"""FastAPI application factory.

    uvicorn riskml.server.main:app --port 8000                    # API only
    SERVE_STATIC=1 uvicorn riskml.server.main:app --port 8000     # API + built web UI
"""

from __future__ import annotations

import os

# OpenMP/BLAS read these when they load: one native thread per request so concurrent trainings
# cannot oversubscribe the CPU (must run before numpy / scikit-learn are imported).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("AGRIM_NATIVE_THREADS", "1"))

import json  # noqa: E402
import logging  # noqa: E402
import threading  # noqa: E402
from collections.abc import Iterator  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.middleware.gzip import GZipMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from riskml import __version__  # noqa: E402
from riskml.config import (  # noqa: E402
    ARTIFACTS_DIR,
    CUSTOM_RATE_PER_MIN,
    CUSTOM_TIME_BUDGET_S,
    MAX_COLS,
    MAX_CONCURRENT_TRAININGS,
    MAX_ROWS,
    MAX_UPLOAD_BYTES,
    ROOT,
    SAMPLE_DATA_DIR,
)
from riskml.custom.io import UploadError, read_table  # noqa: E402
from riskml.custom.profile import profile  # noqa: E402
from riskml.custom.trainer import TrainConfig, TrainingError, train_custom  # noqa: E402
from riskml.modeling.zoo import SERVING_MODELS  # noqa: E402
from riskml.server.schemas import PredictRequest, PredictResponse, TrainConfigIn  # noqa: E402
from riskml.server.security import (  # noqa: E402
    BodySizeLimitMiddleware,
    RateLimiter,
    SecurityMiddleware,
    client_key,
)
from riskml.serving.predictor import FeatureValidationError, predict  # noqa: E402
from riskml.serving.registry import ArtifactIntegrityError, ModelRegistry, UnknownCutoffError  # noqa: E402

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(message)s")
logger = logging.getLogger("agrim")

MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 256_000  # multipart overhead + config field
STATIC_DIR = ROOT / "web" / "out"


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None,
                 **extra):
        self.status, self.code, self.message = status, code, message
        self.headers, self.extra = headers or {}, extra


def _error(request: Request, status: int, code: str, message: str, headers=None, **extra):
    rid = getattr(request.state, "request_id", None)
    body = {"error": {"code": code, "message": message, "request_id": rid, **extra}}
    return JSONResponse(body, status_code=status, headers=headers)


NATIVE_THREADS = int(os.environ.get("AGRIM_NATIVE_THREADS", "1"))


@contextmanager
def native_thread_limit() -> Iterator[None]:
    """Limit OpenMP/BLAS threads *in the calling thread*.

    OpenMP thread counts are per-thread settings, and FastAPI runs sync endpoints in worker
    threads, so the limit must be applied inside each request, not only at start-up.
    """
    import sklearn.ensemble  # noqa: F401, PLC0415  (make sure the OpenMP runtime is loaded)
    from threadpoolctl import threadpool_limits  # noqa: PLC0415

    with threadpool_limits(limits=NATIVE_THREADS):
        yield


def create_app(artifacts_dir: Path | None = None, static_dir: Path | None = None,
               rate_per_min: int | None = None, max_trainings: int | None = None,
               time_budget_s: float | None = None, cors_origins: list[str] | None = None,
               trust_proxy: bool | None = None) -> FastAPI:
    app = FastAPI(
        title="Agrim API", version=__version__,
        description="Early academic-risk prediction: OULAD-trained models and "
                    "bring-your-own-data training.",
        docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json",
    )
    app.state.registry = ModelRegistry(artifacts_dir or ARTIFACTS_DIR)
    app.state.training_slots = threading.BoundedSemaphore(max_trainings or
                                                          MAX_CONCURRENT_TRAININGS)
    app.state.limiter = RateLimiter(rate_per_min or CUSTOM_RATE_PER_MIN)
    app.state.time_budget_s = time_budget_s if time_budget_s is not None \
        else CUSTOM_TIME_BUDGET_S
    app.state.trust_proxy = trust_proxy if trust_proxy is not None else \
        os.environ.get("AGRIM_TRUST_PROXY", "1") == "1"

    origins = cors_origins if cors_origins is not None else [
        o.strip() for o in os.environ.get("AGRIM_CORS_ORIGINS", "http://localhost:3000")
        .split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["content-type", "x-request-id"], max_age=600)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)
    app.add_middleware(SecurityMiddleware)

    # ---------------------------------------------------------------- error handlers
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return _error(request, exc.status, exc.code, exc.message, exc.headers, **exc.extra)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        fields = [{"loc": [str(p) for p in e.get("loc", [])], "message": e.get("msg", "")}
                  for e in exc.errors()]
        return _error(request, 422, "validation_error", "The request is not valid.",
                      details=fields)

    @app.exception_handler(FeatureValidationError)
    async def _features(request: Request, exc: FeatureValidationError):
        return _error(request, 422, "invalid_features", "Some feature values are not valid.",
                      fields=exc.errors)

    @app.exception_handler(UnknownCutoffError)
    async def _cutoff(request: Request, exc: UnknownCutoffError):
        return _error(request, 404, "unknown_cutoff",
                      f"No model for that prediction day. Available: {app.state.registry.cutoffs}")

    @app.exception_handler(UploadError)
    async def _upload(request: Request, exc: UploadError):
        return _error(request, 413 if exc.code == "too_large" else 400, exc.code, exc.message)

    @app.exception_handler(TrainingError)
    async def _training(request: Request, exc: TrainingError):
        return _error(request, 422, exc.code, exc.message)

    @app.exception_handler(ArtifactIntegrityError)
    async def _integrity(request: Request, exc: ArtifactIntegrityError):
        logger.error(json.dumps({"request_id": request.state.request_id,
                                 "integrity_error": str(exc)}))
        return _error(request, 503, "model_unavailable", "The model files failed an integrity "
                                                         "check and were not loaded.")

    # ---------------------------------------------------------------- dependencies
    def rate_limited(request: Request) -> None:
        wait = request.app.state.limiter.check(client_key(request, request.app.state.trust_proxy))
        if wait > 0:
            raise ApiError(429, "rate_limited", "Too many uploads. Please wait a moment.",
                           headers={"Retry-After": str(max(1, round(wait)))})

    def read_upload(file: UploadFile) -> bytes:
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise UploadError("too_large",
                              f"Files up to {MAX_UPLOAD_BYTES // 1_000_000} MB are supported.")
        return data

    # ---------------------------------------------------------------- routes
    @app.get("/api/health", tags=["meta"])
    def health(request: Request):
        reg = request.app.state.registry
        try:
            available = reg.cutoffs
            status = "ok"
        except ArtifactIntegrityError:
            available, status = [], "degraded"
        return {"status": status, "version": __version__,
                "models": {"available": available, "loaded": reg.loaded_cutoffs()}}

    @app.get("/api/meta", tags=["meta"])
    def meta(request: Request):
        reg = request.app.state.registry
        return {
            "version": __version__, "cutoffs": reg.cutoffs, "primary_cutoff": reg.primary_cutoff,
            "custom_models": list(SERVING_MODELS),
            "limits": {"max_upload_bytes": MAX_UPLOAD_BYTES, "max_rows": MAX_ROWS,
                       "max_cols": MAX_COLS, "time_budget_s": request.app.state.time_budget_s},
        }

    @app.get("/api/schema/{cutoff}", tags=["oulad"])
    def schema(cutoff: int, request: Request):
        return request.app.state.registry.get(cutoff).schema

    @app.get("/api/samples/{cutoff}", tags=["oulad"])
    def samples(cutoff: int, request: Request):
        return request.app.state.registry.samples(cutoff)

    @app.post("/api/predict", response_model=PredictResponse, tags=["oulad"])
    def predict_route(body: PredictRequest, request: Request):
        loaded = request.app.state.registry.get(body.cutoff)
        with native_thread_limit():
            result = asdict(predict(loaded, body.features))
        return {**result, "model": loaded.schema.get("model", ""),
                "model_version": loaded.version}

    @app.get("/api/custom/sample", tags=["custom"], response_class=FileResponse)
    def sample_dataset():
        return FileResponse(SAMPLE_DATA_DIR / "uci_student_dropout.csv", media_type="text/csv",
                            filename="uci_student_dropout.csv")

    @app.get("/api/custom/template", tags=["custom"], response_class=FileResponse)
    def template():
        return FileResponse(SAMPLE_DATA_DIR / "college_template.csv", media_type="text/csv",
                            filename="college_template.csv")

    @app.post("/api/custom/profile", tags=["custom"], dependencies=[Depends(rate_limited)])
    def custom_profile(file: UploadFile = File(...)):
        df = read_table(read_upload(file), file.filename or "upload")
        with native_thread_limit():
            return profile(df)

    @app.post("/api/custom/train", tags=["custom"], dependencies=[Depends(rate_limited)])
    def custom_train(request: Request, file: UploadFile = File(...),
                     config: str = Form(..., max_length=20_000)):
        try:
            cfg = TrainConfigIn.model_validate(json.loads(config))
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise ApiError(422, "invalid_config", "The training configuration is not valid.",
                           details=str(exc)[:500]) from exc
        slots = request.app.state.training_slots
        if not slots.acquire(blocking=False):
            raise ApiError(503, "busy", "The server is training other models. Try again in a "
                                        "few seconds.", headers={"Retry-After": "15"})
        try:
            df = read_table(read_upload(file), file.filename or "upload")
            with native_thread_limit():
                return train_custom(df, TrainConfig(**cfg.model_dump()),
                                    time_budget_s=request.app.state.time_budget_s)
        finally:
            slots.release()

    static = static_dir if static_dir is not None else (
        STATIC_DIR if os.environ.get("SERVE_STATIC") == "1" else None)
    if static is not None and Path(static).is_dir():
        app.mount("/", StaticFiles(directory=static, html=True), name="site")
    return app


app = create_app()
