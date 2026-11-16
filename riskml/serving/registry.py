"""Thread-safe, integrity-checked registry of the deployed OULAD models.

Security model
* Every file is read only if its SHA-256 matches ``artifacts/manifest.json``.
* Models are stored in the skops format and loaded with an explicit allow-list of types, so a
  tampered file cannot execute code the way a pickle could.
* Only file names matching ``[A-Za-z0-9_.-]+.skops`` are accepted (no path traversal).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import skops.io as sio

from riskml.config import ARTIFACTS_DIR

TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]
_MODEL_FILE = re.compile(r"^[A-Za-z0-9_.-]+\.skops$")


class ArtifactIntegrityError(RuntimeError):
    """An artifact is missing from the manifest, altered, or of an unexpected shape."""


class UnknownCutoffError(KeyError):
    """No deployed model exists for the requested prediction day."""


@dataclass(frozen=True)
class LoadedModel:
    cutoff: int
    model: Any
    schema: dict
    threshold: float
    version: str
    feature_names: list[str]
    baseline: dict[str, Any]
    labels: dict[str, str] = field(repr=False)


class ModelRegistry:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        self._dir = Path(artifacts_dir)
        self._lock = threading.Lock()
        self._cache: dict[int, LoadedModel] = {}
        self._samples: dict[int, list] = {}
        self._manifest: dict | None = None

    # -- manifest -----------------------------------------------------------------------
    @property
    def manifest(self) -> dict:
        if self._manifest is None:
            path = self._dir / "manifest.json"
            if not path.exists():
                raise ArtifactIntegrityError(f"manifest not found in {self._dir}")
            self._manifest = json.loads(path.read_text())
        return self._manifest

    @property
    def cutoffs(self) -> list[int]:
        return sorted(int(c) for c in self.manifest["cutoffs"])

    @property
    def primary_cutoff(self) -> int:
        return int(self.manifest.get("primary_cutoff", self.cutoffs[0]))

    def loaded_cutoffs(self) -> list[int]:
        return sorted(self._cache)

    def _read_verified(self, rel: str) -> bytes:
        expected = self.manifest.get("files", {}).get(rel)
        if expected is None:
            raise ArtifactIntegrityError(f"{rel} is not listed in the manifest")
        path = (self._dir / rel).resolve()
        if self._dir.resolve() not in path.parents or not path.exists():
            raise ArtifactIntegrityError(f"{rel} is missing")
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if not hmac.compare_digest(actual, expected):
            raise ArtifactIntegrityError(f"{rel} failed its SHA-256 integrity check")
        return data

    # -- models -------------------------------------------------------------------------
    def get(self, cutoff: int) -> LoadedModel:
        cached = self._cache.get(cutoff)
        if cached is not None:
            return cached
        if cutoff not in self.cutoffs:
            raise UnknownCutoffError(cutoff)
        with self._lock:  # double-checked locking: load each model exactly once
            cached = self._cache.get(cutoff)
            if cached is None:
                cached = self._load(cutoff)
                self._cache[cutoff] = cached
            return cached

    def _load(self, cutoff: int) -> LoadedModel:
        schema = json.loads(self._read_verified(f"schema/day{cutoff}.json"))
        model_file = schema.get("model_file", "")
        if not _MODEL_FILE.match(model_file):
            raise ArtifactIntegrityError(f"invalid model file name {model_file!r}")
        model = sio.loads(self._read_verified(f"models/{model_file}"), trusted=TRUSTED_TYPES)
        feats = schema["features"]
        baseline = {
            f["name"]: (f.get("median") if f["kind"] == "numeric" else f.get("mode")) for f in feats
        }
        return LoadedModel(
            cutoff=cutoff,
            model=model,
            schema=schema,
            threshold=float(schema["threshold"]),
            version=str(schema.get("version", "")),
            feature_names=[f["name"] for f in feats],
            baseline=baseline,
            labels={f["name"]: f.get("label", f["name"]) for f in feats},
        )

    def samples(self, cutoff: int) -> list[dict]:
        if cutoff not in self.cutoffs:
            raise UnknownCutoffError(cutoff)
        if cutoff not in self._samples:
            with self._lock:
                if cutoff not in self._samples:
                    self._samples[cutoff] = json.loads(
                        self._read_verified(f"samples/day{cutoff}.json")
                    )
        return self._samples[cutoff]
