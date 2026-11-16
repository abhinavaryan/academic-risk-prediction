"""Pure-ASGI security middleware, rate limiting and client identification."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid

logger = logging.getLogger("agrim")

_RID = re.compile(r"^[A-Za-z0-9-]{8,64}$")

CSP_API = "default-src 'none'; frame-ancestors 'none'"
CSP_DOCS = ("default-src 'none'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: "
            "https://fastapi.tiangolo.com; connect-src 'self'; frame-ancestors 'none'")
CSP_SITE = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' "
            "'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")


def _header(scope, name: bytes) -> str | None:
    for k, v in scope.get("headers", []):
        if k == name:
            return v.decode("latin-1")
    return None


class SecurityMiddleware:
    """Request IDs, security headers, no-store caching for API responses, access logging."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = _header(scope, b"x-request-id")
        rid = incoming if incoming and _RID.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = rid
        path = scope.get("path", "")
        is_api = path.startswith("/api")
        csp = CSP_DOCS if path.startswith("/api/docs") else CSP_API if is_api else CSP_SITE
        https = (_header(scope, b"x-forwarded-proto") or scope.get("scheme")) == "https"
        started = time.perf_counter()
        status = {"code": 0}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                headers = list(message.get("headers", []))
                extra = {
                    b"x-request-id": rid.encode(),
                    b"x-content-type-options": b"nosniff",
                    b"x-frame-options": b"DENY",
                    b"referrer-policy": b"strict-origin-when-cross-origin",
                    b"permissions-policy": b"camera=(), microphone=(), geolocation=()",
                    b"cross-origin-opener-policy": b"same-origin",
                    b"content-security-policy": csp.encode(),
                }
                if is_api:
                    extra[b"cache-control"] = b"no-store"
                if https:
                    extra[b"strict-transport-security"] = b"max-age=63072000; includeSubDomains"
                present = {k.lower() for k, _ in headers}
                headers += [(k, v) for k, v in extra.items() if k not in present]
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            # Handled here (inside our header wrapper) so 500s also carry security headers and
            # never expose a traceback.
            logger.exception("unhandled error request_id=%s", rid)
            if status["code"] == 0:
                body = json.dumps({"error": {"code": "internal_error", "request_id": rid,
                                             "message": "Something went wrong on our side."}})
                await send_wrapper({"type": "http.response.start", "status": 500,
                                    "headers": [(b"content-type", b"application/json")]})
                await send_wrapper({"type": "http.response.body", "body": body.encode()})
        finally:
            logger.info(json.dumps({
                "request_id": rid, "method": scope.get("method"), "path": path,
                "status": status["code"],
                "ms": round((time.perf_counter() - started) * 1000, 1)}))


class _TooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Reject request bodies above ``max_bytes`` (declared or streamed) with HTTP 413."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def _reject(self, send):
        body = json.dumps({"error": {"code": "too_large", "message":
                                     "The upload is larger than the allowed size."}}).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = _header(scope, b"content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            await self._reject(send)
            return
        received = 0
        started = {"value": False}

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _TooLarge
            return message

        async def tracking_send(message):
            if message["type"] == "http.response.start":
                started["value"] = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _TooLarge:
            if not started["value"]:
                await self._reject(send)


class RateLimiter:
    """Thread-safe token bucket per client key (best effort: per server instance)."""

    def __init__(self, per_minute: int):
        self.capacity = float(max(per_minute, 1))
        self.rate = self.capacity / 60.0
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float:
        """Consume a token. Returns 0 if allowed, else seconds until the next token."""
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens >= 1:
                self._buckets[key] = (tokens - 1, now)
                wait = 0.0
            else:
                self._buckets[key] = (tokens, now)
                wait = (1 - tokens) / self.rate
            if len(self._buckets) > 10_000:
                cutoff = now - 120
                self._buckets = {k: v for k, v in self._buckets.items() if v[1] > cutoff}
            return wait


def client_key(request, trust_proxy: bool) -> str:
    if trust_proxy:
        for header in ("x-vercel-forwarded-for", "x-forwarded-for", "x-real-ip"):
            value = request.headers.get(header)
            if value:
                return value.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
