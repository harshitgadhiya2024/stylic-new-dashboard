"""Attach a per-request correlation ID and lightweight access log.

- Reads an incoming `X-Request-ID` header (useful for client-side tracing
  or load-balancer correlation), falls back to a UUID4.
- Stores it on `request.state.request_id` so handlers/exception handlers
  can reference it.
- Echoes it back on the response as `X-Request-ID` so clients can
  include it in bug reports / support tickets.
- Logs one structured line per request with method, path, status, and
  duration. No request *body* is logged — bodies may contain PII,
  passwords, OTPs, or image bytes.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

logger = logging.getLogger("http.access")

# Allow only sane-looking IDs through from clients (defense against log
# injection). Anything longer or containing control chars is discarded
# and replaced with a fresh UUID.
_MAX_INBOUND_ID_LEN = 128


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        inbound = request.headers.get("x-request-id", "")
        if (
            inbound
            and len(inbound) <= _MAX_INBOUND_ID_LEN
            and inbound.replace("-", "").replace("_", "").isalnum()
        ):
            request_id = inbound
        else:
            request_id = uuid.uuid4().hex

        request.state.request_id = request_id

        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            # Never log Authorization / Cookie / query-strings-with-tokens.
            logger.info(
                "rid=%s %s %s -> %d in %.1fms",
                request_id,
                request.method,
                request.url.path,
                status_code,
                duration_ms,
            )
            # Attach the header if we produced a response. If an exception
            # escapes, FastAPI's default 500 handler doesn't go through
            # this path but our custom handler does.
            try:
                response.headers["X-Request-ID"] = request_id  # type: ignore[name-defined]
            except Exception:
                pass
