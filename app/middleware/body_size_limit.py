"""Reject requests whose body is larger than MAX_REQUEST_BODY_MB.

Two layers of defense:

1. If the client sends `Content-Length`, reject early with 413 before
   reading any bytes.
2. If the client streams without a known length, wrap the receive
   channel and count bytes; abort the request once the cap is exceeded.

This protects Celery workers / FastAPI processes from memory exhaustion
and classic slow-post attacks. For *very* large legitimate uploads, the
reverse proxy should cap first; this is a defense-in-depth layer.
"""

from __future__ import annotations

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import settings


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.max_bytes = max(1, settings.MAX_REQUEST_BODY_MB) * 1024 * 1024

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_bytes:
                    await self._reject(send)
                    return
            except ValueError:
                pass

        total = 0
        max_bytes = self.max_bytes

        async def limited_receive() -> Message:
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"") or b""
                total += len(body)
                if total > max_bytes:
                    return {"type": "http.disconnect"}
            return message

        await self.app(scope, limited_receive, send)

    @staticmethod
    async def _reject(send: Send) -> None:
        response = JSONResponse(
            {"detail": "Request body too large."},
            status_code=413,
        )
        await response({"type": "http"}, receive=_empty_receive, send=send)  # type: ignore[arg-type]


async def _empty_receive() -> Message:
    return {"type": "http.disconnect"}
