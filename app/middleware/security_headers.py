"""Inject standard HTTP security headers on every response.

These are the headers recommended by OWASP Secure Headers Project. We
set them conservatively so they don't interfere with the JSON API:

- HSTS is only sent when we believe we're served over HTTPS (behind a
  reverse proxy that sets `X-Forwarded-Proto=https`, or in production).
- CSP is intentionally API-oriented (`default-src 'none'`) — this is a
  JSON API, not an HTML app, so we lock everything down. The Swagger
  docs page (when enabled) still works because it's served by FastAPI
  with its own inline resources; we relax CSP for the docs paths.

None of this changes request/response *bodies* — only headers — so the
existing frontend keeps working.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from app.config import settings

_DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._is_prod = settings.is_production

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        headers = response.headers

        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        headers.setdefault(
            "Permissions-Policy",
            "geolocation=(), microphone=(), camera=(), payment=(), usb=()",
        )
        headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")

        path = request.url.path
        if path.startswith(_DOCS_PATHS):
            # Swagger/Redoc need inline styles + their CDN for scripts.
            headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; "
                "img-src 'self' data: https:; "
                "style-src 'self' 'unsafe-inline' https:; "
                "script-src 'self' 'unsafe-inline' https:; "
                "font-src 'self' data: https:; "
                "connect-src 'self'",
            )
        else:
            # Pure JSON API — deny everything by default.
            headers.setdefault(
                "Content-Security-Policy",
                "default-src 'none'; frame-ancestors 'none'",
            )

        # HSTS only over HTTPS. Trust `X-Forwarded-Proto` from our own
        # reverse proxy (we already run behind nginx/Cloudflare per
        # DEPLOYMENT.md). In dev we intentionally skip this so browsers
        # don't cache a permanent HTTPS-only rule for localhost.
        forwarded_proto = request.headers.get("x-forwarded-proto", "").lower()
        is_https = request.url.scheme == "https" or forwarded_proto == "https"
        if is_https and self._is_prod:
            headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains; preload",
            )

        # Don't advertise stack details.
        if "server" in headers:
            del headers["server"]
        if "x-powered-by" in headers:
            del headers["x-powered-by"]

        return response
