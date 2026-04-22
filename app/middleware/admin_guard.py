"""Require a shared admin secret on ops-only endpoints.

The frontend does not call these, so protecting them with an
`X-Admin-Key` header is a pure backend change.

Behaviour:

- In **production** with `ADMIN_API_KEY` set → header must match.
- In **production** with `ADMIN_API_KEY` empty → endpoints return 404
  (we don't want to leak their existence at all).
- In **development** → endpoints are open (so local debugging works).

The endpoints protected are configured via `PROTECTED_PATHS` — currently
the `/queue/*` queue introspection endpoints in `main.py`.
"""

from __future__ import annotations

import hmac

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from app.config import settings

# Prefix-matched; keep these minimal. Anything here is considered
# "operator only" — never call them from the browser.
PROTECTED_PATHS: tuple[str, ...] = ("/queue/",)


class AdminKeyGuardMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._is_prod = settings.is_production
        self._key = settings.ADMIN_API_KEY or ""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not any(path.startswith(p) for p in PROTECTED_PATHS):
            return await call_next(request)

        if not self._is_prod:
            return await call_next(request)

        if not self._key:
            # Pretend it doesn't exist rather than advertising a locked
            # endpoint. Ops can enable it by setting ADMIN_API_KEY.
            return JSONResponse({"detail": "Not Found"}, status_code=404)

        supplied = request.headers.get("x-admin-key", "")
        if not supplied or not hmac.compare_digest(supplied, self._key):
            return JSONResponse({"detail": "Not Found"}, status_code=404)

        return await call_next(request)
