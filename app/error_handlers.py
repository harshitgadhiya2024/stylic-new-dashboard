"""Centralised exception handlers.

Goals
-----
- Never leak tracebacks, file paths, SQL, or library internals to the
  client. The existing frontend only needs a `detail` string on errors,
  so we keep that shape for 4xx responses (unchanged behaviour) and
  return a generic one for 5xx.
- Always include the `X-Request-ID` so operators can pivot from a
  user-reported error to the exact log line.
- Log the full exception server-side at ERROR level with the request ID.

What does *not* change
----------------------
- FastAPI's default 422 (validation) shape is preserved — we just
  forward to FastAPI's own handler. The frontend already parses it.
- HTTPException(status=..., detail=...) responses from route handlers
  keep their `{"detail": ...}` shape.

What changes
------------
- Uncaught `Exception` used to produce Starlette's default
  `{"detail": "Internal Server Error"}` with no correlation to logs. It
  still returns that shape, but now carries an `X-Request-ID` header and
  is logged with full traceback on our side.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("app.errors")


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "") or ""


async def _http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    headers = dict(exc.headers or {})
    rid = _request_id(request)
    if rid:
        headers["X-Request-ID"] = rid
    return JSONResponse(
        {"detail": exc.detail},
        status_code=exc.status_code,
        headers=headers or None,
    )


async def _validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    rid = _request_id(request)
    headers = {"X-Request-ID": rid} if rid else None
    # Preserve FastAPI's default 422 body shape so existing clients keep
    # parsing errors identically.
    return JSONResponse(
        {"detail": exc.errors()},
        status_code=422,
        headers=headers,
    )


async def _unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    rid = _request_id(request)
    logger.exception(
        "Unhandled exception rid=%s path=%s method=%s",
        rid, request.url.path, request.method,
    )
    headers = {"X-Request-ID": rid} if rid else None
    return JSONResponse(
        {"detail": "Internal server error."},
        status_code=500,
        headers=headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(HTTPException, _http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
