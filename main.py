import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler

from app.routers import auth, user, model_face, background, photoshoot, poses, branding_template, storage
from app.firebase_config import get_firebase_app
from app.config import settings
from app.database import create_indexes
from app.services.modal_enhance_service import warmup_modal
from app.rate_limit import limiter
from app.middleware import (
    SecurityHeadersMiddleware,
    RequestContextMiddleware,
    BodySizeLimitMiddleware,
    AdminKeyGuardMiddleware,
)
from app.error_handlers import register_exception_handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_firebase_app()
        logger.info("[startup] Firebase Admin SDK initialized.")
    except RuntimeError as exc:
        logger.warning("[startup] Firebase init skipped: %s", exc)

    await create_indexes()
    logger.info("[startup] MongoDB indexes created.")

    await warmup_modal()
    logger.info("[startup] Modal GPU warmup triggered.")

    logger.info(
        "[startup] environment=%s debug=%s docs=%s cors_origins=%s trusted_hosts=%s",
        settings.ENVIRONMENT,
        settings.DEBUG,
        "enabled" if _docs_enabled() else "disabled",
        settings.cors_origins_list,
        settings.trusted_hosts_list,
    )

    yield


def _docs_enabled() -> bool:
    """Swagger/Redoc are only served in non-prod unless explicitly opted in."""
    if settings.is_production:
        return settings.ENABLE_DOCS_IN_PRODUCTION
    return True


_docs_url = "/docs" if _docs_enabled() else None
_redoc_url = "/redoc" if _docs_enabled() else None
_openapi_url = "/openapi.json" if _docs_enabled() else None


app = FastAPI(
    title="Stylic AI API",
    description="Backend API for Stylic AI – authentication, user management, file uploads, and 2FA.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=_docs_url,
    redoc_url=_redoc_url,
    openapi_url=_openapi_url,
)


# ─────────────────────────── Rate limiting ────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ─────────────────────────── Exception handlers ───────────────────────────
# Registered *before* middleware so the middleware stack (request ID,
# security headers) wraps around the generated error responses.
register_exception_handlers(app)


# ─────────────────────────── Middleware stack ─────────────────────────────
# Starlette applies middleware in REVERSE order of add_middleware() calls,
# i.e. the LAST added runs FIRST on the way in. We want the request
# context (correlation ID) to be set first, then body-size guard, then
# rate limiter, then CORS/TrustedHost, then the app. So we add them in
# the opposite order below.

# (Innermost) Attach security headers to every outbound response.
app.add_middleware(SecurityHeadersMiddleware)

# GZip compress large JSON responses (e.g. photoshoot result lists).
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Admin-key guard for /queue/* (ops-only endpoints).
app.add_middleware(AdminKeyGuardMiddleware)

# slowapi rate-limit middleware. Reads `app.state.limiter`.
app.add_middleware(SlowAPIMiddleware)

# CORS — restrict to the configured origins. In production we refuse to
# start with wildcard+credentials (it's both a security bug and browser-
# rejected). Credentials are only enabled when origins are explicit.
_cors_origins = settings.cors_origins_list
_allow_credentials = bool(_cors_origins) and _cors_origins != ["*"]
if settings.is_production and (not _cors_origins or _cors_origins == ["*"]):
    logger.error(
        "[startup] CORS_ORIGINS is not set in production. No browser origin "
        "will be allowed. Set CORS_ORIGINS to an explicit comma-separated "
        "list (e.g. 'https://app.stylic.ai,https://stylic.ai')."
    )
    _cors_origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or [],
    allow_origin_regex=None,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "Origin",
        "X-Requested-With",
        "X-Request-ID",
    ],
    expose_headers=["X-Request-ID"],
    max_age=600,
)

# Trusted Host — defends against Host header poisoning / cache attacks.
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.trusted_hosts_list,
)

# Body size cap — defense-in-depth against slow-post / OOM.
app.add_middleware(BodySizeLimitMiddleware)

# (Outermost) Request context: correlation ID + access log.
app.add_middleware(RequestContextMiddleware)


# ─────────────────────────── Routers ──────────────────────────────────────
app.include_router(auth.router)
app.include_router(user.router)
app.include_router(model_face.router)
app.include_router(background.router)
app.include_router(photoshoot.router)
app.include_router(poses.router)
app.include_router(branding_template.router)
app.include_router(storage.router)


# ─────────────────────────── Public routes ────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {
        "service": "Stylic AI API",
        "version": "1.0.0",
        "status": "running",
        "docs": _docs_url or "disabled",
    }


@app.get("/health", tags=["Health"])
def health_check():
    # Keep this intentionally minimal — never expose internal dependency
    # state to unauthenticated callers (version, commit SHA, DB status,
    # queue depth, secrets health, etc. are reconnaissance material).
    return {"status": "ok", "service": "Stylic AI API"}


# ─────────────────────────── Ops-only routes ──────────────────────────────
# These are protected by AdminKeyGuardMiddleware above. In production
# callers MUST send `X-Admin-Key: <ADMIN_API_KEY>`; otherwise the
# endpoints return 404. Frontend does not call these.

@app.get("/queue/status", tags=["Queue"])
def queue_status():
    """Return the number of photoshoot jobs waiting in the Celery queue."""
    from app.tasks.photoshoot_tasks import get_queue_length
    pending = get_queue_length()
    return {
        "queue":         "photoshoots",
        "pending_jobs":  pending,
        "note":          "pending_jobs=-1 means Redis is unreachable",
    }


@app.get("/queue/task/{task_id}", tags=["Queue"])
def task_status(task_id: str):
    """Check the status of a specific Celery task by its task_id."""
    from celery.result import AsyncResult
    from app.worker import celery_app
    result = AsyncResult(task_id, app=celery_app)
    response = {
        "task_id": task_id,
        "status":  result.status,
    }
    if result.status == "SUCCESS":
        response["result"] = result.result
    elif result.status == "FAILURE":
        response["error"] = str(result.result)
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        # Honor X-Forwarded-* headers from our reverse proxy (nginx / CF).
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
