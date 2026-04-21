import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import auth, user, model_face, background, photoshoot, poses, branding_template, storage, webhooks
from app.firebase_config import get_firebase_app
from app.config import settings
from app.database import create_indexes
from app.services.modal_enhance_service import warmup_modal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Firebase
    try:
        get_firebase_app()
        logger.info("[startup] Firebase Admin SDK initialized.")
    except RuntimeError as exc:
        logger.warning("[startup] Firebase init skipped: %s", exc)

    # MongoDB indexes
    await create_indexes()
    logger.info("[startup] MongoDB indexes created.")

    # Modal GPU warmup — downloads weights and loads models in background
    await warmup_modal()
    logger.info("[startup] Modal GPU warmup triggered.")

    yield


app = FastAPI(
    title="Stylic AI API",
    description="Backend API for Stylic AI – authentication, user management, file uploads, and 2FA.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(user.router)
app.include_router(model_face.router)
app.include_router(background.router)
app.include_router(photoshoot.router)
app.include_router(poses.router)
app.include_router(branding_template.router)
app.include_router(storage.router)
app.include_router(webhooks.router)


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "Stylic AI API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "Stylic AI API"}


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
        "status":  result.status,  # PENDING | STARTED | SUCCESS | FAILURE | RETRY
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
    )
