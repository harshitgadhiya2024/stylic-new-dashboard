import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import auth, user, model_face, background, photoshoot
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
