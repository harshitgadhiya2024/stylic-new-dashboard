from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import auth, user, model_face
from app.firebase_config import get_firebase_app
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_firebase_app()
        print("[Firebase] Admin SDK initialized.")
    except RuntimeError as exc:
        print(exc)
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
    
