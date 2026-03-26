from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # MongoDB
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "stylicai"

    # JWT
    JWT_SECRET_KEY: str = "change-this-secret-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # SMTP (Hostinger)
    SMTP_SERVER: str = "smtp.hostinger.com"
    SMTP_PORT: int = 587
    SMTP_EMAIL: str = ""
    SMTP_PASSWORD: str = ""

    # AWS S3
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"
    AWS_S3_BUCKET_NAME: str = ""

    # Firebase (Google Sign-In)
    FIREBASE_SERVICE_ACCOUNT_KEY: str = "./stylic-ai-d1ee0-firebase-adminsdk-fbsvc-a4a36772f6.json"

    # Gemini AI
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash-image"
    GEMINI_VISION_MODEL: str = "gemini-2.5-flash"

    # SeedDream / kie.ai
    SEEDDREAM_API_KEY: str = ""
    SEEDDREAM_MODEL: str = "seedream/4.5-edit"
    SEEDDREAM_QUALITY: str = "basic"
    SEEDDREAM_ASPECT: str = "9:16"
    SEEDDREAM_MAX_RETRIES: int = 120
    SEEDDREAM_RETRY_DELAY: int = 5

    # App defaults
    APP_NAME: str = "Stylic AI"
    DEFAULT_CREDITS: int = 5
    DEFAULT_PLAN: str = "free"
    OTP_EXPIRE_MINUTES: int = 10

    # Modal GPU enhancement pipeline
    # Class names must match what is currently DEPLOYED on Modal.
    # After running `modal deploy modal_realism_pipeline.py` with the new GPU classes,
    # update these via .env:
    #   MODAL_CLS_PRIMARY=FashionRealismT4
    #   MODAL_CLS_FALLBACK=FashionRealismL4
    MODAL_APP_NAME:    str = "fashion-realism"
    MODAL_CLS_PRIMARY: str = "FashionRealismT4"   # L40S — primary GPU class
    MODAL_CLS_FALLBACK: str = "FashionRealismL4" 

    # Redis / Celery queue
    REDIS_URL: str = "redis://localhost:6379/0"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
