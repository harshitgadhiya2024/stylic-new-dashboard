from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Runtime environment ───────────────────────────────────────────────
    # One of: "development" | "staging" | "production".
    # Controls CORS defaults, /docs exposure, error verbosity, HSTS, cookie
    # Secure flag, and rate limiter enforcement.
    ENVIRONMENT: str = "development"

    # MongoDB
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "stylicai"

    # JWT
    JWT_SECRET_KEY: str = "change-this-secret-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    # Optional iss/aud binding. Leave empty to keep existing tokens valid;
    # once every client has refreshed, set these in production for stricter
    # token validation (defense-in-depth against token confusion attacks).
    JWT_ISSUER: str = ""
    JWT_AUDIENCE: str = ""

    # SMTP (Hostinger)
    SMTP_SERVER: str = "smtp.hostinger.com"
    SMTP_PORT: int = 587
    SMTP_EMAIL: str = ""
    SMTP_PASSWORD: str = ""
    # Resend (preferred mail provider; falls back to SMTP when unset)
    RESEND_API_KEY: str = ""
    RESEND_FROM_EMAIL: str = ""

    # Cloudflare R2 (S3-compatible API — see cloudflare-r2-guide.md)
    R2_ACCOUNT_ID: str = ""
    R2_ACCESS_KEY_ID: str = ""
    R2_SECRET_ACCESS_KEY: str = ""
    R2_BUCKET_NAME: str = ""
    # Public base URL (no trailing slash), e.g. https://cdn.yourdomain.com or https://pub-xxx.r2.dev
    R2_PUBLIC_URL: str = ""
    # Optional full endpoint override; default is https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com
    R2_ENDPOINT_URL: str = ""

    # Firebase (Google Sign-In)
    FIREBASE_SERVICE_ACCOUNT_KEY: str = "./stylic-ai-d1ee0-firebase-adminsdk-fbsvc-a4a36772f6.json"

    # Gemini AI
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash-image"
    GEMINI_VISION_MODEL: str = "gemini-2.5-flash"

    # SeedDream / kie.ai
    SEEDDREAM_API_KEY: str = ""
    SEEDDREAM_MODEL: str = "seedream/4.5-edit"
    # Legacy — kept for backward compatibility; photoshoot pipeline now uses SEEDDREAM_MODEL only.
    PHOTOSHOOT_GENERATE_MODEL: str = "nano-banana-pro"
    SEEDDREAM_QUALITY: str = "high"
    SEEDDREAM_ASPECT: str = "9:16"
    SEEDDREAM_MAX_RETRIES: int = 120
    SEEDDREAM_RETRY_DELAY: int = 5

    # Realism pass — nano-banana-pro (runs after SeedDream, before Modal)
    REALISM_MODEL: str = "nano-banana-pro"
    REALISM_QUALITY: str = "4K"
    REALISM_ASPECT: str = "9:16"

    # Model face generation (kie.ai) — same model family as scripts/generate_model_faces.py (Gemini 3 Pro Image via Kie)
    MODEL_FACE_GENERATE_MODEL: str = "nano-banana-pro"
    MODEL_FACE_GENERATE_ASPECT: str = "4:5"
    MODEL_FACE_GENERATE_RESOLUTION: str = "1K"

    # Reference-photo model face — SeedDream 5.0 Lite (upload flow uses image-to-image + user photo URL)
    MODEL_FACE_REFERENCE_SEEDREAM_IMG2IMG_MODEL: str = "seedream/5-lite-image-to-image"
    # Legacy text-to-image id (unused by reference upload; kept for env / tooling compatibility)
    MODEL_FACE_REFERENCE_SEEDREAM_MODEL: str = "seedream/5-lite-text-to-image"
    # kie 5-lite enum: 1:1, 4:3, 3:4, 16:9, 9:16, 2:3, 3:2, 21:9 (no 4:5 — 3:4 is portrait)
    MODEL_FACE_REFERENCE_SEEDREAM_ASPECT: str = "3:4"
    MODEL_FACE_REFERENCE_SEEDREAM_QUALITY: str = "basic"

    # Pose mannequin (create pose from image / from prompt) — SeedDream 5.0 Lite
    POSE_MANNEQUIN_SEEDREAM_TEXT_MODEL: str = "seedream/5-lite-text-to-image"
    POSE_MANNEQUIN_SEEDREAM_IMG2IMG_MODEL: str = "seedream/5-lite-image-to-image"
    POSE_MANNEQUIN_SEEDREAM_ASPECT: str = "9:16"
    POSE_MANNEQUIN_SEEDREAM_QUALITY: str = "basic"

    # App defaults
    APP_NAME: str = "Stylic AI"
    DEFAULT_CREDITS: int = 5
    DEFAULT_PLAN: str = "free"
    OTP_EXPIRE_MINUTES: int = 10

    # Credit limits (per image)
    CREDIT_SINGLE_PHOTOSHOOT_PER_IMAGE: float = 2.0
    CREDIT_REGENERATE_PER_IMAGE: float = 2.0
    CREDIT_BACKGROUND_CHANGE_PER_IMAGE: float = 2.0
    CREDIT_UPSCALE_4X_PER_IMAGE: float = 4.0
    CREDIT_UPSCALE_8X_PER_IMAGE: float = 8.0
    CREDIT_BRANDING_PER_IMAGE: float = 1.0
    CREDIT_RESIZE_PER_IMAGE: float = 1.0
    CREDIT_ADJUST_IMAGE_PER_IMAGE: float = 1.0
    CREDIT_FABRIC_CHANGE_PER_IMAGE: float = 3.0
    CREDIT_TEXTURE_CHANGE_PER_IMAGE: float = 3.0
    CREDIT_COLOR_CHANGE_PER_IMAGE: float = 3.0

    # Modal GPU enhancement pipeline
    # Class names must match what is currently DEPLOYED on Modal.
    # After running `modal deploy modal_realism_pipeline.py` with the new GPU classes,
    # update these via .env:
    #   MODAL_CLS_PRIMARY=FashionRealismT4
    #   MODAL_CLS_FALLBACK=FashionRealismL4
    MODAL_APP_NAME:    str = "fashion-realism"
    MODAL_CLS_PRIMARY: str = "FashionRealismT4"   # L40S — primary GPU class
    MODAL_CLS_FALLBACK: str = "FashionRealismL4" 
    WHICH_UPSCALE: str = "modal"  # "modal" | "kie"
    KIE_UPSCALE_MODEL: str = "topaz/image-upscale"
    KIE_UPSCALE_FACTOR: int = 2
    KIE_REQUEST_RETRIES: int = 3
    KIE_HTTP_TIMEOUT: int = 300

    # --- Upscale pipeline tuning (added 2026-04) -------------------------------
    # Output encoding for the 8k/4k/2k/1k variants. Default is PNG with a low
    # compress_level, which is fully LOSSLESS (bit-identical pixels) and ~15-20x
    # faster than the legacy PNG `optimize=True` path. Trade-off: files are
    # ~30% larger on disk vs compress_level=6.
    #
    #   png_fast     — lossless PNG, compress_level=1 (~3-6s @ 8K, ~130 MB). DEFAULT.
    #   png          — lossless PNG, compress_level=6 (~15-25s @ 8K, ~100 MB).
    #   webp_lossless — lossless WebP (~4-8s @ 8K, ~40 MB). Same pixels, smaller.
    #   jpeg         — LOSSY; only use if you explicitly accept JPEG artifacts.
    KIE_VARIANT_FORMAT: str = "png_fast"       # "png_fast" | "png" | "webp_lossless" | "jpeg"
    # Only used when KIE_VARIANT_FORMAT=jpeg (kept for backward compat).
    KIE_VARIANT_JPEG_QUALITY: int = 95
    # Hard wall-clock ceiling for a single upscale job (including wait + download).
    KIE_UPSCALE_MAX_WAIT_S: float = 900.0      # 15 min

    # Multi-provider photoshoot generation pipeline (see pipeline.py architecture).
    # KIE is shared with SEEDDREAM_API_KEY; Vertex uses GOOGLE_CLOUD_API_KEY; Evolink uses EVOLINK_API_KEY.
    KIE_API_KEY: str = ""
    GOOGLE_CLOUD_API_KEY: str = ""
    EVOLINK_API_KEY: str = ""
    # Generation tuning (aspect/resolution for KIE nano-banana-2 & Vertex models)
    PHOTOSHOOT_GEN_ASPECT_RATIO: str = "9:16"
    PHOTOSHOOT_GEN_RESOLUTION: str = "4K"
    # Per-image wall clock timeout for one generation attempt (seconds).
    PHOTOSHOOT_GEN_PER_IMAGE_TIMEOUT_S: float = 600.0
    # KIE status polling interval for Stage-1 generation tasks (seconds).
    # Reduced from 5s to 2s because polls now share an event loop with heavy
    # Stage-2 work; tighter interval catches completions sooner without
    # materially increasing API load.
    PHOTOSHOOT_KIE_POLL_INTERVAL_S: float = 2.0

    # --- KIE account-wide rate limiter (cross-worker, Redis sliding window) ---
    # KIE.ai enforces ~20 createTask / 10 s per account. We pin submissions at
    # KIE_RATE_LIMIT_MAX per KIE_RATE_LIMIT_WINDOW_S so every Celery worker and
    # FastAPI process consults the same Redis counter before POSTing
    # /createTask. Set KIE_RATE_LIMIT_ENABLED=false only for local dev.
    KIE_RATE_LIMIT_ENABLED:     bool  = True
    KIE_RATE_LIMIT_MAX:         int   = 10         # submissions per window
    KIE_RATE_LIMIT_WINDOW_S:    float = 10.0       # window seconds
    KIE_RATE_LIMIT_KEY_PREFIX:  str   = "kie:createTask"
    # Max seconds a caller waits for a token before raising TimeoutError
    # (falls through to the next provider in the generator chain).
    KIE_RATE_LIMIT_MAX_WAIT_S:  float = 60.0
    # Extra cool-down applied when a KIE HTTP 429 is actually observed.
    KIE_RATE_LIMIT_429_SLEEP_S: float = 5.0

    # Redis / Celery queue
    REDIS_URL: str = "redis://localhost:6379/0"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # ── HTTP security ─────────────────────────────────────────────────────
    # Comma-separated list of allowed browser origins. In production you
    # MUST set this explicitly (e.g. "https://app.stylic.ai,https://stylic.ai").
    # Empty string in production falls back to a deny-all list.
    CORS_ORIGINS: str = "*"
    # Comma-separated list of Host header values allowed to reach the app
    # (defends against Host header poisoning and cache attacks). Empty /
    # "*" disables the check. Example: "api.stylic.ai,stylic.ai".
    TRUSTED_HOSTS: str = "*"
    # Hard ceiling on request body size (MiB). Protects workers from OOM /
    # slow-post DoS. Tune upward only if you intentionally accept larger
    # uploads at the app tier (usually you'd do this at the reverse proxy).
    MAX_REQUEST_BODY_MB: int = 20
    # Shared secret for ops-only endpoints (/queue/*). Clients send it in
    # the `X-Admin-Key` header. If empty, those endpoints are disabled in
    # production and open in development.
    ADMIN_API_KEY: str = ""
    # Expose /docs, /redoc, /openapi.json even in production. Leave False
    # unless you have an external gateway protecting them.
    ENABLE_DOCS_IN_PRODUCTION: bool = False

    # ── Generic HTTP rate limiting (slowapi, Redis-backed) ────────────────
    # These apply *in addition* to KIE_RATE_LIMIT_* which protects the
    # outbound KIE API. Limits here protect *our* API from abuse.
    RATE_LIMIT_ENABLED: bool = True
    # Default limit applied to every route that doesn't override it.
    RATE_LIMIT_DEFAULT: str = "120/minute"
    # Stricter limit for auth endpoints (login / register / forgot-password
    # / resend-otp). Per-IP; per-email lockout is handled separately.
    RATE_LIMIT_AUTH: str = "10/minute"
    # Limit for the Google Sign-In endpoint (Firebase verification is
    # relatively cheap, but still don't let anyone flood it).
    RATE_LIMIT_GOOGLE: str = "20/minute"

    # `extra="ignore"` lets the .env carry ops-only variables that aren't
    # consumed by the app itself (e.g. FLOWER_BASIC_AUTH, which is read
    # by the stylicai-flower.service systemd unit). Without this,
    # pydantic-settings raises `extra_forbidden` on startup.
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "case_sensitive": False,
    }

    # ── Derived helpers ──────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    @property
    def cors_origins_list(self) -> List[str]:
        raw = (self.CORS_ORIGINS or "").strip()
        if not raw:
            return []
        if raw == "*":
            # Wildcard is only safe in dev and without credentials. In prod
            # we treat "*" as "not configured" and deny everything.
            return ["*"] if not self.is_production else []
        return [o.strip() for o in raw.split(",") if o.strip()]

    @property
    def trusted_hosts_list(self) -> List[str]:
        raw = (self.TRUSTED_HOSTS or "").strip()
        if not raw or raw == "*":
            return ["*"]
        return [h.strip() for h in raw.split(",") if h.strip()]

    @field_validator("ENVIRONMENT")
    @classmethod
    def _normalize_env(cls, v: str) -> str:
        v = (v or "development").strip().lower()
        if v not in {"development", "staging", "production"}:
            v = "development"
        return v


settings = Settings()
