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
    CREDIT_REMOVE_BACKGROUND: float = 1.0
    # Standalone user upscale (KIE topaz only) — per upscale_factor
    CREDIT_USER_UPSCALE_2X: float = 3.0
    CREDIT_USER_UPSCALE_4X: float = 5.0
    CREDIT_USER_UPSCALE_8X: float = 7.0

    # fal.ai (optional — remove-background API falls back after KIE failures)
    FAL_API_KEY: str = ""

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

    # After KIE Topaz returns 8K bytes, build 8K/4K/2K/1K variants. When True, run
    # resize+encode on Modal T4 (``modal deploy modal_kie_downsample.py``). On Modal
    # failure or False, fall back to in-process Pillow (asyncio threads).
    MODAL_KIE_DOWNSAMPLE_ENABLED: bool = True
    MODAL_KIE_DOWNSAMPLE_APP_NAME: str = "stylic-kie-downsample"
    MODAL_KIE_DOWNSAMPLE_CLS: str = "KieDownsampleT4"

    # Multi-provider photoshoot generation pipeline (see pipeline.py architecture).
    # KIE is shared with SEEDDREAM_API_KEY; Vertex uses GOOGLE_CLOUD_API_KEY; Evolink uses EVOLINK_API_KEY.
    KIE_API_KEY: str = ""
    # Pre-photoshoot garment analyzer model (KIE Gemini family) used to infer
    # missing upper/lower garment details and footwear text from garment images.
    KIE_GARMENT_ANALYZER_MODEL: str = "gemini-3-pro"
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

    # Feature-generation fallback chain (custom pose/face/background/fabric/color):
    # nano-banana-2 (3x) -> gpt-image-2 (3x), fixed 3:4 and 1K.
    KIE_PRIMARY_IMAGE_MODEL: str = "nano-banana-2"
    KIE_FALLBACK_IMAGE_MODEL: str = "gpt-image-2"
    KIE_FEATURE_MODEL_RETRIES: int = 3
    KIE_FEATURE_ASPECT_RATIO: str = "3:4"
    KIE_FEATURE_RESOLUTION: str = "1K"

    # Razorpay (USD; ensure your Razorpay account supports international / USD)
    RAZORPAY_KEY_ID: str = ""
    RAZORPAY_KEY_SECRET: str = ""
    RAZORPAY_WEBHOOK_SECRET: str = ""
    RAZORPAY_CURRENCY: str = "USD"
    FX_RATE_API_URL: str = "https://api.exchangerate.host/convert"
    FX_RATE_TIMEOUT_S: int = 10

    # Redis / Celery queue
    REDIS_URL: str = "redis://localhost:6379/0"

    # Flower dashboard basic auth (format: user:password)
    FLOWER_BASIC_AUTH: str = ""

    # Public contact / sales form (unauthenticated, rate-limited, strict validation)
    CONTACT_SALES_MAX_PER_IP: int = 5
    CONTACT_SALES_IP_WINDOW_S: int = 3600
    CONTACT_SALES_MAX_PER_EMAIL_PER_24H: int = 5
    # When true, use first IP in X-Forwarded-For (first hop) — enable behind reverse proxy / CDNs.
    CONTACT_SALES_TRUST_X_FORWARDED_FOR: bool = False
    # Contact / transactional email (Gmail and others block inline SVG — use a public <img> URL for the logo)
    STYLIC_MARKETING_BASE_URL: str = "https://stylic.ai"
    # Logotype shown in email header; use a PNG/WebP on your domain or CDN for best results
    STYLIC_MARKETING_LOGO_URL: str = (
        "https://pub-51c3a7dccc2448f792c2fb1bacf8e05d.r2.dev/users/"
        "5971e90a-2682-4c24-a16e-2eda4162f4e8/0f63e79b0d16475288340dfcd5b23444.png"
    )
    STYLIC_EMAIL_FOOTER_CONTACT: str = "contact@stylic.ai"
    STYLIC_EMAIL_FOOTER_COMPANY: str = "Aavish AI Labs"
    STYLIC_SOCIAL_INSTAGRAM_URL: str = "https://www.instagram.com/stylicai/"
    STYLIC_SOCIAL_FACEBOOK_URL: str = "https://www.facebook.com/stylicai/"
    # Web app (welcome email primary CTA)
    STYLIC_APP_BASE_URL: str = "https://app.stylic.ai"
    # Transactional email layout (defaults match email_templates_preview.html)
    STYLIC_EMAIL_BODY_BG: str = "#0f0f12"
    STYLIC_EMAIL_TEXT_PRIMARY: str = "#e8e8ed"
    STYLIC_EMAIL_FONT_STACK: str = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
    STYLIC_EMAIL_MAX_WIDTH: str = "560px"
    STYLIC_EMAIL_INNER_PADDING: str = "40px 20px 48px"
    STYLIC_EMAIL_LOGO_MARGIN_BOTTOM: str = "24px"
    STYLIC_EMAIL_CARD_GRADIENT: str = "linear-gradient(180deg, #1a1a20 0%, #12121a 100%)"
    STYLIC_EMAIL_CARD_BOX_SHADOW: str = "0 24px 64px rgba(0, 0, 0, 0.4)"
    STYLIC_EMAIL_CTA_GRADIENT: str = "linear-gradient(90deg, #f7e6a0 0%, #f0b0c8 45%, #a8b8ff 100%)"
    STYLIC_EMAIL_CTA_LABEL_COLOR: str = "#0f0f12"
    STYLIC_EMAIL_FOOTER_TOP_BORDER: str = "1px solid rgba(255, 255, 255, 0.1)"
    STYLIC_EMAIL_SOCIAL_INSTAGRAM_SVG: str = (
        "https://cdn.jsdelivr.net/npm/simple-icons@11.11.0/icons/instagram.svg"
    )
    STYLIC_EMAIL_SOCIAL_FACEBOOK_SVG: str = (
        "https://cdn.jsdelivr.net/npm/simple-icons@11.11.0/icons/facebook.svg"
    )
    # CSS filter on footer social <img> so black SVGs render white on dark background
    STYLIC_EMAIL_SOCIAL_ICON_FILTER: str = "brightness(0) invert(1)"

    # Admin panel — protect /api/v1/admin/* write routes. Send header:
    #   X-Admin-API-Key: <this value>
    # Empty string disables admin routes (they return 503).
    ADMIN_API_KEY: str = ""

    # JWT for dashboard admins (`/api/v1/admins/*`) — separate from end-user tokens.
    # If empty, falls back to ``JWT_SECRET_KEY`` (set a distinct secret in production).
    ADMIN_JWT_SECRET_KEY: str = ""
    ADMIN_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ADMIN_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    # If set, ``POST /api/v1/admins/auth/bootstrap`` can create the first superadmin when the
    # ``admins`` collection is empty (one-time; header ``X-Admin-Bootstrap-Key``).
    ADMIN_BOOTSTRAP_KEY: str = ""

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
