from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pymongo import ASCENDING
from app.config import settings

_client: AsyncIOMotorClient = None


def get_db_client() -> AsyncIOMotorClient:
    """Process-global client for FastAPI / long-lived asyncio app.

    Do **not** use from Celery tasks wrapped in ``asyncio.run()`` — that closes
    the loop while this client may still be bound to it. Use
    :func:`make_motor_client` per task instead.
    """
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.MONGO_URI)
    return _client


def get_database():
    return get_db_client()[settings.MONGO_DB_NAME]


def make_motor_client() -> AsyncIOMotorClient:
    """Return a brand-new Motor client bound to the current event loop.

    Use this inside Celery tasks where asyncio.run() creates a fresh loop
    for every task execution.  The caller is responsible for closing the
    client when the coroutine finishes.
    """
    return AsyncIOMotorClient(settings.MONGO_URI)


def get_users_collection() -> AsyncIOMotorCollection:
    return get_database()["users"]


def get_otps_collection() -> AsyncIOMotorCollection:
    return get_database()["otps"]


def get_refresh_tokens_collection() -> AsyncIOMotorCollection:
    return get_database()["refresh_tokens"]


def get_model_faces_collection() -> AsyncIOMotorCollection:
    return get_database()["model_faces"]


def get_credit_history_collection() -> AsyncIOMotorCollection:
    return get_database()["credit_history"]


def get_backgrounds_collection() -> AsyncIOMotorCollection:
    return get_database()["backgrounds"]


def get_photoshoots_collection() -> AsyncIOMotorCollection:
    return get_database()["photoshoots"]


def get_poses_collection() -> AsyncIOMotorCollection:
    return get_database()["poses_data"]


def get_upscaling_collection() -> AsyncIOMotorCollection:
    return get_database()["upscaling_data"]


def get_templates_collection() -> AsyncIOMotorCollection:
    return get_database()["templates"]


async def create_indexes() -> None:
    """Create all collection indexes. Call once at application startup."""
    db = get_database()

    otps = db["otps"]
    await otps.create_index("expires_at", expireAfterSeconds=0)
    await otps.create_index([("email", ASCENDING), ("purpose", ASCENDING)])

    refresh_tokens = db["refresh_tokens"]
    await refresh_tokens.create_index("expires_at", expireAfterSeconds=0)

    model_faces = db["model_faces"]
    await model_faces.create_index("model_id", unique=True)
    await model_faces.create_index("user_id")
    await model_faces.create_index("model_category")

    credit_history = db["credit_history"]
    await credit_history.create_index("history_id", unique=True)
    await credit_history.create_index("user_id")
    await credit_history.create_index("created_at")

    backgrounds = db["backgrounds"]
    await backgrounds.create_index("background_id", unique=True)
    await backgrounds.create_index("user_id")
    await backgrounds.create_index("is_default")
    await backgrounds.create_index("is_favorite")
    await backgrounds.create_index("background_type")

    photoshoots = db["photoshoots"]
    await photoshoots.create_index("photoshoot_id", unique=True)
    await photoshoots.create_index("user_id")
    await photoshoots.create_index("status")
    await photoshoots.create_index("is_active")
    await photoshoots.create_index("created_at")

    upscaling = db["upscaling_data"]
    await upscaling.create_index([("photoshoot_id", ASCENDING), ("image_id", ASCENDING)])
    await upscaling.create_index("photoshoot_id")
    await upscaling.create_index("image_id")

    poses_data = db["poses_data"]
    await poses_data.create_index("pose_id")

    templates = db["templates"]
    await templates.create_index("template_id", unique=True)
    await templates.create_index("user_id")
    await templates.create_index([("user_id", ASCENDING), ("is_default", ASCENDING)])
