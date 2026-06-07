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


def get_payment_history_collection() -> AsyncIOMotorCollection:
    return get_database()["payment_history"]


def get_backgrounds_collection() -> AsyncIOMotorCollection:
    return get_database()["backgrounds"]


def get_photoshoots_collection() -> AsyncIOMotorCollection:
    return get_database()["photoshoots"]


def get_poses_collection() -> AsyncIOMotorCollection:
    return get_database()["poses_data"]


def get_upscaling_collection() -> AsyncIOMotorCollection:
    return get_database()["upscaling_data"]


def get_remove_background_collection() -> AsyncIOMotorCollection:
    return get_database()["remove_background_data"]


def get_user_upscaled_collection() -> AsyncIOMotorCollection:
    return get_database()["user_upscaled"]


def get_templates_collection() -> AsyncIOMotorCollection:
    return get_database()["templates"]


def get_contact_sales_collection() -> AsyncIOMotorCollection:
    return get_database()["contact_sales"]


def get_blogs_collection() -> AsyncIOMotorCollection:
    return get_database()["blogs"]


def get_admins_collection() -> AsyncIOMotorCollection:
    return get_database()["admins"]


def get_mail_templates_collection() -> AsyncIOMotorCollection:
    return get_database()["mail_templates"]


def get_mail_sends_collection() -> AsyncIOMotorCollection:
    return get_database()["mail_sends"]


def get_promo_codes_collection() -> AsyncIOMotorCollection:
    return get_database()["promo_codes"]


def get_tickets_collection() -> AsyncIOMotorCollection:
    return get_database()["tickets"]


def get_cancel_subscription_collection() -> AsyncIOMotorCollection:
    return get_database()["cancel_subscription"]


def get_delete_account_request_collection() -> AsyncIOMotorCollection:
    return get_database()["delete_account_request"]


def get_header_meta_tags_collection() -> AsyncIOMotorCollection:
    return get_database()["header_meta_tags"]


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

    payment_history = db["payment_history"]
    await payment_history.create_index("payment_history_id", unique=True)
    await payment_history.create_index("user_id")
    await payment_history.create_index("razorpay_order_id")
    await payment_history.create_index("status")
    await payment_history.create_index("created_at")

    backgrounds = db["backgrounds"]
    await backgrounds.create_index("background_id", unique=True)
    await backgrounds.create_index("user_id")
    await backgrounds.create_index("is_default")
    await backgrounds.create_index("is_favorite")
    await backgrounds.create_index("background_type")
    await backgrounds.create_index("favorite_list")

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
    # Named explicitly: legacy DBs often have non-unique auto-named "pose_id_1";
    # requesting unique=True with the same default name causes IndexKeySpecsConflict.
    await poses_data.create_index(
        [("pose_id", ASCENDING)], unique=True, name="pose_id_unique"
    )
    await poses_data.create_index("user_id")
    await poses_data.create_index("pose_type")
    await poses_data.create_index("is_favorite")
    await poses_data.create_index("is_default")
    await poses_data.create_index("favorite_list")

    templates = db["templates"]
    await templates.create_index("template_id", unique=True)
    await templates.create_index("user_id")
    await templates.create_index([("user_id", ASCENDING), ("is_default", ASCENDING)])

    remove_bg = db["remove_background_data"]
    await remove_bg.create_index("remove_background_id", unique=True)
    await remove_bg.create_index("user_id")
    await remove_bg.create_index("is_active")
    await remove_bg.create_index("is_favorite")
    await remove_bg.create_index("created_at")

    user_up = db["user_upscaled"]
    await user_up.create_index("upscale_id", unique=True)
    await user_up.create_index("user_id")
    await user_up.create_index("is_active")
    await user_up.create_index("is_favorite")
    await user_up.create_index("created_at")

    contact_sales = db["contact_sales"]
    await contact_sales.create_index("submission_id", unique=True)
    await contact_sales.create_index("work_email")
    await contact_sales.create_index("status")
    await contact_sales.create_index("created_at")

    admins = db["admins"]
    await admins.create_index("admin_id", unique=True)
    await admins.create_index("email", unique=True)

    blogs = db["blogs"]
    await blogs.create_index("blog_id", unique=True, name="blog_id_unique")
    await blogs.create_index("blog_url", name="blog_url_lookup")
    await blogs.create_index("status")
    await blogs.create_index("blog_post_date_and_time")
    await blogs.create_index("created_at")

    mail_templates = db["mail_templates"]
    await mail_templates.create_index("mail_template_id", unique=True, name="mail_template_id_unique")
    await mail_templates.create_index("admin_id")
    await mail_templates.create_index("created_at")

    mail_sends = db["mail_sends"]
    await mail_sends.create_index("mail_sender_id", unique=True, name="mail_sender_id_unique")
    await mail_sends.create_index("mail_template_id")
    await mail_sends.create_index("created_at")
    await mail_sends.create_index("admin_id")

    promo_codes = db["promo_codes"]
    await promo_codes.create_index("promo_id", unique=True, name="promo_id_unique")
    await promo_codes.create_index("promo_code", unique=True, name="promo_code_unique")
    await promo_codes.create_index("is_active")
    await promo_codes.create_index("promo_type")
    await promo_codes.create_index("expiry_date")
    await promo_codes.create_index("created_at")

    tickets = db["tickets"]
    await tickets.create_index("ticket_id", unique=True, name="ticket_id_unique")
    await tickets.create_index("user_id")
    await tickets.create_index("status")
    await tickets.create_index("created_at")

    cancel_subscription = db["cancel_subscription"]
    await cancel_subscription.create_index(
        "cancel_subscription_id", unique=True, name="cancel_subscription_id_unique"
    )
    await cancel_subscription.create_index("user_id")
    await cancel_subscription.create_index("created_at")

    delete_account_request = db["delete_account_request"]
    await delete_account_request.create_index(
        "delete-request-id", unique=True, name="delete_request_id_unique"
    )
    await delete_account_request.create_index("user-id")
    await delete_account_request.create_index("created_at")

    header_meta_tags = db["header_meta_tags"]
    await header_meta_tags.create_index(
        "meta_tag_id", unique=True, name="meta_tag_id_unique"
    )
    await header_meta_tags.create_index("created_at")
