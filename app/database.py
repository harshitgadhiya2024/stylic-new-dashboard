from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from app.config import settings

_client: MongoClient = None


def get_db_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(settings.MONGO_URI)
    return _client


def get_database():
    return get_db_client()[settings.MONGO_DB_NAME]


def get_users_collection() -> Collection:
    return get_database()["users"]


def get_otps_collection() -> Collection:
    col = get_database()["otps"]
    col.create_index("expires_at", expireAfterSeconds=0)
    col.create_index([("email", ASCENDING), ("purpose", ASCENDING)])
    return col


def get_refresh_tokens_collection() -> Collection:
    col = get_database()["refresh_tokens"]
    col.create_index("expires_at", expireAfterSeconds=0)
    return col


def get_model_faces_collection() -> Collection:
    col = get_database()["model_faces"]
    col.create_index("model_id", unique=True)
    col.create_index("user_id")
    col.create_index("model_category")
    return col
