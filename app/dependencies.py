from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings
from app.services.jwt_service import decode_token
from app.database import get_users_collection
from app.utils.user_response import user_dict_for_api

bearer_scheme = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    payload = decode_token(credentials.credentials, token_type="access")
    user_id = payload.get("sub")

    col = get_users_collection()
    user = await col.find_one({"user_id": user_id, "is_active": True})

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or account deactivated.",
        )

    return user_dict_for_api(user)


def verify_admin_api_key(
    x_admin_api_key: str | None = Header(default=None, alias="X-Admin-API-Key"),
) -> None:
    """Shared secret for server-side admin panel calls (not end-user JWT)."""
    if not (settings.ADMIN_API_KEY and settings.ADMIN_API_KEY.strip()):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API is not configured (set ADMIN_API_KEY in the server environment).",
        )
    if not x_admin_api_key or x_admin_api_key.strip() != settings.ADMIN_API_KEY.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-API-Key header.",
        )
