from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
