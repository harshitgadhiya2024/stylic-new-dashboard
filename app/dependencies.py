from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import settings
from app.services.jwt_service import decode_token
from app.services.admin_jwt_service import decode_admin_token
from app.database import get_users_collection, get_admins_collection
from app.utils.user_response import user_dict_for_api

bearer_scheme = HTTPBearer()
admin_bearer_scheme = HTTPBearer()


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


async def get_current_admin(
    credentials: HTTPAuthorizationCredentials = Depends(admin_bearer_scheme),
) -> dict:
    """Validate admin **access** JWT (Authorization: Bearer …) and load the admin document."""
    payload = decode_admin_token(credentials.credentials, token_type="admin_access")
    admin_id = payload.get("sub")
    token_role = payload.get("role")
    if not admin_id or not token_role:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token payload.",
        )
    col = get_admins_collection()
    admin = await col.find_one({"admin_id": admin_id, "is_active": True})
    if not admin:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin not found or account deactivated.",
        )
    if (admin.get("role") or "") != str(token_role):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin role changed. Please sign in again.",
        )
    return admin


def require_admin_roles(*allowed_roles: str):
    """Dependency factory: require the admin's role to be one of ``allowed_roles``."""

    async def _dep(admin: dict = Depends(get_current_admin)) -> dict:
        r = admin.get("role")
        if r not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for this action.",
            )
        return admin

    return _dep


# End-user management (`/api/v1/admins/user-management/*`):
#   superadmin + admin + developer: read; superadmin: write. Blogger: no access (omitted from allowed).
def require_user_management_read():
    return require_admin_roles("superadmin", "admin", "developer")


def require_user_management_write():
    return require_admin_roles("superadmin")


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
