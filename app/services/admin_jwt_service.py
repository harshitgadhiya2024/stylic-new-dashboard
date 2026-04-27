"""JWT helpers for dashboard admin users (separate secret from end-user auth)."""

from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from fastapi import HTTPException, status

from app.config import settings


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _admin_secret() -> str:
    s = (settings.ADMIN_JWT_SECRET_KEY or "").strip()
    if s:
        return s
    return settings.JWT_SECRET_KEY


def create_admin_access_token(admin_id: str, role: str) -> str:
    exp = _utc_now() + timedelta(minutes=settings.ADMIN_ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": admin_id,
        "role": role,
        "exp": exp,
        "type": "admin_access",
    }
    return jwt.encode(payload, _admin_secret(), algorithm=settings.JWT_ALGORITHM)


def create_admin_refresh_token(admin_id: str) -> str:
    exp = _utc_now() + timedelta(days=settings.ADMIN_REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {"sub": admin_id, "exp": exp, "type": "admin_refresh"}
    return jwt.encode(payload, _admin_secret(), algorithm=settings.JWT_ALGORITHM)


def decode_admin_token(token: str, token_type: str) -> dict:
    try:
        payload = jwt.decode(
            token, _admin_secret(), algorithms=[settings.JWT_ALGORITHM]
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin token has expired. Please sign in again.",
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token.",
        )
    if payload.get("type") != token_type:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Expected {token_type} admin token.",
        )
    return payload
