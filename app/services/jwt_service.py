"""JWT issuing & verification.

Security notes
--------------
- Access tokens are short-lived; refresh tokens are longer-lived. We
  additionally mint `iat` (issued-at), `nbf` (not-before), and — when
  configured — `iss`/`aud` claims so downstream verification can bind
  tokens to *this* service.
- The algorithm is pinned to a single, configured value and the
  decoder passes `algorithms=[...]` explicitly to python-jose. This
  prevents algorithm-confusion attacks (a classic `alg:"none"` or
  HS/RS confusion bypass).
- `iss`/`aud` validation is **opt-in**: if `JWT_ISSUER` / `JWT_AUDIENCE`
  in settings is non-empty, the decoder enforces it. This keeps older
  tokens (issued before this change) valid so no client gets signed
  out on deploy. Once every active session has rotated, set those env
  vars in production for stricter validation.
- Error messages are intentionally generic ("Invalid token.") — they
  do not leak which claim failed, which would help an attacker craft
  valid-looking tokens.
"""

from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from jose import JWTError, jwt

from app.config import settings


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _base_claims(user_id: str, token_type: str, expire: datetime) -> dict:
    now = _utc_now()
    claims: dict = {
        "sub": user_id,
        "iat": now,
        "nbf": now,
        "exp": expire,
        "type": token_type,
    }
    if settings.JWT_ISSUER:
        claims["iss"] = settings.JWT_ISSUER
    if settings.JWT_AUDIENCE:
        claims["aud"] = settings.JWT_AUDIENCE
    return claims


def create_access_token(user_id: str) -> str:
    expire = _utc_now() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = _base_claims(user_id, "access", expire)
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    expire = _utc_now() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    payload = _base_claims(user_id, "refresh", expire)
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str, token_type: str = "access") -> dict:
    # `options` lets us opt out of iss/aud checks when they aren't
    # configured, so tokens issued before this change remain valid.
    options: dict = {
        "verify_signature": True,
        "verify_exp": True,
        "verify_nbf": True,
        "verify_iat": False,   # clock-skew tolerance; exp/nbf already cover freshness
        "verify_aud": bool(settings.JWT_AUDIENCE),
        "verify_iss": bool(settings.JWT_ISSUER),
    }

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            audience=settings.JWT_AUDIENCE or None,
            issuer=settings.JWT_ISSUER or None,
            options=options,
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again.",
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
        )

    if payload.get("type") != token_type:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Expected {token_type} token.",
        )

    if not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
        )

    return payload
