from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, EmailStr


# Dashboard roles (must match DB + JWT). Only these may be set on create / role change.
ALLOWED_ADMIN_ROLES: frozenset[str] = frozenset(
    ("superadmin", "admin", "developer", "blogger")
)
AdminRole = Literal["superadmin", "admin", "developer", "blogger"]


def _validate_role(v: str) -> str:
    s = (v or "").strip().lower()
    if s not in ALLOWED_ADMIN_ROLES:
        raise ValueError(
            "role must be one of: superadmin, admin, developer, blogger"
        )
    return s


class AdminLoginRequest(BaseModel):
    email:    EmailStr
    password: str = Field(..., min_length=1)


class AdminLoginVerifyOtpRequest(BaseModel):
    email: EmailStr
    otp:   str = Field(..., min_length=4, max_length=10)


class AdminResendOtpRequest(BaseModel):
    email: EmailStr


class AdminRefreshTokenRequest(BaseModel):
    refresh_token: str = Field(..., min_length=1)


class AdminCreateRequest(BaseModel):
    name:     str = Field(..., min_length=1, max_length=200)
    email:    EmailStr
    password: str = Field(..., min_length=8, max_length=200)
    role:     str = Field(
        ...,
        description="One of: superadmin, admin, developer, blogger",
    )

    @field_validator("role", mode="before")
    @classmethod
    def role_ok(cls, v) -> str:
        return _validate_role(str(v) if v is not None else "")


class AdminChangeRoleRequest(BaseModel):
    role: str = Field(
        ...,
        description="One of: superadmin, admin, developer, blogger",
    )

    @field_validator("role", mode="before")
    @classmethod
    def role_ok(cls, v) -> str:
        return _validate_role(str(v) if v is not None else "")


class AdminUpdateRequest(BaseModel):
    name:  Optional[str] = Field(default=None, max_length=200)
    email: Optional[EmailStr] = None

    @field_validator("name", mode="before")
    @classmethod
    def name_strip(cls, v) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None


class AdminChangePasswordRequest(BaseModel):
    old_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=200)


class AdminBootstrapRequest(BaseModel):
    """Create the first superadmin when the ``admins`` collection is empty (requires ``X-Admin-Bootstrap-Key``)."""

    name:     str = Field(..., min_length=1, max_length=200)
    email:    EmailStr
    password: str = Field(..., min_length=8, max_length=200)


def admin_public_dict(doc: dict) -> dict:
    """Admin record safe for API responses (no password)."""
    d = dict(doc)
    d.pop("_id", None)
    d.pop("password", None)
    return d


class AdminTokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    admin:         dict
