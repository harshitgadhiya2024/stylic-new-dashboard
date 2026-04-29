"""
Dashboard admin API — separate JWT from end-user auth (`/api/v1/auth`).

- Public: login → OTP email → verify OTP → access + refresh tokens.
- Secured: ``Authorization: Bearer <admin access token>``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, List

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Path, status

from app.config import settings
from app.database import get_admins_collection
from app.dependencies import get_current_admin, require_admin_roles
from app.models.admin import (
    AdminBootstrapRequest,
    AdminChangePasswordRequest,
    AdminChangeRoleRequest,
    AdminCreateRequest,
    AdminLoginRequest,
    AdminLoginVerifyOtpRequest,
    AdminRefreshTokenRequest,
    AdminResendOtpRequest,
    AdminTokenResponse,
    AdminUpdateRequest,
    admin_public_dict,
)
from app.models.user import MessageResponse
from app.services.admin_jwt_service import (
    create_admin_access_token,
    create_admin_refresh_token,
    decode_admin_token,
)
from app.services.email_service import send_otp_email
from app.services.otp_service import (
    generate_otp,
    save_otp,
    verify_otp,
    consume_otp,
)
from pymongo import ReturnDocument
from app.utils.password import hash_password, verify_password, validate_password_strength

router = APIRouter(prefix="/api/v1/admins", tags=["Admin dashboard"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _public(doc: dict) -> dict:
    return admin_public_dict(doc)


def _build_token_bundle(doc: dict) -> dict:
    a = {**doc}
    a.pop("_id", None)
    return {
        "access_token":  create_admin_access_token(doc["admin_id"], doc["role"]),
        "refresh_token": create_admin_refresh_token(doc["admin_id"]),
        "token_type":     "bearer",
        "admin":          _public(a),
    }


async def _count_superadmins() -> int:
    col = get_admins_collection()
    return await col.count_documents({"role": "superadmin", "is_active": True})


# ── Public: bootstrap (optional first superadmin) ────────────────────────────


@router.post(
    "/auth/bootstrap",
    response_model=AdminTokenResponse,
    summary="Bootstrap first superadmin (one-time, when collection is empty)",
    description="Requires env ``ADMIN_BOOTSTRAP_KEY`` and matching header ``X-Admin-Bootstrap-Key``.",
)
async def admin_bootstrap(
    body: AdminBootstrapRequest,
    x_bootstrap: str | None = Header(default=None, alias="X-Admin-Bootstrap-Key"),
):
    key = (getattr(settings, "ADMIN_BOOTSTRAP_KEY", "") or "").strip()
    if not key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin bootstrap is not enabled (set ADMIN_BOOTSTRAP_KEY).",
        )
    if (x_bootstrap or "").strip() != key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Admin-Bootstrap-Key.",
        )
    col = get_admins_collection()
    if await col.count_documents({}) > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one admin already exists. Use a superadmin account to add more.",
        )
    err = validate_password_strength(body.password)
    if err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err)
    email = str(body.email).lower().strip()
    now = _now()
    admin_id = str(uuid.uuid4())
    doc: dict[str, Any] = {
        "admin_id":    admin_id,
        "name":        body.name.strip(),
        "email":       email,
        "password":    hash_password(body.password),
        "role":        "superadmin",
        "is_active":   True,
        "created_at":  now,
        "updated_at":  now,
    }
    try:
        await col.insert_one(doc)
    except Exception as exc:  # noqa: BLE001
        if "duplicate" in str(exc).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An admin with this email already exists.",
            ) from exc
        raise
    doc.pop("_id", None)
    return _build_token_bundle(doc)


# ── Public: login + OTP ──────────────────────────────────────────────────────


@router.post(
    "/auth/login",
    response_model=MessageResponse,
    summary="Admin login – step 1: validate email/password and send OTP",
)
async def admin_login(
    body: AdminLoginRequest,
    background_tasks: BackgroundTasks,
):
    col = get_admins_collection()
    email = str(body.email).lower().strip()
    admin = await col.find_one({"email": email, "is_active": True})
    if not admin:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    if not verify_password(body.password, admin.get("password", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    otp = generate_otp()
    await save_otp(email=email, otp=otp, purpose="admin_login")
    background_tasks.add_task(
        send_otp_email,
        to_email=email,
        otp=otp,
        purpose="admin_login",
    )
    return {
        "success": True,
        "message":  f"OTP sent to {email}. Valid for {settings.OTP_EXPIRE_MINUTES} minutes.",
    }


@router.post(
    "/auth/verify-otp",
    response_model=AdminTokenResponse,
    summary="Admin login – step 2: verify OTP and return admin tokens",
)
async def admin_verify_otp(body: AdminLoginVerifyOtpRequest):
    email = str(body.email).lower().strip()
    await verify_otp(email=email, otp=body.otp, purpose="admin_login")
    col = get_admins_collection()
    admin = await col.find_one({"email": email, "is_active": True})
    if not admin:
        await consume_otp(email, "admin_login")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Admin not found.",
        )
    await consume_otp(email, "admin_login")
    admin = {**admin}
    admin.pop("_id", None)
    return _build_token_bundle(admin)


@router.post(
    "/auth/resend-otp",
    response_model=MessageResponse,
    summary="Admin login – resend OTP",
)
async def admin_resend_otp(
    body: AdminResendOtpRequest,
    background_tasks: BackgroundTasks,
):
    email = str(body.email).lower().strip()
    col = get_admins_collection()
    if not await col.find_one({"email": email, "is_active": True}):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Admin not found for this email.",
        )
    otp = generate_otp()
    await save_otp(email=email, otp=otp, purpose="admin_login")
    background_tasks.add_task(
        send_otp_email, to_email=email, otp=otp, purpose="admin_login"
    )
    return {"success": True, "message": f"OTP resent to {email}."}


@router.post(
    "/auth/refresh-token",
    response_model=AdminTokenResponse,
    summary="Refresh admin access & refresh tokens",
)
async def admin_refresh_token(body: AdminRefreshTokenRequest):
    payload = decode_admin_token(body.refresh_token, token_type="admin_refresh")
    admin_id = payload.get("sub")
    col = get_admins_collection()
    admin = await col.find_one({"admin_id": admin_id, "is_active": True})
    if not admin:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin not found or deactivated.",
        )
    admin = {**admin}
    admin.pop("_id", None)
    return _build_token_bundle(admin)


# ── Secured: self + listing + superadmin management ─────────────────────────

@router.post(
    "/me/change-password",
    response_model=MessageResponse,
    summary="Change my password (any admin role; admin access token required)",
)
async def admin_change_my_password(
    body: AdminChangePasswordRequest,
    me: dict = Depends(get_current_admin),
):
    col = get_admins_collection()
    if not verify_password(body.old_password, me.get("password", "")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect.",
        )
    err = validate_password_strength(body.new_password)
    if err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err)
    await col.update_one(
        {"admin_id": me["admin_id"]},
        {"$set": {
            "password":   hash_password(body.new_password),
            "updated_at": _now(),
        }},
    )
    return {"success": True, "message": "Password updated successfully."}


@router.get(
    "",
    response_model=List[dict],
    summary="List all admins (superadmin, admin, developer, blogger)",
)
async def list_admins(
    _viewer: dict = Depends(
        require_admin_roles("superadmin", "admin", "developer", "blogger")
    ),
) -> list[dict]:
    col = get_admins_collection()
    cursor = col.find({}).sort("created_at", 1)
    out: list[dict] = []
    async for doc in cursor:
        out.append(_public(doc))
    return out


@router.post(
    "",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    summary="Add a new admin (superadmin only); returns the new admin (no password)",
)
async def create_admin(
    body: AdminCreateRequest,
    _me: dict = Depends(require_admin_roles("superadmin")),
):
    col = get_admins_collection()
    err = validate_password_strength(body.password)
    if err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err)
    email = str(body.email).lower().strip()
    now = _now()
    admin_id = str(uuid.uuid4())
    role = str(body.role).lower().strip()
    doc: dict[str, Any] = {
        "admin_id":    admin_id,
        "name":        body.name.strip(),
        "email":       email,
        "password":    hash_password(body.password),
        "role":        role,
        "is_active":   True,
        "created_at":  now,
        "updated_at":  now,
    }
    try:
        await col.insert_one(doc)
    except Exception as exc:  # noqa: BLE001
        if "duplicate" in str(exc).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An admin with this email already exists.",
            ) from exc
        raise
    return _public(doc)


@router.patch(
    "/{admin_id}/role",
    response_model=dict,
    summary="Change an admin's role (superadmin only); request body: { role }",
)
async def change_admin_role(
    admin_id: str = Path(
        ...,
        pattern=r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
        description="Admin UUID",
    ),
    body: AdminChangeRoleRequest,
    _me: dict = Depends(require_admin_roles("superadmin")),
):
    col = get_admins_collection()
    target = await col.find_one({"admin_id": admin_id, "is_active": True})
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Admin not found.",
        )
    new_role = body.role
    old_role = (target.get("role") or "").lower()
    if new_role == old_role:
        return _public(target)

    if old_role == "superadmin" and new_role != "superadmin":
        if await _count_superadmins() <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot change role: you must keep at least one superadmin.",
            )
    r = await col.find_one_and_update(
        {"admin_id": admin_id, "is_active": True},
        {"$set": {"role": new_role, "updated_at": _now()}},
        return_document=ReturnDocument.AFTER,
    )
    if not r:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Admin not found.")
    r.pop("_id", None)
    return _public(r)


@router.patch(
    "/{admin_id}",
    response_model=dict,
    summary="Update an admin's name and/or email (superadmin only)",
)
async def update_admin(
    admin_id: str = Path(
        ...,
        pattern=r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
        description="Admin UUID",
    ),
    body: AdminUpdateRequest,
    _me: dict = Depends(require_admin_roles("superadmin")),
):
    if body.name is None and body.email is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one of: name, email.",
        )
    col = get_admins_collection()
    target = await col.find_one({"admin_id": admin_id, "is_active": True})
    if not target:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Admin not found.",
        )
    set_fields: dict[str, Any] = {"updated_at": _now()}
    if body.name is not None:
        set_fields["name"] = body.name.strip()
    if body.email is not None:
        new_email = str(body.email).lower().strip()
        if new_email != (target.get("email") or ""):
            dup = await col.find_one({
                "email": new_email,
                "admin_id": {"$ne": admin_id},
            })
            if dup:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="That email is already in use by another admin.",
                )
        set_fields["email"] = new_email
    await col.update_one({"admin_id": admin_id}, {"$set": set_fields})
    fresh = await col.find_one({"admin_id": admin_id})
    if not fresh:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Admin not found.")
    return _public(fresh)