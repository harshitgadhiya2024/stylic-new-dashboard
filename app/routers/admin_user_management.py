"""
End-user account management for dashboard admins (JWT: ``/api/v1/admins/*``).

Permissions: **read** = superadmin, admin, developer. **write** = superadmin only.
Blogger is not included → 403 on all routes here.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.config import settings
from app.database import (
    get_cancel_subscription_collection,
    get_credit_history_collection,
    get_payment_history_collection,
    get_users_collection,
)
from app.dependencies import require_user_management_read, require_user_management_write
from app.models.admin_user_management import (
    AddUserCreditsRequest,
    UserBlockUnblockRequest,
    UserPlanPartialRequest,
    UserRoleMappingPartialRequest,
)
from app.utils.user_response import user_dict_for_api

router = APIRouter(
    prefix="/api/v1/admins/user-management",
    tags=["Admin — user management"],
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def _user_by_id(user_id: str) -> dict:
    col = get_users_collection()
    u = await col.find_one({"user_id": user_id})
    if not u:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found for this user_id.",
        )
    return u


@router.get(
    "/users",
    summary="List all end users (paginated)",
)
async def list_users(
    page: int = Query(1, ge=1, description="1-based page number"),
    limit: int = Query(25, ge=1, description="Page size (integer)."),
    _viewer: dict = Depends(require_user_management_read()),
) -> dict[str, Any]:
    _ = _viewer
    col = get_users_collection()
    skip = (page - 1) * int(limit)
    total = await col.count_documents({})
    total_pages = max(1, (total + int(limit) - 1) // int(limit))
    cursor = col.find({}).sort("created_at", -1).skip(skip).limit(int(limit))
    items: list[dict[str, Any]] = []
    async for doc in cursor:
        items.append(user_dict_for_api(doc))
    return {
        "total": total,
        "page": page,
        "limit": int(limit),
        "total_pages": total_pages,
        "users": items,
    }


@router.get(
    "/cancel-subscription-data",
    summary="Get cancel subscription data (admin, paginated)",
    description=(
        "Returns rows from the ``cancel_subscription`` collection (no query filters), "
        "sorted by ``created_at`` descending, with page-based pagination. "
        "Each document omits ``_id``. "
        "Requires dashboard admin JWT with user-management **read** "
        "(**superadmin**, **admin**, **developer**)."
    ),
)
async def admin_get_all_cancel_subscription_data(
    page: int = Query(1, ge=1, description="1-based page number"),
    limit: int = Query(
        100,
        ge=1,
        le=500,
        description="Items per page (max 500).",
    ),
    _viewer: dict = Depends(require_user_management_read()),
) -> dict[str, Any]:
    _ = _viewer
    col = get_cancel_subscription_collection()
    query: dict[str, Any] = {}
    skip = (page - 1) * int(limit)
    total = await col.count_documents(query)
    lim = int(limit)
    total_pages = max(1, (total + lim - 1) // lim) if total else 1

    cursor = (
        col.find(query)
        .sort("created_at", -1)
        .skip(skip)
        .limit(lim)
    )
    rows = await cursor.to_list(length=lim)
    items: list[dict[str, Any]] = []
    for row in rows:
        doc = dict(row)
        doc.pop("_id", None)
        items.append(doc)
    return {
        "total": total,
        "page": page,
        "limit": lim,
        "total_pages": total_pages,
        "cancel_subscription_data": items,
    }


@router.get(
    "/users/{user_id}",
    summary="Get one end user by user_id",
)
async def get_user(
    user_id: str,
    _viewer: dict = Depends(require_user_management_read()),
) -> dict:
    _ = _viewer
    u = await _user_by_id(user_id)
    return user_dict_for_api(u)


@router.get(
    "/users/{user_id}/credit-history",
    summary="Get credit history for a user (admin)",
    description=(
        "Returns ``credit_history`` collection rows for ``user_id``, newest first. "
        "Each item is the full stored document (``_id`` omitted). "
        "Requires dashboard admin JWT with user-management **read** "
        "(**superadmin**, **admin**, **developer**)."
    ),
)
async def admin_get_user_credit_history(
    user_id: str,
    page: int = Query(1, ge=1, description="1-based page number"),
    limit: int = Query(
        100,
        ge=1,
        le=500,
        description="Items per page (max 500).",
    ),
    _viewer: dict = Depends(require_user_management_read()),
) -> dict[str, Any]:
    _ = _viewer
    await _user_by_id(user_id)

    col = get_credit_history_collection()
    skip = (page - 1) * int(limit)
    query = {"user_id": user_id}
    total = await col.count_documents(query)
    lim = int(limit)
    total_pages = max(1, (total + lim - 1) // lim) if total else 1

    cursor = (
        col.find(query)
        .sort("created_at", -1)
        .skip(skip)
        .limit(lim)
    )
    rows = await cursor.to_list(length=lim)
    items: list[dict[str, Any]] = []
    for row in rows:
        doc = dict(row)
        doc.pop("_id", None)
        items.append(doc)

    return {
        "user_id": user_id,
        "total": total,
        "page": page,
        "limit": lim,
        "total_pages": total_pages,
        "credit_history": items,
    }


@router.get(
    "/users/{user_id}/payment-history",
    summary="Get payment history for a user (admin)",
    description=(
        "Returns ``payment_history`` collection rows for ``user_id``, newest first. "
        "Each item is the full stored document (``_id`` omitted). "
        "Requires dashboard admin JWT with user-management **read** "
        "(**superadmin**, **admin**, **developer**)."
    ),
)
async def admin_get_user_payment_history(
    user_id: str,
    page: int = Query(1, ge=1, description="1-based page number"),
    limit: int = Query(
        100,
        ge=1,
        le=500,
        description="Items per page (max 500).",
    ),
    _viewer: dict = Depends(require_user_management_read()),
) -> dict[str, Any]:
    _ = _viewer
    await _user_by_id(user_id)

    col = get_payment_history_collection()
    skip = (page - 1) * int(limit)
    query = {"user_id": user_id}
    total = await col.count_documents(query)
    lim = int(limit)
    total_pages = max(1, (total + lim - 1) // lim) if total else 1

    cursor = (
        col.find(query)
        .sort("created_at", -1)
        .skip(skip)
        .limit(lim)
    )
    rows = await cursor.to_list(length=lim)
    items: list[dict[str, Any]] = []
    for row in rows:
        doc = dict(row)
        doc.pop("_id", None)
        items.append(doc)

    return {
        "user_id": user_id,
        "total": total,
        "page": page,
        "limit": lim,
        "total_pages": total_pages,
        "payment_history": items,
    }


@router.patch(
    "/users/{user_id}/role-mapping",
    summary="Merge into users.role_mapping_dict (partial)",
)
async def patch_user_role_mapping(
    user_id: str,
    body:    UserRoleMappingPartialRequest,
    me:     dict = Depends(require_user_management_write()),
) -> dict:
    _ = me
    u = await _user_by_id(user_id)
    patch = body.model_dump(exclude_none=True)
    if not patch:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one field to update.",
        )
    current = u.get("role_mapping_dict")
    if not isinstance(current, dict):
        from app.constants.free_plan import FREE_ROLE_MAPPING_DICT

        current = dict(FREE_ROLE_MAPPING_DICT)
    else:
        current = {**current}
    current.update(patch)
    col = get_users_collection()
    await col.update_one(
        {"user_id": user_id},
        {"$set": {
            "role_mapping_dict": current,
            "updated_at":         _now(),
        }},
    )
    fresh = await col.find_one({"user_id": user_id})
    return user_dict_for_api(fresh) if fresh else {}


@router.patch(
    "/users/{user_id}/plan",
    summary="Update plan, plan_mapping_dict, and end-user role field (partial)",
)
async def patch_user_plan(
    user_id: str,
    body:    UserPlanPartialRequest,
    me:     dict = Depends(require_user_management_write()),
) -> dict:
    _ = me
    d = body.model_dump(exclude_none=True)
    if not d:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one of: plan, role, start_date, renew_date.",
        )
    u   = await _user_by_id(user_id)
    col = get_users_collection()
    pdoc = u.get("plan_mapping_dict")
    if not isinstance(pdoc, dict):
        from app.constants.free_plan import build_free_plan_mapping_dict

        pdoc = build_free_plan_mapping_dict(
            u.get("created_at") or _now(),
            str(u.get("plan") or settings.DEFAULT_PLAN or "free"),
        )
    else:
        pdoc = {**pdoc}
    set_fields: dict[str, Any] = {"updated_at": _now()}
    if "plan" in d and d["plan"] is not None:
        set_fields["plan"] = d["plan"]
        pdoc["plan"] = d["plan"]
    if "start_date" in d and d["start_date"] is not None:
        pdoc["start_date"] = _as_utc(d["start_date"])
    if "renew_date" in d and d["renew_date"] is not None:
        pdoc["renew_date"] = _as_utc(d["renew_date"])
    if "role" in d and d["role"] is not None:
        set_fields["role"] = d["role"]
    set_fields["plan_mapping_dict"] = pdoc
    await col.update_one({"user_id": user_id}, {"$set": set_fields})
    fresh = await col.find_one({"user_id": user_id})
    return user_dict_for_api(fresh) if fresh else {}


@router.post(
    "/users/{user_id}/credits",
    summary="Add credits to a user; records credit_history",
    status_code=status.HTTP_200_OK,
)
async def add_user_credits(
    user_id: str,
    body:    AddUserCreditsRequest,
    me:     dict = Depends(require_user_management_write()),
) -> dict[str, Any]:
    u = await _user_by_id(user_id)
    col = get_users_collection()
    add = round(float(body.credit), 4)
    old = float(u.get("credits", 0) or 0)
    new = round(old + add, 4)
    now = _now()
    await col.update_one(
        {"user_id": user_id},
        {"$set": {
            "credits":   new,
            "updated_at": now,
        }},
    )
    hist = get_credit_history_collection()
    admin_id = me.get("admin_id", "")
    admin_label = f"admin {admin_id}"
    notes = (body.notes or "").strip() or f"Manual credit grant by {admin_label}"
    history_doc = {
        "history_id":   str(uuid.uuid4()),
        "user_id":      user_id,
        "feature_name": "admin_add_credits",
        "credit":       add,
        "type":         "add",
        "thumbnail_image": "",
        "notes":         notes,
        "created_at":   now,
        "admin_id":     admin_id,
    }
    await hist.insert_one(history_doc)
    return {
        "user_id":     user_id,
        "credits":     new,
        "credit_added": add,
        "credits_before": old,
    }


@router.patch(
    "/users/{user_id}/status",
    summary="Block or unblock user; reason required (audit log)",
)
async def set_user_status(
    user_id: str,
    body:    UserBlockUnblockRequest,
    me:     dict = Depends(require_user_management_write()),
) -> dict:
    u = await _user_by_id(user_id)
    now = _now()
    col = get_users_collection()
    await col.update_one(
        {"user_id": user_id},
        {"$set": {
            "is_active":   body.is_active,
            "updated_at":  now,
            "status_change_reason":  body.reason.strip(),
            "status_change_at":      now,
            "status_changed_by_admin": me.get("admin_id", ""),
        }},
    )
    fresh = await col.find_one({"user_id": user_id})
    return user_dict_for_api(fresh) if fresh else {}
