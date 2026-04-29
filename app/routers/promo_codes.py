"""Promo code APIs (admin CRUD + user validate/apply)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, status
from pymongo import ReturnDocument

from app.database import (
    get_credit_history_collection,
    get_promo_codes_collection,
    get_users_collection,
)
from app.dependencies import get_current_user, require_admin_roles
from app.models.promo_code import (
    CreatePromoCodeRequest,
    PromoCodeInput,
    PromoCodeRecord,
    PromoLookupRequest,
    UpdatePromoCodeRequest,
)

admin_router = APIRouter(
    prefix="/api/v1/admins/promo-codes",
    tags=["Admin — Promo Codes"],
    dependencies=[Depends(require_admin_roles("superadmin", "admin"))],
)

user_router = APIRouter(
    prefix="/api/v1/promo-codes",
    tags=["Promo Codes"],
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _doc_public(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if k != "_id"}


def _is_promo_expired(doc: dict[str, Any]) -> bool:
    exp = doc.get("expiry_date")
    if not isinstance(exp, datetime):
        return False
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)
    return exp <= _now()


def _promo_filter(promo_id: str | None, promo_code: str | None) -> dict[str, Any]:
    if promo_id:
        return {"promo_id": promo_id}
    if promo_code:
        return {"promo_code": promo_code}
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Either promo_id or promo_code is required.",
    )


@admin_router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=PromoCodeRecord,
    summary="Create promo code",
)
async def create_promo_code(body: CreatePromoCodeRequest) -> dict[str, Any]:
    col = get_promo_codes_collection()
    existing = await col.find_one({"promo_code": body.promo_code})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Promo code already exists.",
        )
    now = _now()
    doc = {
        "promo_id": str(uuid.uuid4()),
        "promo_code": body.promo_code,
        "promo_type": body.promo_type,
        "promo_discount": int(body.promo_discount),
        "promo_credit": int(body.promo_credit),
        "expiry_date": body.expiry_date,
        "is_active": True,
        "created_at": now,
        "updated_at": now,
    }
    await col.insert_one(doc)
    return _doc_public(doc)


@admin_router.get(
    "",
    summary="Get all promo codes (paginated)",
)
async def get_all_promo_codes(
    page: int = Query(1, ge=1, description="1-based page number"),
    limit: int = Query(25, ge=1, description="Page size (integer)."),
) -> dict[str, Any]:
    col = get_promo_codes_collection()
    skip = (page - 1) * int(limit)
    total = await col.count_documents({})
    total_pages = max(1, (total + int(limit) - 1) // int(limit))
    cur = col.find({}).sort("created_at", -1).skip(skip).limit(int(limit))
    items = [_doc_public(d) async for d in cur]
    return {
        "total": total,
        "page": page,
        "limit": int(limit),
        "total_pages": total_pages,
        "promo_codes": items,
    }


@admin_router.post(
    "/get-specific",
    response_model=PromoCodeRecord,
    summary="Get specific promo by promo_id or promo_code",
)
async def get_specific_promo(body: PromoLookupRequest) -> dict[str, Any]:
    col = get_promo_codes_collection()
    doc = await col.find_one(_promo_filter(body.promo_id, body.promo_code))
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Promo code not found.")
    return _doc_public(doc)


@admin_router.delete(
    "/{promo_id}",
    response_model=PromoCodeRecord,
    summary="Delete promo code (soft delete)",
)
async def delete_promo_code(
    promo_id: str = Path(..., min_length=1, max_length=64),
) -> dict[str, Any]:
    col = get_promo_codes_collection()
    doc = await col.find_one_and_update(
        {"promo_id": promo_id},
        {"$set": {"is_active": False, "updated_at": _now()}},
        return_document=ReturnDocument.AFTER,
    )
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Promo code not found.")
    return _doc_public(doc)


@admin_router.patch(
    "",
    response_model=PromoCodeRecord,
    summary="Update promo code by promo_id or promo_code",
)
async def update_promo_code(body: UpdatePromoCodeRequest) -> dict[str, Any]:
    col = get_promo_codes_collection()
    selector = _promo_filter(body.promo_id, body.promo_code)
    existing = await col.find_one(selector)
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Promo code not found.")

    set_fields: dict[str, Any] = {"updated_at": _now()}
    for key in ("promo_type", "promo_discount", "promo_credit", "expiry_date", "is_active"):
        value = getattr(body, key)
        if value is not None:
            set_fields[key] = value
    if body.promo_code is not None:
        if body.promo_code != existing.get("promo_code"):
            dup = await col.find_one({"promo_code": body.promo_code})
            if dup:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Promo code already exists.",
                )
        set_fields["promo_code"] = body.promo_code

    effective_type = str(set_fields.get("promo_type", existing.get("promo_type")))
    effective_discount = int(set_fields.get("promo_discount", existing.get("promo_discount", 0)))
    effective_credit = int(set_fields.get("promo_credit", existing.get("promo_credit", 0)))
    if effective_type == "credit":
        if effective_credit <= 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="promo_credit must be > 0 for credit promo_type.",
            )
        set_fields["promo_discount"] = 0
    elif effective_type == "discount":
        if effective_discount <= 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="promo_discount must be > 0 for discount promo_type.",
            )
        set_fields["promo_credit"] = 0

    await col.update_one({"promo_id": existing["promo_id"]}, {"$set": set_fields})
    fresh = await col.find_one({"promo_id": existing["promo_id"]})
    return _doc_public(fresh)  # type: ignore[arg-type]


@user_router.post(
    "/validate",
    response_model=PromoCodeRecord,
    summary="Validate promo code (user)",
)
async def validate_promo_code(
    body: PromoCodeInput,
    _user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    _ = _user
    col = get_promo_codes_collection()
    doc = await col.find_one({"promo_code": body.promo_code, "is_active": True})
    if not doc or _is_promo_expired(doc):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired promo code.")
    return _doc_public(doc)


@user_router.post(
    "/apply",
    summary="Apply promo code (user)",
)
async def apply_promo_code(
    body: PromoCodeInput,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    user_id = str(user.get("user_id") or "")
    col = get_promo_codes_collection()
    promo = await col.find_one({"promo_code": body.promo_code, "is_active": True})
    if not promo or _is_promo_expired(promo):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired promo code.")

    promo_type = str(promo.get("promo_type") or "")
    if promo_type != "credit":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Currently we only support credit promo codes.",
        )

    history_col = get_credit_history_collection()
    already_used = await history_col.find_one(
        {
            "user_id": user_id,
            "$or": [
                {"promo_id": promo.get("promo_id")},
                {"promo_code": promo.get("promo_code")},
            ],
        }
    )
    if already_used:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You already used this code type.",
        )

    promo_credit = round(float(promo.get("promo_credit", 0) or 0), 4)
    if promo_credit <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Promo credit must be greater than zero.",
        )

    users_col = get_users_collection()
    old_credits = float(user.get("credits", 0) or 0)
    new_credits = round(old_credits + promo_credit, 4)
    now = _now()
    await users_col.update_one(
        {"user_id": user_id},
        {"$set": {"credits": new_credits, "updated_at": now}},
    )

    history_doc = {
        "history_id": str(uuid.uuid4()),
        "user_id": user_id,
        "feature_name": "promo_code_credit",
        "credit": promo_credit,
        "type": "add",
        "thumbnail_image": "",
        "notes": f"Promo applied: {promo.get('promo_code')} ({promo.get('promo_id')})",
        "created_at": now,
        "promo_id": promo.get("promo_id"),
        "promo_code": promo.get("promo_code"),
        "promo_type": promo.get("promo_type"),
        "promo_credit": promo_credit,
        "promo_discount": promo.get("promo_discount", 0),
    }
    await history_col.insert_one(history_doc)

    return {
        "message": "Promo code applied successfully.",
        "promo_details": _doc_public(promo),
        "user_id": user_id,
        "credits_before": round(old_credits, 4),
        "credit_added": promo_credit,
        "credits_after": new_credits,
    }
