from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Query, status

from app.database import (
    get_backgrounds_collection,
    get_credit_history_collection,
    get_model_faces_collection,
    get_payment_history_collection,
    get_photoshoots_collection,
    get_poses_collection,
    get_remove_background_collection,
    get_user_upscaled_collection,
    get_users_collection,
)
from app.dependencies import get_current_user
from app.models.user import (
    ChangeEmailRequest,
    ChangePasswordRequest,
    MessageResponse,
    PartialNotificationPreferences,
    StoreOnboardingRequest,
    UpdateUserRequest,
    UserSchema,
    VerifyEmailChangeRequest,
)
from app.routers.auth import _generate_username
from app.services.email_service import send_otp_email
from app.services.otp_service import generate_otp, save_otp, verify_otp, consume_otp
from app.services.r2_service import upload_file_to_r2
from app.utils.password import hash_password, verify_password, validate_password_strength
from app.utils.user_response import user_dict_for_api_with_credit_metrics

router = APIRouter(prefix="/api/v1/user", tags=["User"])

_ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "application/pdf", "video/mp4",
}


# ─────────────────────────── Helpers ──────────────────────────────────────

async def _clean_user(user: dict) -> dict:
    return await user_dict_for_api_with_credit_metrics(user)


async def _sum_numeric_field(collection, match: dict, field_name: str) -> float:
    pipeline = [
        {"$match": match},
        {
            "$group": {
                "_id": None,
                "total": {
                    "$sum": {
                        "$convert": {
                            "input": f"${field_name}",
                            "to": "double",
                            "onError": 0.0,
                            "onNull": 0.0,
                        }
                    }
                },
            }
        },
    ]
    rows = await collection.aggregate(pipeline).to_list(length=1)
    if not rows:
        return 0.0
    return float(rows[0].get("total", 0.0) or 0.0)


def _feature_percentages(feature_counts: dict[str, int]) -> dict[str, float]:
    total = sum(max(0, int(v)) for v in feature_counts.values())
    if total <= 0:
        return {k: 0.0 for k in feature_counts}

    raw_bp = {k: (max(0, int(v)) * 10000) / total for k, v in feature_counts.items()}
    floored_bp = {k: int(v) for k, v in raw_bp.items()}
    remainder = 10000 - sum(floored_bp.values())
    order = sorted(raw_bp.keys(), key=lambda k: (raw_bp[k] - floored_bp[k]), reverse=True)
    for idx in range(remainder):
        floored_bp[order[idx % len(order)]] += 1
    return {k: round(floored_bp[k] / 100.0, 2) for k in feature_counts}


# ══════════════════════════════════════════════════════════════════════════
# ONBOARDING
# ══════════════════════════════════════════════════════════════════════════

@router.post(
    "/store-onboarding",
    response_model=UserSchema,
    summary="Store onboarding data",
    description=(
        "Persists onboarding answers on the authenticated user document (`onboarding` subdocument). "
        "Requires a valid access token; `user_id` is taken from the token."
    ),
)
async def store_onboarding(
    body: StoreOnboardingRequest,
    current_user: dict = Depends(get_current_user),
):
    now = datetime.now(timezone.utc)
    onboarding = body.model_dump(mode="python", by_alias=False)
    onboarding["stored_at"] = now

    col = get_users_collection()
    await col.update_one(
        {"user_id": current_user["user_id"]},
        {"$set": {"onboarding": onboarding, "updated_at": now}},
    )
    updated = await col.find_one({"user_id": current_user["user_id"]})
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    return await _clean_user(updated)


# ══════════════════════════════════════════════════════════════════════════
# PROFILE
# ══════════════════════════════════════════════════════════════════════════

@router.get(
    "/me",
    response_model=UserSchema,
    summary="Get My Profile",
)
async def get_me(current_user: dict = Depends(get_current_user)):
    return await _clean_user(current_user)


@router.get(
    "/dashboard-analytics",
    summary="Get user dashboard analytics",
    description="Returns user-level spend, credits, generation counts, feature usage counts, and feature usage percentages.",
)
async def get_dashboard_analytics(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]

    payment_history_col = get_payment_history_collection()
    photoshoots_col = get_photoshoots_collection()
    backgrounds_col = get_backgrounds_collection()
    model_faces_col = get_model_faces_collection()
    poses_col = get_poses_collection()
    credit_history_col = get_credit_history_collection()
    remove_bg_col = get_remove_background_collection()
    user_upscaled_col = get_user_upscaled_collection()

    total_cost = await _sum_numeric_field(
        payment_history_col,
        {"user_id": user_id},
        "amount_converted",
    )
    available_credits = round(float(current_user.get("credits", 0) or 0), 4)
    generations = await photoshoots_col.count_documents({"user_id": user_id})

    feature_counts = {
        "total_custom_background": await backgrounds_col.count_documents(
            {"user_id": user_id, "is_default": False}
        ),
        "total_custom_models": await model_faces_col.count_documents(
            {"user_id": user_id, "is_default": False}
        ),
        "total_custom_poses": await poses_col.count_documents(
            {"user_id": user_id, "is_default": False}
        ),
        "total_background_change": await credit_history_col.count_documents(
            {"user_id": user_id, "regeneration_type": "background_change"}
        ),
        "total_regerated": await credit_history_col.count_documents(
            {"user_id": user_id, "regeneration_type": "regenerate"}
        ),
        "total_color_change": await credit_history_col.count_documents(
            {"user_id": user_id, "regeneration_type": "color_change"}
        ),
        "total_upscaled": await credit_history_col.count_documents(
            {
                "user_id": user_id,
                "regeneration_type": {"$in": ["upscale (4x)", "upscale (2x)", "upscale (8x)"]},
            }
        ),
        "total_fabric_change": await credit_history_col.count_documents(
            {"user_id": user_id, "regeneration_type": "fabric_change"}
        ),
        "total_branding": await credit_history_col.count_documents(
            {"user_id": user_id, "regeneration_type": "branding"}
        ),
        "total_remove_background": await remove_bg_col.count_documents({"user_id": user_id}),
        "total_users_upscaled": await user_upscaled_col.count_documents({"user_id": user_id}),
    }

    return {
        "total_cost": round(total_cost, 4),
        "available_credits": available_credits,
        "generations": generations,
        "used_features": feature_counts,
        "used_feature_percentages": _feature_percentages(feature_counts),
    }


@router.get(
    "/credit-history",
    summary="Get my credit history (paginated)",
    description="Returns paginated credit history records for the authenticated user with selected fields only.",
)
async def get_my_credit_history(
    page: int = Query(1, ge=1, description="1-based page number"),
    limit: Literal[25, 50, 75, 100] = Query(
        25,
        description="Page size. Supported values: 25, 50, 75, 100.",
    ),
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    col = get_credit_history_collection()
    skip = (page - 1) * int(limit)
    total = await col.count_documents({"user_id": user_id})
    total_pages = max(1, (total + int(limit) - 1) // int(limit))

    cur = (
        col.find({"user_id": user_id})
        .sort("created_at", -1)
        .skip(skip)
        .limit(int(limit))
    )

    items = []
    async for row in cur:
        items.append(
            {
                "history_id": row.get("history_id", ""),
                "type": row.get("type", ""),
                "feature_name": row.get("feature_name", ""),
                "credit": row.get("credit", 0),
                "updated_at": row.get("updated_at") or row.get("created_at"),
            }
        )

    return {
        "total": total,
        "page": page,
        "limit": int(limit),
        "total_pages": total_pages,
        "credit_history": items,
    }


@router.get(
    "/get-invoices",
    summary="Get my invoices",
    description="Returns all payment_history records for the authenticated user.",
)
async def get_my_invoices(
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    col = get_payment_history_collection()
    cur = col.find({"user_id": user_id}).sort("created_at", -1)
    invoices = []
    async for row in cur:
        row.pop("_id", None)
        invoices.append(row)
    return {"invoices": invoices}


@router.put(
    "/me",
    response_model=UserSchema,
    summary="Update My Profile",
)
async def update_me(body: UpdateUserRequest, current_user: dict = Depends(get_current_user)):
    updates: dict = {"updated_at": datetime.now(timezone.utc)}

    for field, value in body.model_dump(exclude_unset=True).items():
        if field == "notifications" and isinstance(value, dict):
            for notif_key, notif_val in value.items():
                if notif_val is not None:
                    updates[f"notifications.{notif_key}"] = notif_val
        elif value is not None:
            updates[field] = value

    if body.user_name is None and any(
        f in updates for f in ("first_name", "last_name", "phone_number")
    ):
        first_name = updates.get("first_name", current_user.get("first_name", ""))
        last_name = updates.get("last_name", current_user.get("last_name", ""))
        phone_number = updates.get("phone_number", current_user.get("phone_number", ""))
        updates["user_name"] = _generate_username(first_name, last_name, phone_number)

    if len(updates) == 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields provided to update.",
        )

    col = get_users_collection()
    await col.update_one(
        {"user_id": current_user["user_id"]},
        {"$set": updates},
    )

    updated = await col.find_one({"user_id": current_user["user_id"]})
    return await _clean_user(updated)


@router.delete(
    "/me",
    response_model=MessageResponse,
    summary="Delete My Account (Soft Delete)",
)
async def delete_me(current_user: dict = Depends(get_current_user)):
    col = get_users_collection()
    await col.update_one(
        {"user_id": current_user["user_id"]},
        {"$set": {"is_active": False, "updated_at": datetime.now(timezone.utc)}},
    )
    return {"success": True, "message": "Account deactivated successfully."}


# ══════════════════════════════════════════════════════════════════════════
# CHANGE PASSWORD
# ══════════════════════════════════════════════════════════════════════════

@router.put(
    "/change-password",
    response_model=MessageResponse,
    summary="Change Password",
)
async def change_password(body: ChangePasswordRequest, current_user: dict = Depends(get_current_user)):
    if current_user.get("auth_provider") == "google":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google-authenticated accounts cannot change passwords here.",
        )

    col = get_users_collection()
    user = await col.find_one({"user_id": current_user["user_id"]})

    if not verify_password(body.old_password, user.get("password", "")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect.",
        )

    err = validate_password_strength(body.new_password)
    if err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=err)

    await col.update_one(
        {"user_id": current_user["user_id"]},
        {
            "$set": {
                "password": hash_password(body.new_password),
                "updated_at": datetime.now(timezone.utc),
            }
        },
    )
    return {"success": True, "message": "Password updated successfully."}


# ══════════════════════════════════════════════════════════════════════════
# CHANGE EMAIL
# ══════════════════════════════════════════════════════════════════════════

@router.put(
    "/me/change-email",
    response_model=MessageResponse,
    summary="Change Email – Step 1: Send OTP to New Email",
    description="Send a verification OTP to the new email address. Call verify-otp to confirm the change.",
)
async def change_email(
    body: ChangeEmailRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    col = get_users_collection()

    if body.new_email == current_user.get("email"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New email is the same as the current email.",
        )

    if await col.find_one({"email": body.new_email}):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This email is already in use by another account.",
        )

    otp = generate_otp()
    await save_otp(
        email=body.new_email,
        otp=otp,
        purpose="change_email",
        extra={"user_id": current_user["user_id"]},
    )
    background_tasks.add_task(send_otp_email, to_email=body.new_email, otp=otp, purpose="change_email")

    return {
        "success": True,
        "message": f"OTP sent to {body.new_email}. Please verify to complete the email change.",
    }


@router.post(
    "/me/change-email/resend-otp",
    response_model=MessageResponse,
    summary="Change Email – Resend OTP",
)
async def change_email_resend_otp(
    body: ChangeEmailRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    col = get_users_collection()

    if body.new_email == current_user.get("email"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New email is the same as the current email.",
        )

    if await col.find_one({"email": body.new_email}):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This email is already in use by another account.",
        )

    otp = generate_otp()
    await save_otp(
        email=body.new_email,
        otp=otp,
        purpose="change_email",
        extra={"user_id": current_user["user_id"]},
    )
    background_tasks.add_task(send_otp_email, to_email=body.new_email, otp=otp, purpose="change_email")

    return {"success": True, "message": f"OTP resent to {body.new_email}."}


@router.post(
    "/me/change-email/verify-otp",
    response_model=UserSchema,
    summary="Change Email – Step 2: Verify OTP & Update Email",
)
async def change_email_verify_otp(
    body: VerifyEmailChangeRequest,
    current_user: dict = Depends(get_current_user),
):
    record = await verify_otp(email=body.new_email, otp=body.otp, purpose="change_email")

    if record.get("user_id") != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="OTP was not issued for this account.",
        )

    col = get_users_collection()

    if await col.find_one({"email": body.new_email}):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This email is already in use by another account.",
        )

    await col.update_one(
        {"user_id": current_user["user_id"]},
        {"$set": {"email": body.new_email, "updated_at": datetime.now(timezone.utc)}},
    )
    await consume_otp(body.new_email, "change_email")

    updated = await col.find_one({"user_id": current_user["user_id"]})
    return await _clean_user(updated)


# ══════════════════════════════════════════════════════════════════════════
# FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════════

@router.post(
    "/upload-file",
    summary="Upload File to Cloudflare R2",
    description="Upload a file (image, PDF, video) and receive a public URL.",
)
async def upload_file(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    if file.content_type not in _ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {', '.join(_ALLOWED_MIME_TYPES)}",
        )

    user_id = current_user["user_id"]
    url = await upload_file_to_r2(file, folder=f"users/{user_id}")

    return {"success": True, "url": url}
