from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, status

from app.database import get_users_collection
from app.dependencies import get_current_user
from app.models.user import (
    UpdateUserRequest,
    ChangePasswordRequest,
    ChangeEmailRequest,
    VerifyEmailChangeRequest,
    MessageResponse,
    UserSchema,
    PartialNotificationPreferences,
)
from app.routers.auth import _generate_username
from app.services.email_service import send_otp_email
from app.services.otp_service import generate_otp, save_otp, verify_otp, consume_otp
from app.services.s3_service import upload_file_to_s3
from app.utils.password import hash_password, verify_password, validate_password_strength

router = APIRouter(prefix="/api/v1/user", tags=["User"])

_ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "application/pdf", "video/mp4",
}


# ─────────────────────────── Helpers ──────────────────────────────────────

def _clean_user(user: dict) -> dict:
    user = dict(user)
    user.pop("_id", None)
    user.pop("password", None)
    return user


# ══════════════════════════════════════════════════════════════════════════
# PROFILE
# ══════════════════════════════════════════════════════════════════════════

@router.get(
    "/me",
    response_model=UserSchema,
    summary="Get My Profile",
)
def get_me(current_user: dict = Depends(get_current_user)):
    return _clean_user(current_user)


@router.put(
    "/me",
    response_model=UserSchema,
    summary="Update My Profile",
)
def update_me(body: UpdateUserRequest, current_user: dict = Depends(get_current_user)):
    updates: dict = {"updated_at": datetime.now(timezone.utc)}

    for field, value in body.model_dump(exclude_unset=True).items():
        if field == "notifications" and isinstance(value, dict):
            # Partial update: write only the keys the caller actually sent,
            # using dot-notation so untouched notification keys are preserved.
            for notif_key, notif_val in value.items():
                if notif_val is not None:
                    updates[f"notifications.{notif_key}"] = notif_val
        elif value is not None:
            updates[field] = value

    # Auto-regenerate user_name if first_name, last_name, or phone_number changed
    # and the caller did not explicitly supply a new user_name
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
    col.update_one(
        {"user_id": current_user["user_id"]},
        {"$set": updates},
    )

    updated = col.find_one({"user_id": current_user["user_id"]})
    return _clean_user(updated)


@router.delete(
    "/me",
    response_model=MessageResponse,
    summary="Delete My Account (Soft Delete)",
)
def delete_me(current_user: dict = Depends(get_current_user)):
    col = get_users_collection()
    col.update_one(
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
def change_password(body: ChangePasswordRequest, current_user: dict = Depends(get_current_user)):
    if current_user.get("auth_provider") == "google":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google-authenticated accounts cannot change passwords here.",
        )

    col = get_users_collection()
    user = col.find_one({"user_id": current_user["user_id"]})

    if not verify_password(body.old_password, user.get("password", "")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect.",
        )

    err = validate_password_strength(body.new_password)
    if err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=err)

    col.update_one(
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
def change_email(
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

    if col.find_one({"email": body.new_email}):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This email is already in use by another account.",
        )

    otp = generate_otp()
    save_otp(
        email=body.new_email,
        otp=otp,
        purpose="change_email",
        extra={"user_id": current_user["user_id"]},
    )
    background_tasks.add_task(send_otp_email, to_email=body.new_email, otp=otp, purpose="register")

    return {
        "success": True,
        "message": f"OTP sent to {body.new_email}. Please verify to complete the email change.",
    }


@router.post(
    "/me/change-email/resend-otp",
    response_model=MessageResponse,
    summary="Change Email – Resend OTP",
)
def change_email_resend_otp(
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

    if col.find_one({"email": body.new_email}):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This email is already in use by another account.",
        )

    otp = generate_otp()
    save_otp(
        email=body.new_email,
        otp=otp,
        purpose="change_email",
        extra={"user_id": current_user["user_id"]},
    )
    background_tasks.add_task(send_otp_email, to_email=body.new_email, otp=otp, purpose="register")

    return {"success": True, "message": f"OTP resent to {body.new_email}."}


@router.post(
    "/me/change-email/verify-otp",
    response_model=UserSchema,
    summary="Change Email – Step 2: Verify OTP & Update Email",
)
def change_email_verify_otp(
    body: VerifyEmailChangeRequest,
    current_user: dict = Depends(get_current_user),
):
    record = verify_otp(email=body.new_email, otp=body.otp, purpose="change_email")

    if record.get("user_id") != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="OTP was not issued for this account.",
        )

    col = get_users_collection()

    if col.find_one({"email": body.new_email}):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This email is already in use by another account.",
        )

    col.update_one(
        {"user_id": current_user["user_id"]},
        {"$set": {"email": body.new_email, "updated_at": datetime.now(timezone.utc)}},
    )
    consume_otp(body.new_email, "change_email")

    updated = col.find_one({"user_id": current_user["user_id"]})
    return _clean_user(updated)


# ══════════════════════════════════════════════════════════════════════════
# FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════════

@router.post(
    "/upload-file",
    summary="Upload File to S3",
    description="Upload a file (image, PDF, video) and receive a public URL.",
)
def upload_file(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    if file.content_type not in _ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {', '.join(_ALLOWED_MIME_TYPES)}",
        )

    user_id = current_user["user_id"]
    url = upload_file_to_s3(file, folder=f"users/{user_id}")

    return {"success": True, "url": url}
