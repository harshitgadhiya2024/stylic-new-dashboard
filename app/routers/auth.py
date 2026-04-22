import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from firebase_admin import auth as firebase_auth

from app.config import settings
from app.constants.free_plan import FREE_ROLE_MAPPING_DICT, build_free_plan_mapping_dict
from app.database import get_users_collection
from app.firebase_config import get_firebase_app
from app.models.user import (
    RegisterRequest,
    VerifyOTPRequest,
    ResendOTPRequest,
    LoginRequest,
    RefreshTokenRequest,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    GoogleSignInRequest,
    TokenResponse,
    MessageResponse,
)
from app.services.jwt_service import (
    create_access_token,
    create_refresh_token,
    decode_token,
)
from app.services.email_service import send_otp_email
from app.services.otp_service import (
    generate_otp,
    save_otp,
    verify_otp,
    check_otp_verified,
    consume_otp,
    get_pending_otp,
)
from app.utils.password import hash_password, verify_password, validate_password_strength
from app.utils.user_response import user_dict_for_api

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


# ─────────────────────────── Helpers ──────────────────────────────────────

def _clean_user(user: dict) -> dict:
    return user_dict_for_api(user)


def _build_token_response(user: dict) -> dict:
    clean = _clean_user(user)
    return {
        "access_token": create_access_token(user["user_id"]),
        "refresh_token": create_refresh_token(user["user_id"]),
        "token_type": "bearer",
        "user": clean,
    }


def _generate_username(first_name: str, last_name: str, phone_number: str) -> str:
    suffix = phone_number[-5:] if len(phone_number) >= 5 else phone_number
    return f"{first_name}_{last_name}_{suffix}".lower()


def _new_user_doc(
    email: str,
    hashed_password: str,
    auth_provider: str = "email",
    first_name: str = "",
    last_name: str = "",
    phone_number: str = "",
    profile_picture: str = "",
    **kwargs,
) -> dict:
    now = datetime.now(timezone.utc)
    doc = {
        "user_id": str(uuid.uuid4()),
        "first_name": first_name,
        "last_name": last_name,
        "user_name": _generate_username(first_name, last_name, phone_number),
        "email": email,
        "password": hashed_password,
        "phone_number": phone_number,
        "bio": "",
        "profile_picture": profile_picture,
        "is_public_to_explore": True,
        "language": "English",
        "time_zone": "",
        "credits": settings.DEFAULT_CREDITS,
        "plan": settings.DEFAULT_PLAN,
        "role_mapping_dict": dict(FREE_ROLE_MAPPING_DICT),
        "plan_mapping_dict": build_free_plan_mapping_dict(now, settings.DEFAULT_PLAN),
        "auth_provider": auth_provider,
        "notifications": {
            "email_notifications": False,
            "push_notifications": False,
            "marketing_emails": False,
        },
        "is_active": True,
        "created_at": now,
        "updated_at": now,
    }
    doc.update(kwargs)
    return doc


# ══════════════════════════════════════════════════════════════════════════
# REGISTER FLOW
# ══════════════════════════════════════════════════════════════════════════

@router.post(
    "/register",
    response_model=MessageResponse,
    summary="Register – Step 1: Send OTP",
    description="Validate email & password, then send a 6-digit OTP to the provided email.",
)
async def register(body: RegisterRequest, background_tasks: BackgroundTasks):
    col = get_users_collection()

    if await col.find_one({"email": body.email}):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    err = validate_password_strength(body.password)
    if err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=err)

    otp = generate_otp()
    await save_otp(
        email=body.email,
        otp=otp,
        purpose="register",
        extra={
            "hashed_password": hash_password(body.password),
            "first_name": body.first_name,
            "last_name": body.last_name,
            "phone_number": body.phone_number,
        },
    )
    background_tasks.add_task(send_otp_email, to_email=body.email, otp=otp, purpose="register")

    return {
        "success": True,
        "message": f"OTP sent to {body.email}. Valid for {settings.OTP_EXPIRE_MINUTES} minutes.",
    }


@router.post(
    "/register/verify-otp",
    response_model=TokenResponse,
    summary="Register – Step 2: Verify OTP & Create Account",
)
async def register_verify_otp(body: VerifyOTPRequest):
    record = await verify_otp(email=body.email, otp=body.otp, purpose="register")

    hashed_password = record.get("hashed_password", "")
    col = get_users_collection()

    if await col.find_one({"email": body.email}):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Account already exists.",
        )

    user_doc = _new_user_doc(
        email=body.email,
        hashed_password=hashed_password,
        auth_provider="email",
        first_name=record.get("first_name", ""),
        last_name=record.get("last_name", ""),
        phone_number=record.get("phone_number", ""),
    )
    await col.insert_one(user_doc)
    await consume_otp(body.email, "register")

    return _build_token_response(user_doc)


@router.post(
    "/register/resend-otp",
    response_model=MessageResponse,
    summary="Register – Resend OTP",
)
async def register_resend_otp(body: ResendOTPRequest, background_tasks: BackgroundTasks):
    col = get_users_collection()
    if await col.find_one({"email": body.email}):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Account already exists. Please log in.",
        )

    existing = await get_pending_otp(email=body.email, purpose="register")
    hashed_password = existing.get("hashed_password", "") if existing else ""

    if not hashed_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration session expired. Please start registration again.",
        )

    otp = generate_otp()
    await save_otp(
        email=body.email,
        otp=otp,
        purpose="register",
        extra={
            "hashed_password": hashed_password,
            "first_name": existing.get("first_name", ""),
            "last_name": existing.get("last_name", ""),
            "phone_number": existing.get("phone_number", ""),
        },
    )
    background_tasks.add_task(send_otp_email, to_email=body.email, otp=otp, purpose="register")

    return {"success": True, "message": f"OTP resent to {body.email}."}


# ══════════════════════════════════════════════════════════════════════════
# LOGIN FLOW
# ══════════════════════════════════════════════════════════════════════════

@router.post(
    "/login",
    response_model=MessageResponse,
    summary="Login – Step 1: Validate Credentials & Send OTP",
    description="Validate email & password, then send a 6-digit OTP to the provided email.",
)
async def login(body: LoginRequest, background_tasks: BackgroundTasks):
    col = get_users_collection()
    user = await col.find_one({"email": body.email, "is_active": True})

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if user.get("auth_provider") == "google":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This account uses Google Sign-In. Please log in with Google.",
        )

    if not verify_password(body.password, user.get("password", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    otp = generate_otp()
    await save_otp(email=body.email, otp=otp, purpose="login")
    background_tasks.add_task(send_otp_email, to_email=body.email, otp=otp, purpose="login")

    return {
        "success": True,
        "message": f"OTP sent to {body.email}. Valid for {settings.OTP_EXPIRE_MINUTES} minutes.",
    }


@router.post(
    "/login/verify-otp",
    response_model=TokenResponse,
    summary="Login – Step 2: Verify OTP",
)
async def login_verify_otp(body: VerifyOTPRequest):
    await verify_otp(email=body.email, otp=body.otp, purpose="login")

    col = get_users_collection()
    user = await col.find_one({"email": body.email, "is_active": True})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")

    await consume_otp(body.email, "login")
    return _build_token_response(user)


@router.post(
    "/login/resend-otp",
    response_model=MessageResponse,
    summary="Login – Resend OTP",
)
async def login_resend_otp(body: ResendOTPRequest, background_tasks: BackgroundTasks):
    col = get_users_collection()
    if not await col.find_one({"email": body.email, "is_active": True}):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    otp = generate_otp()
    await save_otp(email=body.email, otp=otp, purpose="login")
    background_tasks.add_task(send_otp_email, to_email=body.email, otp=otp, purpose="login")

    return {"success": True, "message": f"OTP resent to {body.email}."}


# ══════════════════════════════════════════════════════════════════════════
# REFRESH TOKEN
# ══════════════════════════════════════════════════════════════════════════

@router.post(
    "/refresh-token",
    response_model=TokenResponse,
    summary="Refresh Access & Refresh Tokens",
)
async def refresh_token(body: RefreshTokenRequest):
    payload = decode_token(body.refresh_token, token_type="refresh")
    user_id = payload.get("sub")

    col = get_users_collection()
    user = await col.find_one({"user_id": user_id, "is_active": True})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")

    return _build_token_response(user)


# ══════════════════════════════════════════════════════════════════════════
# FORGOT PASSWORD FLOW
# ══════════════════════════════════════════════════════════════════════════

@router.post(
    "/forgot-password",
    response_model=MessageResponse,
    summary="Forgot Password – Step 1: Send OTP",
)
async def forgot_password(body: ForgotPasswordRequest, background_tasks: BackgroundTasks):
    col = get_users_collection()
    user = await col.find_one({"email": body.email, "is_active": True})

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No account found with this email.")

    if user.get("auth_provider") == "google":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This account uses Google Sign-In and has no password to reset.",
        )

    otp = generate_otp()
    await save_otp(email=body.email, otp=otp, purpose="forgot_password")
    background_tasks.add_task(send_otp_email, to_email=body.email, otp=otp, purpose="forgot_password")

    return {
        "success": True,
        "message": f"Password reset OTP sent to {body.email}.",
    }


@router.post(
    "/forgot-password/resend-otp",
    response_model=MessageResponse,
    summary="Forgot Password – Resend OTP",
)
async def forgot_password_resend_otp(body: ResendOTPRequest, background_tasks: BackgroundTasks):
    col = get_users_collection()
    if not await col.find_one({"email": body.email, "is_active": True}):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No account found with this email.")

    otp = generate_otp()
    await save_otp(email=body.email, otp=otp, purpose="forgot_password")
    background_tasks.add_task(send_otp_email, to_email=body.email, otp=otp, purpose="forgot_password")

    return {"success": True, "message": f"OTP resent to {body.email}."}


@router.post(
    "/forgot-password/verify-otp",
    response_model=MessageResponse,
    summary="Forgot Password – Step 2: Verify OTP",
)
async def forgot_password_verify_otp(body: VerifyOTPRequest):
    await verify_otp(email=body.email, otp=body.otp, purpose="forgot_password")
    return {"success": True, "message": "OTP verified. You may now reset your password."}


@router.post(
    "/forgot-password/reset",
    response_model=MessageResponse,
    summary="Forgot Password – Step 3: Reset Password",
)
async def forgot_password_reset(body: ResetPasswordRequest):
    otp_record = await check_otp_verified(email=body.email, purpose="forgot_password")
    if not otp_record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OTP not verified or session expired. Please request a new OTP.",
        )

    err = validate_password_strength(body.new_password)
    if err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=err)

    col = get_users_collection()
    result = await col.update_one(
        {"email": body.email, "is_active": True},
        {
            "$set": {
                "password": hash_password(body.new_password),
                "updated_at": datetime.now(timezone.utc),
            }
        },
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    await consume_otp(body.email, "forgot_password")
    return {"success": True, "message": "Password updated successfully."}


# ══════════════════════════════════════════════════════════════════════════
# GOOGLE SIGN-IN / SIGN-UP
# ══════════════════════════════════════════════════════════════════════════

@router.post(
    "/google",
    response_model=TokenResponse,
    summary="Google Sign-In / Sign-Up",
    description="Verify a Firebase ID token from Google sign-in, then login or auto-register the user.",
)
async def google_sign_in(body: GoogleSignInRequest):
    try:
        get_firebase_app()
        decoded = firebase_auth.verify_id_token(body.id_token)
    except firebase_auth.ExpiredIdTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Google token has expired.")
    except firebase_auth.InvalidIdTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid Google token: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Token verification failed: {exc}")

    provider = decoded.get("firebase", {}).get("sign_in_provider", "")
    if provider != "google.com":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token was not issued by Google.",
        )

    email = decoded.get("email")
    full_name = decoded.get("name", "")
    picture = decoded.get("picture", "")
    phone_number = decoded.get("phone_number", "")
    parts = (full_name or "").strip().split(" ", 1)
    first_name = parts[0] if parts else ""
    last_name = parts[1] if len(parts) > 1 else ""

    col = get_users_collection()
    user = await col.find_one({"email": email})

    if user:
        user.pop("_id", None)
        return _build_token_response(user)

    user_doc = _new_user_doc(
        email=email,
        hashed_password="",
        auth_provider="google",
        first_name=first_name,
        last_name=last_name,
        phone_number=phone_number,
        profile_picture=picture,
    )
    await col.insert_one(user_doc)
    user_doc.pop("_id", None)

    return _build_token_response(user_doc)
