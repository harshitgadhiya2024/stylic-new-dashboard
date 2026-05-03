from datetime import datetime
from typing import Any, Literal, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.utils.phone_validation import (
    normalize_phone_optional,
    normalize_phone_profile_field,
    normalize_phone_to_e164,
)


# ─────────────────────────── Shared sub-schemas ───────────────────────────

class NotificationPreferences(BaseModel):
    email_notifications: bool = False
    push_notifications: bool = False
    marketing_emails: bool = False


# ─────────────────────────── Onboarding ───────────────────────────────────

class OnboardingData(BaseModel):
    """Stored on the user document under ``onboarding``."""

    model_config = ConfigDict(extra="ignore")
    customer_type: Literal["solo", "business"]
    country: str
    city: str
    phone_number: Optional[str] = None
    business_name: Optional[str] = None
    gst_no: Optional[str] = None
    team_size: Optional[str] = None
    choices: list[str] = Field(default_factory=list)
    stored_at: Optional[datetime] = None


class StoreOnboardingRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid", populate_by_name=True)

    customer_type: str = Field(
        ...,
        description="Type of account: 'solo' or 'business' (any case).",
    )
    country: str = Field(..., min_length=1, description="Country (required).")
    city: str = Field(..., min_length=1, description="City (required).")
    business_name: Optional[str] = Field(default=None, description="Optional; relevant for business customers.")
    gst_no: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("gst_no", "GST_no"),
        description="Optional GST number (accepts `gst_no` or `GST_no` in JSON).",
    )
    team_size: Optional[str] = Field(default=None, description="Optional team size (free text, e.g. a range or number).")
    choices: list[str] = Field(default_factory=list, description="List of string choices (e.g. use-case tags).")
    phone_number: Optional[str] = Field(
        default=None,
        max_length=24,
        description="Optional; international format (+country …) validated via libphonenumber (E.164 stored).",
    )

    @field_validator("customer_type", mode="before")
    @classmethod
    def _normalize_customer_type(cls, v: Any) -> str:
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ValueError("customer_type is required; use 'solo' or 'business'.")
        s = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        if s in ("solo", "business"):
            return s
        raise ValueError("customer_type must be 'solo' or 'business'.")

    @field_validator("phone_number", mode="after")
    @classmethod
    def _validate_onboarding_phone(cls, v: Optional[str]) -> Optional[str]:
        return normalize_phone_optional(v)


# ─────────────────────────── User schema (response) ───────────────────────

class UserSchema(BaseModel):
    user_id: str
    first_name: str = ""
    last_name: str = ""
    user_name: str = ""
    email: str
    phone_number: Optional[str] = ""
    bio: Optional[str] = ""
    profile_picture: Optional[str] = ""
    is_public_to_explore: bool = True
    language: str = "English"
    time_zone: Optional[str] = ""
    credits: float = 5.0
    plan: str = ""
    total_credit: float = 0.0
    used_credit: float = 0.0
    credit_percentage: int = 0
    role_mapping_dict: dict[str, Any]
    plan_mapping_dict: dict[str, Any]
    auth_provider: str = "email"
    notifications: NotificationPreferences = Field(default_factory=NotificationPreferences)
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    onboarding: Optional[OnboardingData] = None


# ─────────────────────────── Auth request models ──────────────────────────

class RegisterRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    first_name: str
    last_name: str
    email: EmailStr
    password: str
    phone_number: str = Field(..., min_length=1, max_length=24)

    @field_validator("phone_number", mode="after")
    @classmethod
    def _validate_register_phone(cls, v: str) -> str:
        return normalize_phone_to_e164(v)


class VerifyOTPRequest(BaseModel):
    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)


class ResendOTPRequest(BaseModel):
    email: EmailStr


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    email: EmailStr
    new_password: str


class GoogleSignInRequest(BaseModel):
    id_token: str


# ─────────────────────────── User request models ──────────────────────────

class PartialNotificationPreferences(BaseModel):
    email_notifications: Optional[bool] = None
    push_notifications: Optional[bool] = None
    marketing_emails: Optional[bool] = None


class UpdateUserRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    user_name: Optional[str] = None
    phone_number: Optional[str] = Field(default=None, max_length=24)
    bio: Optional[str] = None
    profile_picture: Optional[str] = None
    is_public_to_explore: Optional[bool] = None
    language: Optional[str] = None
    time_zone: Optional[str] = None
    plan: Optional[str] = None
    notifications: Optional[PartialNotificationPreferences] = None

    @field_validator("phone_number", mode="before")
    @classmethod
    def _validate_update_phone(cls, v: object) -> Union[str, None]:
        if v is None:
            return None
        return normalize_phone_profile_field(v)  # type: ignore[arg-type]


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class ChangeEmailRequest(BaseModel):
    new_email: EmailStr


class VerifyEmailChangeRequest(BaseModel):
    new_email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)


class CancelSubscriptionRequest(BaseModel):
    reason: str = Field(..., min_length=1, max_length=2000)
    comments: str = Field(default="", max_length=5000)


# ─────────────────────────── Response models ──────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserSchema


class MessageResponse(BaseModel):
    success: bool
    message: str
