from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


# ─────────────────────────── Shared sub-schemas ───────────────────────────

class NotificationPreferences(BaseModel):
    email_notifications: bool = False
    push_notifications: bool = False
    marketing_emails: bool = False


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
    auth_provider: str = "email"
    notifications: NotificationPreferences = Field(default_factory=NotificationPreferences)
    is_active: bool = True
    created_at: datetime
    updated_at: datetime


# ─────────────────────────── Auth request models ──────────────────────────

class RegisterRequest(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    password: str
    phone_number: str


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
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    user_name: Optional[str] = None
    phone_number: Optional[str] = None
    bio: Optional[str] = None
    profile_picture: Optional[str] = None
    is_public_to_explore: Optional[bool] = None
    language: Optional[str] = None
    time_zone: Optional[str] = None
    plan: Optional[str] = None
    notifications: Optional[PartialNotificationPreferences] = None


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class ChangeEmailRequest(BaseModel):
    new_email: EmailStr


class VerifyEmailChangeRequest(BaseModel):
    new_email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)


# ─────────────────────────── Response models ──────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserSchema


class MessageResponse(BaseModel):
    success: bool
    message: str
