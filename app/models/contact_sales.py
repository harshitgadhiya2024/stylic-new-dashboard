"""
Public contact-sales form (no authentication).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic import EmailStr

from app.utils.phone_validation import normalize_phone_optional


class ContactSalesRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    work_email: EmailStr
    phone: Optional[str] = Field(default=None, max_length=24)
    message: str = Field(..., min_length=1, max_length=5000)
    # Honeypot: must stay empty; not advertised to humans in marketing copy
    website: Optional[str] = Field(
        default=None,
        max_length=500,
    )

    @field_validator("work_email", mode="after")
    @classmethod
    def _lower_email(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("phone", mode="after")
    @classmethod
    def _phone_international(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not str(v).strip():
            return None
        s = str(v).strip()
        if "<" in s or ">" in s:
            raise ValueError("Invalid characters in field")
        if any(ord(c) < 32 and c not in "\t" for c in s):
            raise ValueError("Invalid control characters")
        return normalize_phone_optional(s)

    @field_validator("first_name", "last_name", mode="after")
    @classmethod
    def _no_brackets_or_controls(cls, v: str) -> str:
        s = (v or "").strip()
        if not s:
            raise ValueError("This field is required")
        if "<" in s or ">" in s:
            raise ValueError("Invalid characters in field")
        if any(ord(c) < 32 and c not in "\t" for c in s):
            raise ValueError("Invalid control characters")
        return s


class ContactSalesResponse(BaseModel):
    ok:      bool
    message: str


class AdminContactSalesStatusUpdate(BaseModel):
    """Admin-only: set workflow status on a contact-sales row."""

    model_config = ConfigDict(str_strip_whitespace=True)

    status: Literal["pending", "processing", "completed"]
