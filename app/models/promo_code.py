"""Pydantic models for promo-code admin and user APIs."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PromoType = Literal["credit", "discount"]


def _normalize_promo_code(raw: str) -> str:
    return str(raw or "").strip().upper()


class PromoCodeRecord(BaseModel):
    promo_id: str
    promo_code: str
    promo_type: PromoType
    promo_discount: int
    promo_credit: int
    expiry_date: datetime
    is_active: bool
    created_at: datetime
    updated_at: datetime


class CreatePromoCodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    promo_code: str = Field(..., min_length=1, max_length=128)
    promo_type: PromoType
    promo_discount: int = Field(default=0, ge=0)
    promo_credit: int = Field(default=0, ge=0)
    expiry_date: datetime

    @field_validator("promo_code")
    @classmethod
    def normalize_promo_code(cls, v: str) -> str:
        code = _normalize_promo_code(v)
        if not code:
            raise ValueError("promo_code is required")
        return code

    @model_validator(mode="after")
    def validate_by_type(self):
        if self.promo_type == "credit":
            if self.promo_credit <= 0:
                raise ValueError("promo_credit must be > 0 for credit promo_type")
            self.promo_discount = 0
        else:
            if self.promo_discount <= 0:
                raise ValueError("promo_discount must be > 0 for discount promo_type")
            self.promo_credit = 0
        return self


class UpdatePromoCodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    promo_id: Optional[str] = None
    promo_code: Optional[str] = None
    promo_type: Optional[PromoType] = None
    promo_discount: Optional[int] = Field(default=None, ge=0)
    promo_credit: Optional[int] = Field(default=None, ge=0)
    expiry_date: Optional[datetime] = None
    is_active: Optional[bool] = None

    @field_validator("promo_code")
    @classmethod
    def normalize_optional_code(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        code = _normalize_promo_code(v)
        if not code:
            raise ValueError("promo_code cannot be empty")
        return code

    @model_validator(mode="after")
    def validate_target(self):
        if not (self.promo_id or self.promo_code):
            raise ValueError("Either promo_id or promo_code is required")
        return self


class PromoLookupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    promo_id: Optional[str] = None
    promo_code: Optional[str] = None

    @field_validator("promo_code")
    @classmethod
    def normalize_optional_code(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        code = _normalize_promo_code(v)
        if not code:
            raise ValueError("promo_code cannot be empty")
        return code

    @model_validator(mode="after")
    def validate_target(self):
        if not (self.promo_id or self.promo_code):
            raise ValueError("Either promo_id or promo_code is required")
        return self


class PromoCodeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    promo_code: str = Field(..., min_length=1, max_length=128)

    @field_validator("promo_code")
    @classmethod
    def normalize_promo_code(cls, v: str) -> str:
        code = _normalize_promo_code(v)
        if not code:
            raise ValueError("promo_code is required")
        return code
