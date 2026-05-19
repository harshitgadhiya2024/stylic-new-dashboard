"""Pydantic models for admin header meta tag APIs."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HeaderMetaTagRecord(BaseModel):
    meta_tag_id: str
    header_meta_tag: str
    created_at: datetime
    updated_at: datetime


class CreateHeaderMetaTagRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    header_meta_tag: str = Field(..., min_length=1, description="Meta tag HTML or content to inject in page head.")

    @field_validator("header_meta_tag", mode="before")
    @classmethod
    def strip_header_meta_tag(cls, v: str) -> str:
        s = str(v or "").strip()
        if not s:
            raise ValueError("header_meta_tag is required")
        return s


class UpdateHeaderMetaTagRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    meta_tag_id: str = Field(..., min_length=1, max_length=64)
    header_meta_tag: Optional[str] = Field(
        default=None,
        description="Updated meta tag content. Omit to leave unchanged.",
    )

    @field_validator("meta_tag_id", mode="before")
    @classmethod
    def strip_meta_tag_id(cls, v: str) -> str:
        s = str(v or "").strip()
        if not s:
            raise ValueError("meta_tag_id is required")
        return s

    @field_validator("header_meta_tag", mode="before")
    @classmethod
    def strip_optional_header_meta_tag(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            raise ValueError("header_meta_tag cannot be empty when provided")
        return s


class HeaderMetaTagListResponse(BaseModel):
    total: int
    meta_tags: list[HeaderMetaTagRecord]
