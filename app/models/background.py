from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from datetime import datetime

_ALLOWED_BACKGROUND_TYPES = frozenset({"indoor", "outdoor", "studio"})


def normalize_background_type_value(value) -> str:
    """Accept Indoor / OUTDOOR / Studio etc.; return lowercase DB value."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError("background_type is required")
    s = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in s:
        s = s.replace("__", "_")
    if s in _ALLOWED_BACKGROUND_TYPES:
        return s
    raise ValueError(
        "background_type must be one of: Indoor, Outdoor, Studio "
        "(stored as indoor, outdoor, studio)."
    )


class DeleteBackgroundsRequest(BaseModel):
    background_ids: list[str]


class CreateBackgroundRequest(BaseModel):
    background_name: str
    background_url:  str
    background_type: str = Field(
        ...,
        description="Indoor, Outdoor, or Studio (any case). Stored in lowercase.",
    )
    tags:            Optional[List[str]] = []
    notes:           Optional[str]       = ""

    @field_validator("background_type", mode="before")
    @classmethod
    def _validate_background_type(cls, v):
        return normalize_background_type_value(v)


class AdminUpdateBackgroundRequest(BaseModel):
    """Admin-only: partial update of any background by ``background_id``."""

    background_name: Optional[str] = Field(None, min_length=1)
    background_type: Optional[str] = None
    background_url: Optional[str] = Field(None, min_length=1)
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    count: Optional[int] = Field(None, ge=0)
    is_active: Optional[bool] = None

    @field_validator("background_name", "background_url", mode="before")
    @classmethod
    def _strip_optional_strings(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("background_type", mode="before")
    @classmethod
    def _validate_optional_background_type(cls, v):
        if v is None:
            return None
        return normalize_background_type_value(v)


class AdminCreateDefaultBackgroundRequest(BaseModel):
    """Admin-only: create a platform default background (``is_default=True``)."""

    background_name: str = Field(..., min_length=1)
    background_url: str = Field(..., min_length=1, description="Public URL, e.g. from upload-file.")
    background_type: str = Field(
        ...,
        description="Indoor, Outdoor, or Studio (any case). Stored in lowercase.",
    )

    @field_validator("background_name", "background_url", mode="before")
    @classmethod
    def _strip_ws(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("background_type", mode="before")
    @classmethod
    def _validate_background_type_admin(cls, v):
        return normalize_background_type_value(v)


class CreateBackgroundWithAIRequest(BaseModel):
    background_name:          str
    background_configuration: str
    background_type: str = Field(
        ...,
        description="Indoor, Outdoor, or Studio (any case). Stored in lowercase.",
    )
    tags:                     Optional[List[str]] = []
    notes:                    Optional[str]       = ""

    @field_validator("background_type", mode="before")
    @classmethod
    def _validate_background_type_ai(cls, v):
        return normalize_background_type_value(v)


class BackgroundSchema(BaseModel):
    """API background record. Defaults: viewer `is_favorite` from `favorite_list`; no `user_id`."""

    model_config = ConfigDict(extra="ignore")

    background_id:   str
    user_id:         Optional[str] = None
    background_type: str
    background_name: str
    background_url:  str
    count:           int           = 0
    tags:            List[str]     = []
    notes:           str           = ""
    favorite_list:   List[Any]     = Field(default_factory=list)
    is_default:      bool          = False
    is_active:       bool          = True
    is_favorite:     bool          = False
    created_at:      datetime
    updated_at:      datetime
