from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
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
    background_id:   str
    user_id:         Optional[str] = None
    background_type: str
    background_name: str
    background_url:  str
    count:           int           = 0
    tags:            List[str]     = []
    notes:           str           = ""
    is_default:      bool          = False
    is_active:       bool          = True
    is_favorite:     bool          = False
    created_at:      datetime
    updated_at:      datetime
