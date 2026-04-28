from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class PoseListItem(BaseModel):
    """Legacy minimal shape (still valid for embedded use)."""

    pose_id: str
    image: str


_POSE_TYPES = frozenset({"front", "back", "side"})
_GARMENT_TYPES = frozenset({"upper_body", "full_body"})


def normalize_pose_type_value(value: str) -> str:
    s = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in s:
        s = s.replace("__", "_")
    if s in _POSE_TYPES:
        return s
    raise ValueError(
        "pose_type must be one of: front, back, side (e.g. Front, Side, Back)."
    )


def normalize_optional_pose_type_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return normalize_pose_type_value(s)


def normalize_optional_garment_type_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in s:
        s = s.replace("__", "_")
    if not s:
        return None
    if s == "lower_body":
        return "full_body"
    if s in _GARMENT_TYPES:
        return s
    raise ValueError("garment_type must be one of: upper_body, full_body.")


class CreatePoseFromImageRequest(BaseModel):
    pose_name: str
    pose_type: Optional[str] = Field(
        default=None,
        description="front, back, or side (any case). Optional; inferred from AI when omitted.",
    )
    garment_type: Optional[str] = Field(
        default=None,
        description="upper_body or full_body. Optional; inferred from AI when omitted.",
    )
    image_url: str
    tags: Optional[List[str]] = None
    notes: Optional[str] = ""

    @field_validator("pose_type", mode="before")
    @classmethod
    def _pose_type(cls, v):
        return normalize_optional_pose_type_value(v)

    @field_validator("garment_type", mode="before")
    @classmethod
    def _garment_type(cls, v):
        return normalize_optional_garment_type_value(v)


class CreatePoseFromPromptRequest(BaseModel):
    pose_name: str
    pose_prompt: str
    pose_type: Optional[str] = Field(
        default=None,
        description="front, back, or side (any case). Optional; inferred from AI when omitted.",
    )
    garment_type: Optional[str] = Field(
        default=None,
        description="upper_body or full_body. Optional; inferred from AI when omitted.",
    )
    tags: Optional[List[str]] = None
    notes: Optional[str] = ""

    @field_validator("pose_type", mode="before")
    @classmethod
    def _pose_type_prompt(cls, v):
        return normalize_optional_pose_type_value(v)

    @field_validator("garment_type", mode="before")
    @classmethod
    def _garment_type_prompt(cls, v):
        return normalize_optional_garment_type_value(v)


class DeletePosesRequest(BaseModel):
    pose_ids: List[str]


class PoseSchema(BaseModel):
    """
    Custom poses: ``user_id`` + document ``is_favorite``.
    Platform/default poses: no ``user_id`` in responses; ``is_favorite`` is true when the
    viewer's id is in ``favorite_list``.
    """

    pose_id: str
    user_id: Optional[str] = None
    pose_name: str
    pose_type: str
    pose_prompt: str
    garment_type: str = Field(
        default="",
        description="upper_body or full_body (from vision when the pose was created).",
    )
    image_url: str
    count: int = 0
    notes: str = ""
    tags: List[str] = []
    favorite_list: List[Any] = Field(default_factory=list)
    is_default: bool = False
    is_active: bool = True
    is_favorite: bool = False
    created_at: datetime
    updated_at: datetime
