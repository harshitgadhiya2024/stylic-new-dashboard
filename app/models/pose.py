from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


MannequinFraming = Literal["full_body", "upper_body", "lower_body", "detail_closeup"]


class PoseListItem(BaseModel):
    """Legacy minimal shape (still valid for embedded use)."""

    pose_id: str
    image: str


_POSE_TYPES = frozenset({"front", "back", "side"})


def normalize_pose_type_value(value: str) -> str:
    s = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in s:
        s = s.replace("__", "_")
    if s in _POSE_TYPES:
        return s
    raise ValueError(
        "pose_type must be one of: front, back, side (e.g. Front, Side, Back)."
    )


class CreatePoseFromImageRequest(BaseModel):
    pose_name: str
    pose_type: str = Field(..., description="front, back, or side (any case).")
    image_url: str
    tags: Optional[List[str]] = None
    notes: Optional[str] = ""
    mannequin_framing: Optional[MannequinFraming] = Field(
        default=None,
        description=(
            "How much of the body the mannequin shows. When set, photoshoot SeedDream uses matching "
            "prompt rules (e.g. no mandatory full-length footwear for upper_body). "
            "Omit to infer from pose_prompt when possible."
        ),
    )

    @field_validator("pose_type", mode="before")
    @classmethod
    def _pose_type(cls, v):
        return normalize_pose_type_value(v)


class CreatePoseFromPromptRequest(BaseModel):
    pose_name: str
    pose_prompt: str
    pose_type: str = Field(..., description="front, back, or side (any case).")
    tags: Optional[List[str]] = None
    notes: Optional[str] = ""
    mannequin_framing: Optional[MannequinFraming] = Field(
        default=None,
        description=(
            "How much of the body the mannequin shows. When set, photoshoot SeedDream uses matching "
            "prompt rules. Omit to infer from pose_prompt when possible."
        ),
    )

    @field_validator("pose_type", mode="before")
    @classmethod
    def _pose_type_prompt(cls, v):
        return normalize_pose_type_value(v)


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
    image_url: str
    mannequin_framing: Optional[MannequinFraming] = None
    count: int = 0
    notes: str = ""
    tags: List[str] = []
    favorite_list: List[Any] = Field(default_factory=list)
    is_default: bool = False
    is_active: bool = True
    is_favorite: bool = False
    created_at: datetime
    updated_at: datetime
