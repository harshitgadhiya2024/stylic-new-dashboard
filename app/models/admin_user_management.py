"""Request bodies for admin user-management API (end-user records in ``users``)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Keys allowed in ``users.role_mapping_dict`` partial updates (align with ``FREE_ROLE_MAPPING_DICT`` + product).
ALLOWED_USER_ROLE_MAPPING_KEYS: frozenset[str] = frozenset(
    {
        "single_photoshoot",
        "max_pose",
        "max_resolution",
        "multiple_photoshoot",
        "catalogue_photoshoot",
        "custom_model",
        "custom_background",
        "custom_poses",
        "resize",
        "branding",
        "background_change",
        "color_change",
        "adjust_image",
        "fabric_change",
        "texture_change",
    }
)

class UserRoleMappingPartialRequest(BaseModel):
    """Pass only the keys to change; merged into existing ``role_mapping_dict``."""

    model_config = ConfigDict(extra="forbid")

    single_photoshoot:    Optional[bool]  = None
    max_pose:            Optional[int]   = None
    max_resolution:      Optional[str]   = None
    multiple_photoshoot:  Optional[bool]  = None
    catalogue_photoshoot: Optional[bool]  = None
    custom_model:         Optional[bool]  = None
    custom_background:   Optional[bool]  = None
    custom_poses:        Optional[bool]  = None
    resize:              Optional[bool]  = None
    branding:            Optional[bool]  = None
    background_change:  Optional[bool]  = None
    color_change:        Optional[bool]  = None
    adjust_image:         Optional[bool]  = None
    fabric_change:        Optional[bool]  = None
    texture_change:      Optional[bool]  = None

    @field_validator("max_pose")
    @classmethod
    def max_pose_pos(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        if v < 0 or v > 64:
            raise ValueError("max_pose must be between 0 and 64")
        return v

    @field_validator("max_resolution")
    @classmethod
    def res_norm(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not str(v).strip():
            return None
        return str(v).strip()


class UserPlanPartialRequest(BaseModel):
    """Partial update for ``plan``, ``plan_mapping_dict`` (start/renew), and end-user ``role``."""

    model_config = ConfigDict(extra="forbid")

    plan: Optional[str] = Field(
        default=None,
        description="Top-level `plan`; also sets `plan_mapping_dict.plan` when present.",
    )
    role: Optional[str] = Field(
        default=None,
        max_length=100,
        description="End-user `role` on the user document (not the admin JWT role).",
    )
    start_date: Optional[datetime] = None
    renew_date: Optional[datetime] = None

    @field_validator("role", "plan", mode="before")
    @classmethod
    def strip_empty(cls, v) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None


class AddUserCreditsRequest(BaseModel):
    credit: float = Field(..., gt=0, le=1_000_000, description="Amount to add to user balance")
    notes:  str = Field(default="", max_length=2000)


class UserBlockUnblockRequest(BaseModel):
    is_active: bool = Field(..., description="true = unblock, false = block")
    reason:   str  = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Required audit note (e.g. why block or who approved unblock).",
    )
