"""
Paid plan tier caps (silver / gold / platinum) and ``plan_mapping_dict`` windows.

``plan_mapping_dict`` uses Python ``datetime`` for Mongo (same as ``free_plan``).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Literal

PlanTier = Literal["silver", "gold", "platinum"]
PlanPeriod = Literal["monthly", "yearly"]

# From product spec (testing.txt) — bools normalized to Python bool.
SILVER_ROLE_MAPPING: dict[str, Any] = {
    "single_photoshoot": True,
    "max_pose": 4,
    "max_resolution": "4k",
    "multiple_photoshoot": False,
    "catalogue_photoshoot": False,
    "custom_model": True,
    "custom_background": True,
    "custom_poses": True,
    "resize": True,
    "branding": True,
    "background_change": True,
    "color_change": False,
    "adjust_image": True,
    "fabric_change": False,
    "texture_change": False,
}

GOLD_ROLE_MAPPING: dict[str, Any] = {
    "single_photoshoot": True,
    "max_pose": 6,
    "max_resolution": "4k",
    "multiple_photoshoot": False,
    "catalogue_photoshoot": True,
    "custom_model": True,
    "custom_background": True,
    "custom_poses": True,
    "resize": True,
    "branding": True,
    "background_change": True,
    "color_change": True,
    "adjust_image": True,
    "fabric_change": False,
    "texture_change": False,
}

PLATINUM_ROLE_MAPPING: dict[str, Any] = {
    "single_photoshoot": True,
    "max_pose": 8,
    "max_resolution": "8k",
    "multiple_photoshoot": True,
    "catalogue_photoshoot": True,
    "custom_model": True,
    "custom_background": True,
    "custom_poses": True,
    "resize": True,
    "branding": True,
    "background_change": True,
    "color_change": True,
    "adjust_image": True,
    "fabric_change": True,
    "texture_change": True,
}

_ROLE_BY_PLAN: dict[str, dict[str, Any]] = {
    "silver":   SILVER_ROLE_MAPPING,
    "gold":     GOLD_ROLE_MAPPING,
    "platinum": PLATINUM_ROLE_MAPPING,
}


def role_mapping_for_plan(plan_type: str) -> dict[str, Any]:
    key = (plan_type or "").strip().lower()
    if key not in _ROLE_BY_PLAN:
        raise ValueError(f"Unknown plan_type: {plan_type}")
    return dict(_ROLE_BY_PLAN[key])


def build_paid_plan_mapping_dict(
    plan_type: str,
    timeperiod: PlanPeriod,
    start: datetime | None = None,
) -> dict[str, Any]:
    """``renew_date`` = start + 30d (monthly) or + 365d (yearly), UTC."""
    s = (plan_type or "").strip().lower()
    start = start if start is not None else datetime.now(timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    days = 30 if timeperiod == "monthly" else 365
    return {
        "plan": s,
        "start_date":  start,
        "renew_date":  start + timedelta(days=days),
    }
