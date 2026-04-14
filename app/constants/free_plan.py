"""Default subscription caps for new users (free tier)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

FREE_ROLE_MAPPING_DICT: dict = {
    "single_photoshoot": True,
    "max_pose": 2,
    "max_resolution": "2k",
    "multiple_photoshoot": False,
    "catalogue_photoshoot": False,
    "custom_model": False,
    "custom_background": True,
    "custom_poses": False,
    "resize": True,
    "branding": True,
    "background_change": True,
    "color_change": False,
    "adjust_image": True,
    "fabric_change": False,
    "texture_change": False,
}


def build_free_plan_mapping_dict(
    now: datetime | None = None,
    plan: str = "free",
) -> dict:
    """Plan window at signup: start = registration time, renew_date = start + 7 days."""
    start = now if now is not None else datetime.now(timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return {
        "plan": plan,
        "start_date": start,
        "renew_date": start + timedelta(days=7),
    }
