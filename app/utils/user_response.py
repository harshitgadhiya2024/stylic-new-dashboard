"""Normalize user documents for API responses (safe fields + plan/role dicts)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.constants.free_plan import FREE_ROLE_MAPPING_DICT, build_free_plan_mapping_dict


def _as_utc_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def user_dict_for_api(user: dict) -> dict:
    """Copy user for JSON: drop secrets, always include ``role_mapping_dict`` and ``plan_mapping_dict``."""
    out: dict[str, Any] = dict(user)
    out.pop("_id", None)
    out.pop("password", None)

    if "role_mapping_dict" not in out or out["role_mapping_dict"] is None:
        out["role_mapping_dict"] = dict(FREE_ROLE_MAPPING_DICT)

    if "plan_mapping_dict" not in out or out["plan_mapping_dict"] is None:
        created = out.get("created_at")
        if isinstance(created, datetime):
            start = _as_utc_aware(created)
        else:
            start = datetime.now(timezone.utc)
        plan_key = out.get("plan") or settings.DEFAULT_PLAN or "free"
        if isinstance(plan_key, str):
            plan_key = plan_key.strip() or "free"
        else:
            plan_key = str(settings.DEFAULT_PLAN or "free")
        out["plan_mapping_dict"] = build_free_plan_mapping_dict(start, plan_key)

    return out
