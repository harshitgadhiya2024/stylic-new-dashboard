"""Normalize user documents for API responses (safe fields + plan/role dicts)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.constants.free_plan import FREE_ROLE_MAPPING_DICT, build_free_plan_mapping_dict
from app.database import get_credit_history_collection


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


async def user_dict_for_api_with_credit_metrics(user: dict) -> dict:
    """
    Normalize user and append credit dashboard fields:
    - credits: current balance (already present)
    - used_credit: total deducted credit from credit_history
    - total_credit: credits + used_credit
    - credit_percentage: int((credits/total_credit)*100), 0 when total_credit is 0
    """
    out = user_dict_for_api(user)
    user_id = str(out.get("user_id") or "").strip()

    try:
        current_credit = float(out.get("credits", 0) or 0)
    except Exception:
        current_credit = 0.0
    if current_credit < 0:
        current_credit = 0.0

    used_credit = 0.0
    if user_id:
        history_col = get_credit_history_collection()
        cursor = history_col.find({"user_id": user_id}, {"credit": 1, "type": 1})
        async for row in cursor:
            row_type = str(row.get("type") or "").strip().lower()
            if row_type and row_type != "deduct":
                continue
            try:
                c = float(row.get("credit", 0) or 0)
            except Exception:
                c = 0.0
            if c > 0:
                used_credit += c

    total_credit = current_credit + used_credit
    credit_percentage = int((current_credit / total_credit) * 100) if total_credit > 0 else 0

    out["credits"] = round(current_credit, 4)
    out["used_credit"] = round(used_credit, 4)
    out["total_credit"] = round(total_credit, 4)
    out["credit_percentage"] = int(credit_percentage)
    return out
