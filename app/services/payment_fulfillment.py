"""
Apply successful Razorpay payment: user credits, optional plan/role/plan maps,
``credit_history``, and final ``payment_history`` status.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from motor.motor_asyncio import AsyncIOMotorCollection

from app.constants.paid_plans import build_paid_plan_mapping_dict, role_mapping_for_plan


async def apply_successful_payment(
    *,
    users_col: AsyncIOMotorCollection,
    credit_history_col: AsyncIOMotorCollection,
    payment_history_col: AsyncIOMotorCollection,
    payment_doc: dict[str, Any],
    razorpay_payment: dict[str, Any],
) -> dict[str, Any]:
    """
    Idempotent only if caller already filtered ``payment_history`` to ``order_created`` and
    moved to a locked state. This function sets user, credit_history, and ``status=paid``.
    """
    user_id = payment_doc["user_id"]
    credit_add = float(payment_doc.get("credit") or 0)
    credit_type = payment_doc.get("credit_type")
    now = datetime.now(timezone.utc)

    user = await users_col.find_one({"user_id": user_id})
    if not user:
        raise ValueError("User not found for payment")

    old_credits = float(user.get("credits", 0))
    new_credits = round(old_credits + credit_add, 4)

    user_set: dict[str, Any] = {
        "credits":     new_credits,
        "updated_at":  now,
    }
    if credit_type == "plan":
        pt = (payment_doc.get("plan_type") or "").lower()
        tp = payment_doc.get("timeperiod")
        if tp not in ("monthly", "yearly"):
            tp = "monthly"
        user_set["plan"] = pt
        user_set["role_mapping_dict"] = role_mapping_for_plan(pt)
        user_set["plan_mapping_dict"] = build_paid_plan_mapping_dict(
            pt,
            tp,  # type: ignore[arg-type]
            start=now,
        )

    await users_col.update_one(
        {"user_id": user_id},
        {"$set": user_set},
    )

    history_id = str(uuid.uuid4())
    rpid = (razorpay_payment or {}).get("id") or payment_doc.get("razorpay_payment_id")
    feature = "razorpay_plan_purchase" if credit_type == "plan" else "razorpay_credit_purchase"
    ch_doc: dict[str, Any] = {
        "history_id":          history_id,
        "user_id":             user_id,
        "feature_name":        feature,
        "credit":              credit_add,
        "type":                "add",
        "thumbnail_image":     "",
        "notes":               f"Razorpay payment {rpid} — {credit_type} — credit +{credit_add}",
        "payment_history_id":  payment_doc.get("payment_history_id"),
        "razorpay_order_id":  payment_doc.get("razorpay_order_id"),
        "razorpay_payment_id": rpid,
        "amount_usd":         payment_doc.get("amount_usd"),
        "currency":           payment_doc.get("currency", "USD"),
        "amount_cents":        payment_doc.get("amount_cents"),
        "credit_type":         credit_type,
        "plan_type":           payment_doc.get("plan_type"),
        "timeperiod":          payment_doc.get("timeperiod"),
        "created_at":          now,
    }
    if isinstance(razorpay_payment, dict) and razorpay_payment:
        ch_doc["razorpay_payment_snapshot"] = dict(
            (k, razorpay_payment[k]) for k in list(razorpay_payment.keys())[:80]
        )

    await credit_history_col.insert_one(ch_doc)

    await payment_history_col.update_one(
        {"payment_history_id": payment_doc["payment_history_id"]},
        {"$set": {
            "status":              "paid",
            "is_success":          True,
            "is_credit_applied":   True,
            "is_plan_applied":        credit_type == "plan",
            "razorpay_payment_id":  rpid,
            "raw_razorpay_payment":  razorpay_payment,
            "updated_at":             now,
        }},
    )

    return {
        "credits":       new_credits,
        "credit_added":  credit_add,
        "feature_name":  feature,
        "history_id":   history_id,
    }


async def mark_payment_failed(
    payment_history_col: AsyncIOMotorCollection,
    payment_history_id: str,
    error: str,
    *,
    raw: Optional[dict] = None,
) -> None:
    now = datetime.now(timezone.utc)
    u: dict[str, Any] = {
        "status":     "failed",
        "error":     error,
        "is_success": False,
        "updated_at": now,
    }
    if raw is not None:
        u["raw_razorpay_error_context"] = raw
    await payment_history_col.update_one(
        {"payment_history_id": payment_history_id},
        {"$set": u},
    )
