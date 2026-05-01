"""
Razorpay payments (USD): create order, verify checkout signature, optional webhook.

Fulfillment: add credits, optional plan/role/plan maps, ``credit_history``, ``payment_history``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import get_credit_history_collection, get_payment_history_collection, get_users_collection
from app.dependencies import get_current_user
from app.models.payment import CreateRazorpayOrderRequest, VerifyRazorpayPaymentRequest
from app.services.email_service import send_payment_receipt_email
from app.services.payment_fulfillment import apply_successful_payment, mark_payment_failed
from app.services.razorpay_api import (
    convert_usd_to_order_amount,
    create_order_amount,
    fetch_payment,
    safe_json,
    verify_payment_signature,
    verify_webhook_signature,
)

logger = logging.getLogger("payment")

router = APIRouter(prefix="/api/v1/payments/razorpay", tags=["Payments — Razorpay"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _format_invoice_dt(dt: Any) -> str:
    if dt is None:
        return ""
    if isinstance(dt, str):
        return dt
    try:
        return dt.strftime("%b %d, %Y %H:%M UTC")
    except Exception:
        return str(dt)


async def _send_payment_receipt_safe(payment_history_id: str) -> None:
    """Best-effort: load the paid payment_history + user and dispatch a receipt email."""
    try:
        ph_col = get_payment_history_collection()
        users_col = get_users_collection()
        ph = await ph_col.find_one({"payment_history_id": payment_history_id})
        if not ph or ph.get("status") != "paid":
            return
        user = await users_col.find_one({"user_id": ph.get("user_id")})
        if not user or not user.get("email"):
            return

        amount_usd = ph.get("amount_usd")
        try:
            paid_amount = f"{float(amount_usd):.2f}"
        except (TypeError, ValueError):
            paid_amount = str(amount_usd or "")
        currency = (ph.get("currency") or "USD").upper()

        is_plan = ph.get("credit_type") == "plan"
        purchase_type = "Plan purchase" if is_plan else "Credit purchase"
        plan_name = (ph.get("plan_type") or "").strip().title() or ("—" if not is_plan else "")
        timeperiod = (ph.get("timeperiod") or "").strip().title()
        plan_validity = f"{plan_name} ({timeperiod})" if (is_plan and timeperiod) else ("—" if not is_plan else plan_name or "—")

        invoice = {
            "invoice_id":      ph.get("payment_history_id"),
            "transaction_id":  ph.get("razorpay_payment_id"),
            "payment_date":    _format_invoice_dt(ph.get("updated_at") or ph.get("created_at")),
            "purchase_type":   purchase_type,
            "plan_name":       plan_name or "—",
            "credits_added":   str(ph.get("credit") or 0),
            "current_credits": str(round(float(user.get("credits", 0) or 0), 4)),
            "plan_validity":   plan_validity,
            "currency":        currency,
            "paid_amount":     paid_amount,
        }
        await send_payment_receipt_email(
            to_email=user["email"],
            first_name=user.get("first_name", ""),
            invoice=invoice,
        )
    except Exception as exc:  # noqa: BLE001 — never break the API on receipt mail
        logger.warning("payment receipt mail failed for %s: %s", payment_history_id, exc)


@router.post(
    "/create-order",
    status_code=status.HTTP_201_CREATED,
    summary="Create Razorpay order (USD)",
    description="Creates a Razorpay order and a ``payment_history`` row (``order_created``). "
    "Client completes payment in Razorpay Checkout, then calls ``/verify``.",
)
async def create_order(
    body: CreateRazorpayOrderRequest,
    current_user: dict = Depends(get_current_user),
):
    if not (getattr(settings, "RAZORPAY_KEY_ID", "") or "").strip() or not (
        getattr(settings, "RAZORPAY_KEY_SECRET", "") or ""
    ).strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Razorpay is not configured (RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET).",
        )

    user_id = current_user["user_id"]
    ph_id = str(uuid.uuid4())
    receipt = ph_id.replace("-", "")[:20]
    target_currency = (getattr(settings, "RAZORPAY_CURRENCY", "USD") or "USD").upper()
    try:
        conversion = await asyncio.to_thread(
            lambda: convert_usd_to_order_amount(body.amount, target_currency)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to convert USD amount to {target_currency}: {exc}",
        ) from exc

    amount_minor = int(conversion["amount_minor"])
    now = _now()

    notes: dict[str, str] = {
        "stylic_payment_id": str(ph_id),
        "user_id":           str(user_id),
        "credit_type":       str(body.credit_type),
        "credit":            str(body.credit),
        "amount_usd":        str(body.amount),
        "amount_converted":  str(conversion["amount_converted"]),
        "currency":          target_currency,
        "fx_rate":           str(conversion["fx_rate"]),
    }
    if body.plan_type is not None:
        notes["plan_type"] = str(body.plan_type)
    if body.timeperiod is not None:
        notes["timeperiod"] = str(body.timeperiod)

    col = get_payment_history_collection()
    try:
        order = await asyncio.to_thread(
            lambda: create_order_amount(
                amount_minor=amount_minor,
                currency=target_currency,
                receipt=receipt,
                notes=notes,
            )
        )
    except Exception as exc:
        await col.insert_one(
            {
                "payment_history_id": ph_id,
                "user_id":            user_id,
                "status":                "failed",
                "error":                 f"order_create: {exc!r}",
                "is_success":            False,
                "request_snapshot":     body.model_dump(),
                "amount_usd":            body.amount,
                "amount_cents":          amount_minor,
                "amount_minor":          amount_minor,
                "amount_converted":      conversion["amount_converted"],
                "fx_rate":               conversion["fx_rate"],
                "fx_provider":           conversion.get("fx_provider"),
                "fx_as_of":              conversion.get("fx_as_of"),
                "currency":            target_currency,
                "credit_type":         body.credit_type,
                "plan_type":            body.plan_type,
                "timeperiod":          body.timeperiod,
                "credit":              body.credit,
                "created_at":          now,
                "updated_at":          now,
            }
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Razorpay order creation failed: {exc}",
        ) from exc

    order_dict = safe_json(order)
    doc: dict[str, Any] = {
        "payment_history_id":  ph_id,
        "user_id":              user_id,
        "status":                "order_created",
        "is_success":            False,
        "is_credit_applied":     False,
        "is_plan_applied":        False,
        "request_snapshot":     body.model_dump(),
        "amount_usd":            body.amount,
        "amount_cents":          int(order.get("amount") or amount_minor),
        "amount_minor":          int(order.get("amount") or amount_minor),
        "amount_converted":      conversion["amount_converted"],
        "fx_rate":               conversion["fx_rate"],
        "fx_provider":           conversion.get("fx_provider"),
        "fx_as_of":              conversion.get("fx_as_of"),
        "currency":              (order.get("currency") or target_currency).upper(),
        "credit_type":            body.credit_type,
        "plan_type":             body.plan_type,
        "timeperiod":            body.timeperiod,
        "credit":                body.credit,
        "razorpay_order_id":     order.get("id") or order.get("order_id"),
        "raw_razorpay_order":   order_dict,
        "created_at":            now,
        "updated_at":            now,
    }
    await col.insert_one(doc)

    key_id = (getattr(settings, "RAZORPAY_KEY_ID", "") or "").strip()
    if not key_id:
        key_id = ""

    return {
        "stylic_payment_id":  ph_id,
        "razorpay_order_id":  doc.get("razorpay_order_id"),
        "order":              order_dict,
        "amount_minor":       int(order.get("amount") or amount_minor),
        "amount_converted":   conversion["amount_converted"],
        "fx_rate":            conversion["fx_rate"],
        "currency":           doc.get("currency", "USD"),
        "key_id":             key_id,
        "message":            "Use key_id and order with Razorpay Checkout, then call POST /verify.",
    }


async def _complete_verified_intent(
    body: VerifyRazorpayPaymentRequest,
    user_id: str,
) -> dict[str, Any]:
    col = get_payment_history_collection()
    ph = await col.find_one(
        {
            "payment_history_id": body.stylic_payment_id,
            "user_id":            user_id,
        }
    )
    if not ph:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Payment intent not found.")

    if ph.get("status") == "paid" and ph.get("razorpay_payment_id") == body.razorpay_payment_id:
        return {
            "message":    "Payment already completed.",
            "idempotent": True,
            "result":     {"credits": None},
        }

    now = _now()
    lock = await col.find_one_and_update(
        {
            "payment_history_id":  body.stylic_payment_id,
            "user_id":               user_id,
            "status":                "order_created",
            "razorpay_order_id":     body.razorpay_order_id,
        },
        {"$set": {"status": "verifying", "updated_at": now}},
    )
    if not lock:
        ex = await col.find_one(
            {"payment_history_id": body.stylic_payment_id, "user_id": user_id}
        )
        if ex and ex.get("status") == "paid":
            return {
                "message":    "Payment already completed.",
                "idempotent": True,
            }
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payment cannot be verified (wrong order id or not pending).",
        )

    try:
        def _verify() -> None:
            verify_payment_signature(
                order_id=body.razorpay_order_id,
                payment_id=body.razorpay_payment_id,
                signature=body.razorpay_signature,
            )
        await asyncio.to_thread(_verify)
    except Exception as exc:
        await mark_payment_failed(
            col, body.stylic_payment_id, f"signature: {exc!r}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Signature verification failed.",
        ) from exc

    try:
        pay = await asyncio.to_thread(fetch_payment, body.razorpay_payment_id)
    except Exception as exc:
        await mark_payment_failed(
            col, body.stylic_payment_id, f"fetch_payment: {exc!r}"
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not fetch payment: {exc}",
        ) from exc

    st = (pay.get("status") or "").lower()
    if st not in ("captured", "authorized"):
        await mark_payment_failed(
            col,
            body.stylic_payment_id,
            f"bad_payment_status:{st}",
            raw=pay,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Payment not successful (status={st}).",
        )

    if (pay.get("order_id") or pay.get("orderId")) and str(pay.get("order_id") or pay.get("orderId")) != str(body.razorpay_order_id):
        await mark_payment_failed(
            col, body.stylic_payment_id, "order_id_mismatch", raw=pay
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Order id mismatch on payment.")

    expected_minor = int(lock.get("amount_minor") or lock.get("amount_cents") or 0)
    if int(pay.get("amount") or 0) != expected_minor:
        await mark_payment_failed(
            col, body.stylic_payment_id, "amount_mismatch", raw=pay
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Paid amount does not match order amount.",
        )

    lock["razorpay_payment_id"] = body.razorpay_payment_id
    users_col = get_users_collection()
    ch_col = get_credit_history_collection()
    try:
        result = await apply_successful_payment(
            users_col=users_col,
            credit_history_col=ch_col,
            payment_history_col=col,
            payment_doc=lock,
            razorpay_payment=pay,
        )
    except Exception as exc:
        await col.update_one(
            {"payment_history_id": body.stylic_payment_id},
            {"$set": {
                "status":     "payout_failed",
                "error":      str(exc),
                "updated_at":  _now(),
            }},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fulfillment failed: {exc}",
        ) from exc

    return {
        "message": "Payment verified and credits applied.",
        "result":  result,
    }


@router.post(
    "/verify",
    status_code=status.HTTP_200_OK,
    summary="Verify Razorpay payment and fulfill",
)
async def verify_payment(
    body: VerifyRazorpayPaymentRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    result = await _complete_verified_intent(body, current_user["user_id"])
    if not result.get("idempotent"):
        background_tasks.add_task(_send_payment_receipt_safe, body.stylic_payment_id)
    return result


@router.post(
    "/webhook",
    status_code=status.HTTP_200_OK,
    summary="Razorpay webhooks (optional backup)",
    description="Verifies ``X-Razorpay-Signature`` when ``RAZORPAY_WEBHOOK_SECRET`` is set. "
    "Idempotent with ``/verify``.",
    include_in_schema=True,
)
async def razorpay_webhook(request: Request, background_tasks: BackgroundTasks):
    wh_secret = (getattr(settings, "RAZORPAY_WEBHOOK_SECRET", "") or "").strip()
    if not wh_secret:
        return JSONResponse(
            status_code=503,
            content={"ok": False, "detail": "Set RAZORPAY_WEBHOOK_SECRET to enable webhooks."},
        )

    body = await request.body()
    sig = request.headers.get("X-Razorpay-Signature", "") or request.headers.get("x-razorpay-signature", "")

    if not verify_webhook_signature(body, sig):
        return JSONResponse(status_code=400, content={"ok": False, "detail": "Invalid signature"})

    try:
        data = json.loads(body.decode("utf-8") if body else "{}")
    except Exception:
        return {"ok": True, "detail": "ignored non-json body"}

    event = (data.get("event") or "").lower()
    payment_entity: dict[str, Any] = {}
    if "payment" in (data.get("payload") or {}):
        ent = (data.get("payload") or {}).get("payment") or {}
        payment_entity = ent.get("entity") or ent
    if not payment_entity and isinstance(data, dict) and "entity" in data:
        # rare shapes
        payment_entity = (data.get("entity") or {})  # type: ignore[assignment]

    order_id = (payment_entity.get("order_id") or payment_entity.get("orderId") or "")
    pay_id = (payment_entity.get("id") or payment_entity.get("payment_id") or "")
    if not order_id or not pay_id:
        return {"ok": True, "event": event, "detail": "no payment in payload"}

    st = (payment_entity.get("status") or "").lower()
    if st and st not in ("captured", "authorized", "pending"):
        col = get_payment_history_collection()
        one = await col.find_one({"razorpay_order_id": order_id})
        if one and one.get("status") == "order_created":
            await col.update_one(
                {"_id": one["_id"]},
                {"$set": {
                    "status":              "failed",
                    "is_success":          False,
                    "raw_webhook_event":  data,
                    "error":               f"webhook_payment_{st}",
                    "updated_at":         _now(),
                }},
            )
        return {"ok": True, "event": event, "handled": "failed_state"}

    col = get_payment_history_collection()
    ph = await col.find_one(
        {
            "razorpay_order_id": order_id,
            "status":            "order_created",
        }
    )
    if not ph:
        return {"ok": True, "detail": "no matching pending intent or already processed"}

    user_id = ph["user_id"]
    if ph.get("status") == "paid":
        return {"ok": True, "idempotent": True}

    try:
        pay = await asyncio.to_thread(fetch_payment, str(pay_id))
    except Exception as exc:
        logger.warning("[webhook] fetch_payment failed: %s", exc)
        return {"ok": True, "detail": "fetch later"}

    if (pay.get("order_id") or pay.get("orderId")) and str(pay.get("order_id") or pay.get("orderId")) != str(order_id):
        return {"ok": True, "detail": "order mismatch"}

    now = _now()
    lo = await col.find_one_and_update(
        {
            "payment_history_id": ph["payment_history_id"],
            "user_id":            user_id,
            "status":               "order_created",
        },
        {"$set": {"status": "verifying", "updated_at": now}},
    )
    if not lo:
        return {"ok": True, "idempotent": True}

    stp = (pay.get("status") or "").lower()
    if stp not in ("captured", "authorized"):
        await mark_payment_failed(col, ph["payment_history_id"], f"webhook_fetch_status:{stp}", raw=pay)
        return {"ok": True}

    expected_minor = int(lo.get("amount_minor") or lo.get("amount_cents") or 0)
    if int(pay.get("amount") or 0) != expected_minor:
        await mark_payment_failed(col, ph["payment_history_id"], "webhook_amount_mismatch", raw=pay)
        return {"ok": True}

    ph_u = {**lo, "razorpay_payment_id": str(pay_id)}
    try:
        out = await apply_successful_payment(
            users_col=get_users_collection(),
            credit_history_col=get_credit_history_collection(),
            payment_history_col=col,
            payment_doc=ph_u,
            razorpay_payment=pay,
        )
    except Exception as exc:
        await col.update_one(
            {"payment_history_id": ph["payment_history_id"]},
            {"$set": {"status": "payout_failed", "error": str(exc), "raw_webhook_event": data, "updated_at": _now()}},
        )
        return {"ok": True, "error": str(exc)}

    background_tasks.add_task(_send_payment_receipt_safe, ph["payment_history_id"])
    return {"ok": True, "fulfilled": True, "credits": out.get("credits")}


# expose whether Razorpay is configured (e.g. health for frontend)
@router.get("/config", summary="Public Razorpay key id for checkout")
def razorpay_config():
    return {
        "key_id":    (getattr(settings, "RAZORPAY_KEY_ID", "") or "").strip(),
        "currency":  (getattr(settings, "RAZORPAY_CURRENCY", "USD") or "USD").upper(),
        "configured": bool((getattr(settings, "RAZORPAY_KEY_ID", "") or "").strip() and (getattr(settings, "RAZORPAY_KEY_SECRET", "") or "").strip()),
    }
