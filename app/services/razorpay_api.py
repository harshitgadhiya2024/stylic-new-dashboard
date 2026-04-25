"""
Razorpay REST helpers (sync client). Use ``asyncio.to_thread`` from FastAPI routes.

Amount input from API is USD. Orders are created in ``settings.RAZORPAY_CURRENCY``
after real-time FX conversion.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import requests
import razorpay

from app.config import settings


def get_razorpay_client() -> "razorpay.Client":
    key_id = (getattr(settings, "RAZORPAY_KEY_ID", "") or "").strip()
    secret = (getattr(settings, "RAZORPAY_KEY_SECRET", "") or "").strip()
    if not key_id or not secret:
        raise RuntimeError("RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET must be set")
    return razorpay.Client(auth=(key_id, secret))


def _minor_unit_decimals(currency: str) -> int:
    c = (currency or "USD").upper()
    zero_decimal = {"JPY", "KRW", "VND"}
    three_decimal = {"BHD", "KWD", "OMR", "JOD", "TND"}
    if c in zero_decimal:
        return 0
    if c in three_decimal:
        return 3
    return 2


def _to_minor(amount: float, currency: str) -> int:
    decimals = _minor_unit_decimals(currency)
    minor = int(round(float(amount) * (10 ** decimals)))
    return max(1, minor)


def fetch_live_fx_rate_usd_to(target_currency: str) -> dict[str, Any]:
    """
    Fetch live FX rate from USD to ``target_currency``.
    Uses exchangerate.host (no auth) by default.
    """
    target = (target_currency or "USD").upper()
    if target == "USD":
        return {
            "rate": 1.0,
            "provider": "identity",
            "as_of": datetime.now(timezone.utc),
        }

    timeout = int(getattr(settings, "FX_RATE_TIMEOUT_S", 10) or 10)
    configured = (getattr(settings, "FX_RATE_API_URL", "") or "").strip()
    providers: list[tuple[str, dict[str, str], str]] = []
    if configured:
        providers.append((configured, {"from": "USD", "to": target, "amount": "1"}, "configured"))
    providers.extend([
        ("https://api.exchangerate.host/convert", {"from": "USD", "to": target, "amount": "1"}, "exchangerate.host"),
        ("https://api.frankfurter.app/latest", {"from": "USD", "to": target, "amount": "1"}, "frankfurter.app"),
        ("https://open.er-api.com/v6/latest/USD", {}, "open.er-api.com"),
    ])

    errors: list[str] = []
    for url, params, name in providers:
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            body = resp.json() or {}
            rate = None
            # exchangerate.host
            if isinstance(body.get("result"), (int, float)):
                rate = float(body["result"])
            elif isinstance(body.get("info"), dict) and isinstance(body["info"].get("rate"), (int, float)):
                rate = float(body["info"]["rate"])
            # frankfurter
            elif isinstance(body.get("rates"), dict) and isinstance(body["rates"].get(target), (int, float)):
                rate = float(body["rates"][target])
            # open.er-api
            elif isinstance(body.get("rates"), dict) and isinstance(body["rates"].get(target), (int, float)):
                rate = float(body["rates"][target])

            if rate and rate > 0:
                return {
                    "rate": rate,
                    "provider": name,
                    "as_of": datetime.now(timezone.utc),
                    "raw": body,
                }
            errors.append(f"{name}: invalid response {body}")
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    raise RuntimeError(f"FX conversion failed for USD->{target}. Errors: {' | '.join(errors)}")


def create_order_amount(
    *,
    amount_minor: int,
    currency: str,
    receipt: str,
    notes: dict[str, str],
) -> dict[str, Any]:
    client = get_razorpay_client()
    cc = (currency or "USD").upper()
    data = {
        "amount":   max(1, int(amount_minor)),
        "currency": cc,
        "receipt":  receipt[:40],
        "notes":    notes,
    }
    return client.order.create(data=data)  # type: ignore[no-any-return]


def convert_usd_to_order_amount(amount_usd: float, target_currency: str) -> dict[str, Any]:
    target = (target_currency or "USD").upper()
    fx = fetch_live_fx_rate_usd_to(target)
    converted = round(float(amount_usd) * float(fx["rate"]), 6)
    amount_minor = _to_minor(converted, target)
    return {
        "amount_usd": amount_usd,
        "currency": target,
        "fx_rate": float(fx["rate"]),
        "amount_converted": converted,
        "amount_minor": amount_minor,
        "fx_provider": fx.get("provider"),
        "fx_as_of": fx.get("as_of"),
        "fx_raw": fx.get("raw"),
    }


def fetch_payment(payment_id: str) -> dict[str, Any]:
    client = get_razorpay_client()
    p = client.payment.fetch(payment_id)  # type: ignore[no-any-return]
    return dict(p) if not isinstance(p, dict) else p


def verify_payment_signature(*, order_id: str, payment_id: str, signature: str) -> None:
    client = get_razorpay_client()
    client.utility.verify_payment_signature(  # raises on failure
        {
            "razorpay_order_id":   order_id,
            "razorpay_payment_id": payment_id,
            "razorpay_signature":  signature,
        },
    )


def verify_webhook_signature(body: bytes, signature_header: str) -> bool:
    secret = (getattr(settings, "RAZORPAY_WEBHOOK_SECRET", "") or "").strip()
    if not secret or not signature_header:
        return False
    try:
        text = body.decode("utf-8") if isinstance(body, (bytes, bytearray)) else str(body)
        get_razorpay_client().utility.verify_webhook_signature(
            text,
            signature_header,
            secret,
        )
        return True
    except Exception:
        return False


def safe_json(obj: Any) -> Any:
    """Make Razorpay dict JSON-serializable (best-effort)."""
    try:
        if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
            return obj
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)
