"""
Contact sales: sanitize → send confirmation email → persist (order required).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from app.database import get_contact_sales_collection
from app.models.contact_sales import ContactSalesRequest
from app.services.contact_sanitize import (
    sanitize_message,
    sanitize_optional_phone,
    sanitize_plain_line,
)
from app.services.contact_rate_limit import record_successful_submission_rl
from app.services.email_service import send_contact_thank_you_email

logger = logging.getLogger("contact_sales")

_MAX = {
    "first_name":  100,
    "last_name":   100,
    "message":     5000,
}


async def process_contact_sales_submission(body: ContactSalesRequest) -> dict[str, Any]:
    first = sanitize_plain_line(body.first_name, _MAX["first_name"])
    last = sanitize_plain_line(body.last_name, _MAX["last_name"])
    phone = sanitize_optional_phone(body.phone)
    msg = sanitize_message(body.message, _MAX["message"])
    email = body.email  # already normalized by Pydantic

    await send_contact_thank_you_email(to_email=email, first_name=first)

    sub_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    doc: Dict[str, Any] = {
        "submission_id":         sub_id,
        "first_name":            first,
        "last_name":             last,
        "email":                 email,
        "phone":                 phone,
        "message":               msg,
        "conversation_channel":  "email",
        "status":                "pending",
        "created_at":            now,
        "updated_at":            now,
    }
    col = get_contact_sales_collection()
    try:
        await col.insert_one(doc)
    except Exception:
        logger.exception(
            "Contact sales: confirmation email was sent but DB insert failed for %s",
            email,
        )
        raise RuntimeError("Failed to save submission after email was sent.") from None

    record_successful_submission_rl(email)
    return {"submission_id": sub_id}
