"""
Public contact / sales form — no auth. Heavily validated, rate-limited, spam-resistant.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status

from app.models.contact_sales import ContactSalesRequest, ContactSalesResponse
from app.services.contact_rate_limit import enforce_rate_limits_for_contact
from app.services.contact_sanitize import should_block_honeypot
from app.services.contact_sales_service import process_contact_sales_submission

logger = logging.getLogger("contact_sales.router")

router = APIRouter(prefix="/api/v1", tags=["Contact / Sales"])


@router.post(
    "/contact-sales",
    status_code=status.HTTP_201_CREATED,
    response_model=ContactSalesResponse,
    summary="Contact sales (public form)",
    description=(
        "Submit the marketing site contact / sales form. Honeypot, strict plaintext validation, "
        "per-IP and per-email rate limits, then confirmation email; "
        "a database record is created only after the message is sent."
    ),
)
async def post_contact_sales(request: Request, body: ContactSalesRequest) -> ContactSalesResponse:
    _ok_msg = "Thank you. Our team will get back to you soon."
    if should_block_honeypot(body.website):
        # Indistinguishable success for bots; no email, no DB, no rate counters
        return ContactSalesResponse(ok=True, message=_ok_msg)

    enforce_rate_limits_for_contact(request, body.email)

    try:
        await process_contact_sales_submission(body)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.exception("Contact sales: send or persist failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to complete your request at the moment. Please try again in a few minutes.",
        ) from exc

    return ContactSalesResponse(ok=True, message=_ok_msg)
