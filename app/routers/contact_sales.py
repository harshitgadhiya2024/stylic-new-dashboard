"""
Public contact / sales form — no auth. Heavily validated, rate-limited, spam-resistant.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import ValidationError

from app.models.contact_sales import ContactSalesRequest, ContactSalesResponse
from app.services.contact_rate_limit import enforce_rate_limits_for_contact
from app.services.contact_sanitize import should_block_honeypot
from app.services.contact_sales_service import process_contact_sales_submission

logger = logging.getLogger("contact_sales.router")

router = APIRouter(prefix="/api/v1", tags=["Contact / Sales"])


def _format_validation_errors(exc: ValidationError) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", ()) if p != "body"]
        field = ".".join(loc) if loc else "body"
        errors.append(
            {
                "field": field,
                "message": err.get("msg", "Invalid value"),
            }
        )
    return errors


@router.post(
    "/contact-sales",
    status_code=status.HTTP_201_CREATED,
    response_model=ContactSalesResponse,
    summary="Contact sales (public form)",
    description=(
        "Submit the marketing site contact / sales form. Honeypot, strict plaintext validation, "
        "per-IP and per-email rate limits, then confirmation email to the work address; "
        "a database record is created only after the message is sent."
    ),
)
async def post_contact_sales(request: Request) -> ContactSalesResponse:
    _ok_msg = "Thank you. Our team will get back to you soon."
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Validation failed",
                "errors": [{"field": "body", "message": "Request body must be valid JSON"}],
            },
        ) from None

    try:
        body = ContactSalesRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Validation failed",
                "errors": _format_validation_errors(exc),
            },
        ) from None

    if should_block_honeypot(body.website):
        # Indistinguishable success for bots; no email, no DB, no rate counters
        return ContactSalesResponse(ok=True, message=_ok_msg)

    enforce_rate_limits_for_contact(request, body.work_email)

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
