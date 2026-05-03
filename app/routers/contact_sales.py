"""
Public contact / sales form — no auth. Heavily validated, rate-limited, spam-resistant.

Admin routes (dashboard JWT): list, status update, hard delete by ``submission_id``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import ValidationError

from app.database import get_contact_sales_collection
from app.dependencies import require_admin_roles, require_user_management_read
from app.models.contact_sales import (
    AdminContactSalesStatusUpdate,
    ContactSalesRequest,
    ContactSalesResponse,
)
from app.services.contact_rate_limit import enforce_rate_limits_for_contact
from app.services.contact_sanitize import should_block_honeypot
from app.services.contact_sales_service import process_contact_sales_submission

logger = logging.getLogger("contact_sales.router")

router = APIRouter(prefix="/api/v1", tags=["Contact / Sales"])

admin_router = APIRouter(
    prefix="/api/v1/admins/contact-sales",
    tags=["Admin — Contact sales"],
)


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


# ══════════════════════════════════════════════════════════════════════════
# Admin (dashboard JWT)
# ══════════════════════════════════════════════════════════════════════════


def _strip_contact_sales_doc(doc: dict[str, Any]) -> dict[str, Any]:
    out = dict(doc)
    out.pop("_id", None)
    return out


@admin_router.get(
    "/",
    summary="List contact sales submissions (paginated)",
    description=(
        "Returns rows from the ``contact_sales`` collection, newest first, with pagination. "
        "Requires dashboard admin JWT with user-management **read** "
        "(**superadmin**, **admin**, **developer**)."
    ),
)
async def admin_list_contact_sales(
    page: int = Query(1, ge=1, description="1-based page number"),
    limit: int = Query(
        100,
        ge=1,
        le=500,
        description="Items per page (max 500).",
    ),
    _viewer: dict = Depends(require_user_management_read()),
) -> dict[str, Any]:
    _ = _viewer
    col = get_contact_sales_collection()
    query: dict[str, Any] = {}
    skip = (page - 1) * int(limit)
    total = await col.count_documents(query)
    lim = int(limit)
    total_pages = max(1, (total + lim - 1) // lim) if total else 1

    cursor = (
        col.find(query)
        .sort("created_at", -1)
        .skip(skip)
        .limit(lim)
    )
    rows = await cursor.to_list(length=lim)
    items = [_strip_contact_sales_doc(r) for r in rows]

    return {
        "total": total,
        "page": page,
        "limit": lim,
        "total_pages": total_pages,
        "contact_sales_data": items,
    }


@admin_router.patch(
    "/{submission_id}/status",
    summary="Update contact sales status",
    description=(
        "Sets ``status`` to one of: **pending**, **processing**, **completed**. "
        "Requires dashboard admin JWT (**superadmin**, **admin** only)."
    ),
)
async def admin_update_contact_sales_status(
    submission_id: str,
    body: AdminContactSalesStatusUpdate,
    _admin: dict = Depends(require_admin_roles("superadmin", "admin")),
) -> dict[str, Any]:
    _ = _admin
    col = get_contact_sales_collection()
    doc = await col.find_one({"submission_id": submission_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contact sales submission not found.",
        )

    now = datetime.now(timezone.utc)
    await col.update_one(
        {"submission_id": submission_id},
        {"$set": {"status": body.status, "updated_at": now}},
    )
    fresh = await col.find_one({"submission_id": submission_id})
    return {
        "success": True,
        "message": "Status updated.",
        "submission_id": submission_id,
        "data": _strip_contact_sales_doc(fresh or doc),
    }


@admin_router.delete(
    "/{submission_id}",
    summary="Hard-delete contact sales submission",
    description=(
        "Permanently removes the MongoDB document for ``submission_id``. "
        "Requires dashboard admin JWT (**superadmin**, **admin** only)."
    ),
)
async def admin_hard_delete_contact_sales(
    submission_id: str,
    _admin: dict = Depends(require_admin_roles("superadmin", "admin")),
) -> dict[str, Any]:
    _ = _admin
    col = get_contact_sales_collection()
    result = await col.delete_one({"submission_id": submission_id})
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Contact sales submission not found.",
        )
    return {
        "success": True,
        "message": "Submission deleted.",
        "submission_id": submission_id,
    }
