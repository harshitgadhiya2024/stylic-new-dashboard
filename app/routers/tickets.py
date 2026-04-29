"""User ticket management (authenticated)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.database import get_tickets_collection
from app.dependencies import get_current_user
from app.models.ticket import CreateTicketRequest, TicketRecord

router = APIRouter(prefix="/api/v1/user/tickets", tags=["User — tickets"])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_public(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if k != "_id"}


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=TicketRecord,
    summary="Create ticket",
)
async def create_ticket(
    body: CreateTicketRequest,
    current_user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    uid = str(current_user["user_id"])
    now = _now()
    images = [str(u).strip() for u in (body.images or []) if str(u).strip()]

    doc: dict[str, Any] = {
        "ticket_id": str(uuid.uuid4()),
        "user_id": uid,
        "ticket_type": body.ticket_type.strip(),
        "descriptions": body.descriptions,
        "images": images,
        "notes": (body.notes or "").strip(),
        "status": "pending",
        "is_active": True,
        "created_at": now,
        "updated_at": now,
    }
    col = get_tickets_collection()
    await col.insert_one(doc)
    return _to_public(doc)


@router.get(
    "/{ticket_id}",
    response_model=TicketRecord,
    summary="Get ticket details",
)
async def get_ticket(
    ticket_id: str,
    current_user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    uid = str(current_user["user_id"])
    col = get_tickets_collection()
    doc = await col.find_one({"ticket_id": ticket_id, "user_id": uid})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ticket not found.",
        )
    return _to_public(doc)


@router.delete(
    "/{ticket_id}",
    response_model=TicketRecord,
    summary="Cancel ticket (soft delete)",
    description="Sets status to cancelled and is_active to false.",
)
async def delete_ticket(
    ticket_id: str,
    current_user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    uid = str(current_user["user_id"])
    col = get_tickets_collection()
    now = _now()
    res = await col.update_one(
        {"ticket_id": ticket_id, "user_id": uid},
        {
            "$set": {
                "status": "cancelled",
                "is_active": False,
                "updated_at": now,
            }
        },
    )
    if res.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ticket not found.",
        )
    fresh = await col.find_one({"ticket_id": ticket_id, "user_id": uid})
    if not fresh:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ticket not found.",
        )
    return _to_public(fresh)
