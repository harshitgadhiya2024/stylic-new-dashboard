"""
Admin APIs for ``delete_account_request`` (dashboard JWT).

Mongo keys match existing inserts: ``delete-request-id``, ``user-id``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.database import get_delete_account_request_collection
from app.dependencies import require_admin_roles, require_user_management_read

router = APIRouter(
    prefix="/api/v1/admins/delete-account-requests",
    tags=["Admin — Delete account requests"],
)

_DELETE_REQUEST_ID_KEY = "delete-request-id"


def _strip_doc(doc: dict[str, Any]) -> dict[str, Any]:
    out = dict(doc)
    out.pop("_id", None)
    return out


@router.get(
    "/",
    summary="List all delete account requests (paginated)",
    description=(
        "Returns rows from the ``delete_account_request`` collection, newest first. "
        "Requires dashboard admin JWT with user-management **read** "
        "(**superadmin**, **admin**, **developer**)."
    ),
)
async def admin_list_delete_account_requests(
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
    col = get_delete_account_request_collection()
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
    items = [_strip_doc(r) for r in rows]

    return {
        "total": total,
        "page": page,
        "limit": lim,
        "total_pages": total_pages,
        "delete_account_requests": items,
    }


@router.patch(
    "/{delete_request_id}/toggle-active",
    summary="Toggle delete-account request active state",
    description=(
        "Flips ``is_active`` for the row identified by ``delete-request-id``. "
        "Missing ``is_active`` is treated as ``True``. "
        "Requires dashboard admin JWT (**superadmin**, **admin** only)."
    ),
)
async def admin_toggle_delete_account_request_active(
    delete_request_id: str,
    _admin: dict = Depends(require_admin_roles("superadmin", "admin")),
) -> dict[str, Any]:
    _ = _admin
    col = get_delete_account_request_collection()
    doc = await col.find_one({_DELETE_REQUEST_ID_KEY: delete_request_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Delete account request not found.",
        )

    currently_active = bool(doc.get("is_active", True))
    new_active = not currently_active
    now = datetime.now(timezone.utc)

    await col.update_one(
        {_DELETE_REQUEST_ID_KEY: delete_request_id},
        {"$set": {"is_active": new_active, "updated_at": now}},
    )
    fresh = await col.find_one({_DELETE_REQUEST_ID_KEY: delete_request_id})
    msg = "Request activated." if new_active else "Request deactivated."

    return {
        "success": True,
        "message": msg,
        "delete_request_id": delete_request_id,
        "is_active": new_active,
        "data": _strip_doc(fresh or doc),
    }


@router.delete(
    "/{delete_request_id}",
    summary="Hard-delete delete account request",
    description=(
        "Permanently removes the MongoDB document whose ``delete-request-id`` matches. "
        "Requires dashboard admin JWT (**superadmin**, **admin** only)."
    ),
)
async def admin_hard_delete_delete_account_request(
    delete_request_id: str,
    _admin: dict = Depends(require_admin_roles("superadmin", "admin")),
) -> dict[str, Any]:
    _ = _admin
    col = get_delete_account_request_collection()
    result = await col.delete_one({_DELETE_REQUEST_ID_KEY: delete_request_id})
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Delete account request not found.",
        )
    return {
        "success": True,
        "message": "Delete account request removed.",
        "delete_request_id": delete_request_id,
    }
