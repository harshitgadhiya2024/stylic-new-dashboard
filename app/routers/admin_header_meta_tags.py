"""Admin APIs for page header meta tags (adding-admin-headers)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pymongo import ReturnDocument

from app.database import get_header_meta_tags_collection
from app.dependencies import require_admin_roles
from app.models.header_meta_tag import (
    CreateHeaderMetaTagRequest,
    HeaderMetaTagListResponse,
    HeaderMetaTagRecord,
    UpdateHeaderMetaTagRequest,
)

router = APIRouter(
    prefix="/api/v1/admins/adding-admin-headers",
    tags=["Admin — Header meta tags"],
    dependencies=[Depends(require_admin_roles("superadmin", "admin"))],
)

# Public (no auth) router — kept under the same Swagger tag so it shows up
# alongside the admin endpoints in the same section of /docs.
public_router = APIRouter(
    prefix="/api/v1/admins/adding-admin-headers",
    tags=["Admin — Header meta tags"],
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _doc_public(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if k != "_id"}


@router.post(
    "/create-header-meta-tag",
    status_code=status.HTTP_201_CREATED,
    response_model=HeaderMetaTagRecord,
    summary="Create header meta tag",
    description="Stores a new ``header_meta_tag`` document in MongoDB.",
)
async def create_header_meta_tag(body: CreateHeaderMetaTagRequest) -> dict[str, Any]:
    now = _now()
    doc = {
        "meta_tag_id": str(uuid.uuid4()),
        "header_meta_tag": body.header_meta_tag,
        "created_at": now,
        "updated_at": now,
    }
    col = get_header_meta_tags_collection()
    await col.insert_one(doc)
    return _doc_public(doc)


@router.get(
    "/get-all-meta-tags",
    response_model=HeaderMetaTagListResponse,
    summary="Get all meta tags",
    description="Returns every stored header meta tag, newest first.",
)
async def get_all_meta_tags() -> dict[str, Any]:
    col = get_header_meta_tags_collection()
    cur = col.find({}).sort("created_at", -1)
    items = [_doc_public(d) async for d in cur]
    return {"total": len(items), "meta_tags": items}


@router.patch(
    "/update-meta-tags",
    response_model=HeaderMetaTagRecord,
    summary="Update meta tag",
    description="Updates the record identified by ``meta_tag_id``.",
)
async def update_meta_tags(body: UpdateHeaderMetaTagRequest) -> dict[str, Any]:
    if body.header_meta_tag is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="header_meta_tag is required to update a record.",
        )

    col = get_header_meta_tags_collection()
    doc = await col.find_one_and_update(
        {"meta_tag_id": body.meta_tag_id},
        {
            "$set": {
                "header_meta_tag": body.header_meta_tag,
                "updated_at": _now(),
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Meta tag not found.",
        )
    return _doc_public(doc)


@router.delete(
    "/{meta_tag_id}",
    summary="Hard delete meta tag",
    description="Permanently removes the meta tag document from MongoDB.",
)
async def hard_delete_meta_tag(
    meta_tag_id: str = Path(..., min_length=1, max_length=64, description="Meta tag id to delete."),
) -> dict[str, Any]:
    col = get_header_meta_tags_collection()
    doc = await col.find_one({"meta_tag_id": meta_tag_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Meta tag not found.",
        )
    result = await col.delete_one({"meta_tag_id": meta_tag_id})
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Meta tag not found.",
        )
    return {
        "success": True,
        "message": "Meta tag deleted.",
        "meta_tag_id": meta_tag_id,
        "deleted": _doc_public(doc),
    }


# ── Public read ──────────────────────────────────────────────────────────────


@public_router.get(
    "/get-all-header-meta-tags",
    response_model=HeaderMetaTagListResponse,
    summary="Get all header meta tags (public)",
    description=(
        "Public endpoint — returns every stored header meta tag, newest first. "
        "No authentication required; intended for the website front-end to inject "
        "header meta tags into rendered pages."
    ),
)
async def public_get_all_header_meta_tags() -> dict[str, Any]:
    col = get_header_meta_tags_collection()
    cur = col.find({}).sort("created_at", -1)
    items = [_doc_public(d) async for d in cur]
    return {"total": len(items), "meta_tags": items}
