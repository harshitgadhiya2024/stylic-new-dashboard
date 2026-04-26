"""
Blogs: public read (published only) and admin CRUD under ``/api/v1/admin/``.

Admin write routes require header ``X-Admin-API-Key`` (see :envvar:`ADMIN_API_KEY`).
Public GETs require no authentication.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pymongo import ReturnDocument

from app.database import get_blogs_collection
from app.dependencies import verify_admin_api_key
from app.models.blog import (
    BlogRecord,
    CreateBlogRequest,
    UpdateBlogRequest,
    UpdateBlogStatusRequest,
)

# ── Public (no API key) ─────────────────────────────────────────────────────

public_router = APIRouter(
    prefix="/api/v1/blogs",
    tags=["Blogs (public)"],
)

# ── Admin (X-Admin-API-Key) — prefix includes ``admin`` ─────────────────────

admin_router = APIRouter(
    prefix="/api/v1/admin/blogs",
    dependencies=[Depends(verify_admin_api_key)],
    tags=["Admin — Blogs"],
)

_ALLOWED_STATUS = frozenset({"draft", "published", "archived"})


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _doc_to_record(doc: dict) -> dict[str, Any]:
    out = {k: v for k, v in doc.items() if k != "_id"}
    return out


@admin_router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=BlogRecord,
    summary="Create blog (admin)",
    description="Creates a blog post with ``status`` default ``draft``.",
)
async def admin_create_blog(body: CreateBlogRequest) -> dict[str, Any]:
    now = _now()
    blog_id = str(uuid.uuid4())
    doc: dict[str, Any] = {
        "blog_id": blog_id,
        "blog_name": body.blog_name,
        "blog_tagline": body.blog_tagline,
        "blog_type": body.blog_type,
        "blog_hero_image": body.blog_hero_image,
        "author_name": body.author_name,
        "blog_html_content": body.blog_html_content,
        "blog_post_date_and_time": None,
        "status": "draft",
        "created_at": now,
        "updated_at": now,
    }
    col = get_blogs_collection()
    await col.insert_one(doc)
    return _doc_to_record(doc)


@admin_router.patch(
    "",
    response_model=BlogRecord,
    summary="Update blog (admin)",
    description="Partial update; include ``blog_id`` and any fields to change.",
)
async def admin_update_blog(body: UpdateBlogRequest) -> dict[str, Any]:
    col = get_blogs_collection()
    existing = await col.find_one({"blog_id": body.blog_id})
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Blog not found for this blog_id.",
        )

    set_fields: dict[str, Any] = {"updated_at": _now()}

    if body.blog_name is not None:
        set_fields["blog_name"] = body.blog_name
    if body.blog_tagline is not None:
        set_fields["blog_tagline"] = body.blog_tagline
    if body.blog_type is not None:
        set_fields["blog_type"] = body.blog_type
    if body.blog_hero_image is not None:
        set_fields["blog_hero_image"] = body.blog_hero_image
    if body.author_name is not None:
        set_fields["author_name"] = body.author_name
    if body.blog_html_content is not None:
        set_fields["blog_html_content"] = body.blog_html_content

    if body.status is not None:
        set_fields["status"] = body.status
        if body.status == "published":
            set_fields["blog_post_date_and_time"] = _now()

    await col.update_one(
        {"blog_id": body.blog_id},
        {"$set": set_fields},
    )
    fresh = await col.find_one({"blog_id": body.blog_id})
    return _doc_to_record(fresh)  # type: ignore[arg-type]


@admin_router.patch(
    "/status",
    response_model=BlogRecord,
    summary="Update blog status (admin)",
    description="When ``updated_status`` is ``published``, sets ``blog_post_date_and_time`` to now.",
)
async def admin_update_blog_status(body: UpdateBlogStatusRequest) -> dict[str, Any]:
    if body.updated_status not in _ALLOWED_STATUS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid updated_status. Allowed: {sorted(_ALLOWED_STATUS)}",
        )
    col = get_blogs_collection()
    existing = await col.find_one({"blog_id": body.blog_id})
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Blog not found for this blog_id.",
        )

    now = _now()
    set_fields: dict[str, Any] = {
        "status": body.updated_status,
        "updated_at": now,
    }
    if body.updated_status == "published":
        set_fields["blog_post_date_and_time"] = now

    await col.update_one({"blog_id": body.blog_id}, {"$set": set_fields})
    fresh = await col.find_one({"blog_id": body.blog_id})
    return _doc_to_record(fresh)  # type: ignore[arg-type]


@admin_router.delete(
    "/{blog_id}",
    response_model=BlogRecord,
    summary="Archive blog (admin)",
    description="Soft delete: sets ``status`` to ``archived`` and refreshes ``updated_at``.",
)
async def admin_archive_blog(
    blog_id: str = Path(..., min_length=1, max_length=64, description="Blog id (UUID)"),
) -> dict[str, Any]:
    col = get_blogs_collection()
    now = _now()
    res = await col.find_one_and_update(
        {"blog_id": blog_id},
        {
            "$set": {
                "status": "archived",
                "updated_at": now,
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Blog not found for this blog_id.",
        )
    return _doc_to_record(res)


# ── Public read ──────────────────────────────────────────────────────────────


@public_router.get(
    "",
    response_model=list[BlogRecord],
    summary="List published blogs (public)",
    description="Returns only posts with ``status == published``.",
)
async def public_list_published_blogs(
    skip: int = Query(0, ge=0, le=1_000_000),
    limit: int = Query(30, ge=1, le=100),
) -> list[dict[str, Any]]:
    col = get_blogs_collection()
    cur = (
        col.find({"status": "published"})
        .sort(
            [
                ("blog_post_date_and_time", -1),
                ("created_at", -1),
            ]
        )
        .skip(skip)
        .limit(limit)
    )
    out: list[dict[str, Any]] = []
    async for doc in cur:
        out.append(_doc_to_record(doc))
    return out


@public_router.get(
    "/{blog_id}",
    response_model=BlogRecord,
    summary="Get one blog (public)",
    description="Full row for a blog that is **published**; other statuses return 404.",
)
async def public_get_blog(
    blog_id: str = Path(..., min_length=1, max_length=64),
) -> dict[str, Any]:
    col = get_blogs_collection()
    doc = await col.find_one({"blog_id": blog_id, "status": "published"})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Blog not found or not published.",
        )
    return _doc_to_record(doc)
