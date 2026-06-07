"""
Blogs: public read (published only) and admin CRUD under ``/api/v1/admin/``.

Admin write routes require admin JWT Bearer token.
Public GETs require no authentication.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from pymongo import ReturnDocument

from app.database import get_blogs_collection
from app.dependencies import require_admin_roles
from app.models.blog import (
    AdminBlogListResponse,
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

# ── Admin (Bearer admin JWT) — prefix includes ``admin`` ────────────────────

admin_router = APIRouter(
    prefix="/api/v1/admin/blogs",
    dependencies=[Depends(require_admin_roles("superadmin", "admin", "blogger"))],
    tags=["Admin — Blogs"],
)

_ALLOWED_STATUS = frozenset({"draft", "published", "archived"})


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _doc_to_record(doc: dict) -> dict[str, Any]:
    out = {k: v for k, v in doc.items() if k != "_id"}
    return out


def _faqs_to_doc(faqs: Any) -> list[dict[str, Any]] | None:
    if faqs is None:
        return None
    return [f.model_dump() if hasattr(f, "model_dump") else dict(f) for f in faqs]


def _toc_to_doc(toc: Any) -> list[dict[str, Any]] | None:
    if toc is None:
        return None
    return [t.model_dump() if hasattr(t, "model_dump") else dict(t) for t in toc]


def _date_to_doc(d: Any) -> str | None:
    return d.isoformat() if d is not None else None


def _time_to_doc(t: Any) -> str | None:
    return t.isoformat() if t is not None else None


async def _find_blog_by_identifier(
    col: Any,
    identifier: str,
    *,
    status: str | None = None,
) -> dict[str, Any] | None:
    """Resolve a blog by ``blog_id`` (UUID) or ``blog_url`` (slug/path)."""
    key = identifier.strip()
    if not key:
        return None

    base: dict[str, Any] = {}
    if status is not None:
        base["status"] = status

    doc = await col.find_one({**base, "blog_id": key})
    if doc:
        return doc
    return await col.find_one({**base, "blog_url": key})


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

        # SEO meta
        "meta_title": body.meta_title,
        "meta_description": body.meta_description,
        "meta_keywords": body.meta_keywords,

        # Open Graph
        "og_title": body.og_title,
        "og_description": body.og_description,
        "og_image": body.og_image,
        "og_image_alt": body.og_image_alt,
        "og_url": body.og_url,

        # Twitter
        "twitter_title": body.twitter_title,
        "twitter_description": body.twitter_description,
        "twitter_image": body.twitter_image,
        "twitter_image_alt": body.twitter_image_alt,

        # Hero image alt
        "blog_hero_image_alt": body.blog_hero_image_alt,

        # URLs
        "blog_url": body.blog_url,
        "canonical_url": body.canonical_url,

        # Schema / TOC / FAQs
        "schema_org": body.schema_org,
        "table_of_contents": _toc_to_doc(body.table_of_contents),
        "faqs": _faqs_to_doc(body.faqs),

        # Date/time (stored as ISO strings)
        "blog_post_date": _date_to_doc(body.blog_post_date),
        "blog_post_time": _time_to_doc(body.blog_post_time),
        "blog_post_date_and_time": None,

        "status": "draft",
        "created_at": now,
        "updated_at": now,
    }
    col = get_blogs_collection()
    await col.insert_one(doc)
    return _doc_to_record(doc)


@admin_router.get(
    "",
    response_model=AdminBlogListResponse,
    summary="List blogs (admin)",
    description=(
        "Returns blogs for admin panel with page-based pagination. "
        "Optional `status` filter supports draft, published, or archived."
    ),
)
async def admin_list_blogs(
    status_filter: str | None = Query(
        default=None,
        alias="status",
        description="Optional status filter: draft, published, archived.",
    ),
    page: int = Query(1, ge=1, description="1-based page number"),
    limit: int = Query(30, ge=1, le=100, description="Items per page"),
) -> dict[str, Any]:
    query: dict[str, Any] = {}
    if status_filter is not None:
        normalized_status = status_filter.strip().lower()
        if normalized_status not in _ALLOWED_STATUS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid status filter. Allowed: {sorted(_ALLOWED_STATUS)}",
            )
        query["status"] = normalized_status

    col = get_blogs_collection()
    total = await col.count_documents(query)
    total_pages = max(1, (total + limit - 1) // limit) if total else 1
    skip = (page - 1) * limit
    cur = (
        col.find(query)
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
    return {
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": total_pages,
        "data": out,
    }


@admin_router.get(
    "/{blog_identifier:path}",
    response_model=BlogRecord,
    summary="Get one blog (admin)",
    description=(
        "Returns one blog for admin panel by `blog_id` (UUID) or `blog_url` "
        "(slug/path, any status)."
    ),
)
async def admin_get_blog(
    blog_identifier: str = Path(
        ...,
        min_length=1,
        max_length=2000,
        description="Blog ID (UUID) or blog_url slug/path.",
    ),
) -> dict[str, Any]:
    col = get_blogs_collection()
    doc = await _find_blog_by_identifier(col, blog_identifier)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Blog not found for this blog_id or blog_url.",
        )
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

    # SEO meta
    if body.meta_title is not None:
        set_fields["meta_title"] = body.meta_title
    if body.meta_description is not None:
        set_fields["meta_description"] = body.meta_description
    if body.meta_keywords is not None:
        set_fields["meta_keywords"] = body.meta_keywords

    # Open Graph
    if body.og_title is not None:
        set_fields["og_title"] = body.og_title
    if body.og_description is not None:
        set_fields["og_description"] = body.og_description
    if body.og_image is not None:
        set_fields["og_image"] = body.og_image
    if body.og_image_alt is not None:
        set_fields["og_image_alt"] = body.og_image_alt
    if body.og_url is not None:
        set_fields["og_url"] = body.og_url

    # Twitter
    if body.twitter_title is not None:
        set_fields["twitter_title"] = body.twitter_title
    if body.twitter_description is not None:
        set_fields["twitter_description"] = body.twitter_description
    if body.twitter_image is not None:
        set_fields["twitter_image"] = body.twitter_image
    if body.twitter_image_alt is not None:
        set_fields["twitter_image_alt"] = body.twitter_image_alt

    # Hero image alt
    if body.blog_hero_image_alt is not None:
        set_fields["blog_hero_image_alt"] = body.blog_hero_image_alt

    # URLs
    if body.blog_url is not None:
        set_fields["blog_url"] = body.blog_url
    if body.canonical_url is not None:
        set_fields["canonical_url"] = body.canonical_url

    # Schema / TOC / FAQs
    if body.schema_org is not None:
        set_fields["schema_org"] = body.schema_org
    if body.table_of_contents is not None:
        set_fields["table_of_contents"] = _toc_to_doc(body.table_of_contents)
    if body.faqs is not None:
        set_fields["faqs"] = _faqs_to_doc(body.faqs)

    # Date/time
    if body.blog_post_date is not None:
        set_fields["blog_post_date"] = _date_to_doc(body.blog_post_date)
    if body.blog_post_time is not None:
        set_fields["blog_post_time"] = _time_to_doc(body.blog_post_time)

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
    "/{blog_identifier:path}",
    response_model=BlogRecord,
    summary="Get one blog (public)",
    description=(
        "Full row for a **published** blog, looked up by `blog_id` (UUID) or "
        "`blog_url` (slug/path). Other statuses return 404."
    ),
)
async def public_get_blog(
    blog_identifier: str = Path(
        ...,
        min_length=1,
        max_length=2000,
        description="Blog ID (UUID) or blog_url slug/path.",
    ),
) -> dict[str, Any]:
    col = get_blogs_collection()
    doc = await _find_blog_by_identifier(
        col, blog_identifier, status="published"
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Blog not found or not published.",
        )
    return _doc_to_record(doc)
