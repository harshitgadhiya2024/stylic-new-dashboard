import json
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.database import get_backgrounds_collection
from app.dependencies import get_current_user
from app.models.background import (
    BackgroundSchema,
    CreateBackgroundRequest,
    CreateBackgroundWithAIRequest,
    DeleteBackgroundsRequest,
)
from app.services.background_service import generate_background_stream, generate_background_with_ai_stream
from app.services.credit_service import check_sufficient_credits, deduct_credits_and_record

router = APIRouter(prefix="/api/v1/backgrounds", tags=["Backgrounds"])

# Stored in Mongo as lowercase; query accepts labels (any case / spaces ok).
_ALLOWED_BACKGROUND_TYPE_DB_VALUES = frozenset({"indoor", "outdoor", "studio"})

_SSE_HEADERS = {
    "Cache-Control":     "no-cache",
    "X-Accel-Buffering": "no",
    "Connection":        "keep-alive",
}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _clean_bg(doc: dict) -> dict:
    doc = dict(doc)
    doc.pop("_id", None)
    doc.setdefault("is_favorite", False)
    return doc


def _normalize_background_type_filter(raw: Optional[str]) -> Optional[str]:
    """
    Map frontend labels (e.g. \"Indoor\", \"Outdoor\") or snake_case to DB value.
    Returns None when ``raw`` is empty (no filter). Raises HTTPException 422 if unknown.
    """
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    key = s.lower().replace(" ", "_").replace("-", "_")
    while "__" in key:
        key = key.replace("__", "_")
    if key in _ALLOWED_BACKGROUND_TYPE_DB_VALUES:
        return key
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=(
            f"Invalid background_type {raw!r}. "
            "Use one of: Indoor, Outdoor, Studio "
            "(or indoor, outdoor, studio)."
        ),
    )


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Create Background (Streaming)",
    description=(
        "Accepts a background image URL, name, and background_type (Indoor | Outdoor | Studio; stored lowercase). "
        "Streams real-time progress via SSE. "
        "Validates the URL, generates an enhanced background via SeedDream (16:9), "
        "uploads to S3, saves to DB, and deducts 2.5 credits in the background. "
        "Secured — user_id is taken from the auth token. "
        "Response is `text/event-stream`. Final event `done` contains the full background record. "
        "Expected duration: 20-30 seconds."
    ),
)
async def create_background(
    body: CreateBackgroundRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    check_sufficient_credits(current_user)

    async def event_stream() -> AsyncGenerator[str, None]:
        generated_bg_url: str | None = None

        try:
            async for step, message, result_url in generate_background_stream(
                body.background_url,
                body.background_name,
            ):
                if step == "done":
                    generated_bg_url = result_url
                    yield _sse("storing_db", {"step": "storing_db", "message": "Storing in database"})

                    now = datetime.now(timezone.utc)
                    doc = {
                        "background_id":   str(uuid.uuid4()),
                        "user_id":         current_user["user_id"],
                        "background_type": body.background_type,
                        "background_name": body.background_name,
                        "background_url":  generated_bg_url,
                        "count":           0,
                        "tags":            body.tags or [],
                        "notes":           body.notes or "",
                        "is_default":      False,
                        "is_active":       True,
                        "is_favorite":     False,
                        "created_at":      now.isoformat(),
                        "updated_at":      now.isoformat(),
                    }

                    col = get_backgrounds_collection()
                    await col.insert_one({**doc, "created_at": now, "updated_at": now})

                    yield _sse("done", {"step": "done", "message": "Background generation complete", "data": doc})
                else:
                    yield _sse(step, {"step": step, "message": message})
        except HTTPException as exc:
            yield _sse("error", {"step": "error", "message": exc.detail})
            return
        except Exception as exc:
            yield _sse("error", {"step": "error", "message": str(exc)})
            return

        if generated_bg_url:
            background_tasks.add_task(
                deduct_credits_and_record,
                user=current_user,
                feature_name="background_generation",
                generated_face_url=generated_bg_url,
                notes=body.notes or "",
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post(
    "/generate-with-ai",
    status_code=status.HTTP_201_CREATED,
    summary="Create Background Using AI (Streaming)",
    description=(
        "Generates a professional fashion photoshoot background from a text description "
        "using Gemini AI (9:16 aspect ratio). Requires background_type: Indoor, Outdoor, or Studio (stored lowercase). "
        "Streams real-time progress via SSE. "
        "Uploads the result to S3, saves to DB, and deducts 2.5 credits in the background. "
        "Secured — user_id is taken from the auth token. "
        "Response is `text/event-stream`. Final event `done` contains the full background record. "
        "Expected duration: 20-30 seconds."
    ),
)
async def create_background_with_ai(
    body: CreateBackgroundWithAIRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    check_sufficient_credits(current_user)

    async def event_stream() -> AsyncGenerator[str, None]:
        generated_bg_url: str | None = None

        try:
            async for step, message, result_url in generate_background_with_ai_stream(
                body.background_name,
                body.background_configuration,
            ):
                if step == "done":
                    generated_bg_url = result_url
                    yield _sse("storing_db", {"step": "storing_db", "message": "Storing in database"})

                    now = datetime.now(timezone.utc)
                    doc = {
                        "background_id":   str(uuid.uuid4()),
                        "user_id":         current_user["user_id"],
                        "background_type": body.background_type,
                        "background_name": body.background_name,
                        "background_url":  generated_bg_url,
                        "count":           0,
                        "tags":            body.tags or [],
                        "notes":           body.notes or "",
                        "is_default":      False,
                        "is_active":       True,
                        "is_favorite":     False,
                        "created_at":      now.isoformat(),
                        "updated_at":      now.isoformat(),
                    }

                    col = get_backgrounds_collection()
                    await col.insert_one({**doc, "created_at": now, "updated_at": now})

                    yield _sse("done", {"step": "done", "message": "Background generation complete", "data": doc})
                else:
                    yield _sse(step, {"step": step, "message": message})
        except HTTPException as exc:
            yield _sse("error", {"step": "error", "message": exc.detail})
            return
        except Exception as exc:
            yield _sse("error", {"step": "error", "message": str(exc)})
            return

        if generated_bg_url:
            background_tasks.add_task(
                deduct_credits_and_record,
                user=current_user,
                feature_name="background_generation_ai",
                generated_face_url=generated_bg_url,
                notes=body.notes or "",
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.get(
    "/",
    summary="Get Backgrounds",
    description=(
        "Returns a paginated list of backgrounds based on `type`. "
        "`type=default` — returns all platform default backgrounds (is_default=True). "
        "`type=custom` — returns backgrounds created by the current user (user_id match, is_active=True), "
        "sorted with favorites first, then newest first. "
        "Optional filters: `is_favorite` (true/false), `background_type` "
        "(Indoor, Outdoor, Studio — matched to stored lowercase). Omit either for no filter. "
        "Use `page` and `limit` to control pagination."
    ),
)
async def get_backgrounds(
    type:  Literal["default", "custom"] = Query(..., description="'default' for platform backgrounds, 'custom' for user-created backgrounds"),
    page:  int = Query(default=1,  ge=1,        description="Page number (1-based)"),
    limit: int = Query(default=10, ge=1, le=100, description="Number of items per page"),
    is_favorite: Optional[bool] = Query(
        default=None,
        description="When set, return only favorites (true) or only non-favorites (false). Omit for no filter.",
    ),
    background_type: Optional[str] = Query(
        default=None,
        description=(
            "Optional. One of: Indoor, Outdoor, Studio (any case / spaces), "
            "or indoor, outdoor, studio."
        ),
    ),
    current_user: dict = Depends(get_current_user),
):
    col = get_backgrounds_collection()

    bg_type = _normalize_background_type_filter(background_type)

    if type == "default":
        query: dict = {"is_default": True}
    else:
        query = {"user_id": current_user["user_id"], "is_active": True}

    if bg_type is not None:
        query["background_type"] = bg_type
    if is_favorite is not None:
        query["is_favorite"] = is_favorite

    if type == "custom":
        cursor = col.find(query).sort([("is_favorite", -1), ("created_at", -1)])
    else:
        cursor = col.find(query)
    docs = await cursor.to_list(length=None)

    total       = len(docs)
    skip        = (page - 1) * limit
    paged       = docs[skip: skip + limit]
    total_pages = (total + limit - 1) // limit if total else 1

    return {
        "type":        type,
        "page":        page,
        "limit":       limit,
        "total":       total,
        "total_pages": total_pages,
        "filters":     {
            "is_favorite":     is_favorite,
            "background_type": bg_type,
        },
        "data":        [_clean_bg(doc) for doc in paged],
    }


@router.patch(
    "/{background_id}/toggle-favorite",
    response_model=BackgroundSchema,
    summary="Toggle Favorite",
    description=(
        "Switch is_favorite between true and false for a background. "
        "Secured — only the owner can toggle. Default platform backgrounds cannot be favorited."
    ),
)
async def toggle_background_favorite(
    background_id: str,
    current_user: dict = Depends(get_current_user),
):
    col = get_backgrounds_collection()

    doc = await col.find_one({"background_id": background_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Background not found.",
        )

    if doc.get("is_default", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Default backgrounds cannot be marked as favorite.",
        )

    if doc.get("user_id") != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to update this background.",
        )

    new_value = not doc.get("is_favorite", False)
    now = datetime.now(timezone.utc)

    await col.update_one(
        {"background_id": background_id},
        {"$set": {"is_favorite": new_value, "updated_at": now}},
    )

    doc["is_favorite"] = new_value
    doc["updated_at"]  = now
    return _clean_bg(doc)


@router.delete(
    "/bulk-delete",
    summary="Delete Multiple Backgrounds",
    description=(
        "Soft-deletes multiple backgrounds by setting is_active=False. "
        "Only backgrounds owned by the current user (user_id match) and not default (is_default=False) "
        "are deleted. Default backgrounds are silently skipped. "
        "Returns counts of deleted and skipped backgrounds."
    ),
)
async def delete_backgrounds_bulk(
    body: DeleteBackgroundsRequest,
    current_user: dict = Depends(get_current_user),
):
    if not body.background_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="background_ids must not be empty.",
        )

    col     = get_backgrounds_collection()
    user_id = current_user["user_id"]

    result = await col.update_many(
        {
            "background_id": {"$in": body.background_ids},
            "user_id":       user_id,
            "is_default":    False,
            "is_active":     True,
        },
        {"$set": {"is_active": False, "updated_at": datetime.now(timezone.utc)}},
    )

    deleted_count = result.modified_count
    skipped_count = len(body.background_ids) - deleted_count

    return {
        "message":       f"{deleted_count} background(s) deleted successfully.",
        "deleted_count": deleted_count,
        "skipped_count": skipped_count,
    }


@router.delete(
    "/{background_id}",
    summary="Delete Background",
    description="Soft-delete a background by setting is_active=False. Secured — only the owner can delete.",
)
async def delete_background(
    background_id: str,
    current_user: dict = Depends(get_current_user),
):
    col = get_backgrounds_collection()

    doc = await col.find_one({"background_id": background_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Background not found.",
        )

    if doc.get("is_default", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Default backgrounds cannot be deleted.",
        )

    if doc.get("user_id") != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this background.",
        )

    if not doc.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Background is already deleted.",
        )

    await col.update_one(
        {"background_id": background_id},
        {"$set": {"is_active": False, "updated_at": datetime.now(timezone.utc)}},
    )

    return {"message": "Background deleted successfully.", "background_id": background_id}
