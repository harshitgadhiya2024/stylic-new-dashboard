import json
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.database import get_backgrounds_collection
from app.dependencies import get_current_user
from app.models.background import CreateBackgroundRequest, CreateBackgroundWithAIRequest, DeleteBackgroundsRequest
from app.services.background_service import generate_background_stream, generate_background_with_ai_stream
from app.services.credit_service import check_sufficient_credits, deduct_credits_and_record

router = APIRouter(prefix="/api/v1/backgrounds", tags=["Backgrounds"])

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
    return doc


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Create Background (Streaming)",
    description=(
        "Accepts a background image URL and name. Streams real-time progress via SSE. "
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
                        "background_type": body.background_name,
                        "background_name": body.background_name,
                        "background_url":  generated_bg_url,
                        "count":           0,
                        "tags":            body.tags or [],
                        "notes":           body.notes or "",
                        "is_default":      False,
                        "is_active":       True,
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
        "using Gemini AI (9:16 aspect ratio). Streams real-time progress via SSE. "
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
                        "background_type": body.background_name,
                        "background_name": body.background_name,
                        "background_url":  generated_bg_url,
                        "count":           0,
                        "tags":            body.tags or [],
                        "notes":           body.notes or "",
                        "is_default":      False,
                        "is_active":       True,
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
        "sorted by creation date (newest first). "
        "Use `page` and `limit` to control pagination."
    ),
)
async def get_backgrounds(
    type:  Literal["default", "custom"] = Query(..., description="'default' for platform backgrounds, 'custom' for user-created backgrounds"),
    page:  int = Query(default=1,  ge=1,        description="Page number (1-based)"),
    limit: int = Query(default=10, ge=1, le=100, description="Number of items per page"),
    current_user: dict = Depends(get_current_user),
):
    col = get_backgrounds_collection()

    if type == "default":
        docs = await col.find({"is_default": True}).to_list(length=None)
    else:
        docs = await col.find({"user_id": current_user["user_id"], "is_active": True}).to_list(length=None)
        docs.sort(key=lambda d: d.get("created_at") or "", reverse=True)

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
        "data":        [_clean_bg(doc) for doc in paged],
    }


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

