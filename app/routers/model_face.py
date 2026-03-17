import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import List, AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.database import get_model_faces_collection
from app.dependencies import get_current_user
from app.models.model_face import (
    CreateModelFaceRequest,
    CreateModelFaceWithAIRequest,
    ModelFaceSchema,
)
from app.services.ai_face_service import generate_and_upload_face_stream
from app.services.face_to_model_service import generate_model_face_from_reference_stream
from app.services.credit_service import check_sufficient_credits, deduct_credits_and_record

router = APIRouter(prefix="/api/v1/model-faces", tags=["Model Faces"])

_SSE_HEADERS = {
    "Cache-Control":    "no-cache",
    "X-Accel-Buffering": "no",
    "Connection":       "keep-alive",
}


def _clean_face(doc: dict) -> dict:
    doc = dict(doc)
    doc.pop("_id", None)
    return doc


def _sse(event: str, data: dict) -> str:
    """Format a single Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _run_sync_generator(gen_fn, *args, **kwargs) -> asyncio.Queue:
    """
    Run a blocking synchronous generator in a thread-pool executor and
    forward each yielded item into an asyncio.Queue so the async caller
    can await items without blocking the event loop.

    Sentinel values pushed to the queue:
      ("ok",  item)   — a normal yielded value
      ("err", exc)    — an exception raised inside the generator
      ("end", None)   — generator exhausted
    """
    loop  = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _worker():
        try:
            for item in gen_fn(*args, **kwargs):
                loop.call_soon_threadsafe(queue.put_nowait, ("ok", item))
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("err", exc))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("end", None))

    loop.run_in_executor(None, _worker)
    return queue


@router.get(
    "/",
    summary="Get All Model Faces",
    description=(
        "Returns paginated model faces for the authenticated user. "
        "Fetches all default faces (is_default=True, any user) and all faces created by the user "
        "(any is_default value), merges them (deduplicating by model_id), then sorts so that "
        "favorites appear first followed by the rest ordered by creation date (newest first). "
        "Use `page` and `limit` query params to control pagination."
    ),
)
def get_model_faces(
    page:  int = Query(default=1,  ge=1,        description="Page number (1-based)"),
    limit: int = Query(default=10, ge=1, le=100, description="Number of items per page"),
    current_user: dict = Depends(get_current_user),
):
    col     = get_model_faces_collection()
    user_id = current_user["user_id"]

    # 1 — all default faces (no user_id condition)
    default_docs = list(col.find({"is_default": True}))

    # 2 — all faces created by this user (no is_default condition)
    user_docs = list(col.find({"user_id": user_id}))

    # 3 — merge, deduplicate by model_id (user docs take precedence)
    seen: dict = {}
    for doc in default_docs:
        seen[doc["model_id"]] = doc
    for doc in user_docs:
        seen[doc["model_id"]] = doc

    merged = list(seen.values())

    # 4 — sort: favorites first, then newest first
    merged.sort(key=lambda d: (not d.get("is_favorite", False), d.get("created_at") or ""), )

    # 5 — paginate in Python
    total  = len(merged)
    skip   = (page - 1) * limit
    paged  = merged[skip: skip + limit]

    return {
        "page":        page,
        "limit":       limit,
        "total":       total,
        "total_pages": (total + limit - 1) // limit if total else 1,
        "data":        [_clean_face(doc) for doc in paged],
    }


@router.post(
    "/",
    summary="Upload / Create a Model Face (Streaming)",
    description=(
        "Accepts a reference face photo URL. Streams real-time progress via SSE. "
        "Validates the image for a real human face (Gemini), generates a professional "
        "model portrait replicating the reference face (SeedDream), uploads to S3, "
        "saves to DB, and deducts 2.5 credits in the background. "
        "Secured — user_id is taken from the auth token. "
        "Response is `text/event-stream`. Final event `done` contains the full model face record."
    ),
)
async def create_model_face(
    body: CreateModelFaceRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    check_sufficient_credits(current_user)

    async def event_stream() -> AsyncGenerator[str, None]:
        generated_face_url: str | None = None

        queue = await _run_sync_generator(
            generate_model_face_from_reference_stream,
            body.reference_face_url,
            body.model_category,
        )

        while True:
            kind, payload = await queue.get()

            if kind == "end":
                break

            if kind == "err":
                exc = payload
                msg = exc.detail if isinstance(exc, HTTPException) else str(exc)
                yield _sse("error", {"step": "error", "message": msg})
                return

            step, message, face_url = payload

            if step == "done":
                generated_face_url = face_url
                yield _sse("storing_db", {"step": "storing_db", "message": "Storing in database"})

                now = datetime.now(timezone.utc)
                doc = {
                    "model_id":            str(uuid.uuid4()),
                    "user_id":             current_user["user_id"],
                    "model_name":          body.model_name,
                    "model_category":      body.model_category,
                    "model_configuration": {},
                    "tags":                body.tags,
                    "notes":               body.notes,
                    "model_used_count":    0,
                    "face_url":            generated_face_url,
                    "reference_face_url":  body.reference_face_url,
                    "is_default":          False,
                    "is_active":           True,
                    "is_favorite":         False,
                    "created_at":          now.isoformat(),
                    "updated_at":          now.isoformat(),
                }
                col = get_model_faces_collection()
                col.insert_one({**doc, "created_at": now, "updated_at": now})
                yield _sse("done", {"step": "done", "message": "Face generation complete", "data": doc})
            else:
                yield _sse(step, {"step": step, "message": message})

        if generated_face_url:
            background_tasks.add_task(
                deduct_credits_and_record,
                user=current_user,
                feature_name="face_generation",
                generated_face_url=generated_face_url,
                notes=body.notes or "",
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post(
    "/generate-with-ai",
    summary="Create Model Face Using AI (Streaming)",
    description=(
        "Streams real-time progress via SSE. Generates a realistic portrait via Gemini AI "
        "using the provided face configurations (all optional — unset fields use "
        "category-appropriate defaults), uploads to S3, saves to DB, and deducts 2.5 credits "
        "in the background. beard_length and beard_color only apply when model_category is "
        "'adult_male'. Response is `text/event-stream`. Final event `done` contains the full "
        "model face record."
    ),
)
async def create_model_face_with_ai(
    body: CreateModelFaceWithAIRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    check_sufficient_credits(current_user)

    overrides = {}
    if body.face_configurations:
        overrides = {
            k: v
            for k, v in body.face_configurations.model_dump().items()
            if v is not None
        }

    async def event_stream() -> AsyncGenerator[str, None]:
        generated_face_url: str | None = None
        final_config: dict | None = None

        queue = await _run_sync_generator(
            generate_and_upload_face_stream,
            body.model_category,
            overrides,
        )

        while True:
            kind, payload = await queue.get()

            if kind == "end":
                break

            if kind == "err":
                exc = payload
                msg = exc.detail if isinstance(exc, HTTPException) else str(exc)
                yield _sse("error", {"step": "error", "message": msg})
                return

            step, message, face_url, config = payload

            if step == "done":
                generated_face_url = face_url
                final_config       = config
                yield _sse("storing_db", {"step": "storing_db", "message": "Storing in database"})

                now = datetime.now(timezone.utc)
                doc = {
                    "model_id":            str(uuid.uuid4()),
                    "user_id":             current_user["user_id"],
                    "model_name":          body.model_name,
                    "model_category":      body.model_category,
                    "model_configuration": final_config,
                    "tags":                body.tags or [],
                    "notes":               body.notes or "",
                    "model_used_count":    0,
                    "face_url":            generated_face_url,
                    "reference_face_url":  None,
                    "is_default":          False,
                    "is_active":           True,
                    "is_favorite":         False,
                    "created_at":          now.isoformat(),
                    "updated_at":          now.isoformat(),
                }
                col = get_model_faces_collection()
                col.insert_one({**doc, "created_at": now, "updated_at": now})
                yield _sse("done", {"step": "done", "message": "Face generation complete", "data": doc})
            else:
                yield _sse(step, {"step": step, "message": message})

        if generated_face_url:
            background_tasks.add_task(
                deduct_credits_and_record,
                user=current_user,
                feature_name="face_generation",
                generated_face_url=generated_face_url,
                notes=body.notes or "",
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.patch(
    "/{model_id}/toggle-favorite",
    response_model=ModelFaceSchema,
    summary="Toggle Favorite",
    description="Switch is_favorite between true and false for a model face. Secured — only the owner can toggle.",
)
def toggle_favorite(
    model_id: str,
    current_user: dict = Depends(get_current_user),
):
    col = get_model_faces_collection()

    doc = col.find_one({"model_id": model_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model face not found.",
        )

    if doc.get("user_id") != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to update this model face.",
        )

    new_value = not doc.get("is_favorite", False)
    now = datetime.now(timezone.utc)

    col.update_one(
        {"model_id": model_id},
        {"$set": {"is_favorite": new_value, "updated_at": now}},
    )

    doc["is_favorite"] = new_value
    doc["updated_at"]  = now
    return _clean_face(doc)


@router.delete(
    "/{model_id}",
    summary="Delete Model Face",
    description="Soft-delete a model face by setting is_active=False. Secured — only the owner can delete.",
)
def delete_model_face(
    model_id: str,
    current_user: dict = Depends(get_current_user),
):
    col = get_model_faces_collection()

    doc = col.find_one({"model_id": model_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model face not found.",
        )

    if doc.get("user_id") != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this model face.",
        )

    if not doc.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model face is already deleted.",
        )

    col.update_one(
        {"model_id": model_id},
        {"$set": {"is_active": False, "updated_at": datetime.now(timezone.utc)}},
    )

    return {"message": "Model face deleted successfully.", "model_id": model_id}
