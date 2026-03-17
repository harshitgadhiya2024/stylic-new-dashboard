import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.database import get_backgrounds_collection
from app.dependencies import get_current_user
from app.models.background import CreateBackgroundRequest, CreateBackgroundWithAIRequest
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


async def _run_sync_generator(gen_fn, *args, **kwargs) -> asyncio.Queue:
    """
    Run a blocking synchronous generator in a thread-pool executor and
    forward each yielded item into an asyncio.Queue.

    Sentinel values:
      ("ok",  item)  — normal yielded value
      ("err", exc)   — exception raised inside the generator
      ("end", None)  — generator exhausted
    """
    loop:  asyncio.AbstractEventLoop = asyncio.get_event_loop()
    queue: asyncio.Queue             = asyncio.Queue()

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

        queue = await _run_sync_generator(
            generate_background_stream,
            body.background_url,
            body.background_name,
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

            step, message, result_url = payload

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
                col.insert_one({**doc, "created_at": now, "updated_at": now})

                yield _sse("done", {"step": "done", "message": "Background generation complete", "data": doc})
            else:
                yield _sse(step, {"step": step, "message": message})

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

        queue = await _run_sync_generator(
            generate_background_with_ai_stream,
            body.background_name,
            body.background_configuration,
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

            step, message, result_url = payload

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
                col.insert_one({**doc, "created_at": now, "updated_at": now})

                yield _sse("done", {"step": "done", "message": "Background generation complete", "data": doc})
            else:
                yield _sse(step, {"step": step, "message": message})

        if generated_bg_url:
            background_tasks.add_task(
                deduct_credits_and_record,
                user=current_user,
                feature_name="background_generation_ai",
                generated_face_url=generated_bg_url,
                notes=body.notes or "",
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)
