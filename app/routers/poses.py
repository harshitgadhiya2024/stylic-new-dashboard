import json
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.database import get_poses_collection
from app.dependencies import get_current_user
from app.models.pose import (
    CreatePoseFromImageRequest,
    CreatePoseFromPromptRequest,
    DeletePosesRequest,
    PoseSchema,
)
from app.services.credit_service import check_sufficient_credits, deduct_credits_and_record
from app.services.pose_mannequin_service import stream_pose_from_image_url, stream_pose_from_text_prompt

router = APIRouter(prefix="/api/v1/poses", tags=["Poses"])

_POSE_TYPE_DB = frozenset({"front", "back", "side"})

_SSE_HEADERS = {
    "Cache-Control":     "no-cache",
    "X-Accel-Buffering": "no",
    "Connection":        "keep-alive",
}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _clean_pose(doc: dict) -> dict:
    d = dict(doc)
    d.pop("_id", None)
    d.setdefault("is_favorite", False)
    d.setdefault("is_active", True)
    d.setdefault("is_default", False)
    d.setdefault("count", 0)
    d.setdefault("tags", [])
    d.setdefault("notes", "")
    d.setdefault("pose_type", "front")
    d.setdefault("pose_name", "")
    d.setdefault("pose_prompt", "")
    d.setdefault("image_url", "")
    return d


def _normalize_pose_type_filter(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    key = s.lower().replace(" ", "_").replace("-", "_")
    while "__" in key:
        key = key.replace("__", "_")
    if key in _POSE_TYPE_DB:
        return key
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=(
            f"Invalid pose_type {raw!r}. Use one of: front, back, side "
            "(any case)."
        ),
    )


def _default_poses_query() -> dict:
    """Platform poses: explicit default OR no user_id."""
    return {
        "$and": [
            {
                "$or": [
                    {"is_default": True},
                    {"user_id": {"$exists": False}},
                    {"user_id": None},
                    {"user_id": ""},
                ]
            },
            {"is_active": {"$ne": False}},
        ]
    }


@router.get(
    "/",
    summary="Get Poses",
    description=(
        "`type=default` — platform poses (`is_default` or no `user_id`). "
        "`type=custom` — current user's poses (`is_active` true). "
        "Sorted: custom lists favorites first, then newest. "
        "Optional filters: `is_favorite`, `pose_type` (front|back|side)."
    ),
)
async def get_poses(
    type: Literal["default", "custom"] = Query(
        ...,
        description="'default' for platform poses, 'custom' for user-created poses",
    ),
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    limit: int = Query(default=10, ge=1, le=100, description="Items per page"),
    is_favorite: Optional[bool] = Query(
        default=None,
        description="Filter by favorite; omit for no filter.",
    ),
    pose_type: Optional[str] = Query(
        default=None,
        description="Optional. One of: front, back, side (any case).",
    ),
    current_user: dict = Depends(get_current_user),
):
    col = get_poses_collection()
    pt = _normalize_pose_type_filter(pose_type)

    if type == "default":
        query: dict = _default_poses_query()
    else:
        query = {
            "user_id":   current_user["user_id"],
            "is_active": {"$ne": False},
        }

    if pt is not None:
        query["pose_type"] = pt
    if is_favorite is not None:
        query["is_favorite"] = is_favorite

    if type == "custom":
        cursor = col.find(query).sort([("is_favorite", -1), ("created_at", -1)])
    else:
        cursor = col.find(query).sort([("pose_id", 1)])

    docs = await cursor.to_list(length=None)
    total = len(docs)
    skip = (page - 1) * limit
    paged = docs[skip : skip + limit]
    total_pages = (total + limit - 1) // limit if total else 1

    return {
        "type":        type,
        "page":        page,
        "limit":       limit,
        "total":       total,
        "total_pages": total_pages,
        "filters":     {
            "is_favorite": is_favorite,
            "pose_type":   pt,
        },
        "data":        [_clean_pose(d) for d in paged],
    }


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Create Pose from Image (Streaming)",
    description=(
        "Downloads `image_url`, converts to mannequin (Gemini image), derives pose prompt "
        "(Gemini flash), uploads PNG to S3, saves document. SSE stream; final `done` has record. "
        "Costs the same credits as face generation (2.5)."
    ),
)
async def create_pose_from_image(
    body: CreatePoseFromImageRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    check_sufficient_credits(current_user)
    tags = body.tags if body.tags is not None else []

    async def event_stream() -> AsyncGenerator[str, None]:
        mannequin_url: str | None = None
        pose_prompt_final: str | None = None
        try:
            async for chunk in stream_pose_from_image_url(body.image_url):
                kind = chunk[0]
                if kind == "progress":
                    yield _sse("progress", {"step": "progress", "message": chunk[1]})
                elif kind == "done":
                    mannequin_url = chunk[1]
                    pose_prompt_final = chunk[2]
                    break
            if not mannequin_url or pose_prompt_final is None:
                yield _sse("error", {"step": "error", "message": "Pipeline returned no result"})
                return

            yield _sse("storing_db", {"step": "storing_db", "message": "Storing in database"})
            now = datetime.now(timezone.utc)
            pose_id = str(uuid.uuid4())
            doc = {
                "pose_id":     pose_id,
                "user_id":     current_user["user_id"],
                "pose_name":   body.pose_name,
                "pose_type":   body.pose_type,
                "pose_prompt": pose_prompt_final,
                "image_url":   mannequin_url,
                "count":       0,
                "notes":       body.notes or "",
                "tags":        tags,
                "is_default":  False,
                "is_favorite": False,
                "is_active":   True,
                "created_at":  now.isoformat(),
                "updated_at":  now.isoformat(),
            }
            col = get_poses_collection()
            await col.insert_one({**doc, "created_at": now, "updated_at": now})
            yield _sse("done", {"step": "done", "message": "Pose saved", "data": doc})

            background_tasks.add_task(
                deduct_credits_and_record,
                user=current_user,
                feature_name="pose_generation",
                generated_face_url=mannequin_url,
                notes=body.notes or "",
            )
        except HTTPException as exc:
            yield _sse("error", {"step": "error", "message": exc.detail})
        except Exception as exc:
            yield _sse("error", {"step": "error", "message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post(
    "/generate-from-prompt",
    status_code=status.HTTP_201_CREATED,
    summary="Create Pose from Prompt (Streaming)",
    description=(
        "Generates mannequin image from `pose_prompt` + `pose_type` (Gemini image), uploads S3, "
        "stores document using the same `pose_prompt` text. SSE stream."
    ),
)
async def create_pose_from_prompt(
    body: CreatePoseFromPromptRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    check_sufficient_credits(current_user)
    tags = body.tags if body.tags is not None else []

    async def event_stream() -> AsyncGenerator[str, None]:
        mannequin_url: str | None = None
        try:
            async for chunk in stream_pose_from_text_prompt(body.pose_prompt, body.pose_type):
                kind = chunk[0]
                if kind == "progress":
                    yield _sse("progress", {"step": "progress", "message": chunk[1]})
                elif kind == "done":
                    mannequin_url = chunk[1]
                    break
            if not mannequin_url:
                yield _sse("error", {"step": "error", "message": "No image URL returned"})
                return

            yield _sse("storing_db", {"step": "storing_db", "message": "Storing in database"})
            now = datetime.now(timezone.utc)
            pose_id = str(uuid.uuid4())
            doc = {
                "pose_id":     pose_id,
                "user_id":     current_user["user_id"],
                "pose_name":   body.pose_name,
                "pose_type":   body.pose_type,
                "pose_prompt": body.pose_prompt.strip(),
                "image_url":   mannequin_url,
                "count":       0,
                "notes":       body.notes or "",
                "tags":        tags,
                "is_default":  False,
                "is_favorite": False,
                "is_active":   True,
                "created_at":  now.isoformat(),
                "updated_at":  now.isoformat(),
            }
            col = get_poses_collection()
            await col.insert_one({**doc, "created_at": now, "updated_at": now})
            yield _sse("done", {"step": "done", "message": "Pose saved", "data": doc})

            background_tasks.add_task(
                deduct_credits_and_record,
                user=current_user,
                feature_name="pose_generation_prompt",
                generated_face_url=mannequin_url,
                notes=body.notes or "",
            )
        except HTTPException as exc:
            yield _sse("error", {"step": "error", "message": exc.detail})
        except Exception as exc:
            yield _sse("error", {"step": "error", "message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.patch(
    "/{pose_id}/toggle-favorite",
    response_model=PoseSchema,
    summary="Toggle Favorite",
    description="Toggle is_favorite for a user pose. Default/platform poses cannot be favorited.",
)
async def toggle_pose_favorite(
    pose_id: str,
    current_user: dict = Depends(get_current_user),
):
    col = get_poses_collection()
    doc = await col.find_one({"pose_id": pose_id})
    if not doc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Pose not found.")
    if doc.get("is_default", False) or not doc.get("user_id"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Default/platform poses cannot be marked as favorite.",
        )
    if doc.get("user_id") != current_user["user_id"]:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to update this pose.",
        )

    new_val = not doc.get("is_favorite", False)
    now = datetime.now(timezone.utc)
    await col.update_one(
        {"pose_id": pose_id},
        {"$set": {"is_favorite": new_val, "updated_at": now}},
    )
    doc["is_favorite"] = new_val
    doc["updated_at"] = now
    return _clean_pose(doc)


@router.delete(
    "/bulk-delete",
    summary="Delete Multiple Poses",
    description="Soft-delete user poses (is_active=False). Skips default/platform poses.",
)
async def delete_poses_bulk(
    body: DeletePosesRequest,
    current_user: dict = Depends(get_current_user),
):
    if not body.pose_ids:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="pose_ids must not be empty.",
        )
    col = get_poses_collection()
    result = await col.update_many(
        {
            "pose_id":    {"$in": body.pose_ids},
            "user_id":    current_user["user_id"],
            "is_default": {"$ne": True},
            "is_active":  {"$ne": False},
        },
        {"$set": {"is_active": False, "updated_at": datetime.now(timezone.utc)}},
    )
    deleted = result.modified_count
    return {
        "message":        f"{deleted} pose(s) deleted successfully.",
        "deleted_count":  deleted,
        "skipped_count":  len(body.pose_ids) - deleted,
    }


@router.delete(
    "/{pose_id}",
    summary="Delete Pose",
    description="Soft-delete one user pose.",
)
async def delete_pose(
    pose_id: str,
    current_user: dict = Depends(get_current_user),
):
    col = get_poses_collection()
    doc = await col.find_one({"pose_id": pose_id})
    if not doc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Pose not found.")
    if doc.get("is_default", False) or not doc.get("user_id"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Default/platform poses cannot be deleted.",
        )
    if doc.get("user_id") != current_user["user_id"]:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this pose.",
        )
    if doc.get("is_active") is False:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="Pose is already deleted.",
        )
    await col.update_one(
        {"pose_id": pose_id},
        {"$set": {"is_active": False, "updated_at": datetime.now(timezone.utc)}},
    )
    return {"message": "Pose deleted successfully.", "pose_id": pose_id}
