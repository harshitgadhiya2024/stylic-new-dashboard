import asyncio
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


def _is_platform_pose(doc: dict) -> bool:
    if doc.get("is_default", False):
        return True
    uid = doc.get("user_id")
    return uid is None or uid == ""


def serialize_pose_response(
    doc: dict,
    viewer_user_id: Optional[str] = None,
) -> dict:
    """
    Custom poses: ``user_id`` + document ``is_favorite``.
    Platform poses: omit ``user_id``; ``is_favorite`` = viewer in ``favorite_list``.
    """
    d = dict(doc)
    d.pop("_id", None)
    fl = list(d.get("favorite_list") or [])
    platform = _is_platform_pose(d)
    fallback_ts = datetime.now(timezone.utc)
    created_at = d.get("created_at") or fallback_ts
    updated_at = d.get("updated_at") or fallback_ts
    out: dict = {
        "pose_id":       d["pose_id"],
        "pose_name":     d.get("pose_name") or "",
        "pose_type":     d.get("pose_type") or "front",
        "pose_prompt":   d.get("pose_prompt") or "",
        "image_url":     d.get("image_url") or "",
        "mannequin_framing": d.get("mannequin_framing"),
        "count":         int(d.get("count", 0) or 0),
        "notes":         d.get("notes") or "",
        "tags":          d.get("tags") or [],
        "favorite_list": fl,
        "is_default":    bool(d.get("is_default", False)),
        "is_active":     bool(d.get("is_active", True)),
        "created_at":    created_at,
        "updated_at":    updated_at,
    }
    if platform:
        if viewer_user_id is not None:
            out["is_favorite"] = viewer_user_id in fl
        else:
            out["is_favorite"] = False
    else:
        out["is_favorite"] = bool(d.get("is_favorite", False))
        out["user_id"] = d.get("user_id")
    return out


def _jsonable_pose_for_sse(doc: dict, viewer_user_id: Optional[str] = None) -> dict:
    data = serialize_pose_response(doc, viewer_user_id=viewer_user_id)
    for key, val in list(data.items()):
        if isinstance(val, datetime):
            data[key] = val.isoformat()
    return data


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
        "When `is_favorite=true`, `type` is ignored: returns your custom poses with "
        "`is_favorite=true` plus platform poses whose `favorite_list` contains your `user_id`. "
        "When `is_favorite=false`, filtering is scoped by `type` (custom: non-favorites; "
        "default: platform poses you have not added to `favorite_list`). "
        "Responses include `favorite_list`; for platform defaults, `is_favorite` reflects the current user. "
        "Sorted: custom (no favorite filter) — favorites first, then newest; default — `pose_id`. "
        "Optional filter: `pose_type` (front|back|side). "
        "`type` is required unless `is_favorite=true`."
    ),
)
async def get_poses(
    type: Optional[Literal["default", "custom"]] = Query(
        default=None,
        description=(
            "'default' for platform poses, 'custom' for user-created poses. "
            "Required unless `is_favorite=true` (favorites union ignores `type`)."
        ),
    ),
    page: int = Query(default=1, ge=1, description="Page number (1-based)"),
    limit: int = Query(default=10, ge=1, le=100, description="Items per page"),
    is_favorite: Optional[bool] = Query(
        default=None,
        description=(
            "true: favorites only — ignores `type`. false: non-favorites within `type`. "
            "Omit: no favorite filter."
        ),
    ),
    pose_type: Optional[str] = Query(
        default=None,
        description="Optional. One of: front, back, side (any case).",
    ),
    current_user: dict = Depends(get_current_user),
):
    if type is None and is_favorite is not True:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query parameter `type` is required unless `is_favorite` is true.",
        )

    col = get_poses_collection()
    uid = current_user["user_id"]
    pt = _normalize_pose_type_filter(pose_type)

    def _sort_key_updated(doc: dict) -> datetime:
        u = doc.get("updated_at")
        if u is None:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)
        if isinstance(u, datetime) and u.tzinfo is None:
            return u.replace(tzinfo=timezone.utc)
        return u

    default_and = _default_poses_query().get("$and") or []

    docs: list[dict]

    if is_favorite is True:
        q_custom: dict = {
            "user_id":     uid,
            "is_active":   {"$ne": False},
            "is_favorite": True,
        }
        q_default: dict = {"$and": [*default_and, {"favorite_list": uid}]}
        if pt is not None:
            q_custom["pose_type"] = pt
            q_default["pose_type"] = pt
        d_custom = await col.find(q_custom).sort([("created_at", -1)]).to_list(length=None)
        d_default = await col.find(q_default).sort([("updated_at", -1)]).to_list(length=None)
        by_pid: dict[str, dict] = {}
        for d in d_custom + d_default:
            by_pid[d["pose_id"]] = d
        docs = list(by_pid.values())
        docs.sort(key=_sort_key_updated, reverse=True)
    elif is_favorite is False:
        if type == "custom":
            query: dict = {
                "user_id":     uid,
                "is_active":   {"$ne": False},
                "is_favorite": False,
            }
        else:
            query = {
                "$and": [
                    *default_and,
                    {
                        "$or": [
                            {"favorite_list": {"$exists": False}},
                            {"favorite_list": []},
                            {"favorite_list": {"$nin": [uid]}},
                        ]
                    },
                ]
            }
        if pt is not None:
            query["pose_type"] = pt
        if type == "custom":
            cursor = col.find(query).sort([("created_at", -1)])
        else:
            cursor = col.find(query).sort([("pose_id", 1)])
        docs = await cursor.to_list(length=None)
    else:
        if type == "default":
            query = dict(_default_poses_query())
        else:
            query = {
                "user_id":   uid,
                "is_active": {"$ne": False},
            }
        if pt is not None:
            query["pose_type"] = pt
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
        "type":        type if is_favorite is not True else None,
        "page":        page,
        "limit":       limit,
        "total":       total,
        "total_pages": total_pages,
        "filters":     {
            "is_favorite": is_favorite,
            "pose_type":   pt,
        },
        "data":        [serialize_pose_response(d, viewer_user_id=uid) for d in paged],
    }


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Create Pose from Image (Streaming)",
    description=(
        "Uses `image_url` with SeedDream 5.0 Lite (image-to-image) to produce a mannequin PNG, "
        "derives pose prompt with Gemini vision, uploads PNG to R2, saves document. "
        "SSE stream; final `done` has record. Costs the same credits as face generation (2.5)."
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
            yield _sse("initialize", {"step": "initialize", "message": "Initializing custom pose generation"})
            await asyncio.sleep(0.6)
            yield _sse("validating_input", {"step": "validating_input", "message": "Validating pose request input"})
            await asyncio.sleep(0.6)
            yield _sse("validating_image", {"step": "validating_image", "message": "Checking source image accessibility"})
            await asyncio.sleep(0.6)
            yield _sse("preparing_pipeline", {"step": "preparing_pipeline", "message": "Preparing mannequin generation pipeline"})
            await asyncio.sleep(0.6)

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
            doc_db = {
                "pose_id":       pose_id,
                "user_id":       current_user["user_id"],
                "pose_name":     body.pose_name,
                "pose_type":     body.pose_type,
                "pose_prompt":   pose_prompt_final,
                "image_url":     mannequin_url,
                "count":         0,
                "notes":         body.notes or "",
                "tags":          tags,
                "favorite_list": [],
                "is_default":    False,
                "is_favorite":   False,
                "is_active":     True,
                "created_at":    now,
                "updated_at":    now,
            }
            if body.mannequin_framing is not None:
                doc_db["mannequin_framing"] = body.mannequin_framing
            col = get_poses_collection()
            await col.insert_one(doc_db)
            saved = await col.find_one({"pose_id": pose_id})
            payload = _jsonable_pose_for_sse(
                saved or doc_db,
                viewer_user_id=current_user["user_id"],
            )
            yield _sse("done", {"step": "done", "message": "Pose saved", "data": payload})

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
        "Generates mannequin image from `pose_prompt` + `pose_type` via SeedDream 5.0 Lite "
        "(text-to-image), uploads to R2, stores document using the same `pose_prompt` text. SSE stream."
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
            yield _sse("initialize", {"step": "initialize", "message": "Initializing custom pose generation"})
            await asyncio.sleep(0.6)
            yield _sse("validating_input", {"step": "validating_input", "message": "Validating pose prompt and pose type"})
            await asyncio.sleep(0.6)
            yield _sse("preparing_prompt", {"step": "preparing_prompt", "message": "Preparing structured mannequin generation prompt"})
            await asyncio.sleep(0.6)
            yield _sse("preparing_pipeline", {"step": "preparing_pipeline", "message": "Preparing mannequin generation pipeline"})
            await asyncio.sleep(0.6)

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
            doc_db = {
                "pose_id":       pose_id,
                "user_id":       current_user["user_id"],
                "pose_name":     body.pose_name,
                "pose_type":     body.pose_type,
                "pose_prompt":   body.pose_prompt.strip(),
                "image_url":     mannequin_url,
                "count":         0,
                "notes":         body.notes or "",
                "tags":          tags,
                "favorite_list": [],
                "is_default":    False,
                "is_favorite":   False,
                "is_active":     True,
                "created_at":    now,
                "updated_at":    now,
            }
            if body.mannequin_framing is not None:
                doc_db["mannequin_framing"] = body.mannequin_framing
            col = get_poses_collection()
            await col.insert_one(doc_db)
            saved = await col.find_one({"pose_id": pose_id})
            payload = _jsonable_pose_for_sse(
                saved or doc_db,
                viewer_user_id=current_user["user_id"],
            )
            yield _sse("done", {"step": "done", "message": "Pose saved", "data": payload})

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
    description=(
        "Query `type=custom`: flip `is_favorite` on a pose you own (not a platform default). "
        "Query `type=default`: add or remove your `user_id` in the platform pose's `favorite_list`. "
        "Response includes viewer-specific `is_favorite` for platform poses (derived from `favorite_list`)."
    ),
)
async def toggle_pose_favorite(
    pose_id: str,
    favorite_type: Literal["default", "custom"] = Query(
        ...,
        alias="type",
        description="'custom' toggles is_favorite on your pose; 'default' toggles your id in favorite_list.",
    ),
    current_user: dict = Depends(get_current_user),
):
    col = get_poses_collection()
    uid = current_user["user_id"]
    doc = await col.find_one({"pose_id": pose_id})
    if not doc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Pose not found.")

    now = datetime.now(timezone.utc)

    if favorite_type == "custom":
        if _is_platform_pose(doc):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail="This pose is a platform default — use type=default to favorite it.",
            )
        if doc.get("user_id") != uid:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to update this pose.",
            )
        new_val = not doc.get("is_favorite", False)
        await col.update_one(
            {"pose_id": pose_id},
            {"$set": {"is_favorite": new_val, "updated_at": now}},
        )
    else:
        if not _is_platform_pose(doc):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail="This pose is user-created — use type=custom to toggle favorite.",
            )
        fl = list(doc.get("favorite_list") or [])
        if uid in fl:
            await col.update_one(
                {"pose_id": pose_id},
                {"$pull": {"favorite_list": uid}, "$set": {"updated_at": now}},
            )
        else:
            await col.update_one(
                {"pose_id": pose_id},
                {"$addToSet": {"favorite_list": uid}, "$set": {"updated_at": now}},
            )

    saved = await col.find_one({"pose_id": pose_id})
    if not saved:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Pose not found after update.")
    return PoseSchema(**serialize_pose_response(saved, viewer_user_id=uid))


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
