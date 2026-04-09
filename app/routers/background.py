import json
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Literal, Optional

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


def _strip_mongo_id(doc: dict) -> dict:
    d = dict(doc)
    d.pop("_id", None)
    return d


def serialize_background_response(
    doc: dict,
    viewer_user_id: Optional[str] = None,
) -> dict:
    """Canonical API shape with favorite_list. Default rows: is_favorite from favorite_list + viewer."""
    d = _strip_mongo_id(doc)
    is_def = bool(d.get("is_default", False))
    fallback_ts = datetime.now(timezone.utc)
    created_at = d.get("created_at") or fallback_ts
    updated_at = d.get("updated_at") or fallback_ts
    out: dict[str, Any] = {
        "background_id":   d["background_id"],
        "user_id":          None,
        "background_type":  d.get("background_type") or "",
        "background_name":  d.get("background_name") or "",
        "background_url":   d.get("background_url") or "",
        "count":            int(d.get("count", 0) or 0),
        "tags":             d.get("tags") or [],
        "notes":            d.get("notes") or "",
        "favorite_list":    d.get("favorite_list") or [],
        "is_default":       is_def,
        "is_active":        bool(d.get("is_active", True)),
        "created_at":       created_at,
        "updated_at":       updated_at,
    }
    if not is_def:
        out["user_id"] = d.get("user_id")
        out["is_favorite"] = bool(d.get("is_favorite", False))
    else:
        if viewer_user_id is not None:
            fl = d.get("favorite_list") or []
            out["is_favorite"] = viewer_user_id in fl
        else:
            out["is_favorite"] = False
    return out


def _jsonable_background_for_sse(doc: dict, viewer_user_id: Optional[str] = None) -> dict:
    data = serialize_background_response(doc, viewer_user_id=viewer_user_id)
    for key, val in list(data.items()):
        if isinstance(val, datetime):
            data[key] = val.isoformat()
    return data


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
                    bid = str(uuid.uuid4())
                    doc_db = {
                        "background_id":   bid,
                        "user_id":         current_user["user_id"],
                        "background_type": body.background_type,
                        "background_name": body.background_name,
                        "background_url":  generated_bg_url,
                        "count":           0,
                        "tags":            body.tags or [],
                        "notes":           body.notes or "",
                        "favorite_list":   [],
                        "is_default":      False,
                        "is_active":       True,
                        "is_favorite":     False,
                        "created_at":      now,
                        "updated_at":      now,
                    }

                    col = get_backgrounds_collection()
                    await col.insert_one(doc_db)
                    saved = await col.find_one({"background_id": bid})
                    payload = _jsonable_background_for_sse(
                        saved or doc_db,
                        viewer_user_id=current_user["user_id"],
                    )
                    yield _sse("done", {"step": "done", "message": "Background generation complete", "data": payload})
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
                    bid = str(uuid.uuid4())
                    doc_db = {
                        "background_id":   bid,
                        "user_id":         current_user["user_id"],
                        "background_type": body.background_type,
                        "background_name": body.background_name,
                        "background_url":  generated_bg_url,
                        "count":           0,
                        "tags":            body.tags or [],
                        "notes":           body.notes or "",
                        "favorite_list":   [],
                        "is_default":      False,
                        "is_active":       True,
                        "is_favorite":     False,
                        "created_at":      now,
                        "updated_at":      now,
                    }

                    col = get_backgrounds_collection()
                    await col.insert_one(doc_db)
                    saved = await col.find_one({"background_id": bid})
                    payload = _jsonable_background_for_sse(
                        saved or doc_db,
                        viewer_user_id=current_user["user_id"],
                    )
                    yield _sse("done", {"step": "done", "message": "Background generation complete", "data": payload})
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
        "Returns a paginated list of backgrounds. `type` is required unless `is_favorite=true`. "
        "`type=default` — platform defaults (is_default=True). "
        "`type=custom` — your backgrounds (user_id, is_active=True), favorites first. "
        "When `is_favorite=true`, `type` is ignored: union of your custom `is_favorite=true` rows "
        "and defaults whose `favorite_list` contains your user_id. "
        "When `is_favorite=false`, filtering is scoped by `type` (custom: your non-favorites; "
        "default: defaults you have not added to `favorite_list`). "
        "Optional `background_type` filter applies whenever used. "
        "Responses include `favorite_list`; for defaults, `is_favorite` reflects the current user."
    ),
)
async def get_backgrounds(
    type: Optional[Literal["default", "custom"]] = Query(
        default=None,
        description=(
            "'default' for platform backgrounds, 'custom' for user-created. "
            "Required unless `is_favorite=true`."
        ),
    ),
    page:  int = Query(default=1,  ge=1,        description="Page number (1-based)"),
    limit: int = Query(default=10, ge=1, le=100, description="Number of items per page"),
    is_favorite: Optional[bool] = Query(
        default=None,
        description=(
            "true: favorites only — ignores `type`. false: non-favorites within `type`. "
            "Omit: no favorite filter."
        ),
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
    if type is None and is_favorite is not True:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query parameter `type` is required unless `is_favorite` is true.",
        )

    col = get_backgrounds_collection()
    uid = current_user["user_id"]
    bg_type = _normalize_background_type_filter(background_type)

    def _sort_key_updated(doc: dict) -> datetime:
        u = doc.get("updated_at")
        if u is None:
            return datetime(1970, 1, 1, tzinfo=timezone.utc)
        if isinstance(u, datetime) and u.tzinfo is None:
            return u.replace(tzinfo=timezone.utc)
        return u

    docs: list[dict]

    if is_favorite is True:
        q_custom: dict = {
            "user_id":     uid,
            "is_active":   True,
            "is_favorite": True,
            "is_default":  False,
        }
        q_default: dict = {"is_default": True, "favorite_list": uid}
        if bg_type is not None:
            q_custom["background_type"] = bg_type
            q_default["background_type"] = bg_type
        d_custom = await col.find(q_custom).sort([("created_at", -1)]).to_list(length=None)
        d_default = await col.find(q_default).sort([("updated_at", -1)]).to_list(length=None)
        by_bid: dict[str, dict] = {}
        for d in d_custom + d_default:
            by_bid[d["background_id"]] = d
        docs = list(by_bid.values())
        docs.sort(key=_sort_key_updated, reverse=True)
    elif is_favorite is False:
        if type == "custom":
            query: dict = {
                "user_id":     uid,
                "is_active":   True,
                "is_favorite": False,
                "is_default":  False,
            }
        else:
            query = {
                "is_default": True,
                "$or":        [
                    {"favorite_list": {"$exists": False}},
                    {"favorite_list": []},
                    {"favorite_list": {"$nin": [uid]}},
                ],
            }
        if bg_type is not None:
            query["background_type"] = bg_type
        if type == "custom":
            cursor = col.find(query).sort([("created_at", -1)])
        else:
            cursor = col.find(query)
        docs = await cursor.to_list(length=None)
    else:
        if type == "default":
            query = {"is_default": True}
        else:
            query = {"user_id": uid, "is_active": True}
        if bg_type is not None:
            query["background_type"] = bg_type
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
        "type":        type if is_favorite is not True else None,
        "page":        page,
        "limit":       limit,
        "total":       total,
        "total_pages": total_pages,
        "filters":     {
            "is_favorite":     is_favorite,
            "background_type": bg_type,
        },
        "data":        [serialize_background_response(d, viewer_user_id=uid) for d in paged],
    }


@router.patch(
    "/{background_id}/toggle-favorite",
    response_model=BackgroundSchema,
    summary="Toggle Favorite",
    description=(
        "Query `type=custom`: flip `is_favorite` on a background you own. "
        "Query `type=default`: add or remove your `user_id` in the default background's `favorite_list`. "
        "Response includes viewer-specific `is_favorite` for defaults."
    ),
)
async def toggle_background_favorite(
    background_id: str,
    favorite_type: Literal["default", "custom"] = Query(
        ...,
        alias="type",
        description="'custom' toggles is_favorite on your background; 'default' toggles your id in favorite_list.",
    ),
    current_user: dict = Depends(get_current_user),
):
    col = get_backgrounds_collection()
    uid = current_user["user_id"]

    doc = await col.find_one({"background_id": background_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Background not found.",
        )

    now = datetime.now(timezone.utc)

    if favorite_type == "custom":
        if doc.get("is_default", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This is a platform default — use type=default to favorite it.",
            )
        if doc.get("user_id") != uid:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to update this background.",
            )
        new_value = not doc.get("is_favorite", False)
        await col.update_one(
            {"background_id": background_id},
            {"$set": {"is_favorite": new_value, "updated_at": now}},
        )
    else:
        if not doc.get("is_default", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This background is user-created — use type=custom to toggle favorite.",
            )
        fl = list(doc.get("favorite_list") or [])
        if uid in fl:
            await col.update_one(
                {"background_id": background_id},
                {"$pull": {"favorite_list": uid}, "$set": {"updated_at": now}},
            )
        else:
            await col.update_one(
                {"background_id": background_id},
                {"$addToSet": {"favorite_list": uid}, "$set": {"updated_at": now}},
            )

    saved = await col.find_one({"background_id": background_id})
    if not saved:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Background not found after update.",
        )
    return BackgroundSchema(**serialize_background_response(saved, viewer_user_id=uid))


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
