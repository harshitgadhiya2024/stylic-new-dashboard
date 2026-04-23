import asyncio
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.config import settings
from app.database import get_credit_history_collection, get_remove_background_collection, get_users_collection
from app.dependencies import get_current_user
from app.models.remove_background import (
    RemoveBackgroundCreateRequest,
    RemoveBackgroundIdRequest,
    RemoveBackgroundIdsRequest,
)
from app.services.remove_background_service import download_image_bytes, remove_background_to_output_url
from app.services.r2_service import upload_bytes_to_r2

# KIE/fal work runs in this HTTP request (async + thread offload only). No Celery, no Redis
# job queue, and no app.services.kie_rate_limiter (photoshoot pipeline only).
router = APIRouter(prefix="/api/v1/remove-background", tags=["Remove background"])


def _require_http_url(url: str) -> None:
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="input_image_public_url must be a valid http(s) URL.",
        )


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Remove background (API-1)",
    description=(
        "Removes background from ``input_image_public_url`` (KIE recraft/remove-background, "
        "optional fal.ai fallback), uploads result to R2, stores a row, deducts credits "
        f"({settings.CREDIT_REMOVE_BACKGROUND} per request), and writes ``credit_history``. "
        "**Synchronous for the client:** the handler waits for KIE/fal, download, R2 upload, "
        "and credit deduction in one request — **no Celery**, **no Redis task queue**, "
        "**no** ``kie_rate_limiter`` (unlike photoshoot generation)."
    ),
)
async def remove_background_create(
    body: RemoveBackgroundCreateRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    url_in = body.input_image_public_url.strip()
    _require_http_url(url_in)

    credit_cost = float(settings.CREDIT_REMOVE_BACKGROUND)
    current_credits = float(current_user.get("credits", 0))
    if current_credits < credit_cost:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"Insufficient credits. Remove background costs {credit_cost} credits "
                f"but you only have {current_credits}."
            ),
        )

    try:
        provider_url = await asyncio.to_thread(remove_background_to_output_url, url_in)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Remove-background provider failed: {exc}",
        ) from exc

    try:
        image_bytes = await download_image_bytes(provider_url)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to download processed image: {exc}",
        ) from exc

    remove_background_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    key = f"remove-background/{user_id}/{remove_background_id}.png"
    try:
        public_url = await upload_bytes_to_r2(image_bytes, key, content_type="image/png")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload to R2: {exc}",
        ) from exc

    col = get_remove_background_collection()
    doc = {
        "remove_background_id": remove_background_id,
        "user_id":               user_id,
        "input_image":           url_in,
        "generated_image":       public_url,
        "is_active":             True,
        "is_credit_deducted":    False,
        "is_favorite":           False,
        "created_at":            now,
        "updated_at":            now,
    }
    await col.insert_one(doc)

    users_col = get_users_collection()
    history_col = get_credit_history_collection()
    new_credits = round(current_credits - credit_cost, 4)
    try:
        upd = await users_col.update_one(
            {"user_id": user_id},
            {"$set": {"credits": new_credits, "updated_at": now}},
        )
        if upd.matched_count == 0:
            await col.delete_one({"remove_background_id": remove_background_id})
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
        await history_col.insert_one({
            "history_id":       str(uuid.uuid4()),
            "user_id":          user_id,
            "feature_name":     "remove_background",
            "credit":           credit_cost,
            "credit_per_image": credit_cost,
            "type":             "deduct",
            "thumbnail_image":  "",
            "notes":            f"Remove background — id {remove_background_id}",
            "remove_background_id": remove_background_id,
            "created_at":       now,
        })
        await col.update_one(
            {"remove_background_id": remove_background_id},
            {"$set": {"is_credit_deducted": True, "updated_at": now}},
        )
    except HTTPException:
        raise
    except Exception as exc:
        await col.delete_one({"remove_background_id": remove_background_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Credit deduction failed: {exc}",
        ) from exc

    final = await col.find_one({"remove_background_id": remove_background_id}, {"_id": 0})
    if not final:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Record missing after create.",
        )
    return final


@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Get all remove-background rows (API-2)",
    description=(
        "Returns all active rows for the authenticated user. "
        "When ``is_favorite=true``, only favorites are returned."
    ),
)
async def remove_background_list(
    current_user: dict = Depends(get_current_user),
    is_favorite: bool | None = Query(
        default=None,
        description="If true, filter to is_favorite=True; if omitted, all active rows.",
    ),
):
    user_id = current_user["user_id"]
    col = get_remove_background_collection()
    query: dict = {"user_id": user_id, "is_active": True}
    if is_favorite is True:
        query["is_favorite"] = True
    cursor = col.find(query, {"_id": 0}).sort("created_at", -1)
    items = await cursor.to_list(length=None)
    return {"remove_backgrounds": items, "count": len(items)}


@router.get(
    "/detail",
    status_code=status.HTTP_200_OK,
    summary="Get one remove-background row (API-3)",
)
async def remove_background_detail(
    remove_background_id: str = Query(..., description="remove_background_id"),
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    col = get_remove_background_collection()
    doc = await col.find_one(
        {"remove_background_id": remove_background_id, "user_id": user_id, "is_active": True},
        {"_id": 0},
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Remove-background record not found.",
        )
    return doc


@router.patch(
    "/delete-multiple",
    status_code=status.HTTP_200_OK,
    summary="Soft-delete multiple (API-4)",
)
async def remove_background_delete_multiple(
    body: RemoveBackgroundIdsRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    now = datetime.now(timezone.utc)
    col = get_remove_background_collection()
    result = await col.update_many(
        {
            "remove_background_id": {"$in": body.remove_background_ids},
            "user_id":               user_id,
        },
        {"$set": {"is_active": False, "updated_at": now}},
    )
    return {
        "message":       f"{result.modified_count} record(s) deactivated.",
        "modified_count": result.modified_count,
    }


@router.patch(
    "/delete",
    status_code=status.HTTP_200_OK,
    summary="Soft-delete one (API-5)",
)
async def remove_background_delete_one(
    body: RemoveBackgroundIdRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    now = datetime.now(timezone.utc)
    col = get_remove_background_collection()
    result = await col.update_one(
        {
            "remove_background_id": body.remove_background_id,
            "user_id":               user_id,
        },
        {"$set": {"is_active": False, "updated_at": now}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found.")
    return {"message": "Record deactivated.", "remove_background_id": body.remove_background_id}


@router.patch(
    "/favorite-toggle",
    status_code=status.HTTP_200_OK,
    summary="Toggle favorite (API-6)",
)
async def remove_background_favorite_toggle(
    body: RemoveBackgroundIdRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    now = datetime.now(timezone.utc)
    col = get_remove_background_collection()
    doc = await col.find_one(
        {"remove_background_id": body.remove_background_id, "user_id": user_id, "is_active": True},
    )
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found.")
    new_fav = not bool(doc.get("is_favorite", False))
    await col.update_one(
        {"remove_background_id": body.remove_background_id, "user_id": user_id},
        {"$set": {"is_favorite": new_fav, "updated_at": now}},
    )
    return {
        "remove_background_id": body.remove_background_id,
        "is_favorite":          new_fav,
    }
