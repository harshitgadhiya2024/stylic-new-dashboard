import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.config import settings
from app.database import get_credit_history_collection, get_user_upscaled_collection, get_users_collection
from app.dependencies import get_current_user
from app.models.user_upscale import UserUpscaleIdRequest, UserUpscaleIdsRequest, UserUpscaleImageRequest
from app.services.r2_service import upload_bytes_to_r2
from app.services.user_upscale_service import run_standalone_kie_upscale

# KIE work runs in this HTTP request only. No Celery, no Redis job queue, no kie_rate_limiter.
router = APIRouter(prefix="/api/v1/upscale-image", tags=["Upscale image"])


def _require_http_url(url: str) -> None:
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="input_image_public_url must be a valid http(s) URL.",
        )


def _credit_for_factor(factor: int) -> float:
    if factor == 2:
        return float(settings.CREDIT_USER_UPSCALE_2X)
    if factor == 4:
        return float(settings.CREDIT_USER_UPSCALE_4X)
    return float(settings.CREDIT_USER_UPSCALE_8X)


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Upscale image via KIE (API-1)",
    description=(
        "Standalone KIE ``topaz/image-upscale`` only. ``upscale_factor`` 2, 4, or 8. "
        "Create-task is retried up to 4 times on failure. Credits: 2x → "
        f"{settings.CREDIT_USER_UPSCALE_2X}, 4x → {settings.CREDIT_USER_UPSCALE_4X}, "
        f"8x → {settings.CREDIT_USER_UPSCALE_8X} (configurable via env). "
        "**Synchronous for the client:** poll + download + R2 + credits complete in one "
        "request — **no Celery**, **no Redis task queue**, **no** ``kie_rate_limiter``."
    ),
)
async def upscale_image_create(
    body: UserUpscaleImageRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    url_in = body.input_image_public_url.strip()
    factor = int(body.upscale_factor)
    _require_http_url(url_in)

    credit_cost = _credit_for_factor(factor)
    current_credits = float(current_user.get("credits", 0))
    if current_credits < credit_cost:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"Insufficient credits. This {factor}x upscale costs {credit_cost} credits "
                f"but you only have {current_credits}."
            ),
        )

    upscale_id = str(uuid.uuid4())
    try:
        png_bytes, resolution_detail = await run_standalone_kie_upscale(
            image_url=url_in,
            upscale_factor=factor,
            trace_id=upscale_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"KIE upscale failed: {exc}",
        ) from exc

    now = datetime.now(timezone.utc)
    key = f"user-upscaled/{user_id}/{upscale_id}_{factor}x.png"
    try:
        public_url = await upload_bytes_to_r2(png_bytes, key, content_type="image/png")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload to R2: {exc}",
        ) from exc

    col = get_user_upscaled_collection()
    doc = {
        "upscale_id":                 upscale_id,
        "user_id":                    user_id,
        "input_image":                url_in,
        "generated_image":            public_url,
        "output_image_resolution":    factor,
        "credit_cutted":              credit_cost,
        "is_active":                  True,
        "is_credit_deducted":         False,
        "is_favorite":                False,
        "created_at":                 now,
        "updated_at":                 now,
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
            await col.delete_one({"upscale_id": upscale_id})
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

        await history_col.insert_one({
            "history_id":                str(uuid.uuid4()),
            "user_id":                   user_id,
            "feature_name":              "user_upscale_image",
            "credit":                    credit_cost,
            "credit_per_image":          credit_cost,
            "type":                      "deduct",
            "thumbnail_image":           "",
            "notes":                     f"{resolution_detail} | upscale_id={upscale_id}",
            "upscale_id":                upscale_id,
            "output_image_resolution": factor,
            "output_resolution_detail": resolution_detail,
            "generated_image_url":     public_url,
            "created_at":                now,
        })
        await col.update_one(
            {"upscale_id": upscale_id},
            {"$set": {"is_credit_deducted": True, "updated_at": now}},
        )
    except HTTPException:
        raise
    except Exception as exc:
        await col.delete_one({"upscale_id": upscale_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Credit deduction failed: {exc}",
        ) from exc

    final = await col.find_one({"upscale_id": upscale_id}, {"_id": 0})
    if not final:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Record missing after create.",
        )
    return final


@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Get all user upscale rows (API-2)",
)
async def upscale_image_list(
    current_user: dict = Depends(get_current_user),
    is_favorite: bool | None = Query(
        default=None,
        description="If true, only rows with is_favorite=True.",
    ),
):
    user_id = current_user["user_id"]
    col = get_user_upscaled_collection()
    query: dict = {"user_id": user_id, "is_active": True}
    if is_favorite is True:
        query["is_favorite"] = True
    items = await col.find(query, {"_id": 0}).sort("created_at", -1).to_list(length=None)
    return {"upscales": items, "count": len(items)}


@router.get(
    "/detail",
    status_code=status.HTTP_200_OK,
    summary="Get one user upscale row (API-3)",
)
async def upscale_image_detail(
    upscale_id: str = Query(..., description="upscale_id"),
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    col = get_user_upscaled_collection()
    doc = await col.find_one(
        {"upscale_id": upscale_id, "user_id": user_id, "is_active": True},
        {"_id": 0},
    )
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found.")
    return doc


@router.patch(
    "/delete-multiple",
    status_code=status.HTTP_200_OK,
    summary="Soft-delete multiple (API-4)",
)
async def upscale_image_delete_multiple(
    body: UserUpscaleIdsRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    now = datetime.now(timezone.utc)
    col = get_user_upscaled_collection()
    result = await col.update_many(
        {"upscale_id": {"$in": body.upscale_ids}, "user_id": user_id},
        {"$set": {"is_active": False, "updated_at": now}},
    )
    return {"message": f"{result.modified_count} record(s) deactivated.", "modified_count": result.modified_count}


@router.patch(
    "/delete",
    status_code=status.HTTP_200_OK,
    summary="Soft-delete one (API-5)",
)
async def upscale_image_delete_one(
    body: UserUpscaleIdRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    now = datetime.now(timezone.utc)
    col = get_user_upscaled_collection()
    result = await col.update_one(
        {"upscale_id": body.upscale_id, "user_id": user_id},
        {"$set": {"is_active": False, "updated_at": now}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found.")
    return {"message": "Record deactivated.", "upscale_id": body.upscale_id}


@router.patch(
    "/favorite-toggle",
    status_code=status.HTTP_200_OK,
    summary="Toggle favorite (API-6)",
)
async def upscale_image_favorite_toggle(
    body: UserUpscaleIdRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    now = datetime.now(timezone.utc)
    col = get_user_upscaled_collection()
    doc = await col.find_one(
        {"upscale_id": body.upscale_id, "user_id": user_id, "is_active": True},
    )
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Record not found.")
    new_fav = not bool(doc.get("is_favorite", False))
    await col.update_one(
        {"upscale_id": body.upscale_id, "user_id": user_id},
        {"$set": {"is_favorite": new_fav, "updated_at": now}},
    )
    return {"upscale_id": body.upscale_id, "is_favorite": new_fav}
