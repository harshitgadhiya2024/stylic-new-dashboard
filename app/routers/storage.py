"""
Cloudflare R2 storage utilities (authenticated).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.database import get_photoshoots_collection
from app.dependencies import get_current_user
from app.models.storage import DeleteR2ByPublicUrlRequest, DeleteR2ByPublicUrlResponse
from app.services.r2_service import delete_object_by_key, public_url_to_object_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/storage", tags=["Storage"])

_PHOTOSHOOT_EDIT_SEGMENTS = frozenset({"fabric", "texture", "color"})


def _photoshoot_id_from_key(key: str) -> str | None:
    """Return photoshoot_id encoded in ``photoshoots/...`` keys, or None."""
    parts = key.split("/")
    if len(parts) < 3 or parts[0] != "photoshoots":
        return None
    if parts[1] in _PHOTOSHOOT_EDIT_SEGMENTS and len(parts) >= 4:
        return parts[2]
    return parts[1]


async def _assert_user_can_delete_object(user_id: str, key: str) -> None:
    """Only allow deletes for objects owned by this user (same rules as upload prefixes)."""
    if key.startswith(f"users/{user_id}/"):
        return
    if key.startswith(f"thumbnails/{user_id}_"):
        return

    if key.startswith("photoshoots/"):
        pid = _photoshoot_id_from_key(key)
        if not pid:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not allowed to delete this object path.",
            )
        col = get_photoshoots_collection()
        doc = await col.find_one(
            {"photoshoot_id": pid, "user_id": user_id},
            {"_id": 1},
        )
        if doc:
            return
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Photoshoot not found or you do not own it.",
        )

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            "You may only delete URLs under your account: "
            "`users/{you}/...`, `thumbnails/{you}_...`, or `photoshoots/...` for photoshoots you own."
        ),
    )


@router.post(
    "/r2/delete-by-public-url",
    response_model=DeleteR2ByPublicUrlResponse,
    summary="Delete R2 object by public URL",
    description=(
        "Deletes a single file from Cloudflare R2 when you pass its **public HTTPS URL** "
        "(must match `R2_PUBLIC_URL`). You must be logged in and may only delete objects "
        "under your `users/` prefix, your `thumbnails/` files, or outputs for **your** photoshoots."
    ),
)
async def delete_r2_by_public_url(
    body: DeleteR2ByPublicUrlRequest,
    current_user: dict = Depends(get_current_user),
) -> DeleteR2ByPublicUrlResponse:
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Missing user_id on session.")

    try:
        key = public_url_to_object_key(body.public_url)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    await _assert_user_can_delete_object(user_id, key)

    await delete_object_by_key(key)
    logger.info("R2 delete user=%s key=%s", user_id, key)

    return DeleteR2ByPublicUrlResponse(
        success=True,
        message="Object deleted (or did not exist).",
        object_key=key,
    )
