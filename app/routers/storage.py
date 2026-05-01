"""
Cloudflare R2 storage utilities.

``delete-by-public-url`` is a **public** route (no login): deletes are allowed only for
object keys under the ``public/`` prefix (same tree as the public upload-file endpoint).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.models.storage import DeleteR2ByPublicUrlRequest, DeleteR2ByPublicUrlResponse
from app.services.r2_service import delete_object_by_key, public_url_to_object_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/storage", tags=["Storage"])


def _assert_public_delete_key_allowed(key: str) -> None:
    """Unauthenticated deletes: only ``public/…`` keys (public upload folder)."""
    if key.startswith("public/"):
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            "Public delete is only allowed for objects under the `public/` prefix "
            "(URLs from the public upload-file flow)."
        ),
    )


@router.post(
    "/r2/delete-by-public-url",
    response_model=DeleteR2ByPublicUrlResponse,
    summary="Delete R2 object by public URL",
    description=(
        "Public endpoint: **no authentication**. Deletes a single file from Cloudflare R2 when you pass "
        "its **public HTTPS URL** (must match `R2_PUBLIC_URL`). Only object keys under the **`public/`** "
        "prefix may be removed (same scope as anonymous uploads to `public/`)."
    ),
)
async def delete_r2_by_public_url(body: DeleteR2ByPublicUrlRequest) -> DeleteR2ByPublicUrlResponse:
    try:
        key = public_url_to_object_key(body.public_url)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    _assert_public_delete_key_allowed(key)

    await delete_object_by_key(key)
    logger.info("R2 public delete key=%s", key)

    return DeleteR2ByPublicUrlResponse(
        success=True,
        message="Object deleted (or did not exist).",
        object_key=key,
    )
