"""
Credit service.

Handles:
  - Checking whether a user has enough credits before generation.
  - Creating a 100×100 thumbnail from a generated face image URL and uploading to S3.
  - Deducting credits from the user document after successful generation.
  - Inserting a record into the credit_history collection.
"""

import io
import uuid
from datetime import datetime, timezone

import requests
from fastapi import HTTPException, status

from app.database import get_users_collection, get_credit_history_collection
from app.services.s3_service import upload_bytes_to_s3

FACE_GENERATION_COST: float = 2.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_thumbnail(image_url: str) -> bytes:
    """Download image_url and return 100×100 JPEG thumbnail bytes."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing dependency for thumbnail: {exc}. Run: pip install Pillow",
        )

    try:
        resp = requests.get(image_url, timeout=30)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.thumbnail((100, 100))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create thumbnail: {exc}",
        )


def _upload_thumbnail(image_url: str, user_id: str) -> str:
    """Create thumbnail from image_url, upload to S3, return public URL."""
    thumb_bytes = _make_thumbnail(image_url)
    key = f"thumbnails/{user_id}_{uuid.uuid4().hex[:8]}.jpg"
    return upload_bytes_to_s3(thumb_bytes, key, content_type="image/jpeg")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_sufficient_credits(user: dict) -> None:
    """
    Raise HTTP 402 if the user does not have enough credits for face generation.
    """
    current = float(user.get("credits", 0))
    if current < FACE_GENERATION_COST:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"Insufficient credits. Face generation costs {FACE_GENERATION_COST} credits "
                f"but you only have {current}."
            ),
        )


def deduct_credits_and_record(
    user: dict,
    feature_name: str,
    generated_face_url: str,
    notes: str = "",
) -> None:
    """
    After successful face generation:
      1. Create a thumbnail from generated_face_url and upload to S3.
      2. Deduct FACE_GENERATION_COST from user.credits in the users collection.
      3. Insert a record into credit_history.

    Raises HTTPException on any failure.
    """
    user_id = user["user_id"]
    current_credits = float(user.get("credits", 0))
    new_credits = round(current_credits - FACE_GENERATION_COST, 4)

    # 1 — thumbnail
    thumbnail_url = _upload_thumbnail(generated_face_url, user_id)

    # 2 — deduct from user
    users_col = get_users_collection()
    result = users_col.update_one(
        {"user_id": user_id},
        {
            "$set": {
                "credits":    new_credits,
                "updated_at": datetime.now(timezone.utc),
            }
        },
    )
    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found while deducting credits.",
        )

    # 3 — insert credit history record
    history_col = get_credit_history_collection()
    history_doc = {
        "history_id":       str(uuid.uuid4()),
        "user_id":          user_id,
        "feature_name":     feature_name,
        "credit":           FACE_GENERATION_COST,
        "type":             "deduct",
        "thumbnail_image":  thumbnail_url,
        "notes":            notes,
        "created_at":       datetime.now(timezone.utc),
    }
    try:
        history_col.insert_one(history_doc)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record credit history: {exc}",
        )
