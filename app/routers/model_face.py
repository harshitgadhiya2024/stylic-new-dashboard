import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from app.database import get_model_faces_collection
from app.dependencies import get_current_user
from app.models.model_face import (
    CreateModelFaceRequest,
    CreateModelFaceWithAIRequest,
    ModelFaceSchema,
)
from app.services.ai_face_service import generate_and_upload_face

router = APIRouter(prefix="/api/v1/model-faces", tags=["Model Faces"])


def _clean_face(doc: dict) -> dict:
    doc = dict(doc)
    doc.pop("_id", None)
    return doc


@router.post(
    "/",
    response_model=ModelFaceSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Upload / Create a Model Face",
    description="Create a new model face entry. Secured — user_id is taken from the auth token.",
)
def create_model_face(
    body: CreateModelFaceRequest,
    current_user: dict = Depends(get_current_user),
):
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
        "face_url":            body.face_url,
        "is_default":          False,
        "is_active":           True,
        "is_favorite":         False,
        "created_at":          now,
        "updated_at":          now,
    }

    col = get_model_faces_collection()

    try:
        col.insert_one(doc)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save model face: {exc}",
        )

    return _clean_face(doc)


@router.post(
    "/generate-with-ai",
    response_model=ModelFaceSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Create Model Face Using AI",
    description=(
        "Generate a realistic portrait image via Gemini AI using the provided face "
        "configurations (all optional — unset fields use category-appropriate defaults), "
        "upload it to S3, and save the result in the database. "
        "beard_length and beard_color are only applied when model_category is 'adult_male'."
    ),
)
def create_model_face_with_ai(
    body: CreateModelFaceWithAIRequest,
    current_user: dict = Depends(get_current_user),
):
    # Convert optional nested model to plain dict of only the explicitly passed fields
    overrides = {}
    if body.face_configurations:
        overrides = {
            k: v
            for k, v in body.face_configurations.model_dump().items()
            if v is not None
        }

    # Generate image via Gemini and upload to S3
    face_url, final_config = generate_and_upload_face(body.model_category, overrides)

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
        "face_url":            face_url,
        "is_default":          False,
        "is_active":           True,
        "is_favorite":         False,
        "created_at":          now,
        "updated_at":          now,
    }

    col = get_model_faces_collection()
    try:
        col.insert_one(doc)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save model face: {exc}",
        )

    return _clean_face(doc)


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
