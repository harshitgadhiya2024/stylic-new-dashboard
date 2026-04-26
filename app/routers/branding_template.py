import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.database import get_templates_collection
from app.dependencies import get_current_user
from app.models.template import (
    CreateBrandingTemplateRequest,
    SetDefaultBrandingTemplateRequest,
    SoftDeleteBrandingTemplateRequest,
    UpdateBrandingTemplateRequest,
)

router = APIRouter(prefix="/api/v1/branding-templates", tags=["Branding templates"])


def _clean_template_doc(doc: dict) -> dict:
    out = dict(doc)
    out.pop("_id", None)
    return out


def _require_matching_user(current_user: dict, user_id: str) -> None:
    if current_user.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="user_id does not match the authenticated user.",
        )


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Create branding template",
    description=(
        "Creates a row in the `templates` collection. "
        "`logo_position` and `serial_code_position` must be one of the nine grid positions. "
        "Secured — `user_id` is taken from the auth token."
    ),
)
async def create_branding_template(
    body: CreateBrandingTemplateRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    now = datetime.now(timezone.utc)
    template_id = str(uuid.uuid4())

    doc = {
        "template_id":          template_id,
        "user_id":              user_id,
        "template_name":        body.template_name,
        "logo_image_url":       body.logo_image_url,
        "serial_code_format":   body.serial_code_format,
        "logo_position":        body.logo_position,
        "serial_code_position": body.serial_code_position,
        "font_size":            body.font_size,
        "logo_size":            body.logo_size,
        "is_default":           False,
        "is_active":            True,
        "created_at":           now,
        "updated_at":           now,
    }

    col = get_templates_collection()
    await col.insert_one(doc)

    return _clean_template_doc(doc)


@router.patch(
    "/",
    summary="Update branding template",
    description=(
        "Partial update: only fields present in the body are written. "
        "Requires `template_id` and `user_id`; `user_id` must match the auth token."
    ),
)
async def update_branding_template(
    body: UpdateBrandingTemplateRequest,
    current_user: dict = Depends(get_current_user),
):
    _require_matching_user(current_user, body.user_id)

    update_fields = body.model_dump(exclude={"template_id", "user_id"}, exclude_none=True)
    if not update_fields:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No fields to update. Provide at least one mutable field.",
        )

    now = datetime.now(timezone.utc)
    col = get_templates_collection()

    existing = await col.find_one({"template_id": body.template_id, "user_id": body.user_id})
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Template not found.",
        )

    await col.update_one(
        {"template_id": body.template_id, "user_id": body.user_id},
        {"$set": {**update_fields, "updated_at": now}},
    )

    updated = await col.find_one({"template_id": body.template_id, "user_id": body.user_id})
    return _clean_template_doc(updated)


@router.post(
    "/soft-delete",
    summary="Soft delete branding template",
    description="Sets `is_active=False` for the template. `user_id` must match the auth token.",
)
async def soft_delete_branding_template(
    body: SoftDeleteBrandingTemplateRequest,
    current_user: dict = Depends(get_current_user),
):
    _require_matching_user(current_user, body.user_id)

    col = get_templates_collection()
    now = datetime.now(timezone.utc)

    result = await col.update_one(
        {"template_id": body.template_id, "user_id": body.user_id, "is_active": True},
        {"$set": {"is_active": False, "updated_at": now}},
    )

    if result.matched_count == 0:
        doc = await col.find_one({"template_id": body.template_id, "user_id": body.user_id})
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found.",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Template is already inactive.",
        )

    updated = await col.find_one({"template_id": body.template_id, "user_id": body.user_id})
    return _clean_template_doc(updated)


@router.get(
    "/",
    summary="List branding templates for user",
    description="Returns all template documents for `user_id` (including inactive). `user_id` must match the auth token.",
)
async def list_branding_templates(
    user_id: str = Query(..., description="Owner user id (must match token)"),
    current_user: dict = Depends(get_current_user),
):
    _require_matching_user(current_user, user_id)

    col = get_templates_collection()
    cursor = col.find({"user_id": user_id}).sort("created_at", -1)
    docs = await cursor.to_list(length=None)
    return [_clean_template_doc(d) for d in docs]


@router.get(
    "/{template_id}",
    summary="Get branding template detail",
    description="Returns one template. `user_id` query param must match the auth token.",
)
async def get_branding_template(
    template_id: str,
    user_id: str = Query(..., description="Owner user id (must match token)"),
    current_user: dict = Depends(get_current_user),
):
    _require_matching_user(current_user, user_id)

    col = get_templates_collection()
    doc = await col.find_one({"template_id": template_id, "user_id": user_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Template not found.",
        )
    return _clean_template_doc(doc)


@router.post(
    "/set-default",
    summary="Set default branding template",
    description=(
        "Sets `is_default=True` on the given template and `is_default=False` on all other "
        "templates for that user. `user_id` must match the auth token."
    ),
)
async def set_default_branding_template(
    body: SetDefaultBrandingTemplateRequest,
    current_user: dict = Depends(get_current_user),
):
    _require_matching_user(current_user, body.user_id)

    col = get_templates_collection()
    now = datetime.now(timezone.utc)

    target = await col.find_one(
        {"template_id": body.template_id, "user_id": body.user_id, "is_active": True},
    )
    if not target:
        missing = await col.find_one({"template_id": body.template_id, "user_id": body.user_id})
        if not missing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found.",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot set default on an inactive template.",
        )

    await col.update_many(
        {"user_id": body.user_id},
        {"$set": {"is_default": False, "updated_at": now}},
    )
    await col.update_one(
        {"template_id": body.template_id, "user_id": body.user_id},
        {"$set": {"is_default": True, "updated_at": now}},
    )

    updated = await col.find_one({"template_id": body.template_id, "user_id": body.user_id})
    return _clean_template_doc(updated)
