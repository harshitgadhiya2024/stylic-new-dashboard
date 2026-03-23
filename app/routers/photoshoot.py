import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.database import (
    get_photoshoots_collection,
    get_upscaling_collection,
    get_users_collection,
    get_credit_history_collection,
)
from app.dependencies import get_current_user
from app.models.photoshoot import CreatePhotoshootRequest, UpscalePhotoshootRequest
from app.services.photoshoot_service import run_photoshoot_job

router = APIRouter(prefix="/api/v1/photoshoots", tags=["Photoshoots"])

_CREDIT_PER_POSE   = 2.0
_CREDIT_UPSCALE_2X = 2.0
_CREDIT_UPSCALE_4X = 4.0


@router.post(
    "/",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create Photoshoot",
    description=(
        "Validates credits (total_poses × 2), stores the photoshoot document with "
        "status='processing', then fires a background job that: resolves pose prompts, "
        "runs all poses concurrently via SeedDream (quality=high, 9:16), generates 4K/2K/1K "
        "images for each pose, uploads all to S3, deducts credits, "
        "and updates the document to status='completed' (or 'failed'). "
        "Secured — user_id is taken from the auth token."
    ),
)
async def create_photoshoot(
    body: CreatePhotoshootRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]

    # ── Step 1: resolve total poses count ────────────────────────────────────
    if body.which_pose_option == "default":
        total_poses = len(body.poses_ids or [])
    elif body.which_pose_option == "custom":
        total_poses = len(body.poses_images or [])
    else:
        total_poses = len(body.poses_prompts or [])

    if total_poses == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"No poses provided for which_pose_option='{body.which_pose_option}'.",
        )

    # ── Step 2: credit check ──────────────────────────────────────────────────
    total_credit    = total_poses * _CREDIT_PER_POSE
    current_credits = float(current_user.get("credits", 0))

    if current_credits < total_credit:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"Insufficient credits. This photoshoot requires {total_credit} credits "
                f"({total_poses} pose(s) × {_CREDIT_PER_POSE}) but you only have {current_credits}."
            ),
        )

    # ── Step 3: store photoshoot document ─────────────────────────────────────
    photoshoot_id = str(uuid.uuid4())
    now           = datetime.now(timezone.utc)

    doc = {
        "photoshoot_id":    photoshoot_id,
        "user_id":          user_id,
        "sku_id":           body.sku_id or "",
        "input_parameter": {
            "front_garment_image":             body.front_garment_image,
            "back_garment_image":              body.back_garment_image or "",
            "ethnicity":                       body.ethnicity,
            "gender":                          body.gender,
            "skin_tone":                       body.skin_tone,
            "age":                             body.age,
            "age_group":                       body.age_group,
            "weight":                          body.weight,
            "height":                          body.height,
            "upper_garment_type":              body.upper_garment_type,
            "upper_garment_specification":     body.upper_garment_specification,
            "lower_garment_type":              body.lower_garment_type,
            "lower_garment_specification":     body.lower_garment_specification,
            "one_piece_garment_type":          body.one_piece_garment_type,
            "one_piece_garment_specification": body.one_piece_garment_specification,
            "fitting":                         body.fitting,
            "background_id":                   body.background_id,
            "which_pose_option":               body.which_pose_option,
            "poses_ids":                       body.poses_ids or [],
            "poses_images":                    body.poses_images or [],
            "poses_prompts":                   body.poses_prompts or [],
            "model_id":                        body.model_id,
            "lighting_style":                  body.lighting_style,
            "ornaments":                       body.ornaments or "",
            "regeneration_type":               body.regeneration_type or "",
            "regenerate_photoshoot_id":        body.regenerate_photoshoot_id or "",
        },
        "output_images":             [],
        "failed_poses":              [],
        "total_credit":              total_credit,
        "is_credit_deducted":        False,
        "is_completed":              False,
        "status":                    "processing",
        "error":                     None,
        "regeneration_type":         body.regeneration_type or "",
        "regenerate_photoshoot_id":  body.regenerate_photoshoot_id or "",
        "created_at":                now,
        "updated_at":                now,
    }

    col = get_photoshoots_collection()
    await col.insert_one(doc)

    # ── Step 4: fire background job ───────────────────────────────────────────
    job_payload = {
        **doc["input_parameter"],
        "user_id": user_id,
    }
    background_tasks.add_task(run_photoshoot_job, photoshoot_id, job_payload)

    return {
        "message":        "Photoshoot started successfully. Processing in background.",
        "photoshoot_id":  photoshoot_id,
        "total_poses":    total_poses,
        "total_credit":   total_credit,
        "status":         "processing",
    }


@router.post(
    "/upscale",
    status_code=status.HTTP_200_OK,
    summary="Upscale Photoshoot",
    description=(
        "Given a list of image_ids from an existing photoshoot, fetches the already-upscaled "
        "images from upscaling_data, copies the original photoshoot document into a new one "
        "with updated output_images, deducts credits (2× → 2 credits/image, 4× → 4 credits/image), "
        "and returns the new photoshoot immediately as completed. "
        "Secured — user_id is taken from the auth token."
    ),
)
async def upscale_photoshoot(
    body: UpscalePhotoshootRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    now     = datetime.now(timezone.utc)

    # ── Step 1: validate image_ids ────────────────────────────────────────────
    if not body.image_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="image_ids must contain at least one image_id.",
        )

    # ── Step 2: credit check ──────────────────────────────────────────────────
    is_4x          = "4x" in body.regeneration_type
    credit_per_img = _CREDIT_UPSCALE_4X if is_4x else _CREDIT_UPSCALE_2X
    total_credit   = credit_per_img * len(body.image_ids)
    current_credits = float(current_user.get("credits", 0))

    if current_credits < total_credit:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"Insufficient credits. This upscale requires {total_credit} credits "
                f"({len(body.image_ids)} image(s) × {credit_per_img}) "
                f"but you only have {current_credits}."
            ),
        )

    # ── Step 3: fetch original photoshoot ─────────────────────────────────────
    ps_col      = get_photoshoots_collection()
    original_ps = await ps_col.find_one({"photoshoot_id": body.photoshoot_id})
    if not original_ps:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photoshoot '{body.photoshoot_id}' not found.",
        )

    # verify ownership
    if original_ps.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this photoshoot.",
        )

    # ── Step 4: fetch upscaling_data for each requested image_id ─────────────
    up_col = get_upscaling_collection()

    # build a map: image_id → upscaling doc
    upscaling_docs = await up_col.find(
        {"photoshoot_id": body.photoshoot_id, "image_id": {"$in": body.image_ids}}
    ).to_list(length=None)
    upscaling_map = {doc["image_id"]: doc for doc in upscaling_docs}

    # build pose_prompt map from original output_images
    pose_prompt_map = {
        img["image_id"]: img.get("pose_prompt", "")
        for img in original_ps.get("output_images", [])
    }

    # ── Step 5: build output_images with upscaled URLs ────────────────────────
    output_images = []
    for image_id in body.image_ids:
        up_doc = upscaling_map.get(image_id)
        if up_doc:
            if is_4x:
                image_url = up_doc.get("4k_upscaled") or up_doc.get("4k", "")
            else:
                image_url = up_doc.get("2k_upscaled") or up_doc.get("2k", "")
        else:
            # fallback: find the original url from output_images
            image_url = next(
                (img["image"] for img in original_ps.get("output_images", [])
                 if img["image_id"] == image_id),
                "",
            )

        output_images.append({
            "image_id":    image_id,
            "pose_prompt": pose_prompt_map.get(image_id, ""),
            "image":       image_url,
        })

    # ── Step 6: build new photoshoot document (copy + override) ──────────────
    new_photoshoot_id = str(uuid.uuid4())

    new_doc = {
        k: v for k, v in original_ps.items()
        if k not in ("_id", "photoshoot_id", "output_images", "failed_poses",
                     "total_credit", "is_credit_deducted", "is_completed",
                     "status", "error", "regeneration_type",
                     "regenerate_photoshoot_id", "created_at", "updated_at")
    }
    new_doc.update({
        "photoshoot_id":           new_photoshoot_id,
        "output_images":           output_images,
        "failed_poses":            [],
        "total_credit":            total_credit,
        "is_credit_deducted":      True,
        "is_completed":            True,
        "status":                  "completed",
        "error":                   None,
        "regeneration_type":       body.regeneration_type,
        "regenerate_photoshoot_id": body.photoshoot_id,
        "created_at":              now,
        "updated_at":              now,
    })

    await ps_col.insert_one(new_doc)

    # ── Step 7: deduct credits ────────────────────────────────────────────────
    users_col   = get_users_collection()
    history_col = get_credit_history_collection()

    new_credits = round(current_credits - total_credit, 4)
    await users_col.update_one(
        {"user_id": user_id},
        {"$set": {"credits": new_credits, "updated_at": now}},
    )
    await history_col.insert_one({
        "history_id":     str(uuid.uuid4()),
        "user_id":        user_id,
        "feature_name":   "photoshoot_upscale",
        "credit":         total_credit,
        "type":           "deduct",
        "thumbnail_image": "",
        "notes":          f"Upscale ({body.regeneration_type}) — new photoshoot {new_photoshoot_id}",
        "created_at":     now,
    })

    return {
        "message":                  "Photoshoot upscaled successfully.",
        "photoshoot_id":            new_photoshoot_id,
        "regenerate_photoshoot_id": body.photoshoot_id,
        "regeneration_type":        body.regeneration_type,
        "total_images":             len(output_images),
        "total_credit":             total_credit,
        "status":                   "completed",
        "output_images":            output_images,
    }
