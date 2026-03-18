import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.database import get_photoshoots_collection, get_users_collection
from app.dependencies import get_current_user
from app.models.photoshoot import CreatePhotoshootRequest
from app.services.photoshoot_service import run_photoshoot_job

router = APIRouter(prefix="/api/v1/photoshoots", tags=["Photoshoots"])

_CREDIT_PER_POSE = 2.0


@router.post(
    "/",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create Photoshoot",
    description=(
        "Validates credits (total_poses × 2), stores the photoshoot document with "
        "status='processing', then fires a background job that: resolves pose prompts, "
        "runs all poses concurrently via SeedDream (quality=high, 9:16), generates 4K/2K/1K "
        "images with deblurred variants for each pose, uploads all to S3, deducts credits, "
        "and updates the document to status='completed' (or 'failed'). "
        "Secured — user_id is taken from the auth token."
    ),
)
def create_photoshoot(
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
    total_credit   = total_poses * _CREDIT_PER_POSE
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
            "front_garment_image":            body.front_garment_image,
            "back_garment_image":             body.back_garment_image or "",
            "ethnicity":                      body.ethnicity,
            "gender":                         body.gender,
            "skin_tone":                      body.skin_tone,
            "age":                            body.age,
            "age_group":                      body.age_group,
            "weight":                         body.weight,
            "height":                         body.height,
            "upper_garment_type":             body.upper_garment_type,
            "upper_garment_specification":    body.upper_garment_specification,
            "lower_garment_type":             body.lower_garment_type,
            "lower_garment_specification":    body.lower_garment_specification,
            "one_piece_garment_type":         body.one_piece_garment_type,
            "one_piece_garment_specification": body.one_piece_garment_specification,
            "fitting":                        body.fitting,
            "background_id":                  body.background_id,
            "which_pose_option":              body.which_pose_option,
            "poses_ids":                      body.poses_ids or [],
            "poses_images":                   body.poses_images or [],
            "poses_prompts":                  body.poses_prompts or [],
            "model_id":                       body.model_id,
            "lighting_style":                 body.lighting_style,
            "ornaments":                      body.ornaments or "",
        },
        "output_images":      [],
        "failed_poses":       [],
        "total_credit":       total_credit,
        "is_credit_deducted": False,
        "is_completed":       False,
        "status":             "processing",
        "error":              None,
        "created_at":         now,
        "updated_at":         now,
    }

    col = get_photoshoots_collection()
    col.insert_one(doc)

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
