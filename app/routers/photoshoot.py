import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import ValidationError

from app.database import (
    get_photoshoots_collection,
    get_upscaling_collection,
    get_users_collection,
    get_credit_history_collection,
    get_backgrounds_collection,
)
from app.dependencies import get_current_user
from app.models.photoshoot import (
    CreatePhotoshootRequest,
    CreateMultiplePhotoshootsRequest,
    UpscalePhotoshootRequest,
    RegeneratePhotoshootRequest,
    DeletePhotoshootsRequest,
    ResizePhotoshootRequest,
    BrandingPhotoshootRequest,
    BackgroundChangeRequest,
    FabricChangeRequest,
    FabricPhotoshootRequest,
    TextureChangeRequest,
    TexturePhotoshootRequest,
    ColorChangeRequest,
    ColorPhotoshootRequest,
)
from app.tasks.photoshoot_tasks import run_photoshoot_task
from app.services.photoshoot_service import merge_photoshoot_batch_configs, count_poses_in_merged_config

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
        "is_active":                 True,
        "created_at":                now,
        "updated_at":                now,
    }

    col = get_photoshoots_collection()
    await col.insert_one(doc)

    # ── Step 4: enqueue background job via Celery ─────────────────────────────
    job_payload = {
        **doc["input_parameter"],
        "user_id": user_id,
    }
    task = run_photoshoot_task.apply_async(
        args=[photoshoot_id, job_payload],
        queue="photoshoots",
    )

    return {
        "message":        "Photoshoot started successfully. Processing in background.",
        "photoshoot_id":  photoshoot_id,
        "task_id":        task.id,
        "total_poses":    total_poses,
        "total_credit":   total_credit,
        "status":         "processing",
    }


@router.post(
    "/batch",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create multiple photoshoots (batch)",
    description=(
        "Merges each entry in ``photoshoot_list_config`` with ``default_config`` (per-field overrides), "
        "validates the result as a full photoshoot payload, checks credits for the entire batch, "
        "then creates one photoshoot document and one Celery job per row — same pipeline as POST /."
    ),
)
async def create_multiple_photoshoots(
    body: CreateMultiplePhotoshootsRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    batch_photoshoot_id = str(uuid.uuid4())
    defaults_dump = body.default_config.model_dump(exclude_none=True)

    validated: list[tuple[int, CreatePhotoshootRequest, int]] = []

    for idx, item in enumerate(body.photoshoot_list_config):
        merged = merge_photoshoot_batch_configs(
            defaults_dump,
            item.model_dump(exclude_none=True),
        )
        try:
            row_req = CreatePhotoshootRequest.model_validate(merged)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": f"Merged config invalid at photoshoot_list_config index {idx}",
                    "index":   idx,
                    "errors":  exc.errors(),
                },
            ) from exc

        n_poses = count_poses_in_merged_config(merged)
        if n_poses == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"No poses resolved at photoshoot_list_config index {idx} "
                    f"(which_pose_option='{row_req.which_pose_option}')."
                ),
            )
        validated.append((idx, row_req, n_poses))

    total_credit_batch = sum(n * _CREDIT_PER_POSE for _, _, n in validated)
    current_credits    = float(current_user.get("credits", 0))

    if current_credits < total_credit_batch:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"Insufficient credits. This batch requires {total_credit_batch} credits "
                f"but you only have {current_credits}."
            ),
        )

    col = get_photoshoots_collection()
    now = datetime.now(timezone.utc)
    results: list[dict] = []

    for idx, row_req, total_poses in validated:
        photoshoot_id = str(uuid.uuid4())
        total_credit    = total_poses * _CREDIT_PER_POSE

        doc = {
            "photoshoot_id":       photoshoot_id,
            "user_id":             user_id,
            "batch_photoshoot_id": batch_photoshoot_id,
            "batch_index":         idx,
            "sku_id":              row_req.sku_id or "",
            "input_parameter": {
                "front_garment_image":             row_req.front_garment_image,
                "back_garment_image":              row_req.back_garment_image or "",
                "ethnicity":                       row_req.ethnicity,
                "gender":                          row_req.gender,
                "skin_tone":                       row_req.skin_tone,
                "age":                             row_req.age,
                "age_group":                       row_req.age_group,
                "weight":                          row_req.weight,
                "height":                          row_req.height,
                "upper_garment_type":              row_req.upper_garment_type,
                "upper_garment_specification":     row_req.upper_garment_specification,
                "lower_garment_type":              row_req.lower_garment_type,
                "lower_garment_specification":     row_req.lower_garment_specification,
                "one_piece_garment_type":          row_req.one_piece_garment_type,
                "one_piece_garment_specification": row_req.one_piece_garment_specification,
                "fitting":                         row_req.fitting,
                "background_id":                   row_req.background_id,
                "which_pose_option":               row_req.which_pose_option,
                "poses_ids":                       row_req.poses_ids or [],
                "poses_images":                    row_req.poses_images or [],
                "poses_prompts":                   row_req.poses_prompts or [],
                "model_id":                        row_req.model_id,
                "lighting_style":                  row_req.lighting_style,
                "ornaments":                       row_req.ornaments or "",
                "regeneration_type":               row_req.regeneration_type or "",
                "regenerate_photoshoot_id":        row_req.regenerate_photoshoot_id or "",
                "batch_photoshoot_id":             batch_photoshoot_id,
                "batch_index":                     idx,
            },
            "output_images":             [],
            "failed_poses":              [],
            "total_credit":              total_credit,
            "is_credit_deducted":      False,
            "is_completed":              False,
            "status":                    "processing",
            "error":                     None,
            "regeneration_type":         row_req.regeneration_type or "",
            "regenerate_photoshoot_id":  row_req.regenerate_photoshoot_id or "",
            "is_active":                 True,
            "created_at":                now,
            "updated_at":                now,
        }

        await col.insert_one(doc)

        job_payload = {
            **doc["input_parameter"],
            "user_id": user_id,
        }
        task = run_photoshoot_task.apply_async(
            args=[photoshoot_id, job_payload],
            queue="photoshoots",
        )

        results.append({
            "batch_index":    idx,
            "photoshoot_id":  photoshoot_id,
            "task_id":        task.id,
            "total_poses":    total_poses,
            "total_credit":   total_credit,
            "status":         "processing",
        })

    return {
        "message":             "Batch photoshoots started successfully. Each job runs in background.",
        "batch_photoshoot_id": batch_photoshoot_id,
        "total_photoshoots":   len(results),
        "total_credit":        total_credit_batch,
        "photoshoots":         results,
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
        "is_active":               True,
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
        "history_id":               str(uuid.uuid4()),
        "user_id":                  user_id,
        "feature_name":             "photoshoot_upscale",
        "credit":                   total_credit,
        "credit_per_image":         credit_per_img,
        "image_ids":                body.image_ids,
        "type":                     "deduct",
        "thumbnail_image":          "",
        "notes":                    f"Upscale ({body.regeneration_type}) — new photoshoot {new_photoshoot_id}",
        "photoshoot_id":            new_photoshoot_id,
        "regeneration_type":        body.regeneration_type,
        "regenerate_photoshoot_id": body.photoshoot_id,
        "created_at":               now,
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


@router.post(
    "/regenerate",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Regenerate Photoshoot",
    description=(
        "Re-runs the full SeedDream + Modal pipeline for selected poses of an existing "
        "photoshoot using the same input_parameter configuration. "
        "If image_ids is empty, all poses are regenerated; otherwise only the specified ones. "
        "Credits: 2 per regenerated image. Secured — user_id from auth token."
    ),
)
async def regenerate_photoshoot(
    body: RegeneratePhotoshootRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    now     = datetime.now(timezone.utc)

    # ── Step 1: fetch original photoshoot ─────────────────────────────────────
    ps_col      = get_photoshoots_collection()
    original_ps = await ps_col.find_one({"photoshoot_id": body.photoshoot_id})
    if not original_ps:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photoshoot '{body.photoshoot_id}' not found.",
        )

    if original_ps.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this photoshoot.",
        )

    # ── Step 2: resolve pose_prompts from original output_images ──────────────
    all_output_images = original_ps.get("output_images", [])
    if not all_output_images:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Original photoshoot has no output_images to regenerate from.",
        )

    requested_ids = body.image_ids or []
    if requested_ids:
        # only regenerate poses matching the given image_ids, preserving order
        id_set = set(requested_ids)
        selected = [img for img in all_output_images if img["image_id"] in id_set]
        if not selected:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="None of the provided image_ids were found in this photoshoot.",
            )
    else:
        # regenerate all poses
        selected = all_output_images

    pose_prompts = [img["pose_prompt"] for img in selected]
    total_poses  = len(pose_prompts)

    # ── Step 3: credit check ──────────────────────────────────────────────────
    total_credit    = total_poses * _CREDIT_PER_POSE
    current_credits = float(current_user.get("credits", 0))

    if current_credits < total_credit:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"Insufficient credits. This regeneration requires {total_credit} credits "
                f"({total_poses} pose(s) × {_CREDIT_PER_POSE}) but you only have {current_credits}."
            ),
        )

    # ── Step 4: build job payload from original input_parameter ───────────────
    original_params = original_ps.get("input_parameter", {})
    new_photoshoot_id = str(uuid.uuid4())

    # Override poses to use the resolved pose_prompts directly (prompt mode)
    job_payload = {
        **original_params,
        "user_id":                   user_id,
        "which_pose_option":         "prompt",
        "poses_prompts":             pose_prompts,
        "poses_ids":                 [],
        "poses_images":              [],
        "regeneration_type":         "regenerate",
        "regenerate_photoshoot_id":  body.photoshoot_id,
    }

    # ── Step 5: store new photoshoot document (processing state) ──────────────
    doc = {
        "photoshoot_id":            new_photoshoot_id,
        "user_id":                  user_id,
        "sku_id":                   original_ps.get("sku_id", ""),
        "input_parameter":          {
            **original_params,
            "which_pose_option":        "prompt",
            "poses_prompts":            pose_prompts,
            "poses_ids":                [],
            "poses_images":             [],
            "regeneration_type":        "regenerate",
            "regenerate_photoshoot_id": body.photoshoot_id,
        },
        "output_images":            [],
        "failed_poses":             [],
        "total_credit":             total_credit,
        "is_credit_deducted":       False,
        "is_completed":             False,
        "status":                   "processing",
        "error":                    None,
        "regeneration_type":        "regenerate",
        "regenerate_photoshoot_id": body.photoshoot_id,
        "is_active":                True,
        "created_at":               now,
        "updated_at":               now,
    }
    await ps_col.insert_one(doc)

    # ── Step 6: enqueue background job via Celery ─────────────────────────────
    task = run_photoshoot_task.apply_async(
        args=[new_photoshoot_id, job_payload],
        queue="photoshoots",
    )

    return {
        "message":                  "Photoshoot regeneration started successfully. Processing in background.",
        "photoshoot_id":            new_photoshoot_id,
        "task_id":                  task.id,
        "regenerate_photoshoot_id": body.photoshoot_id,
        "regeneration_type":        "regenerate",
        "total_poses":              total_poses,
        "total_credit":             total_credit,
        "status":                   "processing",
    }


@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Get All Photoshoots",
    description=(
        "Returns paginated photoshoots belonging to the authenticated user, "
        "sorted by created_at descending (newest first). "
        "Use `page` (1-based) and `page_size` query params to control pagination. "
        "Secured — user_id is taken from the auth token."
    ),
)
async def get_all_photoshoots(
    current_user: dict = Depends(get_current_user),
    page:      int = Query(default=1,  ge=1,  description="Page number (1-based)"),
    page_size: int = Query(default=10, ge=1, le=100, description="Number of records per page (max 100)"),
):
    user_id = current_user["user_id"]
    col     = get_photoshoots_collection()

    query = {"user_id": user_id, "is_active": True}

    total       = await col.count_documents(query)
    skip        = (page - 1) * page_size
    total_pages = (total + page_size - 1) // page_size

    photoshoots = await col.find(
        query,
        {"_id": 0},
    ).sort("created_at", -1).skip(skip).limit(page_size).to_list(length=None)

    return {
        "total":       total,
        "page":        page,
        "page_size":   page_size,
        "total_pages": total_pages,
        "has_next":    page < total_pages,
        "has_prev":    page > 1,
        "photoshoots": photoshoots,
    }


@router.get(
    "/detail",
    status_code=status.HTTP_200_OK,
    summary="Get Photoshoot Detail",
    description=(
        "Returns full details of a single photoshoot by photoshoot_id. "
        "Only the authenticated user's own photoshoot is accessible. "
        "Secured — user_id is taken from the auth token."
    ),
)
async def get_photoshoot_detail(
    photoshoot_id: str = Query(..., description="ID of the photoshoot to fetch"),
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    col     = get_photoshoots_collection()

    photoshoot = await col.find_one(
        {"photoshoot_id": photoshoot_id, "user_id": user_id},
        {"_id": 0},
    )

    if not photoshoot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photoshoot '{photoshoot_id}' not found.",
        )

    return photoshoot


@router.patch(
    "/delete",
    status_code=status.HTTP_200_OK,
    summary="Soft Delete Photoshoots",
    description=(
        "Sets is_active=False on the given list of photoshoot_ids for the authenticated user. "
        "Soft-deleted photoshoots are excluded from the get-all listing. "
        "Secured — user_id is taken from the auth token."
    ),
)
async def soft_delete_photoshoots(
    body: DeletePhotoshootsRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]

    if not body.photoshoot_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="photoshoot_ids must contain at least one id.",
        )

    col    = get_photoshoots_collection()
    now    = datetime.now(timezone.utc)

    result = await col.update_many(
        {
            "photoshoot_id": {"$in": body.photoshoot_ids},
            "user_id":       user_id,
        },
        {"$set": {"is_active": False, "updated_at": now}},
    )

    return {
        "message":        f"{result.modified_count} photoshoot(s) deleted successfully.",
        "deleted_count":  result.modified_count,
        "photoshoot_ids": body.photoshoot_ids,
    }


@router.post(
    "/resize",
    status_code=status.HTTP_200_OK,
    summary="Resize Photoshoot",
    description=(
        "Copies an existing photoshoot into a new document, replacing the image URL "
        "for each matching image_id with the one provided in resize_list. "
        "Sets regeneration_type='resize' and regenerate_photoshoot_id to the original "
        "photoshoot_id. No credits are deducted. "
        "Secured — user_id is taken from the auth token."
    ),
)
async def resize_photoshoot(
    body: ResizePhotoshootRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]

    if not body.resize_list:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="resize_list must contain at least one item.",
        )

    # ── Step 1: fetch original photoshoot ─────────────────────────────────────
    ps_col      = get_photoshoots_collection()
    original_ps = await ps_col.find_one({"photoshoot_id": body.photoshoot_id, "user_id": user_id})
    if not original_ps:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photoshoot '{body.photoshoot_id}' not found.",
        )

    # ── Step 2: build updated output_images ───────────────────────────────────
    # Only include entries whose image_id is in resize_list, with the new image URL
    resize_map = {item.image_id: item.image for item in body.resize_list}

    output_images = [
        {**img, "image": resize_map[img["image_id"]]}
        for img in original_ps.get("output_images", [])
        if img["image_id"] in resize_map
    ]

    if not output_images:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="None of the provided image_ids were found in this photoshoot's output_images.",
        )

    # ── Step 3: credit check ──────────────────────────────────────────────────
    total_credit    = 1.0 * len(output_images)
    current_credits = float(current_user.get("credits", 0))

    if current_credits < total_credit:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"Insufficient credits. This resize requires {total_credit} credits "
                f"({len(output_images)} image(s) × 1) but you only have {current_credits}."
            ),
        )

    # ── Step 4: build and store new photoshoot document ───────────────────────
    new_photoshoot_id = str(uuid.uuid4())
    now               = datetime.now(timezone.utc)

    new_doc = {
        k: v for k, v in original_ps.items()
        if k not in ("_id", "photoshoot_id", "output_images", "total_credit",
                     "is_credit_deducted", "regeneration_type",
                     "regenerate_photoshoot_id", "created_at", "updated_at")
    }
    new_doc.update({
        "photoshoot_id":            new_photoshoot_id,
        "output_images":            output_images,
        "total_credit":             total_credit,
        "is_credit_deducted":       True,
        "regeneration_type":        "resize",
        "regenerate_photoshoot_id": body.photoshoot_id,
        "is_active":                True,
        "created_at":               now,
        "updated_at":               now,
    })

    await ps_col.insert_one(new_doc)

    # ── Step 5: deduct credits + record history ───────────────────────────────
    users_col   = get_users_collection()
    history_col = get_credit_history_collection()

    new_credits = round(current_credits - total_credit, 4)
    await users_col.update_one(
        {"user_id": user_id},
        {"$set": {"credits": new_credits, "updated_at": now}},
    )
    resized_image_ids = [img["image_id"] for img in output_images]
    await history_col.insert_one({
        "history_id":               str(uuid.uuid4()),
        "user_id":                  user_id,
        "feature_name":             "photoshoot_resize",
        "credit":                   total_credit,
        "credit_per_image":         1.0,
        "image_ids":                resized_image_ids,
        "type":                     "deduct",
        "thumbnail_image":          "",
        "notes":                    f"Resize — new photoshoot {new_photoshoot_id}",
        "photoshoot_id":            new_photoshoot_id,
        "regeneration_type":        "resize",
        "regenerate_photoshoot_id": body.photoshoot_id,
        "created_at":               now,
    })

    return {
        "message":                  "Photoshoot resized successfully.",
        "photoshoot_id":            new_photoshoot_id,
        "regenerate_photoshoot_id": body.photoshoot_id,
        "regeneration_type":        "resize",
        "total_credit":             total_credit,
        "output_images":            output_images,
    }


@router.post(
    "/branding",
    status_code=status.HTTP_200_OK,
    summary="Branding Photoshoot",
    description=(
        "Deducts 1 credit per new image_id for branding. "
        "If a credit history record already exists for this user + photoshoot + branding, "
        "only the image_ids not yet charged are billed and the existing record is updated. "
        "Secured — user_id is taken from the auth token."
    ),
)
async def branding_photoshoot(
    body: BrandingPhotoshootRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]

    if not body.image_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="image_ids must contain at least one id.",
        )

    # ── Step 1: verify photoshoot exists and belongs to user ──────────────────
    ps_col      = get_photoshoots_collection()
    original_ps = await ps_col.find_one({"photoshoot_id": body.photoshoot_id, "user_id": user_id})
    if not original_ps:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photoshoot '{body.photoshoot_id}' not found.",
        )

    # ── Step 2: check existing branding credit history record ─────────────────
    history_col     = get_credit_history_collection()
    existing_record = await history_col.find_one({
        "user_id":       user_id,
        "photoshoot_id": body.photoshoot_id,
        "feature_name":  "photoshoot_branding",
    })

    if existing_record:
        already_charged  = set(existing_record.get("image_ids", []))
        new_image_ids    = [iid for iid in body.image_ids if iid not in already_charged]

        if not new_image_ids:
            # All image_ids already charged — nothing to do
            return {
                "message":       "All image_ids have already been charged for branding. No credits deducted.",
                "photoshoot_id": body.photoshoot_id,
                "image_ids":     body.image_ids,
                "new_image_ids": [],
                "total_credit":  0.0,
            }

        total_credit    = 1.0 * len(new_image_ids)
        current_credits = float(current_user.get("credits", 0))

        if current_credits < total_credit:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=(
                    f"Insufficient credits. Branding requires {total_credit} credits "
                    f"({len(new_image_ids)} new image(s) × 1) but you only have {current_credits}."
                ),
            )

        # Deduct credits
        now         = datetime.now(timezone.utc)
        users_col   = get_users_collection()
        new_credits = round(current_credits - total_credit, 4)
        await users_col.update_one(
            {"user_id": user_id},
            {"$set": {"credits": new_credits, "updated_at": now}},
        )

        # Update existing credit history record with merged image_ids and updated credit
        merged_image_ids    = list(already_charged) + new_image_ids
        updated_total_credit = round(existing_record.get("credit", 0) + total_credit, 4)
        await history_col.update_one(
            {"_id": existing_record["_id"]},
            {"$set": {
                "image_ids":  merged_image_ids,
                "credit":     updated_total_credit,
                "updated_at": now,
            }},
        )

        return {
            "message":       "Branding credits deducted for new images.",
            "photoshoot_id": body.photoshoot_id,
            "image_ids":     merged_image_ids,
            "new_image_ids": new_image_ids,
            "total_credit":  total_credit,
        }

    # ── No existing record — create fresh ────────────────────────────────────
    total_credit    = 1.0 * len(body.image_ids)
    current_credits = float(current_user.get("credits", 0))

    if current_credits < total_credit:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"Insufficient credits. Branding requires {total_credit} credits "
                f"({len(body.image_ids)} image(s) × 1) but you only have {current_credits}."
            ),
        )

    now         = datetime.now(timezone.utc)
    users_col   = get_users_collection()
    new_credits = round(current_credits - total_credit, 4)
    await users_col.update_one(
        {"user_id": user_id},
        {"$set": {"credits": new_credits, "updated_at": now}},
    )
    await history_col.insert_one({
        "history_id":               str(uuid.uuid4()),
        "user_id":                  user_id,
        "feature_name":             "photoshoot_branding",
        "credit":                   total_credit,
        "credit_per_image":         1.0,
        "image_ids":                body.image_ids,
        "type":                     "deduct",
        "thumbnail_image":          "",
        "notes":                    f"Branding — photoshoot {body.photoshoot_id}",
        "photoshoot_id":            body.photoshoot_id,
        "regeneration_type":        "branding",
        "regenerate_photoshoot_id": body.photoshoot_id,
        "created_at":               now,
    })

    return {
        "message":       "Branding credits deducted successfully.",
        "photoshoot_id": body.photoshoot_id,
        "image_ids":     body.image_ids,
        "new_image_ids": body.image_ids,
        "total_credit":  total_credit,
    }


@router.post(
    "/adjust-image",
    status_code=status.HTTP_200_OK,
    summary="Adjust Image Photoshoot",
    description=(
        "Copies an existing photoshoot into a new document, replacing the image URL "
        "for each matching image_id with the one provided in resize_list. "
        "Sets regeneration_type='adjust_image'. Deducts 1 credit per image. "
        "Secured — user_id is taken from the auth token."
    ),
)
async def adjust_image_photoshoot(
    body: ResizePhotoshootRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]

    if not body.resize_list:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="resize_list must contain at least one item.",
        )

    # ── Step 1: fetch original photoshoot ─────────────────────────────────────
    ps_col      = get_photoshoots_collection()
    original_ps = await ps_col.find_one({"photoshoot_id": body.photoshoot_id, "user_id": user_id})
    if not original_ps:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photoshoot '{body.photoshoot_id}' not found.",
        )

    # ── Step 2: build updated output_images ───────────────────────────────────
    adjust_map = {item.image_id: item.image for item in body.resize_list}

    output_images = [
        {**img, "image": adjust_map[img["image_id"]]}
        for img in original_ps.get("output_images", [])
        if img["image_id"] in adjust_map
    ]

    if not output_images:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="None of the provided image_ids were found in this photoshoot's output_images.",
        )

    # ── Step 3: credit check ──────────────────────────────────────────────────
    total_credit    = 1.0 * len(output_images)
    current_credits = float(current_user.get("credits", 0))

    if current_credits < total_credit:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"Insufficient credits. Adjust image requires {total_credit} credits "
                f"({len(output_images)} image(s) × 1) but you only have {current_credits}."
            ),
        )

    # ── Step 4: build and store new photoshoot document ───────────────────────
    new_photoshoot_id = str(uuid.uuid4())
    now               = datetime.now(timezone.utc)

    new_doc = {
        k: v for k, v in original_ps.items()
        if k not in ("_id", "photoshoot_id", "output_images", "total_credit",
                     "is_credit_deducted", "regeneration_type",
                     "regenerate_photoshoot_id", "created_at", "updated_at")
    }
    new_doc.update({
        "photoshoot_id":            new_photoshoot_id,
        "output_images":            output_images,
        "total_credit":             total_credit,
        "is_credit_deducted":       True,
        "regeneration_type":        "adjust_image",
        "regenerate_photoshoot_id": body.photoshoot_id,
        "is_active":                True,
        "created_at":               now,
        "updated_at":               now,
    })

    await ps_col.insert_one(new_doc)

    # ── Step 5: deduct credits + record history ───────────────────────────────
    users_col   = get_users_collection()
    history_col = get_credit_history_collection()

    adjusted_image_ids = [img["image_id"] for img in output_images]
    new_credits        = round(current_credits - total_credit, 4)
    await users_col.update_one(
        {"user_id": user_id},
        {"$set": {"credits": new_credits, "updated_at": now}},
    )
    await history_col.insert_one({
        "history_id":               str(uuid.uuid4()),
        "user_id":                  user_id,
        "feature_name":             "photoshoot_adjust_image",
        "credit":                   total_credit,
        "credit_per_image":         1.0,
        "image_ids":                adjusted_image_ids,
        "type":                     "deduct",
        "thumbnail_image":          "",
        "notes":                    f"Adjust image — new photoshoot {new_photoshoot_id}",
        "photoshoot_id":            new_photoshoot_id,
        "regeneration_type":        "adjust_image",
        "regenerate_photoshoot_id": body.photoshoot_id,
        "created_at":               now,
    })

    return {
        "message":                  "Photoshoot image adjusted successfully.",
        "photoshoot_id":            new_photoshoot_id,
        "regenerate_photoshoot_id": body.photoshoot_id,
        "regeneration_type":        "adjust_image",
        "total_credit":             total_credit,
        "output_images":            output_images,
    }


@router.post(
    "/background-change",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Background Change Photoshoot",
    description=(
        "Re-runs the full SeedDream + Modal pipeline for selected poses using a new background. "
        "Fetches pose_prompts from the original photoshoot's output_images for given image_ids, "
        "swaps the background_id, and keeps all other input_parameter values unchanged. "
        "Credits: 2 per image. Secured — user_id from auth token."
    ),
)
async def background_change_photoshoot(
    body: BackgroundChangeRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]

    if not body.image_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="image_ids must contain at least one id.",
        )

    # ── Step 1: fetch new background URL ──────────────────────────────────────
    bg_col = get_backgrounds_collection()
    bg_doc = await bg_col.find_one({"background_id": body.background_id})
    if not bg_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Background '{body.background_id}' not found.",
        )

    # ── Step 2: fetch original photoshoot and resolve pose_prompts ────────────
    ps_col      = get_photoshoots_collection()
    original_ps = await ps_col.find_one({"photoshoot_id": body.photoshoot_id, "user_id": user_id})
    if not original_ps:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photoshoot '{body.photoshoot_id}' not found.",
        )

    id_set   = set(body.image_ids)
    selected = [img for img in original_ps.get("output_images", []) if img["image_id"] in id_set]
    if not selected:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="None of the provided image_ids were found in this photoshoot.",
        )

    pose_prompts = [img["pose_prompt"] for img in selected]
    total_poses  = len(pose_prompts)

    # ── Step 3: credit check ──────────────────────────────────────────────────
    total_credit    = total_poses * _CREDIT_PER_POSE
    current_credits = float(current_user.get("credits", 0))

    if current_credits < total_credit:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"Insufficient credits. Background change requires {total_credit} credits "
                f"({total_poses} pose(s) × {_CREDIT_PER_POSE}) but you only have {current_credits}."
            ),
        )

    # ── Step 4: build job payload — original params with new background ───────
    original_params   = original_ps.get("input_parameter", {})
    new_photoshoot_id = str(uuid.uuid4())
    now               = datetime.now(timezone.utc)

    job_payload = {
        **original_params,
        "user_id":                   user_id,
        "which_pose_option":         "prompt",
        "poses_prompts":             pose_prompts,
        "poses_ids":                 [],
        "poses_images":              [],
        "background_id":             body.background_id,
        "regeneration_type":         "background_change",
        "regenerate_photoshoot_id":  body.photoshoot_id,
    }

    # ── Step 5: store new photoshoot document ─────────────────────────────────
    doc = {
        "photoshoot_id":            new_photoshoot_id,
        "user_id":                  user_id,
        "sku_id":                   original_ps.get("sku_id", ""),
        "input_parameter": {
            **original_params,
            "which_pose_option":        "prompt",
            "poses_prompts":            pose_prompts,
            "poses_ids":                [],
            "poses_images":             [],
            "background_id":            body.background_id,
            "regeneration_type":        "background_change",
            "regenerate_photoshoot_id": body.photoshoot_id,
        },
        "output_images":            [],
        "failed_poses":             [],
        "total_credit":             total_credit,
        "is_credit_deducted":       False,
        "is_completed":             False,
        "status":                   "processing",
        "error":                    None,
        "regeneration_type":        "background_change",
        "regenerate_photoshoot_id": body.photoshoot_id,
        "is_active":                True,
        "created_at":               now,
        "updated_at":               now,
    }
    await ps_col.insert_one(doc)

    # ── Step 6: enqueue background job via Celery ─────────────────────────────
    task = run_photoshoot_task.apply_async(
        args=[new_photoshoot_id, job_payload],
        queue="photoshoots",
    )

    return {
        "message":                  "Background change started successfully. Processing in background.",
        "photoshoot_id":            new_photoshoot_id,
        "task_id":                  task.id,
        "regenerate_photoshoot_id": body.photoshoot_id,
        "regeneration_type":        "background_change",
        "background_id":            body.background_id,
        "total_poses":              total_poses,
        "total_credit":             total_credit,
        "status":                   "processing",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# API-1: Change fabric on garment image (Gemini only, no photoshoot re-run)
# ═══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/change-fabric",
    status_code=status.HTTP_200_OK,
    summary="Change Fabric on Garment Image",
    description=(
        "Fetches the front_garment_image (and back_garment_image if present) from the original "
        "photoshoot's input_parameter, passes each to Gemini to apply the requested fabric "
        "texture, uploads the result(s) to S3, and returns the new garment image URL(s). "
        "No new photoshoot is created. No credits are deducted. "
        "Secured — user_id from auth token."
    ),
)
async def change_fabric_garment(
    body: FabricChangeRequest,
    current_user: dict = Depends(get_current_user),
):
    from app.services.fabric_service import change_fabric
    from app.services.s3_service import upload_bytes_to_s3

    user_id = current_user["user_id"]

    # ── Step 1: fetch photoshoot ───────────────────────────────────────────────
    ps_col = get_photoshoots_collection()
    ps_doc = await ps_col.find_one({"photoshoot_id": body.photoshoot_id, "user_id": user_id})
    if not ps_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photoshoot '{body.photoshoot_id}' not found.",
        )

    # ── Step 2: extract garment image URLs from input_parameter ───────────────
    input_params      = ps_doc.get("input_parameter", {})
    front_garment_url = input_params.get("front_garment_image", "")
    back_garment_url  = input_params.get("back_garment_image", "")

    if not front_garment_url:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Original photoshoot has no front_garment_image in input_parameter.",
        )

    # ── Step 3: call Gemini for each garment image and upload to S3 ───────────
    now    = datetime.now(timezone.utc)
    result: dict = {}

    front_bytes  = await change_fabric(front_garment_url, body.fabric)
    front_key    = f"photoshoots/fabric/{body.photoshoot_id}/front_{body.fabric}_{uuid.uuid4().hex[:8]}.png"
    front_s3_url = await upload_bytes_to_s3(front_bytes, front_key, content_type="image/png")
    result["front_garment_image"] = front_s3_url

    if back_garment_url:
        back_bytes  = await change_fabric(back_garment_url, body.fabric)
        back_key    = f"photoshoots/fabric/{body.photoshoot_id}/back_{body.fabric}_{uuid.uuid4().hex[:8]}.png"
        back_s3_url = await upload_bytes_to_s3(back_bytes, back_key, content_type="image/png")
        result["back_garment_image"] = back_s3_url

    return {
        "message":        "Fabric changed successfully.",
        "photoshoot_id":  body.photoshoot_id,
        "fabric":         body.fabric,
        "garment_images": result,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# API-2: Re-generate photoshoot with new garment images (fabric change variant)
# ═══════════════════════════════════════════════════════════════════════════════

_CREDIT_FABRIC_CHANGE = 3.0


@router.post(
    "/fabric-change-photoshoot",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Fabric Change Photoshoot",
    description=(
        "Re-runs the full SeedDream + Modal pipeline for selected poses using new garment images "
        "(fabric-changed). Fetches pose_prompts from the original photoshoot's output_images for "
        "the given image_ids, replaces front/back garment images with those supplied in the request, "
        "and keeps all other input_parameter values unchanged. "
        "Credits: 3 per image. Secured — user_id from auth token."
    ),
)
async def fabric_change_photoshoot(
    body: FabricPhotoshootRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]

    if not body.image_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="image_ids must contain at least one id.",
        )

    # ── Step 1: fetch original photoshoot ─────────────────────────────────────
    ps_col      = get_photoshoots_collection()
    original_ps = await ps_col.find_one({"photoshoot_id": body.photoshoot_id, "user_id": user_id})
    if not original_ps:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photoshoot '{body.photoshoot_id}' not found.",
        )

    # ── Step 2: credit check (3 credits per image) ────────────────────────────
    total_poses     = len(body.image_ids)
    total_credit    = total_poses * _CREDIT_FABRIC_CHANGE
    current_credits = float(current_user.get("credits", 0))

    if current_credits < total_credit:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"Insufficient credits. Fabric change photoshoot requires {total_credit} credits "
                f"({total_poses} image(s) × {_CREDIT_FABRIC_CHANGE}) but you only have {current_credits}."
            ),
        )

    # ── Step 3: resolve pose_prompts for the selected image_ids ───────────────
    id_set   = set(body.image_ids)
    selected = [img for img in original_ps.get("output_images", []) if img["image_id"] in id_set]
    if not selected:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="None of the provided image_ids were found in this photoshoot.",
        )

    pose_prompts = [img["pose_prompt"] for img in selected]

    # ── Step 4: build job payload — original params with new garment images ───
    original_params   = original_ps.get("input_parameter", {})
    new_photoshoot_id = str(uuid.uuid4())
    now               = datetime.now(timezone.utc)

    updated_garments: dict = {"front_garment_image": body.front_garment_image}
    if body.back_garment_image:
        updated_garments["back_garment_image"] = body.back_garment_image

    job_payload = {
        **original_params,
        **updated_garments,
        "user_id":                   user_id,
        "which_pose_option":         "prompt",
        "poses_prompts":             pose_prompts,
        "poses_ids":                 [],
        "poses_images":              [],
        "regeneration_type":         "fabric_change",
        "regenerate_photoshoot_id":  body.photoshoot_id,
    }

    # ── Step 5: store new photoshoot document ─────────────────────────────────
    doc = {
        "photoshoot_id":            new_photoshoot_id,
        "user_id":                  user_id,
        "sku_id":                   original_ps.get("sku_id", ""),
        "input_parameter": {
            **original_params,
            **updated_garments,
            "which_pose_option":        "prompt",
            "poses_prompts":            pose_prompts,
            "poses_ids":                [],
            "poses_images":             [],
            "regeneration_type":        "fabric_change",
            "regenerate_photoshoot_id": body.photoshoot_id,
        },
        "output_images":            [],
        "failed_poses":             [],
        "total_credit":             total_credit,
        "is_credit_deducted":       False,
        "is_completed":             False,
        "status":                   "processing",
        "error":                    None,
        "regeneration_type":        "fabric_change",
        "regenerate_photoshoot_id": body.photoshoot_id,
        "is_active":                True,
        "created_at":               now,
        "updated_at":               now,
    }
    await ps_col.insert_one(doc)

    # ── Step 6: enqueue background job via Celery ─────────────────────────────
    task = run_photoshoot_task.apply_async(
        args=[new_photoshoot_id, job_payload],
        queue="photoshoots",
    )

    return {
        "message":                  "Fabric change photoshoot started successfully. Processing in background.",
        "photoshoot_id":            new_photoshoot_id,
        "task_id":                  task.id,
        "regenerate_photoshoot_id": body.photoshoot_id,
        "regeneration_type":        "fabric_change",
        "total_poses":              total_poses,
        "total_credit":             total_credit,
        "status":                   "processing",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# API-1: Change texture on garment image (Gemini only, no photoshoot re-run)
# ═══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/change-texture",
    status_code=status.HTTP_200_OK,
    summary="Change Texture on Garment Image",
    description=(
        "Fetches the front_garment_image (and back_garment_image if present) from the original "
        "photoshoot's input_parameter, passes each to Gemini to apply the requested surface "
        "texture/pattern, uploads the result(s) to S3, and returns the new garment image URL(s). "
        "No new photoshoot is created. No credits are deducted. "
        "Secured — user_id from auth token."
    ),
)
async def change_texture_garment(
    body: TextureChangeRequest,
    current_user: dict = Depends(get_current_user),
):
    from app.services.fabric_service import change_texture
    from app.services.s3_service import upload_bytes_to_s3

    user_id = current_user["user_id"]

    # ── Step 1: fetch photoshoot ───────────────────────────────────────────────
    ps_col = get_photoshoots_collection()
    ps_doc = await ps_col.find_one({"photoshoot_id": body.photoshoot_id, "user_id": user_id})
    if not ps_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photoshoot '{body.photoshoot_id}' not found.",
        )

    # ── Step 2: extract garment image URLs from input_parameter ───────────────
    input_params      = ps_doc.get("input_parameter", {})
    front_garment_url = input_params.get("front_garment_image", "")
    back_garment_url  = input_params.get("back_garment_image", "")

    if not front_garment_url:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Original photoshoot has no front_garment_image in input_parameter.",
        )

    # ── Step 3: call Gemini for each garment image and upload to S3 ───────────
    result: dict = {}

    front_bytes  = await change_texture(front_garment_url, body.texture)
    front_key    = f"photoshoots/texture/{body.photoshoot_id}/front_{body.texture.replace(' ', '_')}_{uuid.uuid4().hex[:8]}.png"
    front_s3_url = await upload_bytes_to_s3(front_bytes, front_key, content_type="image/png")
    result["front_garment_image"] = front_s3_url

    if back_garment_url:
        back_bytes  = await change_texture(back_garment_url, body.texture)
        back_key    = f"photoshoots/texture/{body.photoshoot_id}/back_{body.texture.replace(' ', '_')}_{uuid.uuid4().hex[:8]}.png"
        back_s3_url = await upload_bytes_to_s3(back_bytes, back_key, content_type="image/png")
        result["back_garment_image"] = back_s3_url

    return {
        "message":        "Texture changed successfully.",
        "photoshoot_id":  body.photoshoot_id,
        "texture":        body.texture,
        "garment_images": result,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# API-2: Re-generate photoshoot with new garment images (texture change variant)
# ═══════════════════════════════════════════════════════════════════════════════

_CREDIT_TEXTURE_CHANGE = 3.0


@router.post(
    "/texture-change-photoshoot",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Texture Change Photoshoot",
    description=(
        "Re-runs the full SeedDream + Modal pipeline for selected poses using new garment images "
        "(texture-changed). Fetches pose_prompts from the original photoshoot's output_images for "
        "the given image_ids, replaces front/back garment images with those supplied in the request, "
        "and keeps all other input_parameter values unchanged. "
        "Credits: 3 per image. Secured — user_id from auth token."
    ),
)
async def texture_change_photoshoot(
    body: TexturePhotoshootRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]

    if not body.image_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="image_ids must contain at least one id.",
        )

    # ── Step 1: fetch original photoshoot ─────────────────────────────────────
    ps_col      = get_photoshoots_collection()
    original_ps = await ps_col.find_one({"photoshoot_id": body.photoshoot_id, "user_id": user_id})
    if not original_ps:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photoshoot '{body.photoshoot_id}' not found.",
        )

    # ── Step 2: credit check (3 credits per image) ────────────────────────────
    total_poses     = len(body.image_ids)
    total_credit    = total_poses * _CREDIT_TEXTURE_CHANGE
    current_credits = float(current_user.get("credits", 0))

    if current_credits < total_credit:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"Insufficient credits. Texture change photoshoot requires {total_credit} credits "
                f"({total_poses} image(s) × {_CREDIT_TEXTURE_CHANGE}) but you only have {current_credits}."
            ),
        )

    # ── Step 3: resolve pose_prompts for the selected image_ids ───────────────
    id_set   = set(body.image_ids)
    selected = [img for img in original_ps.get("output_images", []) if img["image_id"] in id_set]
    if not selected:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="None of the provided image_ids were found in this photoshoot.",
        )

    pose_prompts = [img["pose_prompt"] for img in selected]

    # ── Step 4: build job payload — original params with new garment images ───
    original_params   = original_ps.get("input_parameter", {})
    new_photoshoot_id = str(uuid.uuid4())
    now               = datetime.now(timezone.utc)

    updated_garments: dict = {"front_garment_image": body.front_garment_image}
    if body.back_garment_image:
        updated_garments["back_garment_image"] = body.back_garment_image

    job_payload = {
        **original_params,
        **updated_garments,
        "user_id":                   user_id,
        "which_pose_option":         "prompt",
        "poses_prompts":             pose_prompts,
        "poses_ids":                 [],
        "poses_images":              [],
        "regeneration_type":         "texture_change",
        "regenerate_photoshoot_id":  body.photoshoot_id,
    }

    # ── Step 5: store new photoshoot document ─────────────────────────────────
    doc = {
        "photoshoot_id":            new_photoshoot_id,
        "user_id":                  user_id,
        "sku_id":                   original_ps.get("sku_id", ""),
        "input_parameter": {
            **original_params,
            **updated_garments,
            "which_pose_option":        "prompt",
            "poses_prompts":            pose_prompts,
            "poses_ids":                [],
            "poses_images":             [],
            "regeneration_type":        "texture_change",
            "regenerate_photoshoot_id": body.photoshoot_id,
        },
        "output_images":            [],
        "failed_poses":             [],
        "total_credit":             total_credit,
        "is_credit_deducted":       False,
        "is_completed":             False,
        "status":                   "processing",
        "error":                    None,
        "regeneration_type":        "texture_change",
        "regenerate_photoshoot_id": body.photoshoot_id,
        "is_active":                True,
        "created_at":               now,
        "updated_at":               now,
    }
    await ps_col.insert_one(doc)

    # ── Step 6: enqueue background job via Celery ─────────────────────────────
    task = run_photoshoot_task.apply_async(
        args=[new_photoshoot_id, job_payload],
        queue="photoshoots",
    )

    return {
        "message":                  "Texture change photoshoot started successfully. Processing in background.",
        "photoshoot_id":            new_photoshoot_id,
        "task_id":                  task.id,
        "regenerate_photoshoot_id": body.photoshoot_id,
        "regeneration_type":        "texture_change",
        "total_poses":              total_poses,
        "total_credit":             total_credit,
        "status":                   "processing",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# API-1: Change color on garment image (Gemini only, no photoshoot re-run)
# ═══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/change-color",
    status_code=status.HTTP_200_OK,
    summary="Change Color on Garment Image",
    description=(
        "Fetches the front_garment_image (and back_garment_image if present) from the original "
        "photoshoot's input_parameter, passes each to Gemini to apply the requested hex color, "
        "uploads the result(s) to S3, and returns the new garment image URL(s). "
        "No new photoshoot is created. No credits are deducted. "
        "Secured — user_id from auth token."
    ),
)
async def change_color_garment(
    body: ColorChangeRequest,
    current_user: dict = Depends(get_current_user),
):
    from app.services.fabric_service import change_color
    from app.services.s3_service import upload_bytes_to_s3

    user_id = current_user["user_id"]

    # ── Step 1: fetch photoshoot ───────────────────────────────────────────────
    ps_col = get_photoshoots_collection()
    ps_doc = await ps_col.find_one({"photoshoot_id": body.photoshoot_id, "user_id": user_id})
    if not ps_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photoshoot '{body.photoshoot_id}' not found.",
        )

    # ── Step 2: extract garment image URLs from input_parameter ───────────────
    input_params      = ps_doc.get("input_parameter", {})
    front_garment_url = input_params.get("front_garment_image", "")
    back_garment_url  = input_params.get("back_garment_image", "")

    if not front_garment_url:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Original photoshoot has no front_garment_image in input_parameter.",
        )

    # ── Step 3: call Gemini for each garment image and upload to S3 ───────────
    safe_hex = body.color_hex.lstrip("#")
    result: dict = {}

    front_bytes  = await change_color(front_garment_url, body.color_hex)
    front_key    = f"photoshoots/color/{body.photoshoot_id}/front_{safe_hex}_{uuid.uuid4().hex[:8]}.png"
    front_s3_url = await upload_bytes_to_s3(front_bytes, front_key, content_type="image/png")
    result["front_garment_image"] = front_s3_url

    if back_garment_url:
        back_bytes  = await change_color(back_garment_url, body.color_hex)
        back_key    = f"photoshoots/color/{body.photoshoot_id}/back_{safe_hex}_{uuid.uuid4().hex[:8]}.png"
        back_s3_url = await upload_bytes_to_s3(back_bytes, back_key, content_type="image/png")
        result["back_garment_image"] = back_s3_url

    return {
        "message":        "Color changed successfully.",
        "photoshoot_id":  body.photoshoot_id,
        "color_hex":      body.color_hex,
        "garment_images": result,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# API-2: Re-generate photoshoot with new garment images (color change variant)
# ═══════════════════════════════════════════════════════════════════════════════

_CREDIT_COLOR_CHANGE = 3.0


@router.post(
    "/color-change-photoshoot",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Color Change Photoshoot",
    description=(
        "Re-runs the full SeedDream + Modal pipeline for selected poses using new garment images "
        "(color-changed). Fetches pose_prompts from the original photoshoot's output_images for "
        "the given image_ids, replaces front/back garment images with those supplied in the request, "
        "and keeps all other input_parameter values unchanged. "
        "Credits: 3 per image. Secured — user_id from auth token."
    ),
)
async def color_change_photoshoot(
    body: ColorPhotoshootRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]

    if not body.image_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="image_ids must contain at least one id.",
        )

    # ── Step 1: fetch original photoshoot ─────────────────────────────────────
    ps_col      = get_photoshoots_collection()
    original_ps = await ps_col.find_one({"photoshoot_id": body.photoshoot_id, "user_id": user_id})
    if not original_ps:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photoshoot '{body.photoshoot_id}' not found.",
        )

    # ── Step 2: credit check (3 credits per image) ────────────────────────────
    total_poses     = len(body.image_ids)
    total_credit    = total_poses * _CREDIT_COLOR_CHANGE
    current_credits = float(current_user.get("credits", 0))

    if current_credits < total_credit:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"Insufficient credits. Color change photoshoot requires {total_credit} credits "
                f"({total_poses} image(s) × {_CREDIT_COLOR_CHANGE}) but you only have {current_credits}."
            ),
        )

    # ── Step 3: resolve pose_prompts for the selected image_ids ───────────────
    id_set   = set(body.image_ids)
    selected = [img for img in original_ps.get("output_images", []) if img["image_id"] in id_set]
    if not selected:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="None of the provided image_ids were found in this photoshoot.",
        )

    pose_prompts = [img["pose_prompt"] for img in selected]

    # ── Step 4: build job payload — original params with new garment images ───
    original_params   = original_ps.get("input_parameter", {})
    new_photoshoot_id = str(uuid.uuid4())
    now               = datetime.now(timezone.utc)

    updated_garments: dict = {"front_garment_image": body.front_garment_image}
    if body.back_garment_image:
        updated_garments["back_garment_image"] = body.back_garment_image

    job_payload = {
        **original_params,
        **updated_garments,
        "user_id":                   user_id,
        "which_pose_option":         "prompt",
        "poses_prompts":             pose_prompts,
        "poses_ids":                 [],
        "poses_images":              [],
        "regeneration_type":         "color_change",
        "regenerate_photoshoot_id":  body.photoshoot_id,
    }

    # ── Step 5: store new photoshoot document ─────────────────────────────────
    doc = {
        "photoshoot_id":            new_photoshoot_id,
        "user_id":                  user_id,
        "sku_id":                   original_ps.get("sku_id", ""),
        "input_parameter": {
            **original_params,
            **updated_garments,
            "which_pose_option":        "prompt",
            "poses_prompts":            pose_prompts,
            "poses_ids":                [],
            "poses_images":             [],
            "regeneration_type":        "color_change",
            "regenerate_photoshoot_id": body.photoshoot_id,
        },
        "output_images":            [],
        "failed_poses":             [],
        "total_credit":             total_credit,
        "is_credit_deducted":       False,
        "is_completed":             False,
        "status":                   "processing",
        "error":                    None,
        "regeneration_type":        "color_change",
        "regenerate_photoshoot_id": body.photoshoot_id,
        "is_active":                True,
        "created_at":               now,
        "updated_at":               now,
    }
    await ps_col.insert_one(doc)

    # ── Step 6: enqueue background job via Celery ─────────────────────────────
    task = run_photoshoot_task.apply_async(
        args=[new_photoshoot_id, job_payload],
        queue="photoshoots",
    )

    return {
        "message":                  "Color change photoshoot started successfully. Processing in background.",
        "photoshoot_id":            new_photoshoot_id,
        "task_id":                  task.id,
        "regenerate_photoshoot_id": body.photoshoot_id,
        "regeneration_type":        "color_change",
        "total_poses":              total_poses,
        "total_credit":             total_credit,
        "status":                   "processing",
    }
