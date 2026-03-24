"""
Photoshoot background job service.

Responsibilities:
  1. Resolve pose prompts (default → fetch from poses_data, custom → Gemini vision,
     prompt → use as-is).
  2. Fetch background image URL from backgrounds collection.
  3. Fetch model face image URL from model_faces collection.
  4. For each pose — concurrently via asyncio.gather:
       a. Build a detailed generation prompt.
       b. Submit SeedDream task (quality=high, aspect=9:16).
       c. Poll until done.
       d. Download 4K result bytes.
       e. Resize to 2K and 1K (SeedDream originals).
       f. Upload SeedDream 4K, 2K, 1K to S3.
       g. Send 4K bytes to Modal GPU pipeline for enhancement.
       h. Upload enhanced 8K, 4K, 2K, 1K to S3 and save to upscaling_data.
  5. Build output_images mapping — each entry stores the upscaled 1K URL as `image`.
  6. Deduct credits and record history.
  7. Update photoshoot document: output_images, status, is_completed, is_credit_deducted.
     On any unhandled error → status="failed", error field set.
"""

import asyncio
import io
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import List

import httpx
from PIL import Image

logger = logging.getLogger("photoshoot")

from app.config import settings
from app.database import (
    get_backgrounds_collection,
    get_model_faces_collection,
    get_photoshoots_collection,
    get_poses_collection,
    get_users_collection,
    get_credit_history_collection,
)
from app.services.s3_service import upload_bytes_to_s3
from app.services.modal_enhance_service import enhance_and_upload

_CREATE_URL = "https://api.kie.ai/api/v1/jobs/createTask"
_STATUS_URL  = "https://api.kie.ai/api/v1/jobs/recordInfo"
_PHOTOSHOOT_CREDIT_PER_POSE = 2.0


# ---------------------------------------------------------------------------
# Pose prompt resolution
# ---------------------------------------------------------------------------

async def _fetch_default_pose_prompts(pose_ids: List[str]) -> List[str]:
    logger.info("[poses] Fetching %d default pose prompt(s) from DB", len(pose_ids))
    col     = get_poses_collection()
    prompts = []
    for pid in pose_ids:
        doc = await col.find_one({"pose_id": pid})
        if doc and doc.get("pose_prompt"):
            logger.info("[poses] Found prompt for pose_id=%s", pid)
            prompts.append(doc["pose_prompt"])
        else:
            logger.warning("[poses] No prompt found for pose_id=%s — using fallback", pid)
            prompts.append(f"Standing in a natural, relaxed fashion model pose — pose id: {pid}")
    logger.info("[poses] Resolved %d default pose prompt(s)", len(prompts))
    return prompts


async def _generate_pose_prompt_from_image(image_url: str) -> str:
    """
    Send a custom pose image to Gemini vision and extract ONLY body position
    and pose details — no gender, age, clothing, or background details.
    """
    logger.info("[poses] Generating pose prompt from custom image: %s", image_url)
    try:
        from google import genai
        from google.genai import types as gtypes
    except ImportError as exc:
        logger.error("[poses] Gemini SDK not available: %s", exc)
        return f"Natural fashion model pose — Gemini unavailable: {exc}"

    try:
        logger.info("[poses] Downloading custom pose image...")
        async with httpx.AsyncClient(timeout=30) as client:
            img_resp = await client.get(image_url)
            img_resp.raise_for_status()
        img_bytes = img_resp.content
        content_type = img_resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        if content_type not in ("image/jpeg", "image/png", "image/webp"):
            content_type = "image/jpeg"
        logger.info("[poses] Custom pose image downloaded (%d bytes, %s)", len(img_bytes), content_type)
    except Exception as exc:
        logger.error("[poses] Failed to download custom pose image: %s", exc)
        return f"Natural fashion model pose — could not download image: {exc}"

    prompt_text = (
        "You are a pose description specialist for fashion photoshoots.\n"
        "Analyze this reference image and describe ONLY the body position and pose.\n\n"
        "STRICT RULES — your description MUST:\n"
        "- Include: exact body orientation (front/back/side/angle), limb positions, "
        "  head direction, weight distribution, framing (full body/half body/close-up)\n"
        "- EXCLUDE: gender, age, age group, clothing details, fabric, colors, "
        "  accessories, background, setting, or environment\n\n"
        "Output: a single concise paragraph, 2-4 sentences, describing only pose and body position."
    )

    loop = asyncio.get_event_loop()

    def _call_gemini():
        g_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return g_client.models.generate_content(
            model=settings.GEMINI_VISION_MODEL,
            contents=[
                gtypes.Content(
                    role="user",
                    parts=[
                        gtypes.Part.from_bytes(mime_type=content_type, data=img_bytes),
                        gtypes.Part.from_text(text=prompt_text),
                    ],
                )
            ],
            config=gtypes.GenerateContentConfig(
                response_modalities=["TEXT"],
                temperature=0,
            ),
        )

    try:
        logger.info("[poses] Calling Gemini vision to extract pose description...")
        response = await loop.run_in_executor(None, _call_gemini)
        result = response.candidates[0].content.parts[0].text.strip()
        logger.info("[poses] Gemini pose description generated (%d chars)", len(result))
        return result
    except Exception as exc:
        logger.error("[poses] Gemini vision error: %s", exc)
        return f"Natural standing fashion model pose — vision error: {exc}"


async def resolve_pose_prompts(
    which_pose_option: str,
    poses_ids: List[str],
    poses_images: List[str],
    poses_prompts: List[str],
) -> List[str]:
    logger.info("[poses] Resolving poses — option=%s", which_pose_option)
    if which_pose_option == "default":
        result = await _fetch_default_pose_prompts(poses_ids)
    elif which_pose_option == "custom":
        result = await asyncio.gather(*[_generate_pose_prompt_from_image(url) for url in poses_images])
        result = list(result)
        logger.info("[poses] %d custom pose prompts generated via Gemini", len(result))
    else:
        logger.info("[poses] Using %d user-provided pose prompts directly", len(poses_prompts))
        result = list(poses_prompts)

    for i, prompt in enumerate(result, 1):
        logger.info("[poses] Pose #%d prompt:\n%s", i, prompt)

    return result


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _is_back_pose(pose_prompt: str) -> bool:
    back_keywords = ["back", "rear", "behind", "back-facing", "facing away", "turned away"]
    lower = pose_prompt.lower()
    return any(kw in lower for kw in back_keywords)


def _build_photoshoot_prompt(
    pose: str,
    has_back_image: bool,
    req: dict,
) -> str:
    if has_back_image:
        image_ref = (
            "You are provided with FOUR reference images:\n"
            "  IMAGE 1 — GARMENT FRONT: exact outfit front view.\n"
            "  IMAGE 2 — GARMENT BACK: exact outfit back view.\n"
            "  IMAGE 3 — MODEL FACE: the exact face to use.\n"
            "  IMAGE 4 — BACKGROUND: the exact background scene."
        )
        garment_note = (
            "- USE EXACT GARMENT from IMAGE 1 (front) and IMAGE 2 (back) — MANDATORY.\n"
            "- Reproduce every detail from both views with 100% accuracy."
        )
    else:
        image_ref = (
            "You are provided with THREE reference images:\n"
            "  IMAGE 1 — GARMENT: exact outfit to be worn.\n"
            "  IMAGE 2 — MODEL FACE: the exact face to use.\n"
            "  IMAGE 3 — BACKGROUND: the exact background scene."
        )
        garment_note = (
            "- USE EXACT GARMENT from IMAGE 1 — MANDATORY.\n"
            "- Reproduce every detail with 100% accuracy."
        )

    ug_type  = req.get("upper_garment_type", "")
    ug_spec  = req.get("upper_garment_specification", "")
    lg_type  = req.get("lower_garment_type", "")
    lg_spec  = req.get("lower_garment_specification", "")
    op_type  = req.get("one_piece_garment_type", "")
    op_spec  = req.get("one_piece_garment_specification", "")
    fitting  = req.get("fitting", "regular fit")
    gender   = req.get("gender", "").strip().lower()

    upper_line    = f"- Upper: {ug_type}{f' — {ug_spec}' if ug_spec else ''}" if ug_type else ""
    lower_line    = f"- Lower: {lg_type}{f' — {lg_spec}' if lg_spec else ''}" if lg_type else ""
    onepiece_line = f"- One-piece: {op_type}{f' — {op_spec}' if op_spec else ''}" if op_type else ""

    garment_lines = "\n".join(filter(None, [upper_line, lower_line, onepiece_line]))

    footwear_instruction = (
        "Choose footwear that naturally complements and matches the outfit style, "
        "color palette, and formality level shown in the garment reference image. "
        "Examples: heels / sandals for formal/ethnic wear, sneakers / loafers for casual, "
        "boots for western/denim, flats for relaxed everyday wear. "
        "Footwear must look realistic, well-fitted, and appropriate for the overall look."
    )

    female_accessory_instruction = ""
    if gender == "female":
        female_accessory_instruction = (
            "\n\n[ACCESSORIES — FEMALE]\n"
            "Evaluate the outfit style from the garment reference image:\n"
            "- If the outfit is formal, ethnic, party-wear, or occasion-wear → add a stylish PURSE "
            "  (clutch or small structured bag) that complements the garment's color and style.\n"
            "- If the outfit is casual, western, denim, or everyday wear → add a HANDBAG "
            "  (tote, sling, or shoulder bag) that matches the outfit's vibe.\n"
            "- The bag must look natural in the model's hand or on her shoulder, matching the "
            "  overall color palette. Do NOT add a bag if the pose makes it physically impossible."
        )

    return f"""{image_ref}

Generate a photorealistic fashion photoshoot image. Zero deviation from reference images.

[FACE — DO NOT CHANGE]
Use EXACT face from model face reference. Match: face shape, eyes, nose, lips, skin tone, eyebrows, hair. FACE VALIDATION: verify face matches reference before finalising — regenerate if any deviation.
Model: {req['gender']}, {req['ethnicity']}, {req['age']} ({req['age_group']}), {req['weight']} build, {req['height']}, {req['skin_tone']} skin.

[GARMENT — DO NOT CHANGE]
{garment_note}
{garment_lines}
- Fitting: {fitting}

[BACKGROUND — DO NOT CHANGE]
Use EXACT background from reference. Match: all objects, colors, lighting, shadows, depth. No alterations.

[POSE] {pose}

[FOOTWEAR]
{footwear_instruction}{female_accessory_instruction}

[STYLE] Lighting: {req.get('lighting_style', 'natural light')} | Ornaments: {req.get('ornaments', 'none')}

[QUALITY] Ultra-high resolution 4K, photorealistic, professional fashion photography, sharp focus, seamless model-background integration, commercial e-commerce grade.

NON-NEGOTIABLE: 1. Face = 100% identical to reference. 2. Garment = 100% identical to reference. 3. Background = 100% identical to reference."""


# ---------------------------------------------------------------------------
# SeedDream submit + poll
# ---------------------------------------------------------------------------

async def _submit_task(prompt: str, image_urls: List[str], pose_label: str) -> str:
    logger.info("[%s] Submitting SeedDream task (quality=high, 9:16, %d images)...", pose_label, len(image_urls))
    payload = json.dumps({
        "model": settings.SEEDDREAM_MODEL,
        "input": {
            "prompt":       prompt,
            "image_urls":   image_urls,
            "aspect_ratio": "9:16",
            "quality":      "high",
        },
    })
    headers = {
        "Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}",
        "Content-Type":  "application/json",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(_CREATE_URL, headers=headers, content=payload)
        resp.raise_for_status()
    task_id = resp.json().get("data", {}).get("taskId")
    if not task_id:
        raise RuntimeError(f"No taskId returned: {resp.text}")
    logger.info("[%s] Task submitted — task_id=%s", pose_label, task_id)
    return task_id


async def _poll_task(task_id: str, pose_label: str) -> str:
    logger.info("[%s] Polling SeedDream task_id=%s ...", pose_label, task_id)
    headers = {"Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}"}
    for attempt in range(1, settings.SEEDDREAM_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(f"{_STATUS_URL}?taskId={task_id}", headers=headers)
                resp.raise_for_status()
            data  = resp.json().get("data", {})
            state = data.get("state")
            if state == "success":
                urls = json.loads(data.get("resultJson", "{}")).get("resultUrls", [])
                if urls:
                    logger.info("[%s] Task complete — result URL obtained (attempt %d)", pose_label, attempt)
                    return urls[0]
                raise RuntimeError("Task succeeded but no resultUrls found.")
            if state == "fail":
                raise RuntimeError("SeedDream task failed.")
            logger.debug("[%s] Poll #%d — state=%s", pose_label, attempt, state)
            await asyncio.sleep(settings.SEEDDREAM_RETRY_DELAY)
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("[%s] Poll #%d error: %s", pose_label, attempt, exc)
            await asyncio.sleep(settings.SEEDDREAM_RETRY_DELAY)
    raise RuntimeError(f"SeedDream task timed out after {settings.SEEDDREAM_MAX_RETRIES} attempts.")


# ---------------------------------------------------------------------------
# Image download + resize helpers
# ---------------------------------------------------------------------------

async def _download_bytes(url: str, label: str = "") -> bytes:
    logger.info("[%s] Downloading image from URL...", label)
    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            data = resp.content
            logger.info("[%s] Image downloaded — %d bytes", label, len(data))
            return data
        except Exception as exc:
            logger.warning("[%s] Download attempt %d/3 failed: %s", label, attempt, exc)
            if attempt == 3:
                raise RuntimeError(f"Failed to download image after 3 attempts: {exc}")
            await asyncio.sleep(3)


def _resize_image(original_bytes: bytes, max_dimension: int) -> bytes:
    img = Image.open(io.BytesIO(original_bytes)).convert("RGB")
    w, h = img.size
    scale = max_dimension / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Single pose worker
# ---------------------------------------------------------------------------

async def _process_one_pose(
    pose_idx:      int,
    pose_prompt:   str,
    image_urls:    List[str],
    photoshoot_id: str,
    req_snapshot:  dict,
) -> dict:
    """
    Full async pipeline for one pose. Returns output_image dict or raises on failure.
    """
    pose_label = f"pose-{pose_idx:02d}"
    image_id   = str(uuid.uuid4())
    logger.info("[%s] ── Starting pose pipeline ──────────────────────", pose_label)
    logger.info("[%s] Pose prompt:\n%s", pose_label, pose_prompt)

    logger.info("[%s] Building SeedDream generation prompt...", pose_label)
    has_back = bool(req_snapshot.get("back_garment_image", ""))
    prompt   = _build_photoshoot_prompt(pose_prompt, has_back, req_snapshot)
    logger.info("[%s] Generation prompt built (%d chars, back_image=%s)", pose_label, len(prompt), has_back)

    task_id = await _submit_task(prompt, image_urls, pose_label)
    result_url_4k = await _poll_task(task_id, pose_label)
    logger.info("[%s] 4K result URL received", pose_label)

    bytes_4k = await _download_bytes(result_url_4k, pose_label)

    # ── SeedDream originals: resize 4K → 2K / 1K ─────────────────────────
    logger.info("[%s] Resizing 4K → 2K...", pose_label)
    bytes_2k = await asyncio.get_event_loop().run_in_executor(None, _resize_image, bytes_4k, 2048)
    logger.info("[%s] Resizing 4K → 1K...", pose_label)
    bytes_1k = await asyncio.get_event_loop().run_in_executor(None, _resize_image, bytes_4k, 1024)
    logger.info("[%s] Resize complete", pose_label)

    prefix = f"photoshoots/{photoshoot_id}/{image_id}"
    logger.info("[%s] Uploading SeedDream 4K / 2K / 1K to S3 (prefix=%s)...", pose_label, prefix)

    url_4k, url_2k, url_1k = await asyncio.gather(
        upload_bytes_to_s3(bytes_4k, f"{prefix}_4k.png", "image/png"),
        upload_bytes_to_s3(bytes_2k, f"{prefix}_2k.png", "image/png"),
        upload_bytes_to_s3(bytes_1k, f"{prefix}_1k.png", "image/png"),
    )
    logger.info("[%s] Uploaded SeedDream 4K, 2K, 1K", pose_label)

    # ── Modal GPU enhancement: 4K → enhanced 8K / 4K / 2K / 1K ──────────
    logger.info("[%s] Sending 4K bytes to Modal GPU pipeline for enhancement...", pose_label)
    upscale_result = await enhance_and_upload(
        image_bytes=bytes_4k,
        photoshoot_id=photoshoot_id,
        image_id=image_id,
        seeddream_4k_url=url_4k,
        seeddream_2k_url=url_2k,
        seeddream_1k_url=url_1k,
    )
    logger.info("[%s] Modal enhancement complete — upscaled 1K: %s", pose_label,
                upscale_result.get("1k_upscaled", "N/A")[:80])

    # Use the upscaled 1K as the primary display image; fall back to SeedDream 1K
    display_image = upscale_result.get("1k_upscaled") or url_1k

    logger.info("[%s] ── Pose pipeline COMPLETE ─────────────────────", pose_label)
    return {
        "image_id":    image_id,
        "pose_prompt": pose_prompt,
        "image":       display_image,
    }


# ---------------------------------------------------------------------------
# Credit deduction helper
# ---------------------------------------------------------------------------

async def _deduct_photoshoot_credits(
    user_id: str,
    total_credit: float,
    photoshoot_id: str,
    regeneration_type: str = "",
    regenerate_photoshoot_id: str = "",
    image_ids: list = None,
    credit_per_image: float = None,
) -> None:
    logger.info("[credits] Deducting %.2f credits from user_id=%s for photoshoot=%s",
                total_credit, user_id, photoshoot_id)
    users_col   = get_users_collection()
    history_col = get_credit_history_collection()

    user = await users_col.find_one({"user_id": user_id})
    if not user:
        logger.error("[credits] User not found: %s — skipping credit deduction", user_id)
        return

    old_credits = float(user.get("credits", 0))
    new_credits = round(old_credits - total_credit, 4)
    await users_col.update_one(
        {"user_id": user_id},
        {"$set": {"credits": new_credits, "updated_at": datetime.now(timezone.utc)}},
    )
    logger.info("[credits] Credits updated: %.4f → %.4f", old_credits, new_credits)

    feature_name = "photoshoot_regenerate" if regeneration_type == "regenerate" else "photoshoot_generation"
    notes        = f"Photoshoot {photoshoot_id}"
    if regeneration_type:
        notes = f"{regeneration_type} — new photoshoot {photoshoot_id}"
        if regenerate_photoshoot_id:
            notes += f" (from {regenerate_photoshoot_id})"

    history_doc = {
        "history_id":       str(uuid.uuid4()),
        "user_id":          user_id,
        "feature_name":     feature_name,
        "credit":           total_credit,
        "credit_per_image": credit_per_image if credit_per_image is not None else _PHOTOSHOOT_CREDIT_PER_POSE,
        "image_ids":        image_ids or [],
        "type":             "deduct",
        "thumbnail_image":  "",
        "notes":            notes,
        "photoshoot_id":    photoshoot_id,
        "created_at":       datetime.now(timezone.utc),
    }
    if regeneration_type:
        history_doc["regeneration_type"]        = regeneration_type
        history_doc["regenerate_photoshoot_id"] = regenerate_photoshoot_id

    await history_col.insert_one(history_doc)
    logger.info("[credits] Credit history record inserted")


# ---------------------------------------------------------------------------
# Public background job entry point
# ---------------------------------------------------------------------------

async def run_photoshoot_job(photoshoot_id: str, req: dict) -> None:
    """
    Called as a FastAPI BackgroundTask. Runs the full photoshoot pipeline,
    updates the photoshoot document, deducts credits.
    """
    logger.info("=" * 70)
    logger.info("[job] Photoshoot job STARTED — photoshoot_id=%s", photoshoot_id)
    logger.info("[job] user_id=%s | pose_option=%s", req.get("user_id"), req.get("which_pose_option"))
    logger.info("=" * 70)

    col       = get_photoshoots_collection()
    job_start = time.time()

    try:
        # ── Step 1: resolve pose prompts ──────────────────────────────────
        logger.info("[job] STEP 1 — Resolving pose prompts...")
        pose_prompts = await resolve_pose_prompts(
            which_pose_option=req["which_pose_option"],
            poses_ids=req.get("poses_ids") or [],
            poses_images=req.get("poses_images") or [],
            poses_prompts=req.get("poses_prompts") or [],
        )

        if not pose_prompts:
            raise ValueError("No pose prompts could be resolved.")

        logger.info("[job] STEP 1 DONE — %d pose prompt(s) ready", len(pose_prompts))

        # ── Step 2: fetch background and model face URLs ──────────────────
        logger.info("[job] STEP 2 — Fetching background and model face from DB...")
        bg_doc = await get_backgrounds_collection().find_one({"background_id": req["background_id"]})
        if not bg_doc:
            raise ValueError(f"Background not found: {req['background_id']}")
        background_url = bg_doc["background_url"]
        logger.info("[job] Background found: %s", background_url[:80])

        mf_doc = await get_model_faces_collection().find_one({"model_id": req["model_id"]})
        if not mf_doc:
            raise ValueError(f"Model face not found: {req['model_id']}")
        model_face_url = mf_doc["face_url"]
        logger.info("[job] Model face found: %s", model_face_url[:80])

        image_urls = [req["front_garment_image"]]
        if req.get("back_garment_image"):
            image_urls.append(req["back_garment_image"])
            logger.info("[job] Back garment image included")
        image_urls.append(model_face_url)
        image_urls.append(background_url)
        logger.info("[job] STEP 2 DONE — %d reference images assembled", len(image_urls))

        # ── Step 3: process all poses concurrently ────────────────────────
        logger.info("[job] STEP 3 — Launching %d pose(s) concurrently...", len(pose_prompts))

        tasks = [
            _process_one_pose(idx, prompt, image_urls, photoshoot_id, req)
            for idx, prompt in enumerate(pose_prompts, 1)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output_images = []
        failed_poses  = []
        for idx, result in enumerate(results, 1):
            if isinstance(result, Exception):
                failed_poses.append({"pose_index": idx, "error": str(result)})
                logger.error("[job] pose-%02d FAILED: %s", idx, result)
            else:
                output_images.append(result)
                logger.info("[job] pose-%02d SUCCEEDED", idx)

        logger.info("[job] STEP 3 DONE — %d succeeded, %d failed",
                    len(output_images), len(failed_poses))

        # ── Step 4: deduct credits ────────────────────────────────────────
        logger.info("[job] STEP 4 — Deducting credits...")
        total_credit = len(pose_prompts) * _PHOTOSHOOT_CREDIT_PER_POSE
        generated_image_ids = [img["image_id"] for img in output_images]
        await _deduct_photoshoot_credits(
            req["user_id"],
            total_credit,
            photoshoot_id,
            regeneration_type=req.get("regeneration_type", ""),
            regenerate_photoshoot_id=req.get("regenerate_photoshoot_id", ""),
            image_ids=generated_image_ids,
            credit_per_image=_PHOTOSHOOT_CREDIT_PER_POSE,
        )
        logger.info("[job] STEP 4 DONE — %.2f credits deducted", total_credit)

        # ── Step 5: update photoshoot document ───────────────────────────
        logger.info("[job] STEP 5 — Updating photoshoot document in DB...")
        final_status = "completed" if not failed_poses else "partial"
        await col.update_one(
            {"photoshoot_id": photoshoot_id},
            {"$set": {
                "output_images":       output_images,
                "failed_poses":        failed_poses,
                "total_credit":        total_credit,
                "is_credit_deducted":  True,
                "is_completed":        True,
                "status":              final_status,
                "updated_at":          datetime.now(timezone.utc),
            }},
        )

        elapsed = round(time.time() - job_start, 1)
        logger.info("=" * 70)
        logger.info("[job] Photoshoot job FINISHED — photoshoot_id=%s | status=%s | elapsed=%.1fs",
                    photoshoot_id, final_status, elapsed)
        logger.info("=" * 70)

    except Exception as exc:
        elapsed = round(time.time() - job_start, 1)
        logger.error("[job] Photoshoot job FAILED — photoshoot_id=%s | error=%s | elapsed=%.1fs",
                     photoshoot_id, exc, elapsed)
        await col.update_one(
            {"photoshoot_id": photoshoot_id},
            {"$set": {
                "status":     "failed",
                "error":      str(exc),
                "updated_at": datetime.now(timezone.utc),
            }},
        )
