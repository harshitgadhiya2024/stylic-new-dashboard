"""
Photoshoot background job service.

Responsibilities:
  1. Resolve pose prompts (default → fetch from poses_data, custom → Gemini vision,
     prompt → use as-is).
  2. Fetch background image URL from backgrounds collection.
  3. Fetch model face image URL from model_faces collection.
  4. For each pose — concurrently via ThreadPoolExecutor:
       a. Build a detailed generation prompt.
       b. Submit SeedDream task (quality=high, aspect=9:16).
       c. Poll until done.
       d. Download 4K result bytes.
       e. Resize to 2K and 1K.
       f. Deblur all three sizes (GFPGAN + Real-ESRGAN).
       g. Upload all 6 images (3 originals + 3 deblurred) to S3.
  5. Build output_images mapping.
  6. Deduct credits and record history.
  7. Update photoshoot document: output_images, status, is_completed, is_credit_deducted.
     On any unhandled error → status="failed", error field set.
"""

import io
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import List

import requests

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

_CREATE_URL = "https://api.kie.ai/api/v1/jobs/createTask"
_STATUS_URL  = "https://api.kie.ai/api/v1/jobs/recordInfo"
_PHOTOSHOOT_CREDIT_PER_POSE = 2.0


# ---------------------------------------------------------------------------
# Pose prompt resolution
# ---------------------------------------------------------------------------

def _fetch_default_pose_prompts(pose_ids: List[str]) -> List[str]:
    logger.info("[poses] Fetching %d default pose prompt(s) from DB", len(pose_ids))
    col     = get_poses_collection()
    prompts = []
    for pid in pose_ids:
        doc = col.find_one({"pose_id": pid})
        if doc and doc.get("pose_prompt"):
            logger.info("[poses] Found prompt for pose_id=%s", pid)
            prompts.append(doc["pose_prompt"])
        else:
            logger.warning("[poses] No prompt found for pose_id=%s — using fallback", pid)
            prompts.append(f"Standing in a natural, relaxed fashion model pose — pose id: {pid}")
    logger.info("[poses] Resolved %d default pose prompt(s)", len(prompts))
    return prompts


def _generate_pose_prompt_from_image(image_url: str) -> str:
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
        img_resp = requests.get(image_url, timeout=30)
        img_resp.raise_for_status()
        img_bytes = img_resp.content
        content_type = img_resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
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

    try:
        logger.info("[poses] Calling Gemini vision to extract pose description...")
        client   = genai.Client(api_key=settings.GEMINI_API_KEY)
        response = client.models.generate_content(
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
        result = response.candidates[0].content.parts[0].text.strip()
        logger.info("[poses] Gemini pose description generated (%d chars)", len(result))
        return result
    except Exception as exc:
        logger.error("[poses] Gemini vision error: %s", exc)
        return f"Natural standing fashion model pose — vision error: {exc}"


def resolve_pose_prompts(
    which_pose_option: str,
    poses_ids: List[str],
    poses_images: List[str],
    poses_prompts: List[str],
) -> List[str]:
    logger.info("[poses] Resolving poses — option=%s", which_pose_option)
    if which_pose_option == "default":
        result = _fetch_default_pose_prompts(poses_ids)
    elif which_pose_option == "custom":
        result = [_generate_pose_prompt_from_image(url) for url in poses_images]
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

    upper_line     = f"- Upper: {ug_type}{f' — {ug_spec}' if ug_spec else ''}" if ug_type else ""
    lower_line     = f"- Lower: {lg_type}{f' — {lg_spec}' if lg_spec else ''}" if lg_type else ""
    onepiece_line  = f"- One-piece: {op_type}{f' — {op_spec}' if op_spec else ''}" if op_type else ""

    garment_lines = "\n".join(filter(None, [upper_line, lower_line, onepiece_line]))

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

[STYLE] Lighting: {req.get('lighting_style', 'natural light')} | Ornaments: {req.get('ornaments', 'none')}

[QUALITY] Ultra-high resolution 4K, photorealistic, professional fashion photography, sharp focus, seamless model-background integration, commercial e-commerce grade.

NON-NEGOTIABLE: 1. Face = 100% identical to reference. 2. Garment = 100% identical to reference. 3. Background = 100% identical to reference."""


# ---------------------------------------------------------------------------
# SeedDream submit + poll
# ---------------------------------------------------------------------------

def _submit_task(prompt: str, image_urls: List[str], pose_label: str) -> str:
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
    resp = requests.post(_CREATE_URL, headers=headers, data=payload, timeout=30)
    resp.raise_for_status()
    task_id = resp.json().get("data", {}).get("taskId")
    if not task_id:
        raise RuntimeError(f"No taskId returned: {resp.text}")
    logger.info("[%s] Task submitted — task_id=%s", pose_label, task_id)
    return task_id


def _poll_task(task_id: str, pose_label: str) -> str:
    logger.info("[%s] Polling SeedDream task_id=%s ...", pose_label, task_id)
    headers = {"Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}"}
    for attempt in range(1, settings.SEEDDREAM_MAX_RETRIES + 1):
        try:
            resp = requests.get(f"{_STATUS_URL}?taskId={task_id}", headers=headers, timeout=30)
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
            time.sleep(settings.SEEDDREAM_RETRY_DELAY)
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("[%s] Poll #%d error: %s", pose_label, attempt, exc)
            time.sleep(settings.SEEDDREAM_RETRY_DELAY)
    raise RuntimeError(f"SeedDream task timed out after {settings.SEEDDREAM_MAX_RETRIES} attempts.")


# ---------------------------------------------------------------------------
# Image download + resize + deblur helpers
# ---------------------------------------------------------------------------

def _download_bytes(url: str, label: str = "") -> bytes:
    logger.info("[%s] Downloading image from URL...", label)
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, stream=True, timeout=(10, 120))
            resp.raise_for_status()
            buf = io.BytesIO()
            for chunk in resp.iter_content(65536):
                if chunk:
                    buf.write(chunk)
            data = buf.getvalue()
            logger.info("[%s] Image downloaded — %d bytes", label, len(data))
            return data
        except Exception as exc:
            logger.warning("[%s] Download attempt %d/3 failed: %s", label, attempt, exc)
            if attempt == 3:
                raise RuntimeError(f"Failed to download image after 3 attempts: {exc}")
            time.sleep(3)




# ---------------------------------------------------------------------------
# Single pose worker
# ---------------------------------------------------------------------------

def _process_one_pose(
    pose_idx:      int,
    pose_prompt:   str,
    image_urls:    List[str],
    photoshoot_id: str,
    req_snapshot:  dict,
) -> dict:
    """
    Full pipeline for one pose. Returns output_image dict or raises on failure.
    """
    pose_label = f"pose-{pose_idx:02d}"
    image_id   = str(uuid.uuid4())
    logger.info("[%s] ── Starting pose pipeline ──────────────────────", pose_label)
    logger.info("[%s] Pose prompt:\n%s", pose_label, pose_prompt)

    # 2.3 — build generation prompt
    logger.info("[%s] Building SeedDream generation prompt...", pose_label)
    has_back = bool(req_snapshot.get("back_garment_image", ""))
    prompt   = _build_photoshoot_prompt(pose_prompt, has_back, req_snapshot)
    logger.info("[%s] Generation prompt built (%d chars, back_image=%s)", pose_label, len(prompt), has_back)
    logger.info("[%s] Generation prompt:\n%s", pose_label, prompt)

    # 2.4-2.5 — submit task
    task_id = _submit_task(prompt, image_urls, pose_label)

    # 2.6 — poll until done
    result_url = _poll_task(task_id, pose_label)
    logger.info("[%s] Result URL received", pose_label)

    # 2.7 — download original image
    image_bytes = _download_bytes(result_url, pose_label)

    # Upload single image to S3
    prefix    = f"photoshoots/{photoshoot_id}/{image_id}"
    s3_key    = f"{prefix}.png"
    logger.info("[%s] Uploading image to S3 (key=%s)...", pose_label, s3_key)
    image_url = upload_bytes_to_s3(image_bytes, s3_key, "image/png")
    logger.info("[%s] Uploaded to S3: %s", pose_label, image_url)

    logger.info("[%s] ── Pose pipeline COMPLETE ─────────────────────", pose_label)
    return {
        "image_id":    image_id,
        "pose_prompt": pose_prompt,
        "image_url":   image_url,
    }


# ---------------------------------------------------------------------------
# Credit deduction helper (inline — avoids importing credit_service circular)
# ---------------------------------------------------------------------------

def _deduct_photoshoot_credits(user_id: str, total_credit: float, photoshoot_id: str) -> None:
    logger.info("[credits] Deducting %.2f credits from user_id=%s for photoshoot=%s",
                total_credit, user_id, photoshoot_id)
    users_col   = get_users_collection()
    history_col = get_credit_history_collection()

    user = users_col.find_one({"user_id": user_id})
    if not user:
        logger.error("[credits] User not found: %s — skipping credit deduction", user_id)
        return

    old_credits = float(user.get("credits", 0))
    new_credits = round(old_credits - total_credit, 4)
    users_col.update_one(
        {"user_id": user_id},
        {"$set": {"credits": new_credits, "updated_at": datetime.now(timezone.utc)}},
    )
    logger.info("[credits] Credits updated: %.4f → %.4f", old_credits, new_credits)

    history_col.insert_one({
        "history_id":      str(uuid.uuid4()),
        "user_id":         user_id,
        "feature_name":    "photoshoot_generation",
        "credit":          total_credit,
        "type":            "deduct",
        "thumbnail_image": "",
        "notes":           f"Photoshoot {photoshoot_id}",
        "created_at":      datetime.now(timezone.utc),
    })
    logger.info("[credits] Credit history record inserted")


# ---------------------------------------------------------------------------
# Public background job entry point
# ---------------------------------------------------------------------------

def run_photoshoot_job(photoshoot_id: str, req: dict) -> None:
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
        pose_prompts = resolve_pose_prompts(
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
        bg_doc = get_backgrounds_collection().find_one({"background_id": req["background_id"]})
        if not bg_doc:
            raise ValueError(f"Background not found: {req['background_id']}")
        background_url = bg_doc["background_url"]
        logger.info("[job] Background found: %s", background_url[:80])

        mf_doc = get_model_faces_collection().find_one({"model_id": req["model_id"]})
        if not mf_doc:
            raise ValueError(f"Model face not found: {req['model_id']}")
        model_face_url = mf_doc["face_url"]
        logger.info("[job] Model face found: %s", model_face_url[:80])

        # ── Build ordered image_urls list ─────────────────────────────────
        image_urls = [req["front_garment_image"]]
        if req.get("back_garment_image"):
            image_urls.append(req["back_garment_image"])
            logger.info("[job] Back garment image included")
        image_urls.append(model_face_url)
        image_urls.append(background_url)
        logger.info("[job] STEP 2 DONE — %d reference images assembled", len(image_urls))

        # ── Step 3: process all poses concurrently ────────────────────────
        logger.info("[job] STEP 3 — Launching %d pose(s) concurrently (max 5 workers)...", len(pose_prompts))
        output_images = []
        failed_poses  = []

        with ThreadPoolExecutor(max_workers=min(len(pose_prompts), 5)) as executor:
            futures = {
                executor.submit(
                    _process_one_pose,
                    idx,
                    prompt,
                    image_urls,
                    photoshoot_id,
                    req,
                ): idx
                for idx, prompt in enumerate(pose_prompts, 1)
            }

            completed_count = 0
            for future in as_completed(futures):
                pose_idx = futures[future]
                try:
                    result = future.result()
                    output_images.append(result)
                    completed_count += 1
                    logger.info("[job] pose-%02d SUCCEEDED (%d/%d complete)",
                                pose_idx, completed_count, len(pose_prompts))
                except Exception as exc:
                    failed_poses.append({"pose_index": pose_idx, "error": str(exc)})
                    logger.error("[job] pose-%02d FAILED: %s", pose_idx, exc)

        logger.info("[job] STEP 3 DONE — %d succeeded, %d failed",
                    len(output_images), len(failed_poses))

        # ── Step 4: deduct credits ────────────────────────────────────────
        logger.info("[job] STEP 4 — Deducting credits...")
        total_credit = len(pose_prompts) * _PHOTOSHOOT_CREDIT_PER_POSE
        _deduct_photoshoot_credits(req["user_id"], total_credit, photoshoot_id)
        logger.info("[job] STEP 4 DONE — %.2f credits deducted", total_credit)

        # ── Step 5: update photoshoot document ───────────────────────────
        logger.info("[job] STEP 5 — Updating photoshoot document in DB...")
        final_status = "completed" if not failed_poses else "partial"
        col.update_one(
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
        col.update_one(
            {"photoshoot_id": photoshoot_id},
            {"$set": {
                "status":     "failed",
                "error":      str(exc),
                "updated_at": datetime.now(timezone.utc),
            }},
        )
