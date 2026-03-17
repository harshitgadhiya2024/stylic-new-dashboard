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
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import List

import requests
from fastapi import HTTPException, status
from PIL import Image

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
    col    = get_poses_collection()
    prompts = []
    for pid in pose_ids:
        doc = col.find_one({"pose_id": pid})
        if doc and doc.get("pose_prompt"):
            prompts.append(doc["pose_prompt"])
        else:
            prompts.append(f"Standing in a natural, relaxed fashion model pose — pose id: {pid}")
    return prompts


def _generate_pose_prompt_from_image(image_url: str) -> str:
    """
    Send a custom pose image to Gemini vision and extract ONLY body position
    and pose details — no gender, age, clothing, or background details.
    """
    try:
        from google import genai
        from google.genai import types as gtypes
    except ImportError as exc:
        return f"Natural fashion model pose — Gemini unavailable: {exc}"

    try:
        img_resp = requests.get(image_url, timeout=30)
        img_resp.raise_for_status()
        img_bytes = img_resp.content
        content_type = img_resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
        if content_type not in ("image/jpeg", "image/png", "image/webp"):
            content_type = "image/jpeg"
    except Exception as exc:
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
        return response.candidates[0].content.parts[0].text.strip()
    except Exception as exc:
        return f"Natural standing fashion model pose — vision error: {exc}"


def resolve_pose_prompts(
    which_pose_option: str,
    poses_ids: List[str],
    poses_images: List[str],
    poses_prompts: List[str],
) -> List[str]:
    if which_pose_option == "default":
        return _fetch_default_pose_prompts(poses_ids)
    elif which_pose_option == "custom":
        return [_generate_pose_prompt_from_image(url) for url in poses_images]
    else:
        return list(poses_prompts)


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

def _submit_task(prompt: str, image_urls: List[str]) -> str:
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
    return task_id


def _poll_task(task_id: str) -> str:
    headers = {"Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}"}
    for _ in range(settings.SEEDDREAM_MAX_RETRIES):
        try:
            resp = requests.get(f"{_STATUS_URL}?taskId={task_id}", headers=headers, timeout=30)
            resp.raise_for_status()
            data  = resp.json().get("data", {})
            state = data.get("state")
            if state == "success":
                urls = json.loads(data.get("resultJson", "{}")).get("resultUrls", [])
                if urls:
                    return urls[0]
                raise RuntimeError("Task succeeded but no resultUrls found.")
            if state == "fail":
                raise RuntimeError("SeedDream task failed.")
            time.sleep(settings.SEEDDREAM_RETRY_DELAY)
        except RuntimeError:
            raise
        except Exception:
            time.sleep(settings.SEEDDREAM_RETRY_DELAY)
    raise RuntimeError(f"SeedDream task timed out after {settings.SEEDDREAM_MAX_RETRIES} attempts.")


# ---------------------------------------------------------------------------
# Image download + resize + deblur helpers
# ---------------------------------------------------------------------------

def _download_bytes(url: str) -> bytes:
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, stream=True, timeout=(10, 120))
            resp.raise_for_status()
            buf = io.BytesIO()
            for chunk in resp.iter_content(65536):
                if chunk:
                    buf.write(chunk)
            return buf.getvalue()
        except Exception as exc:
            if attempt == 3:
                raise RuntimeError(f"Failed to download image after 3 attempts: {exc}")
            time.sleep(3)


def _resize_image(original_bytes: bytes, max_dimension: int) -> bytes:
    img = Image.open(io.BytesIO(original_bytes)).convert("RGB")
    w, h = img.size
    scale = max_dimension / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# Deblur globals — loaded once, shared across threads via a lock
_deblur_restorer   = None
_deblur_load_lock  = threading.Lock()
_deblur_infer_lock = threading.Lock()


def _get_deblur_restorer():
    global _deblur_restorer
    if _deblur_restorer is not None:
        return _deblur_restorer
    with _deblur_load_lock:
        if _deblur_restorer is not None:
            return _deblur_restorer
        try:
            import importlib, subprocess, sys, urllib.request, os

            for pkg, mod in [
                ("gfpgan", "gfpgan"), ("basicsr", "basicsr"),
                ("facexlib", "facexlib"), ("realesrgan", "realesrgan"),
            ]:
                try:
                    importlib.import_module(mod)
                except ImportError:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

            try:
                import torchvision.transforms.functional_tensor  # noqa
            except (ImportError, ModuleNotFoundError):
                import torchvision.transforms.functional as _ftf
                sys.modules["torchvision.transforms.functional_tensor"] = _ftf

            from gfpgan import GFPGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            weights_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
            os.makedirs(weights_dir, exist_ok=True)

            gfpgan_path  = os.path.join(weights_dir, "GFPGANv1.4.pth")
            esrgan_path  = os.path.join(weights_dir, "RealESRGAN_x2plus.pth")

            if not os.path.exists(gfpgan_path):
                urllib.request.urlretrieve(
                    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                    gfpgan_path,
                )
            if not os.path.exists(esrgan_path):
                urllib.request.urlretrieve(
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
                    esrgan_path,
                )

            bg_upsampler = RealESRGANer(
                scale=2,
                model_path=esrgan_path,
                model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                              num_block=23, num_grow_ch=32, scale=2),
                tile=512, tile_pad=16, pre_pad=0, half=False, device="cpu",
            )
            _deblur_restorer = GFPGANer(
                model_path=gfpgan_path,
                upscale=2,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=bg_upsampler,
                device="cpu",
            )
        except Exception:
            _deblur_restorer = None
    return _deblur_restorer


def _deblur_bytes(image_bytes: bytes) -> bytes:
    """Deblur image bytes using GFPGAN. Returns sharpened bytes or original if deblur fails."""
    restorer = _get_deblur_restorer()
    if restorer is None:
        return image_bytes
    try:
        import cv2
        import numpy as np

        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img       = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return image_bytes

        with _deblur_infer_lock:
            _, _, sharp = restorer.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=0.5,
            )

        _, encoded = cv2.imencode(".png", sharp)
        return encoded.tobytes()
    except Exception:
        return image_bytes


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
    image_id = str(uuid.uuid4())

    # 2.3 — build prompt
    has_back = bool(req_snapshot.get("back_garment_image", ""))
    prompt   = _build_photoshoot_prompt(pose_prompt, has_back, req_snapshot)

    # 2.4-2.5 — submit + get task_id
    task_id = _submit_task(prompt, image_urls)

    # 2.6 — poll
    result_url_4k = _poll_task(task_id)

    # 2.7 — download 4K
    bytes_4k = _download_bytes(result_url_4k)

    # 2.8 — resize to 2K and 1K
    bytes_2k = _resize_image(bytes_4k, 2048)
    bytes_1k = _resize_image(bytes_4k, 1024)

    # 2.9-2.11 — deblur all three
    deblurred_1k = _deblur_bytes(bytes_1k)
    deblurred_2k = _deblur_bytes(bytes_2k)
    deblurred_4k = _deblur_bytes(bytes_4k)

    prefix = f"photoshoots/{photoshoot_id}/{image_id}"

    url_4k_orig = upload_bytes_to_s3(bytes_4k,      f"{prefix}_4k.png",          "image/png")
    url_2k_orig = upload_bytes_to_s3(bytes_2k,      f"{prefix}_2k.png",          "image/png")
    url_1k_orig = upload_bytes_to_s3(bytes_1k,      f"{prefix}_1k.png",          "image/png")
    url_4k_dblr = upload_bytes_to_s3(deblurred_4k,  f"{prefix}_4k_deblurred.png", "image/png")
    url_2k_dblr = upload_bytes_to_s3(deblurred_2k,  f"{prefix}_2k_deblurred.png", "image/png")
    url_1k_dblr = upload_bytes_to_s3(deblurred_1k,  f"{prefix}_1k_deblurred.png", "image/png")

    return {
        "image_id":   image_id,
        "pose_prompt": pose_prompt,
        "images": {
            "1k": [url_1k_orig, url_1k_dblr],
            "2k": [url_2k_orig, url_2k_dblr],
            "4k": [url_4k_orig, url_4k_dblr],
        },
    }


# ---------------------------------------------------------------------------
# Credit deduction helper (inline — avoids importing credit_service circular)
# ---------------------------------------------------------------------------

def _deduct_photoshoot_credits(user_id: str, total_credit: float, photoshoot_id: str) -> None:
    users_col   = get_users_collection()
    history_col = get_credit_history_collection()

    user = users_col.find_one({"user_id": user_id})
    if not user:
        return

    new_credits = round(float(user.get("credits", 0)) - total_credit, 4)
    users_col.update_one(
        {"user_id": user_id},
        {"$set": {"credits": new_credits, "updated_at": datetime.now(timezone.utc)}},
    )

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


# ---------------------------------------------------------------------------
# Public background job entry point
# ---------------------------------------------------------------------------

def run_photoshoot_job(photoshoot_id: str, req: dict) -> None:
    """
    Called as a FastAPI BackgroundTask. Runs the full photoshoot pipeline,
    updates the photoshoot document, deducts credits.
    """
    col = get_photoshoots_collection()

    try:
        # ── Step 1: resolve pose prompts ──────────────────────────────────
        pose_prompts = resolve_pose_prompts(
            which_pose_option=req["which_pose_option"],
            poses_ids=req.get("poses_ids") or [],
            poses_images=req.get("poses_images") or [],
            poses_prompts=req.get("poses_prompts") or [],
        )

        if not pose_prompts:
            raise ValueError("No pose prompts could be resolved.")

        # ── Step 2: fetch background and model face URLs ──────────────────
        bg_doc = get_backgrounds_collection().find_one({"background_id": req["background_id"]})
        if not bg_doc:
            raise ValueError(f"Background not found: {req['background_id']}")
        background_url = bg_doc["background_url"]

        mf_doc = get_model_faces_collection().find_one({"model_id": req["model_id"]})
        if not mf_doc:
            raise ValueError(f"Model face not found: {req['model_id']}")
        model_face_url = mf_doc["face_url"]

        # ── Build ordered image_urls list ─────────────────────────────────
        # Order: front_garment, [back_garment], model_face, background
        image_urls = [req["front_garment_image"]]
        if req.get("back_garment_image"):
            image_urls.append(req["back_garment_image"])
        image_urls.append(model_face_url)
        image_urls.append(background_url)

        # ── Step 3: process all poses concurrently ────────────────────────
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

            for future in as_completed(futures):
                pose_idx = futures[future]
                try:
                    result = future.result()
                    output_images.append(result)
                except Exception as exc:
                    failed_poses.append({"pose_index": pose_idx, "error": str(exc)})

        # ── Step 4: deduct credits ────────────────────────────────────────
        total_credit = len(pose_prompts) * _PHOTOSHOOT_CREDIT_PER_POSE
        _deduct_photoshoot_credits(req["user_id"], total_credit, photoshoot_id)

        # ── Step 5: update photoshoot document ───────────────────────────
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

    except Exception as exc:
        col.update_one(
            {"photoshoot_id": photoshoot_id},
            {"$set": {
                "status":     "failed",
                "error":      str(exc),
                "updated_at": datetime.now(timezone.utc),
            }},
        )
