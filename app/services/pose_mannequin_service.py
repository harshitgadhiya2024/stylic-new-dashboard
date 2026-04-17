"""
Pose mannequin pipeline (from pose_to_mannequin.py), adapted for the API.

1. SeedDream 5.0 Lite (image-to-image) → mannequin PNG from reference photo URL
2. Gemini flash (vision) → pose-only prompt from mannequin PNG
3. R2 upload → public URL

Also: text-only mannequin generation from a written pose description (API-3)
via SeedDream 5.0 Lite text-to-image.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import uuid
from typing import AsyncGenerator

import httpx
from google import genai
from google.genai import types as gtypes
from PIL import Image

from app.config import settings
from app.services.r2_service import upload_bytes_to_r2

logger = logging.getLogger("pose_mannequin")

MAX_RETRIES = 3
RETRY_DELAY = 5.0
MANNEQUIN_PREFIX = "mannequin-output"

_CREATE_URL = "https://api.kie.ai/api/v1/jobs/createTask"
_STATUS_URL = "https://api.kie.ai/api/v1/jobs/recordInfo"

MANNEQUIN_PROMPT = """\
You are a professional fashion-tech image editor.
Convert the provided photograph of a person into a clean mannequin image.
Follow every requirement below without exception.

MANNEQUIN APPEARANCE:
- Solid medium grey skin (#9E9E9E). Smooth matte finish, no skin texture.
- Completely smooth, featureless face — absolutely no eyes, nose, mouth, ears, or any facial detail.
- Completely bald — no hair, no stubble, no hairline. Smooth, rounded head.
- Male mannequin body form with masculine proportions.
- Body proportions stay identical to the person in the photo.
- Skin color must remain grey across all visible skin/mannequin areas (no skin-tone variation).

GARMENTS AND ACCESSORIES (critical):
- The mannequin must wear ONLY these two garments:
  1) Plain white half-sleeve t-shirt (solid white, no print, no logo, no texture graphics).
  2) Plain dark grey jeans (solid dark charcoal grey, no print, no logo, no pattern).
- Do NOT include any other garments or layers: no jacket, hoodie, shirt, vest, blazer,
  sweater, shorts, joggers, innerwear visibility, or extra fabric.
- Do NOT include any accessories: no belt, jewellery, watch, hat, bag, scarf, or ornament.
- Do NOT include footwear or socks.
- Preserve clean garment fit while keeping the exact original body pose.

POSE (critical):
- Replicate the EXACT pose from the original: every joint angle, limb position,
  weight shift, hand placement, foot placement, torso lean, and head tilt.
- Do NOT simplify, straighten, or alter the pose in any way.

FRAMING / CROP LOCK (critical):
- Preserve the EXACT camera framing from the input photo.
- If the input is a close-up crop (face/neck/shoulder/chest/waist detail), keep it as close-up.
- If the input shows only upper half or torso details, do NOT zoom out to full body.
- Keep the same visible garment regions and crop boundaries as the source image.
- Maintain the same camera distance/composition so garment detailing visibility remains equivalent.
- Never invent hidden body parts outside the original frame.

BACKGROUND:
- Critical: background must be ONLY pure white (#FFFFFF).
- Do not use any other background color.
- No gradients, color casts, textures, reflections, or floor lines.
- Subject perfectly centred in frame.

SCENE OBJECTS / PROPS:
- Prefer removing props entirely when possible.
- If a prop is essential to preserve the exact pose (for example: table, chair, stool, bench, support object),
  keep it minimal and render it in solid black (#000000) so it is clearly visible.
- Do not introduce any new props or decorative objects.

OUTPUT:
- Clean, sharp, fashion-industry mannequin photograph.
- No watermarks, text, or overlays.
"""

POSE_PROMPT_INSTRUCTION_TEMPLATE = """\
You are an expert prompt engineer for AI image generation.
Examine the mannequin image carefully. The user selected pose orientation VIEW = **{pose_type}** (front = facing camera, side = profile, back = back toward camera).

Your reply will be stored and reused as a pose prompt. It MUST lead with explicit shoot / framing metadata, then the body pose.

MANDATORY LINE 1 — start your entire output with this exact prefix (fill the bracketed choices from the image only):
Output tags: FRAMING = <choose exactly one: full body | upper half body | lower half body | head and shoulders | close-up upper garment fabric detail>; VIEW = <front | back | side — use **{pose_type}** unless the mannequin clearly faces another way, then pick the closest>; FOCUS = <choose exactly one: full-body pose | upper-body garment drape and fabric detailing pose | lower-body stance | head and neck region>.

MANDATORY LINE 2 — blank line after line 1, then dense comma-separated pose-only phrases (no prose sentences): stance, torso angle, shoulders, hips, head tilt/turn, neck, arms, elbows, wrists, hands/fingers if visible, legs/knees/feet only if visible in frame, weight distribution.

STRICT RULES for line 2:
- NEVER mention: background, clothing colours, garment logos, gender, hair, skin tone, lighting, shadows, accessories, jewellery, facial features, or props.
- Line 2 length: 70–120 words.
- Do not repeat line 1 inside line 2.

Return ONLY line 1, one blank line, then line 2 — no other text.
"""


def _ensure_seedream_configured() -> None:
    if not (settings.SEEDDREAM_API_KEY or "").strip():
        raise RuntimeError("SeedDream is not configured (SEEDDREAM_API_KEY).")


def _to_png_bytes(data: bytes) -> bytes:
    buf = io.BytesIO()
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        img = Image.open(io.BytesIO(base64.b64decode(data)))
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _generate_pose_prompt_from_png(
    png_bytes: bytes,
    label: str,
    *,
    pose_type: str = "front",
) -> str:
    pt = (pose_type or "front").strip().lower()
    if pt not in ("front", "back", "side"):
        pt = "front"
    instruction = POSE_PROMPT_INSTRUCTION_TEMPLATE.format(pose_type=pt)
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    parts = [
        gtypes.Part.from_bytes(mime_type="image/png", data=png_bytes),
        gtypes.Part.from_text(text=instruction),
    ]
    response = client.models.generate_content(
        model=settings.GEMINI_VISION_MODEL,
        contents=[gtypes.Content(role="user", parts=parts)],
        config=gtypes.GenerateContentConfig(
            response_modalities=["TEXT"],
            temperature=0.2,
        ),
    )
    for part in response.candidates[0].content.parts:
        if part.text:
            return part.text.strip()
    raise RuntimeError("Gemini returned no pose prompt.")


def _with_retry_sync(fn, label: str):
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            last_err = exc
            logger.warning("[%s] attempt %s/%s failed: %s", label, attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                import time
                time.sleep(RETRY_DELAY)
    raise last_err  # type: ignore[misc]


TEXT_MANNEQUIN_PROMPT_TEMPLATE = """\
You are a professional fashion-tech image generator.
Create a single clean photograph of a fashion mannequin (not a real person).

OUTPUT — READ FIRST (camera, framing, and view — obey before all else):
- Set exactly ONE framing to match the written pose: **full body** (head to feet in frame); OR **upper half body** (waist-up / bust-up, no full legs); OR **lower half body** (waist-down); OR **head and shoulders**; OR **close-up upper garment fabric detail** (tight crop for neckline/chest/fabric fold behaviour, not a full-length shot).
- If the description is about bust, portrait, torso, blouse detail, or fabric fold, you MUST use upper half body or close-up upper garment fabric detail — NOT full body.
- If the description clearly covers stance from head through feet, use **full body**.
- Body orientation / VIEW: **{pose_type}** — front = facing camera, side = 90° profile, back = back toward camera.
- Do not contradict the chosen framing: do not pull back to full body when FRAMING is upper half or close-up fabric detail.

MANNEQUIN APPEARANCE:
- Solid medium grey skin (#9E9E9E), smooth matte, featureless face (no eyes, nose, mouth, ears), completely bald.
- Male mannequin proportions.

CLOTHING (only these):
- Plain white half-sleeve t-shirt, plain dark charcoal grey jeans. No other garments, no accessories, no footwear.

BACKGROUND:
- Pure white (#FFFFFF) only, no gradients.

POSE AND VIEW:
- The mannequin MUST display this exact body pose under the OUTPUT framing rules above (replicate precisely):
{pose_prompt}

OUTPUT: One sharp fashion mannequin photo, centred, no watermark or text.
"""


async def _submit_kie_task(model: str, input_payload: dict) -> str:
    body = json.dumps({"model": model, "input": input_payload})
    headers = {
        "Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(_CREATE_URL, headers=headers, content=body)
        resp.raise_for_status()
    task_id = resp.json().get("data", {}).get("taskId")
    if not task_id:
        raise RuntimeError("Mannequin generation could not be started (no task id).")
    return task_id


async def _poll_kie_task(task_id: str) -> str:
    headers = {"Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}"}
    for _ in range(1, settings.SEEDDREAM_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{_STATUS_URL}?taskId={task_id}",
                    headers=headers,
                )
                resp.raise_for_status()
            data = resp.json().get("data", {})
            state = data.get("state")
            if state == "success":
                result_urls = json.loads(data.get("resultJson", "{}")).get("resultUrls", [])
                if result_urls:
                    return result_urls[0]
                raise RuntimeError("Mannequin generation finished but no image URL was returned.")
            if state == "fail":
                raise RuntimeError("Mannequin image generation failed.")
            await asyncio.sleep(settings.SEEDDREAM_RETRY_DELAY)
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("[mannequin] poll error: %s", exc)
            await asyncio.sleep(settings.SEEDDREAM_RETRY_DELAY)
    raise RuntimeError("Mannequin image generation timed out.")


async def _download_result_image(result_url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.get(result_url)
                resp.raise_for_status()
            return resp.content
        except Exception as exc:
            last = exc
            if attempt < 3:
                await asyncio.sleep(3)
    raise RuntimeError(f"Failed to download mannequin image: {last}")


async def _run_seedream_mannequin_once(
    model: str,
    input_payload: dict,
) -> bytes:
    _ensure_seedream_configured()
    inp = {**input_payload, "nsfw_checker": False}
    task_id = await _submit_kie_task(model, inp)
    result_url = await _poll_kie_task(task_id)
    return await _download_result_image(result_url)


async def _seedream_mannequin_from_image_url_with_retries(image_url: str) -> bytes:
    payload = {
        "prompt": MANNEQUIN_PROMPT,
        "image_urls": [image_url],
        "aspect_ratio": settings.POSE_MANNEQUIN_SEEDREAM_ASPECT,
        "quality": settings.POSE_MANNEQUIN_SEEDREAM_QUALITY,
    }
    model = settings.POSE_MANNEQUIN_SEEDREAM_IMG2IMG_MODEL
    last: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await _run_seedream_mannequin_once(model, payload)
        except Exception as exc:
            last = exc
            logger.warning(
                "[mannequin img2img] attempt %s/%s failed: %s",
                attempt,
                MAX_RETRIES,
                exc,
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)
    raise last  # type: ignore[misc]


async def _seedream_mannequin_from_text_with_retries(pose_prompt: str, pose_type: str) -> bytes:
    text = TEXT_MANNEQUIN_PROMPT_TEMPLATE.format(
        pose_type=pose_type,
        pose_prompt=pose_prompt.strip(),
    )
    if len(text) > 10000:
        raise RuntimeError("Pose description exceeds the maximum prompt length (10000 characters).")
    payload = {
        "prompt": text,
        "aspect_ratio": settings.POSE_MANNEQUIN_SEEDREAM_ASPECT,
        "quality": settings.POSE_MANNEQUIN_SEEDREAM_QUALITY,
    }
    model = settings.POSE_MANNEQUIN_SEEDREAM_TEXT_MODEL
    last: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await _run_seedream_mannequin_once(model, payload)
        except Exception as exc:
            last = exc
            logger.warning(
                "[mannequin text2img] attempt %s/%s failed: %s",
                attempt,
                MAX_RETRIES,
                exc,
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)
    raise last  # type: ignore[misc]


async def upload_mannequin_png(png_bytes: bytes) -> str:
    key = f"{MANNEQUIN_PREFIX}/{uuid.uuid4()}.png"
    return await upload_bytes_to_r2(png_bytes, key, content_type="image/png")


async def stream_pose_from_image_url(
    image_url: str,
    *,
    pose_type: str = "front",
) -> AsyncGenerator[tuple, None]:
    """Yield ``("progress", msg)`` then ``("done", mannequin_url, pose_prompt)``."""
    yield ("progress", "Generating mannequin (SeedDream)…")
    raw = await _seedream_mannequin_from_image_url_with_retries(image_url)
    png_bytes = _to_png_bytes(raw)

    yield ("progress", "Deriving pose description…")
    loop = asyncio.get_running_loop()
    pt = (pose_type or "front").strip().lower()
    if pt not in ("front", "back", "side"):
        pt = "front"
    pose_prompt = await loop.run_in_executor(
        None,
        lambda: _with_retry_sync(
            lambda: _generate_pose_prompt_from_png(png_bytes, "pose_prompt", pose_type=pt),
            "pose prompt",
        ),
    )

    yield ("progress", "Uploading mannequin to R2…")
    url = await upload_mannequin_png(png_bytes)

    yield ("done", url, pose_prompt)


async def stream_pose_from_text_prompt(
    pose_prompt: str,
    pose_type: str,
) -> AsyncGenerator[tuple, None]:
    yield ("progress", "Generating mannequin from pose description (SeedDream)…")
    raw = await _seedream_mannequin_from_text_with_retries(pose_prompt, pose_type)
    png_bytes = _to_png_bytes(raw)

    yield ("progress", "Deriving pose description…")
    loop = asyncio.get_running_loop()
    pt = (pose_type or "front").strip().lower()
    if pt not in ("front", "back", "side"):
        pt = "front"
    derived_pose_prompt = await loop.run_in_executor(
        None,
        lambda: _with_retry_sync(
            lambda: _generate_pose_prompt_from_png(png_bytes, "pose_prompt_text", pose_type=pt),
            "pose prompt",
        ),
    )

    yield ("progress", "Uploading mannequin to R2…")
    url = await upload_mannequin_png(png_bytes)
    yield ("done", url, derived_pose_prompt)
