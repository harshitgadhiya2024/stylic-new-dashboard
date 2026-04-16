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

POSE_PROMPT_INSTRUCTION = """\
You are an expert prompt engineer for AI image generation.
Examine the mannequin image carefully and write a precise, high-quality
pose description to be embedded directly inside an image generation prompt.

STRICT OUTPUT RULES:
1. Describe ONLY: overall body stance, torso angle, shoulder alignment, hip
   position, head tilt/turn, neck angle, arm positions, elbow angles, wrist
   angles, hand placement, finger positions (if visible), leg stance, knee
   bend, foot placement and direction, weight distribution, and centre of gravity.
2. NEVER mention: background, clothing, colours, gender, hair, skin tone,
   lighting, shadows, accessories, jewellery, facial features, or any objects.
3. Format: dense, specific, comma-separated descriptive phrases — no prose sentences.
4. Length: 70–120 words exactly.
5. Begin directly with the pose (no preamble such as "The pose shows…").

Return ONLY the pose description — no other text.
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


def _generate_pose_prompt_from_png(png_bytes: bytes, label: str) -> str:
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    parts = [
        gtypes.Part.from_bytes(mime_type="image/png", data=png_bytes),
        gtypes.Part.from_text(text=POSE_PROMPT_INSTRUCTION),
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

MANNEQUIN APPEARANCE:
- Solid medium grey skin (#9E9E9E), smooth matte, featureless face (no eyes, nose, mouth, ears), completely bald.
- Male mannequin proportions.

CLOTHING (only these):
- Plain white half-sleeve t-shirt, plain dark charcoal grey jeans. No other garments, no accessories, no footwear.

BACKGROUND:
- Pure white (#FFFFFF) only, no gradients.

POSE AND VIEW:
- Pose type context: {pose_type} — front means facing camera, side means profile, back means back toward camera.
- The mannequin MUST display this exact body pose (replicate precisely):
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


async def stream_pose_from_image_url(image_url: str) -> AsyncGenerator[tuple, None]:
    """Yield ``("progress", msg)`` then ``("done", mannequin_url, pose_prompt)``."""
    yield ("progress", "Generating mannequin (SeedDream)…")
    raw = await _seedream_mannequin_from_image_url_with_retries(image_url)
    png_bytes = _to_png_bytes(raw)

    yield ("progress", "Deriving pose description…")
    loop = asyncio.get_running_loop()
    pose_prompt = await loop.run_in_executor(
        None,
        lambda: _with_retry_sync(
            lambda: _generate_pose_prompt_from_png(png_bytes, "pose_prompt"),
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
    yield ("progress", "Uploading mannequin to R2…")
    url = await upload_mannequin_png(png_bytes)
    yield ("done", url)
