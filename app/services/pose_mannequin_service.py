"""
Pose mannequin pipeline (from pose_to_mannequin.py), adapted for the API.

1. gemini-2.5-flash-image → mannequin PNG from reference photo
2. gemini-2.5-flash → pose-only prompt from mannequin PNG
3. S3 upload → public URL

Also: text-only mannequin generation from a written pose description (API-3).
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import uuid
from typing import AsyncGenerator, Tuple

import httpx
from google import genai
from google.genai import types as gtypes
from PIL import Image

from app.config import settings
from app.services.s3_service import upload_bytes_to_s3

logger = logging.getLogger("pose_mannequin")

MAX_RETRIES = 3
RETRY_DELAY = 5.0
MANNEQUIN_PREFIX = "mannequin-output"

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


def _to_png_bytes(data: bytes) -> bytes:
    buf = io.BytesIO()
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        img = Image.open(io.BytesIO(base64.b64decode(data)))
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _generate_mannequin_from_bytes(image_bytes: bytes, mime_type: str, label: str) -> bytes:
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    parts = [
        gtypes.Part.from_bytes(mime_type=mime_type, data=image_bytes),
        gtypes.Part.from_text(text=MANNEQUIN_PROMPT),
    ]
    config = gtypes.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])
    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=[gtypes.Content(role="user", parts=parts)],
        config=config,
    )
    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.data:
            return part.inline_data.data
        if part.text:
            logger.info("[%s] mannequin model note: %s", label, part.text[:120])
    raise RuntimeError("Gemini returned no image for mannequin generation.")


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


def sync_image_url_to_mannequin_and_prompt(image_bytes: bytes, mime_type: str) -> Tuple[bytes, str]:
    """Return (png_bytes, pose_prompt)."""
    raw = _with_retry_sync(
        lambda: _generate_mannequin_from_bytes(image_bytes, mime_type, "mannequin"),
        "mannequin generation",
    )
    png = _to_png_bytes(raw)
    pose_prompt = _with_retry_sync(
        lambda: _generate_pose_prompt_from_png(png, "pose_prompt"),
        "pose prompt",
    )
    return png, pose_prompt


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


def sync_generate_mannequin_png_from_text_pose(pose_prompt: str, pose_type: str) -> bytes:
    """Generate mannequin image from written pose only; returns PNG bytes."""

    def _run():
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        text = TEXT_MANNEQUIN_PROMPT_TEMPLATE.format(
            pose_type=pose_type,
            pose_prompt=pose_prompt.strip(),
        )
        parts = [gtypes.Part.from_text(text=text)]
        config = gtypes.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[gtypes.Content(role="user", parts=parts)],
            config=config,
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                return _to_png_bytes(part.inline_data.data)
            if part.text:
                logger.info("[text-mannequin] model note: %s", part.text[:120])
        raise RuntimeError("Gemini returned no image for text-based mannequin.")

    return _with_retry_sync(_run, "text mannequin")


async def download_image(url: str) -> tuple[bytes, str]:
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    ctype = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    if ctype not in ("image/jpeg", "image/png", "image/webp", "image/avif"):
        ctype = "image/jpeg"
    return resp.content, ctype


async def upload_mannequin_png(png_bytes: bytes) -> str:
    key = f"{MANNEQUIN_PREFIX}/{uuid.uuid4()}.png"
    return await upload_bytes_to_s3(png_bytes, key, content_type="image/png")


async def stream_pose_from_image_url(image_url: str) -> AsyncGenerator[tuple, None]:
    """Yield ``("progress", msg)`` then ``("done", mannequin_url, pose_prompt)``."""
    yield ("progress", "Downloading reference image…")
    image_bytes, mime = await download_image(image_url)

    loop = asyncio.get_running_loop()

    yield ("progress", "Generating mannequin (Gemini image)…")
    png_bytes, pose_prompt = await loop.run_in_executor(
        None,
        lambda: sync_image_url_to_mannequin_and_prompt(image_bytes, mime),
    )

    yield ("progress", "Uploading mannequin to S3…")
    url = await upload_mannequin_png(png_bytes)

    yield ("done", url, pose_prompt)


async def stream_pose_from_text_prompt(
    pose_prompt: str,
    pose_type: str,
) -> AsyncGenerator[tuple, None]:
    yield ("progress", "Generating mannequin from pose description…")
    loop = asyncio.get_running_loop()
    png_bytes = await loop.run_in_executor(
        None,
        lambda: sync_generate_mannequin_png_from_text_pose(pose_prompt, pose_type),
    )
    yield ("progress", "Uploading mannequin to S3…")
    url = await upload_mannequin_png(png_bytes)
    yield ("done", url)
