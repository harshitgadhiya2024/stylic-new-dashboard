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

VISIBLE BODY PARTS ONLY — HIGHEST PRIORITY (read before all else):
- Show ONLY the same body regions that appear in the input. Frame edges are hard limits — never reveal what was cropped out.
- If the input does NOT show the full head (or shows no head at all), do NOT add a head, face, or hair — keep the crop exactly as implied by the source (neck-up, chin-up, or head fully absent).
- If only ONE arm or ONE hand is visible, render only that arm/hand — do NOT invent a second arm, hand, or symmetrical pose outside the crop.
- Bust-up / upper-chest garment detail / collar-button focus: keep that tight framing — no waist, hips, legs, or feet unless they appear in the source.
- Midsection / waist–hip garment detail (no head in frame): output must also have NO head, NO shoulders above the crop, and NO legs or feet below the crop — same macro crop as the source.
- Head-and-shoulders or over-the-shoulder portrait: match visible shoulders and head coverage only; omit any body parts not in frame.
- Full body in the source only: then you may show head through feet on the mannequin; otherwise NEVER zoom out to full body.
- Do NOT zoom in tighter than the source unless the source is already that tight; do NOT reframe to a standard catalogue full-length shot.

MANNEQUIN APPEARANCE (apply only to visible mannequin areas):
- Solid medium grey skin (#9E9E9E). Smooth matte finish, no skin texture.
- Where the head is INSIDE the crop: completely smooth, featureless face — no eyes, nose, mouth, ears; completely bald, no hairline.
- Where the head is OUTSIDE or cropped off: do not draw a head — let the image end at the crop (e.g. neck, upper chest, or garment edge only).
- Male mannequin segment proportions for visible regions only — match the source silhouette; do not infer a hidden full-body shape.
- Skin color must remain grey across all visible mannequin areas (no skin-tone variation).

GARMENTS AND ACCESSORIES (critical):
- The mannequin must wear ONLY these two garments on every visible body region:
  1) Plain white half-sleeve t-shirt (solid white, no print, no logo, no texture graphics).
  2) Plain dark grey jeans (solid dark charcoal grey, no print, no logo, no pattern).
- Map garments to what the source shows: if only upper torso is visible, only the shirt (and shirt hem over jeans if that is in frame); if only jeans/waist band is visible, show only that portion — do not add the missing garment half as a full new layer beyond the crop.
- Do NOT include any other garments or layers: no jacket, hoodie, shirt, vest, blazer,
  sweater, shorts, joggers, innerwear visibility, or extra fabric.
- Do NOT include any accessories: no belt, jewellery, watch, hat, bag, scarf, or ornament.
- Do NOT include footwear or socks unless feet are clearly visible in the source (then plain neutral feet only, no shoes).
- Preserve clean garment fit and fabric silhouette while keeping the exact visible pose.

POSE (critical — only for limbs and segments that appear in the input):
- Replicate the EXACT pose for every visible joint angle, limb segment, hand placement, torso lean, and head tilt that appears in frame.
- Do NOT invent pose detail for legs, feet, a second arm, or head if those are not visible in the source.

FRAMING / CROP LOCK (critical):
- Preserve the EXACT camera framing, scale, and subject placement from the input photo.
- Garment-detail crops, half-body, one-sided upper body, and macro waist shots must stay that way — never pull back to show a complete mannequin.

BACKGROUND:
- Critical: background must be ONLY pure white (#FFFFFF).
- Do not use any other background color.
- No gradients, color casts, textures, reflections, or floor lines.
- Match composition to the source (do not force recentre if the source subject is off-centre).

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
Output tags: FRAMING = <choose exactly one: full body | upper half body | lower half body | head and shoulders | bust and upper chest (garment detail, may exclude full head if cropped) | midsection waist–hip garment detail (no head in frame) | single-side partial upper body (only one arm/hand visible)>; VIEW = <front | back | side — use **{pose_type}** unless the mannequin clearly faces another way, then pick the closest>; FOCUS = <choose exactly one: full-body pose | upper-body garment drape and fabric detailing pose | lower-body stance | head and neck region | macro garment intersection only>.

MANDATORY LINE 2 — blank line after line 1, then dense comma-separated pose-only phrases (no prose sentences). Describe ONLY anatomy that is VISIBLE in the image: e.g. if no head in frame, do not mention head; if only one arm, describe only that arm; if no legs, do not mention legs or feet.

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

OUTPUT — READ FIRST (camera, framing, visible body parts, and view):
- Choose exactly ONE framing to match the written pose — do NOT default to full body:
  **full body** (head to feet in frame) ONLY if the text clearly describes full stance head-to-toe;
  **upper half body** (waist-up / bust-up, no full legs);
  **lower half body** (waist-down, no full head) when the text is about hips/legs only;
  **head and shoulders** or **bust / upper-chest garment detail** when the text is portrait, collar, buttons, neckline, or knit texture on chest — no waist or legs unless described;
  **midsection waist–hip garment detail** when the text is about hem, jeans intersection, side slit, pocket — NO head above crop and NO feet below crop in the output;
  **single-side partial upper body** when the text describes only one visible arm or over-one-shoulder — do NOT add a second arm or mirror a full torso.
- Only show body parts implied by that framing: if framing omits head, the image must OMIT the head entirely; if only one arm is described, show only that arm.
- If the description is about bust, portrait, torso, blouse detail, or fabric fold, you MUST use bust-up, head-and-shoulders, or close-up upper garment detail — NOT full body.
- Body orientation / VIEW: **{pose_type}** — front = facing camera, side = 90° profile, back = back toward camera.
- Never pull back to reveal a “complete” mannequin when the intent is garment detail or partial body.

MANNEQUIN APPEARANCE:
- Solid medium grey skin (#9E9E9E), smooth matte on all visible mannequin areas.
- Featureless face (no eyes, nose, mouth, ears) and bald head ONLY where the head is inside the chosen frame; if framing excludes the head, do not render a head.
- Male mannequin proportions for visible segments only.

CLOTHING (only these):
- Plain white half-sleeve t-shirt, plain dark charcoal grey jeans, only on visible body regions matching the framing (e.g. midsection crop shows only the relevant shirt hem + jeans band).
- No other garments, no accessories, no footwear unless full legs/feet are in frame (then no shoes, bare mannequin feet only if needed).

BACKGROUND:
- Pure white (#FFFFFF) only, no gradients.

POSE AND VIEW:
- The mannequin MUST display this exact body pose under the OUTPUT rules above (replicate precisely, partial body if partial):
{pose_prompt}

OUTPUT: One sharp fashion mannequin photo matching the chosen crop (not forced full-length), no watermark or text.
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
