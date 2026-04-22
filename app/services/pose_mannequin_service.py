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

# kie.ai SeedDream prompt limit (same cap as photoshoot compact prompts).
POSE_MANNEQUIN_SEEDDREAM_PROMPT_MAX_CHARS = 3000

MANNEQUIN_PROMPT = """\
Fashion-tech editor: convert the person photo into a clean grey mannequin on pure white (#FFFFFF) only.

VISIBILITY (strict — highest priority):
- Only regions visible in the input; crop edges are hard limits — never reveal cropped-out body parts.
- No head/face/hair if head is absent or cut off; end at neck, upper chest, or garment edge like the source.
- One arm/hand visible only — never add a second arm or mirrored torso outside the crop.
- Bust/collar/button/knit-detail crops stay tight — no waist, hips, legs, feet unless they appear in source.
- Waist–hip macro (no head in source): no head/shoulders above crop, no legs/feet below — match macro crop.
- Head-and-shoulders / over-shoulder: only what is in frame. Full head-to-feet mannequin only if source is full body — never zoom out to a catalogue full body.

MANNEQUIN (visible areas only):
- Skin #9E9E9E matte grey, no texture. Featureless face + bald only if head is in frame; if head cropped out, do not draw a head.
- Male segment proportions for visible parts; match source silhouette — do not infer hidden full-body shape.

GARMENTS (visible regions only; no extras):
- Only plain white half-sleeve tee + plain dark charcoal jeans — no prints, logos, patterns, jacket, hoodie, vest, belt, jewellery, hat, bag, scarf, socks, shoes.
- Map to source: upper torso only = shirt (+ hem over jeans if in frame); waist/jeans band only = that band — no invented garment beyond crop.
- Bare neutral mannequin feet only if feet clearly visible in source; otherwise no footwear.

POSE: Replicate exact visible joints, lean, tilt, hands; never invent legs, second arm, or head if not in frame.

FRAMING: Same framing, scale, and placement as source — no forced recentre.

BG #FFFFFF flat; no gradients, floor lines, or cast. PROPS: remove if possible; if essential to pose, minimal solid black (#000000). No new props.

OUTPUT: Sharp mannequin, no watermark or text.
"""

if len(MANNEQUIN_PROMPT) > POSE_MANNEQUIN_SEEDDREAM_PROMPT_MAX_CHARS:
    raise RuntimeError(
        f"MANNEQUIN_PROMPT is {len(MANNEQUIN_PROMPT)} chars; "
        f"max {POSE_MANNEQUIN_SEEDDREAM_PROMPT_MAX_CHARS} for SeedDream."
    )

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
Fashion mannequin (not a real person), pure white #FFFFFF background, no watermark.

FRAMING (choose ONE; never default to full body): full body only if text clearly describes head-to-toe stance; waist-up/bust-up; waist-down (no head); head-and-shoulders or bust/collar/button/knit detail; waist–hip macro (no head above crop, no feet below); one-side partial (one arm only — no mirror arm).

VISIBILITY: Omit head if framing omits it; one arm if only one is described. VIEW: **{pose_type}** (front|side|back). No pull-back to a "complete" mannequin for garment-detail intent.

MANNEQUIN: #9E9E9E matte grey; featureless+bald head only if head is in frame.

CLOTHES: Plain white half-sleeve tee + dark charcoal jeans on visible areas only; no extra garments, belt, jewellery, shoes (bare feet only if feet in frame).

POSE (match written + framing):
{pose_prompt}

Sharp output matching crop, not forced full-length.
"""

_text_mannequin_overhead_max = max(
    len(TEXT_MANNEQUIN_PROMPT_TEMPLATE.format(pose_type=pt, pose_prompt=""))
    for pt in ("front", "back", "side")
)
if _text_mannequin_overhead_max >= POSE_MANNEQUIN_SEEDDREAM_PROMPT_MAX_CHARS:
    raise RuntimeError(
        f"TEXT_MANNEQUIN_PROMPT_TEMPLATE overhead is {_text_mannequin_overhead_max} chars; "
        f"must be <= {POSE_MANNEQUIN_SEEDDREAM_PROMPT_MAX_CHARS} so user pose text can fit."
    )


def _compose_text_mannequin_seeddream_prompt(pose_prompt: str, pose_type: str) -> str:
    """Build text-to-image mannequin prompt; total length <= POSE_MANNEQUIN_SEEDDREAM_PROMPT_MAX_CHARS."""
    pt = (pose_type or "front").strip().lower()
    if pt not in ("front", "back", "side"):
        pt = "front"
    limit = POSE_MANNEQUIN_SEEDDREAM_PROMPT_MAX_CHARS
    pp = pose_prompt.strip()
    overhead = len(TEXT_MANNEQUIN_PROMPT_TEMPLATE.format(pose_type=pt, pose_prompt=""))
    max_pp = max(0, limit - overhead)
    if len(pp) > max_pp:
        if max_pp <= 16:
            pp = pp[:max_pp]
        else:
            pp = pp[: max_pp - 3].rstrip() + "..."
        logger.warning(
            "[mannequin text2img] pose_prompt truncated to %d chars (limit=%d, overhead=%d)",
            len(pp),
            limit,
            overhead,
        )
    text = TEXT_MANNEQUIN_PROMPT_TEMPLATE.format(pose_type=pt, pose_prompt=pp)
    if len(text) > limit:
        # Defensive trim (template edits should keep overhead safe).
        text = text[:limit]
        logger.warning("[mannequin text2img] final prompt hard-trimmed to %d chars", limit)
    return text


async def _submit_kie_task(model: str, input_payload: dict) -> str:
    from app.services.kie_rate_limiter import acquire_kie_token_async

    body = json.dumps({"model": model, "input": input_payload})
    headers = {
        "Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}",
        "Content-Type": "application/json",
    }
    # Account-wide rate limit (shared across all Celery workers).
    await acquire_kie_token_async()
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
    text = _compose_text_mannequin_seeddream_prompt(pose_prompt, pose_type)
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
