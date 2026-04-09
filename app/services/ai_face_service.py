"""
Model face image generation via kie.ai nano-banana-pro.

Uses the same prompt template and configuration JSON style as
scripts/generate_model_faces.py (passport-style headshot instructions).
"""

import asyncio
import io
import json
import uuid
from typing import Any, AsyncGenerator, Optional as Opt, Tuple

import httpx
from fastapi import HTTPException, status
from PIL import Image

from app.config import settings
from app.services.s3_service import upload_bytes_to_s3

_CREATE_URL = "https://api.kie.ai/api/v1/jobs/createTask"
_STATUS_URL = "https://api.kie.ai/api/v1/jobs/recordInfo"

# ── Default configuration values ──────────────────────────────────────────
_DEFAULTS_COMMON = {
    "face_shape":       "oval",
    "jawline_type":     "soft",
    "cheekbone_height": "medium",
    "face_skin_tone":   "medium",
    "skin_undertone":   "warm",
    "hair_color":       "dark_black",
    "hair_length":      "medium",
    "hair_style":       "straight",
    "eye_shape":        "almond",
    "eye_color":        "dark_brown",
    "nose_shape":       "straight",
    "lip_shape":        "medium",
    "eyebrow_shape":    "natural",
    "age":              "25 years",
    "ethnicity":        "Indian",
}

# Beard only for adult_male — all other categories force none
_BEARD_DEFAULTS = {
    "beard_length": "none",
    "beard_color":  "none",
}

_MALE_CATEGORIES = {"adult_male"}


def build_configuration(category: str, overrides: dict) -> dict:
    """
    Merge caller-supplied overrides on top of category-appropriate defaults.
    beard_length / beard_color are stripped and forced to 'none' for non-male categories.
    """
    if "female" in category or "girl" in category:
        gender = "female" if "adult" in category else "girl"
    elif "boy" in category:
        gender = "boy"
    else:
        gender = "male"

    config = {**_DEFAULTS_COMMON, "gender": gender}

    if category in _MALE_CATEGORIES:
        config.update(_BEARD_DEFAULTS)
        if overrides.get("beard_length"):
            config["beard_length"] = overrides["beard_length"]
        if overrides.get("beard_color"):
            config["beard_color"] = overrides["beard_color"]
    else:
        config["beard_length"] = "none"
        config["beard_color"]  = "none"

    for key, value in overrides.items():
        if key in ("beard_length", "beard_color"):
            continue
        if value is not None:
            config[key] = value

    return config


def build_face_prompt(model_cfg: dict[str, Any]) -> str:
    """
    Exact same prompt structure as scripts/generate_model_faces.build_face_prompt:
    JSON block of configuration + passport-style headshot instructions.
    """
    cfg_without_image = {k: v for k, v in model_cfg.items() if k != "image"}
    config_json = json.dumps(cfg_without_image, indent=2)
    return f"""Generate a highly realistic passport-size photograph of a person based on the face configuration below.

MODEL FACE CONFIGURATION:
{config_json}

Use every attribute from the configuration above (age, gender, ethnicity, bodyType, and all nested attributes like eyeColor, hairType, hairColor, hairStyle, hairLength, skinColor, faceShape, faceExpression, beardType, eyebrow, jawline, originalAttributes, etc.) to accurately construct the person's appearance.

PHOTO STYLE: Professional headshot / passport photo. Head and upper shoulders visible. Shot from chest-up, centered framing. Clean plain white background. Soft, even studio lighting from front with subtle fill light. No harsh shadows. Sharp focus on face.

CLOTHING: Wearing a plain black crew-neck t-shirt. The t-shirt should be clean, unwrinkled, and clearly visible at the shoulders and upper chest area.

REALISM: This must look like a real photograph taken with a professional camera. Natural skin texture with pores and subtle imperfections, realistic hair strands, proper light interaction with skin and fabric. No artificial smoothing, no CGI look, no illustration style. Photorealistic quality similar to a real ID photo or modeling headshot."""


async def _submit_model_face_kie_task(prompt: str) -> str:
    if not settings.SEEDDREAM_API_KEY.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="kie.ai API key not configured (SEEDDREAM_API_KEY).",
        )
    payload = json.dumps({
        "model": settings.MODEL_FACE_GENERATE_MODEL,
        "input": {
            "prompt":         prompt,
            "image_input":    [],
            "aspect_ratio":   settings.MODEL_FACE_GENERATE_ASPECT,
            "resolution":     settings.MODEL_FACE_GENERATE_RESOLUTION,
            "output_format":  "png",
        },
    })
    headers = {
        "Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}",
        "Content-Type":  "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(_CREATE_URL, headers=headers, content=payload)
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"kie.ai task submission failed: {exc.response.text[:500]}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"kie.ai task submission failed: {exc}",
        ) from exc

    task_id = resp.json().get("data", {}).get("taskId")
    if not task_id:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"kie.ai returned no taskId: {resp.text[:800]}",
        )
    return task_id


async def _poll_model_face_kie_task(task_id: str) -> str:
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
                    return urls[0]
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="kie.ai task succeeded but no resultUrls in response.",
                )
            if state == "fail":
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="kie.ai model face generation task failed.",
                )
            await asyncio.sleep(settings.SEEDDREAM_RETRY_DELAY)
        except HTTPException:
            raise
        except Exception:
            await asyncio.sleep(settings.SEEDDREAM_RETRY_DELAY)
    raise HTTPException(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        detail=f"kie.ai task did not complete after {settings.SEEDDREAM_MAX_RETRIES} attempts.",
    )


async def _download_result_image(url: str) -> bytes:
    try:
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to download generated image: {exc}",
        ) from exc


def _normalize_to_png_bytes(raw: bytes) -> bytes:
    try:
        img = Image.open(io.BytesIO(raw))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return raw


async def generate_face_image(config: dict) -> bytes:
    """Submit prompt to kie.ai nano-banana-pro, poll, download, return PNG bytes."""
    prompt = build_face_prompt(config)
    if len(prompt) > 10000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Face configuration produces a prompt over kie.ai 10000 character limit.",
        )

    task_id = await _submit_model_face_kie_task(prompt)
    image_url = await _poll_model_face_kie_task(task_id)
    raw = await _download_result_image(image_url)
    return _normalize_to_png_bytes(raw)


async def generate_and_upload_face(category: str, overrides: dict) -> tuple[str, dict]:
    """
    Full pipeline: build config → generate image → upload to S3.
    Returns (face_url, final_configuration).
    """
    config    = build_configuration(category, overrides)
    img_bytes = await generate_face_image(config)

    face_id  = str(uuid.uuid4())
    s3_key   = f"model-faces/{category}_{face_id[:8]}.png"
    face_url = await upload_bytes_to_s3(img_bytes, s3_key, content_type="image/png")

    return face_url, config


# ---------------------------------------------------------------------------
# Async streaming generator — yields (step, message, face_url | None, config | None)
# ---------------------------------------------------------------------------

async def generate_and_upload_face_stream(
    category: str,
    overrides: dict,
) -> AsyncGenerator[Tuple[str, str, Opt[str], Opt[dict]], None]:
    """
    Same pipeline as generate_and_upload_face but yields progress tuples
    at each stage so the caller can stream them to the client.

    Yields: (step, message, face_url, config)
      - face_url and config are None for all steps except the final "done" step.
    """
    yield ("initialize", "Initializing face generation process", None, None)
    await asyncio.sleep(1)

    yield ("validating_config", "Validating face configurations", None, None)
    config = build_configuration(category, overrides)
    await asyncio.sleep(0.5)
    yield ("validating_config_done", "Face configurations validated", None, None)
    await asyncio.sleep(1)

    yield ("starting_generation", "Starting face generation (kie.ai nano-banana-pro)", None, None)
    await asyncio.sleep(1)
    yield ("training", "Training face specifications, facial expression, skin overlaying, skin tone management", None, None)

    img_bytes = await generate_face_image(config)
    await asyncio.sleep(0.5)

    yield ("generated", "Successfully generated face", None, None)
    await asyncio.sleep(1)

    yield ("uploading", "Uploading generated face to storage", None, None)
    face_id  = str(uuid.uuid4())
    s3_key   = f"model-faces/{category}_{face_id[:8]}.png"
    face_url = await upload_bytes_to_s3(img_bytes, s3_key, content_type="image/png")
    await asyncio.sleep(0.5)

    yield ("done", "Face generation complete", face_url, config)
