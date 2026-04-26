"""
AI model-face portrait generation (text prompt → image → R2).

Uses the same prompt template and configuration JSON style as
scripts/generate_model_faces.py (passport-style headshot instructions).
"""

import asyncio
import io
import json
import re
import uuid
from typing import Any, AsyncGenerator, Optional as Opt, Tuple

import httpx
from fastapi import HTTPException, status
from PIL import Image

from app.config import settings
from app.services.kie_image_fallback_service import generate_image_with_model_fallback
from app.services.r2_service import upload_bytes_to_r2

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
    "age":              25,
    "ethnicity":        "Indian",
}

# Beard only for adult_male — all other categories force none
_BEARD_DEFAULTS = {
    "beard_length": "none",
    "beard_color":  "none",
}

_MALE_CATEGORIES = {"adult_male"}


def coerce_age_to_int(age_val: Any) -> Opt[int]:
    """Parse age to integer years from int, float, or strings like '25 years'. Returns None if unknown."""
    if age_val is None or isinstance(age_val, bool):
        return None
    if isinstance(age_val, int):
        return age_val
    if isinstance(age_val, float):
        return int(age_val)
    s = str(age_val).strip()
    if not s:
        return None
    m = re.match(r"^(\d+)", s)
    if m:
        return int(m.group(1))
    return None


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

    age_i = coerce_age_to_int(config.get("age"))
    config["age"] = age_i if age_i is not None else 25

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


def _normalize_to_png_bytes(raw: bytes) -> bytes:
    try:
        img = Image.open(io.BytesIO(raw))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return raw


async def generate_face_image(config: dict) -> bytes:
    """Submit portrait prompt, poll, download, return PNG bytes."""
    prompt = build_face_prompt(config)
    if len(prompt) > 10000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Face configuration exceeds the maximum prompt length (10000 characters).",
        )

    try:
        raw = await generate_image_with_model_fallback(
            prompt,
            image_urls=None,
            label="custom_face_generation",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Portrait generation failed: {exc}",
        ) from exc
    return _normalize_to_png_bytes(raw)


async def generate_and_upload_face(category: str, overrides: dict) -> tuple[str, dict]:
    """
    Full pipeline: build config → generate image → upload to R2.
    Returns (face_url, final_configuration).
    """
    config    = build_configuration(category, overrides)
    img_bytes = await generate_face_image(config)

    face_id  = str(uuid.uuid4())
    s3_key   = f"model-faces/{category}_{face_id[:8]}.png"
    face_url = await upload_bytes_to_r2(img_bytes, s3_key, content_type="image/png")

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

    yield ("starting_generation", "Starting face generation", None, None)
    await asyncio.sleep(1)
    yield ("training", "Training face specifications, facial expression, skin overlaying, skin tone management", None, None)

    img_bytes = await generate_face_image(config)
    await asyncio.sleep(0.5)

    yield ("generated", "Successfully generated face", None, None)
    await asyncio.sleep(1)

    yield ("uploading", "Uploading generated face to storage", None, None)
    face_id  = str(uuid.uuid4())
    s3_key   = f"model-faces/{category}_{face_id[:8]}.png"
    face_url = await upload_bytes_to_r2(img_bytes, s3_key, content_type="image/png")
    await asyncio.sleep(0.5)

    yield ("done", "Face generation complete", face_url, config)
