"""
Face-to-Model service (reference photo upload).

Pipeline:
  1. build_configuration — category defaults for DB storage (not sent to image generation).
  2. KIE — primary model (retries) then fallback (retries) with user's photo URL +
     passport / black t-shirt prompt to preserve identity while standardizing framing and clothing.
  3. Download result, upload to R2.
"""

import asyncio
import uuid
from typing import Any, AsyncGenerator, Tuple, Optional as Opt

from fastapi import HTTPException, status

from app.config import settings
from app.services.ai_face_service import build_configuration
from app.services.kie_image_fallback_service import generate_image_with_model_fallback
from app.services.r2_service import upload_bytes_to_r2


def _reference_face_kie_model_chain() -> list[str]:
    primary = (
        getattr(settings, "MODEL_FACE_REFERENCE_KIE_PRIMARY_MODEL", "") or "nano-banana-2"
    ).strip()
    fb = (
        getattr(settings, "MODEL_FACE_REFERENCE_KIE_FALLBACK_MODEL", "")
        or "gpt-image-1.5-image-to-image"
    ).strip()
    out: list[str] = []
    for m in (primary, fb):
        if m and m not in out:
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# Reference portrait — image-to-image (identity from user photo, standardized output)
# ---------------------------------------------------------------------------

_REFERENCE_PASSPORT_IMG2IMG_PROMPT = """\
Transform this reference photo into a professional passport-style headshot while preserving the exact same person.

IDENTITY (critical): Keep the same face, bone structure, skin tone, eyes, nose, mouth, facial hair pattern, \
hair style, hair color, age appearance, and ethnicity as the person in the reference image. \
The result must be clearly the same individual — do not invent a different face.

FRAMING: Head and upper shoulders visible, chest-up framing, centered, facing the camera with a neutral \
passport-appropriate expression (mouth closed, eyes open, looking at camera).

BACKGROUND: Clean plain white background only. Soft, even studio lighting from the front with subtle fill; \
sharp focus on the face. No harsh shadows.

CLOTHING: Plain black crew-neck t-shirt only — clean, unwrinkled, clearly visible at the shoulders and upper \
chest. Replace any original clothing with this garment.

OUTPUT: Photorealistic, like a real government ID or modeling headshot. Natural skin texture. \
No watermark, no text, no illustration, no CGI or plastic skin.
"""


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def generate_model_face_from_reference(image_url: str, model_category: str) -> str:
    """
    Run KIE portrait chain (reference URL + passport prompt), upload result to R2, return public URL.
    """
    try:
        img_bytes = await generate_image_with_model_fallback(
            _REFERENCE_PASSPORT_IMG2IMG_PROMPT,
            image_urls=[image_url],
            label="custom_face_reference_img2img",
            model_chain=_reference_face_kie_model_chain(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Portrait generation failed. {exc}",
        ) from exc
    face_id = str(uuid.uuid4())
    s3_key = f"model-faces/{model_category}_{face_id[:8]}.png"
    return await upload_bytes_to_r2(img_bytes, s3_key, content_type="image/png")


async def generate_model_face_from_reference_stream(
    image_url: str,
    model_category: str,
) -> AsyncGenerator[Tuple[str, str, Opt[str], Opt[dict[str, Any]]], None]:
    """
    Same pipeline as generate_model_face_from_reference with progress tuples.

    Yields: (step, message, face_url, persist_meta)
      - face_url and persist_meta are None until the final ``done`` step.
      - On ``done``, persist_meta is
        ``{ "age", "ethnicity", "gender", "model_configuration" }`` for DB storage.
    """
    yield ("initialize", "Initializing face generation process", None, None)
    await asyncio.sleep(1)

    yield (
        "building_profile",
        "Applying category defaults for stored profile fields",
        None,
        None,
    )
    config = build_configuration(model_category, {})
    await asyncio.sleep(0.3)
    yield ("building_profile_done", "Profile fields ready", None, None)
    await asyncio.sleep(0.5)

    yield ("starting_generation", "Starting portrait generation from your photo", None, None)
    await asyncio.sleep(1)
    yield (
        "training",
        "Generating passport-style portrait (same face, black t-shirt, white background)",
        None,
        None,
    )

    try:
        img_bytes = await generate_image_with_model_fallback(
            _REFERENCE_PASSPORT_IMG2IMG_PROMPT,
            image_urls=[image_url],
            label="custom_face_reference_stream",
            model_chain=_reference_face_kie_model_chain(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Portrait generation failed. {exc}",
        ) from exc
    await asyncio.sleep(0.5)

    yield ("generated", "Successfully generated face", None, None)
    await asyncio.sleep(1)

    yield ("uploading", "Uploading generated face to storage", None, None)
    face_id = str(uuid.uuid4())
    s3_key = f"model-faces/{model_category}_{face_id[:8]}.png"
    s3_url = await upload_bytes_to_r2(img_bytes, s3_key, content_type="image/png")
    await asyncio.sleep(0.5)

    persist_meta: dict[str, Any] = {
        "age": config.get("age"),
        "ethnicity": config.get("ethnicity"),
        "gender": config.get("gender"),
        "model_configuration": dict(config),
    }
    yield ("done", "Face generation complete", s3_url, persist_meta)
