"""
Face-to-Model service (reference photo upload).

Pipeline:
  1. Vision — validate a clear human face and extract structured attributes for DB storage only.
  2. build_configuration — merge vision overrides with category defaults (persisted; not sent to image).
  3. SeedDream 5.0 Lite image-to-image — user's reference photo URL + passport / black t-shirt prompt
     to preserve identity while standardizing framing and clothing.
  4. Download result, upload to S3.
"""

import asyncio
import json
import uuid
from typing import Any, AsyncGenerator, FrozenSet, Tuple, Optional as Opt

import httpx
from fastapi import HTTPException, status

from app.config import settings
from app.services.ai_face_service import build_configuration
from app.services.s3_service import upload_bytes_to_s3

_ALLOWED_VISION_OVERRIDE_KEYS: FrozenSet[str] = frozenset(
    {
        "face_shape",
        "jawline_type",
        "cheekbone_height",
        "face_skin_tone",
        "skin_undertone",
        "hair_color",
        "hair_length",
        "hair_style",
        "eye_shape",
        "eye_color",
        "nose_shape",
        "lip_shape",
        "eyebrow_shape",
        "beard_length",
        "beard_color",
        "age",
        "ethnicity",
        "gender",
    }
)


def _sanitize_vision_overrides(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k not in _ALLOWED_VISION_OVERRIDE_KEYS or v is None:
            continue
        if isinstance(v, str):
            s = v.strip()
            if not s:
                continue
            out[k] = s
        elif isinstance(v, (int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v).strip()
    return out


def _assert_required_vision_demographics(overrides: dict[str, Any]) -> None:
    """Vision must infer age, ethnicity, and gender so we can persist them on the model face."""
    for key in ("age", "ethnicity", "gender"):
        v = overrides.get(key)
        if v is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Vision did not return `{key}` for this image. "
                    "Use a clearer front-facing photo or try again."
                ),
            )
        if isinstance(v, str) and not v.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Vision returned an empty `{key}` for this image. "
                    "Use a clearer front-facing photo or try again."
                ),
            )


# ---------------------------------------------------------------------------
# Step 1 — Vision: validate face + structured configuration
# ---------------------------------------------------------------------------

async def validate_face(image_url: str) -> dict[str, Any]:
    """
    Download the image and use vision analysis to verify a usable human face.

    Returns:
        {
          "description": str,   # short narrative (for logs / debugging)
          "overrides": dict,    # snake_case keys for build_configuration
        }
    """
    try:
        from google import genai
        from google.genai import types as gtypes
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing dependency: {exc}. Run: pip install google-genai",
        )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            img_resp = await client.get(image_url)
            img_resp.raise_for_status()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not download the provided face_url: {exc}",
        )

    img_bytes = img_resp.content
    content_type = img_resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    if content_type not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        content_type = "image/jpeg"

    keys_list = ", ".join(sorted(_ALLOWED_VISION_OVERRIDE_KEYS))
    validation_prompt = (
        "You are an image validation assistant for a fashion photoshoot platform. "
        "Check whether this image contains a clear, usable human face.\n\n"
        "Respond with ONLY valid JSON (no markdown fences):\n"
        "{\n"
        '  "has_face": true or false,\n'
        '  "reason": "<if has_face is false: explain why. Empty string if true.>",\n'
        '  "description": "<if has_face: concise visible appearance summary. Empty if false.>",\n'
        '  "overrides": {\n'
        "    /* Only if has_face is true. REQUIRED non-empty fields: age, ethnicity, gender. */\n"
        "    /* Other keys: omit or null. Use snake_case strings (e.g. dark_brown, medium). */\n"
        f"    /* Allowed keys: {keys_list} */\n"
        "  }\n"
        "}\n\n"
        "When has_face is true, overrides MUST include:\n"
        "- age: integer years (e.g. 28) preferred, or a string like \"28 years\" (estimated from the photo)\n"
        "- ethnicity: short label (e.g. South Asian, European, African)\n"
        "- gender: one of male, female, boy, girl matching the subject\n"
        "Also fill other visible attributes when possible (face_shape, hair_color, eye_color, etc.).\n\n"
        "has_face must be false if:\n"
        "- No human face is visible\n"
        "- Face is heavily blurred, masked, or occluded\n"
        "- Image is a cartoon, illustration, or AI art without a real face\n"
        "- Image contains only animals, objects, or scenery\n"
        "- Face is turned away and features are not visible"
    )

    loop = asyncio.get_event_loop()

    def _call_gemini():
        g_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return g_client.models.generate_content(
            model=settings.GEMINI_VISION_MODEL,
            contents=[
                gtypes.Content(
                    role="user",
                    parts=[
                        gtypes.Part.from_bytes(mime_type=content_type, data=img_bytes),
                        gtypes.Part.from_text(text=validation_prompt),
                    ],
                )
            ],
            config=gtypes.GenerateContentConfig(
                response_modalities=["TEXT"],
                temperature=0,
            ),
        )

    try:
        response = await loop.run_in_executor(None, _call_gemini)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Reference image analysis failed. Please try again later.",
        )

    raw = response.candidates[0].content.parts[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Reference image analysis returned an unexpected response. Please try again.",
        )

    if not result.get("has_face"):
        reason = result.get("reason", "No valid human face detected in the image.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Input image does not contain a valid face: {reason}",
        )

    overrides = _sanitize_vision_overrides(result.get("overrides"))
    _assert_required_vision_demographics(overrides)
    return {
        "description": result.get("description") or "",
        "overrides":   overrides,
    }


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
# Step 2 — Image-to-image portrait job (kie.ai)
# ---------------------------------------------------------------------------

_CREATE_URL = "https://api.kie.ai/api/v1/jobs/createTask"
_STATUS_URL = "https://api.kie.ai/api/v1/jobs/recordInfo"


async def _submit_seedream_lite_reference_img2img_task(image_url: str) -> str:
    if not (settings.SEEDDREAM_API_KEY or "").strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Portrait generation is not configured.",
        )
    payload = json.dumps({
        "model": settings.MODEL_FACE_REFERENCE_SEEDREAM_IMG2IMG_MODEL,
        "input": {
            "prompt":       _REFERENCE_PASSPORT_IMG2IMG_PROMPT,
            "image_urls":   [image_url],
            "aspect_ratio": settings.MODEL_FACE_REFERENCE_SEEDREAM_ASPECT,
            "quality":      settings.MODEL_FACE_REFERENCE_SEEDREAM_QUALITY,
            "nsfw_checker": False,
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
            detail="Portrait generation request failed. Please try again later.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Portrait generation request failed. Please try again later.",
        )

    task_id = resp.json().get("data", {}).get("taskId")
    if not task_id:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Portrait generation could not be started. Please try again later.",
        )
    return task_id


async def _poll_task(task_id: str) -> str:
    headers = {"Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}"}

    for _attempt in range(1, settings.SEEDDREAM_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{_STATUS_URL}?taskId={task_id}",
                    headers=headers,
                )
                resp.raise_for_status()
            data  = resp.json().get("data", {})
            state = data.get("state")

            if state == "success":
                result_urls = json.loads(data.get("resultJson", "{}")).get("resultUrls", [])
                if result_urls:
                    return result_urls[0]
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Portrait generation finished but no image URL was returned.",
                )

            if state == "fail":
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="Portrait generation failed.",
                )

            await asyncio.sleep(settings.SEEDDREAM_RETRY_DELAY)

        except HTTPException:
            raise
        except Exception:
            await asyncio.sleep(settings.SEEDDREAM_RETRY_DELAY)

    raise HTTPException(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        detail="Portrait generation timed out. Please try again later.",
    )


# ---------------------------------------------------------------------------
# Step 3 — Download generated image bytes
# ---------------------------------------------------------------------------

async def _download_image(result_url: str) -> bytes:
    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.get(result_url)
                resp.raise_for_status()
            return resp.content
        except Exception as exc:
            if attempt == 3:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to download generated image: {exc}",
                )
            await asyncio.sleep(3)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def generate_model_face_from_reference(image_url: str, model_category: str) -> str:
    """
    Validate reference photo, merge vision attributes into config for storage semantics,
    run SeedDream image-to-image (reference URL + passport prompt), upload to S3.
    """
    parsed = await validate_face(image_url)
    config = build_configuration(model_category, parsed["overrides"])

    task_id = await _submit_seedream_lite_reference_img2img_task(image_url)
    result_url = await _poll_task(task_id)
    img_bytes  = await _download_image(result_url)
    face_id    = str(uuid.uuid4())
    s3_key     = f"model-faces/{model_category}_{face_id[:8]}.png"
    return await upload_bytes_to_s3(img_bytes, s3_key, content_type="image/png")


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

    yield ("validating_image", "Validating reference image", None, None)
    parsed = await validate_face(image_url)
    await asyncio.sleep(0.5)
    yield ("validating_image_done", "Reference image validated — face detected", None, None)
    await asyncio.sleep(1)

    yield (
        "building_prompt",
        "Merging vision attributes for your profile (used for storage, not for image generation)",
        None,
        None,
    )
    config = build_configuration(model_category, parsed["overrides"])
    await asyncio.sleep(0.3)
    yield ("building_prompt_done", "Ready to generate passport portrait from your photo", None, None)
    await asyncio.sleep(0.5)

    yield ("starting_generation", "Starting portrait generation from your photo", None, None)
    await asyncio.sleep(1)
    yield (
        "training",
        "Generating passport-style portrait (same face, black t-shirt, white background)",
        None,
        None,
    )

    task_id = await _submit_seedream_lite_reference_img2img_task(image_url)
    result_url = await _poll_task(task_id)
    await asyncio.sleep(0.5)

    yield ("generated", "Successfully generated face", None, None)
    await asyncio.sleep(1)

    yield ("uploading", "Uploading generated face to storage", None, None)
    img_bytes = await _download_image(result_url)
    face_id   = str(uuid.uuid4())
    s3_key    = f"model-faces/{model_category}_{face_id[:8]}.png"
    s3_url    = await upload_bytes_to_s3(img_bytes, s3_key, content_type="image/png")
    await asyncio.sleep(0.5)

    persist_meta: dict[str, Any] = {
        "age":                 config.get("age"),
        "ethnicity":           config.get("ethnicity"),
        "gender":              config.get("gender"),
        "model_configuration": dict(config),
    }
    yield ("done", "Face generation complete", s3_url, persist_meta)
