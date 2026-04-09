"""
Face-to-Model service (reference photo upload).

Pipeline:
  1. Gemini vision — validate a clear human face and extract structured attributes
     (same keys as AI face configuration / scripts/generate_model_faces JSON style).
  2. Merge with category defaults via build_configuration (shared with generate-with-AI).
  3. build_face_prompt — identical passport-style prompt as generate-with-AI / script.
  4. kie.ai SeedDream 5.0 Lite text-to-image (no reference image in the generation call).
  5. Download result, upload to S3.
"""

import asyncio
import json
import uuid
from typing import Any, AsyncGenerator, FrozenSet, Tuple, Optional as Opt

import httpx
from fastapi import HTTPException, status

from app.config import settings
from app.services.ai_face_service import build_configuration, build_face_prompt
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
# Step 1 — Gemini vision: validate face + structured configuration
# ---------------------------------------------------------------------------

async def validate_face(image_url: str) -> dict[str, Any]:
    """
    Download the image and use Gemini vision to verify a usable human face.

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
        "- age: string like \"28 years\" or a number (estimated from the photo)\n"
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
            detail=f"Gemini face validation failed: {exc}",
        )

    raw = response.candidates[0].content.parts[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini returned unexpected response: {raw}",
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
# Step 2 — SeedDream 5.0 Lite text-to-image (kie.ai)
# ---------------------------------------------------------------------------

_CREATE_URL = "https://api.kie.ai/api/v1/jobs/createTask"
_STATUS_URL = "https://api.kie.ai/api/v1/jobs/recordInfo"


async def _submit_seedream_lite_text_task(prompt: str) -> str:
    payload = json.dumps({
        "model": settings.MODEL_FACE_REFERENCE_SEEDREAM_MODEL,
        "input": {
            "prompt":         prompt,
            "aspect_ratio":   settings.MODEL_FACE_REFERENCE_SEEDREAM_ASPECT,
            "quality":        settings.MODEL_FACE_REFERENCE_SEEDREAM_QUALITY,
            "nsfw_checker":   False,
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
            detail=f"SeedDream 5 Lite task submission failed: {exc.response.text[:800]}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"SeedDream 5 Lite task submission failed: {exc}",
        )

    task_id = resp.json().get("data", {}).get("taskId")
    if not task_id:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"SeedDream returned no taskId: {resp.text[:800]}",
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
                    detail="SeedDream task succeeded but returned no result URLs.",
                )

            if state == "fail":
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="SeedDream image generation task failed.",
                )

            await asyncio.sleep(settings.SEEDDREAM_RETRY_DELAY)

        except HTTPException:
            raise
        except Exception:
            await asyncio.sleep(settings.SEEDDREAM_RETRY_DELAY)

    raise HTTPException(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        detail=f"SeedDream task did not complete after {settings.SEEDDREAM_MAX_RETRIES} attempts.",
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
    Validate with Gemini vision, build script-style prompt from merged config,
    generate via SeedDream 5.0 Lite text-to-image, upload to S3.
    """
    parsed    = await validate_face(image_url)
    config    = build_configuration(model_category, parsed["overrides"])
    prompt    = build_face_prompt(config)
    if len(prompt) > 10000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Face configuration prompt exceeds 10000 characters (kie.ai limit).",
        )

    task_id    = await _submit_seedream_lite_text_task(prompt)
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

    yield ("validating_image", "Validating reference image (Gemini vision)", None, None)
    parsed = await validate_face(image_url)
    await asyncio.sleep(0.5)
    yield ("validating_image_done", "Reference image validated — face detected", None, None)
    await asyncio.sleep(1)

    yield ("building_prompt", "Building face configuration and passport-style prompt", None, None)
    config = build_configuration(model_category, parsed["overrides"])
    prompt = build_face_prompt(config)
    if len(prompt) > 10000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Face configuration prompt exceeds 10000 characters (kie.ai limit).",
        )
    await asyncio.sleep(0.3)
    yield ("building_prompt_done", "Prompt ready", None, None)
    await asyncio.sleep(0.5)

    yield ("starting_generation", "Starting SeedDream 5.0 Lite text-to-image", None, None)
    await asyncio.sleep(1)
    yield ("training", "Generating portrait from face configuration", None, None)

    task_id    = await _submit_seedream_lite_text_task(prompt)
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
