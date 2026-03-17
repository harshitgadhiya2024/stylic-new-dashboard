"""
Background generation service.

Two pipelines:
  A) generate_background_stream        — reference URL → SeedDream enhancement
  B) generate_background_with_ai_stream — text config  → Gemini image generation

Both are synchronous generators that yield (step, message, result_url | None)
tuples for real-time SSE streaming via the thread-pool pattern.
"""

import io
import json
import time
import uuid
from typing import Generator, Tuple, Optional

import requests
from fastapi import HTTPException, status

from app.config import settings
from app.services.s3_service import upload_bytes_to_s3

_CREATE_URL = "https://api.kie.ai/api/v1/jobs/createTask"
_STATUS_URL = "https://api.kie.ai/api/v1/jobs/recordInfo"


# ---------------------------------------------------------------------------
# Step 1 — Validate background URL is reachable
# ---------------------------------------------------------------------------

def _validate_background_url(background_url: str) -> None:
    try:
        resp = requests.head(background_url, timeout=15, allow_redirects=True)
        resp.raise_for_status()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not reach the provided background_url: {exc}",
        )


# ---------------------------------------------------------------------------
# Step 2 — Build SeedDream prompt
# ---------------------------------------------------------------------------

def _build_background_prompt(background_name: str) -> str:
    readable = background_name.replace("_", " ").strip()
    return (
        f"You are provided with ONE reference background image.\n\n"
        f"Generate a highly realistic, professional fashion photoshoot background "
        f"that closely matches the provided reference image.\n\n"
        f"Background style: {readable}\n\n"
        f"[REQUIREMENTS]\n"
        f"- Maintain the exact mood, color palette, and composition of the reference\n"
        f"- Professional studio or location photography quality\n"
        f"- Clean, distraction-free background suitable for fashion model placement\n"
        f"- High resolution, photorealistic quality\n"
        f"- No people, models, or text in the background\n"
        f"- Seamless, well-lit environment\n\n"
        f"Do not add any text, watermarks, or overlays."
    )


# ---------------------------------------------------------------------------
# Step 3 — Submit SeedDream task
# ---------------------------------------------------------------------------

def _submit_task(prompt: str, background_url: str) -> str:
    payload = json.dumps({
        "model": settings.SEEDDREAM_MODEL,
        "input": {
            "prompt":       prompt,
            "image_urls":   [background_url],
            "aspect_ratio": "16:9",
            "quality":      settings.SEEDDREAM_QUALITY,
        },
    })
    headers = {
        "Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}",
        "Content-Type":  "application/json",
    }

    try:
        resp = requests.post(_CREATE_URL, headers=headers, data=payload, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"SeedDream task submission failed: {exc}",
        )

    task_id = resp.json().get("data", {}).get("taskId")
    if not task_id:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"SeedDream returned no taskId: {resp.text}",
        )
    return task_id


# ---------------------------------------------------------------------------
# Step 4 — Poll SeedDream task
# ---------------------------------------------------------------------------

def _poll_task(task_id: str) -> str:
    headers = {"Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}"}

    for attempt in range(1, settings.SEEDDREAM_MAX_RETRIES + 1):
        try:
            resp = requests.get(
                f"{_STATUS_URL}?taskId={task_id}",
                headers=headers,
                timeout=30,
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
                    detail="SeedDream background generation task failed.",
                )

            time.sleep(settings.SEEDDREAM_RETRY_DELAY)

        except HTTPException:
            raise
        except Exception:
            time.sleep(settings.SEEDDREAM_RETRY_DELAY)

    raise HTTPException(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        detail=f"SeedDream task did not complete after {settings.SEEDDREAM_MAX_RETRIES} attempts.",
    )


# ---------------------------------------------------------------------------
# Step 5 — Download generated image bytes
# ---------------------------------------------------------------------------

def _download_image(result_url: str) -> bytes:
    for attempt in range(1, 4):
        try:
            resp = requests.get(result_url, stream=True, timeout=(10, 60))
            resp.raise_for_status()
            buf = io.BytesIO()
            for chunk in resp.iter_content(65536):
                if chunk:
                    buf.write(chunk)
            return buf.getvalue()
        except Exception as exc:
            if attempt == 3:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to download generated background: {exc}",
                )
            time.sleep(3)


# ---------------------------------------------------------------------------
# Public streaming generator
# ---------------------------------------------------------------------------

def generate_background_stream(
    background_url:  str,
    background_name: str,
) -> Generator[Tuple[str, str, Optional[str]], None, None]:
    """
    Full pipeline as a synchronous generator that yields (step, message, result_url).
    result_url is None for all steps except the final "done" step.

    Designed to run in a thread-pool executor via _run_sync_generator so the
    async event loop is never blocked.

    Total expected duration: 20-30 seconds.
    """
    yield ("initialize", "Initializing background generation process", None)
    time.sleep(1)

    yield ("validating_background", "Validating background image URL", None)
    _validate_background_url(background_url)
    time.sleep(0.5)
    yield ("validating_background_done", "Background image validated successfully", None)
    time.sleep(1)

    yield ("starting_generation", "Starting background generation", None)
    time.sleep(1)
    yield ("processing", "Processing background — analyzing composition, lighting, color palette and environment", None)

    prompt  = _build_background_prompt(background_name)
    task_id = _submit_task(prompt, background_url)
    result_url = _poll_task(task_id)
    time.sleep(0.5)

    yield ("generated", "Successfully generated background", None)
    time.sleep(1)

    yield ("uploading", "Uploading generated background to storage", None)
    img_bytes = _download_image(result_url)
    bg_id     = str(uuid.uuid4())
    s3_key    = f"backgrounds/generated_{bg_id[:8]}.png"
    s3_url    = upload_bytes_to_s3(img_bytes, s3_key, content_type="image/png")
    time.sleep(0.5)

    yield ("done", "Background generation complete", s3_url)


# ---------------------------------------------------------------------------
# Gemini AI background generation
# ---------------------------------------------------------------------------

def _generate_background_image_with_gemini(
    background_name: str,
    background_configuration: str,
) -> bytes:
    """
    Call Gemini to generate a 9:16 background image from a text description.
    Returns raw PNG bytes.
    """
    try:
        from google import genai
        from google.genai import types as gtypes
        from PIL import Image
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing dependency: {exc}. Run: pip install google-genai Pillow",
        )

    readable_name = background_name.replace("_", " ").strip()

    prompt = (
        f"Generate a highly realistic, professional fashion photoshoot background image.\n\n"
        f"Background name: {readable_name}\n"
        f"Background description and configuration: {background_configuration}\n\n"
        f"[REQUIREMENTS]\n"
        f"- Professional studio or location photography quality\n"
        f"- Clean, distraction-free background suitable for placing a fashion model in front\n"
        f"- No people, models, mannequins, or text in the image\n"
        f"- High resolution, photorealistic, cinematic quality\n"
        f"- Seamless, well-lit environment with natural depth\n"
        f"- Colors and lighting should complement fashion photography\n\n"
        f"Do not add any text, watermarks, or overlays on the image."
    )

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    contents = [
        gtypes.Content(
            role="user",
            parts=[gtypes.Part.from_text(text=prompt)],
        )
    ]
    cfg = gtypes.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
        image_config=gtypes.ImageConfig(aspect_ratio="9:16"),
    )

    try:
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=contents,
            config=cfg,
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                img = Image.open(io.BytesIO(part.inline_data.data))
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                return buf.getvalue()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini background generation failed: {exc}",
        )

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Gemini returned no image data for background generation.",
    )


def generate_background_with_ai_stream(
    background_name:          str,
    background_configuration: str,
) -> Generator[Tuple[str, str, Optional[str]], None, None]:
    """
    Gemini AI pipeline as a synchronous generator.
    Yields (step, message, result_url | None).

    Designed to run in a thread-pool executor via _run_sync_generator.
    Expected duration: 20-30 seconds.
    """
    yield ("initialize", "Initializing background generation process", None)
    time.sleep(1)

    yield ("validating_config", "Validating background configurations", None)
    time.sleep(0.5)
    yield ("validating_config_done", "Background configurations validated", None)
    time.sleep(1)

    yield ("starting_generation", "Starting background generation", None)
    time.sleep(1)
    yield ("processing", "Processing background — generating composition, lighting, color palette and environment", None)

    img_bytes = _generate_background_image_with_gemini(background_name, background_configuration)
    time.sleep(0.5)

    yield ("generated", "Successfully generated background", None)
    time.sleep(1)

    yield ("uploading", "Uploading generated background to storage", None)
    bg_id  = str(uuid.uuid4())
    s3_key = f"backgrounds/ai_{bg_id[:8]}.png"
    s3_url = upload_bytes_to_s3(img_bytes, s3_key, content_type="image/png")
    time.sleep(0.5)

    yield ("done", "Background generation complete", s3_url)
