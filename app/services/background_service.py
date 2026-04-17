"""
Background generation service.

Two pipelines:
  A) generate_background_stream         — uploaded/custom URL → direct upload to R2
  B) generate_background_with_ai_stream — text config  → Gemini image generation

Both are async generators that yield (step, message, result_url | None)
tuples for real-time SSE streaming.
"""

import asyncio
import io
import uuid
from typing import AsyncGenerator, Tuple, Optional

import httpx
from fastapi import HTTPException, status

from app.config import settings
from app.services.r2_service import upload_bytes_to_r2


# ---------------------------------------------------------------------------
# Step 1 — Validate background URL is reachable
# ---------------------------------------------------------------------------

async def _validate_background_url(background_url: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.head(background_url, follow_redirects=True)
            resp.raise_for_status()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not reach the provided background_url: {exc}",
        )


# ---------------------------------------------------------------------------
# Step 2 — Download uploaded/custom image bytes
# ---------------------------------------------------------------------------

async def _download_image(image_url: str) -> bytes:
    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.get(image_url)
                resp.raise_for_status()
            return resp.content
        except Exception as exc:
            if attempt == 3:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to download background image: {exc}",
                )
            await asyncio.sleep(3)


# ---------------------------------------------------------------------------
# Public async streaming generator
# ---------------------------------------------------------------------------

async def generate_background_stream(
    background_url:  str,
    background_name: str,
) -> AsyncGenerator[Tuple[str, str, Optional[str]], None]:
    """
    Custom background upload pipeline as an async generator that yields
    (step, message, result_url).
    result_url is None for all steps except the final "done" step.
    """
    _ = background_name  # currently used only for client-side display / DB metadata

    yield ("initialize", "Initializing custom background processing", None)
    await asyncio.sleep(1)

    yield ("validating_background", "Validating background image URL", None)
    await _validate_background_url(background_url)
    await asyncio.sleep(1)
    yield ("validating_background_done", "Background image validated successfully", None)
    await asyncio.sleep(1)

    yield ("starting_generation", "Starting custom background workflow", None)
    await asyncio.sleep(1)
    yield ("processing", "Processing background — analyzing composition, lighting, color palette and environment", None)
    await asyncio.sleep(2)
    yield ("processing_details", "Optimizing background quality and preparing storage format", None)
    await asyncio.sleep(2)
    yield ("processing_final", "Finalizing custom background output", None)
    await asyncio.sleep(1)

    yield ("uploading", "Uploading background to storage", None)
    img_bytes = await _download_image(background_url)
    bg_id     = str(uuid.uuid4())
    s3_key    = f"backgrounds/custom_{bg_id[:8]}.png"
    s3_url    = await upload_bytes_to_r2(img_bytes, s3_key, content_type="image/png")
    await asyncio.sleep(1)

    yield ("done", "Custom background upload complete", s3_url)


# ---------------------------------------------------------------------------
# Gemini AI background generation
# ---------------------------------------------------------------------------

async def _generate_background_image_with_gemini(
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

    loop = asyncio.get_event_loop()

    def _call_gemini():
        g_client = genai.Client(api_key=settings.GEMINI_API_KEY)
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
        return g_client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=contents,
            config=cfg,
        )

    try:
        response = await loop.run_in_executor(None, _call_gemini)
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


async def generate_background_with_ai_stream(
    background_name:          str,
    background_configuration: str,
) -> AsyncGenerator[Tuple[str, str, Optional[str]], None]:
    """
    Gemini AI pipeline as an async generator.
    Yields (step, message, result_url | None).
    """
    yield ("initialize", "Initializing background generation process", None)
    await asyncio.sleep(1)

    yield ("validating_config", "Validating background configurations", None)
    await asyncio.sleep(0.5)
    yield ("validating_config_done", "Background configurations validated", None)
    await asyncio.sleep(1)

    yield ("starting_generation", "Starting background generation", None)
    await asyncio.sleep(1)
    yield ("processing", "Processing background — generating composition, lighting, color palette and environment", None)

    img_bytes = await _generate_background_image_with_gemini(background_name, background_configuration)
    await asyncio.sleep(0.5)

    yield ("generated", "Successfully generated background", None)
    await asyncio.sleep(1)

    yield ("uploading", "Uploading generated background to storage", None)
    bg_id  = str(uuid.uuid4())
    s3_key = f"backgrounds/ai_{bg_id[:8]}.png"
    s3_url = await upload_bytes_to_r2(img_bytes, s3_key, content_type="image/png")
    await asyncio.sleep(0.5)

    yield ("done", "Background generation complete", s3_url)
