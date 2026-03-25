"""
Fabric change service.

Uses Gemini 2.5 Flash Image (gemini-2.5-flash-image) to apply a new fabric
texture to a garment image while preserving the garment's shape, cut, and
structure.

Public function:
    change_fabric(image_url: str, fabric: str) -> bytes
        Downloads the garment image, sends it to Gemini with an instruction
        to apply the requested fabric, and returns the resulting PNG bytes.
"""

import asyncio
import io
import logging

import httpx
from fastapi import HTTPException, status

from app.config import settings

logger = logging.getLogger("fabric_service")


async def _download_image(url: str) -> tuple[bytes, str]:
    """Download an image from a URL and return (bytes, mime_type)."""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
        content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        if content_type not in ("image/jpeg", "image/png", "image/webp"):
            content_type = "image/jpeg"
        logger.info("[fabric] Downloaded garment image: %d bytes (%s)", len(resp.content), content_type)
        return resp.content, content_type
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to download garment image from '{url}': {exc}",
        )


def _build_fabric_prompt(fabric: str) -> str:
    return (
        f"You are a professional fashion image editor.\n\n"
        f"I am providing you with a garment image. Your task is to change the fabric/material "
        f"of this garment to: **{fabric}**.\n\n"
        f"[STRICT REQUIREMENTS]\n"
        f"- Keep the exact same garment shape, cut, silhouette, and style\n"
        f"- Keep the exact same garment color and pattern (only change the material texture)\n"
        f"- Realistically apply the {fabric} texture — show the correct drape, sheen, "
        f"  weave pattern, and surface finish that is characteristic of {fabric}\n"
        f"- Preserve all stitching, seams, buttons, zippers, or other garment details\n"
        f"- Keep the same background, lighting, and composition as the original image\n"
        f"- Do NOT add any text, watermarks, logos, or overlays\n"
        f"- Output a high-resolution, photorealistic image\n\n"
        f"Return ONLY the edited garment image with {fabric} fabric applied."
    )


def _build_texture_prompt(texture: str) -> str:
    return (
        f"You are a professional fashion image editor.\n\n"
        f"I am providing you with a garment image. Your task is to change the surface texture/pattern "
        f"of this garment to: **{texture}**.\n\n"
        f"[STRICT REQUIREMENTS]\n"
        f"- Keep the exact same garment shape, cut, silhouette, and style\n"
        f"- Keep the same base color tones — only change the surface texture/pattern\n"
        f"- Realistically apply the {texture} texture — ensure the pattern repeats naturally "
        f"  across the fabric surface, follows the garment's contours, and looks physically accurate\n"
        f"- Preserve all stitching, seams, buttons, zippers, or other garment construction details\n"
        f"- Keep the same background, lighting, and composition as the original image\n"
        f"- Do NOT add any text, watermarks, logos, or overlays\n"
        f"- Output a high-resolution, photorealistic image\n\n"
        f"Return ONLY the edited garment image with the {texture} texture applied."
    )


async def change_fabric(image_url: str, fabric: str) -> bytes:
    """
    Apply a new fabric texture to a garment image using Gemini.

    Args:
        image_url: Public URL of the garment image to edit.
        fabric:    The desired fabric name (e.g. "silk", "cotton", "denim").

    Returns:
        PNG bytes of the fabric-changed garment image.
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

    img_bytes, mime_type = await _download_image(image_url)
    prompt_text          = _build_fabric_prompt(fabric)

    loop = asyncio.get_event_loop()

    def _call_gemini() -> bytes:
        g_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        cfg = gtypes.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        )
        response = g_client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[
                gtypes.Content(
                    role="user",
                    parts=[
                        gtypes.Part.from_bytes(mime_type=mime_type, data=img_bytes),
                        gtypes.Part.from_text(text=prompt_text),
                    ],
                )
            ],
            config=cfg,
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                img = Image.open(io.BytesIO(part.inline_data.data))
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                return buf.getvalue()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Gemini returned no image data for fabric change.",
        )

    try:
        logger.info("[fabric] Calling Gemini for fabric='%s' on image: %s", fabric, image_url)
        result = await loop.run_in_executor(None, _call_gemini)
        logger.info("[fabric] Gemini fabric change complete (%d bytes)", len(result))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[fabric] Gemini fabric change failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini fabric change failed: {exc}",
        )


async def change_texture(image_url: str, texture: str) -> bytes:
    """
    Apply a new surface texture/pattern to a garment image using Gemini.

    Args:
        image_url: Public URL of the garment image to edit.
        texture:   The desired texture/pattern name (e.g. "plain weave", "checked", "printed").

    Returns:
        PNG bytes of the texture-changed garment image.
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

    img_bytes, mime_type = await _download_image(image_url)
    prompt_text          = _build_texture_prompt(texture)

    loop = asyncio.get_event_loop()

    def _call_gemini() -> bytes:
        g_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        cfg = gtypes.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        )
        response = g_client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[
                gtypes.Content(
                    role="user",
                    parts=[
                        gtypes.Part.from_bytes(mime_type=mime_type, data=img_bytes),
                        gtypes.Part.from_text(text=prompt_text),
                    ],
                )
            ],
            config=cfg,
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                img = Image.open(io.BytesIO(part.inline_data.data))
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                return buf.getvalue()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Gemini returned no image data for texture change.",
        )

    try:
        logger.info("[texture] Calling Gemini for texture='%s' on image: %s", texture, image_url)
        result = await loop.run_in_executor(None, _call_gemini)
        logger.info("[texture] Gemini texture change complete (%d bytes)", len(result))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[texture] Gemini texture change failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini texture change failed: {exc}",
        )


def _hex_to_rgb_name(hex_code: str) -> str:
    """Convert a hex color code to an approximate human-readable color name for the prompt."""
    hex_code = hex_code.lstrip("#")
    try:
        r, g, b = int(hex_code[0:2], 16), int(hex_code[2:4], 16), int(hex_code[4:6], 16)
        return f"RGB({r}, {g}, {b}) / hex #{hex_code.upper()}"
    except Exception:
        return f"hex #{hex_code}"


def _build_color_prompt(color_hex: str) -> str:
    color_desc = _hex_to_rgb_name(color_hex)
    return (
        f"You are a professional fashion image editor.\n\n"
        f"I am providing you with a garment image. Your task is to change the color of this "
        f"garment to the following exact color: **{color_desc}**.\n\n"
        f"[STRICT REQUIREMENTS]\n"
        f"- Change the primary garment color to exactly match {color_desc}\n"
        f"- Keep the exact same garment shape, cut, silhouette, and style\n"
        f"- Preserve all fabric texture, pattern, weave, and surface details — only the hue changes\n"
        f"- Maintain realistic lighting, shadows, highlights, and shading on the garment "
        f"  as they would appear with the new color under the same lighting conditions\n"
        f"- Preserve all stitching, seams, buttons, zippers, labels, or other garment details\n"
        f"- Keep the same background, lighting direction, and composition as the original image\n"
        f"- Do NOT add any text, watermarks, logos, or overlays\n"
        f"- Output a high-resolution, photorealistic image\n\n"
        f"Return ONLY the edited garment image with the new color applied."
    )


async def change_color(image_url: str, color_hex: str) -> bytes:
    """
    Apply a new color to a garment image using Gemini.

    Args:
        image_url:  Public URL of the garment image to edit.
        color_hex:  Target color as a hex code (e.g. "#fff28f", "#fe2a3e").

    Returns:
        PNG bytes of the color-changed garment image.
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

    img_bytes, mime_type = await _download_image(image_url)
    prompt_text          = _build_color_prompt(color_hex)

    loop = asyncio.get_event_loop()

    def _call_gemini() -> bytes:
        g_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        cfg = gtypes.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        )
        response = g_client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[
                gtypes.Content(
                    role="user",
                    parts=[
                        gtypes.Part.from_bytes(mime_type=mime_type, data=img_bytes),
                        gtypes.Part.from_text(text=prompt_text),
                    ],
                )
            ],
            config=cfg,
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                img = Image.open(io.BytesIO(part.inline_data.data))
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                return buf.getvalue()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Gemini returned no image data for color change.",
        )

    try:
        logger.info("[color] Calling Gemini for color='%s' on image: %s", color_hex, image_url)
        result = await loop.run_in_executor(None, _call_gemini)
        logger.info("[color] Gemini color change complete (%d bytes)", len(result))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[color] Gemini color change failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini color change failed: {exc}",
        )
