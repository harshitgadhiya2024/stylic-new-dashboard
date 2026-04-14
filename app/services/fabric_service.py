"""
Garment image edits (fabric / texture / color).

Uses kie.ai with the same image-edit contract as the photoshoot realism pass:
``nano-banana-pro`` via ``createTask`` + ``recordInfo``, with ``image_input`` as a
URL list and ``aspect_ratio`` / ``resolution`` from app settings.

Public functions:
    change_fabric(image_url: str, fabric: str) -> bytes
    change_garment_upper_lower_fabrics(image_url: str, upper_fabric: str, lower_fabric: str) -> bytes
    change_texture(image_url: str, texture: str) -> bytes
    change_garment_upper_lower_textures(image_url: str, upper_texture: str, lower_texture: str) -> bytes
    change_color(image_url: str, color_hex: str) -> bytes
    change_garment_upper_lower_colors(image_url: str, upper_color_hex: str, lower_color_hex: str) -> bytes
"""

import asyncio
import json
import logging

import httpx
from fastapi import HTTPException, status

from app.config import settings

logger = logging.getLogger("fabric_service")

_CREATE_URL = "https://api.kie.ai/api/v1/jobs/createTask"
_STATUS_URL = "https://api.kie.ai/api/v1/jobs/recordInfo"


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


def _build_upper_lower_fabric_prompt(upper_fabric: str, lower_fabric: str) -> str:
    return (
        "You are a professional fashion image editor.\n\n"
        "You receive a garment reference image (flat lay, ghost mannequin, or similar) that may show "
        "**upper-body garments** (shirts, tops, jackets, kurti tops, etc.) and/or **lower-body garments** "
        "(pants, jeans, skirts, salwar bottoms, lehenga skirts, etc.).\n\n"
        "Change fabric / material exactly as follows:\n"
        f"- **Upper garment(s)**: change the visible fabric/material of every upper-body garment to **{upper_fabric}** "
        f"— correct drape, sheen, weave, and surface finish typical of {upper_fabric}.\n"
        f"- **Lower garment(s)**: change the visible fabric/material of every lower-body garment to **{lower_fabric}** "
        f"— correct drape, sheen, weave, and surface finish typical of {lower_fabric}.\n\n"
        "[REGION RULES]\n"
        "- Split upper vs lower using normal fashion anatomy (waistline, shirt hem vs pant top, dupatta vs skirt).\n"
        "- If the outfit is a **one-piece** (dress, jumpsuit, gown), apply **upper_fabric** to the bodice / torso "
        "  through the natural waist, and **lower_fabric** from the waist through the hem.\n"
        "- If only upper or only lower appears in frame, change only those garment(s) to the matching requested "
        "  material and leave other regions unchanged.\n"
        "- Do not alter skin, hands, shoes, or background.\n\n"
        "[STRICT REQUIREMENTS]\n"
        "- Keep the exact same garment color and **printed pattern layout** where visible — only the material "
        "  read (fiber, weave, sheen, hand) should change to match each requested fabric\n"
        "- Preserve garment shape, cut, silhouette, stitching, buttons, zippers, pleats, and construction\n"
        "- Keep lighting and composition consistent with the original\n"
        "- No text, watermarks, or logos\n"
        "- Photorealistic output\n\n"
        "Return ONLY the edited garment image."
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


def _build_upper_lower_texture_prompt(upper_texture: str, lower_texture: str) -> str:
    return (
        "You are a professional fashion image editor.\n\n"
        "You receive a garment reference image (flat lay, ghost mannequin, or similar) that may show "
        "**upper-body garments** (shirts, tops, jackets, kurti tops, etc.) and/or **lower-body garments** "
        "(pants, jeans, skirts, salwar bottoms, lehenga skirts, etc.).\n\n"
        "Apply surface textures / patterns exactly as follows:\n"
        f"- **Upper garment fabric(s)**: change the visible surface texture or pattern of every upper-body "
        f"garment to: **{upper_texture}**.\n"
        f"- **Lower garment fabric(s)**: change the visible surface texture or pattern of every lower-body "
        f"garment to: **{lower_texture}**.\n\n"
        "[REGION RULES]\n"
        "- Split upper vs lower using normal fashion anatomy (waistline, shirt hem vs pant top, dupatta vs skirt).\n"
        "- If the outfit is a **one-piece** (dress, jumpsuit, gown), apply **upper_texture** to the bodice / "
        "  torso through the natural waist, and **lower_texture** from the waist through the hem.\n"
        "- If only upper or only lower appears in frame, apply only the matching requested texture to those "
        "  garment(s) and leave other regions unchanged.\n"
        "- Do not alter skin, hands, shoes, or background.\n\n"
        "[STRICT REQUIREMENTS]\n"
        "- Keep the same base color tones per region unless the new texture inherently implies a different read "
        "  (e.g. metallic sheen) — still preserve overall hue family where possible\n"
        "- Preserve garment shape, cut, silhouette, stitching, buttons, zippers, pleats, and construction\n"
        "- Make each texture repeat naturally, follow fabric drape and contours, and look physically accurate\n"
        "- Keep lighting, shadows, and composition consistent with the original\n"
        "- No text, watermarks, or logos\n"
        "- Photorealistic output\n\n"
        "Return ONLY the edited garment image."
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


def _build_upper_lower_color_prompt(upper_hex: str, lower_hex: str) -> str:
    upper_desc = _hex_to_rgb_name(upper_hex)
    lower_desc = _hex_to_rgb_name(lower_hex)
    return (
        "You are a professional fashion image editor.\n\n"
        "You receive a garment reference image (flat lay, ghost mannequin, or similar) that may show "
        "**upper-body garments** (shirts, tops, jackets, kurti tops, cholis, etc.) and/or "
        "**lower-body garments** (pants, jeans, skirts, salwar bottoms, lehenga skirts, etc.).\n\n"
        "Apply colors exactly as follows:\n"
        f"- **Upper garment fabric(s)**: recolor every visible upper-body garment to **{upper_desc}** "
        f"(target hex **{upper_hex}**).\n"
        f"- **Lower garment fabric(s)**: recolor every visible lower-body garment to **{lower_desc}** "
        f"(target hex **{lower_hex}**).\n\n"
        "[REGION RULES]\n"
        "- Split upper vs lower using normal fashion anatomy: waistline, shirt hem vs pant top, "
        "  dupatta vs skirt, etc. Do not recolor skin, hands, shoes, or background.\n"
        "- If the outfit is a **one-piece** (dress, jumpsuit, gown, anarkali without a clear skirt break), "
        "  apply the **upper** color from the shoulders / bodice down through the natural waist, and the "
        "  **lower** color from the waist through the hem / legs.\n"
        "- If only upper or only lower garments appear in frame, recolor only those pieces with the "
        "  matching requested color and leave unrelated regions unchanged.\n\n"
        "[STRICT REQUIREMENTS]\n"
        f"- Match **{upper_hex}** and **{lower_hex}** precisely on their respective garment regions\n"
        "- Preserve garment shape, cut, silhouette, pleats, embroidery layout, and construction details\n"
        "- Preserve fabric texture and weave — only adjust color / hue as needed\n"
        "- Keep lighting, shadows, and highlights physically plausible\n"
        "- Keep background and composition unchanged\n"
        "- No text, watermarks, or logos\n"
        "- Photorealistic output\n\n"
        "Return ONLY the edited garment image."
    )


def _require_kie_api_key() -> None:
    if not (settings.SEEDDREAM_API_KEY or "").strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Garment editing is not configured (missing kie.ai API key).",
        )


async def _submit_nano_banana_garment_task(prompt: str, image_url: str, log_label: str) -> str:
    """Submit a kie.ai nano-banana-pro image edit (same input shape as photoshoot realism pass)."""
    _require_kie_api_key()
    model = settings.REALISM_MODEL
    logger.info(
        "[%s] Submitting kie.ai garment edit (%s, %d chars)...",
        log_label, model, len(prompt),
    )
    payload = json.dumps({
        "model": model,
        "input": {
            "prompt":       prompt,
            "image_input":  [image_url],
            "aspect_ratio": settings.REALISM_ASPECT,
            "resolution":   settings.REALISM_QUALITY,
        },
    })
    headers = {
        "Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}",
        "Content-Type":  "application/json",
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(_CREATE_URL, headers=headers, content=payload)
        resp.raise_for_status()
    task_id = resp.json().get("data", {}).get("taskId")
    if not task_id:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"No taskId returned from kie.ai ({log_label}): {resp.text}",
        )
    logger.info("[%s] kie.ai task submitted — task_id=%s", log_label, task_id)
    return task_id


async def _poll_kie_garment_task(task_id: str, log_label: str) -> str:
    """Poll kie.ai until success and return first result URL."""
    logger.info("[%s] Polling kie.ai task_id=%s ...", log_label, task_id)
    headers = {"Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}"}
    for attempt in range(1, settings.SEEDDREAM_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.get(f"{_STATUS_URL}?taskId={task_id}", headers=headers)
                resp.raise_for_status()
            data = resp.json().get("data", {})
            state = data.get("state")
            if state == "success":
                urls = json.loads(data.get("resultJson", "{}")).get("resultUrls", [])
                if urls:
                    logger.info("[%s] kie.ai garment edit complete (attempt %d)", log_label, attempt)
                    return urls[0]
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Garment edit succeeded but no resultUrls ({log_label}).",
                )
            if state == "fail":
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Garment edit task failed on kie.ai ({log_label}).",
                )
            await asyncio.sleep(settings.SEEDDREAM_RETRY_DELAY)
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("[%s] Poll #%d error: %s", log_label, attempt, exc)
            await asyncio.sleep(settings.SEEDDREAM_RETRY_DELAY)
    raise HTTPException(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        detail=f"Garment edit timed out after {settings.SEEDDREAM_MAX_RETRIES} attempts ({log_label}).",
    )


async def _download_result_bytes(url: str, log_label: str) -> bytes:
    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
            logger.info("[%s] Downloaded result (%d bytes)", log_label, len(resp.content))
            return resp.content
        except Exception as exc:
            if attempt == 3:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to download edited garment image ({log_label}): {exc}",
                )
            await asyncio.sleep(3)


async def _edit_garment_via_kie(prompt: str, image_url: str, log_label: str) -> bytes:
    task_id = await _submit_nano_banana_garment_task(prompt, image_url, log_label)
    out_url = await _poll_kie_garment_task(task_id, log_label)
    return await _download_result_bytes(out_url, log_label)


async def change_fabric(image_url: str, fabric: str) -> bytes:
    """
    Apply a new fabric texture to a garment image via kie.ai (nano-banana-pro).

    Args:
        image_url: Public URL of the garment image to edit.
        fabric:    The desired fabric name (e.g. "silk", "cotton", "denim").

    Returns:
        Image bytes of the fabric-changed garment (format from model output).
    """
    prompt = _build_fabric_prompt(fabric)
    label = f"fabric:{fabric}"
    logger.info("[%s] Garment fabric edit — image_url=%s", label, image_url[:120])
    return await _edit_garment_via_kie(prompt, image_url, label)


async def change_garment_upper_lower_fabrics(
    image_url: str,
    upper_fabric: str,
    lower_fabric: str,
) -> bytes:
    """
    Apply distinct upper vs lower garment fabrics/materials via kie.ai (nano-banana-pro).
    """
    prompt = _build_upper_lower_fabric_prompt(upper_fabric, lower_fabric)
    u_short = upper_fabric[:24].replace("\n", " ")
    l_short = lower_fabric[:24].replace("\n", " ")
    label = f"fabric:upper={u_short}_lower={l_short}"
    logger.info("[%s] Garment upper/lower fabric edit — image_url=%s", label, image_url[:120])
    return await _edit_garment_via_kie(prompt, image_url, label)


async def change_texture(image_url: str, texture: str) -> bytes:
    """
    Apply a new surface texture/pattern to a garment image via kie.ai (nano-banana-pro).

    Args:
        image_url: Public URL of the garment image to edit.
        texture:   The desired texture/pattern name (e.g. "plain weave", "checked", "printed").

    Returns:
        Image bytes of the texture-changed garment (format from model output).
    """
    prompt = _build_texture_prompt(texture)
    label = f"texture:{texture}"
    logger.info("[%s] Garment texture edit — image_url=%s", label, image_url[:120])
    return await _edit_garment_via_kie(prompt, image_url, label)


async def change_garment_upper_lower_textures(
    image_url: str,
    upper_texture: str,
    lower_texture: str,
) -> bytes:
    """
    Apply distinct upper vs lower garment surface textures via kie.ai (nano-banana-pro).
    """
    prompt = _build_upper_lower_texture_prompt(upper_texture, lower_texture)
    u_short = upper_texture[:24].replace("\n", " ")
    l_short = lower_texture[:24].replace("\n", " ")
    label = f"texture:upper={u_short}_lower={l_short}"
    logger.info("[%s] Garment upper/lower texture edit — image_url=%s", label, image_url[:120])
    return await _edit_garment_via_kie(prompt, image_url, label)


async def change_color(image_url: str, color_hex: str) -> bytes:
    """
    Apply a new color to a garment image via kie.ai (nano-banana-pro).

    Args:
        image_url:  Public URL of the garment image to edit.
        color_hex:  Target color as a hex code (e.g. "#fff28f", "#fe2a3e").

    Returns:
        Image bytes of the color-changed garment (format from model output).
    """
    prompt = _build_color_prompt(color_hex)
    label = f"color:{color_hex}"
    logger.info("[%s] Garment color edit — image_url=%s", label, image_url[:120])
    return await _edit_garment_via_kie(prompt, image_url, label)


async def change_garment_upper_lower_colors(
    image_url: str,
    upper_color_hex: str,
    lower_color_hex: str,
) -> bytes:
    """
    Recolor upper-body vs lower-body garment regions via kie.ai (nano-banana-pro).

    Args:
        image_url:        Public URL of the garment image to edit.
        upper_color_hex:  Target hex for upper garment(s), e.g. ``#fe2a3e``.
        lower_color_hex:  Target hex for lower garment(s), e.g. ``#112233``.
    """
    prompt = _build_upper_lower_color_prompt(upper_color_hex, lower_color_hex)
    label = f"color:upper={upper_color_hex}_lower={lower_color_hex}"
    logger.info("[%s] Garment upper/lower color edit — image_url=%s", label, image_url[:120])
    return await _edit_garment_via_kie(prompt, image_url, label)
