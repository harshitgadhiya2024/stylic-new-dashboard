"""
Shared KIE image generation/edit fallback:
1) nano-banana-2 (N retries)
2) gpt-image-2 (N retries)

Used by custom-pose, custom-face, custom-background, and garment edit features.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

import httpx

from app.config import settings

logger = logging.getLogger("kie_image_fallback")

_CREATE_URL = "https://api.kie.ai/api/v1/jobs/createTask"
_STATUS_URL = "https://api.kie.ai/api/v1/jobs/recordInfo"


def _kie_api_key() -> str:
    key = (getattr(settings, "KIE_API_KEY", "") or "").strip()
    if not key:
        key = (getattr(settings, "SEEDDREAM_API_KEY", "") or "").strip()
    if not key:
        raise RuntimeError("KIE_API_KEY / SEEDDREAM_API_KEY not configured")
    return key


def _model_chain() -> list[str]:
    return [
        (getattr(settings, "KIE_PRIMARY_IMAGE_MODEL", "") or "nano-banana-2").strip(),
        (getattr(settings, "KIE_FALLBACK_IMAGE_MODEL", "") or "gpt-image-2").strip(),
    ]


def _build_input_payload(
    model: str,
    prompt: str,
    image_urls: Optional[list[str]] = None,
) -> dict[str, Any]:
    aspect = (getattr(settings, "KIE_FEATURE_ASPECT_RATIO", "") or "3:4").strip()
    res = (getattr(settings, "KIE_FEATURE_RESOLUTION", "") or "1K").strip()
    images = [u for u in (image_urls or []) if u]
    if model == "gpt-image-2":
        payload: dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": aspect,
            "resolution": res,
        }
        if images:
            payload["input_urls"] = images
        return payload

    payload = {
        "prompt": prompt,
        "aspect_ratio": aspect,
        "resolution": res,
        "output_format": "png",
    }
    if images:
        payload["image_input"] = images
    return payload


async def _submit_task(model: str, prompt: str, image_urls: Optional[list[str]]) -> str:
    body = {
        "model": model,
        "input": _build_input_payload(model, prompt, image_urls=image_urls),
    }
    headers = {"Authorization": f"Bearer {_kie_api_key()}", "Content-Type": "application/json"}
    timeout = int(getattr(settings, "KIE_HTTP_TIMEOUT", 300) or 300)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(_CREATE_URL, headers=headers, json=body)
        r.raise_for_status()
        data = r.json() or {}
    if data.get("code") != 200:
        raise RuntimeError(f"KIE createTask failed: {data}")
    task_id = (data.get("data") or {}).get("taskId")
    if not task_id:
        raise RuntimeError(f"KIE createTask missing taskId: {data}")
    return str(task_id)


async def _poll_result_url(task_id: str) -> str:
    headers = {"Authorization": f"Bearer {_kie_api_key()}"}
    max_retries = int(getattr(settings, "SEEDDREAM_MAX_RETRIES", 120) or 120)
    delay = float(getattr(settings, "SEEDDREAM_RETRY_DELAY", 5) or 5)
    timeout = int(getattr(settings, "KIE_HTTP_TIMEOUT", 300) or 300)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for _ in range(max_retries):
            r = await client.get(_STATUS_URL, headers=headers, params={"taskId": task_id})
            r.raise_for_status()
            body = r.json() or {}
            if body.get("code") != 200:
                await asyncio.sleep(delay)
                continue
            data = body.get("data") or {}
            state = (data.get("state") or "").lower()
            if state == "success":
                raw = data.get("resultJson")
                parsed = json.loads(raw) if isinstance(raw, str) else (raw or {})
                urls = (
                    parsed.get("resultUrls")
                    or parsed.get("result_urls")
                    or parsed.get("urls")
                    or parsed.get("images")
                    or []
                )
                if isinstance(urls, list) and urls:
                    first = urls[0]
                    if isinstance(first, dict):
                        first = first.get("url") or first.get("image_url") or ""
                    if isinstance(first, str) and first:
                        return first
                for k in ("url", "image", "image_url"):
                    v = parsed.get(k)
                    if isinstance(v, str) and v:
                        return v
                raise RuntimeError(f"KIE success but no result URL: {body}")
            if state in {"fail", "failed", "error"}:
                raise RuntimeError(f"KIE task failed: {body}")
            await asyncio.sleep(delay)
    raise RuntimeError("KIE task polling timed out")


async def _download(url: str) -> bytes:
    timeout = int(getattr(settings, "KIE_HTTP_TIMEOUT", 300) or 300)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.content


async def generate_image_with_model_fallback(
    prompt: str,
    image_urls: Optional[list[str]] = None,
    *,
    label: str = "feature",
) -> bytes:
    """
    Run configured model chain with retries per model and return image bytes.
    """
    retries = int(getattr(settings, "KIE_FEATURE_MODEL_RETRIES", 3) or 3)
    errors: list[str] = []
    for model in _model_chain():
        for attempt in range(1, retries + 1):
            try:
                logger.info("[%s] model=%s attempt=%d/%d", label, model, attempt, retries)
                task_id = await _submit_task(model, prompt, image_urls=image_urls)
                out_url = await _poll_result_url(task_id)
                return await _download(out_url)
            except Exception as exc:
                err = f"{model} attempt {attempt}/{retries}: {exc}"
                logger.warning("[%s] %s", label, err)
                errors.append(err)
                if attempt < retries:
                    await asyncio.sleep(2.0)
    raise RuntimeError(f"All fallback models failed. {' | '.join(errors)}")

