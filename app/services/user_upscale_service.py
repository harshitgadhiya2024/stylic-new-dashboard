"""
Standalone user image upscale via KIE.ai only (topaz/image-upscale).

Separate from photoshoot upscale. Create-task uses **4 retries**; poll + download
reuse the same KIE endpoints and timing settings as ``modal_enhance_service``.

**Transport:** KIE HTTP calls run directly in the FastAPI request (no Celery, no Redis
job queue, no ``kie_rate_limiter``).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from typing import Optional, Tuple

import httpx
from PIL import Image

from app.config import settings

logger = logging.getLogger("user_upscale")

_KIE_CREATE_URL = "https://api.kie.ai/api/v1/jobs/createTask"
_KIE_STATUS_URL = "https://api.kie.ai/api/v1/jobs/recordInfo"

USER_UPSCALE_CREATE_RETRIES = 4


def _kie_bearer() -> str:
    key = (getattr(settings, "KIE_API_KEY", "") or "").strip()
    if not key:
        key = (getattr(settings, "SEEDDREAM_API_KEY", "") or "").strip()
    if not key:
        raise RuntimeError("KIE_API_KEY / SEEDDREAM_API_KEY not set for KIE upscale")
    return key


async def _kie_create_upscale_task(
    *,
    source_image_url: str,
    upscale_factor: int,
    trace_id: str,
) -> str:
    headers = {
        "Authorization": f"Bearer {_kie_bearer()}",
        "Content-Type": "application/json",
    }
    body: dict = {
        "model": settings.KIE_UPSCALE_MODEL,
        "input": {
            "image_url": source_image_url,
            "upscale_factor": str(int(upscale_factor)),
        },
        "metadata": {
            "purpose": "user_upscale",
            "upscale_trace_id": trace_id,
        },
    }
    last_exc: Exception | None = None
    for attempt in range(1, USER_UPSCALE_CREATE_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=settings.KIE_HTTP_TIMEOUT) as client:
                r = await client.post(_KIE_CREATE_URL, headers=headers, json=body)
                r.raise_for_status()
                data = r.json() or {}
            if data.get("code") != 200:
                raise RuntimeError(f"KIE createTask error: {data}")
            task_id = (data.get("data") or {}).get("taskId")
            if not task_id:
                raise RuntimeError(f"KIE createTask missing taskId: {data}")
            return str(task_id)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "[user-upscale] createTask attempt %d/%d trace=%s: %s",
                attempt, USER_UPSCALE_CREATE_RETRIES, trace_id, exc,
            )
            if attempt < USER_UPSCALE_CREATE_RETRIES:
                await asyncio.sleep(min(2**attempt, 8))
    raise RuntimeError(f"KIE createTask failed after {USER_UPSCALE_CREATE_RETRIES} attempts: {last_exc}")


async def _kie_poll_once(task_id: str) -> Tuple[str, Optional[str]]:
    headers = {"Authorization": f"Bearer {_kie_bearer()}"}
    async with httpx.AsyncClient(timeout=settings.KIE_HTTP_TIMEOUT) as client:
        resp = await client.get(f"{_KIE_STATUS_URL}?taskId={task_id}", headers=headers)
        resp.raise_for_status()
    body = resp.json() or {}
    if body.get("code") != 200:
        raise RuntimeError(f"KIE recordInfo error: {body}")
    data = body.get("data") or {}
    state = (data.get("state") or "").lower()
    if state == "success":
        raw = data.get("resultJson", "{}")
        parsed = json.loads(raw) if isinstance(raw, str) else (raw or {})
        urls = parsed.get("resultUrls") or parsed.get("urls") or []
        if isinstance(urls, list) and urls:
            return state, str(urls[0])
        maybe = parsed.get("url") or parsed.get("image_url")
        if maybe:
            return state, str(maybe)
        raise RuntimeError(f"KIE upscale success but no URL: {parsed}")
    return state, None


async def _await_upscale_result(task_id: str, trace_id: str) -> str:
    deadline = time.monotonic() + float(settings.KIE_UPSCALE_MAX_WAIT_S or 900)
    interval = float(
        getattr(settings, "PHOTOSHOOT_KIE_POLL_INTERVAL_S", 0)
        or settings.SEEDDREAM_RETRY_DELAY
        or 3
    )
    max_iters = int(settings.SEEDDREAM_MAX_RETRIES or 300)
    consecutive_errors = 0
    for i in range(max_iters):
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"KIE upscale task {task_id} exceeded {settings.KIE_UPSCALE_MAX_WAIT_S}s"
            )
        try:
            state, url = await _kie_poll_once(task_id)
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            logger.warning(
                "[user-upscale] poll error task_id=%s attempt=%d err=%s",
                task_id, consecutive_errors, exc,
            )
            if consecutive_errors >= 10:
                raise RuntimeError(
                    f"KIE recordInfo failed {consecutive_errors}x for task_id={task_id}: {exc}"
                ) from exc
            await asyncio.sleep(interval)
            continue
        if state == "success" and url:
            logger.info("[user-upscale] result task_id=%s trace=%s poll=%d", task_id, trace_id, i + 1)
            return url
        if state == "fail":
            raise RuntimeError(f"KIE upscale task failed: task_id={task_id}")
        await asyncio.sleep(interval)
    raise TimeoutError(f"KIE upscale task {task_id} did not finish in {max_iters} polls")


async def _stream_download(url: str, trace_id: str) -> bytes:
    max_attempts = int(getattr(settings, "KIE_REQUEST_RETRIES", 3) or 3)
    for attempt in range(1, max_attempts + 1):
        try:
            chunks: list[bytes] = []
            async with httpx.AsyncClient(timeout=settings.KIE_HTTP_TIMEOUT) as client:
                async with client.stream("GET", url, follow_redirects=True) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 256):
                        if chunk:
                            chunks.append(chunk)
            data = b"".join(chunks)
            logger.info(
                "[user-upscale] downloaded trace=%s bytes=%d",
                trace_id, len(data),
            )
            return data
        except Exception as exc:
            logger.warning(
                "[user-upscale] download attempt %d/%d trace=%s: %s",
                attempt, max_attempts, trace_id, exc,
            )
            if attempt >= max_attempts:
                raise
            await asyncio.sleep(min(2**attempt, 8))
    raise RuntimeError("download unreachable")


def output_resolution_label(upscale_factor: int, width: int, height: int) -> str:
    return f"{upscale_factor}x ({width}x{height}px)"


def image_dimensions_from_bytes(data: bytes) -> tuple[int, int]:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img.size


def image_to_png_bytes(data: bytes) -> bytes:
    """Normalize to PNG for stable R2 storage and metadata."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=1)
    return buf.getvalue()


async def run_standalone_kie_upscale(
    image_url: str,
    upscale_factor: int,
    trace_id: str,
) -> tuple[bytes, str]:
    """
    Submit KIE topaz upscale → poll → download. Returns (image_bytes, resolution_notes).
    """
    task_id = await _kie_create_upscale_task(
        source_image_url=image_url,
        upscale_factor=upscale_factor,
        trace_id=trace_id,
    )
    result_url = await _await_upscale_result(task_id, trace_id)
    raw = await _stream_download(result_url, trace_id)
    png_bytes = await asyncio.to_thread(image_to_png_bytes, raw)
    w, h = await asyncio.to_thread(image_dimensions_from_bytes, png_bytes)
    label = output_resolution_label(upscale_factor, w, h)
    return png_bytes, label
