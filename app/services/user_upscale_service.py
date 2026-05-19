"""
Standalone user image upscale with provider fallback:

  1. KIE.ai (topaz/image-upscale) — create task (4 retries), poll, download
  2. Modal GPU enhancement pipeline (FashionRealism*)
  3. fal.ai Topaz (``fal-ai/topaz/upscale/image`` — same family as KIE ``topaz/image-upscale``)

**Transport:** HTTP calls run in the FastAPI request (no Celery, no ``kie_rate_limiter``).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
from typing import Any, Callable, Optional, Tuple

import httpx
from PIL import Image

from app.config import settings

logger = logging.getLogger("user_upscale")

_KIE_CREATE_URL = "https://api.kie.ai/api/v1/jobs/createTask"
_KIE_STATUS_URL = "https://api.kie.ai/api/v1/jobs/recordInfo"

USER_UPSCALE_CREATE_RETRIES = 4
_MODAL_VARIANT_BY_FACTOR = {2: "2k", 4: "4k", 8: "8k"}
_KIE_FAIL_STATES = frozenset({"fail", "failed", "error"})


def _kie_bearer() -> str:
    key = (getattr(settings, "KIE_API_KEY", "") or "").strip()
    if not key:
        key = (getattr(settings, "SEEDDREAM_API_KEY", "") or "").strip()
    if not key:
        raise RuntimeError("KIE_API_KEY / SEEDDREAM_API_KEY not set for KIE upscale")
    return key


def _fal_api_key() -> str:
    return (getattr(settings, "FAL_API_KEY", "") or "").strip()


def _extract_output_url(data: Any) -> str:
    """Best-effort URL extraction from KIE/fal result payloads."""
    if isinstance(data, str) and data.startswith("http"):
        return data
    if not isinstance(data, dict):
        return ""
    for key in ("url", "image_url"):
        value = data.get(key)
        if isinstance(value, str) and value.startswith("http"):
            return value
    image_obj = data.get("image")
    if isinstance(image_obj, str) and image_obj.startswith("http"):
        return image_obj
    if isinstance(image_obj, dict):
        value = image_obj.get("url") or image_obj.get("image_url")
        if isinstance(value, str) and value.startswith("http"):
            return value
    for candidate in (data.get("output"), data.get("result"), data.get("data")):
        if isinstance(candidate, str) and candidate.startswith("http"):
            return candidate
        if isinstance(candidate, dict):
            nested = _extract_output_url(candidate)
            if nested:
                return nested
        if isinstance(candidate, list):
            for item in candidate:
                nested = _extract_output_url(item)
                if nested:
                    return nested
    urls = (
        data.get("resultUrls")
        or data.get("result_urls")
        or data.get("urls")
        or data.get("images")
        or []
    )
    if isinstance(urls, list):
        for item in urls:
            if isinstance(item, str) and item.startswith("http"):
                return item
            if isinstance(item, dict):
                nested = _extract_output_url(item)
                if nested:
                    return nested
    return ""


def _parse_kie_result_json(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


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
                attempt,
                USER_UPSCALE_CREATE_RETRIES,
                trace_id,
                exc,
            )
            if attempt < USER_UPSCALE_CREATE_RETRIES:
                await asyncio.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"KIE createTask failed after {USER_UPSCALE_CREATE_RETRIES} attempts: {last_exc}"
    )


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
    if state in _KIE_FAIL_STATES:
        fail_msg = data.get("failMsg") or data.get("fail_msg") or "Unknown KIE failure"
        fail_code = data.get("failCode") or data.get("fail_code")
        raise RuntimeError(f"KIE upscale task failed: {fail_msg} (code={fail_code})")
    if state == "success":
        parsed = _parse_kie_result_json(data.get("resultJson"))
        url = _extract_output_url(parsed)
        if url:
            return state, url
        return state, None
    return state, None


async def _await_upscale_result(task_id: str, trace_id: str) -> str:
    deadline = time.monotonic() + float(settings.KIE_UPSCALE_MAX_WAIT_S or 900)
    interval = float(
        getattr(settings, "USER_UPSCALE_POLL_INTERVAL_S", 0)
        or getattr(settings, "PHOTOSHOOT_KIE_POLL_INTERVAL_S", 0)
        or settings.SEEDDREAM_RETRY_DELAY
        or 3
    )
    max_iters = int(
        getattr(settings, "USER_UPSCALE_POLL_MAX_ITERS", 0)
        or settings.SEEDDREAM_MAX_RETRIES
        or 200
    )
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
                task_id,
                consecutive_errors,
                exc,
            )
            if consecutive_errors >= 10:
                raise RuntimeError(
                    f"KIE recordInfo failed {consecutive_errors}x for task_id={task_id}: {exc}"
                ) from exc
            await asyncio.sleep(interval)
            continue
        if state == "success" and url:
            logger.info(
                "[user-upscale] result task_id=%s trace=%s poll=%d",
                task_id,
                trace_id,
                i + 1,
            )
            return url
        if state == "success" and not url:
            logger.debug(
                "[user-upscale] success without URL yet task_id=%s poll=%d",
                task_id,
                i + 1,
            )
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
            logger.info("[user-upscale] downloaded trace=%s bytes=%d", trace_id, len(data))
            return data
        except Exception as exc:
            logger.warning(
                "[user-upscale] download attempt %d/%d trace=%s: %s",
                attempt,
                max_attempts,
                trace_id,
                exc,
            )
            if attempt >= max_attempts:
                raise
            await asyncio.sleep(min(2**attempt, 8))
    raise RuntimeError("download unreachable")


async def _download_source_image(url: str, trace_id: str) -> bytes:
    headers_list: list[dict[str, str]] = [{}]
    if _fal_api_key():
        lowered = _fal_api_key().lower()
        auth = (
            _fal_api_key()
            if lowered.startswith("key ") or lowered.startswith("bearer ")
            else f"Key {_fal_api_key()}"
        )
        headers_list.append({"Authorization": auth})
    last_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        for attempt in range(1, 4):
            for headers in headers_list:
                try:
                    resp = await client.get(url, headers=headers)
                    resp.raise_for_status()
                    return resp.content
                except Exception as exc:
                    last_exc = exc
            await asyncio.sleep(min(2**attempt, 6))
    raise RuntimeError(f"Failed to download source image trace={trace_id}: {last_exc}") from last_exc


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


async def _finalize_upscale_bytes(
    raw: bytes,
    upscale_factor: int,
) -> tuple[bytes, str]:
    png_bytes = await asyncio.to_thread(image_to_png_bytes, raw)
    w, h = await asyncio.to_thread(image_dimensions_from_bytes, png_bytes)
    label = output_resolution_label(upscale_factor, w, h)
    return png_bytes, label


async def run_standalone_kie_upscale(
    image_url: str,
    upscale_factor: int,
    trace_id: str,
) -> tuple[bytes, str]:
    """Submit KIE topaz upscale → poll → download. Returns (image_bytes, resolution_notes)."""
    task_id = await _kie_create_upscale_task(
        source_image_url=image_url,
        upscale_factor=upscale_factor,
        trace_id=trace_id,
    )
    result_url = await _await_upscale_result(task_id, trace_id)
    raw = await _stream_download(result_url, trace_id)
    return await _finalize_upscale_bytes(raw, upscale_factor)


async def run_standalone_modal_upscale(
    image_url: str,
    upscale_factor: int,
    trace_id: str,
) -> tuple[bytes, str]:
    """Download source image and run Modal FashionRealism enhance pipeline."""
    from app.services.modal_enhance_service import _run_modal_upscale

    source_bytes = await _download_source_image(image_url, trace_id)
    outputs = await _run_modal_upscale(source_bytes, trace_id)
    if not isinstance(outputs, dict) or not outputs:
        raise RuntimeError(f"Modal upscale returned invalid payload: {type(outputs)}")

    preferred = _MODAL_VARIANT_BY_FACTOR.get(int(upscale_factor), "4k")
    order = [preferred, "8k", "4k", "2k", "1k"]
    seen: set[str] = set()
    raw: bytes | None = None
    for key in order:
        if key in seen:
            continue
        seen.add(key)
        candidate = outputs.get(key)
        if isinstance(candidate, bytes) and candidate:
            raw = candidate
            break
    if raw is None:
        raise RuntimeError(
            f"Modal upscale produced no usable variant (wanted {preferred}, keys={list(outputs.keys())})"
        )
    return await _finalize_upscale_bytes(raw, upscale_factor)


def _fal_topaz_model_id() -> str:
    return (
        getattr(settings, "FAL_UPSCALE_MODEL", "") or "fal-ai/topaz/upscale/image"
    ).strip()


def _build_fal_topaz_arguments(image_url: str, upscale_factor: int) -> dict[str, Any]:
    """Mirror KIE topaz/image-upscale inputs: public image URL + upscale factor."""
    enhance_model = (
        getattr(settings, "FAL_TOPAZ_ENHANCE_MODEL", "") or "Standard V2"
    ).strip()
    return {
        "image_url": image_url,
        "upscale_factor": float(int(upscale_factor)),
        "output_format": "png",
        "model": enhance_model,
        "face_enhancement": True,
    }


def _run_fal_upscale_sync(image_url: str, upscale_factor: int) -> str:
    try:
        import fal_client  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "fal-client is not installed; install fal-client for fal upscale fallback."
        ) from exc

    key = _fal_api_key()
    if not key:
        raise RuntimeError("FAL_API_KEY is missing.")

    model_id = _fal_topaz_model_id()
    if "topaz" not in model_id.lower():
        logger.warning(
            "[user-upscale] FAL_UPSCALE_MODEL=%s is not Topaz; expected fal-ai/topaz/upscale/image",
            model_id,
        )

    client = fal_client.SyncClient(key=key)
    arguments = _build_fal_topaz_arguments(image_url, upscale_factor)
    result = client.subscribe(model_id, arguments=arguments, with_logs=False)
    output_url = _extract_output_url(result if isinstance(result, dict) else {})
    if output_url:
        return output_url
    raise RuntimeError(f"fal Topaz upscale completed but no output URL found: {result!r}")


async def run_standalone_fal_upscale(
    image_url: str,
    upscale_factor: int,
    trace_id: str,
) -> tuple[bytes, str]:
    """Run fal.ai Topaz upscale and download the result."""
    loop = asyncio.get_running_loop()
    output_url = await loop.run_in_executor(
        None,
        lambda: _run_fal_upscale_sync(image_url, upscale_factor),
    )
    raw = await _stream_download(output_url, trace_id)
    return await _finalize_upscale_bytes(raw, upscale_factor)


async def run_user_upscale_with_fallback(
    image_url: str,
    upscale_factor: int,
    trace_id: str,
) -> tuple[bytes, str, str]:
    """
    Try KIE → Modal → fal. Returns (png_bytes, resolution_label, provider_used).
    """
    errors: list[str] = []
    providers: list[tuple[str, Callable[..., Any]]] = [
        ("kie", run_standalone_kie_upscale),
        ("modal", run_standalone_modal_upscale),
        ("fal", run_standalone_fal_upscale),
    ]

    for name, fn in providers:
        if name == "fal" and not _fal_api_key():
            errors.append("fal: FAL_API_KEY not configured")
            continue
        try:
            logger.info("[user-upscale] trying provider=%s trace=%s", name, trace_id)
            png_bytes, label = await fn(image_url, upscale_factor, trace_id)
            logger.info("[user-upscale] success provider=%s trace=%s", name, trace_id)
            return png_bytes, label, name
        except Exception as exc:
            err = f"{name}: {exc}"
            errors.append(err)
            logger.warning("[user-upscale] provider failed trace=%s %s", trace_id, err)

    raise RuntimeError(f"All upscale providers failed. {' | '.join(errors)}")
