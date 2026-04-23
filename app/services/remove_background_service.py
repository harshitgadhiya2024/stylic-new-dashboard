"""
Remove-background pipeline (KIE recraft/remove-background, optional fal.ai fallback).

Logic mirrors ``remove-background.py`` at repo root; API keys come from ``app.config.settings``.

**Transport:** KIE/fal HTTP calls are made directly from the API process (no Celery worker,
no Redis-backed task queue, and this module must not use ``kie_rate_limiter`` — that is
only for the photoshoot SeedDream submit path).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Optional

import httpx
import requests

from app.config import settings


def _kie_api_key() -> str:
    key = (getattr(settings, "KIE_API_KEY", "") or "").strip()
    if not key:
        key = (getattr(settings, "SEEDDREAM_API_KEY", "") or "").strip()
    if not key:
        raise RuntimeError("KIE_API_KEY / SEEDDREAM_API_KEY not set")
    return key


def _fal_api_key() -> str:
    return (getattr(settings, "FAL_API_KEY", "") or "").strip()


def _extract_output_url(data: dict) -> str:
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
            for key in ("url", "image", "image_url"):
                value = candidate.get(key)
                if isinstance(value, str) and value.startswith("http"):
                    return value
            images = candidate.get("images")
            if isinstance(images, list):
                for item in images:
                    if isinstance(item, str) and item.startswith("http"):
                        return item
                    if isinstance(item, dict):
                        value = item.get("url") or item.get("image") or item.get("image_url")
                        if isinstance(value, str) and value.startswith("http"):
                            return value
        if isinstance(candidate, list):
            for item in candidate:
                if isinstance(item, str) and item.startswith("http"):
                    return item
                if isinstance(item, dict):
                    value = item.get("url") or item.get("image") or item.get("image_url")
                    if isinstance(value, str) and value.startswith("http"):
                        return value
    return ""


def _fal_auth_header() -> dict[str, str]:
    key = _fal_api_key()
    if not key:
        raise RuntimeError("FAL_API_KEY is missing.")
    lowered = key.lower()
    if lowered.startswith("key ") or lowered.startswith("bearer "):
        auth_value = key
    else:
        auth_value = f"Key {key}"
    return {"Authorization": auth_value}


def run_kie_remove_bg(image_url: str) -> str:
    create_url = "https://api.kie.ai/api/v1/jobs/createTask"
    result_url = "https://api.kie.ai/api/v1/jobs/recordInfo"
    headers = {
        "Authorization": f"Bearer {_kie_api_key()}",
        "Content-Type": "application/json",
    }
    create_payload = {
        "model": "recraft/remove-background",
        "input": {"image": image_url},
    }
    create_response = requests.post(create_url, json=create_payload, headers=headers, timeout=60)
    create_response.raise_for_status()
    create_json = create_response.json()
    task_id = (
        create_json.get("data", {}).get("taskId")
        or create_json.get("data", {}).get("id")
        or create_json.get("taskId")
        or create_json.get("id")
    )
    if not task_id:
        raise RuntimeError(f"Kie createTask did not return task id: {create_json}")

    interval = float(getattr(settings, "PHOTOSHOOT_KIE_POLL_INTERVAL_S", 2.0) or 2.0)
    for _ in range(120):
        poll_response = requests.get(
            result_url,
            headers={"Authorization": headers["Authorization"]},
            params={"taskId": task_id},
            timeout=60,
        )
        poll_response.raise_for_status()
        poll_json = poll_response.json()
        data = poll_json.get("data") or {}
        state = str(data.get("state") or "").lower()
        if state == "success":
            raw = data.get("resultJson")
            parsed: Any = raw
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
            urls = parsed.get("resultUrls") or parsed.get("result_urls") or []
            if urls and isinstance(urls[0], str) and urls[0].startswith("http"):
                return urls[0]
            output_url = _extract_output_url(parsed) or _extract_output_url(data) or _extract_output_url(poll_json)
            if output_url:
                return output_url
            raise RuntimeError(f"Kie task succeeded but output URL not found: {poll_json}")
        if state in {"fail", "failed", "error"}:
            fail_msg = data.get("failMsg") or data.get("fail_msg") or "Unknown Kie failure"
            fail_code = data.get("failCode") or data.get("fail_code")
            raise RuntimeError(f"Kie task failed: {fail_msg} (code={fail_code})")
        time.sleep(interval)
    raise TimeoutError("Kie polling timed out.")


def run_fal_remove_bg(image_url: str) -> str:
    try:
        import fal_client  # type: ignore
    except ImportError as exc:
        raise RuntimeError("fal-client is not installed; install fal-client or rely on KIE only.") from exc

    key = _fal_api_key()
    if not key:
        raise RuntimeError("FAL_API_KEY is missing.")

    fal_model_id = "fal-ai/birefnet/v2"
    fal_params = {
        "model": "General Use (Heavy)",
        "operating_resolution": "2048x2048",
        "output_format": "png",
        "refine_foreground": True,
    }
    client = fal_client.SyncClient(key=key)
    result = client.subscribe(
        fal_model_id,
        arguments={"image_url": image_url, **fal_params},
        with_logs=False,
    )
    output_url = _extract_output_url(result if isinstance(result, dict) else {})
    if output_url:
        return output_url
    raise RuntimeError(f"Fal completed but no output URL found: {result!r}")


def _with_retries(label: str, attempts: int, func: Callable[..., str], *args: Any) -> str:
    last_error: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            return func(*args)
        except Exception as error:
            last_error = error
            if attempt < attempts:
                time.sleep(1.0)
    raise RuntimeError(f"{label} failed after {attempts} attempts.") from last_error


def remove_background_to_output_url(image_url: str) -> str:
    """Return a temporary provider URL for the background-removed image."""
    try:
        return _with_retries("kie.ai", 2, run_kie_remove_bg, image_url)
    except Exception as kie_exc:
        if _fal_api_key():
            try:
                return _with_retries("fal.ai", 2, run_fal_remove_bg, image_url)
            except Exception as fal_exc:
                raise RuntimeError(f"KIE failed ({kie_exc!r}); fal failed ({fal_exc!r})") from fal_exc
        raise kie_exc


async def download_image_bytes(url: str) -> bytes:
    headers_list: list[dict[str, str]] = [{}]
    if _fal_api_key():
        headers_list.append(_fal_auth_header())
    last_error: Optional[BaseException] = None
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        for _attempt in range(1, 6):
            for headers in headers_list:
                try:
                    resp = await client.get(url, headers=headers)
                    resp.raise_for_status()
                    return resp.content
                except Exception as error:
                    last_error = error
            await asyncio.sleep(1.5)
    raise RuntimeError(f"Failed to download output image from {url}") from last_error
