"""
Enhancement service with switchable upscale provider.

Providers:
  - modal: existing Modal GPU enhancement pipeline
  - kie:   kie.ai topaz/image-upscale (4K public URL -> 8K), then downsample to 4K/2K/1K
"""

import asyncio
import io
import json
import logging
from datetime import datetime, timezone

import httpx
from PIL import Image

from app.config import settings
from app.database import get_upscaling_collection
from app.services.s3_service import upload_bytes_to_s3

logger = logging.getLogger("modal_enhance")

_KIE_CREATE_URL = "https://api.kie.ai/api/v1/jobs/createTask"
_KIE_STATUS_URL = "https://api.kie.ai/api/v1/jobs/recordInfo"


async def warmup_modal() -> None:
    """Best-effort Modal import check during app boot."""
    try:
        import modal  # noqa: F401
        logger.info(
            "[modal] Modal package found — primary=%s fallback=%s app=%s",
            settings.MODAL_CLS_PRIMARY,
            settings.MODAL_CLS_FALLBACK,
            settings.MODAL_APP_NAME,
        )
    except ImportError:
        logger.warning("[modal] Modal package not installed — modal upscaling unavailable.")


def _fit_within_box(orig_w: int, orig_h: int, max_w: int, max_h: int) -> tuple[int, int]:
    if orig_w <= 0 or orig_h <= 0:
        return max_w, max_h
    aspect = orig_w / orig_h
    if aspect >= (max_w / max_h):
        new_w = max_w
        new_h = max(1, round(max_w / aspect))
    else:
        new_h = max_h
        new_w = max(1, round(max_h * aspect))
    return new_w, new_h


def _downsample_from_8k_bytes(img_bytes: bytes) -> dict[str, bytes]:
    """
    Downsample one 8K (or near-8K) image into 8K/4K/2K/1K PNG bytes.
    Mirrors the aspect-preserving strategy from upscaling_data.py.
    """
    with Image.open(io.BytesIO(img_bytes)) as img:
        img = img.convert("RGB")
        src_w, src_h = img.size
        boxes = {
            "8k": (7680, 4320),
            "4k": (3840, 2160),
            "2k": (2560, 1440),
            "1k": (1280, 720),
        }
        out: dict[str, bytes] = {}
        for label, (max_w, max_h) in boxes.items():
            w, h = _fit_within_box(src_w, src_h, max_w, max_h)
            resized = img.resize((w, h), Image.LANCZOS)
            buf = io.BytesIO()
            resized.save(buf, format="PNG")
            out[label] = buf.getvalue()
        return out


async def _upload_upscaled_outputs(prefix: str, outputs: dict[str, bytes]) -> dict[str, str]:
    label_to_key = {
        "8k": "8k_upscaled",
        "4k": "4k_upscaled",
        "2k": "2k_upscaled",
        "1k": "1k_upscaled",
    }
    upload_tasks = {
        label: upload_bytes_to_s3(data, f"{prefix}_{label}_upscaled.png", "image/png")
        for label, data in outputs.items()
        if label in label_to_key
    }
    upscaled_urls: dict[str, str] = {}
    if upload_tasks:
        labels = list(upload_tasks.keys())
        urls = await asyncio.gather(*upload_tasks.values())
        for label, url in zip(labels, urls):
            upscaled_urls[label_to_key[label]] = url
            logger.info("[upscale] Uploaded %s → %s", label_to_key[label], url[:80])
    return upscaled_urls


async def _run_modal_upscale(image_bytes: bytes, image_id: str) -> dict[str, bytes]:
    import modal  # noqa: E402

    filename = f"{image_id}.png"
    app_name = settings.MODAL_APP_NAME
    cls_primary = settings.MODAL_CLS_PRIMARY
    cls_fallback = settings.MODAL_CLS_FALLBACK

    try:
        logger.info("[modal] Trying %s (%s) ...", cls_primary, app_name)
        cls = modal.Cls.from_name(app_name, cls_primary)
        return await cls().enhance.remote.aio(image_bytes, filename)
    except Exception as primary_err:
        logger.warning(
            "[modal] %s failed (%s: %s) — fallback to %s",
            cls_primary, type(primary_err).__name__, primary_err, cls_fallback,
        )
        cls = modal.Cls.from_name(app_name, cls_fallback)
        return await cls().enhance.remote.aio(image_bytes, filename)


async def _run_kie_upscale(source_image_url: str, image_id: str) -> dict[str, bytes]:
    if not source_image_url:
        raise ValueError("source_image_url is required for KIE upscale.")
    if not (settings.SEEDDREAM_API_KEY or "").strip():
        raise ValueError("SEEDDREAM_API_KEY is missing for KIE upscale.")

    payload = json.dumps({
        "model": settings.KIE_UPSCALE_MODEL,
        "input": {
            "image_url": source_image_url,
            "upscale_factor": str(settings.KIE_UPSCALE_FACTOR),
        },
    })
    headers = {
        "Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}",
        "Content-Type": "application/json",
    }
    logger.info(
        "[kie-upscale] Submit image_id=%s model=%s factor=%s",
        image_id, settings.KIE_UPSCALE_MODEL, settings.KIE_UPSCALE_FACTOR,
    )
    async with httpx.AsyncClient(timeout=60) as client:
        create_resp = await client.post(_KIE_CREATE_URL, headers=headers, content=payload)
        create_resp.raise_for_status()
    task_id = create_resp.json().get("data", {}).get("taskId")
    if not task_id:
        raise RuntimeError(f"No taskId from kie upscale createTask: {create_resp.text}")

    result_url = ""
    async with httpx.AsyncClient(timeout=60) as client:
        for attempt in range(1, settings.SEEDDREAM_MAX_RETRIES + 1):
            poll_resp = await client.get(f"{_KIE_STATUS_URL}?taskId={task_id}", headers=headers)
            poll_resp.raise_for_status()
            data = poll_resp.json().get("data", {})
            state = data.get("state")
            if state == "success":
                result_json = data.get("resultJson", "{}")
                parsed = json.loads(result_json) if isinstance(result_json, str) else (result_json or {})
                urls = parsed.get("resultUrls") or parsed.get("urls") or []
                if isinstance(urls, list) and urls:
                    result_url = str(urls[0])
                    break
                maybe_url = parsed.get("url") or parsed.get("image_url")
                if maybe_url:
                    result_url = str(maybe_url)
                    break
                raise RuntimeError("KIE upscale succeeded but no output URL was found.")
            if state == "fail":
                raise RuntimeError("KIE upscale task failed.")
            await asyncio.sleep(settings.SEEDDREAM_RETRY_DELAY)
        else:
            raise RuntimeError(f"KIE upscale timed out after {settings.SEEDDREAM_MAX_RETRIES} polls.")

    async with httpx.AsyncClient(timeout=180) as client:
        out_resp = await client.get(result_url, follow_redirects=True)
        out_resp.raise_for_status()
        bytes_8k = out_resp.content
    logger.info("[kie-upscale] Downloaded 8K output for image_id=%s (%d bytes)", image_id, len(bytes_8k))
    return _downsample_from_8k_bytes(bytes_8k)


async def enhance_and_upload(
    image_bytes: bytes,
    photoshoot_id: str,
    image_id: str,
    seeddream_4k_url: str,
    seeddream_2k_url: str,
    seeddream_1k_url: str,
    source_image_url: str = "",
    upscaling_col=None,
) -> dict:
    """
    Upscale via provider configured by ``WHICH_UPSCALE`` and insert upscaling_data doc.
    """
    prefix = f"photoshoots/{photoshoot_id}/{image_id}"
    now = datetime.now(timezone.utc)
    upscaled_urls: dict[str, str] = {}
    provider = (settings.WHICH_UPSCALE or "modal").strip().lower()

    try:
        if provider == "kie":
            logger.info("[upscale] Using KIE provider for image_id=%s", image_id)
            outputs = await _run_kie_upscale(source_image_url=source_image_url, image_id=image_id)
            upscaled_urls = await _upload_upscaled_outputs(prefix, outputs)
        else:
            logger.info("[upscale] Using Modal provider for image_id=%s", image_id)
            outputs = await _run_modal_upscale(image_bytes=image_bytes, image_id=image_id)
            upscaled_urls = await _upload_upscaled_outputs(prefix, outputs)
    except ImportError:
        logger.warning("[upscale] Modal import failed — skipping upscale.")
    except Exception as exc:
        logger.error("[upscale] Provider=%s failed for image_id=%s: %s", provider, image_id, exc)

    doc = {
        "photoshoot_id": photoshoot_id,
        "image_id": image_id,
        "1k_upscaled": upscaled_urls.get("1k_upscaled", ""),
        "2k_upscaled": upscaled_urls.get("2k_upscaled", ""),
        "4k_upscaled": upscaled_urls.get("4k_upscaled", ""),
        "8k_upscaled": upscaled_urls.get("8k_upscaled", ""),
        "1k": seeddream_1k_url,
        "2k": seeddream_2k_url,
        "4k": seeddream_4k_url,
        "created_at": now,
        "updated_at": now,
    }
    col = upscaling_col if upscaling_col is not None else get_upscaling_collection()
    await col.insert_one(doc)
    logger.info("[upscale] upscaling_data record inserted for image_id=%s", image_id)
    return doc
