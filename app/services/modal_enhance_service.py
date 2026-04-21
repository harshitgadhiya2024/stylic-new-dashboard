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
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from PIL import Image

from app.config import settings
from app.database import get_upscaling_collection
from app.services.r2_service import upload_bytes_to_r2

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


_VALID_FORMATS = ("png_fast", "png", "webp_lossless", "jpeg")


def _normalized_format() -> str:
    fmt = (settings.KIE_VARIANT_FORMAT or "png_fast").strip().lower()
    return fmt if fmt in _VALID_FORMATS else "png_fast"


def _encode_variant(
    img: "Image.Image",
    new_size: tuple[int, int],
    fmt: str,
    jpeg_quality: int,
) -> bytes:
    """
    Resize + encode ONE variant. CPU-bound; designed to run in a thread.

    All `png_*` and `webp_lossless` outputs are bit-identical to the input
    pixels (true lossless). `jpeg` is lossy — only use when explicitly opted in.
    """
    # Only resize when shrinking; for the "8k" passthrough this is a no-op.
    if new_size != img.size:
        resized = img.resize(new_size, Image.LANCZOS)
    else:
        resized = img

    buf = io.BytesIO()
    if fmt == "png_fast":
        # LOSSLESS — compress_level=1 is ~15-20x faster than optimize=True and
        # produces identical pixels (PNG compression never touches image data).
        resized.save(buf, format="PNG", optimize=False, compress_level=1)
    elif fmt == "png":
        # LOSSLESS — default PNG compression. Balanced speed/size.
        resized.save(buf, format="PNG", optimize=False, compress_level=6)
    elif fmt == "webp_lossless":
        # LOSSLESS WebP — same pixels, ~60% smaller than PNG, encoded ~3x faster
        # than compress_level=6 PNG. method=4 balances speed and size.
        resized.save(buf, format="WEBP", lossless=True, quality=100, method=4)
    elif fmt == "jpeg":
        # LOSSY — only used if explicitly selected. subsampling=0 preserves
        # full chroma resolution (no 4:2:0 downsampling) which is the main
        # visible JPEG artifact on fashion imagery.
        resized.save(
            buf,
            format="JPEG",
            quality=jpeg_quality,
            optimize=False,
            progressive=True,
            subsampling=0,
        )
    else:
        # Defensive — should never hit because _normalized_format guards it.
        resized.save(buf, format="PNG", optimize=False, compress_level=1)
    return buf.getvalue()


async def _downsample_from_8k_bytes(img_bytes: bytes) -> dict[str, bytes]:
    """
    Downsample one 8K source image into 8K/4K/2K/1K variants, in parallel,
    off the main event loop (threads). Honors KIE_VARIANT_FORMAT.

    Default is `png_fast` — fully lossless, ~15-20x faster encode than the
    legacy `optimize=True` path. Callers must await.
    """
    fmt = _normalized_format()
    jpeg_q = int(settings.KIE_VARIANT_JPEG_QUALITY or 95)

    # Decode once; share the Pillow Image across all four encode threads.
    # Pillow releases the GIL during resize/encode, so threading gives real parallelism.
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size

    sizes = {
        "8k": (w, h),
        "4k": (w // 2, h // 2),
        "2k": (w // 4, h // 4),
        "1k": (w // 8, h // 8),
    }

    results_list = await asyncio.gather(*[
        asyncio.to_thread(_encode_variant, img, sz, fmt, jpeg_q)
        for sz in sizes.values()
    ])
    return dict(zip(sizes.keys(), results_list))


def variant_content_type() -> str:
    fmt = _normalized_format()
    if fmt == "jpeg":
        return "image/jpeg"
    if fmt == "webp_lossless":
        return "image/webp"
    return "image/png"


def variant_extension() -> str:
    fmt = _normalized_format()
    if fmt == "jpeg":
        return "jpg"
    if fmt == "webp_lossless":
        return "webp"
    return "png"

async def _upload_upscaled_outputs(prefix: str, outputs: dict[str, bytes]) -> dict[str, str]:
    label_to_key = {
        "8k": "8k_upscaled",
        "4k": "4k_upscaled",
        "2k": "2k_upscaled",
        "1k": "1k_upscaled",
    }
    ext = variant_extension()
    ctype = variant_content_type()
    upload_tasks = {
        label: upload_bytes_to_r2(data, f"{prefix}_{label}_upscaled.{ext}", ctype)
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


def _callback_url() -> str:
    """Compute the public webhook URL; returns "" when webhooks are disabled."""
    base = (settings.PUBLIC_BASE_URL or "").strip().rstrip("/")
    path = (settings.KIE_WEBHOOK_PATH or "").strip()
    secret = (settings.KIE_WEBHOOK_SECRET or "").strip()
    if not base or not path or not secret:
        return ""
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}"


async def _kie_create_upscale_task(
    *, source_image_url: str, upscale_factor: int, photoshoot_id: str, image_id: str,
) -> str:
    """Submit a topaz/image-upscale job. Returns the KIE taskId."""
    headers = {
        "Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}",
        "Content-Type": "application/json",
    }
    body: dict = {
        "model": settings.KIE_UPSCALE_MODEL,
        "input": {
            "image_url": source_image_url,
            "upscale_factor": str(upscale_factor),
        },
    }
    cb = _callback_url()
    if cb:
        body["callBackUrl"] = cb
        # Round-trip identifiers so the webhook handler can re-associate results.
        body["metadata"] = {
            "photoshoot_id": photoshoot_id,
            "image_id": image_id,
            "purpose": "upscale",
        }

    max_attempts = int(getattr(settings, "KIE_REQUEST_RETRIES", 3) or 3)
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
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
                "[kie-upscale] createTask attempt %d/%d failed for image_id=%s: %s",
                attempt, max_attempts, image_id, exc,
            )
            if attempt < max_attempts:
                await asyncio.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"KIE createTask failed after {max_attempts} attempts: {last_exc}")


async def _kie_poll_once(task_id: str) -> tuple[str, Optional[str]]:
    """
    One-shot poll of KIE recordInfo. Returns (state, result_url_or_None).
    Raises on HTTP error — caller decides whether to keep polling.
    """
    headers = {"Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}"}
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


async def _await_upscale_result(task_id: str, image_id: str) -> str:
    """
    Wait for KIE to finish, preferring webhooks + safety poll.

    Strategy:
      - If PUBLIC_BASE_URL is set: block on Redis BLPOP for up to
        KIE_UPSCALE_SAFETY_POLL_INTERVAL_S seconds. On each timeout, do a
        single recordInfo poll. Total wall-clock capped by KIE_UPSCALE_MAX_WAIT_S.
      - If webhooks disabled: pure poll loop at SEEDDREAM_RETRY_DELAY cadence,
        bounded by SEEDDREAM_MAX_RETRIES.

    Returns the result URL.
    """
    from app.services import kie_task_registry

    has_webhook = bool(_callback_url())
    deadline = time.monotonic() + float(settings.KIE_UPSCALE_MAX_WAIT_S or 900)
    safety_interval = float(settings.KIE_UPSCALE_SAFETY_POLL_INTERVAL_S or 30)

    if has_webhook:
        # Hybrid: BLPOP + periodic safety poll.
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            wait_slot = min(safety_interval, max(1.0, remaining))
            payload = await kie_task_registry.wait_for_async(task_id, wait_slot)
            if payload:
                state = (payload.get("state") or "").lower()
                if state == "success" and payload.get("result_url"):
                    logger.info(
                        "[kie-upscale] result via WEBHOOK task_id=%s image_id=%s",
                        task_id, image_id,
                    )
                    return str(payload["result_url"])
                if state == "fail":
                    raise RuntimeError(
                        f"KIE upscale failed (webhook): {payload.get('raw')}"
                    )
            # BLPOP timed out — safety-check via direct poll (catches dropped callbacks).
            try:
                state, url = await _kie_poll_once(task_id)
                if state == "success" and url:
                    logger.info(
                        "[kie-upscale] result via SAFETY-POLL task_id=%s image_id=%s",
                        task_id, image_id,
                    )
                    return url
                if state == "fail":
                    raise RuntimeError("KIE upscale task failed (safety poll).")
            except Exception as exc:
                logger.warning(
                    "[kie-upscale] safety poll error task_id=%s: %s", task_id, exc,
                )
        raise TimeoutError(
            f"KIE upscale task {task_id} did not finish within {settings.KIE_UPSCALE_MAX_WAIT_S}s"
        )

    # Webhook disabled — legacy poll loop.
    interval = float(settings.SEEDDREAM_RETRY_DELAY or 3)
    max_iters = int(settings.SEEDDREAM_MAX_RETRIES or 40)
    for _ in range(max_iters):
        if time.monotonic() >= deadline:
            break
        try:
            state, url = await _kie_poll_once(task_id)
        except Exception as exc:
            logger.warning("[kie-upscale] poll error task_id=%s: %s", task_id, exc)
            await asyncio.sleep(interval)
            continue
        if state == "success" and url:
            return url
        if state == "fail":
            raise RuntimeError("KIE upscale task failed.")
        await asyncio.sleep(interval)
    raise TimeoutError(f"KIE upscale timed out after {max_iters} polls.")


async def _stream_download(url: str, image_id: str) -> bytes:
    """
    Stream-download the KIE 8K output. Uses stream=True so the bytes are
    assembled once without the intermediate 'response.content' copy.
    """
    max_attempts = int(getattr(settings, "KIE_REQUEST_RETRIES", 3) or 3)
    for attempt in range(1, max_attempts + 1):
        try:
            chunks: list[bytes] = []
            t0 = time.monotonic()
            async with httpx.AsyncClient(timeout=settings.KIE_HTTP_TIMEOUT) as client:
                async with client.stream("GET", url, follow_redirects=True) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 256):
                        if chunk:
                            chunks.append(chunk)
            data = b"".join(chunks)
            logger.info(
                "[kie-upscale] Downloaded 8K output for image_id=%s (%d bytes, %.1fs)",
                image_id, len(data), time.monotonic() - t0,
            )
            return data
        except Exception as exc:
            logger.warning(
                "[kie-upscale] download attempt %d/%d failed for image_id=%s: %s",
                attempt, max_attempts, image_id, exc,
            )
            if attempt >= max_attempts:
                raise
            await asyncio.sleep(min(2 ** attempt, 8))
    raise RuntimeError("unreachable")


async def _run_kie_upscale(
    source_image_url: str, image_id: str, seeddream_url: str, photoshoot_id: str = "",
) -> dict[str, bytes]:
    """
    Submit -> wait (webhook+safety-poll, or pure poll) -> stream download ->
    parallel resize+encode. Returns {8k,4k,2k,1k: bytes}.
    """
    if not source_image_url:
        raise ValueError("source_image_url is required for KIE upscale.")
    if not (settings.SEEDDREAM_API_KEY or "").strip():
        raise ValueError("SEEDDREAM_API_KEY is missing for KIE upscale.")

    from app.services import kie_task_registry

    t_submit = time.monotonic()
    logger.info(
        "[kie-upscale] Submit image_id=%s model=%s factor=%s webhook=%s",
        image_id, settings.KIE_UPSCALE_MODEL, settings.KIE_UPSCALE_FACTOR,
        "on" if _callback_url() else "off",
    )

    task_id = await _kie_create_upscale_task(
        source_image_url=source_image_url,
        upscale_factor=int(settings.KIE_UPSCALE_FACTOR),
        photoshoot_id=photoshoot_id,
        image_id=image_id,
    )

    # Register in Redis so the webhook handler (running in FastAPI process)
    # can push the completion signal back to this worker.
    if _callback_url():
        await asyncio.to_thread(
            kie_task_registry.register_sync, task_id, photoshoot_id, image_id,
        )

    result_url = await _await_upscale_result(task_id, image_id)
    logger.info(
        "[kie-upscale] task_id=%s ready in %.1fs — downloading",
        task_id, time.monotonic() - t_submit,
    )

    bytes_8k = await _stream_download(result_url, image_id)
    return await _downsample_from_8k_bytes(bytes_8k)


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
            # Single retry-aware call — retries now live inside _run_kie_upscale
            # (createTask, poll, download). The old nested "while counter < 3"
            # loop silently mutated the request payload on retry (factor 2 -> 4,
            # url -> seeddream_2k_url) and could spawn duplicate Topaz jobs.
            outputs = await _run_kie_upscale(
                source_image_url=source_image_url,
                image_id=image_id,
                seeddream_url=seeddream_2k_url,
                photoshoot_id=photoshoot_id,
            )
            if outputs and all(outputs.get(k) for k in ("8k", "4k", "2k", "1k")):
                upscaled_urls = await _upload_upscaled_outputs(prefix, outputs)
            else:
                logger.error(
                    "[upscale] KIE returned incomplete outputs for image_id=%s keys=%s",
                    image_id, list(outputs.keys()) if outputs else None,
                )
        else:
            # Modal branch intentionally disabled for now.
            # logger.info("[upscale] Using Modal provider for image_id=%s", image_id)
            # outputs = await _run_modal_upscale(image_bytes=image_bytes, image_id=image_id)
            # upscaled_urls = await _upload_upscaled_outputs(prefix, outputs)
            logger.warning(
                "[upscale] Provider '%s' is disabled in code path. Set WHICH_UPSCALE=kie.",
                provider,
            )
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
        # Keep legacy keys populated with upscaled URLs for downstream consumers.
        "1k": upscaled_urls.get("1k_upscaled", seeddream_1k_url),
        "2k": upscaled_urls.get("2k_upscaled", seeddream_2k_url),
        "4k": upscaled_urls.get("4k_upscaled", seeddream_4k_url),
        "created_at": now,
        "updated_at": now,
    }
    col = upscaling_col if upscaling_col is not None else get_upscaling_collection()
    await col.insert_one(doc)
    logger.info("[upscale] upscaling_data record inserted for image_id=%s", image_id)
    return doc
