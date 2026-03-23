"""
Modal GPU enhancement service.

Provides two entry-points:
  • warmup_modal()        — download weights & load models once at app startup.
                            Must be called from the FastAPI lifespan handler.
  • enhance_and_upload()  — accept raw 4K image bytes, run the GPU pipeline on
                            Modal, upload 8K / 4K / 2K / 1K outputs to S3, persist
                            the result to the `upscaling_data` collection and return
                            a dict of public URLs keyed by resolution label.

All network / I/O is async; the synchronous Modal remote call is offloaded to a
thread-pool so it never blocks the event loop.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from app.database import get_upscaling_collection
from app.services.s3_service import upload_bytes_to_s3

logger = logging.getLogger("modal_enhance")

# ---------------------------------------------------------------------------
# Module-level Modal handle (resolved once after import)
# ---------------------------------------------------------------------------

_modal_cls = None          # FashionRealismT4 or FashionRealismL4 class reference
_modal_instance = None     # instantiated Modal cls object reused across calls
_modal_available = False


def _import_modal():
    """
    Attempt to import Modal and resolve the FashionRealismT4 class from the
    pipeline script. Sets module-level _modal_cls / _modal_available.
    Called lazily so the rest of the app starts fine if Modal is absent.
    """
    global _modal_cls, _modal_available
    try:
        import importlib.util, sys
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "modal_realism_pipeline",
            Path(__file__).resolve().parent.parent.parent / "modal_realism_pipeline.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        _modal_cls = getattr(mod, "FashionRealismT4", None)
        if _modal_cls is None:
            logger.warning("[modal] FashionRealismT4 not found in pipeline script — Modal disabled.")
            return
        _modal_available = True
        logger.info("[modal] Modal pipeline module loaded successfully.")
    except Exception as exc:
        logger.warning("[modal] Modal pipeline load failed (%s: %s) — GPU enhancement disabled.", type(exc).__name__, exc)


# ---------------------------------------------------------------------------
# Warmup (called at startup)
# ---------------------------------------------------------------------------

async def warmup_modal() -> None:
    """
    Trigger Modal container warm-up so that weights are downloaded and models
    are loaded before the first real request arrives.

    This fires a lightweight no-op `enhance` call on a 1×1 white PNG so Modal
    spins up the container and caches the weights volume in the background.
    The call is intentionally not awaited for completion — it runs in a
    background thread so startup is non-blocking.
    """
    _import_modal()
    if not _modal_available:
        logger.warning("[modal] Skipping warmup — Modal not available.")
        return

    asyncio.get_event_loop().run_in_executor(None, _do_warmup_sync)
    logger.info("[modal] Modal warmup triggered in background thread.")


def _do_warmup_sync() -> None:
    """Synchronous warmup executed in a thread pool at startup."""
    try:
        import cv2
        import numpy as np

        # 1×1 white PNG as a minimal valid payload
        dummy = np.ones((1, 1, 3), dtype=np.uint8) * 255
        ok, enc = cv2.imencode(".png", dummy)
        if not ok:
            logger.warning("[modal] Could not encode dummy warmup image.")
            return

        global _modal_instance
        _modal_instance = _modal_cls()
        _modal_instance.enhance.remote(enc.tobytes(), "warmup.png")
        logger.info("[modal] Modal warmup call completed — container is warm.")
    except Exception as exc:
        logger.warning("[modal] Modal warmup call failed (%s: %s) — will retry on first real request.", type(exc).__name__, exc)


# ---------------------------------------------------------------------------
# Core enhancement + upload
# ---------------------------------------------------------------------------

async def enhance_and_upload(
    image_bytes: bytes,
    photoshoot_id: str,
    image_id: str,
    seeddream_4k_url: str,
    seeddream_2k_url: str,
    seeddream_1k_url: str,
) -> dict:
    """
    Run the Modal GPU pipeline on `image_bytes` (expected to be the 4K SeedDream
    output), upload all four resolution outputs to S3, store the combined record
    in `upscaling_data`, and return a dict:

        {
            "8k_upscaled":  "<url>",
            "4k_upscaled":  "<url>",
            "2k_upscaled":  "<url>",
            "1k_upscaled":  "<url>",
            "4k":           "<seeddream_4k_url>",
            "2k":           "<seeddream_2k_url>",
            "1k":           "<seeddream_1k_url>",
        }

    If Modal is unavailable or fails, falls back gracefully — upscaled URLs are
    set to empty strings and the function still records the SeedDream URLs.
    """
    if not _modal_available:
        _import_modal()

    upscaled_bytes: dict[str, bytes] = {}

    if _modal_available:
        try:
            upscaled_bytes = await asyncio.get_event_loop().run_in_executor(
                None,
                _call_enhance_sync,
                image_bytes,
                f"{photoshoot_id}_{image_id}.png",
            )
            logger.info("[modal] Enhancement complete for image_id=%s — resolutions: %s",
                        image_id, list(upscaled_bytes.keys()))
        except Exception as exc:
            logger.error("[modal] Enhancement failed for image_id=%s: %s — skipping upscale.", image_id, exc)

    # Upload enhanced outputs concurrently
    prefix = f"photoshoots/{photoshoot_id}/{image_id}/upscaled"
    upload_tasks = {}
    for label in ("8k", "4k", "2k", "1k"):
        if label in upscaled_bytes:
            key = f"{prefix}_{label}.png"
            upload_tasks[label] = upload_bytes_to_s3(upscaled_bytes[label], key, "image/png")

    upscaled_urls: dict[str, str] = {label: "" for label in ("8k", "4k", "2k", "1k")}
    if upload_tasks:
        labels_ordered = list(upload_tasks.keys())
        results = await asyncio.gather(*upload_tasks.values(), return_exceptions=True)
        for label, result in zip(labels_ordered, results):
            if isinstance(result, Exception):
                logger.error("[modal] S3 upload failed for %s/%s: %s", image_id, label, result)
            else:
                upscaled_urls[label] = result

    now = datetime.now(timezone.utc)
    record = {
        "upscaling_id":  str(uuid.uuid4()),
        "photoshoot_id": photoshoot_id,
        "image_id":      image_id,
        "8k_upscaled":   upscaled_urls.get("8k", ""),
        "4k_upscaled":   upscaled_urls.get("4k", ""),
        "2k_upscaled":   upscaled_urls.get("2k", ""),
        "1k_upscaled":   upscaled_urls.get("1k", ""),
        "4k":            seeddream_4k_url,
        "2k":            seeddream_2k_url,
        "1k":            seeddream_1k_url,
        "created_at":    now,
        "updated_at":    now,
    }

    try:
        col = get_upscaling_collection()
        await col.insert_one(record)
        logger.info("[modal] upscaling_data record saved for image_id=%s", image_id)
    except Exception as exc:
        logger.error("[modal] Failed to save upscaling_data for image_id=%s: %s", image_id, exc)

    return {
        "8k_upscaled": record["8k_upscaled"],
        "4k_upscaled": record["4k_upscaled"],
        "2k_upscaled": record["2k_upscaled"],
        "1k_upscaled": record["1k_upscaled"],
        "4k":          seeddream_4k_url,
        "2k":          seeddream_2k_url,
        "1k":          seeddream_1k_url,
    }


def _call_enhance_sync(image_bytes: bytes, filename: str) -> dict:
    """Synchronous Modal remote call; run inside a thread pool."""
    global _modal_instance
    try:
        if _modal_instance is None:
            _modal_instance = _modal_cls()
        return _modal_instance.enhance.remote(image_bytes, filename)
    except Exception as t4_err:
        logger.warning("[modal] T4 enhance failed (%s) — trying L4 fallback.", t4_err)

    # Fallback: import L4 class and retry once
    try:
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "modal_realism_pipeline",
            Path(__file__).resolve().parent.parent.parent / "modal_realism_pipeline.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        l4_cls = getattr(mod, "FashionRealismL4", None)
        if l4_cls is not None:
            return l4_cls().enhance.remote(image_bytes, filename)
    except Exception as l4_err:
        raise RuntimeError(f"Both T4 and L4 enhance attempts failed: {l4_err}") from l4_err
    raise RuntimeError("Modal enhance failed — L4 class not found.")
