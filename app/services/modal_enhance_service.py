"""
Modal enhancement service.

Calls the deployed Modal GPU pipeline (FashionRealismL40S / FashionRealismA100)
from modal_realism_pipeline.py, uploads the returned 8K / 4K / 2K / 1K PNG bytes
to S3, inserts a record in the upscaling_data collection, and returns the URL map.

GPU configuration (set in app/config.py / .env):
    MODAL_APP_NAME     = "fashion-realism"
    MODAL_CLS_PRIMARY  = "FashionRealismL40S"   ← L40S, 48 GB VRAM (primary)
    MODAL_CLS_FALLBACK = "FashionRealismA100"   ← A100-40GB, 40 GB VRAM (fallback)

Modal is called via .remote.aio() (native async) so multiple poses dispatched
via asyncio.gather each get their own auto-scaled GPU container.
"""

import asyncio
import logging
from datetime import datetime, timezone

from app.config import settings
from app.database import get_upscaling_collection
from app.services.s3_service import upload_bytes_to_s3

logger = logging.getLogger("modal_enhance")


async def warmup_modal() -> None:
    """
    Verify the Modal package is present at startup.
    Failures are silently swallowed — the app must boot even when Modal is unavailable.
    """
    try:
        import modal  # noqa: F401
        logger.info(
            "[modal] Modal package found — primary=%s fallback=%s app=%s",
            settings.MODAL_CLS_PRIMARY,
            settings.MODAL_CLS_FALLBACK,
            settings.MODAL_APP_NAME,
        )
    except ImportError:
        logger.warning("[modal] Modal package not installed — GPU enhancement will be skipped at runtime.")


async def enhance_and_upload(
    image_bytes: bytes,
    photoshoot_id: str,
    image_id: str,
    seeddream_4k_url: str,
    seeddream_2k_url: str,
    seeddream_1k_url: str,
) -> dict:
    """
    Send ``image_bytes`` (SeedDream 4K output) to the Modal GPU pipeline,
    upload the resulting resolutions to S3, save a record in upscaling_data,
    and return a dict with all URL fields.

    Return schema (mirrors the upscaling_data document fields):
        {
          "1k_upscaled": str | "",
          "2k_upscaled": str | "",
          "4k_upscaled": str | "",
          "8k_upscaled": str | "",
          "1k": str,   # SeedDream original
          "2k": str,
          "4k": str,
        }

    On any Modal failure the upscaled_* fields are left empty and the
    SeedDream originals are returned so the photoshoot still completes.
    """
    prefix = f"photoshoots/{photoshoot_id}/{image_id}"
    now    = datetime.now(timezone.utc)

    upscaled_urls: dict[str, str] = {}

    try:
        logger.info("[modal] Calling Modal GPU pipeline for image_id=%s ...", image_id)

        import modal  # noqa: E402

        filename    = f"{image_id}.png"
        app_name    = settings.MODAL_APP_NAME
        cls_primary = settings.MODAL_CLS_PRIMARY
        cls_fallback = settings.MODAL_CLS_FALLBACK

        async def _call_modal_async() -> dict:
            """Try primary GPU class first; fall back to secondary on any error."""
            try:
                logger.info(
                    "[modal] Trying %s (%s) via Modal from_name (async) ...",
                    cls_primary, app_name,
                )
                cls = modal.Cls.from_name(app_name, cls_primary)
                outputs = await cls().enhance.remote.aio(image_bytes, filename)
                logger.info("[modal] %s enhancement succeeded for image_id=%s", cls_primary, image_id)
                return outputs
            except Exception as primary_err:
                logger.warning(
                    "[modal] %s failed for image_id=%s (%s: %s) — falling back to %s",
                    cls_primary, image_id, type(primary_err).__name__, primary_err, cls_fallback,
                )
                cls = modal.Cls.from_name(app_name, cls_fallback)
                outputs = await cls().enhance.remote.aio(image_bytes, filename)
                logger.info("[modal] %s enhancement succeeded for image_id=%s", cls_fallback, image_id)
                return outputs

        outputs = await _call_modal_async()

        # outputs = {"8k": bytes, "4k": bytes, "2k": bytes, "1k": bytes}
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

        if upload_tasks:
            labels  = list(upload_tasks.keys())
            results = await asyncio.gather(*upload_tasks.values())
            for label, url in zip(labels, results):
                upscaled_urls[label_to_key[label]] = url
                logger.info("[modal] Uploaded %s → %s", label_to_key[label], url[:80])

    except ImportError:
        logger.warning("[modal] modal package not installed — skipping GPU enhancement")
    except Exception as exc:
        logger.error("[modal] Enhancement failed for image_id=%s: %s", image_id, exc)

    # Build the full upscaling_data document
    doc = {
        "photoshoot_id": photoshoot_id,
        "image_id":      image_id,
        # Upscaled outputs (empty string when Modal was not called / failed)
        "1k_upscaled":   upscaled_urls.get("1k_upscaled", ""),
        "2k_upscaled":   upscaled_urls.get("2k_upscaled", ""),
        "4k_upscaled":   upscaled_urls.get("4k_upscaled", ""),
        "8k_upscaled":   upscaled_urls.get("8k_upscaled", ""),
        # SeedDream originals
        "1k":            seeddream_1k_url,
        "2k":            seeddream_2k_url,
        "4k":            seeddream_4k_url,
        "created_at":    now,
        "updated_at":    now,
    }

    col = get_upscaling_collection()
    await col.insert_one(doc)
    logger.info("[modal] upscaling_data record inserted for image_id=%s", image_id)

    return doc
