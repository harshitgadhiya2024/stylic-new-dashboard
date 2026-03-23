"""
Modal enhancement service.

Calls the deployed Modal GPU pipeline (FashionRealismT4 / FashionRealismL4) from
modal_realism_pipeline.py, uploads the returned 8K / 4K / 2K / 1K PNG bytes to S3,
inserts a record in the upscaling_data collection, and returns the URL map.

The modal_realism_pipeline.py is used AS-IS — no changes to that file.
We only swap the input image bytes before sending to Modal.
"""

import asyncio
import logging
from datetime import datetime, timezone

from app.database import get_upscaling_collection
from app.services.s3_service import upload_bytes_to_s3

logger = logging.getLogger("modal_enhance")


async def warmup_modal() -> None:
    """
    Trigger a lightweight warmup call so the Modal container starts loading
    weights in the background.  Failures are silently swallowed — the app must
    boot even when Modal is unavailable.
    """
    try:
        import modal  # noqa: F401 — just verify the package is present
        logger.info("[modal] Modal package found — warmup skipped (weights load on first real call).")
    except ImportError:
        logger.warning("[modal] Modal package not installed — enhancement will be skipped at runtime.")


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

        # Use Modal's from_name API to call the already-deployed app remotely.
        # This avoids importing modal_realism_pipeline.py as a local module
        # (which would trigger sys.exit(1) when local GPU deps like timm are absent).
        import modal  # noqa: E402

        filename = f"{image_id}.png"

        def _call_modal() -> dict:
            """Blocking Modal .enhance.remote() call — runs in a thread executor."""
            try:
                logger.info("[modal] Trying T4 GPU via Modal from_name ...")
                cls_t4 = modal.Cls.from_name("fashion-realism", "FashionRealismT4")
                outputs = cls_t4().enhance.remote(image_bytes, filename)
                logger.info("[modal] T4 enhancement succeeded")
                return outputs
            except Exception as t4_err:
                logger.warning("[modal] T4 failed (%s: %s) — falling back to L4", type(t4_err).__name__, t4_err)
                cls_l4 = modal.Cls.from_name("fashion-realism", "FashionRealismL4")
                outputs = cls_l4().enhance.remote(image_bytes, filename)
                logger.info("[modal] L4 enhancement succeeded")
                return outputs

        loop    = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(None, _call_modal)

        # outputs = {"8k": bytes, "4k": bytes, "2k": bytes, "1k": bytes}
        label_to_key = {"8k": "8k_upscaled", "4k": "4k_upscaled", "2k": "2k_upscaled", "1k": "1k_upscaled"}
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
