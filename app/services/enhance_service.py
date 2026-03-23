"""
Enhancement service — wraps the Modal FashionRealismT4/L4 pipeline.

Responsibilities:
  - warmup(): called once at app startup to pre-load model weights into the
    Modal Volume and warm the GPU container (no image input needed).
  - enhance_image(): called per pose after SeedDream generation.  Sends the
    raw 4K bytes to the Modal GPU, receives back upscaled PNG bytes keyed by
    resolution label ("8k", "4k", "2k", "1k"), and returns them.

The Modal classes (FashionRealismT4 / FashionRealismL4) are imported lazily
so the FastAPI process never crashes if Modal is not installed.
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger("enhance_service")

# ---------------------------------------------------------------------------
# Lazy Modal import helpers
# ---------------------------------------------------------------------------

def _get_modal_classes():
    """
    Import enhance_pipeline and return (FashionRealismT4, FashionRealismL4).
    Returns (None, None) if Modal is not installed or the pipeline file is missing.
    """
    pipeline_path = Path(__file__).resolve().parent.parent.parent / "enhance_pipeline.py"
    if not pipeline_path.exists():
        logger.warning("[enhance] enhance_pipeline.py not found — enhancement disabled")
        return None, None

    try:
        import importlib.util
        spec   = importlib.util.spec_from_file_location("enhance_pipeline", pipeline_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        T4 = getattr(module, "FashionRealismT4", None)
        L4 = getattr(module, "FashionRealismL4", None)
        if T4 is None or L4 is None:
            logger.warning("[enhance] Modal classes not found in enhance_pipeline.py — Modal may not be installed")
            return None, None
        return T4, L4
    except Exception as exc:
        logger.warning("[enhance] Could not load enhance_pipeline.py: %s — enhancement disabled", exc)
        return None, None


# ---------------------------------------------------------------------------
# Startup warmup — downloads weights + warms GPU container, NO image input
# ---------------------------------------------------------------------------

def warmup_enhance_pipeline() -> None:
    """
    Call once at application startup (in a thread executor so the async event
    loop is not blocked).  Triggers Modal container startup + weight download
    into the persistent Volume without processing any image.
    """
    logger.info("[enhance] Warming up Modal enhancement pipeline...")
    T4, L4 = _get_modal_classes()
    if T4 is None:
        logger.warning("[enhance] Modal not available — warmup skipped")
        return

    # Send a tiny 1×1 white pixel just to trigger container startup and weight
    # download without wasting GPU time on real inference.
    try:
        import cv2
        import numpy as np
        tiny = np.full((1, 1, 3), 255, dtype=np.uint8)
        _, enc = cv2.imencode(".png", tiny)
        dummy_bytes = enc.tobytes()
    except Exception:
        # If cv2 is not available locally just use a minimal PNG header.
        # The Modal container will decode it and skip (or error silently).
        import struct, zlib
        def _png1x1():
            sig  = b'\x89PNG\r\n\x1a\n'
            ihdr = b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
            idat = b'\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x18\xdd\x8d\x64'
            iend = b'\x00\x00\x00\x00IEND\xaeB`\x82'
            return sig + ihdr + idat + iend
        dummy_bytes = _png1x1()

    try:
        T4().enhance.remote(dummy_bytes, "warmup.png")
        logger.info("[enhance] Modal T4 container warmed up successfully")
    except Exception as t4_err:
        logger.warning("[enhance] T4 warmup failed (%s) — trying L4...", t4_err)
        try:
            L4().enhance.remote(dummy_bytes, "warmup.png")
            logger.info("[enhance] Modal L4 container warmed up successfully")
        except Exception as l4_err:
            logger.warning("[enhance] L4 warmup also failed (%s) — containers will cold-start on first request", l4_err)


# ---------------------------------------------------------------------------
# Per-image enhancement
# ---------------------------------------------------------------------------

async def enhance_image(image_bytes: bytes, filename: str = "image.png") -> dict[str, bytes] | None:
    """
    Send image_bytes to the Modal GPU pipeline and return a dict:
        {"8k": <bytes>, "4k": <bytes>, "2k": <bytes>, "1k": <bytes>}

    Returns None if Modal is not available or enhancement fails, so the caller
    can fall back gracefully (store original sizes only).
    """
    import asyncio
    T4, L4 = _get_modal_classes()
    if T4 is None:
        logger.warning("[enhance] Modal not available — skipping enhancement for %s", filename)
        return None

    loop = asyncio.get_event_loop()

    def _call_modal():
        try:
            logger.info("[enhance] Sending %s to Modal T4...", filename)
            result = T4().enhance.remote(image_bytes, filename)
            logger.info("[enhance] T4 enhancement complete for %s", filename)
            return result
        except Exception as t4_err:
            logger.warning("[enhance] T4 failed for %s (%s) — trying L4...", filename, t4_err)
            logger.info("[enhance] Sending %s to Modal L4...", filename)
            result = L4().enhance.remote(image_bytes, filename)
            logger.info("[enhance] L4 enhancement complete for %s", filename)
            return result

    try:
        outputs = await loop.run_in_executor(None, _call_modal)
        # Normalise keys to lowercase ("1k", "2k", "4k", "8k")
        return {k.lower(): v for k, v in outputs.items()}
    except Exception as exc:
        logger.error("[enhance] Enhancement failed for %s: %s — returning None", filename, exc)
        return None
