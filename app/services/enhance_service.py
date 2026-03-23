"""
Enhancement service — calls the deployed Modal FashionRealismT4/L4 pipeline
remotely using modal.Cls.from_name().

The enhance_pipeline.py file is NEVER imported here. All heavy ML dependencies
(torch, cv2, timm, etc.) live only inside the Modal container image and are
never needed on the EC2 / local server process.

At startup:  warmup_enhance_pipeline() triggers container startup + weight
             download into the Modal Volume (no image input).
Per-pose:    enhance_image() sends raw 4K bytes → Modal GPU → returns
             {"8k": bytes, "4k": bytes, "2k": bytes, "1k": bytes}.
"""

import asyncio
import logging

logger = logging.getLogger("enhance_service")

# Modal app name and class names as defined in enhance_pipeline.py
_MODAL_APP_NAME   = "fashion-realism"
_T4_CLASS_NAME    = "FashionRealismT4"
_L4_CLASS_NAME    = "FashionRealismL4"


def _get_modal_cls(class_name: str):
    """
    Look up a deployed Modal class by app + class name.
    Returns the Modal Cls handle, or None if Modal is not installed / class
    not found (so the server never crashes when Modal is unavailable).
    """
    try:
        import modal
        return modal.Cls.from_name(_MODAL_APP_NAME, class_name)
    except Exception as exc:
        logger.warning("[enhance] Could not get Modal class '%s': %s", class_name, exc)
        return None


# ---------------------------------------------------------------------------
# Startup warmup — triggers container start + weight download, no image input
# ---------------------------------------------------------------------------

def warmup_enhance_pipeline() -> None:
    """
    Call once at application startup via loop.run_in_executor so the async
    event loop is not blocked. Sends a tiny dummy image to trigger the Modal
    container to start and download weights into the persistent Volume.
    No real image processing is done.
    """
    logger.info("[enhance] Warming up Modal enhancement pipeline...")

    T4 = _get_modal_cls(_T4_CLASS_NAME)
    if T4 is None:
        logger.warning("[enhance] Modal not available — warmup skipped")
        return

    # Minimal 1×1 white PNG — just enough to wake the container.
    dummy_bytes = _make_1x1_png()

    try:
        T4().enhance.remote(dummy_bytes, "warmup.png")
        logger.info("[enhance] Modal T4 container warmed up successfully")
    except Exception as t4_err:
        logger.warning("[enhance] T4 warmup failed (%s) — trying L4...", t4_err)
        L4 = _get_modal_cls(_L4_CLASS_NAME)
        if L4 is None:
            return
        try:
            L4().enhance.remote(dummy_bytes, "warmup.png")
            logger.info("[enhance] Modal L4 container warmed up successfully")
        except Exception as l4_err:
            logger.warning(
                "[enhance] L4 warmup also failed (%s) — containers will cold-start on first request",
                l4_err,
            )


# ---------------------------------------------------------------------------
# Per-image enhancement
# ---------------------------------------------------------------------------

async def enhance_image(image_bytes: bytes, filename: str = "image.png") -> dict | None:
    """
    Send image_bytes to the Modal GPU pipeline and return:
        {"8k": <bytes>, "4k": <bytes>, "2k": <bytes>, "1k": <bytes>}

    Returns None on any failure so the caller can fall back gracefully.
    """
    loop = asyncio.get_event_loop()

    def _call_modal():
        T4 = _get_modal_cls(_T4_CLASS_NAME)
        if T4 is None:
            logger.warning("[enhance] Modal not available — skipping enhancement for %s", filename)
            return None
        try:
            logger.info("[enhance] Sending %s to Modal T4...", filename)
            result = T4().enhance.remote(image_bytes, filename)
            logger.info("[enhance] T4 enhancement complete for %s", filename)
            return result
        except Exception as t4_err:
            logger.warning("[enhance] T4 failed for %s (%s) — trying L4...", filename, t4_err)
            L4 = _get_modal_cls(_L4_CLASS_NAME)
            if L4 is None:
                return None
            try:
                logger.info("[enhance] Sending %s to Modal L4...", filename)
                result = L4().enhance.remote(image_bytes, filename)
                logger.info("[enhance] L4 enhancement complete for %s", filename)
                return result
            except Exception as l4_err:
                logger.error("[enhance] Both T4 and L4 failed for %s: %s", filename, l4_err)
                return None

    try:
        outputs = await loop.run_in_executor(None, _call_modal)
        if outputs is None:
            return None
        # Normalise keys to lowercase ("1k", "2k", "4k", "8k")
        return {k.lower(): v for k, v in outputs.items()}
    except Exception as exc:
        logger.error("[enhance] Enhancement failed for %s: %s — returning None", filename, exc)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_1x1_png() -> bytes:
    """Return a minimal valid 1×1 white PNG as bytes without any dependencies."""
    import struct, zlib

    def chunk(name: bytes, data: bytes) -> bytes:
        c = name + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw  = b"\x00\xff\xff\xff"          # filter byte + RGB white pixel
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return signature + ihdr + idat + iend
