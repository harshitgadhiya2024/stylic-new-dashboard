"""
Free-plan photoshoot tiled watermark — same algorithm as ``testing.py``:
Poppins Bold from npm tarball, one rotated stamp, tiled on a fixed grid.

Corner logo from ``testing.py`` is intentionally omitted (it was commented out there).
"""
from __future__ import annotations

import io
import logging
import os
import tarfile
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
from urllib.request import Request, urlopen

from app.config import settings

logger = logging.getLogger(__name__)

# Font source: @expo-google-fonts/poppins on npm — matches ``testing.py``.
_NPM_POPPINS_BOLD: tuple[str, str] = (
    "https://registry.npmjs.org/@expo-google-fonts/poppins/-/poppins-0.2.3.tgz",
    "package/Poppins_700Bold.ttf",
)

_poppins_fetch_lock = threading.Lock()


def _watermark_fonts_dir() -> Path:
    raw = (getattr(settings, "FREE_PLAN_WATERMARK_FONT_CACHE_DIR", "") or "").strip()
    if raw:
        p = Path(os.path.expanduser(raw))
    else:
        p = Path.home() / ".cache" / "stylicai" / "watermark_fonts"
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "stylicai_watermark_fonts"
        fallback.mkdir(parents=True, exist_ok=True)
        logger.warning("[watermark] using temp font cache (could not mkdir %s): %s", p, fallback)
        return fallback


def _ensure_poppins_bold_ttf() -> Optional[Path]:
    """Return path to Poppins-Bold.ttf, downloading the npm package once if needed."""
    dest = _watermark_fonts_dir() / "Poppins-Bold.ttf"
    if dest.is_file() and dest.stat().st_size >= 10_000:
        return dest

    pkg_url, member_path = _NPM_POPPINS_BOLD
    timeout_s = float(getattr(settings, "FREE_PLAN_WATERMARK_FONT_DOWNLOAD_TIMEOUT_S", 45.0) or 45.0)

    with _poppins_fetch_lock:
        if dest.is_file() and dest.stat().st_size >= 10_000:
            return dest
        tmp = dest.parent / f"{dest.name}.{uuid.uuid4().hex}.part"
        try:
            logger.info("[watermark] Downloading Poppins Bold from npm → %s", dest)
            req = Request(pkg_url, headers={"User-Agent": "StylicAI-free-plan-watermark/1.0"})
            with urlopen(req, timeout=timeout_s) as resp:
                tgz_data = resp.read()
            with tarfile.open(fileobj=io.BytesIO(tgz_data), mode="r:gz") as tar:
                member = tar.getmember(member_path)
                bio = tar.extractfile(member)
                if bio is None:
                    raise RuntimeError(f"could not extract {member_path!r} from npm tarball")
                data = bio.read()
            if len(data) < 10_000:
                raise RuntimeError(f"extracted font too small ({len(data)} bytes)")
            tmp.write_bytes(data)
            os.replace(tmp, dest)
        except Exception as exc:
            logger.error("[watermark] Poppins download failed: %s", exc, exc_info=True)
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            return dest if dest.is_file() else None

    return dest if dest.is_file() else None


def _tile_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    explicit = (getattr(settings, "FREE_PLAN_WATERMARK_TILE_FONT_PATH", "") or "").strip()
    if explicit:
        p = Path(os.path.expanduser(explicit))
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception as exc:
                logger.warning("[watermark] FREE_PLAN_WATERMARK_TILE_FONT_PATH load failed (%s): %s", p, exc)
    path = _ensure_poppins_bold_ttf()
    if path is not None:
        try:
            return ImageFont.truetype(str(path), size)
        except Exception as exc:
            logger.warning("[watermark] Poppins truetype failed (%s): %s", path, exc)
    try:
        return ImageFont.load_default(size=max(10, min(size, 96)))
    except TypeError:
        return ImageFont.load_default()


def build_tile_layer(width: int, height: int) -> Image.Image:
    """Full-image tiled diagonal semi-transparent text layer (``testing.py`` logic)."""
    tile_text = (settings.FREE_PLAN_WATERMARK_TEXT or "Stylic").strip() or "Stylic"
    opacity = max(0, min(255, int(getattr(settings, "FREE_PLAN_WATERMARK_OPACITY", 95) or 95)))
    rotation = float(getattr(settings, "FREE_PLAN_WATERMARK_ROTATION", -30.0) or -30.0)
    font_size = max(10, int(getattr(settings, "FREE_PLAN_WATERMARK_TILE_FONT_SIZE", 52) or 52))
    spacing_x = max(40, int(getattr(settings, "FREE_PLAN_WATERMARK_TILE_SPACING_X", 280) or 280))
    spacing_y = max(40, int(getattr(settings, "FREE_PLAN_WATERMARK_TILE_SPACING_Y", 200) or 200))

    tr = max(0, min(255, int(getattr(settings, "FREE_PLAN_WATERMARK_TILE_R", 255) or 255)))
    tg = max(0, min(255, int(getattr(settings, "FREE_PLAN_WATERMARK_TILE_G", 255) or 255)))
    tb = max(0, min(255, int(getattr(settings, "FREE_PLAN_WATERMARK_TILE_B", 255) or 255)))

    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    font = _tile_font(font_size)

    tmp = ImageDraw.Draw(layer)
    bb = tmp.textbbox((0, 0), tile_text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad = max(tw, th) + 40

    stamp_img = Image.new("RGBA", (tw + pad, th + pad), (0, 0, 0, 0))
    ImageDraw.Draw(stamp_img).text(
        (pad // 2, pad // 2),
        tile_text,
        font=font,
        fill=(tr, tg, tb, opacity),
    )
    stamp = stamp_img.rotate(rotation, expand=True)
    sw, sh = stamp.size

    cols = (width // spacing_x) + 5
    rows = (height // spacing_y) + 5

    for row in range(-2, rows):
        for col in range(-2, cols):
            x = col * spacing_x + (row % 2) * (spacing_x // 2)
            y = row * spacing_y
            layer.paste(stamp, (x - sw // 2, y - sh // 2), stamp)

    return layer


def apply_tile_watermark_png_bytes(original_bytes: bytes) -> bytes:
    """Decode PNG/JPEG bytes, composite tiled watermark, re-encode as PNG."""
    base = Image.open(io.BytesIO(original_bytes)).convert("RGBA")
    w, h = base.size
    combined = Image.alpha_composite(base, build_tile_layer(w, h))
    buf = io.BytesIO()
    combined.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()
