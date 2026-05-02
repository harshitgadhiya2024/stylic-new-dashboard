"""
Modal T4 worker: decode Topaz 8K output → 8K / 4K / 2K / 1K variants (GPU bicubic downscale + encode).

Deploy:
  modal deploy modal_kie_downsample.py

Must match keys/encoding semantics in ``app/services/modal_enhance_service.py`` (KIE_VARIANT_FORMAT).
"""

from __future__ import annotations

import io
from typing import Any

import modal

app = modal.App("stylic-kie-downsample")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install("pillow", "numpy")
)

_VALID_FORMATS = frozenset({"png_fast", "png", "webp_lossless", "jpeg"})


def _normalized_format(fmt: str) -> str:
    f = (fmt or "png_fast").strip().lower()
    return f if f in _VALID_FORMATS else "png_fast"


def _encode_variant(
    img: Any,
    new_size: tuple[int, int],
    fmt: str,
    jpeg_quality: int,
) -> bytes:
    """Resize + encode one variant (Pillow). Mirrors modal_enhance_service._encode_variant."""
    from PIL import Image

    if new_size != img.size:
        resized = img.resize(new_size, Image.LANCZOS)
    else:
        resized = img

    buf = io.BytesIO()
    if fmt == "png_fast":
        resized.save(buf, format="PNG", optimize=False, compress_level=1)
    elif fmt == "png":
        resized.save(buf, format="PNG", optimize=False, compress_level=6)
    elif fmt == "webp_lossless":
        resized.save(buf, format="WEBP", lossless=True, quality=100, method=4)
    elif fmt == "jpeg":
        resized.save(
            buf,
            format="JPEG",
            quality=jpeg_quality,
            optimize=False,
            progressive=True,
            subsampling=0,
        )
    else:
        resized.save(buf, format="PNG", optimize=False, compress_level=1)
    return buf.getvalue()


def _tensor_to_pil(t_nchw: Any) -> Any:
    """Float NCHW [0,1] on CPU/GPU → RGB PIL Image."""
    import numpy as np
    from PIL import Image
    import torch

    x = t_nchw.squeeze(0).permute(1, 2, 0).clamp(0, 1)
    arr = (x * 255.0).byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


@app.cls(
    image=image,
    gpu="T4",
    timeout=2000,
    memory=16384,
    max_containers=20,
    allow_concurrent_inputs=10,
)
@modal.concurrent(max_inputs=10)
class KieDownsampleT4:
    """
    Bicubic downscale on CUDA (torch), encode with Pillow — same size ladder as the API server.
    """

    @modal.method()
    def run(self, image_bytes: bytes, variant_format: str, jpeg_quality: int) -> dict[str, bytes]:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image

        fmt = _normalized_format(variant_format)
        jq = int(jpeg_quality or 95)

        pil_full = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = pil_full.size

        device = torch.device("cuda")
        arr = np.asarray(pil_full).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

        sizes_hw = {
            "8k": (h, w),
            "4k": (max(1, h // 2), max(1, w // 2)),
            "2k": (max(1, h // 4), max(1, w // 4)),
            "1k": (max(1, h // 8), max(1, w // 8)),
        }

        out: dict[str, bytes] = {}
        for label, (nh, nw) in sizes_hw.items():
            if label == "8k":
                small = pil_full
            else:
                tt = F.interpolate(t, size=(nh, nw), mode="bicubic", align_corners=False)
                small = _tensor_to_pil(tt)
            out[label] = _encode_variant(small, (nw, nh), fmt, jq)

        return out
