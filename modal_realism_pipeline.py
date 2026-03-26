#!/usr/bin/env python3
"""
Fashion Realism Enhancement Pipeline — L40S / A100-40GB GPU (Modal) Edition
============================================================================
High-fidelity multi-model pipeline for fashion photoshoot images.

Stage 1  │ SwinIR-L x4          │ Base super-resolution (x4 upscale)
Stage 2  │ HAT (tiled)          │ Fabric & pattern detail refinement
Stage 3  │ RestoreFormer++ / GFPGAN │ Face & skin photorealism
Stage 4  │ Post-processing      │ Bilateral smooth → CLAHE → deband

Run on Modal:
    modal run modal_realism_pipeline.py

Run locally (needs CUDA GPU):
    python modal_realism_pipeline.py
"""

import os
import sys
import math
import time
import warnings
import requests
import urllib.request
from pathlib import Path
from tqdm import tqdm

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

try:
    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    pass

# ═══════════════════════════════════════════════════════════════
#  CONFIG  ← only edit this block
# ═══════════════════════════════════════════════════════════════

INPUT_IMAGE = "input1_4k.png"   # ← path to your input image

# ── S3 Upload ───────────────────────────────────────────────────
# Images are uploaded to S3 directly from Modal and public URLs are returned.
AWS_REGION        = "eu-north-1"
AWS_S3_BUCKET     = "aavishailabs-uploads-prod"
S3_KEY_PREFIX     = "fashion-realism"   # folder path inside the bucket

# ── Outputs ────────────────────────────────────────────────────
SAVE_8K = True   # useful when SR_UPSCALE_MODE=2 on 4K input
SAVE_1K = True
SAVE_2K = True
SAVE_4K = True

# ── Stage 1: SwinIR base upscaler ──────────────────────────────
# Tile size: L40S (48 GB) handles 1536 comfortably; A100-40GB handles 1280.
# Larger tiles = fewer tile boundaries = sharper output + significantly fewer passes = faster.
SWINIR_TILE         = 1536   # was 1024 — L40S/A100 have ample VRAM for 1536
SWINIR_TILE_OVERLAP = 96     # was 128 — sufficient at 1536; saves ~25% overlap waste
# SR scale mode for SwinIR/HAT pipeline:
# "auto" = 1x for >=4K input, 2x for mid-res, 4x for small input
# 1, 2, 4 are also allowed.
SR_UPSCALE_MODE = 2

# ── Stage 2: HAT fabric/pattern refinement ─────────────────────
# HAT runs a refinement pass (no extra upscale) to recover fine
# fabric weave, stitching, and pattern detail that SwinIR smooths.
# True = on (recommended) | False = skip (faster, less detail)
USE_HAT = True
HAT_TILE         = 768   # was 512 — L40S/A100 handle 768 easily; ~2.25× fewer HAT tiles
HAT_TILE_OVERLAP = 64    # was 96 — proportionally fine at 768; keeps seams clean
# 0.0 = only SwinIR, 1.0 = only HAT. Lower values avoid hallucinated texture.
HAT_BLEND_WEIGHT = 0.30
# Re-inject high-frequency detail from original image upsample (non-hallucinatory).
DETAIL_PRESERVE_BLEND = 0.36

# ── Stage 3: Face & skin realism ───────────────────────────────
USE_FACE_ENHANCE    = True
# "codeformer"    = best stability + identity preservation (recommended)
# "restoreformer" = photorealistic skin texture (optional)
# "gfpgan"        = fastest fallback
FACE_BACKEND        = "codeformer"
CODEFORMER_FIDELITY = 0.75
# GFPGAN paste weight: 0.5 = natural blend, 1.0 = full GFPGAN (over-smooth)
# Lower = more original texture preserved, less plastic look
GFPGAN_WEIGHT = 0.22
# Blend enhanced face back with original face region to avoid "AI skin".
# 0.0 = use full enhanced face, 1.0 = keep original face only.
FACE_NATURAL_BLEND = 0.28
# Stage 3.5: skin-region-only refinement for body skin (arms/legs/hands).
USE_BODY_SKIN_REFINE = True
# "texture_transfer" keeps real skin texture from source (recommended)
# "legacy_smooth" keeps old smoothing behavior
# "diffusion_retouch" uses model-based skin retouch (heavier/slower)
BODY_SKIN_BACKEND = "diffusion_retouch"
# Overall blend strength for skin-only refinement.
BODY_SKIN_REFINE_STRENGTH = 0.38
# How much source microtexture is re-injected after smoothing.
BODY_SKIN_TEXTURE_RESTORE = 0.90
# Very subtle grain to avoid waxy/plastic flat skin patches.
# 0.0 = off, typical useful range 0.2 - 0.8
BODY_SKIN_MICRO_GRAIN = 0.18
# Additional synthetic pore boost (used mostly when source detail is weak).
# 0.0 = off, 0.3~0.8 recommended
BODY_SKIN_SYNTH_PORE_STRENGTH = 0.55
# Skin-local luminance contrast recovery (keeps natural volume).
BODY_SKIN_LOCAL_CONTRAST = 0.18
# Use face-tone guidance to avoid modifying non-skin objects.
SKIN_FACE_GUIDED_MASK = True
# Diffusion skin retouch settings (only used when BODY_SKIN_BACKEND=diffusion_retouch)
# Use public inpainting model IDs (no gated access required).
BODY_SKIN_DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-inpainting"
BODY_SKIN_DIFFUSION_FALLBACK_IDS = [
    "runwayml/stable-diffusion-inpainting",
    "stabilityai/stable-diffusion-2-inpainting",
]
BODY_SKIN_DIFFUSION_STEPS = 14
BODY_SKIN_DIFFUSION_GUIDANCE = 3.5
BODY_SKIN_DIFFUSION_STRENGTH = 0.20
# Max long side for diffusion pass to keep runtime practical.
BODY_SKIN_DIFFUSION_MAX_LONG = 1536

# ── Stage 4: Post-processing ────────────────────────────────────
# Bilateral: 0 = OFF. Embroidered/printed fabric needs detail preserved.
BILATERAL_STRENGTH = 0

# CLAHE: local contrast boost. Off — handled by gamma correction below.
CLAHE_CLIP = 0.0

# Deband: imperceptible dither to break SR colour banding.
DEBAND_STRENGTH = 1

# Sharpening: unsharp mask for embroidery thread crispness.
# 0.0 = off | 0.12 = subtle | 0.25 = strong | 0.4 = very strong
SHARPEN_STRENGTH = 0.10
# Reduce sharpening intensity on detected face regions to prevent noisy skin.
# 0.0 = no protection, 1.0 = full protection on faces.
FACE_SHARPEN_PROTECT = 0.85

# Gamma correction: the CORRECT way to fix SR model brightness shift.
# gamma < 1.0 = brighter  |  gamma > 1.0 = darker  |  1.0 = no change
# 0.85 lifts midtones to match input while preserving highlight rolloff.
GAMMA = 1.0

# Preserve original exposure/brightness from input image in the final output.
# 0.0 = off, 1.0 = fully match input tonality.
PRESERVE_INPUT_TONALITY = True
TONALITY_MATCH_STRENGTH = 1.0
# Preserve original color palette so background wall/floor lighting stays same.
PRESERVE_INPUT_COLOR = True
COLOR_MATCH_STRENGTH = 0.9
# Keep flat background regions very close to source look.
PRESERVE_BACKGROUND_LOOK = True
BACKGROUND_BLEND_STRENGTH = 0.9

# Face enhancement: minimum face height as fraction of image height.
# If detected face is smaller than this fraction, skip GFPGAN to avoid
# warping artefacts on small/distant/profile faces.
# 0.0 = always enhance (recommended with RestoreFormer/CodeFormer)
FACE_MIN_SIZE_FRACTION = 0.0

# ═══════════════════════════════════════════════════════════════
#  END OF CONFIG — do not edit below this line
# ═══════════════════════════════════════════════════════════════

QUALITY_PRESETS = {"1K": 1024, "2K": 2048, "4K": 4096, "8K": 8192}

MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)

# SwinIR-L real-world SR x4 (GAN variant — sharpest real-photo output)
SWINIR_URL  = (
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
    "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
)
SWINIR_PATH = MODEL_DIR / "swinir_realsr_x4.pth"

# HAT real-world GAN model (Real_HAT_GAN_SRx4.pth, ~170 MB).
# Primary: HuggingFace mirror (stable, no auth required, direct HTTP).
# Fallback: correct Google Drive folder file ID (1Ma12vCWT27P9M99-s2RXnynKN-OQsBrv).
HAT_URL       = (
    "https://huggingface.co/jaideepsingh/upscale_models/resolve/main/"
    "HAT/Real_HAT_GAN_SRx4.pth"
)
HAT_GDRIVE_ID = "1Ma12vCWT27P9M99-s2RXnynKN-OQsBrv"
HAT_PATH      = MODEL_DIR / "hat_real_gan_x4.pth"

# RestoreFormer++ — photorealistic face restoration (skin pores, texture)
# NOTE: the '+' chars must be URL-encoded as %2B in the GitHub release URL.
RESTOREFORMER_URL  = (
    "https://github.com/wzhouxiff/RestoreFormerPlusPlus/releases/download/v1.0.0/"
    "RestoreFormer%2B%2B.ckpt"
)
RESTOREFORMER_PATH = MODEL_DIR / "RestoreFormerPlusPlus.ckpt"

CODEFORMER_URL  = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
CODEFORMER_PATH = MODEL_DIR / "codeformer.pth"

# GFPGAN v1.4 — fallback face model
GFPGAN_URL  = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
)
GFPGAN_PATH = MODEL_DIR / "GFPGANv1.4.pth"

# facexlib weights used internally by GFPGAN face detection/parsing.
FACEXLIB_DETECTION_URL = (
    "https://github.com/xinntao/facexlib/releases/download/v0.1.0/"
    "detection_Resnet50_Final.pth"
)
FACEXLIB_PARSING_URL = (
    "https://github.com/xinntao/facexlib/releases/download/v0.2.2/"
    "parsing_parsenet.pth"
)


# ─────────────────────────────────────────────
# TIMER
# ─────────────────────────────────────────────
class StageTimer:
    def __init__(self):
        self._pipeline_start = time.time()
        self._stage_start    = None

    def start(self, name):
        self._stage_start = time.time()
        print(f"\n[⏱] Starting: {name}")

    def end(self):
        elapsed = time.time() - self._stage_start
        print(f"[✓] Done in {_fmt(elapsed)}")
        return elapsed

    def eta(self, done, total):
        elapsed = time.time() - self._stage_start
        if done == 0:
            return "calculating..."
        return _fmt((elapsed / done) * (total - done))

    def total(self):
        return time.time() - self._pipeline_start


def _fmt(s):
    s = int(s)
    return f"{s}s" if s < 60 else f"{s // 60}m {s % 60:02d}s"


TIMER = StageTimer()


# ─────────────────────────────────────────────
# 1. DEVICE
# ─────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[✓] GPU: {name}  ({vram:.1f} GB VRAM)")
        return torch.device("cuda")
    print("[!] No CUDA GPU — falling back to CPU (slow)")
    return torch.device("cpu")


# ─────────────────────────────────────────────
# 2. DOWNLOAD
# ─────────────────────────────────────────────
def download_file(url, dest):
    dest = Path(dest)
    if dest.exists():
        print(f"[✓] Already downloaded: {dest.name}")
        return
    print(f"[↓] Downloading {dest.name} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"[✓] Saved to {dest}")


# ─────────────────────────────────────────────
# 3. SWINIR — Stage 1 (base x4 upscaler)
# ─────────────────────────────────────────────
def install_swinir():
    candidates = [
        Path("SwinIR"),
        Path("/SwinIR"),
        Path("/root/SwinIR"),
        Path("/opt/SwinIR"),
    ]

    def _safe_exists(p: Path) -> bool:
        try:
            return p.exists()
        except (PermissionError, OSError):
            return False

    repo = next((p for p in candidates if _safe_exists(p)), None)
    if repo is None:
        print("[→] Cloning SwinIR ...")
        os.system("git clone https://github.com/JingyunLiang/SwinIR.git --depth 1")
        repo = Path("SwinIR")
    repo_str = str(repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

install_swinir()

try:
    from models.network_swinir import SwinIR as SwinIRNet
except ImportError:
    print("[!] SwinIR import failed.")
    sys.exit(1)


def load_swinir(model_path, device):
    model = SwinIRNet(
        upscale=4, in_chans=3, img_size=64, window_size=8,
        img_range=1.0, depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
        embed_dim=240, num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
        mlp_ratio=2, upsampler="nearest+conv", resi_connection="3conv",
    )
    state = torch.load(model_path, map_location=device)
    state = state.get("params_ema", state.get("params", state))
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    if device.type == "cuda":
        model = model.half()
        print("[✓] SwinIR loaded — fp16 GPU mode")
    else:
        print("[✓] SwinIR loaded — fp32 CPU mode")
    return model


def make_cosine_window(size, device):
    ramp = torch.hann_window(size, periodic=False, device=device)
    return ramp.unsqueeze(0) * ramp.unsqueeze(1)


def _tiled_infer(model, img_np, device, tile, overlap, scale=1, use_fp16=True):
    """
    Generic tiled inference for any SR/refinement model.
    scale=4  for SwinIR (x4 upscale)
    scale=1  for HAT used as a refinement-only pass
    """
    window_size = 8
    img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    _, _, h, w = img_t.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    img_t = F.pad(img_t, (0, pad_w, 0, pad_h), mode="reflect")

    _, _, H, W = img_t.shape
    out_t  = torch.zeros(1, 3, H * scale, W * scale, device=device, dtype=torch.float32)
    weight = torch.zeros(1, 1, H * scale, W * scale, device=device, dtype=torch.float32)
    win    = make_cosine_window(tile * scale, device).float()

    tiles_y = math.ceil(H / (tile - overlap))
    tiles_x = math.ceil(W / (tile - overlap))
    total   = tiles_y * tiles_x

    TIMER.start(f"Tiled inference: {total} tiles ({tiles_y}×{tiles_x}), "
                f"tile={tile}, overlap={overlap}")
    count = 0
    for yi in range(tiles_y):
        for xi in range(tiles_x):
            y0 = min(yi * (tile - overlap), H - tile)
            x0 = min(xi * (tile - overlap), W - tile)
            y1, x1 = y0 + tile, x0 + tile

            patch = img_t[:, :, y0:y1, x0:x1].float()
            with torch.no_grad():
                if use_fp16 and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        patch_out = model(patch).clamp(0, 1).float()
                else:
                    patch_out = model(patch).clamp(0, 1)

            sy0, sy1 = y0 * scale, y1 * scale
            sx0, sx1 = x0 * scale, x1 * scale
            out_t[:, :, sy0:sy1, sx0:sx1]  += patch_out * win
            weight[:, :, sy0:sy1, sx0:sx1] += win

            count += 1
            print(f"    tile {count}/{total}  |  ETA: {TIMER.eta(count, total)}      ", end="\r")

    out_t  = (out_t / weight.clamp(min=1e-6))[:, :, :h * scale, :w * scale]
    out_np = out_t.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    out_np = np.clip(out_np * 255, 0, 255).astype(np.uint8)
    out_np = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    print(f"\n[✓] Output: {out_np.shape[1]}x{out_np.shape[0]}")
    return out_np


def resolve_sr_scale_factor(h, w):
    """
    Decide effective upscale factor used by SwinIR/HAT stage.
    """
    mode = SR_UPSCALE_MODE
    if isinstance(mode, str) and mode.lower() == "auto":
        long_side = max(h, w)
        if long_side >= 3800:
            return 1
        if long_side >= 1900:
            return 2
        return 4
    try:
        factor = int(mode)
    except Exception as e:
        raise ValueError(f"Invalid SR_UPSCALE_MODE={mode!r}: {e}")
    if factor not in (1, 2, 4):
        raise ValueError("SR_UPSCALE_MODE must be one of: 'auto', 1, 2, 4")
    return factor


def _infer_with_target_scale(model, img_np, device, tile, overlap, target_scale, use_fp16=True):
    """
    Run fixed x4 SR model to emulate x1/x2/x4 pipeline scale.
    target_scale:
      4 -> direct x4
      2 -> downscale input by 0.5 then x4, yielding net x2
      1 -> downscale input by 0.25 then x4, yielding net x1
    """
    if target_scale == 4:
        return _tiled_infer(model, img_np, device, tile, overlap, scale=4, use_fp16=use_fp16)

    if target_scale not in (1, 2):
        raise ValueError(f"Unsupported target_scale={target_scale}")

    h, w = img_np.shape[:2]
    shrink = target_scale / 4.0
    in_w = max(32, int(round((w * shrink) / 8) * 8))
    in_h = max(32, int(round((h * shrink) / 8) * 8))
    resized_in = cv2.resize(img_np, (in_w, in_h), interpolation=cv2.INTER_AREA)

    sr = _tiled_infer(model, resized_in, device, tile, overlap, scale=4, use_fp16=use_fp16)
    out_w = int(round(w * target_scale))
    out_h = int(round(h * target_scale))
    interp = cv2.INTER_CUBIC if target_scale > 1 else cv2.INTER_LANCZOS4
    out = cv2.resize(sr, (out_w, out_h), interpolation=interp)
    print(f"[✓] Effective upscale x{target_scale} -> {out_w}x{out_h}")
    return out


def swinir_upscale(model, img_np, device, target_scale=4):
    use_fp16 = (device.type == "cuda")
    print(f"[→] Stage 1: SwinIR effective x{target_scale} base upscale")
    return _infer_with_target_scale(
        model, img_np, device,
        tile=SWINIR_TILE, overlap=SWINIR_TILE_OVERLAP,
        target_scale=target_scale, use_fp16=use_fp16,
    )


# ─────────────────────────────────────────────
# 4. HAT — Stage 2 (fabric & pattern detail)
# ─────────────────────────────────────────────
def install_hat():
    candidates = [
        Path("HAT"),
        Path("/HAT"),
        Path("/root/HAT"),
        Path("/opt/HAT"),
    ]
    repo = next((p for p in candidates if p.exists()), None)
    if repo is None:
        print("[→] Cloning HAT ...")
        os.system("git clone https://github.com/XPixelGroup/HAT.git --depth 1")
        repo = Path("HAT")
    # Add the archs directory directly to sys.path so we can import hat_arch
    # WITHOUT triggering HAT's __init__.py (which imports hat.data, which
    # imports the old basicsr degradations module that uses the removed
    # torchvision.transforms.functional_tensor).
    archs_path = str((repo / "hat/archs").resolve())
    if archs_path not in sys.path:
        sys.path.insert(0, archs_path)


def download_hat(dest: Path):
    """
    Download Real_HAT_GAN_SRx4.pth (~170 MB).
    Primary: HuggingFace direct HTTP (no auth, no redirect tricks).
    Fallback: Google Drive via gdown.
    """
    if dest.exists() and dest.stat().st_size > 100_000_000:
        print(f"[✓] Already downloaded: {dest.name}")
        return True
    dest.unlink(missing_ok=True)

    # Primary — HuggingFace
    print(f"[↓] Downloading {dest.name} from HuggingFace ...")
    try:
        download_file(HAT_URL, dest)
        if dest.exists() and dest.stat().st_size > 100_000_000:
            print(f"[✓] Saved to {dest}")
            return True
        print(f"[!] HuggingFace download too small ({dest.stat().st_size // 1024} KB) — trying gdown fallback ...")
        dest.unlink(missing_ok=True)
    except Exception as e:
        print(f"[!] HuggingFace download failed: {e} — trying gdown fallback ...")
        dest.unlink(missing_ok=True)

    # Fallback — Google Drive
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={HAT_GDRIVE_ID}"
        print(f"[↓] Downloading {dest.name} via gdown ...")
        gdown.download(url, str(dest), quiet=False)
        if dest.exists() and dest.stat().st_size > 100_000_000:
            print(f"[✓] Saved to {dest}")
            return True
        print("[!] gdown download also too small — skipping HAT.")
        dest.unlink(missing_ok=True)
        return False
    except Exception as e:
        print(f"[!] HAT download failed entirely: {e}")
        dest.unlink(missing_ok=True)
        return False


def load_hat(model_path, device):
    """
    Load Real_HAT_GAN_SRx4 for real-world image refinement.

    We import hat_arch directly (not via `from hat.archs...`) to avoid
    triggering hat/__init__.py → hat/data/__init__.py → realesrgan_dataset.py
    → old basicsr.data.degradations which imports the removed
    torchvision.transforms.functional_tensor module.

    We also bypass the ARCH_REGISTRY double-registration assert by importing
    the module file directly via importlib instead of re-importing the same
    module object.
    """
    install_hat()
    hat_arch_candidates = [
        Path("HAT/hat/archs/hat_arch.py"),
        Path("/HAT/hat/archs/hat_arch.py"),
        Path("/root/HAT/hat/archs/hat_arch.py"),
        Path("/opt/HAT/hat/archs/hat_arch.py"),
    ]
    hat_arch_file = next((p for p in hat_arch_candidates if p.exists()), None)
    if hat_arch_file is None:
        print("[!] HAT source code not found — skipping fabric refinement.")
        return None
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "hat_arch_module",
            str(hat_arch_file.resolve()),
        )
        _mod = _ilu.module_from_spec(_spec)
        # Register under a unique name so basicsr registry won't collide
        # if load_hat is ever called a second time in the same process.
        import sys as _sys
        _mod_name = "hat_arch_module"
        if _mod_name not in _sys.modules:
            _spec.loader.exec_module(_mod)
            _sys.modules[_mod_name] = _mod
        else:
            _mod = _sys.modules[_mod_name]
        HATNet = _mod.HAT
    except Exception as e:
        print(f"[!] HAT import failed ({type(e).__name__}: {e}) — skipping fabric refinement.")
        return None

    model = HATNet(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )
    state = torch.load(model_path, map_location=device, weights_only=False)
    state = state.get("params_ema", state.get("params", state))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[!] HAT: {len(missing)} missing keys (e.g. {missing[0]}) — wrong checkpoint?")
        if len(missing) > 5:
            print("[!] Too many missing keys — HAT model skipped.")
            return None
    model.eval().to(device)
    if device.type == "cuda":
        model = model.half()
        print("[✓] HAT loaded — fp16 GPU mode")
    else:
        print("[✓] HAT loaded — fp32 CPU mode")
    return model


def hat_refine(model, img_np, device, target_scale=4):
    if model is None:
        return img_np
    use_fp16 = (device.type == "cuda")
    print(f"[→] Stage 2: HAT effective x{target_scale} fabric/pattern refinement")
    return _infer_with_target_scale(
        model, img_np, device,
        tile=HAT_TILE, overlap=HAT_TILE_OVERLAP,
        target_scale=target_scale, use_fp16=use_fp16,
    )


# ─────────────────────────────────────────────
# 5. FACE & SKIN — Stage 3
# ─────────────────────────────────────────────

def enhance_faces_restoreformer(img_bgr, device, model_dir: Path = None):
    """
    RestoreFormer++: state-of-the-art photorealistic face restoration.
    Recovers natural skin pores, subtle texture, and realistic micro-detail
    without the over-smoothed 'AI face' look of older models.
    Falls back to CodeFormer if RestoreFormer++ is unavailable.
    """
    try:
        from basicsr.utils import img2tensor, tensor2img
        # facexlib bundles face restoration helpers under facexlib.utils,
        # not the old standalone 'facelib' package.
        try:
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        except ImportError:
            from facelib.utils.face_restoration_helper import FaceRestoreHelper
        from basicsr.archs.restoreformer_arch import RestoreFormer
    except Exception as e:
        print(f"[!] RestoreFormer++ deps failed ({type(e).__name__}: {e}) — trying CodeFormer ...")
        return enhance_faces_codeformer(img_bgr, device, model_dir)

    mdir = Path(model_dir) if model_dir else MODEL_DIR
    rf_path = mdir / "RestoreFormerPlusPlus.ckpt"
    try:
        if not rf_path.exists():
            download_file(RESTOREFORMER_URL, rf_path)
    except Exception as e:
        print(f"[!] RestoreFormer++ download failed: {e} — trying CodeFormer ...")
        return enhance_faces_codeformer(img_bgr, device, model_dir)

    try:
        ckpt = torch.load(rf_path, map_location=device)
        # Handle both raw state_dict and Lightning checkpoint formats
        state = ckpt.get("state_dict", ckpt)
        # Strip "vqvae." prefix if present (some checkpoints have it)
        state = {k.replace("vqvae.", ""): v for k, v in state.items()}

        net = RestoreFormer().to(device)
        net.load_state_dict(state, strict=False)
        net.eval()
    except Exception as e:
        print(f"[!] RestoreFormer++ load failed ({type(e).__name__}: {e}) — trying CodeFormer ...")
        return enhance_faces_codeformer(img_bgr, device, model_dir)

    face_helper = FaceRestoreHelper(
        upscale_factor=1, face_size=512, crop_ratio=(1, 1),
        det_model="retinaface_resnet50", save_ext="png",
        use_parse=True, device=device,
    )
    face_helper.clean_all()
    face_helper.read_image(img_bgr)
    face_helper.get_face_landmarks_5(
        only_center_face=False, resize=640, eye_dist_threshold=5
    )
    face_helper.align_warp_face()

    if len(face_helper.cropped_faces) == 0:
        print("[!] No faces detected — skipping face enhancement.")
        return img_bgr

    print(f"[→] RestoreFormer++ enhancing {len(face_helper.cropped_faces)} face(s) ...")
    for cropped_face in face_helper.cropped_faces:
        face_t = (
            img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            .unsqueeze(0).to(device)
        )
        with torch.no_grad():
            output   = net(face_t)[0]
            restored = tensor2img(output, rgb2bgr=True, min_max=(-1, 1)).astype("uint8")
        face_helper.add_restored_face(restored)

    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image()
    n = len(face_helper.cropped_faces)
    print(f"[✓] RestoreFormer++ enhanced {n} face(s).")
    return restored_img


def _face_too_small(img_bgr) -> bool:
    """
    Quick heuristic: run a lightweight face detector and check whether the
    largest detected face is smaller than FACE_MIN_SIZE_FRACTION of the image
    height.  Returns True if we should skip GFPGAN to avoid warping artefacts
    on small/distant/profile faces.
    """
    if FACE_MIN_SIZE_FRACTION <= 0.0:
        return False
    try:
        import cv2 as _cv2
        h_img = img_bgr.shape[0]
        # Use OpenCV's built-in frontal-face detector — fast, no extra deps
        _cascade_path = _cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = _cv2.CascadeClassifier(_cascade_path)
        gray    = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2GRAY)
        faces   = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(faces) == 0:
            # Haar misses profile/angled faces often; don't skip blindly.
            print("[!] No frontal face detected by Haar; continuing face enhancement.")
            return False
        largest_h = max(h for (_, _, _, h) in faces)
        fraction  = largest_h / h_img
        if fraction < FACE_MIN_SIZE_FRACTION:
            print(f"[!] Face too small ({fraction:.2%} of image) — skipping GFPGAN to avoid warping.")
            return True
        print(f"[✓] Face size OK ({fraction:.2%} of image) — applying GFPGAN.")
        return False
    except Exception as e:
        print(f"[!] Face-size check failed ({e}) — applying GFPGAN anyway.")
        return False


def enhance_faces_gfpgan(img_bgr, device, model_dir: Path = None):
    try:
        from gfpgan import GFPGANer
    except Exception as e:
        print(f"[!] GFPGAN import failed ({type(e).__name__}: {e}) — skipping face enhancement.")
        return img_bgr

    # Skip on small / profile faces to prevent eye/face warping artefacts
    if _face_too_small(img_bgr):
        return img_bgr

    mdir = Path(model_dir) if model_dir else MODEL_DIR
    gfpgan_path = mdir / "GFPGANv1.4.pth"
    if not gfpgan_path.exists():
        download_file(GFPGAN_URL, gfpgan_path)

    restorer = GFPGANer(
        model_path=str(gfpgan_path), upscale=1, arch="clean",
        channel_multiplier=2, bg_upsampler=None, device=device,
    )
    _, restored_faces, restored_img = restorer.enhance(
        img_bgr, has_aligned=False, only_center_face=False,
        paste_back=True, weight=GFPGAN_WEIGHT,
    )
    if restored_img is None:
        return img_bgr
    print(f"[✓] GFPGAN enhanced {len(restored_faces)} face(s).")
    return restored_img


def enhance_faces_codeformer(img_bgr, device, model_dir: Path = None):
    try:
        import sys as _sys
        cf_root_candidates = [
            Path("/opt/CodeFormer"),
            Path("/CodeFormer"),
            Path("CodeFormer"),
        ]
        cf_root = next((p for p in cf_root_candidates if p.exists()), None)
        if cf_root and str(cf_root.resolve()) not in _sys.path:
            _sys.path.insert(0, str(cf_root.resolve()))

        # If pip basicsr is missing CodeFormer-specific arch files, extend its
        # module search path with CodeFormer's bundled basicsr package.
        if cf_root:
            try:
                import basicsr as _basicsr
                local_bs = str((cf_root / "basicsr").resolve())
                if hasattr(_basicsr, "__path__") and local_bs not in list(_basicsr.__path__):
                    _basicsr.__path__.append(local_bs)
                try:
                    import basicsr.archs as _bs_archs
                    local_archs = str((cf_root / "basicsr/archs").resolve())
                    if hasattr(_bs_archs, "__path__") and local_archs not in list(_bs_archs.__path__):
                        _bs_archs.__path__.append(local_archs)
                except Exception:
                    pass
            except Exception:
                pass

        from basicsr.utils import img2tensor, tensor2img
        try:
            from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        except ImportError:
            from facelib.utils.face_restoration_helper import FaceRestoreHelper
        try:
            from basicsr.archs.codeformer_arch import CodeFormer
        except Exception:
            # Some runtime basicsr builds don't ship codeformer_arch.
            # Load directly from cloned CodeFormer repo as a robust fallback.
            import importlib.util as _ilu
            candidates = [
                Path("/opt/CodeFormer/basicsr/archs/codeformer_arch.py"),
                Path("/CodeFormer/basicsr/archs/codeformer_arch.py"),
                Path("CodeFormer/basicsr/archs/codeformer_arch.py"),
            ]
            arch_file = next((p for p in candidates if p.exists()), None)
            if arch_file is None:
                raise ModuleNotFoundError(
                    "No module named 'basicsr.archs.codeformer_arch' and no local codeformer_arch.py found"
                )

            # Ensure vqgan dependency exists under the expected basicsr namespace.
            vqgan_file = arch_file.parent / "vqgan_arch.py"
            if vqgan_file.exists() and "basicsr.archs.vqgan_arch" not in _sys.modules:
                vq_spec = _ilu.spec_from_file_location(
                    "basicsr.archs.vqgan_arch", str(vqgan_file.resolve())
                )
                vq_mod = _ilu.module_from_spec(vq_spec)
                _sys.modules["basicsr.archs.vqgan_arch"] = vq_mod
                vq_spec.loader.exec_module(vq_mod)

            _spec = _ilu.spec_from_file_location(
                "basicsr.archs.codeformer_arch", str(arch_file.resolve())
            )
            _mod = _ilu.module_from_spec(_spec)
            _sys.modules["basicsr.archs.codeformer_arch"] = _mod
            _spec.loader.exec_module(_mod)
            CodeFormer = _mod.CodeFormer
    except Exception as e:
        print(f"[!] CodeFormer import failed ({type(e).__name__}: {e}) — skipping CodeFormer.")
        return img_bgr

    mdir    = Path(model_dir) if model_dir else MODEL_DIR
    cf_url  = CODEFORMER_URL
    cf_path = mdir / "codeformer.pth"
    if not cf_path.exists():
        print("[↓] Downloading CodeFormer model ...")
        download_file(cf_url, cf_path)

    try:
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
    except ImportError:
        from facelib.utils.face_restoration_helper import FaceRestoreHelper
    net = CodeFormer(
        dim_embd=512, codebook_size=1024, n_head=8,
        n_layers=9, connect_list=["32", "64", "128", "256"],
    ).to(device)
    ckpt = torch.load(cf_path, map_location=device)
    net.load_state_dict(ckpt["params_ema"])
    net.eval()

    face_helper = FaceRestoreHelper(
        upscale_factor=1, face_size=512, crop_ratio=(1, 1),
        det_model="retinaface_resnet50", save_ext="png",
        use_parse=True, device=device,
    )
    face_helper.clean_all()
    face_helper.read_image(img_bgr)
    face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
    face_helper.align_warp_face()

    if len(face_helper.cropped_faces) == 0:
        print("[!] No faces detected by CodeFormer.")
        return img_bgr

    print(f"[→] CodeFormer enhancing {len(face_helper.cropped_faces)} face(s) ...")
    for cropped_face in face_helper.cropped_faces:
        face_t = (
            img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            .unsqueeze(0).to(device)
        )
        with torch.no_grad():
            output   = net(face_t, w=CODEFORMER_FIDELITY, adain=True)[0]
            restored = tensor2img(output, rgb2bgr=True, min_max=(0, 1)).astype("uint8")
        face_helper.add_restored_face(restored)

    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image()
    n = len(face_helper.cropped_faces)
    print(f"[✓] CodeFormer enhanced {n} face(s).")
    return restored_img


def enhance_faces(img_bgr, device, model_dir: Path = None):
    if FACE_BACKEND == "restoreformer":
        return enhance_faces_restoreformer(img_bgr, device, model_dir)
    if FACE_BACKEND == "codeformer":
        return enhance_faces_codeformer(img_bgr, device, model_dir)
    return enhance_faces_gfpgan(img_bgr, device, model_dir)


# ─────────────────────────────────────────────
# 6. POST-PROCESSING — Stage 4
# ─────────────────────────────────────────────

def bilateral_smooth(img_bgr):
    """
    Removes the 'cracked-mud / hallucinated texture' artifact that SR models
    invent on smooth fabric (denim, jersey, satin) while keeping hard edges
    (seams, zippers, stitching) perfectly sharp.
    """
    if BILATERAL_STRENGTH <= 0:
        return img_bgr
    d       = 9
    sigma_c = BILATERAL_STRENGTH * 8
    sigma_s = BILATERAL_STRENGTH * 4
    result  = cv2.bilateralFilter(img_bgr, d, sigma_c, sigma_s)
    print(f"[✓] Bilateral smooth (strength={BILATERAL_STRENGTH})")
    return result


def gamma_correction(img_bgr):
    """
    Proper gamma correction to restore brightness lost during SR upscaling.
    Builds a per-pixel LUT (lookup table) — zero extra allocations, very fast.
    gamma < 1.0 brightens; gamma > 1.0 darkens; 1.0 = no change.
    """
    if abs(GAMMA - 1.0) < 1e-4:
        return img_bgr
    inv_gamma = 1.0 / GAMMA
    lut = (np.arange(256, dtype=np.float32) / 255.0) ** inv_gamma * 255.0
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    result = cv2.LUT(img_bgr, lut)
    print(f"[✓] Gamma correction (γ={GAMMA}  →  midtones {'↑' if GAMMA < 1 else '↓'})")
    return result


def clahe_color_correction(img_bgr):
    lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))
    result  = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    print(f"[✓] CLAHE color correction (clip={CLAHE_CLIP})")
    return result


def deband(img_bgr):
    rng   = np.random.default_rng(seed=42)
    noise = rng.integers(-DEBAND_STRENGTH, DEBAND_STRENGTH + 1,
                         img_bgr.shape, dtype=np.int16)
    result = np.clip(img_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    print(f"[✓] Deband (strength={DEBAND_STRENGTH})")
    return result


def adaptive_sharpen(img_bgr):
    """
    Two-pass unsharp mask:
      Pass 1 — global base sharpening (applied everywhere, combats SR softness)
      Pass 2 — extra boost on detected edges (fabric threads, embroidery)
    This is far more effective than edge-only sharpening for SR output.
    """
    s = SHARPEN_STRENGTH
    if s <= 0:
        return img_bgr

    # ── Pass 1: global unsharp mask ─────────────────────────────────────
    # sigma=1.5 targets fine detail; amount = s * 2 for noticeable crispness
    blurred_fine = cv2.GaussianBlur(img_bgr, (0, 0), 1.5)
    sharpened    = cv2.addWeighted(img_bgr, 1.0 + s * 2.0, blurred_fine, -s * 2.0, 0)

    # ── Pass 2: edge-boost mask ─────────────────────────────────────────
    gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges     = cv2.Canny(gray, 20, 80)
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_mask = cv2.GaussianBlur(
        cv2.dilate(edges, kernel).astype(np.float32) / 255.0, (0, 0), 3.0
    )
    # Extra sharpening on edges with a coarser radius to pop fabric weave
    blurred_coarse  = cv2.GaussianBlur(img_bgr, (0, 0), 3.0)
    sharp_edge_only = cv2.addWeighted(img_bgr, 1.0 + s * 1.5, blurred_coarse, -s * 1.5, 0)
    mask_3c  = np.stack([edge_mask] * 3, axis=-1)
    result   = np.clip(
        sharp_edge_only.astype(np.float32) * mask_3c
        + sharpened.astype(np.float32) * (1.0 - mask_3c),
        0, 255,
    ).astype(np.uint8)

    # Protect face skin from over-sharpening noise.
    protect = float(np.clip(FACE_SHARPEN_PROTECT, 0.0, 1.0))
    if protect > 0:
        try:
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.08, minNeighbors=4, minSize=(36, 36)
            )
            if len(faces) > 0:
                h, w = gray.shape[:2]
                yy, xx = np.mgrid[0:h, 0:w]
                fmask = np.zeros((h, w), dtype=np.float32)
                for (x, y, fw, fh) in faces:
                    cx, cy = x + fw / 2.0, y + fh / 2.0
                    rx, ry = fw * 0.82, fh * 0.95
                    ell = ((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2
                    local = np.clip(1.0 - ell, 0.0, 1.0)
                    local = cv2.GaussianBlur(local, (0, 0), 4.0)
                    fmask = np.maximum(fmask, local)
                fmask = np.clip(fmask * protect, 0.0, 1.0)
                fmask3 = np.stack([fmask, fmask, fmask], axis=-1)
                result = np.clip(
                    result.astype(np.float32) * (1.0 - fmask3)
                    + img_bgr.astype(np.float32) * fmask3,
                    0, 255
                ).astype(np.uint8)
        except Exception:
            pass
    print(f"[✓] Adaptive sharpen (strength={s}, global+edge-boost)")
    return result


def preserve_original_texture(sr_bgr, original_bgr):
    """
    Preserve true fabric micro-detail from the source image (upsampled) to
    reduce GAN hallucination and avoid over-sharpened synthetic texture.
    """
    b = DETAIL_PRESERVE_BLEND
    if b <= 0:
        return sr_bgr

    h, w = sr_bgr.shape[:2]
    src_up = cv2.resize(original_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

    # Multi-scale residuals keep pattern geometry faithful without adding
    # synthetic GAN-only texture.
    src_low_fine = cv2.GaussianBlur(src_up, (0, 0), 1.1)
    src_low_mid = cv2.GaussianBlur(src_up, (0, 0), 2.4)
    residual = (
        (src_up.astype(np.float32) - src_low_fine.astype(np.float32)) * 0.65
        + (src_low_fine.astype(np.float32) - src_low_mid.astype(np.float32)) * 0.35
    )

    # Edge-confidence mask so we preserve fabric/pattern detail, not flat walls.
    gray = cv2.cvtColor(src_up, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)
    mask = cv2.GaussianBlur(np.clip((mag - 0.08) / 0.35, 0.0, 1.0), (0, 0), 1.4)
    mask3 = np.stack([mask, mask, mask], axis=-1)

    out = sr_bgr.astype(np.float32) + residual * b * mask3
    out = np.clip(out, 0, 255).astype(np.uint8)
    print(f"[✓] Texture preserve blend (strength={b:.2f})")
    return out


def blend_faces_natural(pre_face_bgr, enhanced_bgr):
    """
    Reduce plastic/AI skin look by softly blending each detected face region
    back with the pre-enhancement image.
    """
    blend = float(np.clip(FACE_NATURAL_BLEND, 0.0, 1.0))
    if blend <= 0.0:
        return enhanced_bgr

    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(pre_face_bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=4, minSize=(36, 36)
        )
    except Exception as e:
        print(f"[!] Face natural blend skipped ({e})")
        return enhanced_bgr

    if len(faces) == 0:
        return enhanced_bgr

    # If enhancement changed only limited region(s), constrain blending there.
    diff = cv2.absdiff(enhanced_bgr, pre_face_bgr)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    changed = cv2.GaussianBlur((diff_gray > 2).astype(np.float32), (0, 0), 2.2)

    h, w = pre_face_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for (x, y, fw, fh) in faces:
        cx, cy = x + fw / 2.0, y + fh / 2.0
        rx, ry = fw * 0.70, fh * 0.85
        if rx < 1 or ry < 1:
            continue
        ell = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        local = np.clip(1.0 - ell, 0.0, 1.0)
        local = cv2.GaussianBlur(local, (0, 0), 5.0)
        mask = np.maximum(mask, local)

    # Keep blend only where enhancement actually modified pixels, avoids
    # over-blending unaffected/false-positive face boxes.
    mask = mask * np.clip(changed * 1.5, 0.0, 1.0)
    mask = np.clip(mask * blend, 0.0, 1.0)
    mask3 = np.stack([mask, mask, mask], axis=-1)
    out = (
        enhanced_bgr.astype(np.float32) * (1.0 - mask3)
        + pre_face_bgr.astype(np.float32) * mask3
    )
    out = np.clip(out, 0, 255).astype(np.uint8)
    print(f"[✓] Face natural blend (faces={len(faces)}, strength={blend:.2f})")
    return out


def _skin_mask_hsv_ycrcb(img_bgr):
    """
    Build robust skin mask using HSV + YCrCb intersection.
    Tuned to be conservative to avoid touching garments/background.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    # HSV ranges for skin-like hues (light to medium/darker tones).
    h, s, v = cv2.split(hsv)
    hsv_mask = (
        (((h <= 25) | (h >= 160)) & (s >= 25) & (s <= 185) & (v >= 35))
        .astype(np.uint8) * 255
    )

    # YCrCb canonical skin cluster.
    y, cr, cb = cv2.split(ycc)
    ycc_mask = (
        (cr >= 130) & (cr <= 180) &
        (cb >= 75) & (cb <= 135) &
        (y >= 25) & (y <= 245)
    ).astype(np.uint8) * 255

    mask = cv2.bitwise_and(hsv_mask, ycc_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (0, 0), 2.0)
    return mask.astype(np.float32) / 255.0


def _face_guided_skin_filter(img_bgr, skin_mask):
    """
    Narrow skin mask using face color priors so chairs/walls/wood tones
    are less likely to be modified.
    """
    if not SKIN_FACE_GUIDED_MASK:
        return skin_mask

    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=4, minSize=(36, 36)
        )
    except Exception:
        return skin_mask

    if len(faces) == 0:
        return skin_mask

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab = lab[:, :, 1:3]
    ab_samples = []
    for (x, y, fw, fh) in faces:
        x0 = int(x + 0.20 * fw)
        x1 = int(x + 0.80 * fw)
        y0 = int(y + 0.18 * fh)
        y1 = int(y + 0.72 * fh)
        x0, x1 = max(0, x0), min(img_bgr.shape[1], x1)
        y0, y1 = max(0, y0), min(img_bgr.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            continue
        roi_ab = ab[y0:y1, x0:x1]
        roi_m = skin_mask[y0:y1, x0:x1] > 0.25
        if roi_m.any():
            ab_samples.append(roi_ab[roi_m])
        else:
            ab_samples.append(roi_ab.reshape(-1, 2))

    if not ab_samples:
        return skin_mask
    ab_mean = np.concatenate(ab_samples, axis=0).mean(axis=0)
    dist = np.sqrt(np.sum((ab - ab_mean) ** 2, axis=2))
    color_gate = np.clip((30.0 - dist) / 30.0, 0.0, 1.0)
    out = skin_mask * color_gate
    return cv2.GaussianBlur(out, (0, 0), 1.6)


def _exclude_face_regions(mask, img_bgr):
    """
    Remove face regions from body-skin mask so Stage 3.5 does not soften
    already-enhanced face details.
    """
    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=4, minSize=(36, 36)
        )
    except Exception:
        return mask

    if len(faces) == 0:
        return mask

    h, w = img_bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    suppress = np.zeros((h, w), dtype=np.float32)
    for (x, y, fw, fh) in faces:
        cx, cy = x + fw / 2.0, y + fh / 2.0
        rx, ry = fw * 0.78, fh * 0.95
        ell = ((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2
        local = np.clip(1.0 - ell, 0.0, 1.0)
        local = cv2.GaussianBlur(local, (0, 0), 3.0)
        suppress = np.maximum(suppress, local)

    out = mask * (1.0 - np.clip(suppress, 0.0, 1.0))
    return cv2.GaussianBlur(out, (0, 0), 1.2)


def _multiscale_pore_noise(h, w, seed=321):
    """
    Generate subtle pore-like micro texture (multi-scale high-frequency noise).
    """
    rng = np.random.default_rng(seed=seed)
    n1 = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    n2 = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    n3 = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)

    # Pore scales: fine + medium, then remove very low-frequency drift.
    n1 = cv2.GaussianBlur(n1, (0, 0), 0.55)
    n2 = cv2.GaussianBlur(n2, (0, 0), 1.10)
    n3 = cv2.GaussianBlur(n3, (0, 0), 2.20)
    noise = n1 * 0.55 + n2 * 0.35 - n3 * 0.20
    noise = cv2.GaussianBlur(noise, (0, 0), 0.45)

    m = float(np.std(noise))
    if m > 1e-6:
        noise = noise / m
    return noise


def load_body_skin_diffuser(device, cache_dir: Path = None):
    """
    Load diffusion inpainting pipeline for model-based body skin retouch.
    """
    try:
        from diffusers import StableDiffusionInpaintPipeline
    except Exception as e:
        print(f"[!] Diffusion deps unavailable ({type(e).__name__}: {e})")
        return None

    model_ids = [BODY_SKIN_DIFFUSION_MODEL_ID] + list(BODY_SKIN_DIFFUSION_FALLBACK_IDS)
    # Preserve order but deduplicate.
    dedup_ids = []
    for mid in model_ids:
        if mid and mid not in dedup_ids:
            dedup_ids.append(mid)

    last_err = None
    hf_token = os.environ.get("HF_TOKEN", "").strip() or None
    for model_id in dedup_ids:
        try:
            kwargs = {
                "torch_dtype": torch.float16 if device.type == "cuda" else torch.float32,
                "use_safetensors": True,
            }
            if cache_dir is not None:
                kwargs["cache_dir"] = str(cache_dir)
            if device.type == "cuda":
                kwargs["variant"] = "fp16"
            if hf_token:
                kwargs["token"] = hf_token
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                **kwargs,
            ).to(device)
            try:
                if device.type == "cuda":
                    pipe.enable_attention_slicing()
            except Exception:
                pass
            print(f"[✓] Loaded body-skin diffusion retouch model: {model_id}")
            return pipe
        except Exception as e:
            print(f"[!] Diffusion model load failed for {model_id} ({type(e).__name__}: {e})")
            last_err = e
            continue

    print(f"[!] Failed to load any diffusion retouch model ({type(last_err).__name__}: {last_err})")
    return None


def _refine_body_skin_diffusion(img_bgr, skin_safe, device, diffuser_pipe):
    """
    Diffusion inpainting over skin mask only (face excluded).
    """
    if diffuser_pipe is None:
        return img_bgr

    h, w = img_bgr.shape[:2]
    mask = np.clip(skin_safe * BODY_SKIN_REFINE_STRENGTH, 0.0, 1.0)
    coverage = float((mask > 0.08).mean() * 100.0)
    if coverage < 0.15:
        print("[!] Body-skin diffusion skipped (mask coverage too low).")
        return img_bgr

    max_long = int(max(256, BODY_SKIN_DIFFUSION_MAX_LONG))
    long_side = max(h, w)
    scale = min(1.0, max_long / float(long_side))
    rw = max(64, int(round(w * scale / 8) * 8))
    rh = max(64, int(round(h * scale / 8) * 8))

    img_small = cv2.resize(img_bgr, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
    mask_small = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_LINEAR)
    mask_small = np.clip(mask_small, 0.0, 1.0)
    mask_u8 = (mask_small * 255).astype(np.uint8)

    try:
        from PIL import Image
        pil_img = Image.fromarray(cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(mask_u8, mode="L")
        prompt = (
            "natural realistic human skin texture, subtle pores, detailed but clean skin, "
            "preserve identity, preserve lighting, preserve colors, no makeup change, no blur"
        )
        negative = (
            "plastic skin, waxy skin, over-smoothed skin, noisy artifacts, fake pores, "
            "airbrushed, painting, cartoon, altered face, changed clothes"
        )
        out = diffuser_pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=pil_img,
            mask_image=pil_mask,
            num_inference_steps=int(BODY_SKIN_DIFFUSION_STEPS),
            guidance_scale=float(BODY_SKIN_DIFFUSION_GUIDANCE),
            strength=float(BODY_SKIN_DIFFUSION_STRENGTH),
        ).images[0]
        out_bgr_small = cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[!] Body-skin diffusion inference failed ({type(e).__name__}: {e})")
        return img_bgr

    out_bgr = cv2.resize(out_bgr_small, (w, h), interpolation=cv2.INTER_CUBIC)
    mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_full = cv2.GaussianBlur(mask_full, (0, 0), 1.4)
    m3 = np.stack([mask_full, mask_full, mask_full], axis=-1)
    blended = np.clip(
        img_bgr.astype(np.float32) * (1.0 - m3)
        + out_bgr.astype(np.float32) * m3,
        0, 255
    ).astype(np.uint8)
    print(f"[✓] Body-skin diffusion refine (coverage={coverage:.1f}%, size={rw}x{rh})")
    return blended


def refine_body_skin_only(img_bgr, original_bgr, device=None, preloaded_models=None):
    """
    Stage 3.5: refine non-face skin realism while protecting fabric edges
    and keeping background untouched.
    """
    strength = float(np.clip(BODY_SKIN_REFINE_STRENGTH, 0.0, 1.0))
    tex = float(np.clip(BODY_SKIN_TEXTURE_RESTORE, 0.0, 1.0))
    grain = float(np.clip(BODY_SKIN_MICRO_GRAIN, 0.0, 2.0))
    pore = float(np.clip(BODY_SKIN_SYNTH_PORE_STRENGTH, 0.0, 1.5))
    local_contrast = float(np.clip(BODY_SKIN_LOCAL_CONTRAST, 0.0, 0.8))
    if (not USE_BODY_SKIN_REFINE) or strength <= 0.0:
        return img_bgr

    h, w = img_bgr.shape[:2]
    src = cv2.resize(original_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

    skin = _skin_mask_hsv_ycrcb(img_bgr)
    skin = _face_guided_skin_filter(img_bgr, skin)
    skin = _exclude_face_regions(skin, img_bgr)

    # Protect embroidery/garment edges via gradient-based edge gating.
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)
    edge_gate = np.clip((mag - 0.10) / 0.22, 0.0, 1.0)
    skin_safe = skin * (1.0 - edge_gate)
    skin_safe = cv2.GaussianBlur(skin_safe, (0, 0), 1.6)

    backend = str(BODY_SKIN_BACKEND).lower().strip()
    if backend == "diffusion_retouch":
        diff_pipe = None
        if preloaded_models:
            diff_pipe = preloaded_models.get("skin_diffuser")
        if diff_pipe is None:
            # Local/non-modal fallback lazy-load.
            cache_dir = MODEL_DIR / "hf_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            if device is None:
                device = get_device()
            diff_pipe = load_body_skin_diffuser(device, cache_dir=cache_dir)
        if diff_pipe is None:
            print("[!] Falling back to texture_transfer skin backend.")
            backend = "texture_transfer"
        else:
            return _refine_body_skin_diffusion(img_bgr, skin_safe, device, diff_pipe)

    if backend == "legacy_smooth":
        # Very mild smoothing only to remove synthetic blotches.
        smooth = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=14, sigmaSpace=7)
        cur_low = cv2.GaussianBlur(img_bgr, (0, 0), 1.0)
        cur_res = img_bgr.astype(np.float32) - cur_low.astype(np.float32)
        src_low = cv2.GaussianBlur(src, (0, 0), 1.0)
        src_res = src.astype(np.float32) - src_low.astype(np.float32)
        detail = cur_res * 0.75 + src_res * 0.25
        if grain > 0:
            rng = np.random.default_rng(seed=123)
            g = rng.normal(0.0, grain, size=(h, w)).astype(np.float32)
            g = cv2.GaussianBlur(g, (0, 0), 0.9)
            detail += np.stack([g, g, g], axis=-1)
        smooth_f = smooth.astype(np.float32)
        skin_refined = np.clip(
            smooth_f * (1.0 - tex * 0.35) + (smooth_f + detail) * (tex * 0.35),
            0, 255
        ).astype(np.uint8)
    else:
        # Texture-transfer backend: avoid smoothing/waxiness by keeping SR base
        # and injecting real high-frequency skin detail from source image.
        sr_f = img_bgr.astype(np.float32)
        src_f = src.astype(np.float32)

        # Multi-scale real detail from source.
        src_lp1 = cv2.GaussianBlur(src_f, (0, 0), 0.9)
        src_lp2 = cv2.GaussianBlur(src_f, (0, 0), 2.0)
        src_hp = (src_f - src_lp1) * 0.65 + (src_lp1 - src_lp2) * 0.35

        # Keep a small amount of existing SR detail too.
        sr_lp = cv2.GaussianBlur(sr_f, (0, 0), 1.0)
        sr_hp = sr_f - sr_lp
        detail = src_hp * tex + sr_hp * (1.0 - tex * 0.35)

        # Optional subtle luminance grain for anti-plastic effect.
        if grain > 0:
            rng = np.random.default_rng(seed=123)
            g = rng.normal(0.0, grain, size=(h, w)).astype(np.float32)
            g = cv2.GaussianBlur(g, (0, 0), 0.8)
            detail += np.stack([g, g, g], axis=-1)

        # Clamp detail gain to avoid noisy/harsh skin.
        detail_gain = np.clip(tex * 0.55, 0.0, 0.65)
        skin_refined = np.clip(sr_f + detail * detail_gain, 0, 255).astype(np.uint8)

        # If source detail is weak/plastic, inject adaptive pore texture
        # and recover local luminance contrast only in skin regions.
        if pore > 0 or local_contrast > 0:
            lab = cv2.cvtColor(skin_refined, cv2.COLOR_BGR2LAB).astype(np.float32)
            l = lab[:, :, 0]

            src_energy = np.mean(np.abs(src_hp), axis=2)
            src_energy = cv2.GaussianBlur(src_energy, (0, 0), 1.2)
            # Higher gate where source has low detail (plastic-like areas).
            low_detail_gate = np.clip((12.0 - src_energy) / 12.0, 0.0, 1.0)
            low_detail_gate = cv2.GaussianBlur(low_detail_gate, (0, 0), 1.2)

            if pore > 0:
                pores = _multiscale_pore_noise(h, w)
                pore_amp = pore * 1.6
                l = l + pores * pore_amp * low_detail_gate * np.clip(skin_safe, 0.0, 1.0)

            if local_contrast > 0:
                l_blur = cv2.GaussianBlur(l, (0, 0), 1.2)
                l_hp = l - l_blur
                lc_amp = local_contrast * 0.9
                l = l + l_hp * lc_amp * np.clip(skin_safe, 0.0, 1.0)

            lab[:, :, 0] = np.clip(l, 0, 255)
            skin_refined = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    m = np.clip(skin_safe * strength, 0.0, 1.0)
    m3 = np.stack([m, m, m], axis=-1)
    out = (
        img_bgr.astype(np.float32) * (1.0 - m3)
        + skin_refined.astype(np.float32) * m3
    )
    out = np.clip(out, 0, 255).astype(np.uint8)
    coverage = float((m > 0.08).mean() * 100.0)
    print(f"[✓] Body-skin refine ({backend}, strength={strength:.2f}, coverage={coverage:.1f}%)")
    return out


def match_input_tonality(sr_bgr, original_bgr):
    """
    Match output luminance to input luminance so exposure/brightness stays
    visually consistent even after SR and face enhancement.
    """
    if not PRESERVE_INPUT_TONALITY:
        return sr_bgr

    strength = float(np.clip(TONALITY_MATCH_STRENGTH, 0.0, 1.0))
    if strength <= 0.0:
        return sr_bgr

    h, w = sr_bgr.shape[:2]
    src = cv2.resize(original_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

    # Work in LAB so only luminance is corrected and colors are preserved.
    sr_lab = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_sr = sr_lab[:, :, 0]
    l_src = src_lab[:, :, 0]

    # Robust percentile mapping prevents being dominated by outliers.
    src_p5, src_p95 = np.percentile(l_src, [5, 95])
    sr_p5, sr_p95 = np.percentile(l_sr, [5, 95])
    scale = (src_p95 - src_p5) / max(sr_p95 - sr_p5, 1e-6)
    bias = src_p5 - scale * sr_p5

    mapped = np.clip(l_sr * scale + bias, 0.0, 255.0)

    # Align means as a gentle second pass (helps stubborn under/over exposure).
    mean_delta = float(l_src.mean() - mapped.mean())
    mapped = np.clip(mapped + mean_delta * 0.6, 0.0, 255.0)

    # Optional blend if user wants partial matching.
    l_final = l_sr * (1.0 - strength) + mapped * strength
    sr_lab[:, :, 0] = np.clip(l_final, 0.0, 255.0)
    out = cv2.cvtColor(sr_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    print(f"[✓] Tonality match to input (strength={strength:.2f})")
    return out


def match_input_color_palette(sr_bgr, original_bgr):
    """
    Keep color combination close to the original photoshoot image
    (background wall, plant tone, floor warmth, etc.).
    """
    if not PRESERVE_INPUT_COLOR:
        return sr_bgr

    strength = float(np.clip(COLOR_MATCH_STRENGTH, 0.0, 1.0))
    if strength <= 0.0:
        return sr_bgr

    h, w = sr_bgr.shape[:2]
    src = cv2.resize(original_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    sr_lab = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Match only chroma channels (a,b), keep luminance handled separately.
    for c in (1, 2):
        sr_c = sr_lab[:, :, c]
        src_c = src_lab[:, :, c]
        sr_mean, sr_std = float(sr_c.mean()), float(sr_c.std())
        src_mean, src_std = float(src_c.mean()), float(src_c.std())
        scale = src_std / max(sr_std, 1e-6)
        scale = float(np.clip(scale, 0.85, 1.15))
        mapped = (sr_c - sr_mean) * scale + src_mean
        sr_lab[:, :, c] = sr_c * (1.0 - strength) + mapped * strength

    out = cv2.cvtColor(np.clip(sr_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    print(f"[✓] Color palette match (strength={strength:.2f})")
    return out


def preserve_background_look(sr_bgr, original_bgr):
    """
    Preserve wall/floor/background lighting and tone by blending flat regions
    with the upscaled source image.
    """
    if not PRESERVE_BACKGROUND_LOOK:
        return sr_bgr

    strength = float(np.clip(BACKGROUND_BLEND_STRENGTH, 0.0, 1.0))
    if strength <= 0.0:
        return sr_bgr

    h, w = sr_bgr.shape[:2]
    src = cv2.resize(original_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Higher mask in flat regions (typical background), lower on detailed areas.
    bg = np.clip((0.18 - mag) / 0.18, 0.0, 1.0)

    # Spatial prior: emphasize outer borders so subject area (center) keeps SR detail.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    rx, ry = max(1.0, w * 0.38), max(1.0, h * 0.42)
    center_ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    border_prior = np.clip((center_ellipse - 0.95) / 0.9, 0.0, 1.0)
    bg = bg * border_prior

    # Preserve colorful/detailed regions (often garments), prioritize neutral backdrop.
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[:, :, 1] / 255.0
    neutral_gate = np.clip((0.45 - sat) / 0.45, 0.0, 1.0)
    bg = bg * neutral_gate

    bg = cv2.GaussianBlur(bg, (0, 0), 4.5)
    bg = np.clip(bg * strength, 0.0, 1.0)
    bg3 = np.stack([bg, bg, bg], axis=-1)

    out = (
        sr_bgr.astype(np.float32) * (1.0 - bg3)
        + src.astype(np.float32) * bg3
    )
    out = np.clip(out, 0, 255).astype(np.uint8)
    print(f"[✓] Background look preserve (strength={strength:.2f})")
    return out


def post_process(img_bgr):
    # 1. Bilateral only if needed (smooth fabrics). Off for embroidery/prints.
    if BILATERAL_STRENGTH > 0:
        img_bgr = bilateral_smooth(img_bgr)
    # 2. Gamma correction — restores brightness lost during SR upscaling
    img_bgr = gamma_correction(img_bgr)
    # 3. CLAHE — off by default (gamma handles brightness; CLAHE can darken)
    if CLAHE_CLIP > 0:
        img_bgr = clahe_color_correction(img_bgr)
    # 4. Sharpen edges (embroidery thread, fabric weave)
    if SHARPEN_STRENGTH > 0:
        img_bgr = adaptive_sharpen(img_bgr)
    # 5. Deband — dither to break colour banding
    if DEBAND_STRENGTH > 0:
        img_bgr = deband(img_bgr)
    return img_bgr


# ─────────────────────────────────────────────
# 7. OUTPUT HELPERS
# ─────────────────────────────────────────────

def resize_to_target(img_bgr, target_long):
    h, w = img_bgr.shape[:2]
    if max(h, w) == target_long:
        return img_bgr
    scale        = target_long / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    interp       = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_LANCZOS4
    resized      = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
    print(f"[✓] Resized to {new_w}x{new_h}")
    return resized


def save_resolution(img_bgr, stem, label, target_long):
    out      = resize_to_target(img_bgr, target_long)
    out_path = f"{stem}_realism_{label.lower()}.png"
    cv2.imwrite(out_path, out, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    h, w = out.shape[:2]
    print(f"[✓] Saved {label} -> {out_path}  ({w}x{h})")
    return out_path


def get_enabled_outputs(sr_img):
    """
    Build output list in descending resolution order.
    Includes 8K only when explicitly enabled.
    """
    outputs = []
    if SAVE_8K:
        outputs.append(("8k", QUALITY_PRESETS["8K"]))
    if SAVE_4K:
        outputs.append(("4k", QUALITY_PRESETS["4K"]))
    if SAVE_2K:
        outputs.append(("2k", QUALITY_PRESETS["2K"]))
    if SAVE_1K:
        outputs.append(("1k", QUALITY_PRESETS["1K"]))
    return outputs


# ─────────────────────────────────────────────
# 8. FULL PIPELINE (local)
# ─────────────────────────────────────────────

def run_pipeline(img_bgr, device, tmp_dir: Path, preloaded_models: dict = None):
    """
    Runs all 4 stages and returns the final enhanced image (full res).
    tmp_dir: directory where model weights are cached.

    Pipeline:
      Stage 1 — SwinIR x{1|2|4} upscale  (base super-resolution)
      Stage 2 — HAT x{1|2|4} upscale     (fabric/pattern detail, run on ORIGINAL
                                          input; outputs blended with SwinIR)
      Stage 3 — Face realism
      Stage 3.5 — Body skin-only refinement
      Stage 4 — Post-processing

    Why blend instead of chain?
    HAT and SwinIR are fixed x4 models. For effective x1/x2/x4 behavior we
    pre-resize input before model inference and post-resize outputs back to the
    target scale. We still run both on the original input branch and blend.
    """
    preloaded_models = preloaded_models or {}
    managed_swinir = False
    managed_hat = False
    target_scale = resolve_sr_scale_factor(img_bgr.shape[0], img_bgr.shape[1])
    print(f"[✓] Effective SR scale selected: x{target_scale} (mode={SR_UPSCALE_MODE})")

    # ── Stage 1: SwinIR x4 ──────────────────────────────────────
    swinir_model = preloaded_models.get("swinir")
    if swinir_model is None:
        TIMER.start("Stage 1 — SwinIR model load")
        # When running on Modal, tmp_dir IS the weights volume (/weights).
        # When running locally, weights are downloaded on demand to tmp_dir.
        swinir_path = tmp_dir / "swinir_realsr_x4.pth"
        if not swinir_path.exists():
            download_file(SWINIR_URL, swinir_path)
        swinir_model = load_swinir(swinir_path, device)
        TIMER.end()
        managed_swinir = True
    else:
        print("[✓] Using preloaded SwinIR model from warm container.")

    TIMER.start(f"Stage 1 — SwinIR effective x{target_scale} upscale")
    sr_swinir = swinir_upscale(swinir_model, img_bgr, device, target_scale=target_scale)
    TIMER.end()
    if managed_swinir:
        del swinir_model
    if device.type == "cuda" and managed_swinir:
        torch.cuda.empty_cache()

    sr = sr_swinir  # default output if HAT is skipped

    # ── Stage 2: HAT x4 (run on original input, blend with SwinIR) ─
    if USE_HAT:
        hat_model = preloaded_models.get("hat")
        if hat_model is None:
            TIMER.start("Stage 2 — HAT model load")
            hat_path = tmp_dir / "hat_real_gan_x4.pth"
            hat_ok   = True
            if not hat_path.exists():
                hat_ok = download_hat(hat_path)
            hat_model = load_hat(hat_path, device) if hat_ok else None
            TIMER.end()
            managed_hat = hat_model is not None
        else:
            print("[✓] Using preloaded HAT model from warm container.")

        if hat_model is not None:
            TIMER.start(f"Stage 2 — HAT effective x{target_scale} fabric/pattern upscale")
            sr_hat = hat_refine(hat_model, img_bgr, device, target_scale=target_scale)
            TIMER.end()
            if managed_hat:
                del hat_model
            if device.type == "cuda" and managed_hat:
                torch.cuda.empty_cache()

            # Blend: SwinIR (colour/structure) + HAT (texture detail).
            # Both are x4 outputs at the same resolution.
            hat_w = float(np.clip(HAT_BLEND_WEIGHT, 0.0, 1.0))
            swinir_w = 1.0 - hat_w
            print(f"[→] Blending SwinIR + HAT outputs ({swinir_w:.2f}/{hat_w:.2f}) ...")
            sr = cv2.addWeighted(sr_swinir, swinir_w, sr_hat, hat_w, 0)
            print("[✓] Blend complete.")

    # Preserve true texture detail from source image to reduce hallucination.
    sr = preserve_original_texture(sr, img_bgr)

    # ── Stage 3: Face realism ───────────────────────────────────
    if USE_FACE_ENHANCE:
        TIMER.start(f"Stage 3 — Face ({FACE_BACKEND})")
        sr_before_face = sr
        sr = enhance_faces(sr_before_face, device, model_dir=tmp_dir)
        sr = blend_faces_natural(sr_before_face, sr)
        TIMER.end()

    # ── Stage 3.5: Body skin-only refinement ────────────────────
    if USE_BODY_SKIN_REFINE:
        TIMER.start("Stage 3.5 — Body skin refinement")
        sr = refine_body_skin_only(sr, img_bgr, device=device, preloaded_models=preloaded_models)
        TIMER.end()

    # ── Stage 4: Post-processing ─────────────────────────────────
    TIMER.start("Stage 4 — Post-processing")
    sr = post_process(sr)
    TIMER.end()

    # Keep final exposure, color combination, and background lighting close
    # to original photoshoot image.
    sr = match_input_tonality(sr, img_bgr)
    sr = match_input_color_palette(sr, img_bgr)
    sr = preserve_background_look(sr, img_bgr)

    return sr


def main():
    input_path = Path(INPUT_IMAGE)
    stem       = input_path.stem
    device     = get_device()

    print("=" * 60)
    print("  Fashion Realism Pipeline  —  L40S / A100-40GB Edition")
    print(f"  Input     : {INPUT_IMAGE}")
    print(f"  Device    : {device}")
    print(f"  Stages    : SwinIR/HAT dynamic scale ({SR_UPSCALE_MODE}) → {FACE_BACKEND} face → body-skin refine → post-proc")
    print(f"  Outputs   : {stem}_realism_[1k|2k|4k|8k].png")
    print("=" * 60)

    if not input_path.exists():
        print(f"[!] Input not found: {INPUT_IMAGE}")
        sys.exit(1)

    img_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print("[!] Failed to read image.")
        sys.exit(1)

    h, w = img_bgr.shape[:2]
    print(f"[✓] Input loaded: {w}x{h}")

    sr = run_pipeline(img_bgr, device, MODEL_DIR)

    saved = []
    for label, target in get_enabled_outputs(sr):
        saved.append(save_resolution(sr, stem, label.upper(), target))

    print(f"\n{'=' * 60}")
    print(f"[✓] All outputs saved:")
    for p in saved:
        print(f"    -> {p}")
    print(f"[✓] Total time -> {_fmt(TIMER.total())}")
    print(f"{'=' * 60}\n")


# ─────────────────────────────────────────────
# 9. MODAL DEPLOYMENT
# ─────────────────────────────────────────────
try:
    import modal

    app            = modal.App("fashion-realism")
    weights_volume = modal.Volume.from_name("fashion-realism-weights", create_if_missing=True)
    WEIGHTS_PATH   = "/weights"   # populated once; reused across all requests
    hf_secret = modal.Secret.from_name("huggingface-secret")

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "libgl1", "libglib2.0-0")
        .pip_install(
            "torch", "torchvision",
            extra_options="--index-url https://download.pytorch.org/whl/cu121",
        )
        .run_commands(
            # basicsr from GitHub FIRST — PyPI version breaks with torchvision >= 0.16
            "pip install 'basicsr @ git+https://github.com/XPixelGroup/BasicSR.git'",
            # gfpgan + facexlib + timm with --no-deps to prevent basicsr downgrade
            "pip install timm facexlib gfpgan opencv-python-headless Pillow "
            "requests tqdm --no-deps",
            # runtime deps skipped by --no-deps; einops needed by HAT; boto3 for S3
            "pip install filterpy huggingface_hub safetensors einops boto3 "
            "diffusers transformers accelerate",
        )
        .run_commands(
            # SwinIR code (no weights — loaded from volume at runtime)
            "git clone https://github.com/JingyunLiang/SwinIR.git --depth 1",
            # HAT code only — weights come from volume, no requirements.txt install
            "git clone https://github.com/XPixelGroup/HAT.git --depth 1",
            # CodeFormer (fallback face model code)
            "git clone https://github.com/sczhou/CodeFormer.git /opt/CodeFormer --depth 1",
            "ln -s /opt/CodeFormer /usr/local/lib/python3.11/site-packages/codeformer",
        )
    )

    class _RealismRuntime:
        def __init__(self):
            self.device = None
            self.weights_dir = None
            self.models = {}

        def _ensure_weights(self):
            self.weights_dir.mkdir(parents=True, exist_ok=True)
            hf_cache_dir = self.weights_dir / "hf_cache"
            hf_cache_dir.mkdir(parents=True, exist_ok=True)

            # Download once into persistent volume if missing.
            swinir_path = self.weights_dir / "swinir_realsr_x4.pth"
            if not swinir_path.exists():
                download_file(SWINIR_URL, swinir_path)

            if USE_HAT:
                hat_path = self.weights_dir / "hat_real_gan_x4.pth"
                if not hat_path.exists():
                    download_hat(hat_path)

            # Face-restoration weights (download once into volume)
            if FACE_BACKEND in ("restoreformer", "codeformer"):
                rf_path = self.weights_dir / "RestoreFormerPlusPlus.ckpt"
                if not rf_path.exists():
                    download_file(RESTOREFORMER_URL, rf_path)
                cf_path = self.weights_dir / "codeformer.pth"
                if not cf_path.exists():
                    download_file(CODEFORMER_URL, cf_path)
            else:
                gfpgan_path = self.weights_dir / "GFPGANv1.4.pth"
                if not gfpgan_path.exists():
                    download_file(GFPGAN_URL, gfpgan_path)

            facelib_dir = self.weights_dir / "gfpgan_weights"
            facelib_dir.mkdir(parents=True, exist_ok=True)
            det_path = facelib_dir / "detection_Resnet50_Final.pth"
            parse_path = facelib_dir / "parsing_parsenet.pth"
            if not det_path.exists():
                download_file(FACEXLIB_DETECTION_URL, det_path)
            if not parse_path.exists():
                download_file(FACEXLIB_PARSING_URL, parse_path)

            # Persist newly downloaded files for future cold starts.
            weights_volume.commit()

            # GFPGAN expects this location by default.
            facelib_dst = Path("/root/gfpgan/weights")
            if not facelib_dst.exists() and facelib_dir.exists():
                facelib_dst.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(str(facelib_dir), str(facelib_dst))

            # CodeFormer/facexlib may download to package-local weights path.
            # Mirror persisted weights there so each container doesn't redownload.
            facexlib_pkg_dst = Path("/usr/local/lib/python3.11/site-packages/facexlib/weights")
            try:
                if not facexlib_pkg_dst.exists():
                    facexlib_pkg_dst.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(str(facelib_dir), str(facexlib_pkg_dst))
                else:
                    import shutil
                    for name in ("detection_Resnet50_Final.pth", "parsing_parsenet.pth"):
                        src_f = facelib_dir / name
                        dst_f = facexlib_pkg_dst / name
                        if src_f.exists() and not dst_f.exists():
                            shutil.copy2(src_f, dst_f)
            except Exception as e:
                print(f"[!] facexlib cache wiring skipped ({type(e).__name__}: {e})")

        def warmup(self):
            self.device = get_device()
            self.weights_dir = Path(WEIGHTS_PATH)
            hf_home = self.weights_dir / "hf_cache"
            os.environ["HF_HOME"] = str(hf_home)
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home)
            self._ensure_weights()

            swinir_path = self.weights_dir / "swinir_realsr_x4.pth"
            self.models["swinir"] = load_swinir(swinir_path, self.device)

            if USE_HAT:
                hat_path = self.weights_dir / "hat_real_gan_x4.pth"
                self.models["hat"] = load_hat(hat_path, self.device)
            else:
                self.models["hat"] = None

            if USE_BODY_SKIN_REFINE and str(BODY_SKIN_BACKEND).lower().strip() == "diffusion_retouch":
                self.models["skin_diffuser"] = load_body_skin_diffuser(
                    self.device, cache_dir=hf_home
                )
            else:
                self.models["skin_diffuser"] = None

            print("[✓] Warm container ready: models loaded once.")

        def infer(self, image_bytes: bytes, filename: str) -> dict:
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f"Could not decode input image: {filename}")

            h, w = img_bgr.shape[:2]
            print(f"[✓] Input: {w}x{h}")

            sr = run_pipeline(
                img_bgr,
                self.device,
                self.weights_dir,
                preloaded_models=self.models,
            )

            outputs = {}
            for label, target in get_enabled_outputs(sr):
                resized = resize_to_target(sr, target)
                ok, enc = cv2.imencode(".png", resized, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if not ok:
                    raise RuntimeError(f"PNG encode failed for {label}")
                outputs[label] = enc.tobytes()
                rh, rw = resized.shape[:2]
                print(f"[✓] Prepared {label.upper()} ({rw}x{rh})")
            return outputs

    @app.cls(
        image=image,
        gpu="L40S",
        timeout=600,
        memory=49152,   # 48 GB VRAM — matches L40S physical memory
        secrets=[hf_secret],
        volumes={WEIGHTS_PATH: weights_volume},
        # Auto-scale: spin up to 10 L40S containers when demand is high,
        # scale back to 0 when idle (cold-start ~15-25s).
        concurrency_limit=10,
        # Each container processes one image at a time (GPU-bound workload).
        allow_concurrent_inputs=1,
    )
    class FashionRealismT4:
        """Primary GPU class — runs on L40S (48 GB, Ada Lovelace).
        Class name kept as FashionRealismT4 for backward compatibility with
        existing service code that calls it by name via modal.Cls.from_name().
        """
        @modal.enter()
        def load(self):
            self.rt = _RealismRuntime()
            self.rt.warmup()

        @modal.method()
        def enhance(self, image_bytes: bytes, filename: str) -> dict:
            return self.rt.infer(image_bytes, filename)

    @app.cls(
        image=image,
        gpu="A100-40GB",
        timeout=600,
        memory=40960,   # 40 GB VRAM — matches A100-40GB physical memory
        secrets=[hf_secret],
        volumes={WEIGHTS_PATH: weights_volume},
        # Auto-scale: spin up to 5 A100-40GB containers as fallback.
        concurrency_limit=5,
        allow_concurrent_inputs=1,
    )
    class FashionRealismL4:
        """Fallback GPU class — runs on A100-40GB (40 GB, Ampere).
        Class name kept as FashionRealismL4 for backward compatibility with
        existing service code that calls it by name via modal.Cls.from_name().
        """
        @modal.enter()
        def load(self):
            self.rt = _RealismRuntime()
            self.rt.warmup()

        @modal.method()
        def enhance(self, image_bytes: bytes, filename: str) -> dict:
            return self.rt.infer(image_bytes, filename)

    @app.local_entrypoint()
    def run():
        inp = Path(INPUT_IMAGE)
        if not inp.exists():
            print(f"[!] Input not found: {INPUT_IMAGE}")
            return

        print(f"[->] Uploading {INPUT_IMAGE} to Modal (L40S preferred, A100-40GB fallback) ...")
        print(f"     Pipeline: SwinIR/HAT dynamic scale ({SR_UPSCALE_MODE}) → {FACE_BACKEND} face → body-skin refine → post-proc")
        print(f"     (weights persist in modal.Volume: fashion-realism-weights)")
        print(f"     (models are loaded once per warm container)")

        payload = inp.read_bytes()
        try:
            print("[→] Trying L40S ...")
            outputs = FashionRealismT4().enhance.remote(payload, inp.name)
            gpu_used = "L40S"
        except Exception as t4_err:
            print(f"[!] L40S failed ({type(t4_err).__name__}: {t4_err})")
            print("[→] Falling back to A100-40GB ...")
            outputs = FashionRealismL4().enhance.remote(payload, inp.name)
            gpu_used = "A100-40GB"

        print("\n" + "=" * 60)
        print(f"  ✓ Enhancement complete on {gpu_used}")
        print("=" * 60)
        stem = inp.stem
        for label in outputs.keys():
            out_name = f"{stem}_realism_{label}.png"
            Path(out_name).write_bytes(outputs[label])
            print(f"  {label.upper()}  →  {out_name}")
        print("=" * 60 + "\n")

except ImportError:
    pass   # Modal not installed — local run still works fine


if __name__ == "__main__":
    main()
