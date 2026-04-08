"""
Single-Garment Photoshoot Generator — SeedDream 4.5-edit API (kie.ai)

Flow:
  1. Read local image files and upload them to S3 to get public URLs:
       - PNG / JPEG / JPG  → uploaded to S3 as-is
       - Any other format  → converted to PNG with Pillow, then uploaded to S3
  2. Build the ordered image_urls list (front, optional back, face, background)
  3. For each pose: submit a task, poll until done, download & save the image
  4. Optionally deblur each saved image (GFPGAN + Real-ESRGAN) — saves both
     the original and a *_sharp.png alongside it
  5. Print a summary

Requirements:
    pip install requests boto3 Pillow
"""

import datetime
import io
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import requests
from PIL import Image


# ── DEBLUR CONFIG ─────────────────────────────────────────────────────────────
# After each image is generated, automatically run GFPGAN + Real-ESRGAN
# deblurring. Both the original and the sharpened file are saved.

ENABLE_DEBLUR     = True   # set False to skip deblurring entirely
DEBLUR_UPSCALE    = 2      # 1 = same size, 2 = 2× output resolution
DEBLUR_FULL_IMAGE = True   # True = full body+bg via Real-ESRGAN | False = face only (faster)
DEBLUR_MAX_SIZE   = None   # disabled — process at full original resolution for best quality


# ── API CONFIG ────────────────────────────────────────────────────────────────

API_KEY      = os.getenv("SEEDDREAM_API_KEY", "")
CREATE_URL   = "https://api.kie.ai/api/v1/jobs/createTask"
STATUS_URL   = "https://api.kie.ai/api/v1/jobs/recordInfo"
MODEL        = "seedream/5-lite-image-to-image"
QUALITY      = "high"          # "low" / "medium" / "high"
ASPECT_RATIO = "9:16"          # portrait — change to "1:1", "16:9" etc. if needed

MAX_RETRIES  = 120             # max status-poll attempts per pose
RETRY_DELAY  = 5               # seconds between polls

RUN_ID       = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
OUTPUT_DIR   = f"./photoshoot_output/{RUN_ID}"


# ── S3 CONFIG ─────────────────────────────────────────────────────────────────
# All local images are uploaded here first so the SeedDream API gets public URLs.

AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
S3_BUCKET             = "aavishailabs-uploads-prod"
S3_REGION             = "eu-north-1"
S3_PREFIX             = "seeddream-inputs"            # folder prefix inside the bucket
# Set to True to grant public-read ACL (requires the bucket to allow it);
# set to False if your bucket uses a public-read bucket policy instead.
S3_PUBLIC_ACL = False


# ── INPUT IMAGES (local file paths) ───────────────────────────────────────────
# Provide absolute or relative paths to the images on your machine.
# Supported natively: PNG, JPG, JPEG — any other format is auto-converted to PNG.

GARMENT_FRONT_PATH = "garment_31.jpeg"    # garment front — required
GARMENT_BACK_PATH  = ""         # garment back  — optional, leave "" to skip
MODEL_FACE_PATH    = "face3.png" # model face    — required
BACKGROUND_PATH    = "background2.png"   # background    — required


# ── MODEL CONFIGURATION ───────────────────────────────────────────────────────

model_config = {
    "ethnicity":      "Indian",         # e.g. Indian, Caucasian, East Asian, African …
    "gender":         "Female",         # Male / Female
    "skin_tone":      "Lighter tone",   # Lighter tone / Medium tone / Darker tone
    "age":            "25 years old",
    "age_group":      "Adult",          # Teen / Young Adult / Adult / Middle-aged / Senior
    "weight":         "Slim",           # Slim / Regular / Fat
    "height":         "Regular",        # Short / Regular / Tall
    "lighting_style": "natural light",  # natural light / studio light / golden hour …
    "ornaments":      "Watch, Necklace, Earrings",
}


# ── GARMENT CONFIGURATION ─────────────────────────────────────────────────────
# Fill upper + lower OR one-piece. Leave unused fields as "".

garment_config = {
    "upper_garment_type":      "top",  # shirt / top / blazer / kurta … (or "" to omit)
    "upper_garment_specs":     "",             # extra detail e.g. "sleeveless, floral print"
    "lower_garment_type":      "jeans",        # jeans / palazzo / dhoti …      (or "" to omit)
    "lower_garment_specs":     "Blue jeans",   # extra detail
    "one_piece_garment_type":  "",             # saree / dress / jumpsuit …     (or "" to omit)
    "one_piece_garment_specs": "",
    "fitting":                 "regular fit",  # slim fit / regular fit / oversized …
}


# ── POSES ─────────────────────────────────────────────────────────────────────
# One image will be generated per pose. Add or remove as needed.

# ── Shared poses ──────────────────────────────────────────────────────────────
poses = [
    "Standing upright facing directly forward, weight evenly distributed, arms relaxed slightly away from the body, chin level, looking straight into the camera with a neutral confident expression, full body visible from head to toe"
    
    # "Standing facing forward, upper body angled very slightly toward camera, one hand gently resting at waist, close-up framed from chest to thigh, relaxed posture, face softly out of frame or slightly tilted down",
    
    # "Standing with back fully facing the camera, feet shoulder-width apart, arms naturally at sides, head slightly turned over one shoulder with a subtle look back, full body visible from head to toe",
    
    # "Back facing the camera, torso framed from shoulder blades to lower hip, both arms hanging naturally, slight body twist to create dimension, close-up crop",
    
    # "Standing in a true 90-degree side profile, one foot slightly in front of the other, chin up, arms softly resting at sides, full body from head to toe visible, relaxed and elongated posture",
    
    # "Three-quarter side angle, torso turned 45 degrees from camera, one arm raised slightly or hand on hip to open the silhouette, crop from neck to upper thigh, slight chin tilt",
    
    # "Standing with arms gently extended forward and slightly outward at waist height, palms facing upward, body facing camera directly, fabric naturally falling and draping, slight head tilt downward toward the fabric",
    
    # "Seated at a minimal table, body angled 45 degrees toward camera, one elbow resting on table with hand under chin in a thoughtful pose, other hand resting flat on table near a coffee cup and small book, legs crossed, relaxed and editorial expression"
]


# ── HELPERS ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── IMAGE UPLOAD / FORMAT NORMALISATION ──────────────────────────────────────

# Extensions the SeedDream API accepts natively.
_SUPPORTED_EXTS = {"png", "jpg", "jpeg"}

# MIME type map for supported formats.
_MIME_TYPES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}


def _s3_client():
    """Return a boto3 S3 client using the hardcoded credentials."""
    return boto3.client(
        "s3",
        region_name=S3_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def _upload_bytes_to_s3(data: bytes, s3_key: str, content_type: str) -> str:
    """Upload raw bytes to S3 and return the public HTTPS URL."""
    put_kwargs: dict = {
        "Bucket":      S3_BUCKET,
        "Key":         s3_key,
        "Body":        data,
        "ContentType": content_type,
    }
    if S3_PUBLIC_ACL:
        put_kwargs["ACL"] = "public-read"
    _s3_client().put_object(**put_kwargs)
    return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"


def normalize_local_path(path: str, label: str) -> str:
    """
    Read a local image file, upload it to S3, and return the public URL.

    - PNG / JPEG / JPG  → uploaded to S3 as-is (no re-encoding)
    - Any other format  → converted to PNG with Pillow, then uploaded to S3
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[{label}] File not found: {path}")

    _, raw_ext = os.path.splitext(path)
    ext = raw_ext.lstrip(".").lower()

    with open(path, "rb") as f:
        file_bytes = f.read()

    if ext in _SUPPORTED_EXTS:
        # Supported format — upload directly without re-encoding.
        s3_key      = f"{S3_PREFIX}/{uuid.uuid4()}.{ext}"
        content_type = _MIME_TYPES[ext]
        log(f"  [{label}] '{ext}' is supported — uploading as-is…")
        public_url = _upload_bytes_to_s3(file_bytes, s3_key, content_type)
    else:
        # Unsupported format — convert to PNG with Pillow first.
        img = Image.open(io.BytesIO(file_bytes))
        detected = img.format or "unknown"
        log(f"  [{label}] Format '{ext or detected}' not supported — converting to PNG…")

        if img.mode in ("P", "LA"):
            img = img.convert("RGBA")
        elif img.mode != "RGBA":
            img = img.convert("RGB")

        png_buf = io.BytesIO()
        img.save(png_buf, format="PNG")

        s3_key     = f"{S3_PREFIX}/{uuid.uuid4()}.png"
        public_url = _upload_bytes_to_s3(png_buf.getvalue(), s3_key, "image/png")

    log(f"  [{label}] Uploaded → {public_url}")
    return public_url


def build_image_urls() -> list[str]:
    """
    Upload every local input image to S3 (converting unsupported formats to
    PNG when necessary) and return the ordered list of public URLs for the model.
    """
    log("Step 1 — Uploading reference images to S3…")
    front = normalize_local_path(GARMENT_FRONT_PATH, "garment-front")
    back  = normalize_local_path(GARMENT_BACK_PATH,  "garment-back") if GARMENT_BACK_PATH else None
    face  = normalize_local_path(MODEL_FACE_PATH,    "model-face")
    bg    = normalize_local_path(BACKGROUND_PATH,    "background")

    urls = [front]
    if back:
        urls.append(back)
    urls.append(face)
    urls.append(bg)
    return urls


def build_prompt(pose: str) -> str:
    """Build the full generation prompt for a single pose."""
    mc       = model_config
    gc       = garment_config
    has_back = bool(GARMENT_BACK_PATH)

    if has_back:
        image_ref_block = (
            "You are provided with FOUR reference images:\n"
            "  IMAGE 1 — GARMENT FRONT IMAGE: the exact outfit/garment front view.\n"
            "  IMAGE 2 — GARMENT BACK IMAGE: the exact outfit/garment back view.\n"
            "  IMAGE 3 — MODEL FACE IMAGE: the exact face the model must have.\n"
            "  IMAGE 4 — BACKGROUND IMAGE: the exact background scene to be used."
        )
        garment_ref_note = (
            "- USE THE EXACT SAME GARMENT from IMAGE 1 and IMAGE 2 (both front and back views) — this is MANDATORY.\n"
            "- Reproduce every garment detail from BOTH views with 100% accuracy."
        )
    else:
        image_ref_block = (
            "You are provided with THREE reference images:\n"
            "  IMAGE 1 — GARMENT IMAGE: the exact outfit/garment to be worn.\n"
            "  IMAGE 2 — MODEL FACE IMAGE: the exact face the model must have.\n"
            "  IMAGE 3 — BACKGROUND IMAGE: the exact background scene to be used."
        )
        garment_ref_note = (
            "- USE THE EXACT SAME GARMENT from IMAGE 1 (garment image) — this is MANDATORY.\n"
            "- Reproduce every garment detail with 100% accuracy."
        )

    return f"""{image_ref_block}

Generate a photorealistic fashion photoshoot image. Zero deviation from reference images.

[FACE — DO NOT CHANGE]
Use EXACT face from model face reference. Match: face shape, eyes, nose, lips, skin tone, eyebrows, hair. Do NOT alter or replace. FACE VALIDATION: verify face matches reference before finalising — regenerate if any deviation.
Model: {mc["gender"]}, {mc["ethnicity"]}, {mc["age"]} ({mc["age_group"]}), {mc["weight"]} build, {mc["height"]}, {mc["skin_tone"]} skin.

[GARMENT — DO NOT CHANGE]
{garment_ref_note}
Match: fabric texture, exact colors, patterns, seams, all design details. No modifications.
- Upper: {gc["upper_garment_type"]}{f' — {gc["upper_garment_specs"]}' if gc.get("upper_garment_specs") else ''}
- Lower: {gc["lower_garment_type"]}{f' — {gc["lower_garment_specs"]}' if gc.get("lower_garment_specs") else ''}
- One-piece: {gc["one_piece_garment_type"]}{f' — {gc["one_piece_garment_specs"]}' if gc.get("one_piece_garment_specs") else ''}
- Fitting: {gc["fitting"]}

[BACKGROUND — DO NOT CHANGE]
Use EXACT background from reference. Match: all objects, colors, lighting, shadows, depth. No alterations.

[POSE] {pose}

[STYLE] Lighting: {mc["lighting_style"]} | Ornaments: {mc["ornaments"]}

[QUALITY] Ultra-high resolution, photorealistic, professional fashion photography, sharp focus, seamless model-background integration, commercial e-commerce grade.

NON-NEGOTIABLE: 1. Face = 100% identical to reference. 2. Garment = 100% identical to reference. 3. Background = 100% identical to reference."""


def submit_task(prompt: str, image_urls: list[str]) -> str:
    """Submit a task to the SeedDream API and return the task ID."""
    payload = json.dumps({
        "model": MODEL,
        "input": {
            "prompt":       prompt,
            "image_urls":   image_urls,
            "aspect_ratio": ASPECT_RATIO,
            "quality":      QUALITY,
        }
    })
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    resp = requests.post(CREATE_URL, headers=headers, data=payload, timeout=30)
    resp.raise_for_status()
    print(resp.json())
    task_id = resp.json().get("data", {}).get("taskId")
    if not task_id:
        raise ValueError(f"No taskId in response: {resp.text}")
    return task_id


def poll_task(task_id: str, pose_label: str):  # -> str | None
    """
    Poll the SeedDream API until the task succeeds or fails.
    Returns the result image URL on success, or None on failure.
    """
    headers = {"Authorization": f"Bearer {API_KEY}"}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                f"{STATUS_URL}?taskId={task_id}",
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            data   = resp.json().get("data", {})
            status = data.get("state")

            if status == "success":
                result_json = json.loads(data.get("resultJson", "{}"))
                result_urls = result_json.get("resultUrls", [])
                if result_urls:
                    log(f"  [{pose_label}] Done — result URL: {result_urls[0][:80]}…")
                    return result_urls[0]
                log(f"  [{pose_label}] Success but no resultUrls found.")
                return None

            elif status == "fail":
                log(f"  [{pose_label}] Task failed.")
                return None

            else:
                log(f"  [{pose_label}] Poll #{attempt:03d} — status: {status}")
                time.sleep(RETRY_DELAY)

        except Exception as exc:
            log(f"  [{pose_label}] Poll #{attempt:03d} error: {exc}")
            time.sleep(RETRY_DELAY)

    log(f"  [{pose_label}] Max retries reached — giving up.")
    return None


def download_image(url: str, out_path: str, retries: int = 3) -> None:
    """Download an image from a URL and save it to out_path.

    Uses a (connect=10s, read=30s) timeout tuple so a stalled CDN stream
    raises ReadTimeout instead of hanging forever. Retries up to `retries` times.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, stream=True, timeout=(10, 30))
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            log(f"  Saved: {out_path}")
            return
        except Exception as exc:
            log(f"  Download attempt {attempt}/{retries} failed: {exc}")
            if attempt == retries:
                raise
            time.sleep(3)


# ── DEBLUR HELPERS ────────────────────────────────────────────────────────────

def _setup_deblur():
    """
    Install dependencies, apply the torchvision compatibility shim, download
    model weights, and return an initialised (GFPGANer, bg_upsampler) tuple.
    Called once before the pose loop so models are loaded only one time.
    """
    import importlib
    import subprocess
    import sys
    import urllib.request

    # Auto-install missing packages
    for pkg, mod in [
        ("gfpgan", "gfpgan"), ("basicsr", "basicsr"),
        ("facexlib", "facexlib"), ("realesrgan", "realesrgan"),
    ]:
        try:
            importlib.import_module(mod)
        except ImportError:
            log(f"[deblur] Installing {pkg} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

    # torchvision >= 0.16 removed functional_tensor; basicsr still imports it
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        import torchvision.transforms.functional as _ftf
        sys.modules["torchvision.transforms.functional_tensor"] = _ftf

    from gfpgan import GFPGANer

    _weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

    def _dl(url, dest):
        if os.path.exists(dest):
            return
        os.makedirs(_weights_dir, exist_ok=True)
        log(f"[deblur] Downloading {os.path.basename(dest)} (once) ...")
        urllib.request.urlretrieve(url, dest)
        log(f"[deblur] Saved → {dest}")

    gfpgan_path = os.path.join(_weights_dir, "GFPGANv1.4.pth")
    _dl("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        gfpgan_path)

    bg_upsampler = None
    if DEBLUR_FULL_IMAGE:
        esrgan_path = os.path.join(_weights_dir, "RealESRGAN_x2plus.pth")
        _dl("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            esrgan_path)

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        bg_upsampler = RealESRGANer(
            scale=2,
            model_path=esrgan_path,
            model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                          num_block=23, num_grow_ch=32, scale=2),
            tile=512, tile_pad=16, pre_pad=0, half=False, device="cpu",
        )
        log("[deblur] Real-ESRGAN loaded (body + background)")

    restorer = GFPGANer(
        model_path=gfpgan_path,
        upscale=DEBLUR_UPSCALE,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=bg_upsampler,
        device="cpu",
    )
    log("[deblur] GFPGAN loaded (face restoration)")
    return restorer


def _run_deblur(restorer, original_path: str):  # -> str | None
    """
    Deblur *original_path* using a pre-built restorer.
    Saves the sharp result as *_sharp.png next to the original.
    Returns the path to the sharp file, or None on error.
    """
    import cv2
    import numpy as np

    try:
        img = cv2.imread(original_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cv2 could not read {original_path}")

        h, w = img.shape[:2]
        log(f"[deblur] Restoring {w}×{h} image ...")
        _, _, sharp = restorer.enhance(
            img, has_aligned=False, only_center_face=False,
            paste_back=True, weight=0.5,
        )

        base, _ = os.path.splitext(original_path)
        sharp_path = f"{base}_sharp.png"
        cv2.imwrite(sharp_path, sharp)
        log(f"[deblur] Sharp image saved → {sharp_path}")
        return sharp_path

    except Exception as exc:
        log(f"[deblur] ERROR: {exc}")
        return None


# ── PARALLEL POSE WORKER ──────────────────────────────────────────────────────

def _process_pose(
    pose_idx:       int,
    pose:           str,
    image_urls:     list[str],
    deblur_restorer,
    deblur_lock,            # threading.Lock or None
) -> dict:
    """
    Worker executed in a thread for one pose:
      submit task → poll until done → download → deblur (serialised via lock).
    Returns a result dict consumed by main().
    """
    pose_label = f"pose-{pose_idx:02d}"
    result = {"label": pose_label, "original": None, "sharp": None, "failed": False}

    prompt = build_prompt(pose)
    print("completed")

    # Submit
    try:
        task_id = submit_task(prompt, image_urls)
        log(f"  [{pose_label}] Task submitted — ID: {task_id}")
    except Exception as exc:
        log(f"  [{pose_label}] ERROR submitting: {exc}")
        result["failed"] = True
        return result

    # Poll
    result_url = poll_task(task_id, pose_label)
    if not result_url:
        result["failed"] = True
        return result

    # Download original
    out_file = f"{OUTPUT_DIR}/{pose_label}_{RUN_ID}.png"
    try:
        log(f"  [{pose_label}] Downloading image ...")
        download_image(result_url, out_file)
        result["original"] = out_file
        log(f"  [{pose_label}] ✓ Original saved → {out_file}")
    except Exception as exc:
        log(f"  [{pose_label}] ERROR downloading: {exc}  |  URL: {result_url}")
        result["failed"] = True
        return result

    # Deblur — serialised so only one pose runs CPU inference at a time
    if deblur_restorer and deblur_lock is not None:
        log(f"  [{pose_label}] Queued for deblur (waiting for CPU slot) ...")
        with deblur_lock:
            log(f"  [{pose_label}] Deblurring started (this takes ~30-90s on CPU) ...")
            sharp = _run_deblur(deblur_restorer, out_file)
            if sharp:
                result["sharp"] = sharp
                log(f"  [{pose_label}] ✓ Deblur complete → {sharp}")
            else:
                log(f"  [{pose_label}] ✗ Deblur failed (original still saved)")

    return result


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    overall_start = time.time()
    has_back      = bool(GARMENT_BACK_PATH)

    log("=" * 65)
    log(f"  SeedDream Photoshoot — Run ID : {RUN_ID}")
    log(f"  Model         : {MODEL}")
    log(f"  Quality       : {QUALITY}  |  Aspect ratio: {ASPECT_RATIO}")
    log(f"  Poses         : {len(poses)}")
    log(f"  Garment back  : {'yes' if has_back else 'no (front only)'}")
    log(f"  Deblur        : {'enabled' if ENABLE_DEBLUR else 'disabled'}")
    log("=" * 65)

    # Normalise all input URLs (converts unsupported formats → PNG on S3).
    image_urls = build_image_urls()

    log(f"\n  Reference images ready ({len(image_urls)}):")
    for i, url in enumerate(image_urls, 1):
        log(f"    [{i}] {url[:90]}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialise deblur pipeline once (loading models is expensive)
    deblur_restorer = None
    if ENABLE_DEBLUR:
        log("\nInitialising deblur pipeline (GFPGAN + Real-ESRGAN) ...")
        deblur_restorer = _setup_deblur()

    saved_images:        list[str] = []
    saved_sharp_images:  list[str] = []
    failed_poses:        list[str] = []

    # One lock so only one pose runs CPU deblur inference at a time
    deblur_lock = threading.Lock() if deblur_restorer else None

    log(f"\nLaunching {len(poses)} pose(s) in parallel ...")

    with ThreadPoolExecutor(max_workers=len(poses)) as executor:
        futures = {
            executor.submit(
                _process_pose, idx, pose, image_urls, deblur_restorer, deblur_lock
            ): idx
            for idx, pose in enumerate(poses, 1)
        }

        for future in as_completed(futures):
            try:
                res = future.result()
            except Exception as exc:
                pose_idx = futures[future]
                log(f"  [pose-{pose_idx:02d}] Unhandled error: {exc}")
                failed_poses.append(f"pose-{pose_idx:02d}")
                continue

            if res["failed"]:
                failed_poses.append(res["label"])
            if res["original"]:
                saved_images.append(res["original"])
            if res["sharp"]:
                saved_sharp_images.append(res["sharp"])

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - overall_start
    elapsed_str   = str(datetime.timedelta(seconds=int(total_elapsed)))

    log("\n" + "=" * 65)
    log("  SUMMARY")
    log("=" * 65)
    log(f"  Run ID              : {RUN_ID}")
    log(f"  Model               : {MODEL}")
    log(f"  Poses submitted     : {len(poses)}")
    log(f"  Images saved        : {len(saved_images)}")
    log(f"  Sharp images saved  : {len(saved_sharp_images)}")
    log(f"  Failed poses        : {len(failed_poses)}")
    log(f"  Output folder       : {OUTPUT_DIR}/")
    log(f"  Total elapsed       : {elapsed_str}")
    log("=" * 65)

    if saved_images:
        log("\nGenerated images (original):")
        for img in saved_images:
            log(f"  • {img}")

    if saved_sharp_images:
        log("\nDeblurred images (sharp):")
        for img in saved_sharp_images:
            log(f"  • {img}")

    if failed_poses:
        log(f"\nFailed poses: {', '.join(failed_poses)}")


if __name__ == "__main__":
    main()
