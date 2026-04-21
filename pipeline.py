"""
Garment photoshoot multi-agent pipeline (LangGraph) — single-file implementation.

Architecture:

  Stage 1  GENERATION (chain of fallbacks, per provider retries):
      1) KIE.ai  nano-banana-2        (max_retries 2, concurrency 10, error_retries 2)
      2) Vertex  nano-banana-2 direct (max_retries 4, concurrency 2,  error_retries 4)
      3) Vertex  nano-banana-pro      (max_retries 4, concurrency 2,  error_retries 4)
      4) Evolink nano-banana-2        (max_retries 2, concurrency 10, error_retries 10)
    A provider is tried for ALL poses. Any pose that is still missing after the
    provider's retries falls through to the next provider.

  Stage 2  UPSCALING (chain of fallbacks):
      1) Topaz via KIE   (max_retries 5, concurrency 10, error_retries 5)
      2) Topaz via fal   (max_retries 5, concurrency 10, error_retries 5)

Pipeline implementation:
  Each provider (and each upscaler) is an agent node. A supervisor-style graph
  routes work between GENERATE -> UPSCALE. Per-pose state is tracked
  inside `PipelineState.poses` so nodes remain pure functions over state.

Env vars required (only for the providers you actually want to run):
  KIE_API_KEY              KIE.ai
  EVOLINK_API_KEY          Evolink.ai
  GOOGLE_CLOUD_API_KEY     Vertex AI (nano-banana-2/pro direct) — also used as
                           fallback auth for google-genai
  FAL_KEY                  fal.ai Topaz fallback

Install:
  pip install -U \
      langgraph \
      google-genai \
      pillow \
      requests \
      fal-client

Run:
  python pipeline_langgraph/pipeline.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import mimetypes
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

import requests

# Optional imports — only required when the corresponding provider runs.
try:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore
    genai_types = None  # type: ignore

try:
    import fal_client  # type: ignore
except Exception:  # pragma: no cover
    fal_client = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

from langgraph.graph import END, StateGraph


# =============================================================================
# .env loader (no external dependency required)
# =============================================================================
def _load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_dotenv_file(Path(__file__).resolve().parents[1] / ".env")


# =============================================================================
# CONFIG — edit top-of-file inputs; everything else flows from state.
# =============================================================================
OUTPUT_BASE_DIR = Path("pipeline_langgraph/outputs")
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:8]}"
RUN_OUTPUT_DIR = OUTPUT_BASE_DIR / f"run_{RUN_ID}"
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Stage-2 (upscaler)
REQUIRED_UPSCALE = False
UPSCALE_FACTOR_KIE = "2"
UPSCALE_MODEL_FAL = "High Fidelity V2"
UPSCALE_FACTOR_FAL = 2.0
UPSCALE_OUTPUT_FORMAT = "png"

# Timeouts / polling
PER_IMAGE_TIMEOUT_S = 600.0           # wall-clock per pose generation attempt
UPSCALE_PER_IMAGE_TIMEOUT_S = 1200.0  # wall-clock per upscale attempt
KIE_POLL_INTERVAL_S = 5.0

# --- INPUTS -----------------------------------------------------------------
FRONT_GARMENT_URL = "https://pub-51c3a7dccc2448f792c2fb1bacf8e05d.r2.dev/users/5971e90a-2682-4c24-a16e-2eda4162f4e8/029eae283e7e463eaf47bd6d6bcfefd4.webp"
BACK_GARMENT_URL = ""
BACKGROUND_URL = "https://aavishailabs-uploads-prod.s3.eu-north-1.amazonaws.com/backgrounds/terracotta_seamless_studio_40d28ca0.png"
MODEL_FACE_URL = "https://aavishailabs-uploads-prod.s3.eu-north-1.amazonaws.com/new_model_faces/268.jpg"

# POSES_MANNEQUIN_URLS: list[str] = [
#     "https://pub-51c3a7dccc2448f792c2fb1bacf8e05d.r2.dev/mannequin-output/052d007f-b7b5-4d3b-a4ff-621a8fd364ca.png",
#     "https://pub-51c3a7dccc2448f792c2fb1bacf8e05d.r2.dev/mannequin-output/4c2619a7-964e-4b42-97cb-45c93b732896.png",
#     "https://pub-51c3a7dccc2448f792c2fb1bacf8e05d.r2.dev/mannequin-output/fa728e4f-41e6-4292-b455-356ad9f3ab9d.png",
#     "https://pub-51c3a7dccc2448f792c2fb1bacf8e05d.r2.dev/mannequin-output/d91faceb-8d34-4408-aa25-a6ddaadbedf1.png",
#     "https://aavishailabs-uploads-prod.s3.eu-north-1.amazonaws.com/mannequin-output/0ddadd8c-5258-465e-a55a-47dcefb6bb80.png",
#     "https://aavishailabs-uploads-prod.s3.eu-north-1.amazonaws.com/mannequin-output/94cbdbd4-1c2e-43d5-8005-2358a44cde5b.png",
#     "https://aavishailabs-uploads-prod.s3.eu-north-1.amazonaws.com/mannequin-output/3bc34ae1-b828-4f28-badd-5dfee3c3ee2d.png",
# ]

POSES_MANNEQUIN_URLS: list[str] =[
    "https://aavishailabs-uploads-prod.s3.eu-north-1.amazonaws.com/mannequin-output/12c8c2c1-249f-4152-ae03-284ee6d5cff8.png",
    "https://aavishailabs-uploads-prod.s3.eu-north-1.amazonaws.com/mannequin-output/0ddadd8c-5258-465e-a55a-47dcefb6bb80.png",
    "https://aavishailabs-uploads-prod.s3.eu-north-1.amazonaws.com/mannequin-output/94cbdbd4-1c2e-43d5-8005-2358a44cde5b.png",
    "https://aavishailabs-uploads-prod.s3.eu-north-1.amazonaws.com/mannequin-output/d5458658-c0d7-459d-90cd-c9cf722ce9b2.png",
    "https://aavishailabs-uploads-prod.s3.eu-north-1.amazonaws.com/mannequin-output/a3f9ab99-f47c-43b5-aa4c-cee91181b6ac.png",
    "https://aavishailabs-uploads-prod.s3.eu-north-1.amazonaws.com/mannequin-output/80d9ffbe-6b15-4b6f-b33c-14696ec97f25.png",
    "https://aavishailabs-uploads-prod.s3.eu-north-1.amazonaws.com/mannequin-output/b5deda7f-14d4-4c75-b45b-87bd13748065.png"
]

WEIGHT = "Fat"
HEIGHT = "short"
GENDER = "female"

UPPER_GARMENT_TYPE = ""
UPPER_GARMENT_SPEC = ""
LOWER_GARMENT_TYPE = ""
LOWER_GARMENT_SPEC = ""
ONE_PIECE_GARMENT_TYPE = ""
ONE_PIECE_GARMENT_SPEC = ""
FITTING = ""

ASPECT_RATIO = "9:16"
RESOLUTION = "4K"


# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=os.environ.get("PIPELINE_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


@dataclass
class TimingEvent:
    name: str
    category: str
    start_epoch_s: float
    end_epoch_s: float
    duration_s: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimelineRecorder:
    events: list[TimingEvent] = field(default_factory=list)
    run_start_epoch_s: float = field(default_factory=time.time)

    def start(self, name: str, category: str, meta: dict[str, Any] | None = None) -> float:
        t0 = time.perf_counter()
        log.info("START %-20s | %s | meta=%s", category, name, meta or {})
        return t0

    def end(
        self,
        t0: float,
        name: str,
        category: str,
        meta: dict[str, Any] | None = None,
    ) -> float:
        duration_s = time.perf_counter() - t0
        now_epoch_s = time.time()
        self.events.append(
            TimingEvent(
                name=name,
                category=category,
                start_epoch_s=now_epoch_s - duration_s,
                end_epoch_s=now_epoch_s,
                duration_s=duration_s,
                meta=meta or {},
            )
        )
        log.info("END   %-20s | %s | took=%.2fs", category, name, duration_s)
        return duration_s


TIMELINE = TimelineRecorder()


# =============================================================================
# PROVIDER POLICIES (from pipeline-making.py)
# =============================================================================
@dataclass
class GenPolicy:
    name: str
    concurrency: int
    retries: int  # both "max retries" and "any error retries" from your spec


GEN_POLICIES: list[GenPolicy] = [
    GenPolicy(name="kie_nb2",       concurrency=10, retries=2),
    GenPolicy(name="vertex_nb2",    concurrency=2,  retries=4),
    GenPolicy(name="vertex_nbpro",  concurrency=2,  retries=4),
    GenPolicy(name="evolink_nb2",   concurrency=10, retries=10),
]

UPSCALE_POLICIES: list[GenPolicy] = [
    GenPolicy(name="kie_topaz", concurrency=10, retries=5),
    GenPolicy(name="fal_topaz", concurrency=10, retries=5),
]


# =============================================================================
# STATE
# =============================================================================
@dataclass
class PoseState:
    index: int
    pose_url: str
    # lifecycle
    generated_path: Path | None = None
    upscaled_path: Path | None = None
    # provider bookkeeping
    generator_attempts: list[str] = field(default_factory=list)
    failed: bool = False
    last_error: str | None = None

    @property
    def is_back(self) -> bool:
        return "back" in (self.pose_url or "").lower()


@dataclass
class PipelineState:
    poses: dict[int, PoseState] = field(default_factory=dict)
    # pointers into GEN_POLICIES / UPSCALE_POLICIES
    gen_stage_index: int = 0
    upscale_stage_index: int = 0
    finished: bool = False

    def pending_generation(self) -> list[PoseState]:
        return [p for p in self.poses.values() if p.generated_path is None and not p.failed]

    def pending_upscale(self) -> list[PoseState]:
        return [
            p for p in self.poses.values()
            if p.generated_path is not None
            and p.upscaled_path is None
            and not p.failed
        ]


# =============================================================================
# PROMPT BUILDERS (shared by KIE/Evolink/Vertex agents)
# =============================================================================
def _body_description() -> str:
    g, w, h = GENDER.lower(), WEIGHT.lower(), HEIGHT.lower()
    if g == "male":
        body = {
            "fat":  "a heavier, fuller male build with natural belly volume and broader torso",
            "slim": "a slim, lean male build with narrow waist",
        }.get(w, "a muscular, athletic male build with defined shoulders and chest")
    else:
        body = {
            "fat":  "a fuller, curvier female build with natural body volume",
            "slim": "a slim, slender female build with delicate frame",
        }.get(w, "a regular, healthy female build with natural proportions")
    stature = {
        "tall":  "tall stature (approx 160-170 cm)",
        "short": "short stature (approx 140-150 cm)",
    }.get(h, "average stature (approx 150-160 cm)")
    return f"{body}, {stature}"


def _garment_description() -> str:
    parts: list[str] = []
    if ONE_PIECE_GARMENT_TYPE:
        parts.append(
            f"A one-piece garment: {ONE_PIECE_GARMENT_TYPE} — {ONE_PIECE_GARMENT_SPEC}. "
            "Reproduce fabric, weave, texture, prints, embroidery, buttons, zippers, "
            "seams, waistband and every small detail EXACTLY as in the front garment reference image."
        )
    else:
        if UPPER_GARMENT_TYPE:
            parts.append(
                f"Upper garment: {UPPER_GARMENT_TYPE} — {UPPER_GARMENT_SPEC}. "
                "Match fabric, texture, color, print, buttons, collar, cuffs, stitching EXACTLY from refs."
            )
        if LOWER_GARMENT_TYPE:
            parts.append(
                f"Lower garment: {LOWER_GARMENT_TYPE} — {LOWER_GARMENT_SPEC}. "
                "If no lower-garment image is provided, generate a matching one."
            )
    if FITTING:
        parts.append(f"Overall fitting: {FITTING}.")
    return " ".join(parts)


def _core_prompt(pose_index: int, is_back_pose: bool) -> str:
    body = _body_description()
    garment = _garment_description()
    garment_ref_note = (
        "Use the BACK garment reference image (this is a back pose)."
        if is_back_pose
        else "Use the FRONT garment reference image."
    )
    return f"""
Generate ONE candid editorial fashion photograph — a real, handheld, in-camera
exposure shot on a full-frame mirrorless body (Sony A7R V / Leica Q3) with a
fast prime lens (24mm f/2.8 environmental or 85mm f/1.8 portrait, chosen to
fit the framing). Render as if a skilled human photographer pressed the
shutter in a real location — NOT a CG render, NOT a polished studio key-art,
NOT AI-smooth, NOT symmetrically composed. Single photorealistic frame only.

ABSOLUTE PRIORITIES (must be satisfied before aesthetics):
1) Preserve exact face identity from the face reference — same bone structure,
   eye shape, skin tone, lip shape, hair color and hairline. Zero drift.
2) Reproduce the garment EXACTLY from the garment reference — fabric weave,
   prints, seams, stitching, trims, buttons, zippers, hardware, color.
3) Copy the mannequin's POSTURE and FRAMING exactly (see dedicated section
   below). This is non-negotiable.
4) Integrate the subject into the background with matching perspective, light
   direction, color spill, and physical contact shadows.

POSE & POSTURE (MANDATORY — the mannequin reference is a POSTURE TEMPLATE,
match it exactly):
- Copy the mannequin's body posture 1:1: overall stance, weight distribution,
  which leg bears weight, hip angle, shoulder line, spine curvature, chest
  direction, and head tilt.
- Copy LEG position exactly: same leg bent / straight, same foot placement
  and spacing, same knee angle, same ankle rotation.
- Copy HAND and ARM position exactly: same arm angle at shoulder and elbow,
  same wrist rotation, same finger position (open / fist / pinch / pointing /
  touching garment / on hip), same left-vs-right asymmetry. If one hand is
  raised and the other is down, mirror that exactly — do NOT substitute a
  generic or 'more flattering' model pose.
- Copy the overall body orientation (facing camera / 3-quarter / profile /
  back) exactly as shown in the mannequin.
- Ignore the mannequin's face, skin, clothing, proportions, background —
  ONLY its posture and framing transfer.
- Do NOT invent a new pose. Do NOT 'improve' the pose. Do NOT default to
  a catalogue contrapposto unless the mannequin shows that exact stance.

FRAMING & CROP (MANDATORY — match the mannequin reference):
- If the mannequin shows the FULL body (head to feet), the output MUST be
  full-body with both feet fully inside the frame and FOOTWEAR ON
  (shoes are required — bare feet are rejected).
- If the mannequin shows only the UPPER body (head to waist/torso), the
  output MUST be an upper-body crop with the same crop line — no feet,
  no legs, no footwear rendered.
- Do NOT change the crop relative to the mannequin. Do NOT zoom out to add
  missing feet, and do NOT zoom in to hide feet. The mannequin's framing IS
  the output's framing.

REALISM / SKIN / TEXTURE — pursue imperfection, not polish:
- Skin shows real pores, peach fuzz, micro-freckles, slight T-zone shine,
  subsurface scattering in ears/nostrils, faint redness near nose bridge and
  cheekbones. Matte-satin finish, never plastic, never airbrushed.
- Hair has stray flyaways, asymmetric parting, catch-light in individual
  strands; avoid helmet-shaped 'AI hair'.
- Lips show subtle texture, slight gloss variation; teeth (if visible) are
  naturally uneven in tone.
- Garment fabric shows weave, pile direction, thread ends, fabric memory
  where the body bends; accessories show fingerprints, dust, micro-scratches.
- Background surfaces have dust, small marks, material grain.

LIGHTING — real physics, mixed sources:
- Motivated mixed real-world lighting (window + bounce, or sun + shade, or
  practical + fill). Directional key with visible falloff. No flat lighting.
- Global illumination: color spill from walls/floor/garment onto skin;
  tight ambient occlusion under jaw, collar, armpits, fabric folds.
- Specular highlights follow skin oil map (forehead, nose tip, cupid's bow);
  specular occlusion inside pores.
- Color temperature consistent across subject and background.

SUBJECT-BACKGROUND INTEGRATION (grounded, not cut-out):
- Soft layered contact shadow with dark core and diffuse penumbra beneath
  feet / lowest contact point; long cast shadow if key is directional.
- Ambient occlusion at clothing-skin contact and subject-near-wall seams.
- Background color bounces onto the side of the face, neck, garment.
- Atmospheric haze / fine dust / humidity in the air for depth.
- Depth of field: sharp on eyes, creamy optical bokeh on background
  (cats-eye near edges, not uniformly round).
- Accurate, slightly distorted reflections of the full environment on shiny
  surfaces (eyes, jewelry, patent leather, glass).

LENS & CAMERA REALISM (subtle, never exaggerated):
- Fine film-grain-like sensor noise, denser in shadows.
- Slight chromatic aberration at high-contrast edges.
- Subtle lens flare only if a light source is in/near frame.
- Handheld feel (not tripod-locked).
- Natural color grading — no teal-and-orange crunch, no oversaturation.

COMPOSITION — candid snapshot feel:
- Handheld, imperfect framing; slight off-axis or subtle Dutch angle allowed;
  eye-line NOT perfectly centered. Editorial / documentary snapshot, not
  beauty-counter advert.
- Natural pose weight distribution; correct physical contact (visible
  pressure and fabric deformation at grip points, not floating).

INPUTS:
Body: {body}
Garment rule: {garment_ref_note}
Garment details: {garment}

STRICT: single photorealistic human subject only. No text, watermarks, logos,
duplicates, extra limbs or warped hands. No HDR crunch, no CGI sheen, no
plastic skin, no AI-smooth background, no perfect symmetry.

FINAL REMINDER (highest weight):
1) POSTURE: the mannequin reference defines body pose, hand position, leg
   position — copy it EXACTLY, do not substitute a generic model pose.
2) FRAMING: the mannequin reference defines the crop — full-body mannequin
   → full-body output with footwear ON; upper-body mannequin → upper-body
   output with no feet and no footwear. No mixing.
Pose variation index: {pose_index + 1}.
""".strip()


def _ordered_refs(pose: PoseState) -> list[tuple[str, str]]:
    """(label, url) pairs in the order: face -> garment(s) -> pose -> background."""
    refs: list[tuple[str, str]] = []
    if MODEL_FACE_URL:
        refs.append(("MODEL FACE (identity lock)", MODEL_FACE_URL))
    if pose.is_back and BACK_GARMENT_URL:
        refs.append(("GARMENT — BACK view (use this)", BACK_GARMENT_URL))
        if FRONT_GARMENT_URL:
            refs.append(("GARMENT — FRONT view (context only)", FRONT_GARMENT_URL))
    else:
        if FRONT_GARMENT_URL:
            refs.append(("GARMENT — FRONT view (use this)", FRONT_GARMENT_URL))
        if BACK_GARMENT_URL:
            refs.append(("GARMENT — BACK view (context only)", BACK_GARMENT_URL))
    if pose.pose_url:
        refs.append(("MANNEQUIN POSE — body posture only", pose.pose_url))
    if BACKGROUND_URL:
        refs.append(("BACKGROUND", BACKGROUND_URL))
    return refs


def _prompt_with_url_preamble(pose: PoseState) -> tuple[str, list[str]]:
    refs = _ordered_refs(pose)
    urls = [u for _, u in refs]
    lines = [
        "The API provides reference images in this order. Treat each index as the role below:",
        "",
    ]
    for i, (label, _) in enumerate(refs):
        lines.append(f"- image[{i}] — {label}")
    preamble = "\n".join(lines)
    full_prompt = f"{preamble}\n\n{_core_prompt(pose.index, pose.is_back)}"
    return full_prompt, urls


# =============================================================================
# HTTP HELPER
# =============================================================================
def _load_image_bytes(src: str, timeout: int = 30) -> tuple[bytes, str]:
    parsed = urlparse(src)
    if parsed.scheme in ("http", "https"):
        r = requests.get(src, timeout=timeout)
        r.raise_for_status()
        data = r.content
        mime = (r.headers.get("Content-Type", "") or "").split(";")[0].strip() or "image/jpeg"
    else:
        p = Path(src)
        data = p.read_bytes()
        mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"

    if mime not in ("image/jpeg", "image/png", "image/webp") and Image is not None:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        data, mime = buf.getvalue(), "image/jpeg"
    return data, mime


def _download_to(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


# =============================================================================
# GENERATOR AGENTS — each returns an async fn: (PoseState) -> Path
# =============================================================================
async def _run_in_thread(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)


def _pose_output_path(pose: PoseState, provider: str, ext: str = "png") -> Path:
    return RUN_OUTPUT_DIR / f"pose_{pose.index + 1}_{provider}.{ext}"


# ----- KIE nano-banana-2 -----------------------------------------------------
KIE_CREATE_URL = "https://api.kie.ai/api/v1/jobs/createTask"
KIE_RECORD_URL = "https://api.kie.ai/api/v1/jobs/recordInfo"


def _kie_headers() -> dict[str, str]:
    key = os.environ.get("KIE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("KIE_API_KEY not set")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _kie_create_nb2_task(prompt: str, image_urls: list[str]) -> str:
    body = {
        "model": "nano-banana-2",
        "input": {
            "prompt": prompt,
            "image_input": image_urls,
            "aspect_ratio": ASPECT_RATIO,
            "resolution": RESOLUTION,
            "output_format": "jpg",
        },
    }
    r = requests.post(KIE_CREATE_URL, headers=_kie_headers(), data=json.dumps(body), timeout=60)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != 200:
        raise RuntimeError(f"KIE createTask failed: {data}")
    task_id = (data.get("data") or {}).get("taskId")
    if not task_id:
        raise RuntimeError(f"KIE createTask missing taskId: {data}")
    return str(task_id)


def _kie_poll(task_id: str, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    auth = _kie_headers()["Authorization"]
    while time.monotonic() < deadline:
        r = requests.get(KIE_RECORD_URL, headers={"Authorization": auth}, params={"taskId": task_id}, timeout=60)
        r.raise_for_status()
        body = r.json() or {}
        if body.get("code") != 200:
            raise RuntimeError(f"KIE recordInfo error: {body}")
        data = body.get("data") or {}
        state = (data.get("state") or "").lower()
        if state == "success":
            raw = data.get("resultJson")
            parsed = json.loads(raw) if isinstance(raw, str) else (raw or {})
            urls = parsed.get("resultUrls") or parsed.get("result_urls") or []
            if not urls:
                raise RuntimeError(f"KIE success but no URLs: {parsed}")
            return str(urls[0])
        if state == "fail":
            raise RuntimeError(
                f"KIE task failed: {data.get('failMsg') or data.get('fail_msg')!r} (code={data.get('failCode')!r})"
            )
        time.sleep(KIE_POLL_INTERVAL_S)
    raise TimeoutError(f"KIE task {task_id!r} did not finish within {timeout_s}s")


def _agent_kie_nb2(pose: PoseState) -> Path:
    prompt, image_urls = _prompt_with_url_preamble(pose)
    task_id = _kie_create_nb2_task(prompt, image_urls)
    result_url = _kie_poll(task_id, timeout_s=PER_IMAGE_TIMEOUT_S)
    return _download_to(result_url, _pose_output_path(pose, "kie"))


# ----- Evolink nano-banana-2 -------------------------------------------------
EVOLINK_URL = "https://api.evolink.ai/v1/images/generations"
EVOLINK_TASK_URL = "https://api.evolink.ai/v1/tasks/{task_id}"


def _evolink_headers() -> dict[str, str]:
    key = os.environ.get("EVOLINK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("EVOLINK_API_KEY not set")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _agent_evolink_nb2(pose: PoseState) -> Path:
    prompt, image_urls = _prompt_with_url_preamble(pose)
    payload = {
        "model": "gemini-3.1-flash-image-preview",
        "prompt": prompt,
        "size": ASPECT_RATIO,
        "quality": RESOLUTION,
        "image_urls": image_urls,
    }
    r = requests.post(EVOLINK_URL, headers=_evolink_headers(), data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    created = r.json()
    task_id = created.get("id")
    if not task_id:
        raise RuntimeError(f"Evolink missing task id: {created}")

    deadline = time.monotonic() + PER_IMAGE_TIMEOUT_S
    while time.monotonic() < deadline:
        q = requests.get(EVOLINK_TASK_URL.format(task_id=task_id), headers=_evolink_headers(), timeout=60)
        q.raise_for_status()
        body = q.json()
        status = (body.get("status") or "").lower()
        if status == "completed":
            urls = body.get("results") or []
            if not urls:
                raise RuntimeError(f"Evolink completed but no results: {body}")
            return _download_to(str(urls[0]), _pose_output_path(pose, "evolink"))
        if status in {"failed", "cancelled"}:
            raise RuntimeError(f"Evolink task failed: {body}")
        time.sleep(3.0)
    raise TimeoutError(f"Evolink task {task_id!r} did not finish within {PER_IMAGE_TIMEOUT_S}s")


# ----- Vertex (google-genai) nano-banana-2 & nano-banana-pro -----------------
def _vertex_client():
    if genai is None:
        raise RuntimeError("google-genai not installed")
    api_key = os.environ.get("GOOGLE_CLOUD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_CLOUD_API_KEY not set")
    return genai.Client(vertexai=True, api_key=api_key)


def _vertex_generate(model_id: str, pose: PoseState, use_thinking: bool) -> Path:
    client = _vertex_client()

    parts: list[Any] = []
    parts.append(genai_types.Part.from_text(text="[REF 1/4] MODEL FACE (identity lock):"))
    face_bytes, face_mime = _load_image_bytes(MODEL_FACE_URL)
    parts.append(genai_types.Part.from_bytes(data=face_bytes, mime_type=face_mime))

    if pose.is_back and BACK_GARMENT_URL:
        d, m = _load_image_bytes(BACK_GARMENT_URL)
        parts.append(genai_types.Part.from_text(text="[REF 2/4] GARMENT — BACK (use this):"))
        parts.append(genai_types.Part.from_bytes(data=d, mime_type=m))
        if FRONT_GARMENT_URL:
            d2, m2 = _load_image_bytes(FRONT_GARMENT_URL)
            parts.append(genai_types.Part.from_text(text="[REF 2b] GARMENT — FRONT (context):"))
            parts.append(genai_types.Part.from_bytes(data=d2, mime_type=m2))
    else:
        if FRONT_GARMENT_URL:
            d, m = _load_image_bytes(FRONT_GARMENT_URL)
            parts.append(genai_types.Part.from_text(text="[REF 2/4] GARMENT — FRONT (use this):"))
            parts.append(genai_types.Part.from_bytes(data=d, mime_type=m))
        if BACK_GARMENT_URL:
            d2, m2 = _load_image_bytes(BACK_GARMENT_URL)
            parts.append(genai_types.Part.from_text(text="[REF 2b] GARMENT — BACK (context):"))
            parts.append(genai_types.Part.from_bytes(data=d2, mime_type=m2))

    d3, m3 = _load_image_bytes(pose.pose_url)
    parts.append(genai_types.Part.from_text(text="[REF 3/4] MANNEQUIN POSE — body posture only:"))
    parts.append(genai_types.Part.from_bytes(data=d3, mime_type=m3))

    d4, m4 = _load_image_bytes(BACKGROUND_URL)
    parts.append(genai_types.Part.from_text(text="[REF 4/4] BACKGROUND:"))
    parts.append(genai_types.Part.from_bytes(data=d4, mime_type=m4))

    parts.append(genai_types.Part.from_text(text=_core_prompt(pose.index, pose.is_back)))

    safety = [
        genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
        genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
        genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]
    image_config = genai_types.ImageConfig(
        aspect_ratio=ASPECT_RATIO,
        image_size=RESOLUTION,
        output_mime_type="image/jpeg",
    )
    thinking_config = genai_types.ThinkingConfig(thinking_level="MINIMAL") if use_thinking else None

    cfg = genai_types.GenerateContentConfig(
        temperature=0.2,
        top_p=0.2,
        max_output_tokens=32768,
        response_modalities=["IMAGE"],
        safety_settings=safety,
        image_config=image_config,
        thinking_config=thinking_config,
    )

    buffer = bytearray()
    contents = [genai_types.Content(role="user", parts=parts)]
    for chunk in client.models.generate_content_stream(model=model_id, contents=contents, config=cfg):
        if not getattr(chunk, "candidates", None):
            continue
        for cand in chunk.candidates:
            if not cand.content or not cand.content.parts:
                continue
            for part in cand.content.parts:
                inline = getattr(part, "inline_data", None)
                if inline and inline.data:
                    raw = inline.data
                    if isinstance(raw, str):
                        raw = base64.b64decode(raw)
                    buffer.extend(raw)

    if not buffer:
        raise RuntimeError(f"No image returned by Vertex model {model_id} for pose {pose.index + 1}")

    suffix = "nb2" if "flash" in model_id else "nbpro"
    out = _pose_output_path(pose, f"vertex-{suffix}", ext="jpg")
    out.write_bytes(bytes(buffer))
    return out


def _agent_vertex_nb2(pose: PoseState) -> Path:
    return _vertex_generate("gemini-3.1-flash-image-preview", pose, use_thinking=True)


def _agent_vertex_nbpro(pose: PoseState) -> Path:
    return _vertex_generate("gemini-3-pro-image-preview", pose, use_thinking=False)


GENERATOR_AGENTS: dict[str, Callable[[PoseState], Path]] = {
    "kie_nb2":      _agent_kie_nb2,
    "vertex_nb2":   _agent_vertex_nb2,
    "vertex_nbpro": _agent_vertex_nbpro,
    "evolink_nb2":  _agent_evolink_nb2,
}


# =============================================================================
# UPSCALER AGENTS
# =============================================================================
def _agent_upscale_kie(pose: PoseState) -> Path:
    assert pose.generated_path is not None
    # KIE requires a publicly-reachable URL; if the user doesn't host outputs,
    # this agent will fail and we will fall through to fal.
    # If pose.generated_path happens to be a URL (e.g. still the remote URL), pass through.
    url_candidate = pose.generated_path.as_posix()
    if not url_candidate.startswith("http"):
        raise RuntimeError(
            "KIE Topaz requires a public URL; local files are not supported in this pipeline."
        )
    body = {
        "model": "topaz/image-upscale",
        "input": {"image_url": url_candidate, "upscale_factor": UPSCALE_FACTOR_KIE},
    }
    r = requests.post(KIE_CREATE_URL, headers=_kie_headers(), data=json.dumps(body), timeout=60)
    r.raise_for_status()
    created = r.json()
    if created.get("code") != 200:
        raise RuntimeError(f"KIE Topaz createTask failed: {created}")
    task_id = (created.get("data") or {}).get("taskId")
    result_url = _kie_poll(str(task_id), timeout_s=UPSCALE_PER_IMAGE_TIMEOUT_S)
    dest = RUN_OUTPUT_DIR / f"pose_{pose.index + 1}_upscaled_kie.png"
    return _download_to(result_url, dest)


def _agent_upscale_fal(pose: PoseState) -> Path:
    if fal_client is None:
        raise RuntimeError("fal-client not installed")
    fal_key = os.environ.get("FAL_KEY", "").strip()
    if not fal_key:
        raise RuntimeError("FAL_KEY not set")
    os.environ["FAL_KEY"] = fal_key

    assert pose.generated_path is not None
    src = pose.generated_path.as_posix()
    if src.startswith("http"):
        image_url = src
    else:
        with open(src, "rb") as fh:
            image_url = fal_client.upload(fh.read(), content_type="image/png")

    def _subscribe():
        return fal_client.subscribe(
            "fal-ai/topaz/upscale/image",
            arguments={
                "image_url": image_url,
                "model": UPSCALE_MODEL_FAL,
                "upscale_factor": UPSCALE_FACTOR_FAL,
                "output_format": UPSCALE_OUTPUT_FORMAT,
                "face_enhancement": True,
                "face_enhancement_strength": 0.8,
                "subject_detection": "All",
            },
            with_logs=False,
        )

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_subscribe)
        result = fut.result(timeout=UPSCALE_PER_IMAGE_TIMEOUT_S)

    image_data = (result or {}).get("image") or {}
    out_url = image_data.get("url")
    if not out_url:
        raise RuntimeError(f"fal Topaz returned no URL: {result}")
    dest = RUN_OUTPUT_DIR / f"pose_{pose.index + 1}_upscaled_fal.png"
    return _download_to(out_url, dest)


UPSCALER_AGENTS: dict[str, Callable[[PoseState], Path]] = {
    "kie_topaz": _agent_upscale_kie,
    "fal_topaz": _agent_upscale_fal,
}


# =============================================================================
# RETRY / BATCH RUNNERS
# =============================================================================
async def _run_with_retry(
    fn_sync: Callable[[PoseState], Path],
    pose: PoseState,
    *,
    retries: int,
    provider: str,
    stage: str,
    per_attempt_timeout_s: float,
) -> Path | None:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        event_name = f"{stage}:{provider}:pose_{pose.index + 1}:attempt_{attempt}"
        t0 = TIMELINE.start(
            event_name,
            "attempt",
            {"pose": pose.index + 1, "provider": provider, "stage": stage, "attempt": attempt},
        )
        try:
            log.info("%s[%s] pose %d attempt %d/%d", stage, provider, pose.index + 1, attempt, retries)
            result = await asyncio.wait_for(_run_in_thread(fn_sync, pose), timeout=per_attempt_timeout_s)
            elapsed_s = TIMELINE.end(
                t0,
                event_name,
                "attempt",
                {"status": "ok", "output": str(result)},
            )
            log.info("%s[%s] pose %d OK in %.1fs -> %s", stage, provider, pose.index + 1, elapsed_s, result)
            return result
        except Exception as e:
            last_err = e
            TIMELINE.end(
                t0,
                event_name,
                "attempt",
                {"status": "failed", "error": repr(e)},
            )
            log.warning("%s[%s] pose %d attempt %d failed: %r", stage, provider, pose.index + 1, attempt, e)
            if attempt < retries:
                delay = min(2.0 * (2 ** (attempt - 1)) + random.uniform(0.0, 1.0), 30.0)
                await asyncio.sleep(delay)
    pose.last_error = repr(last_err) if last_err else "unknown"
    log.error("%s[%s] pose %d EXHAUSTED %d retries", stage, provider, pose.index + 1, retries)
    return None


async def _run_batch(
    poses: list[PoseState],
    policy: GenPolicy,
    fn_sync: Callable[[PoseState], Path],
    *,
    stage: str,
    per_attempt_timeout_s: float,
    assign_attr: str,
) -> None:
    """Run `poses` through `fn_sync` with concurrency/retries from `policy`."""
    if not poses:
        return
    sem = asyncio.Semaphore(policy.concurrency)

    async def _one(p: PoseState) -> None:
        async with sem:
            p.generator_attempts.append(policy.name)
            result = await _run_with_retry(
                fn_sync,
                p,
                retries=policy.retries,
                provider=policy.name,
                stage=stage,
                per_attempt_timeout_s=per_attempt_timeout_s,
            )
            if result is not None:
                setattr(p, assign_attr, result)

    batch_name = f"{stage}:{policy.name}:batch"
    t0 = TIMELINE.start(
        batch_name,
        "batch",
        {
            "poses": len(poses),
            "concurrency": policy.concurrency,
            "retries": policy.retries,
        },
    )
    log.info(
        "%s[%s] batch start poses=%d concurrency=%d retries=%d",
        stage,
        policy.name,
        len(poses),
        policy.concurrency,
        policy.retries,
    )
    await asyncio.gather(*[_one(p) for p in poses])
    ok_count = sum(1 for p in poses if getattr(p, assign_attr) is not None)
    TIMELINE.end(
        t0,
        batch_name,
        "batch",
        {"ok": ok_count, "total": len(poses), "assign_attr": assign_attr},
    )
    log.info("%s[%s] batch done; ok=%d/%d",
             stage, policy.name,
             ok_count,
             len(poses))


# =============================================================================
# LANGGRAPH NODES
# =============================================================================
def node_init(state: PipelineState) -> PipelineState:
    t0 = TIMELINE.start("node_init", "node")
    if not state.poses:
        for i, url in enumerate(POSES_MANNEQUIN_URLS):
            if url:
                state.poses[i] = PoseState(index=i, pose_url=url)
    log.info("[init] poses=%d", len(state.poses))
    TIMELINE.end(t0, "node_init", "node", {"poses": len(state.poses)})
    return state


def node_generate(state: PipelineState) -> PipelineState:
    """Stage-1 supervisor: picks the current provider and runs it for all pending poses."""
    policy = GEN_POLICIES[state.gen_stage_index]
    agent = GENERATOR_AGENTS[policy.name]
    t0 = TIMELINE.start("node_generate", "node", {"provider": policy.name})

    # Pending = poses that either have no generation yet OR failed validation (and
    # were reset back to None) and haven't been generated by this provider yet.
    pending = [p for p in state.poses.values() if p.generated_path is None and not p.failed]
    log.info("[stage-1] provider=%s pending=%d", policy.name, len(pending))

    asyncio.run(_run_batch(
        pending,
        policy,
        agent,
        stage="stage-1",
        per_attempt_timeout_s=PER_IMAGE_TIMEOUT_S,
        assign_attr="generated_path",
    ))
    TIMELINE.end(
        t0,
        "node_generate",
        "node",
        {"provider": policy.name, "pending": len(pending)},
    )
    return state


def route_after_generate(state: PipelineState) -> str:
    # Anyone still missing? Move to next provider if one is available; else fail those poses.
    missing = [p for p in state.poses.values() if p.generated_path is None and not p.failed]
    if missing:
        if state.gen_stage_index + 1 < len(GEN_POLICIES):
            state.gen_stage_index += 1
            log.info("[stage-1] %d pose(s) failed -> next provider (%s)",
                     len(missing), GEN_POLICIES[state.gen_stage_index].name)
            return "generate"
        for p in missing:
            p.failed = True
            log.error("[stage-1] pose %d permanently failed: %s", p.index + 1, p.last_error)
    if REQUIRED_UPSCALE:
        return "upscale"
    state.finished = True
    log.info("[stage-2] REQUIRED_UPSCALE=false -> skipping upscaling and finishing run")
    return "end"


def node_upscale(state: PipelineState) -> PipelineState:
    targets = state.pending_upscale()
    t0_node = TIMELINE.start("node_upscale", "node", {"targets": len(targets)})
    log.info("[stage-2] to upscale=%d", len(targets))

    # Try each upscale policy as a fallback chain, one after the other.
    for policy in UPSCALE_POLICIES:
        pending = [p for p in targets if p.upscaled_path is None]
        if not pending:
            break
        agent = UPSCALER_AGENTS[policy.name]
        t0_policy = TIMELINE.start(
            f"stage-2:{policy.name}",
            "upscale_policy",
            {"pending": len(pending)},
        )
        asyncio.run(_run_batch(
            pending,
            policy,
            agent,
            stage="stage-2",
            per_attempt_timeout_s=UPSCALE_PER_IMAGE_TIMEOUT_S,
            assign_attr="upscaled_path",
        ))
        TIMELINE.end(
            t0_policy,
            f"stage-2:{policy.name}",
            "upscale_policy",
            {"pending": len(pending)},
        )

    for p in targets:
        if p.upscaled_path is None:
            log.warning("[stage-2] pose %d not upscaled (keeping original)", p.index + 1)

    state.finished = True
    TIMELINE.end(t0_node, "node_upscale", "node", {"targets": len(targets)})
    return state


# =============================================================================
# GRAPH
# =============================================================================
def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("init", node_init)
    g.add_node("generate", node_generate)
    g.add_node("upscale", node_upscale)

    g.set_entry_point("init")
    g.add_edge("init", "generate")
    g.add_conditional_edges("generate", route_after_generate, {
        "generate": "generate",
        "upscale": "upscale",
        "end": END,
    })
    g.add_edge("upscale", END)

    return g.compile()


# =============================================================================
# MAIN
# =============================================================================
def _write_summary(final: PipelineState) -> Path:
    summary = {
        "poses": [
            {
                "index": p.index,
                "pose_url": p.pose_url,
                "generated": str(p.generated_path) if p.generated_path else None,
                "providers_tried": p.generator_attempts,
                "upscaled": str(p.upscaled_path) if p.upscaled_path else None,
                "failed": p.failed,
                "last_error": p.last_error,
            }
            for p in sorted(final.poses.values(), key=lambda x: x.index)
        ],
        "stages": {
            "gen_stage_index": final.gen_stage_index,
            "upscale_stage_index": final.upscale_stage_index,
            "finished": final.finished,
        },
    }
    path = RUN_OUTPUT_DIR / "pipeline_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    log.info("Wrote summary -> %s", path)
    return path


def _write_timeline(total_duration_s: float) -> Path:
    payload = {
        "run_id": RUN_ID,
        "run_output_dir": str(RUN_OUTPUT_DIR),
        "run_start_epoch_s": TIMELINE.run_start_epoch_s,
        "run_end_epoch_s": time.time(),
        "total_duration_s": total_duration_s,
        "events": [
            {
                "name": e.name,
                "category": e.category,
                "start_epoch_s": e.start_epoch_s,
                "end_epoch_s": e.end_epoch_s,
                "duration_s": e.duration_s,
                "meta": e.meta,
            }
            for e in TIMELINE.events
        ],
    }
    path = RUN_OUTPUT_DIR / "pipeline_timing.json"
    path.write_text(json.dumps(payload, indent=2))
    log.info("Wrote timing -> %s", path)
    return path


def main() -> None:
    t0_total = TIMELINE.start("pipeline_run", "overall", {"run_id": RUN_ID})
    log.info("Pipeline start (%s) run_id=%s output_dir=%s", _stamp(), RUN_ID, RUN_OUTPUT_DIR)
    graph = build_graph()
    final_state: PipelineState = graph.invoke(PipelineState())
    # LangGraph may return a dict-like when using pydantic; coerce if needed.
    if not isinstance(final_state, PipelineState):
        coerced = PipelineState()
        coerced.poses = final_state.get("poses", {})  # type: ignore[attr-defined]
        coerced.gen_stage_index = final_state.get("gen_stage_index", 0)  # type: ignore[attr-defined]
        coerced.upscale_stage_index = final_state.get("upscale_stage_index", 0)  # type: ignore[attr-defined]
        coerced.finished = bool(final_state.get("finished", False))  # type: ignore[attr-defined]
        final_state = coerced
    _write_summary(final_state)
    total_duration_s = TIMELINE.end(t0_total, "pipeline_run", "overall", {"status": "completed"})
    _write_timeline(total_duration_s)
    log.info("Pipeline done (%s) total_time=%.2fs", _stamp(), total_duration_s)


if __name__ == "__main__":
    main()
