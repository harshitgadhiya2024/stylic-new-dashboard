"""
Photoshoot background job service — LangGraph multi-agent pipeline.

Architecture (mirrors ``pipeline.py`` at the repo root, adapted to this app):

  Stage-1 GENERATION — provider fallback chain (each provider tried for all
  pending poses before moving to the next provider):

      1) KIE.ai  nano-banana-2        (concurrency 10, retries 2)
      2) Vertex  nano-banana-2 direct (concurrency 2,  retries 4)
      3) Vertex  nano-banana-pro      (concurrency 2,  retries 4)
      4) Evolink nano-banana-2        (concurrency 10, retries 10)

  Stage-2 UPSCALE — streamed per-pose materialize + upscale.  Each pose
  runs downstream work as soon as its own generation finishes; successful
  poses are ``$push``-written into the photoshoot document immediately and
  credits are deducted incrementally.  The job does NOT wait for every pose
  to finish generation before starting upscale.

LangGraph graph (``build_photoshoot_graph``):

  init  ─► gen_kie_nb2  ─► gen_vertex_nb2  ─► gen_vertex_nbpro  ─► gen_evolink_nb2  ─► finalize ─► END

Each ``gen_*`` node kicks off two things concurrently for every pending pose:
  • generation with the node's provider (with retries+timeout), and
  • as soon as a pose generates successfully, its Stage-2 task
    (``_materialize_and_upscale``) is scheduled; that task pushes the
    finished output_image into Mongo and deducts credits on success.

The ``finalize`` node waits for any in-flight Stage-2 tasks and marks the
photoshoot document ``completed`` / ``partial`` / ``failed``.

KIE upscale logic is preserved exactly (``enhance_and_upload`` in
``app.services.modal_enhance_service`` is invoked unchanged).
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional, TypedDict
from urllib.parse import urlparse

import httpx
import requests
from PIL import Image

logger = logging.getLogger("photoshoot")

from app.config import settings
from app.database import (
    get_backgrounds_collection,
    get_credit_history_collection,
    get_model_faces_collection,
    get_photoshoots_collection,
    get_poses_collection,
    get_users_collection,
)
from app.services.body_prompt_terms import (
    body_height_instruction_long,
    body_weight_instruction_long,
    normalize_body_height,
    normalize_body_weight,
    raw_height_from_req,
    raw_weight_from_req,
)
from app.services.modal_enhance_service import enhance_and_upload
from app.services.r2_service import upload_bytes_to_r2

# Optional SDKs — only required for the providers you enable.
try:  # pragma: no cover
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore
    genai_types = None  # type: ignore

from langgraph.graph import END, StateGraph  # type: ignore


# ---------------------------------------------------------------------------
# Provider policies (mirror pipeline.py GEN_POLICIES)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenPolicy:
    name: str
    concurrency: int
    retries: int


# Per-photoshoot concurrency values are env-tunable (see app/config.py).
# Max poses per photoshoot is 8, so values >8 are wasted; values <8 serialize
# some poses within the same job.  Cross-job throttling for KIE is handled by
# the Redis rate limiter in ``app.services.kie_rate_limiter``.
GEN_POLICIES: List[GenPolicy] = [
    GenPolicy(name="kie_nb2",       concurrency=settings.PHOTOSHOOT_KIE_CONCURRENCY,          retries=2),
    GenPolicy(name="vertex_nb2",    concurrency=settings.PHOTOSHOOT_VERTEX_NB2_CONCURRENCY,   retries=4),
    GenPolicy(name="vertex_nbpro",  concurrency=settings.PHOTOSHOOT_VERTEX_NBPRO_CONCURRENCY, retries=4),
    GenPolicy(name="evolink_nb2",   concurrency=settings.PHOTOSHOOT_EVOLINK_CONCURRENCY,      retries=10),
]


# ---------------------------------------------------------------------------
# Pose runtime
# ---------------------------------------------------------------------------

@dataclass
class PoseRuntime:
    index: int                          # 1-based within the photoshoot
    pose_prompt: str
    mannequin_url: str
    # Stage-1 output
    generated_url: Optional[str] = None
    generated_bytes: Optional[bytes] = None
    generator_used: Optional[str] = None
    # Stage-2 output
    output_image: Optional[dict] = None  # entry appended to output_images on success
    # Bookkeeping
    providers_tried: List[str] = field(default_factory=list)
    last_error: Optional[str] = None
    # Stage-2 async handle (so gen nodes can fire-and-forget upscale work)
    upscale_task: Optional[asyncio.Task] = None

    @property
    def label(self) -> str:
        return f"pose-{self.index:02d}"

    @property
    def is_back(self) -> bool:
        return "back" in (self.pose_prompt or "").lower()

    @property
    def is_upper_body(self) -> bool:
        """True when the pose clearly only shows the upper half (head → waist).

        Used to decide whether footwear must be rendered.  Anything that is
        explicitly a headshot / portrait / half / bust / torso / waist-up
        framing counts as upper-body.  Full-body (the default) requires shoes.
        """
        text = (self.pose_prompt or "").lower()
        upper_markers = (
            "upper body", "upper-body", "upper half", "half body", "half-body",
            "waist up", "waist-up", "waist-level", "bust shot", "bust-shot",
            "headshot", "head shot", "portrait", "torso", "chest up", "chest-up",
            "close up", "close-up", "closeup", "shoulders up", "shoulder up",
        )
        return any(m in text for m in upper_markers)

    @property
    def framing_label(self) -> str:
        return "upper-body" if self.is_upper_body else "full-body"

    @property
    def is_generated(self) -> bool:
        return self.generated_url is not None or self.generated_bytes is not None


# LangGraph state — a mutable container carried across nodes.  We use a
# ``TypedDict`` with a single key pointing at a runtime object, because the
# runtime carries async tasks/bytes that aren't JSON serializable.
class PipelineState(TypedDict):
    ctx: "PipelineContext"


@dataclass
class PipelineContext:
    photoshoot_id: str
    req: dict
    poses: List[PoseRuntime]
    model_face_url: str
    background_url: str
    # Mongo/IO
    photoshoots_col: Any
    upscaling_col: Any
    users_col: Any
    history_col: Any
    # Credit book-keeping
    credit_per_image: float
    credit_total_accum: float = 0.0
    successful_image_ids: List[str] = field(default_factory=list)
    # Stage-1 tracking (only used for logging / final summary)
    stage1_failed: List[PoseRuntime] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pose prompt resolution (public; still used by router path)
# ---------------------------------------------------------------------------

async def _fetch_pose_data(pose_ids: List[str], poses_col=None) -> List[dict]:
    """
    Return ``[{"image_url": ..., "pose_prompt": ...}, ...]`` per pose_id, in
    the same order as ``pose_ids``. Uses a single ``$in`` query instead of
    N round-trips to Mongo (was ~500ms, now ~30ms for 7 poses).
    """
    logger.info("[poses] Fetching %d pose doc(s) from DB", len(pose_ids))
    col = poses_col if poses_col is not None else get_poses_collection()

    cursor = col.find({"pose_id": {"$in": list(pose_ids)}})
    by_id: dict[str, dict] = {}
    async for doc in cursor:
        pid = doc.get("pose_id")
        if pid:
            by_id[str(pid)] = doc

    results: List[dict] = []
    for pid in pose_ids:
        doc = by_id.get(str(pid))
        if doc:
            logger.info("[poses] Found pose_id=%s image_url=%s", pid, bool(doc.get("image_url")))
            results.append({
                "image_url":   doc.get("image_url") or "",
                "pose_prompt": doc.get("pose_prompt") or "",
            })
        else:
            logger.warning("[poses] No doc for pose_id=%s — text fallback only", pid)
            results.append({
                "image_url":   "",
                "pose_prompt": f"Standing in a natural, relaxed fashion model pose — pose id: {pid}",
            })
    logger.info("[poses] Resolved %d pose doc(s)", len(results))
    return results


async def resolve_poses(req: dict, poses_col=None) -> List[dict]:
    """Resolve pose data from either pre-supplied ``pose_data`` or ``poses_ids``."""
    pose_data = req.get("pose_data")
    if pose_data and isinstance(pose_data, list):
        result = [
            {"image_url": pd.get("image_url") or "", "pose_prompt": pd.get("pose_prompt") or ""}
            for pd in pose_data
        ]
        logger.info("[poses] Using %d pre-resolved pose_data entries (regeneration)", len(result))
    else:
        ids = req.get("poses_ids") or []
        logger.info("[poses] Resolving %d pose id(s) from database", len(ids))
        result = await _fetch_pose_data(ids, poses_col=poses_col)

    for i, pd in enumerate(result, 1):
        logger.info(
            "[poses] Pose #%d mannequin_image=%s prompt_len=%d",
            i, bool(pd.get("image_url")), len(pd.get("pose_prompt", "")),
        )
    return result


# ---------------------------------------------------------------------------
# Prompt builder (adapted from pipeline.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Prompt builders — MIRROR of ``new_pipeline.py`` (nano-banana-2 / nano-banana-pro
# Vertex reference pipeline).
#
# The text and structure below MUST stay bit-identical to ``new_pipeline.py``
# so that KIE nb2, Vertex nb2, Vertex nb-pro and Evolink all receive the same
# instruction set.  Dynamic variables (weight / height / gender / garment
# fields) are sourced from the ``req`` dict the same way as the rest of the
# service; no new inputs are introduced.
# ---------------------------------------------------------------------------

def _gender_from_req(req: dict) -> str:
    """Return ``male`` | ``female`` (defaults to ``female`` — same default
    as ``new_pipeline.py`` when the variable is left empty)."""
    g = (req.get("gender") or "").strip().lower()
    return "male" if g == "male" else "female"


def _body_description(req: dict) -> str:
    """Gender-aware body description — bit-identical to ``new_pipeline.py``.

    Inputs come from ``req``:
      * ``gender`` → ``male`` / ``female``
      * ``weight`` / ``body_weight`` → ``slim`` / ``regular`` / ``fat``
      * ``height`` / ``body_height`` → ``short`` / ``regular`` / ``tall``

    Unknown/empty values fall back to ``regular`` (weight) and ``regular``
    (height), matching the reference pipeline.
    """
    g = _gender_from_req(req)
    w = normalize_body_weight(raw_weight_from_req(req))
    h = normalize_body_height(raw_height_from_req(req))

    if g == "male":
        if w == "fat":
            body = "a heavier, fuller male build with natural belly volume and broader torso"
        elif w == "slim":
            body = "a slim, lean male build with narrow waist"
        else:
            body = "a muscular, athletic male build with defined shoulders and chest"
    else:
        if w == "fat":
            body = "a fuller, curvier female build with natural body volume"
        elif w == "slim":
            body = "a slim, slender female build with delicate frame"
        else:
            body = "a regular, healthy female build with natural proportions"

    if h == "tall":
        stature = "tall stature (approx 160-170 cm)"
    elif h == "short":
        stature = "short stature (approx 140-150 cm)"
    else:
        stature = "average stature (approx 150-160 cm)"

    return f"{body}, {stature}"


def _garment_description(req: dict) -> str:
    """Garment description — bit-identical to ``new_pipeline.py``.

    Adds the saree (Gujarati Seedha Pallu) and 3-piece (kurta + bottom +
    dupatta) styling hints when those one-piece types are detected, so
    nano-banana-pro / nano-banana-2 drape the outfit correctly.
    """
    upper_garment_type = (req.get("upper_garment_type") or "").strip()
    upper_garment_specification = (req.get("upper_garment_specification") or "").strip()
    lower_garment_type = (req.get("lower_garment_type") or "").strip()
    lower_garment_specification = (req.get("lower_garment_specification") or "").strip()
    one_piece_garment_type = (req.get("one_piece_garment_type") or "").strip()
    one_piece_garment_specification = (req.get("one_piece_garment_specification") or "").strip()
    fitting = (req.get("fitting") or "").strip()

    parts: List[str] = []
    if one_piece_garment_type:
        otype = one_piece_garment_type.lower()
        parts.append(
            f"A one-piece garment: {one_piece_garment_type} — {one_piece_garment_specification}. "
            f"Reproduce fabric, weave, texture, prints, embroidery, buttons, zippers, seams, "
            f"waistband and every small detail EXACTLY as in the front garment reference image."
        )
        if "saree" in otype:
            parts.append(
                "Drape the saree in authentic Gujarati (Seedha Pallu) style: pallu brought over "
                "the right shoulder and spread across the front. Include a matching blouse and "
                "petticoat color-coordinated with the saree if not explicitly provided."
            )
        elif "three" in otype or "3-piece" in otype or "3 piece" in otype:
            parts.append(
                "Style as a traditional 3-piece Indian set (kurta/kameez + bottom + dupatta). "
                "Drape the dupatta in Gujarati style across the front/shoulder. If any piece is "
                "missing from the reference, add a matching piece that complements the main garment."
            )
    else:
        if upper_garment_type:
            parts.append(
                f"Upper garment: {upper_garment_type} — {upper_garment_specification}. "
                f"Match fabric, texture, color, print, buttons, collar, cuffs, stitching and "
                f"every detail EXACTLY from the front (and back, when applicable) garment reference."
            )
        if lower_garment_type:
            parts.append(
                f"Lower garment: {lower_garment_type} — {lower_garment_specification}. "
                f"If no lower-garment image is provided, generate one that realistically matches "
                f"the upper garment's style, color palette and formality."
            )

    if fitting:
        parts.append(f"Overall fitting: {fitting}.")
    return " ".join(parts)


def _core_prompt(pose: PoseRuntime, req: dict) -> str:
    """Full studio-fashion prompt — bit-identical to ``_build_prompt`` in
    ``new_pipeline.py``.  Dynamic fields: ``body``, ``garment``,
    ``garment_ref_note`` and ``pose.index``.
    """
    body = _body_description(req)
    garment = _garment_description(req)

    garment_ref_note = (
        "Use the BACK garment reference image for the garment (this is a back pose)."
        if pose.is_back and req.get("back_garment_image")
        else "Use the FRONT garment reference image for the garment."
    )

    return f"""
You are generating a hyper-realistic studio fashion photograph shot on a full-frame DSLR
with an 85mm prime lens at f/2.0. Output a single photorealistic image.

=================== INPUT REFERENCES (in order) ===================
1. MODEL FACE reference — the subject's face identity (highest priority).
2. GARMENT reference(s) — front and/or back flat-lay of the garment to be worn.
3. MANNEQUIN POSE reference — ONLY for body posture (ignore its face, skin, clothes, background).
4. BACKGROUND reference — the environment the photo must look shot in.

=================== STRICT PRIORITY ORDER ===================

[PRIORITY 1 — FACE & SKIN IDENTITY LOCK + SKIN REALISM]  (MOST IMPORTANT)
- Preserve the model's face IDENTITY from the face reference with 100% fidelity:
  eye shape/color, eyebrows, nose, lips, jawline, cheekbones, ears, hairline, face proportions.
- DO NOT beautify, slim, reshape, age or de-age the face. Keep exact likeness.
- Physically based skin: render subsurface scattering (SSS) so light transmits subtly through
  ears, nostrils, lip edges and thin skin; natural warm glow where light meets skin.
- Readable micro-pores on face/neck/forearms/hands at 100% crop distance; fine peach-fuzz
  (vellus hair) on cheeks/jaw/temples/forehead catching rim/backlight.
- Micro-specular highlights from SCENE light on the oil-rich zones (nose bridge, nose tip,
  forehead T-zone, cheekbones, cupid's bow, chin) — tiny, soft, not shiny or greasy.
- Anatomy-plausible tendons on the backs of hands, subtle veins on forearms, hands and
  temples/neck where the pose exposes them; natural knuckle creases and nail-bed variation.
- Eye realism: sharp catchlights matching the scene light source, visible iris texture, red
  caruncle near tear duct, faint lower-lid moisture line, tiny lash shadows on cheek.
- Lip realism: vertical lip lines, slight asymmetry, matte-to-semi-gloss micro-variation;
  no uniform lipstick sheen unless garment styling implies it.
- Skin imperfections are mandatory and subtle: mild tonal variation, faint freckles/moles
  where the reference shows them, soft under-eye texture, tiny redness at nostril base/ears.
- ABSOLUTELY NO beauty-smoothing, plastic/waxy skin, airbrush, frequency-separation look,
  porcelain glow, or AI "clean" skin. Keep physically plausible high-frequency detail.
- Body type: {body}. Keep proportions consistent with this body type throughout the image.

[PRIORITY 2 — GARMENT FIDELITY]
- {garment_ref_note}
- {garment}
- Replicate fabric weave, sheen, drape physics, wrinkles, stitching, buttons, zippers,
  embroidery, prints, waistband, hems, cuffs, collar EXACTLY as in the reference.
- Fabric micro-texture must be visible: weave/knit direction, thread highlights, seam puckers,
  subtle fuzz on wool/cotton, specular roll-off on silk/satin, light scatter through sheer fabric.
- No color shift, no pattern drift, no invented logos or extra details.
- Footwear rule (strict): if the mannequin pose is full-body or feet are visible, footwear is
  REQUIRED. Never output barefoot feet unless explicitly requested. Choose footwear that matches
  the worn garment style, color harmony, and context (e.g., ethnic outfit -> matching
  sandals/jutti; western casual -> coordinated shoes/heels/sneakers). Footwear must be realistic,
  proportionate, and naturally integrated with pose, lighting, contact shadows, and perspective.

[PRIORITY 3 — POSE / BODY POSTURE]
- Copy the body posture from the MANNEQUIN POSE reference EXACTLY:
  hand position, finger placement, arm angles, shoulder tilt, hip sway, leg stance,
  foot placement, head tilt and overall silhouette must match the mannequin 1:1.
- Use ONLY the mannequin for posture — its face/skin/clothes/background are irrelevant.

[PRIORITY 4 — SCENE LIGHTING, SHADOWS, LENS & GROUNDING]
- Place the subject INTO the background reference as if the photo was actually shot there
  with a DSLR — NOT composited, NOT green-screened, NOT pasted. No cutout edges, no halo,
  no rim-outline around hair/shoulders, no floating subject.
- Match camera geometry to the background: camera height, horizon line, vertical perspective,
  and vanishing lines must align with room architecture/furniture.
- Subject scale must be physically correct for the scene: no oversized or undersized person.
  Keep head size, shoulder width, torso length, and leg length proportionate to nearby objects
  (chairs, walls, display units, doors, floor tiles).

- LIGHTING PHYSICS (mandatory):
  * Infer the dominant light(s) (direction, softness, color temperature, intensity) from the
    background reference and light the subject with the SAME light — including mixed sources
    (e.g. warm tungsten + cool window skylight) and global illumination.
  * Directional key + soft fill + subtle rim/hair light that wraps around the silhouette so the
    subject sits INSIDE the scene; avoid flat, even, textureless "studio cutout" lighting.
  * Bounce light from the floor/walls onto under-chin, jawline, inner arms and garment
    under-folds, tinted by the bounce surface color.
  * Micro-specular highlights on cheeks/nose/forehead/cupid's bow from the scene light — tiny,
    placement-accurate, never plasticky shine.
  * Match exposure and white balance to the background; skin, hair and garment must share the
    same scene WB, not a separate "portrait look".

- SHADOWS & AMBIENT OCCLUSION (mandatory):
  * Layered, soft, physically plausible shadows: strong near contact points (feet on floor,
    hand on hip, hair on shoulder) and softer/larger farther away.
  * Ambient occlusion in neck folds, under-chin, under-arm, hair parting, garment folds,
    between fingers — grounding the subject in the scene.
  * Single consistent shadow direction across face, body, garment, hair and surroundings
    that matches the scene light.
  * When legs/feet are visible: feet planted on the actual floor plane with correct contact
    shadow and micro-occlusion; no halo, no floating, no drifting shadow.
  * If floor is glossy/reflective, add a subtle physically correct reflection of the subject
    with proper fall-off and occlusion.

- ATMOSPHERE & DEPTH:
  * Add subtle atmospheric depth (very light haze/dust/humidity) consistent with the scene;
    do not wash out contrast.
  * Natural depth of field: subject tack-sharp at eyes/face, background with creamy 85mm f/2.0
    bokeh; foreground objects may show gentle out-of-focus blur.

- LENS REALISM:
  * Fine, even film grain / sensor noise across the whole frame (skin AND background
    must share the same grain).
  * Very slight chromatic aberration at high-contrast edges.
  * Optional subtle lens flare / veiling glare ONLY when a bright light source is in/near frame
    and the scene justifies it — never invented.
  * Natural color grading tied to the scene lighting: no oversaturation, no teal-orange
    over-grade, no sterile clinical edit.
  * Accurate, slightly distorted reflections of the full surrounding environment on glossy
    surfaces (floor, mirrors, metal, lacquered wood, jewelry, eyewear).

=================== HARD CONSTRAINTS ===================
- Single subject, fully visible as implied by the mannequin pose framing.
- Photorealistic only. No illustration, no 3D-render look, no CGI sheen, no over-saturation.
- No text, no watermark, no logo overlays.
- No extra limbs, fingers, or distortions.
- No plastic/waxy/airbrushed skin, no AI-clean complexion, no beauty-smoothing.
- No cutout/halo/float: the subject must live INSIDE the scene's light, not on top of it.
- Skin grain and background grain must match — never a clean subject on a grainy background
  or vice-versa.
- Reject results where subject looks pasted, floating, or incorrectly scaled vs background.
- Output resolution: high, print-quality.

Pose variation index: {pose.index + 1}.
""".strip()


def _ordered_refs(
    pose: PoseRuntime, req: dict, model_face_url: str, background_url: str,
) -> List[tuple[str, str]]:
    """(label, url) pairs in order: face → garment(s) → mannequin → background."""
    front = req.get("front_garment_image") or ""
    back = req.get("back_garment_image") or ""
    refs: List[tuple[str, str]] = []
    if model_face_url:
        refs.append(("MODEL FACE (identity lock)", model_face_url))
    if pose.is_back and back:
        refs.append(("GARMENT — BACK view (use this)", back))
        if front:
            refs.append(("GARMENT — FRONT view (context only)", front))
    else:
        if front:
            refs.append(("GARMENT — FRONT view (use this)", front))
        if back:
            refs.append(("GARMENT — BACK view (context only)", back))
    if pose.mannequin_url:
        refs.append(("MANNEQUIN POSE — body posture only", pose.mannequin_url))
    if background_url:
        refs.append(("BACKGROUND", background_url))
    return refs


def _prompt_with_url_preamble(
    pose: PoseRuntime, req: dict, model_face_url: str, background_url: str,
) -> tuple[str, List[str]]:
    refs = _ordered_refs(pose, req, model_face_url, background_url)
    urls = [u for _, u in refs]
    lines = [
        "The API provides reference images in this order. Treat each index as the role below:",
        "",
    ]
    for i, (label, _) in enumerate(refs):
        lines.append(f"- image[{i}] — {label}")
    return f"{chr(10).join(lines)}\n\n{_core_prompt(pose, req)}", urls


# ----- Evolink prompt (hard cap: <2000 chars) -------------------------------


def _evolink_compact_body(req: dict) -> str:
    """Ultra-short body description for Evolink's 2000-char budget.

    The full ``_body_description`` returns ~200 char weight+height prose;
    here we return just the normalized tokens (e.g. ``medium weight, average
    height``) which carries the same signal in ~40 chars.
    """
    weight = normalize_body_weight(raw_weight_from_req(req))
    height = normalize_body_height(raw_height_from_req(req))
    return f"{weight} weight, {height} height"


def _evolink_compact_garment(req: dict) -> str:
    """Terse garment description for Evolink's 2000-char budget.

    Skips long phrases like "Match fabric, texture, color, print, buttons,
    collar, cuffs, stitching EXACTLY from refs" because the core prompt
    already says "Preserve garment fidelity exactly" in PRIORITY #2.
    """
    op = (req.get("one_piece_garment_type") or "").strip()
    op_spec = (req.get("one_piece_garment_specification") or "").strip()
    ug = (req.get("upper_garment_type") or "").strip()
    us = (req.get("upper_garment_specification") or "").strip()
    lg = (req.get("lower_garment_type") or "").strip()
    ls = (req.get("lower_garment_specification") or "").strip()
    fit = (req.get("fitting") or "regular fit").strip()

    if op:
        g = f"one-piece {op}" + (f" ({op_spec})" if op_spec else "")
    else:
        bits: List[str] = []
        if ug:
            bits.append(f"top: {ug}" + (f" ({us})" if us else ""))
        if lg:
            bits.append(f"bottom: {lg}" + (f" ({ls})" if ls else ""))
        g = "; ".join(bits) if bits else "garment per reference"
    return f"{g}; fit: {fit}"


def _evolink_compact_prompt(
    pose: PoseRuntime, req: dict, model_face_url: str, background_url: str,
) -> tuple[str, List[str]]:
    """Build the Evolink prompt — same structure as ``_core_prompt`` but
    with ultra-compact body/garment fields so the total stays under
    Evolink's 2000-char prompt limit even with verbose user inputs.

    A 1990-char safety cap is retained; with compact body+garment the
    typical length is ~1100 chars and worst-case ~1400 chars, so the cap
    should never actually trip.
    """
    refs = _ordered_refs(pose, req, model_face_url, background_url)
    urls = [u for _, u in refs]

    role_map = {
        "MODEL FACE (identity lock)":            "face",
        "GARMENT — FRONT view (use this)":       "garment-front (use)",
        "GARMENT — FRONT view (context only)":   "garment-front (ctx)",
        "GARMENT — BACK view (use this)":        "garment-back (use)",
        "GARMENT — BACK view (context only)":    "garment-back (ctx)",
        "MANNEQUIN POSE — body posture only":    "mannequin-pose",
        "BACKGROUND":                            "background",
    }
    legend = "; ".join(
        f"[{i}]={role_map.get(label, label)}" for i, (label, _) in enumerate(refs)
    )

    body = _evolink_compact_body(req)
    garment = _evolink_compact_garment(req)
    # Hard-trim user fields as a last line of defense. These limits are
    # generous enough that nothing useful is lost in normal operation.
    if len(body) > 120:
        body = body[:117].rstrip() + "..."
    if len(garment) > 300:
        garment = garment[:297].rstrip() + "..."

    garment_ref_note = (
        "Use BACK garment ref (back pose)."
        if pose.is_back and req.get("back_garment_image")
        else "Use FRONT garment ref."
    )
    if pose.is_upper_body:
        footwear_rule = "Framing: upper-body only; NO footwear, feet out of frame."
    else:
        footwear_rule = (
            "Framing: full-body; footwear MANDATORY (realistic shoes matching "
            "outfit, both feet visible with contact shadow, no bare feet)."
        )

    prompt = (
        f"Images: {legend}.\n\n"
        "Hyper-realistic studio fashion photo, full-frame DSLR, 85mm f/2.0. "
        "Single photorealistic subject.\n"
        "PRIORITY: (1) EXACT face identity from face ref; "
        "(2) garment fabric/print/seams/trims/color EXACT from ref; "
        "(3) copy mannequin posture exactly (ignore its face/skin/clothes/bg); "
        "(4) subject INSIDE scene light (no cutout/halo/float).\n"
        "SKIN: SSS, visible pores, peach-fuzz on cheeks/jaw, micro-specular on "
        "nose/cheeks/forehead, subtle veins/tendons on hands/forearms, sharp eye "
        "catchlights, tiny lip lines. NO airbrush/plastic/AI-smooth skin.\n"
        "LIGHT: match scene direction+WB+exposure; key+soft fill+rim/hair wrap; "
        "bounce on under-chin/arms/garment; global illumination; mixed real sources "
        "(no flat lighting).\n"
        "SHADOWS: layered soft contact shadows + AO in neck/folds/fingers; single "
        "consistent direction across face/body/garment/bg. Feet grounded on same "
        "floor plane (no halo/float); glossy floor → subtle reflection.\n"
        "LENS: creamy 85mm f/2.0 bokeh, fine film grain (subject+bg match), slight "
        "chromatic aberration, subtle lens flare only if light source in frame, "
        "light haze for depth, natural color grade (no oversaturation), "
        "accurate slightly-distorted reflections on glossy surfaces.\n"
        f"Body: {body}.\n"
        f"{garment_ref_note}\n"
        f"Garment: {garment}.\n"
        f"{footwear_rule}\n"
        "Output: single subject, photorealistic, no text/watermark/logo, no distortions, "
        f"no cutout/halo/float. Variation: {pose.index + 1}."
    )

    if len(prompt) > 1990:
        prompt = prompt[:1987].rstrip() + "..."
    return prompt, urls


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load_image_bytes_sync(src: str, timeout: int = 30) -> tuple[bytes, str]:
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

    if mime not in ("image/jpeg", "image/png", "image/webp"):
        img = Image.open(io.BytesIO(data)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        data, mime = buf.getvalue(), "image/jpeg"
    return data, mime


async def _download_bytes(url: str, label: str = "") -> bytes:
    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(timeout=settings.KIE_HTTP_TIMEOUT) as client:
                resp = await client.get(url, follow_redirects=True)
                resp.raise_for_status()
            return resp.content
        except Exception as exc:
            logger.warning("[%s] Download attempt %d/3 failed: %s", label, attempt, exc)
            if attempt == 3:
                raise RuntimeError(f"Failed to download image after 3 attempts: {exc}")
            await asyncio.sleep(3)
    raise RuntimeError("unreachable")


def _resize_image(original_bytes: bytes, max_dimension: int) -> bytes:
    """
    Resize + encode. Delegates to the shared encoder in modal_enhance_service
    so pre-upscale variants (here) and post-upscale variants use the exact
    same lossless format (default: png_fast).
    """
    # Imported lazily to avoid a circular import at module load.
    from app.services.modal_enhance_service import (
        _encode_variant, _normalized_format,
    )

    fmt = _normalized_format()
    jpeg_q = int(settings.KIE_VARIANT_JPEG_QUALITY or 95)

    img = Image.open(io.BytesIO(original_bytes)).convert("RGB")
    w, h = img.size
    scale = max_dimension / max(w, h)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
    else:
        new_size = (w, h)

    return _encode_variant(img, new_size, fmt, jpeg_q)


# ---------------------------------------------------------------------------
# GENERATOR AGENTS (sync — invoked via asyncio.to_thread)
# ---------------------------------------------------------------------------

_KIE_CREATE_URL = "https://api.kie.ai/api/v1/jobs/createTask"
_KIE_RECORD_URL = "https://api.kie.ai/api/v1/jobs/recordInfo"


def _kie_api_key() -> str:
    key = (os.environ.get("KIE_API_KEY") or getattr(settings, "KIE_API_KEY", "") or "").strip()
    if not key:
        key = (getattr(settings, "SEEDDREAM_API_KEY", "") or "").strip()
    if not key:
        raise RuntimeError("KIE_API_KEY / SEEDDREAM_API_KEY not set")
    return key


def _kie_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_kie_api_key()}", "Content-Type": "application/json"}


def _kie_create_nb2_task_sync(prompt: str, image_urls: List[str]) -> str:
    from app.services.kie_rate_limiter import acquire_kie_token_sync

    body = {
        "model": "nano-banana-2",
        "input": {
            "prompt": prompt,
            "image_input": image_urls,
            "aspect_ratio": settings.PHOTOSHOOT_GEN_ASPECT_RATIO,
            "resolution": settings.PHOTOSHOOT_GEN_RESOLUTION,
            "output_format": "jpg",
        },
    }
    # Account-wide rate limit: KIE allows 20 createTask requests / 10 s.  Block
    # here until a slot is free (shared across every Celery worker via Redis).
    acquire_kie_token_sync()
    r = requests.post(_KIE_CREATE_URL, headers=_kie_headers(), data=json.dumps(body), timeout=60)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != 200:
        raise RuntimeError(f"KIE createTask failed: {data}")
    task_id = (data.get("data") or {}).get("taskId")
    if not task_id:
        raise RuntimeError(f"KIE createTask missing taskId: {data}")
    return str(task_id)


def _kie_poll_sync(task_id: str, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    auth = _kie_headers()["Authorization"]
    interval = float(settings.PHOTOSHOOT_KIE_POLL_INTERVAL_S)
    while time.monotonic() < deadline:
        r = requests.get(
            _KIE_RECORD_URL,
            headers={"Authorization": auth},
            params={"taskId": task_id},
            timeout=60,
        )
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
                f"KIE task failed: {data.get('failMsg') or data.get('fail_msg')!r} "
                f"(code={data.get('failCode')!r})"
            )
        time.sleep(interval)
    raise TimeoutError(f"KIE task {task_id!r} did not finish within {timeout_s}s")


def _agent_kie_nb2_sync(pose: PoseRuntime, req: dict, mf: str, bg: str) -> str:
    prompt, urls = _prompt_with_url_preamble(pose, req, mf, bg)
    task_id = _kie_create_nb2_task_sync(prompt, urls)
    return _kie_poll_sync(task_id, timeout_s=settings.PHOTOSHOOT_GEN_PER_IMAGE_TIMEOUT_S)


# ----- Evolink nano-banana-2 -------------------------------------------------

_EVOLINK_URL = "https://api.evolink.ai/v1/images/generations"
_EVOLINK_TASK_URL = "https://api.evolink.ai/v1/tasks/{task_id}"


def _evolink_headers() -> dict[str, str]:
    key = (os.environ.get("EVOLINK_API_KEY") or getattr(settings, "EVOLINK_API_KEY", "") or "").strip()
    if not key:
        raise RuntimeError("EVOLINK_API_KEY not set")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _agent_evolink_nb2_sync(pose: PoseRuntime, req: dict, mf: str, bg: str) -> str:
    # Evolink enforces a ~2000-character prompt limit, so use the compact
    # prompt builder (hard-capped at 1990 chars) instead of the full one.
    prompt, urls = _evolink_compact_prompt(pose, req, mf, bg)
    logger.info(
        "[%s][evolink] prompt_len=%d (limit 2000) framing=%s",
        pose.label, len(prompt), pose.framing_label,
    )
    payload = {
        "model": "gemini-3.1-flash-image-preview",
        "prompt": prompt,
        "size": settings.PHOTOSHOOT_GEN_ASPECT_RATIO,
        "quality": settings.PHOTOSHOOT_GEN_RESOLUTION,
        "image_urls": urls,
    }
    r = requests.post(_EVOLINK_URL, headers=_evolink_headers(), data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    created = r.json()
    task_id = created.get("id")
    if not task_id:
        raise RuntimeError(f"Evolink missing task id: {created}")

    deadline = time.monotonic() + settings.PHOTOSHOOT_GEN_PER_IMAGE_TIMEOUT_S
    while time.monotonic() < deadline:
        q = requests.get(
            _EVOLINK_TASK_URL.format(task_id=task_id),
            headers=_evolink_headers(),
            timeout=60,
        )
        q.raise_for_status()
        body = q.json()
        status = (body.get("status") or "").lower()
        if status == "completed":
            out_urls = body.get("results") or []
            if not out_urls:
                raise RuntimeError(f"Evolink completed but no results: {body}")
            return str(out_urls[0])
        if status in {"failed", "cancelled"}:
            raise RuntimeError(f"Evolink task failed: {body}")
        time.sleep(3.0)
    raise TimeoutError(
        f"Evolink task {task_id!r} did not finish within "
        f"{settings.PHOTOSHOOT_GEN_PER_IMAGE_TIMEOUT_S}s"
    )


# ----- Vertex (google-genai) nano-banana-2 & nano-banana-pro -----------------

def _vertex_client():
    if genai is None:
        raise RuntimeError("google-genai not installed")
    api_key = (
        os.environ.get("GOOGLE_CLOUD_API_KEY")
        or getattr(settings, "GOOGLE_CLOUD_API_KEY", "")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError("GOOGLE_CLOUD_API_KEY not set")
    return genai.Client(vertexai=True, api_key=api_key)


def _vertex_generate_sync(
    model_id: str,
    pose: PoseRuntime,
    req: dict,
    mf: str,
    bg: str,
    use_thinking: bool,
) -> bytes:
    client = _vertex_client()
    front = req.get("front_garment_image") or ""
    back = req.get("back_garment_image") or ""

    # Reference ordering mirrors ``_build_parts`` in ``new_pipeline.py``
    # exactly so that nano-banana-2 / nano-banana-pro see the same
    # multimodal layout: face → garment(use) → garment(context) →
    # mannequin → background → prompt text.
    parts: List[Any] = []

    if mf:
        parts.append(genai_types.Part.from_text(text="[REF 1/4] MODEL FACE (identity lock):"))
        face_bytes, face_mime = _load_image_bytes_sync(mf)
        parts.append(genai_types.Part.from_bytes(data=face_bytes, mime_type=face_mime))

    if pose.is_back and back:
        d, m = _load_image_bytes_sync(back)
        parts.append(genai_types.Part.from_text(text="[REF 2/4] GARMENT — BACK view (use this):"))
        parts.append(genai_types.Part.from_bytes(data=d, mime_type=m))
        if front:
            d2, m2 = _load_image_bytes_sync(front)
            parts.append(genai_types.Part.from_text(text="[REF 2b] GARMENT — FRONT view (context only):"))
            parts.append(genai_types.Part.from_bytes(data=d2, mime_type=m2))
    else:
        if front:
            d, m = _load_image_bytes_sync(front)
            parts.append(genai_types.Part.from_text(text="[REF 2/4] GARMENT — FRONT view (use this):"))
            parts.append(genai_types.Part.from_bytes(data=d, mime_type=m))
        if back:
            d2, m2 = _load_image_bytes_sync(back)
            parts.append(genai_types.Part.from_text(text="[REF 2b] GARMENT — BACK view (context only):"))
            parts.append(genai_types.Part.from_bytes(data=d2, mime_type=m2))

    if pose.mannequin_url:
        d3, m3 = _load_image_bytes_sync(pose.mannequin_url)
        parts.append(genai_types.Part.from_text(
            text="[REF 3/4] MANNEQUIN POSE — copy BODY POSTURE ONLY (ignore face/skin/clothes/bg):"
        ))
        parts.append(genai_types.Part.from_bytes(data=d3, mime_type=m3))

    if bg:
        d4, m4 = _load_image_bytes_sync(bg)
        parts.append(genai_types.Part.from_text(text="[REF 4/4] BACKGROUND — scene the photo is shot in:"))
        parts.append(genai_types.Part.from_bytes(data=d4, mime_type=m4))

    parts.append(genai_types.Part.from_text(text=_core_prompt(pose, req)))

    safety = [
        genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
        genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
        genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]
    image_config = genai_types.ImageConfig(
        aspect_ratio=settings.PHOTOSHOOT_GEN_ASPECT_RATIO,
        image_size=settings.PHOTOSHOOT_GEN_RESOLUTION,
        output_mime_type="image/jpeg",
    )
    thinking_config = (
        genai_types.ThinkingConfig(thinking_level="MINIMAL") if use_thinking else None
    )
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
    for chunk in client.models.generate_content_stream(
        model=model_id, contents=contents, config=cfg,
    ):
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
        raise RuntimeError(f"No image returned by Vertex model {model_id} for pose {pose.index}")
    return bytes(buffer)


def _agent_vertex_nb2_sync(pose: PoseRuntime, req: dict, mf: str, bg: str) -> bytes:
    return _vertex_generate_sync(
        "gemini-3.1-flash-image-preview", pose, req, mf, bg, use_thinking=True,
    )


def _agent_vertex_nbpro_sync(pose: PoseRuntime, req: dict, mf: str, bg: str) -> bytes:
    return _vertex_generate_sync(
        "gemini-3-pro-image-preview", pose, req, mf, bg, use_thinking=False,
    )


GENERATOR_AGENTS: dict[str, Callable[[PoseRuntime, dict, str, str], Any]] = {
    "kie_nb2":      _agent_kie_nb2_sync,
    "vertex_nb2":   _agent_vertex_nb2_sync,
    "vertex_nbpro": _agent_vertex_nbpro_sync,
    "evolink_nb2":  _agent_evolink_nb2_sync,
}


# ---------------------------------------------------------------------------
# Per-pose post-generation stream:
#   materialize → R2 upload → kie upscale → push to photoshoots doc →
#   deduct credits — all runs per-pose as soon as generation finishes.
# ---------------------------------------------------------------------------

async def _materialize_and_upscale(pose: PoseRuntime, ctx: PipelineContext) -> None:
    """Download / resize / upload / upscale + live-stream updates to Mongo."""
    image_id = str(uuid.uuid4())
    prefix = f"photoshoots/{ctx.photoshoot_id}/{image_id}"

    try:
        # 1) Get 4K bytes
        if pose.generated_bytes is not None:
            bytes_4k = pose.generated_bytes
            source_image_url = pose.generated_url or ""
            logger.info(
                "[%s] Using in-memory bytes from %s (%d bytes)",
                pose.label, pose.generator_used, len(bytes_4k),
            )
        elif pose.generated_url:
            source_image_url = pose.generated_url
            bytes_4k = await _download_bytes(pose.generated_url, pose.label)
        else:
            raise RuntimeError(f"[{pose.label}] no generated image available")

        # 2) Re-encode 4K (from generator's PNG to target format) + resize to 2K/1K,
        #    all three in parallel in a thread pool so the event loop stays free.
        loop = asyncio.get_running_loop()
        bytes_4k_enc, bytes_2k, bytes_1k = await asyncio.gather(
            loop.run_in_executor(None, _resize_image, bytes_4k, 4096),
            loop.run_in_executor(None, _resize_image, bytes_4k, 2048),
            loop.run_in_executor(None, _resize_image, bytes_4k, 1024),
        )

        # 3) Upload 4K/2K/1K to R2 (JPEG by default — see KIE_VARIANT_FORMAT)
        from app.services.modal_enhance_service import (
            variant_content_type, variant_extension,
        )
        ext = variant_extension()
        ctype = variant_content_type()
        url_4k, url_2k, url_1k = await asyncio.gather(
            upload_bytes_to_r2(bytes_4k_enc, f"{prefix}_4k.{ext}", ctype),
            upload_bytes_to_r2(bytes_2k,     f"{prefix}_2k.{ext}", ctype),
            upload_bytes_to_r2(bytes_1k,     f"{prefix}_1k.{ext}", ctype),
        )
        effective_source_url = source_image_url or url_4k

        # 4) Configured upscaler (this also inserts the upscaling_data doc)
        upscale_result = await enhance_and_upload(
            image_bytes=bytes_4k,
            photoshoot_id=ctx.photoshoot_id,
            image_id=image_id,
            seeddream_4k_url=url_4k,
            seeddream_2k_url=url_2k,
            seeddream_1k_url=url_1k,
            source_image_url=effective_source_url,
            upscaling_col=ctx.upscaling_col,
        )
        display_image = upscale_result.get("2k_upscaled") or url_2k

        output_image = {
            "image_id":       image_id,
            "pose_prompt":    pose.pose_prompt,
            "pose_image_url": pose.mannequin_url,
            "image":          display_image,
        }
        pose.output_image = output_image

        # 5) STREAM — push into photoshoots doc and deduct credits NOW.
        now = datetime.now(timezone.utc)
        await ctx.photoshoots_col.update_one(
            {"photoshoot_id": ctx.photoshoot_id},
            {
                "$push": {"output_images": output_image},
                "$set":  {"updated_at": now},
            },
        )
        logger.info(
            "[%s] streamed into photoshoots.output_images (image_id=%s)",
            pose.label, image_id,
        )

        await _deduct_one_pose_credit(pose, image_id, ctx)

    except Exception as exc:
        pose.last_error = f"stage-2 failed: {exc}"
        logger.error("[%s] Stage-2 FAILED: %s", pose.label, exc)
        now = datetime.now(timezone.utc)
        await ctx.photoshoots_col.update_one(
            {"photoshoot_id": ctx.photoshoot_id},
            {
                "$push": {"failed_poses": {
                    "pose_index":      pose.index,
                    "error":           pose.last_error,
                    "providers_tried": pose.providers_tried,
                }},
                "$set":  {"updated_at": now},
            },
        )


async def _deduct_one_pose_credit(
    pose: PoseRuntime, image_id: str, ctx: PipelineContext,
) -> None:
    """Deduct `credit_per_image` for ONE successful pose + write a history record."""
    cpi = ctx.credit_per_image
    user_id = ctx.req.get("user_id")
    if not user_id or cpi <= 0:
        return

    now = datetime.now(timezone.utc)
    user = await ctx.users_col.find_one({"user_id": user_id})
    if not user:
        logger.error("[credits] user not found: %s — skipping deduction for %s", user_id, pose.label)
        return

    old_credits = float(user.get("credits", 0))
    new_credits = round(old_credits - cpi, 4)
    await ctx.users_col.update_one(
        {"user_id": user_id},
        {"$set": {"credits": new_credits, "updated_at": now}},
    )

    regen_type = ctx.req.get("regeneration_type", "") or ""
    regen_from = ctx.req.get("regenerate_photoshoot_id", "") or ""
    feature_name = (
        "photoshoot_regenerate" if regen_type == "regenerate" else "photoshoot_generation"
    )
    notes = f"Photoshoot {ctx.photoshoot_id} — {pose.label}"
    if regen_type:
        notes = f"{regen_type} — photoshoot {ctx.photoshoot_id} ({pose.label})"
        if regen_from:
            notes += f" (from {regen_from})"

    history_doc = {
        "history_id":       str(uuid.uuid4()),
        "user_id":          user_id,
        "feature_name":     feature_name,
        "credit":           cpi,
        "credit_per_image": cpi,
        "image_ids":        [image_id],
        "type":             "deduct",
        "thumbnail_image":  "",
        "notes":            notes,
        "photoshoot_id":    ctx.photoshoot_id,
        "created_at":       now,
    }
    if regen_type:
        history_doc["regeneration_type"]        = regen_type
        history_doc["regenerate_photoshoot_id"] = regen_from

    await ctx.history_col.insert_one(history_doc)

    # Update aggregate running totals on the photoshoot doc so clients see real-time totals.
    ctx.credit_total_accum = round(ctx.credit_total_accum + cpi, 4)
    ctx.successful_image_ids.append(image_id)
    await ctx.photoshoots_col.update_one(
        {"photoshoot_id": ctx.photoshoot_id},
        {"$set": {
            "total_credit":       ctx.credit_total_accum,
            "is_credit_deducted": True,
            "updated_at":         now,
        }},
    )
    logger.info(
        "[%s] credit deducted=%.2f (running total=%.2f, credits %.4f → %.4f)",
        pose.label, cpi, ctx.credit_total_accum, old_credits, new_credits,
    )


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

async def _node_init(state: PipelineState) -> PipelineState:
    ctx = state["ctx"]
    logger.info(
        "[graph] init photoshoot=%s poses=%d",
        ctx.photoshoot_id, len(ctx.poses),
    )
    return state


def _make_gen_node(policy: GenPolicy):
    """Build a LangGraph node that runs one provider across all pending poses."""

    async def _node(state: PipelineState) -> PipelineState:
        ctx = state["ctx"]
        pending = [p for p in ctx.poses if not p.is_generated]
        if not pending:
            logger.info("[graph][%s] skipped (no pending poses)", policy.name)
            return state

        logger.info(
            "[graph][%s] batch start poses=%d concurrency=%d retries=%d",
            policy.name, len(pending), policy.concurrency, policy.retries,
        )
        sem = asyncio.Semaphore(policy.concurrency)
        fn = GENERATOR_AGENTS[policy.name]

        async def _one(pose: PoseRuntime) -> None:
            pose.providers_tried.append(policy.name)
            async with sem:
                last_err: Optional[Exception] = None
                for attempt in range(1, policy.retries + 1):
                    logger.info(
                        "[%s][%s] attempt %d/%d",
                        pose.label, policy.name, attempt, policy.retries,
                    )
                    try:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(
                                fn, pose, ctx.req, ctx.model_face_url, ctx.background_url,
                            ),
                            timeout=float(settings.PHOTOSHOOT_GEN_PER_IMAGE_TIMEOUT_S),
                        )
                        if isinstance(result, (bytes, bytearray)):
                            pose.generated_bytes = bytes(result)
                        elif isinstance(result, str) and result:
                            pose.generated_url = result
                        else:
                            raise RuntimeError("empty result")
                        pose.generator_used = policy.name
                        logger.info(
                            "[%s][%s] OK (url=%s bytes=%s) — firing stage-2",
                            pose.label, policy.name,
                            bool(pose.generated_url),
                            len(pose.generated_bytes) if pose.generated_bytes else 0,
                        )
                        # Fire stage-2 immediately — do NOT wait for other poses.
                        pose.upscale_task = asyncio.create_task(
                            _materialize_and_upscale(pose, ctx)
                        )
                        return
                    except Exception as exc:
                        last_err = exc
                        logger.warning(
                            "[%s][%s] attempt %d failed: %r",
                            pose.label, policy.name, attempt, exc,
                        )
                        if attempt < policy.retries:
                            delay = min(2.0 * (2 ** (attempt - 1)) + random.uniform(0.0, 1.0), 30.0)
                            await asyncio.sleep(delay)
                pose.last_error = repr(last_err) if last_err else "unknown"
                logger.error(
                    "[%s][%s] EXHAUSTED %d retries",
                    pose.label, policy.name, policy.retries,
                )

        await asyncio.gather(*[_one(p) for p in pending])
        ok = sum(1 for p in pending if p.is_generated)
        logger.info("[graph][%s] batch done ok=%d/%d", policy.name, ok, len(pending))
        return state

    _node.__name__ = f"gen_{policy.name}"
    return _node


def _route_after_gen(state: PipelineState) -> str:
    """Decide whether to fall through to the next provider or finalize."""
    ctx = state["ctx"]
    missing = [p for p in ctx.poses if not p.is_generated]
    if not missing:
        return "finalize"
    return "continue"


async def _node_finalize(state: PipelineState) -> PipelineState:
    ctx = state["ctx"]
    # Collect stage-1 losers for structured failed_poses logging.
    stage1_losers = [p for p in ctx.poses if not p.is_generated]
    if stage1_losers:
        now = datetime.now(timezone.utc)
        to_push = [
            {
                "pose_index":      p.index,
                "error":           p.last_error or "stage-1 generation failed for all providers",
                "providers_tried": p.providers_tried,
            }
            for p in stage1_losers
        ]
        await ctx.photoshoots_col.update_one(
            {"photoshoot_id": ctx.photoshoot_id},
            {
                "$push": {"failed_poses": {"$each": to_push}},
                "$set":  {"updated_at": now},
            },
        )
        for p in stage1_losers:
            logger.error(
                "[graph] %s FAILED in stage-1 (tried=%s): %s",
                p.label, p.providers_tried, p.last_error,
            )

    # Wait for any in-flight stage-2 tasks.
    pending_tasks = [p.upscale_task for p in ctx.poses if p.upscale_task]
    if pending_tasks:
        logger.info("[graph] finalize: awaiting %d in-flight stage-2 task(s)", len(pending_tasks))
        await asyncio.gather(*pending_tasks, return_exceptions=True)

    # Final status
    ok = sum(1 for p in ctx.poses if p.output_image is not None)
    failed = len(ctx.poses) - ok
    if failed == 0:
        final_status = "completed"
    elif ok > 0:
        final_status = "partial"
    else:
        final_status = "failed"

    now = datetime.now(timezone.utc)
    await ctx.photoshoots_col.update_one(
        {"photoshoot_id": ctx.photoshoot_id},
        {"$set": {
            "status":       final_status,
            "is_completed": True,
            "updated_at":   now,
        }},
    )
    logger.info(
        "[graph] finalize done photoshoot=%s ok=%d/%d status=%s",
        ctx.photoshoot_id, ok, len(ctx.poses), final_status,
    )
    return state


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_photoshoot_graph():
    """Compile the LangGraph StateGraph for a photoshoot run.

    Graph shape (each provider → next provider if any pose still missing,
    else → finalize):

        init → gen_kie_nb2 → gen_vertex_nb2 → gen_vertex_nbpro → gen_evolink_nb2 → finalize → END
    """
    g = StateGraph(PipelineState)
    g.add_node("init", _node_init)

    gen_node_names: List[str] = []
    for policy in GEN_POLICIES:
        node_name = f"gen_{policy.name}"
        g.add_node(node_name, _make_gen_node(policy))
        gen_node_names.append(node_name)

    g.add_node("finalize", _node_finalize)

    g.set_entry_point("init")
    g.add_edge("init", gen_node_names[0])

    for idx, node_name in enumerate(gen_node_names):
        next_node = gen_node_names[idx + 1] if idx + 1 < len(gen_node_names) else "finalize"
        g.add_conditional_edges(
            node_name,
            _route_after_gen,
            {
                "continue": next_node,
                "finalize": "finalize",
            },
        )

    g.add_edge("finalize", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Public entry point (called by Celery task)
# ---------------------------------------------------------------------------

async def run_photoshoot_job(photoshoot_id: str, req: dict, motor_client=None) -> None:
    """Run the multi-agent LangGraph photoshoot pipeline.

    Per-pose results are streamed into the photoshoot document as they
    complete — the caller does NOT need to wait for every pose.

    On top-level failure the photoshoot doc is marked ``status='failed'``.
    """
    logger.info("=" * 70)
    logger.info("[job] Photoshoot job STARTED — photoshoot_id=%s", photoshoot_id)
    logger.info(
        "[job] user_id=%s | poses_ids=%d | pose_data=%d",
        req.get("user_id"),
        len(req.get("poses_ids") or []),
        len(req.get("pose_data") or []),
    )
    logger.info("=" * 70)

    if motor_client is not None:
        from app.config import settings as _s
        _db = motor_client[_s.MONGO_DB_NAME]
        photoshoots_col = _db["photoshoots"]
        poses_col       = _db["poses_data"]
        upscaling_col   = _db["upscaling_data"]
        backgrounds_col = _db["backgrounds"]
        model_faces_col = _db["model_faces"]
        users_col       = _db["users"]
        history_col     = _db["credit_history"]
    else:
        photoshoots_col = get_photoshoots_collection()
        poses_col       = None
        upscaling_col   = None
        backgrounds_col = get_backgrounds_collection()
        model_faces_col = get_model_faces_collection()
        users_col       = get_users_collection()
        history_col     = get_credit_history_collection()

    job_start = time.time()

    try:
        # Step 1 — resolve pose data
        pose_data_list = await resolve_poses(req, poses_col=poses_col)
        if not pose_data_list:
            raise ValueError("No poses could be resolved.")

        # Step 2 — background + model face URLs
        bg_doc = await backgrounds_col.find_one({"background_id": req["background_id"]})
        if not bg_doc:
            raise ValueError(f"Background not found: {req['background_id']}")
        background_url = bg_doc["background_url"]

        mf_doc = await model_faces_col.find_one({"model_id": req["model_id"]})
        if not mf_doc:
            raise ValueError(f"Model face not found: {req['model_id']}")
        model_face_url = mf_doc["face_url"]

        # Step 3 — build runtime context
        poses = [
            PoseRuntime(
                index=i,
                pose_prompt=(pd.get("pose_prompt") or "").strip(),
                mannequin_url=(pd.get("image_url") or "").strip(),
            )
            for i, pd in enumerate(pose_data_list, 1)
        ]
        ctx = PipelineContext(
            photoshoot_id=photoshoot_id,
            req=req,
            poses=poses,
            model_face_url=model_face_url,
            background_url=background_url,
            photoshoots_col=photoshoots_col,
            upscaling_col=upscaling_col,
            users_col=users_col,
            history_col=history_col,
            credit_per_image=float(
                req.get("credit_per_image", settings.CREDIT_SINGLE_PHOTOSHOOT_PER_IMAGE)
            ),
        )

        # Reset output_images/failed_poses so repeated runs don't duplicate.
        await photoshoots_col.update_one(
            {"photoshoot_id": photoshoot_id},
            {"$set": {
                "output_images":      [],
                "failed_poses":       [],
                "total_credit":       0.0,
                "is_credit_deducted": False,
                "status":             "processing",
                "updated_at":         datetime.now(timezone.utc),
            }},
        )

        # Step 4 — compile and run the LangGraph pipeline.
        graph = build_photoshoot_graph()
        await graph.ainvoke({"ctx": ctx})

        elapsed = round(time.time() - job_start, 1)
        logger.info("=" * 70)
        logger.info(
            "[job] Photoshoot job FINISHED — photoshoot_id=%s | elapsed=%.1fs",
            photoshoot_id, elapsed,
        )
        logger.info("=" * 70)

    except Exception as exc:
        elapsed = round(time.time() - job_start, 1)
        logger.error(
            "[job] Photoshoot job FAILED — photoshoot_id=%s | error=%s | elapsed=%.1fs",
            photoshoot_id, exc, elapsed,
        )
        await photoshoots_col.update_one(
            {"photoshoot_id": photoshoot_id},
            {"$set": {
                "status":     "failed",
                "error":      str(exc),
                "updated_at": datetime.now(timezone.utc),
            }},
        )


# ---------------------------------------------------------------------------
# Batch helpers (unchanged API — used by router)
# ---------------------------------------------------------------------------

def merge_photoshoot_batch_configs(default_config: dict, list_item: dict) -> dict:
    """Merge shared defaults with one row. ``list_item`` keys override ``default_config``.

    Keys with value ``None`` are ignored (they do not override).
    """
    base = {k: v for k, v in default_config.items() if v is not None}
    over = {k: v for k, v in list_item.items() if v is not None}
    return {**base, **over}


def count_poses_in_merged_config(merged: dict) -> int:
    """Return pose count for credit calculation."""
    pd = merged.get("pose_data")
    if pd and isinstance(pd, list):
        return len(pd)
    return len(merged.get("poses_ids") or [])
