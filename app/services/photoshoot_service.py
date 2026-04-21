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


GEN_POLICIES: List[GenPolicy] = [
    GenPolicy(name="kie_nb2",       concurrency=10, retries=2),
    GenPolicy(name="vertex_nb2",    concurrency=2,  retries=4),
    GenPolicy(name="vertex_nbpro",  concurrency=2,  retries=4),
    GenPolicy(name="evolink_nb2",   concurrency=10, retries=10),
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

def _garment_description(req: dict) -> str:
    ug = (req.get("upper_garment_type") or "").strip()
    us = (req.get("upper_garment_specification") or "").strip()
    lg = (req.get("lower_garment_type") or "").strip()
    ls = (req.get("lower_garment_specification") or "").strip()
    op = (req.get("one_piece_garment_type") or "").strip()
    os_ = (req.get("one_piece_garment_specification") or "").strip()
    fitting = (req.get("fitting") or "regular fit").strip()

    parts: List[str] = []
    if op:
        parts.append(
            f"A one-piece garment: {op}"
            + (f" — {os_}" if os_ else "")
            + ". Reproduce fabric, weave, texture, prints, embroidery, buttons, zippers, "
              "seams, waistband and every small detail EXACTLY as in the front garment reference image."
        )
    else:
        if ug:
            parts.append(
                f"Upper garment: {ug}"
                + (f" — {us}" if us else "")
                + ". Match fabric, texture, color, print, buttons, collar, cuffs, stitching EXACTLY from refs."
            )
        if lg:
            parts.append(
                f"Lower garment: {lg}"
                + (f" — {ls}" if ls else "")
                + ". If no lower-garment image is provided, generate a matching one."
            )
    if fitting:
        parts.append(f"Overall fitting: {fitting}.")
    return " ".join(parts)


def _body_description(req: dict) -> str:
    weight = normalize_body_weight(raw_weight_from_req(req))
    height = normalize_body_height(raw_height_from_req(req))
    return f"{body_weight_instruction_long(weight)} {body_height_instruction_long(height)}"


def _footwear_rule(pose: PoseRuntime) -> str:
    """Pose-aware framing + footwear rule driven by the mannequin image.

    The framing of the OUTPUT must match the framing shown in the mannequin
    reference exactly:
      * mannequin shows full body  → generate full body with footwear.
      * mannequin shows upper body → generate upper body only, no feet, no shoes.

    These two cases are written as hard, non-negotiable rules because the
    model otherwise tends to "split the difference" (e.g. half-body crop even
    when the mannequin shows the full body, or hallucinate shoes in a
    head-to-waist crop).
    """
    if pose.is_upper_body:
        return (
            "FRAMING (MANDATORY, matches mannequin reference): UPPER-BODY ONLY. "
            "The mannequin reference shows the subject from head to waist/"
            "torso, so the output MUST show the same crop — head to waist, "
            "feet and legs entirely out of frame. "
            "Do NOT render footwear, do NOT render shoes, sandals, heels, or "
            "any leg/feet elements. Any hint of footwear is a hard rejection."
        )
    return (
        "FRAMING (MANDATORY, matches mannequin reference): FULL-BODY, head "
        "to feet. The mannequin reference shows the complete body, so the "
        "output MUST show the complete body with both feet fully inside the "
        "frame — no cropping at the ankles, knees, or thighs. "
        "FOOTWEAR IS MANDATORY: render realistic, well-fitting shoes that "
        "match the outfit style (default to neutral studio-appropriate "
        "footwear if none is specified — closed-toe leather shoes, clean "
        "sneakers, or minimal heels depending on outfit formality). "
        "Both feet visible, properly laced/styled, with natural contact "
        "shadows on the ground. Bare feet, missing shoes, or ankle-cropped "
        "framing are hard rejections."
    )


def _core_prompt(pose: PoseRuntime, req: dict) -> str:
    body = _body_description(req)
    garment = _garment_description(req)
    garment_ref_note = (
        "Use the BACK garment reference image (this is a back pose)."
        if pose.is_back and req.get("back_garment_image")
        else "Use the FRONT garment reference image."
    )
    pose_text = (pose.pose_prompt or "").strip() or "Natural relaxed full-body fashion model pose."
    bg_type = (req.get("background_type") or "").strip().lower() or "general"
    ornaments = (req.get("ornaments") or "none").strip() or "none"
    footwear = _footwear_rule(pose)

    return (
        # === 1. CAMERA / RENDER INTENT — concrete real-camera language ========
        "Generate ONE candid editorial fashion photograph — a real, handheld, "
        "in-camera exposure shot on a full-frame mirrorless body (Sony A7R V / "
        "Leica Q3) with a fast prime lens (24mm f/2.8 environmental or 85mm "
        "f/1.8 portrait, chosen to fit the framing). Render as if a skilled "
        "human photographer pressed the shutter in a real location — NOT a "
        "CG render, NOT a polished studio key-art, NOT AI-smooth, NOT "
        "symmetrically composed. Single photorealistic frame only.\n\n"

        # === 2. PRIORITY LOCK — unchanged hard constraints ====================
        "ABSOLUTE PRIORITIES (in order, must be satisfied before aesthetics):\n"
        "1) Preserve exact face identity from the face reference — same bone "
        "   structure, eye shape, skin tone, lip shape, hair color and "
        "   hairline. Zero drift.\n"
        "2) Reproduce the garment EXACTLY from the garment reference — fabric "
        "   weave, prints, seams, stitching, trims, buttons, zippers, "
        "   hardware, color. No reinterpretation.\n"
        "3) Copy the mannequin's POSTURE and FRAMING exactly (see dedicated "
        "   section below). This is non-negotiable.\n"
        "4) Integrate the subject into the background with correct "
        "   perspective, matching light direction, color-cast spill, and "
        "   physical contact shadows.\n\n"

        # === 2b. POSTURE MIMICRY — dedicated non-negotiable block =============
        "POSE & POSTURE (MANDATORY — the mannequin reference is a POSTURE "
        "TEMPLATE, match it exactly):\n"
        "- Copy the mannequin's body posture 1:1: overall stance, weight "
        "  distribution, which leg bears weight, hip angle, shoulder line, "
        "  spine curvature, chest direction, and head tilt.\n"
        "- Copy LEG position exactly: same leg bent / straight, same foot "
        "  placement and spacing, same knee angle, same ankle rotation.\n"
        "- Copy HAND and ARM position exactly: same arm angle at shoulder "
        "  and elbow, same wrist rotation, same finger position (open, "
        "  fist, pinch, pointing, touching garment, on hip, etc.), same "
        "  left-vs-right asymmetry. If one hand is raised and the other is "
        "  down, mirror that exactly — do NOT substitute a 'more flattering' "
        "  generic model pose.\n"
        "- Copy the overall body orientation (facing camera / 3-quarter / "
        "  profile / back) exactly as shown in the mannequin.\n"
        "- Ignore the mannequin's face, skin tone, clothing, proportions, "
        "  and background — ONLY its posture and framing transfer.\n"
        "- Do NOT invent a new pose. Do NOT 'improve' the pose. Do NOT "
        "  default to a catalogue contrapposto unless the mannequin shows "
        "  that exact stance.\n\n"

        # === 2c. FRAMING / CROP — mandatory match to mannequin =================
        "FRAMING & CROP (MANDATORY — match the mannequin reference):\n"
        "- If the mannequin reference shows the FULL body (head to feet), "
        "  the output MUST be a full-body image, head to feet, with both "
        "  feet fully inside the frame and FOOTWEAR ON (shoes are required; "
        "  bare feet are rejected).\n"
        "- If the mannequin reference shows only the UPPER body (head to "
        "  waist/torso), the output MUST be an upper-body crop with the "
        "  exact same crop line — no feet, no legs, no footwear rendered.\n"
        "- Do NOT change the crop relative to the mannequin. Do NOT "
        "  zoom out to add missing feet, and do NOT zoom in to hide feet. "
        "  The mannequin's framing IS the output's framing.\n\n"

        # === 3. REALISM LADDER — micro-textures, anti-AI markers ==============
        "REALISM / SKIN / TEXTURE — pursue imperfection, not polish:\n"
        "- Skin shows real pores, peach fuzz, micro-freckles, slight sebum "
        "  shine on T-zone, subsurface scattering in ears/nostrils, faint "
        "  redness near nose bridge and cheekbones. Matte-satin finish, never "
        "  plastic, never airbrushed, never uniform tone.\n"
        "- Hair has stray flyaways around the hairline, natural asymmetry in "
        "  parting, catch-light in individual strands — avoid helmet-shaped "
        "  'AI hair'.\n"
        "- Lips show subtle texture lines, slight gloss variation, micro-"
        "  cracks; teeth (if visible) are naturally uneven in tone.\n"
        "- Garment fabric shows weave, pile direction, thread ends, small "
        "  wrinkles at joints and fabric memory where the body bends; "
        "  accessories show fingerprints, dust, micro-scratches.\n"
        "- Background surfaces have dust, small marks, material grain; no "
        "  surface is perfectly clean or perfectly flat.\n\n"

        # === 4. LIGHT PHYSICS ================================================
        "LIGHTING — real physics, mixed sources:\n"
        "- Use motivated, mixed real-world lighting (window + bounce, or sun "
        "  + shade, or practical + fill). Add directional key with visible "
        "  falloff; avoid flat even lighting.\n"
        "- Global illumination: color spill from walls, floor, and the "
        "  garment onto skin; ambient occlusion tight under the jaw, collar, "
        "  armpits, and where fabric meets body.\n"
        "- Specular highlights follow skin oil map (forehead, nose tip, "
        "  cupid's bow, chin); specular occlusion damps highlights inside "
        "  pores. Slight blown highlight allowed on one cheek or shoulder.\n"
        "- Color temperature consistent across subject and background — "
        "  whatever illuminates the background also illuminates the subject.\n\n"

        # === 5. SUBJECT ↔ BACKGROUND BLENDING ================================
        "SUBJECT-BACKGROUND INTEGRATION — this is how the model gets GROUNDED:\n"
        "- Feet / lowest contact point casts a soft, layered contact shadow "
        "  with a dark core and diffuse penumbra; additional long cast shadow "
        "  if directional light is present.\n"
        "- Ambient occlusion visible where clothing folds touch skin, where "
        "  the subject is near a wall, and along the floor seam.\n"
        "- Background color bounces onto the side of the face, neck, and "
        "  garment nearest to it — no clean cut-out look.\n"
        "- Atmospheric haze / humidity / fine dust suspended in the air for "
        "  depth separation; slight volumetric light if a source is directional.\n"
        "- Depth of field: sharp focus on the face/eyes, progressive falloff "
        "  on hands and garment edges, strong creamy bokeh on background; "
        "  bokeh shape is optical (cats-eye near edges), not uniformly round.\n"
        "- Accurate, slightly distorted reflections of the full environment "
        "  on any shiny surface (jewelry, eyes, patent leather, glass).\n\n"

        # === 6. LENS / CAMERA ARTIFACTS ======================================
        "LENS & CAMERA REALISM — subtle, never exaggerated:\n"
        "- Fine film-grain-like sensor noise, denser in shadows.\n"
        "- Slight chromatic aberration at high-contrast edges.\n"
        "- A whisper of lens flare only if a light source is in/near frame.\n"
        "- Barely perceptible motion feel (handheld, not tripod-locked).\n"
        "- Color grading: natural skin tones, mid-contrast, no teal-and-"
        "  orange over-grade, no Instagram filter, no oversaturation.\n\n"

        # === 7. COMPOSITION / ANTI-SYMMETRY ==================================
        "COMPOSITION — candid snapshot feel:\n"
        "- Handheld, imperfect framing; slight off-axis or subtle Dutch angle "
        "  acceptable; eye-line is NOT perfectly centered.\n"
        "- Documentary / editorial snapshot aesthetic — think a 35mm street-"
        "  fashion frame, not a beauty-counter advert.\n"
        "- Natural pose weight distribution; correct physical contact where "
        "  hands/arms touch body or garment (visible pressure and fabric "
        "  deformation, not floating).\n\n"

        # === 8. SUBJECT / WARDROBE / SCENE INPUTS ============================
        f"Body: {body}\n"
        f"Garment rule: {garment_ref_note}\n"
        f"Garment details: {garment}\n"
        f"Pose: {pose_text}\n"
        f"{footwear}\n"
        f"Background type: {bg_type}. Ornaments: {ornaments}.\n\n"

        # === 9. HARD EXCLUSIONS (phrased as affirmative opposites) ===========
        "STRICT: output is a single photorealistic human subject only. No "
        "text, no watermarks, no logos, no UI overlays, no duplicates, no "
        "extra limbs or fingers, no warped hands. No HDR crunch. No CGI "
        "sheen. No plastic skin. No AI-smooth background. No perfect teeth. "
        "No catalogue-style frontal centering.\n\n"

        # === 10. FINAL REINFORCEMENT — last thing the model reads ============
        "FINAL REMINDER (highest weight):\n"
        "1) POSTURE: the mannequin reference defines body pose, hand "
        "   position, leg position — copy it EXACTLY, do not substitute a "
        "   generic model pose.\n"
        "2) FRAMING: the mannequin reference defines the crop — "
        "   full-body mannequin → full-body output with footwear ON; "
        "   upper-body mannequin → upper-body output with no feet and no "
        "   footwear. No mixing.\n"
        f"Pose variation index: {pose.index}."
    )


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


# ----- Evolink-specific prompt (hard cap: <2000 chars) ----------------------

def _compact_garment(req: dict) -> str:
    """Terse garment description for the Evolink prompt budget."""
    ug = (req.get("upper_garment_type") or "").strip()
    us = (req.get("upper_garment_specification") or "").strip()
    lg = (req.get("lower_garment_type") or "").strip()
    ls = (req.get("lower_garment_specification") or "").strip()
    op = (req.get("one_piece_garment_type") or "").strip()
    os_ = (req.get("one_piece_garment_specification") or "").strip()
    fit = (req.get("fitting") or "regular fit").strip()

    if op:
        g = f"one-piece {op}" + (f" ({os_})" if os_ else "")
    else:
        bits: List[str] = []
        if ug:
            bits.append(f"top: {ug}" + (f" ({us})" if us else ""))
        if lg:
            bits.append(f"bottom: {lg}" + (f" ({ls})" if ls else ""))
        g = "; ".join(bits) if bits else "garment per reference"
    return f"{g}; fit: {fit}; match fabric/print/trims/colors EXACTLY from refs"


def _evolink_compact_prompt(
    pose: PoseRuntime, req: dict, model_face_url: str, background_url: str,
) -> tuple[str, List[str]]:
    """Build a <2000 char prompt for Evolink (gemini-3.1-flash-image-preview).

    Evolink enforces a ~2000 character limit on ``prompt``.  We keep the same
    essential instructions as the full prompt but drop verbose language,
    collapse multi-line sections to single lines, and trim long user-supplied
    fields.  The final string is hard-capped at 1990 chars.
    """
    refs = _ordered_refs(pose, req, model_face_url, background_url)
    urls = [u for _, u in refs]

    # Short image-role legend (abbreviated labels).
    role_map = {
        "MODEL FACE (identity lock)":            "face",
        "GARMENT — FRONT view (use this)":       "garment-front (use)",
        "GARMENT — FRONT view (context only)":   "garment-front (ctx)",
        "GARMENT — BACK view (use this)":        "garment-back (use)",
        "GARMENT — BACK view (context only)":    "garment-back (ctx)",
        "MANNEQUIN POSE — body posture only":    "mannequin-pose",
        "BACKGROUND":                            "background",
    }
    legend = ", ".join(
        f"[{i}]={role_map.get(label, label)}" for i, (label, _) in enumerate(refs)
    )

    garment_src = "back" if (pose.is_back and req.get("back_garment_image")) else "front"
    pose_text = (pose.pose_prompt or "").strip() or "natural relaxed fashion pose"
    # Hard trim pose text — it's the most variable field.
    if len(pose_text) > 220:
        pose_text = pose_text[:217].rstrip() + "..."

    body = _body_description(req)
    if len(body) > 260:
        body = body[:257].rstrip() + "..."

    garment = _compact_garment(req)
    if len(garment) > 260:
        garment = garment[:257].rstrip() + "..."

    bg_type = (req.get("background_type") or "general").strip().lower() or "general"
    ornaments = (req.get("ornaments") or "none").strip() or "none"

    if pose.is_upper_body:
        footwear = (
            "FRAMING: UPPER-BODY only (head-to-waist), matching mannequin crop. "
            "NO feet, NO legs, NO footwear rendered."
        )
    else:
        footwear = (
            "FRAMING: FULL-BODY (head-to-feet), matching mannequin crop. "
            "FOOTWEAR MANDATORY — realistic shoes matching outfit, both feet "
            "fully in frame with contact shadow, NO bare feet, NO ankle crop."
        )

    # Budget-conscious compact prompt. Evolink cap is ~2000 chars; we
    # hard-cap at 1990. User-supplied inputs (body/garment/pose/bg) come
    # BEFORE the realism boilerplate so that any tail-truncation trims
    # realism prose (which degrades gracefully) rather than user data
    # (which would silently drop critical garment/pose info).
    prompt = (
        # --- Camera + intent ---
        "Candid editorial fashion photo, real in-camera frame, full-frame "
        "mirrorless + fast prime (24mm f/2.8 or 85mm f/1.8). Handheld, "
        "off-axis, NOT CG, NOT AI-smooth. Single subject. "
        f"Images: {legend}. "
        # --- Priority lock (posture + framing are highest weight) ---
        "HARD: (1) EXACT face identity from face ref; "
        "(2) garment fabric/weave/print/seams/trims/hardware/color EXACT "
        f"from {garment_src} ref; "
        "(3) COPY MANNEQUIN POSE 1:1 — same stance, weight shift, leg "
        "position (which leg bent/straight, foot placement), hand+arm "
        "position (shoulder/elbow/wrist angles, finger position), head "
        "tilt, body orientation. Do NOT substitute a generic model pose. "
        "Ignore mannequin face/skin/clothes/bg. "
        "(4) MATCH MANNEQUIN CROP: full-body mannequin→full-body output "
        "with shoes ON; upper-body mannequin→upper-body output, no feet. "
        "(5) integrate with matching perspective, light direction, color "
        "spill, contact shadows. "
        # --- USER INPUTS (placed early so tail-truncation can't touch them) ---
        f"Body: {body} Garment: {garment}. Pose: {pose_text} {footwear} "
        f"Background: {bg_type}. Ornaments: {ornaments}. "
        # --- Condensed realism/light (tail-safe to truncate) ---
        "Skin: pores, peach fuzz, T-zone shine, matte-satin, NEVER plastic. "
        "Hair: flyaways. Fabric: visible weave, joint wrinkles. Light: mixed "
        "sources, directional key, environment color spill, tight AO, "
        "contact shadow. Fine grain, creamy bokeh, sharp eye focus, "
        "natural grading. One photorealistic human, correct hands/anatomy, "
        "no text/watermark/logo. "
        # --- Final highest-weight reminder ---
        "FINAL: mannequin = POSE template AND CROP template — match both "
        f"exactly. Variation: {pose.index}."
    )

    # Hard cap — guarantee we never hit Evolink's 2000-char limit.
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

    parts: List[Any] = []

    if mf:
        parts.append(genai_types.Part.from_text(text="[REF] MODEL FACE (identity lock):"))
        face_bytes, face_mime = _load_image_bytes_sync(mf)
        parts.append(genai_types.Part.from_bytes(data=face_bytes, mime_type=face_mime))

    if pose.is_back and back:
        d, m = _load_image_bytes_sync(back)
        parts.append(genai_types.Part.from_text(text="[REF] GARMENT — BACK (use this):"))
        parts.append(genai_types.Part.from_bytes(data=d, mime_type=m))
        if front:
            d2, m2 = _load_image_bytes_sync(front)
            parts.append(genai_types.Part.from_text(text="[REF] GARMENT — FRONT (context):"))
            parts.append(genai_types.Part.from_bytes(data=d2, mime_type=m2))
    else:
        if front:
            d, m = _load_image_bytes_sync(front)
            parts.append(genai_types.Part.from_text(text="[REF] GARMENT — FRONT (use this):"))
            parts.append(genai_types.Part.from_bytes(data=d, mime_type=m))
        if back:
            d2, m2 = _load_image_bytes_sync(back)
            parts.append(genai_types.Part.from_text(text="[REF] GARMENT — BACK (context):"))
            parts.append(genai_types.Part.from_bytes(data=d2, mime_type=m2))

    if pose.mannequin_url:
        d3, m3 = _load_image_bytes_sync(pose.mannequin_url)
        parts.append(genai_types.Part.from_text(text="[REF] MANNEQUIN POSE — body posture only:"))
        parts.append(genai_types.Part.from_bytes(data=d3, mime_type=m3))

    if bg:
        d4, m4 = _load_image_bytes_sync(bg)
        parts.append(genai_types.Part.from_text(text="[REF] BACKGROUND:"))
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
