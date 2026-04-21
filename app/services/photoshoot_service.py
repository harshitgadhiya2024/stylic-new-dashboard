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
    """Return ``[{"image_url": ..., "pose_prompt": ...}, ...]`` per pose_id."""
    logger.info("[poses] Fetching %d pose doc(s) from DB", len(pose_ids))
    col = poses_col if poses_col is not None else get_poses_collection()
    results: List[dict] = []
    for pid in pose_ids:
        doc = await col.find_one({"pose_id": pid})
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
    """Pose-aware footwear instruction used by every nano-banana prompt.

    Full-body poses MUST include footwear (so the feet are never bare / cut
    off mid-ankle); upper-body / half-body framings must NOT render footwear
    because the feet aren't in frame.
    """
    if pose.is_upper_body:
        return (
            "Framing: upper-body only (head to waist / torso). "
            "Do NOT render footwear — feet must be out of frame. "
            "Do not add shoes, sandals, heels or any leg/feet elements."
        )
    return (
        "Framing: full-body (head to feet). Footwear is REQUIRED — render "
        "realistic, well-fitting shoes that match the outfit style (neutral "
        "studio-appropriate footwear if none is specified). Feet must be "
        "fully visible with natural contact shadows; no bare feet and no "
        "cropped ankles."
    )


def _core_prompt(pose: PoseRuntime, req: dict) -> str:
    body = _body_description(req)
    garment = _garment_description(req)
    garment_ref_note = (
        "Use the BACK garment reference image for the garment (this is a back pose)."
        if pose.is_back and req.get("back_garment_image")
        else "Use the FRONT garment reference image for the garment."
    )
    pose_text = (pose.pose_prompt or "").strip() or "Natural relaxed full-body fashion model pose."
    bg_type = (req.get("background_type") or "").strip().lower() or "general"
    ornaments = (req.get("ornaments") or "none").strip() or "none"
    footwear = _footwear_rule(pose)

    return (
        "You are generating a hyper-realistic studio fashion photograph shot on a full-frame DSLR\n"
        "with an 85mm prime lens at f/2.0. Output a single photorealistic image.\n\n"
        "PRIORITY:\n"
        "1) Preserve exact face identity from the face reference.\n"
        "2) Preserve garment fidelity exactly (fabric, prints, seams, trims, colors, hardware).\n"
        "3) Copy mannequin posture exactly (ignore mannequin face/skin/clothes/background).\n"
        "4) Blend naturally into the background with correct perspective, lighting, and contact shadows.\n\n"
        f"Body type: {body}\n"
        f"Garment reference rule: {garment_ref_note}\n"
        f"Garment details: {garment}\n"
        f"Pose: {pose_text}\n"
        f"{footwear}\n"
        f"Background type: {bg_type}. Ornaments: {ornaments}.\n"
        "Output: single subject, photorealistic only, no text/watermark/logo, no distortions.\n"
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
        footwear = "Framing: upper-body (head-to-waist); NO footwear, feet out of frame."
    else:
        footwear = (
            "Framing: full-body (head-to-feet); footwear REQUIRED — realistic "
            "shoes matching outfit, natural contact shadow, no bare feet."
        )

    prompt = (
        "Hyper-real studio fashion photo, 85mm DSLR f/2.0, single photorealistic subject. "
        f"Images: {legend}. "
        "Rules: (1) keep exact face identity from face ref; "
        "(2) reproduce garment fabric/print/trims/colors EXACTLY from the "
        f"{garment_src} garment ref; "
        "(3) copy mannequin posture only (ignore mannequin face/skin/clothes/bg); "
        "(4) blend into background with correct perspective, lighting, contact shadows. "
        f"Body: {body} "
        f"Garment: {garment}. "
        f"Pose: {pose_text} "
        f"{footwear} "
        f"Background: {bg_type}. Ornaments: {ornaments}. "
        "No text/watermark/logo, no distortions, single subject. "
        f"Variation: {pose.index}."
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
    img = Image.open(io.BytesIO(original_bytes)).convert("RGB")
    w, h = img.size
    scale = max_dimension / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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

        # 2) Resize 4K → 2K / 1K
        loop = asyncio.get_running_loop()
        bytes_2k, bytes_1k = await asyncio.gather(
            loop.run_in_executor(None, _resize_image, bytes_4k, 2048),
            loop.run_in_executor(None, _resize_image, bytes_4k, 1024),
        )

        # 3) Upload 4K/2K/1K to R2
        url_4k, url_2k, url_1k = await asyncio.gather(
            upload_bytes_to_r2(bytes_4k, f"{prefix}_4k.png", "image/png"),
            upload_bytes_to_r2(bytes_2k, f"{prefix}_2k.png", "image/png"),
            upload_bytes_to_r2(bytes_1k, f"{prefix}_1k.png", "image/png"),
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
