"""
Photoshoot background job service.

Responsibilities:
  1. Resolve pose prompts (default → fetch from poses_data, custom → Gemini vision,
     prompt → use as-is).
  2. Fetch background image URL from backgrounds collection.
  3. Fetch model face image URL from model_faces collection.
  4. For each pose — concurrently via asyncio.gather:
       a. Build compact generation prompt (<=3000 chars).
       b. Submit to kie.ai SEEDDREAM_MODEL (seedream/4.5-edit): full photoshoot
          generation from garment(s) + face + background refs.
       c. Poll until complete; get SeedDream result URL.
       d. Submit SeedDream result to nano-banana-pro realism pass (skin, eyes,
          hands, hair, shoes, background blending). Poll until complete.
       e. Download realism 4K bytes; resize to 2K and 1K; upload 4K / 2K / 1K to S3.
       f. Send realism 4K bytes to Modal GPU pipeline for enhancement.
       g. Upload enhanced 8K, 4K, 2K, 1K to S3 and save to upscaling_data.
  5. Build output_images mapping — each entry stores the upscaled 2K URL as `image`.
  6. Deduct credits and record history only for poses that fully succeeded (full pipeline + S3 output).
  7. Update photoshoot document: output_images, status, is_completed, is_credit_deducted.
     On any unhandled error → status="failed", error field set.
"""

import asyncio
import io
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import List

import httpx
from PIL import Image

logger = logging.getLogger("photoshoot")

from app.config import settings
from app.database import (
    get_backgrounds_collection,
    get_model_faces_collection,
    get_photoshoots_collection,
    get_poses_collection,
    get_users_collection,
    get_credit_history_collection,
)
from app.services.s3_service import upload_bytes_to_s3
from app.services.modal_enhance_service import enhance_and_upload

_CREATE_URL = "https://api.kie.ai/api/v1/jobs/createTask"
_STATUS_URL  = "https://api.kie.ai/api/v1/jobs/recordInfo"


# ---------------------------------------------------------------------------
# Pose prompt resolution
# ---------------------------------------------------------------------------

async def _fetch_pose_data(pose_ids: List[str], poses_col=None) -> List[dict]:
    """Return ``[{"image_url": ..., "pose_prompt": ...}, ...]`` for each pose_id."""
    logger.info("[poses] Fetching %d pose doc(s) from DB", len(pose_ids))
    col = poses_col if poses_col is not None else get_poses_collection()
    results: List[dict] = []
    for pid in pose_ids:
        doc = await col.find_one({"pose_id": pid})
        if doc:
            logger.info("[poses] Found pose_id=%s  image_url=%s", pid, bool(doc.get("image_url")))
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
    """
    Return ``[{"image_url": str, "pose_prompt": str}, ...]`` for each pose.

    Two paths:

    1. **pose_data** list already present (regeneration / background / fabric / texture / color
       change): each entry is ``{"image_url": mannequin, "pose_prompt": text}`` — used directly.
    2. **poses_ids** (normal create): loads ``image_url`` and ``pose_prompt`` from Mongo per id.
    """
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
        has_img = bool(pd.get("image_url"))
        logger.info("[poses] Pose #%d  mannequin_image=%s  prompt_len=%d",
                     i, has_img, len(pd.get("pose_prompt", "")))

    return result


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

import re as _re

def _is_back_pose(pose_prompt: str) -> bool:
    back_keywords = ["back", "rear", "behind", "back-facing", "facing away", "turned away"]
    lower = pose_prompt.lower()
    return any(kw in lower for kw in back_keywords)


def _sanitize_pose_prompt(pose: str) -> str:
    """Strip clothing, background, gender, and garment-related words from a pose prompt.

    The pose prompt should describe ONLY body position and limb placement.
    Removing these words prevents SeedDream from re-interpreting the garment
    or background based on pose description text.
    """
    strip_patterns = [
        r'\b(wearing|dressed in|outfit|garment|clothing|cloth|clothes|fabric|'
        r'top|bottom|skirt|pants|trousers|shirt|dress|blouse|jacket|coat|suit|'
        r'saree|sari|kurta|lehenga|churidar|dupatta|salwar|kameez|gown|frock|'
        r'shorts|jeans|denim|sweater|hoodie|cardigan|vest|crop|bralette|'
        r'sleeve|collar|neckline|hem|waist|belt)\b',
        r'\b(street|garden|park|beach|office|store|shop)\b',
        r'\b(male|female|man|woman|boy|girl|he|she|his|her|they|them)\b',
    ]
    result = pose
    for pattern in strip_patterns:
        result = _re.sub(pattern, '', result, flags=_re.IGNORECASE)
    result = _re.sub(r'\s{2,}', ' ', result).strip()
    return result


def _build_paired_garment_instruction(req: dict) -> str:
    """Build the paired-garment instruction for whichever garment piece is missing.

    Rules:
    - If ONLY upper garment is provided → describe a complementary lower garment
      that forms a natural outfit pair with it.
    - If ONLY lower garment is provided → describe a complementary upper garment
      that forms a natural outfit pair with it.
    - If both are provided → no pairing needed (both come from the reference image).
    - If one-piece is provided → no pairing needed.
    - The pairing must be derived from the garment reference image style/color,
      not invented out of thin air.
    """
    ug_type = req.get("upper_garment_type", "").strip()
    lg_type = req.get("lower_garment_type", "").strip()
    op_type = req.get("one_piece_garment_type", "").strip()

    has_upper    = bool(ug_type)
    has_lower    = bool(lg_type)
    has_onepiece = bool(op_type)

    if has_onepiece or (has_upper and has_lower):
        return ""

    if has_upper and not has_lower:
        return "\n[PAIRED LOWER GARMENT] Add a complementary lower garment matching the upper garment's style and color."

    if has_lower and not has_upper:
        return "\n[PAIRED UPPER GARMENT] Add a complementary upper garment matching the lower garment's style and color."

    return ""


def _build_seeddream_prompt(
    pose: str,
    has_back_image: bool,
    req: dict,
) -> str:
    """Build the SeedDream generation prompt \u2014 single-stage photoshoot pipeline."""
    face_img_num = 3 if has_back_image else 2
    bg_img_num   = 4 if has_back_image else 3

    if has_back_image:
        image_ref = (
            "You are provided with FOUR reference images:\n"
            "  IMG1 \u2014 GARMENT FRONT: exact outfit front view.\n"
            "  IMG2 \u2014 GARMENT BACK: exact outfit back view.\n"
            f"  IMG{face_img_num} \u2014 MODEL FACE: the exact face to use.\n"
            f"  IMG{bg_img_num} \u2014 BACKGROUND: the exact background scene."
        )
        garment_imgs = "IMG1 (front) and IMG2 (back)"
    else:
        image_ref = (
            "You are provided with THREE reference images:\n"
            "  IMG1 \u2014 GARMENT: exact outfit to be worn.\n"
            f"  IMG{face_img_num} \u2014 MODEL FACE: the exact face to use.\n"
            f"  IMG{bg_img_num} \u2014 BACKGROUND: the exact background scene."
        )
        garment_imgs = "IMG1"

    fitting = req.get("fitting", "regular fit")
    gender  = req.get("gender", "").strip().lower()

    # \u2500\u2500 Garment type lines \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    ug_type = req.get("upper_garment_type", "").strip()
    ug_spec = req.get("upper_garment_specification", "").strip()
    lg_type = req.get("lower_garment_type", "").strip()
    lg_spec = req.get("lower_garment_specification", "").strip()
    op_type = req.get("one_piece_garment_type", "").strip()
    op_spec = req.get("one_piece_garment_specification", "").strip()

    garment_type_lines = []
    if ug_type:
        garment_type_lines.append(f"  Upper garment: {ug_type}" + (f" ({ug_spec})" if ug_spec else ""))
    if lg_type:
        garment_type_lines.append(f"  Lower garment: {lg_type}" + (f" ({lg_spec})" if lg_spec else ""))
    if op_type:
        garment_type_lines.append(f"  One-piece: {op_type}" + (f" ({op_spec})" if op_spec else ""))
    garment_type_section = ("\nGarment type(s):\n" + "\n".join(garment_type_lines)) if garment_type_lines else ""

    # \u2500\u2500 Paired garment \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    paired_garment_block = _build_paired_garment_instruction(req)
    clean_pose           = _sanitize_pose_prompt(pose)

    paired_note = ""
    if paired_garment_block:
        if "LOWER" in paired_garment_block:
            paired_note = "\n- IMPORTANT: Add a matching bottom garment \u2014 no bare legs."
        else:
            paired_note = "\n- IMPORTANT: Add a matching top garment \u2014 no bare torso."

    _ornaments_lower = req.get("ornaments", "").lower()
    _bag_requested   = any(kw in _ornaments_lower for kw in ("bag", "purse", "handbag"))
    bag_note = (
        "\n- Add a matching bag/purse (clutch for ethnic/formal, handbag for casual) "
        "held naturally, matching the outfit color palette."
    ) if (gender == "female" and _bag_requested) else ""

    # \u2500\u2500 Weight & height \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    weight = req.get("weight", "regular").strip().lower()
    height = req.get("height", "regular").strip().lower()

    weight_instruction = {
        "fat": (
            "PLUS-SIZE / HEAVY BUILD (non-negotiable): visibly larger body mass \u2014 "
            "broader torso, thicker upper arms and thighs, fuller waist/abdomen, "
            "wider hips. The silhouette must read as genuinely heavy, not "
            "\"slightly curvy\" or standard fashion-model thin."
        ),
        "slim": (
            "SLIM / LEAN BUILD: narrow shoulders, defined waist, slender limbs. "
            "Visibly leaner than an average model."
        ),
    }.get(weight, "Regular / average build \u2014 natural proportions, neither slim nor heavy.")

    height_instruction = {
        "short": "Noticeably shorter than average \u2014 shorter limbs and torso relative to the frame.",
        "tall":  "Noticeably tall \u2014 longer limbs and torso, above-average stature.",
    }.get(height, "Average / regular height \u2014 standard proportions.")

    # \u2500\u2500 Background type \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    bg_type = req.get("background_type", "").strip().lower()
    bg_type_label = bg_type if bg_type in ("indoor", "outdoor", "studio") else "general"

    return f"""You are a HIGH-QUALITY IMAGE GENERATION model. Generate a single photorealistic fashion photoshoot image.
The model must look PHYSICALLY PRESENT in the scene \u2014 as if a photographer clicked a candid shot on location.

{image_ref}

=====================================================================
  PRIORITY ORDER (strictly follow \u2014 higher priority wins any conflict)
  P1 = GARMENT  |  P2 = FACE  |  P3 = BACKGROUND BLEND  |  P4 = BODY (weight + height)  |  P5 = POSE/STYLE
=====================================================================

[P1 \u2014 GARMENT \u2014 ABSOLUTE LOCK \u2014 DO NOT CHANGE ANYTHING]{garment_type_section}
Reproduce the EXACT garment from {garment_imgs}:
- Same fabric, pattern, color, print, embroidery, buttons, pockets, stitching, hem length, neckline, sleeves, cuffs.
- Same micro-level details: weave texture, closures, trims, raw edges, pleats, gathers, ruffles, and how the garment drapes.
- COLOR LOCK: Every fabric, print, border, and trim \u2014 same hues and values. No re-tinting, no warming/cooling drift, no extra saturation.
- WEARING STYLE LOCK: Tucked/untucked, rolled/unrolled sleeves, buttoned/unbuttoned, draped/wrapped, loose/fitted \u2014 match exactly.
- Fitting: {fitting} \u2014 only as the reference garment allows; never override design, color, or wearing style.
- Do NOT simplify, genericize, or \u201cclean up\u201d the garment. Imperfections (wrinkles, asymmetry) are intentional \u2014 preserve them.{paired_note}

[P2 \u2014 FACE \u2014 EXACT IDENTITY LOCK \u2014 DO NOT CHANGE ANYTHING]
The face MUST be an exact copy of the person in IMG{face_img_num}. This is the SAME person \u2014 not \u201cinspired by\u201d or \u201csimilar to\u201d.
- Lock EVERYTHING: face shape, forehead, eye shape/size/color/spacing, eyebrow shape/thickness/arch, nose bridge/tip/width, lip shape/fullness/color, jaw angle, chin shape, cheekbone prominence, ear shape/size, hairline, hair color/texture/length/parting/style, facial hair (if any), skin tone/undertone, all moles/marks/freckles.
- Do NOT beautify, slim, smooth, enlarge eyes, narrow nose, plump lips, or alter ANY facial proportion.
- Expression stays identical to IMG{face_img_num}. Head angle may follow the pose; facial expression and features do NOT change.
- Do NOT change ethnicity, age appearance, or skin color.

[P3 \u2014 BACKGROUND BLENDING \u2014 REALISTIC SCENE INTEGRATION]
The model must look like they are PHYSICALLY PRESENT in the background from IMG{bg_img_num} \u2014 not composited, not edited in, not pasted.
Background type: {bg_type_label}.
- LIGHTING MATCH: Analyze the background \u2014 morning (soft golden light), afternoon (harsh overhead sun), evening (warm orange/pink), indoor (diffused/artificial), outdoor (natural sky), studio (controlled directional). Apply the SAME lighting on the model:
  * Direction: light hits the model from the same direction as the scene key light.
  * Softness/hardness: match the shadow edge quality.
  * Color temperature and intensity: match the background.
- SHADOW INTEGRATION: Natural shadows on ground/surfaces \u2014 same direction, softness, length as scene lighting. Contact shadows at feet. Ambient occlusion at all touch points.
- SKIN & FACE LIGHTING: Scene-consistent highlights and shadows on skin and face. Same sun-side brightness, shadow-side falloff, specular highlights.
- REFLECTION & BOUNCE: If background has reflective surfaces, show subtle model reflections. Color bounce from nearby surfaces.
- ATMOSPHERIC MATCH: If background shows haze, fog, dust \u2014 apply same to model at correct depth.
- Do NOT change the background itself (geometry, colors, objects, sky).

Canon DSLR / full-frame. 4K, 9:16. Candid composition. Shallow depth of field \u2014 subject sharp, background with natural bokeh. Subtle film grain (Kodak Portra style). Natural lens characteristics.

[P4 \u2014 MODEL BODY \u2014 WEIGHT & HEIGHT]
- Gender: {req['gender']}, Ethnicity: {req['ethnicity']}, Age: {req['age']} ({req['age_group']}), Skin tone: {req['skin_tone']}.
- WEIGHT ({weight}): {weight_instruction} The model\u2019s body must visibly reflect this build \u2014 do not default to a standard fashion-model physique.
- HEIGHT ({height}): {height_instruction} Adjust proportions and how the model fills the frame.
- Face identity is ONLY from IMG{face_img_num} (P2); body build follows weight/height here.

[P5 \u2014 POSE] {clean_pose if clean_pose else "Natural, relaxed full-body fashion model pose."}

[FOOTWEAR] MANDATORY: footwear is required in every image; it must match garment style/color theme, stay clearly visible on ground, and bare feet are not allowed.{bag_note}

[STYLE] Ornaments: {req.get('ornaments', 'none')}.

[QUALITY]
- Seamless integration \u2014 no halos, cutouts, or composited look.
- High resolution, fashion e-commerce grade.
- Output: one image, 9:16 aspect ratio."""


# ---------------------------------------------------------------------------
# kie.ai \u2014 SeedDream generation + poll
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Compact prompt builder (<=3 000 chars for kie.ai SeedDream limit)
# ---------------------------------------------------------------------------

_COMPACT_PROMPT_LIMIT = 3000

_COMPACT_SANITIZE_PATTERNS = [
    _re.compile(
        r'\b(wearing|dressed in|outfit|garment|clothing|cloth|clothes|fabric|'
        r'top|bottom|skirt|pants|trousers|shirt|dress|blouse|jacket|coat|suit|'
        r'saree|sari|kurta|lehenga|churidar|dupatta|salwar|kameez|gown|frock|'
        r'shorts|jeans|denim|sweater|hoodie|cardigan|vest|crop|bralette|'
        r'sleeve|collar|neckline|hem|waist|belt)\b', _re.IGNORECASE),
    _re.compile(r'\b(street|garden|park|beach|office|store|shop)\b', _re.IGNORECASE),
    _re.compile(r'\b(male|female|man|woman|boy|girl|he|she|his|her|they|them)\b', _re.IGNORECASE),
]


def _sanitize_pose_compact(pose: str) -> str:
    result = pose
    for pat in _COMPACT_SANITIZE_PATTERNS:
        result = pat.sub('', result)
    return _re.sub(r'\s{2,}', ' ', result).strip()


def _build_compact_prompt(
    pose: str,
    has_back: bool,
    req: dict,
    *,
    has_mannequin_image: bool = False,
) -> str:
    """Build a <=3000-char SeedDream prompt.

    When ``has_mannequin_image`` is True the last reference image is a grey
    mannequin whose ONLY purpose is body posture — the prompt tells SeedDream
    to copy limb / torso / head position and ignore everything else about it.
    """
    fi = 3 if has_back else 2
    bi = 4 if has_back else 3
    mi = bi + 1  # mannequin image number (only used when has_mannequin_image)

    if has_back:
        img_ref = f"IMG1=garment front, IMG2=garment back, IMG{fi}=face, IMG{bi}=background"
        g_imgs = "IMG1+IMG2"
    else:
        img_ref = f"IMG1=garment, IMG{fi}=face, IMG{bi}=background"
        g_imgs = "IMG1"

    if has_mannequin_image:
        img_ref += f", IMG{mi}=POSE MANNEQUIN (body posture reference ONLY)"

    ug  = req.get("upper_garment_type", "").strip()
    us  = req.get("upper_garment_specification", "").strip()
    lg  = req.get("lower_garment_type", "").strip()
    ls  = req.get("lower_garment_specification", "").strip()
    op  = req.get("one_piece_garment_type", "").strip()
    os_ = req.get("one_piece_garment_specification", "").strip()
    op_lower = op.lower()

    parts = []
    if ug:
        parts.append(f"Upper: {ug}" + (f" ({us})" if us else ""))
    if lg:
        parts.append(f"Lower: {lg}" + (f" ({ls})" if ls else ""))
    if op:
        parts.append(f"One-piece: {op}" + (f" ({os_})" if os_ else ""))
    gt_line = f"\nTypes: {' | '.join(parts)}" if parts else ""

    fitting = req.get("fitting", "regular fit")

    paired = ""
    if not op and not (ug and lg):
        if ug and not lg:
            paired = " Add matching lower garment\u2014no bare legs."
        elif lg and not ug:
            paired = " Add matching upper garment\u2014no bare torso."

    # ── garment-specific wearing rules ────────────────────────────────────
    garment_rules = ""
    hands_must_show = False
    if op:
        if "saree" in op_lower or "sari" in op_lower:
            garment_rules = (
                "\n[GARMENT]\n"
                "SAREE\u2014Gujarati drape: pallu over LEFT shoulder to back; neat waist pleats; "
                "fitted blouse + petticoat (chaniya); clear front pleats. "
                "Hands and all fingers visible\u2014pallu must not cover them.\n"
            )
            hands_must_show = True
        elif "dress" in op_lower:
            spec_lower = os_.lower()
            is_3piece = any(kw in spec_lower for kw in (
                "3 piece", "3-piece", "three piece", "dupatta", "duppata",
                "chunni", "stole",
            ))
            if is_3piece:
                garment_rules = (
                    "\n[GARMENT]\n"
                    "3-PIECE dress: top + bottom + dupatta (all mandatory, do not skip dupatta). "
                    "Dupatta Gujarati\u2014one shoulder or across chest. "
                    "Hands and all fingers visible\u2014dupatta must not cover them.\n"
                )
                hands_must_show = True
            else:
                garment_rules = (
                    "\n[GARMENT WEARING RULE]\n"
                    "This is a 2-PIECE DRESS. Wear only the top and bottom\u2014"
                    "do NOT add any dupatta, stole, or scarf.\n"
                )
        elif "lehenga" in op_lower:
            garment_rules = (
                "\n[GARMENT WEARING RULE]\n"
                "This is a LEHENGA. Wear it in Gujarati style with the matching choli (blouse) "
                "and dupatta draped over one shoulder. "
                "BOTH HANDS must be clearly visible and unobstructed.\n"
            )
            hands_must_show = True
        else:
            garment_rules = (
                "\n[GARMENT WEARING RULE]\n"
                f"This is a ONE-PIECE garment ({op}). Wear it exactly as a single piece\u2014"
                "do NOT add extra layers, scarves, or separate top/bottom.\n"
            )

    weight = req.get("weight", "regular").strip().lower()
    height = req.get("height", "regular").strip().lower()
    w_desc = {
        "fat":  "plus-size/heavy build: broad torso, thick limbs, full waist, wide hips\u2014genuinely heavy silhouette",
        "slim": "slim/lean: narrow shoulders, defined waist, slender limbs",
    }.get(weight, "average build, natural proportions")
    h_desc = {
        "short": "shorter than average",
        "tall":  "taller than average",
    }.get(height, "average height")

    bg_type  = req.get("background_type", "").strip().lower()
    bg_label = bg_type if bg_type in ("indoor", "outdoor", "studio") else "general"

    clean_pose = _sanitize_pose_compact(pose) if pose else "Natural relaxed full-body fashion pose"

    gender    = req.get("gender", "").strip().lower()
    orn       = req.get("ornaments", "").strip()
    orn_lower = orn.lower()
    bag = ""
    if gender == "female" and any(kw in orn_lower for kw in ("bag", "purse", "handbag")):
        bag = " Add matching bag/purse held naturally."

    hands_note = ""
    if hands_must_show:
        hands_note = " Do not let dupatta/pallu/fabric hide hands or fingers."

    prompt = (
        f"Hyperrealistic editorial fashion photograph\u2014real person, NOT illustration, NOT AI-looking, NOT plastic skin.\n"
        f"\n"
        f"Refs: {img_ref}\n"
        f"\n"
        f"[GARMENT\u2014EXACT MATCH {g_imgs}]{gt_line}\n"
        f"Copy exact fabric, pattern, color, print, embroidery, buttons, pockets, stitching, hem, neckline, "
        f"sleeves, cuffs, weave, closures, trims, pleats, natural drape and wrinkles. "
        f"No color shift, no simplification. Fitting: {fitting}.{paired}\n"
        + garment_rules
        + f"\n"
        f"[FACE\u2014EXACT COPY IMG{fi}]\n"
        f"Lock all features: face shape, eyes, brows, nose, lips, jaw, chin, cheekbones, ears, hairline, "
        f"hair style/color/length, skin tone/undertone, moles/marks. No beautification. Expression identical to IMG{fi}.\n"
        f"\n"
        f"[SKIN REALISM\u2014NO BEAUTY FILTER]\n"
        f"Visible pores, peach-fuzz, subtle veins/SSS, natural tone variation, knuckle and crease lines, "
        f"realistic elbow texture. No beauty smoothing\u2014raw hyperreal skin.\n"
        f"\n"
        f"[BG + IN-SCENE SUBJECT\u2014IMG{bi} ({bg_label})]\n"
        f"Single real capture\u2014not a composite paste. Light the subject ONLY from the same light field as IMG{bi} "
        f"(key direction, WB, shadow softness); avoid flat frontal studio fill. "
        f"Ambient color bounce from walls, floor, foliage, visible lamps onto skin, garment, and shoes. "
        f"Ambient occlusion plus tight contact shadow at soles and hem where fabric meets floor; cast shadow matches scene key. "
        f"Feet on the floor plane following tile/arch perspective. Rim on hair/shoulders from scene sources; "
        f"subtle light wrap on silhouette edges toward bright areas\u2014no razor cutout. "
        f"Match subject exposure, contrast, and saturation to the environment (not brighter/punchier than the room). "
        f"No floating, no halo. Do not alter background.\n"
        f"\n"
        f"[BODY] {req.get('gender','')}, {req.get('ethnicity','')}, age {req.get('age','')} ({req.get('age_group','')}), "
        f"skin: {req.get('skin_tone','')}.\n"
        f"Weight: {w_desc}. Height: {h_desc}. Body must visibly reflect this build.\n"
        f"\n"
        + (
            f"[POSE \u2014 EXACT COPY FROM IMG{mi} MANNEQUIN]\n"
            f"IMG{mi} is a grey featureless mannequin. It is ONLY a body-posture reference.\n"
            f"You MUST replicate the EXACT pose from IMG{mi} with ZERO deviation:\n"
            f"- HANDS: Copy exact hand position\u2014if hand is in pocket, keep it in pocket; if hand is on hip, "
            f"keep it on hip; if hands are clasped, keep them clasped. Reproduce exact finger curl, wrist angle, "
            f"and palm orientation. Do NOT invent a different hand placement.\n"
            f"- BODY/TORSO: Copy exact torso lean angle, shoulder tilt, chest orientation, spine curvature, "
            f"hip rotation, and weight shift. Do NOT straighten or alter the torso.\n"
            f"- LEGS/FEET: Copy exact leg stance\u2014crossed, apart, bent, straight\u2014exactly as shown. "
            f"Replicate knee bend angle, foot placement, foot direction, and weight distribution between legs.\n"
            f"- HEAD: Copy exact head tilt, turn direction, chin angle, and gaze direction.\n"
            f"IGNORE everything else about IMG{mi}\u2014ignore its grey skin, featureless face, bald head, "
            f"white background, and clothing. Only replicate the body posture on the real human model.{hands_note}\n"
            if has_mannequin_image else
            f"[POSE] {clean_pose}\n"
        )
        + f"\n"
        f"MANDATORY footwear in every image: match garment style/color theme, keep visible on ground, and no bare feet.{bag} Ornaments: {orn or 'none'}.\n"
        f"85mm f/2.8, shallow DOF on subject, 4K 9:16, editorial color; natural hair detail; one coherent in-camera exposure."
    )

    if len(prompt) <= _COMPACT_PROMPT_LIMIT:
        return prompt

    compact = _build_compact_prompt_optimized(
        pose=clean_pose,
        img_ref=img_ref,
        g_imgs=g_imgs,
        fi=fi,
        bi=bi,
        mi=mi,
        gt_line=gt_line,
        fitting=fitting,
        paired=paired,
        garment_rules=garment_rules,
        req=req,
        w_desc=w_desc,
        h_desc=h_desc,
        bg_label=bg_label,
        has_mannequin_image=has_mannequin_image,
        hands_note=hands_note,
        bag=bag,
        orn=orn,
        op=op,
        os_=os_,
    )
    if len(compact) <= _COMPACT_PROMPT_LIMIT:
        return compact

    raise RuntimeError(
        f"Compact prompt exceeds {_COMPACT_PROMPT_LIMIT} chars after optimization "
        f"(len={len(compact)})."
    )


def _build_compact_prompt_optimized(
    *,
    pose: str,
    img_ref: str,
    g_imgs: str,
    fi: int,
    bi: int,
    mi: int,
    gt_line: str,
    fitting: str,
    paired: str,
    garment_rules: str,
    req: dict,
    w_desc: str,
    h_desc: str,
    bg_label: str,
    has_mannequin_image: bool,
    hands_note: str,
    bag: str,
    orn: str,
    op: str,
    os_: str,
) -> str:
    """Optimized <=3000-char variant with preserved instructions (especially one-piece)."""
    onepiece_hint = ""
    if op:
        onepiece_hint = (
            "\n[ONE-PIECE LOCK]\n"
            f"Type: {op}" + (f" ({os_})" if os_ else "") + ". "
            "Follow one-piece wearing rules exactly; do not invent extra layers."
        )

    pose_block = (
        f"[POSE FROM IMG{mi}]\n"
        f"Copy exact posture from IMG{mi}: hands/fingers/wrists, torso lean, shoulder tilt, hip rotation, "
        f"legs/knees/feet, head tilt/turn/chin/gaze. Use mannequin only for body pose; ignore mannequin face/clothes/bg.{hands_note}\n"
        if has_mannequin_image
        else f"[POSE] {pose}\n"
    )

    return (
        "Hyperrealistic editorial fashion photo (real DSLR look; no AI/plastic skin).\n"
        f"Refs: {img_ref}\n"
        f"[GARMENT EXACT {g_imgs}]{gt_line}\n"
        "Keep exact garment color/pattern/fabric/embroidery/details/stitching/fit/drape/wrinkles. "
        f"Fitting: {fitting}.{paired}\n"
        f"{garment_rules}"
        f"{onepiece_hint}\n"
        f"[FACE EXACT IMG{fi}] Same identity and expression; no beautification or feature edits.\n"
        "[SKIN] Pores, peach fuzz, veins, tone variation; no smoothing.\n"
        f"[BG+IN-SCENE IMG{bi} {bg_label}] Subject lit from bg only (dir/WB/shadows); bounce on skin/clothes/shoes; "
        f"AO+sole contact; cast matches key; feet on floor grid; rim+hair light wrap not cutout; match exposure to scene. "
        f"No halo/float; do not alter bg.\n"
        f"[BODY] {req.get('gender','')}, {req.get('ethnicity','')}, age {req.get('age','')} ({req.get('age_group','')}), "
        f"skin {req.get('skin_tone','')}; weight {w_desc}; height {h_desc}.\n"
        f"{pose_block}"
        f"[STYLE] MANDATORY footwear; visible on ground; no bare feet.{bag} Ornaments: {orn or 'none'}.\n"
        "85mm f/2.8, shallow DOF, 4K 9:16, editorial color, natural hair; single coherent in-camera exposure."
    )


# ---------------------------------------------------------------------------
# Realism pass \u2014 nano-banana-pro prompt + submit/poll
# ---------------------------------------------------------------------------

_REALISM_PROMPT = (
    "You are an expert photo retoucher. You receive a single fashion photoshoot image (IMG1).\n"
    "Your ONLY job is to make it look like a real, unretouched DSLR photograph \u2014 NOT an AI render.\n"
    "Do NOT change anything about the composition, pose, garment, background scene, or face identity.\n"
    "\n"
    "=== ABSOLUTE PRESERVATION LOCKS (do NOT alter) ===\n"
    "\n"
    "[FACE IDENTITY LOCK]\n"
    "- Preserve every facial feature exactly: face shape, eye shape/size/color/spacing, eyebrow shape/thickness/arch, "
    "nose bridge/tip/width, lip shape/fullness/color, jaw angle, chin shape, cheekbone prominence, ear shape, "
    "skin tone/undertone, all moles/marks/freckles, facial hair if present.\n"
    "- Preserve exact expression, head angle, gaze direction.\n"
    "- Do NOT beautify, reshape, slim, smooth, or alter any facial proportion.\n"
    "\n"
    "[BODY & POSE LOCK]\n"
    "- Preserve exact body shape, weight, proportions, height, limb positions, posture, hand placement, "
    "finger positions, foot placement.\n"
    "- Do NOT alter body silhouette, muscle definition, or body mass in any way.\n"
    "\n"
    "[GARMENT LOCK]\n"
    "- Preserve every detail of clothing: fabric type, pattern, color, print, embroidery, buttons, pockets, "
    "stitching, hem, neckline, sleeves, cuffs, weave texture, closures, trims, pleats, wrinkles, drape, fit, "
    "tucked/untucked state.\n"
    "- Do NOT change any garment color, pattern, or styling.\n"
    "\n"
    "[BACKGROUND & COMPOSITION LOCK]\n"
    "- Preserve the entire background scene: geometry, objects, colors, sky, furniture, walls, floor, props \u2014 everything.\n"
    "- Preserve the exact framing, crop, aspect ratio, and camera angle.\n"
    "\n"
    "=== REALISM ENHANCEMENTS (apply these ONLY) ===\n"
    "\n"
    "[1. HANDS \u2014 CRITICAL PRIORITY]\n"
    "Hands are the #1 AI giveaway. Fix them to look like real human hands:\n"
    "- FINGER ANATOMY: Each finger must have natural slight curvature \u2014 never perfectly straight or rigid. "
    "Fingers at rest have a gentle curl. Knuckle joints should show bony protrusion and skin bunching when bent. "
    "Inter-finger webbing visible between spread fingers.\n"
    "- SKIN TEXTURE ON HANDS: Deep knuckle wrinkles and creases, finger joint lines, visible tendons on back of hand, "
    "dorsal veins (blue-green subcutaneous), bony knuckle ridges. "
    "Palm side: visible palm lines (life line, heart line, head line), finger pad skin ridges.\n"
    "- NAILS: Natural nail shape with visible cuticles, lunula (half-moon at base), slight translucency showing pink nail bed. "
    "Not perfectly uniform \u2014 slight variation in shape, minor ridges, natural sheen not plastic gloss.\n"
    "- NATURAL GRIP: When hand is in pocket, gripping arm, or resting on surface \u2014 fingers wrap naturally with "
    "proper pressure dimpling on contacted surface. No hovering fingers. Skin compresses where it touches.\n"
    "- RELAXED HAND: Slightly curled fingers with natural spacing \u2014 not stiff mannequin hands. "
    "Thumb sits naturally opposed to fingers, not rigidly parallel.\n"
    "- WRIST: Visible wrist bones (ulnar styloid), skin crease lines, subtle vein detail on inner wrist.\n"
    "\n"
    "[2. EYES \u2014 MICRO-DETAIL]\n"
    "Eyes must look alive, not CG-rendered:\n"
    "- IRIS: Complex iris texture \u2014 visible radial fibres (collarette pattern), crypts (dark irregular patches), "
    "furrows, and pigment spots. Iris is NOT a uniform solid color; it has concentric rings and radial striations.\n"
    "- SCLERA: NOT perfectly white. Faint pink/yellowish tint, very subtle micro blood vessels near inner and outer corners. "
    "Slight natural discoloration near the limbus (iris border).\n"
    "- LIMBAL RING: Darker ring at the outer edge of the iris where it meets the sclera.\n"
    "- TEAR FILM / MOISTURE: Wet glossy reflection on the cornea \u2014 a single bright catchlight plus subtle secondary reflections. "
    "Slight moisture at inner corner (lacrimal caruncle). Very faint wet line along lower lid margin.\n"
    "- PUPIL: True black, perfectly round, with sharp edge. Subtle depth \u2014 it's a hole, not a painted circle.\n"
    "- EYELIDS: Fine skin texture on eyelids, visible eyelid crease, tiny blood vessels if visible. "
    "Lower lid has slight puffiness/texture, not a clean line.\n"
    "\n"
    "[3. FOOTWEAR \u2014 REAL WORN SHOES]\n"
    "Shoes must NOT look like 3D renders or brand-new pristine objects:\n"
    "- WEAR MARKS: Subtle scuff marks, minor sole edge dirt, slight creasing at the toe flex point. Lace/strap wear.\n"
    "- MATERIAL TEXTURE: Canvas/leather grain visible. Stitching detail \u2014 individual stitch holes and thread. "
    "Sole rubber texture, not smooth plastic.\n"
    "- GROUND INTERACTION: Shoe sole FLAT on the ground surface \u2014 not floating. Contact shadow and slight dirt/dust "
    "accumulation at the sole-ground junction. Laces have natural drape and slight twist, not rigid symmetrical bows.\n"
    "- LIGHT RESPONSE: Proper shadow under shoe tongue, highlight on toe cap, matte vs glossy areas. Not uniformly lit.\n"
    "\n"
    "[4. SKIN TEXTURE \u2014 HYPERREALISTIC (PRIORITIZE PORES + VEINS)]\n"
    "Push skin toward a real 100% zoom DSLR read: micro-structure must be obvious but natural\u2014never plastic or airbrushed.\n"
    "Apply across ALL visible skin (face, neck, d\u00e9collet\u00e9 if shown, arms, hands, legs, feet):\n"
    "- MICRO-PORES: Fine irregular pore field on nose, inner cheeks, forehead, chin\u2014visible at normal viewing distance. "
    "Vary pore size and depth; cluster slightly around nose sides and mid-cheek; subtle shadow inside each pore opening "
    "(tiny dark wells), not a flat dot pattern. T-zone slightly more visible sebaceous texture; do NOT erase pores on face.\n"
    "- PERIORBITAL + CHEEK MICRO-TEXTURE: Very fine micro-grain and tiny creases under eyes and at smile lines; "
    "keep them light\u2014not deep wrinkles\u2014unless already strong in the source.\n"
    "- VEINS (SUBCUTANEOUS, READABLE): Make thin blue-green venous lines slightly more visible where real skin shows them: "
    "dorsal hands and knuckles, inner wrists, forearms, temples, side of neck, optional faint periorbital/temple branches. "
    "They must follow natural anatomy (branching, tapering), sit under the skin with soft edges, and respond to scene light "
    "(not neon stickers). Slightly stronger read on arms/hands than on face\u2014face veins stay subtle.\n"
    "- PEACH FUZZ: Fine translucent vellus on cheeks, jawline, upper lip, temples, forearms, upper arms.\n"
    "- SKIN TONE VARIATION: Slightly redder on knuckles, elbows, knees, nose tip; darker in joint creases, under-eye.\n"
    "- TEXTURE: Knuckle folds, finger joint wrinkles, nail detail with cuticles, wrist crease lines, realistic elbow texture.\n"
    "- SUBSURFACE SCATTERING: Ear edges glow reddish when backlit, fingertip translucency, nostril light transmission.\n"
    "- IMPERFECTIONS: Occasional tiny bumps, slight blemish, milia. No beauty filter, no airbrushing, no porcelain look.\n"
    "\n"
    "[5. HAIR REALISM + EDGE FIX]\n"
    "- Individual strand detail \u2014 not a smooth blob. Each strand has its own path with natural irregularity.\n"
    "- Flyaway hairs and baby hairs at hairline, temples, nape, part line.\n"
    "- Specular highlights on individual strands from scene light, not uniform sheen.\n"
    "- HAIR EDGE \u2014 CRITICAL: NO compositing halo or bright fringe around hair edges against the background. "
    "Hair edges must have natural semi-transparent wisps that blend into the background \u2014 not a hard cutout. "
    "Individual stray hairs extend beyond the main hair silhouette and are semi-transparent where they overlap the background.\n"
    "- Eyebrows: individual hairs with natural direction. Eyelashes: natural variation in length/curl.\n"
    "\n"
    "[6. SPATIAL GROUNDING \u2014 ANTI-PASTED FIX]\n"
    "The model must look like a REAL PERSON physically present at that location \u2014 NOT pasted or composited:\n"
    "- ANALYZE the scene: ground plane, perspective lines, vanishing point, object scale. "
    "Model's feet/seated contact must sit ON the ground plane at correct perspective depth.\n"
    "- If spatially misplaced (wrong scale, feet not touching ground, perspective mismatch, cutout look), "
    "ADJUST spatial position so they belong naturally. KEEP exact same pose, limb placement, gesture.\n"
    "- If already grounded correctly \u2014 do NOT change anything.\n"
    "- PERSPECTIVE: Model foreshortening must match camera angle of the scene.\n"
    "\n"
    "[7. BACKGROUND BLENDING & LIGHTING]\n"
    "Make the person genuinely PRESENT in the scene:\n"
    "- Match light direction, color temp, shadow hardness between model and scene.\n"
    "- Ground contact shadow with soft penumbra at feet. Shoes cast realistic shadow onto floor.\n"
    "- Rim light on hair/shoulder edges from scene backlight. Color bounce from nearby surfaces.\n"
    "- Atmospheric depth matching. Reflections on glossy surfaces.\n"
    "- Remove ANY visible halo, edge glow, bright fringe, or compositing artifacts \u2014 especially around hair and shoulder edges.\n"
    "- EDGE BLENDING: Seamless boundary \u2014 no sharp cutout edges, no aliasing, no color fringe.\n"
    "\n"
    "[8. FABRIC & MATERIAL REALISM]\n"
    "- Thread weave micro-texture visible. Material-appropriate sheen (cotton matte, silk specular, denim rough).\n"
    "- Realistic wrinkle shadow/highlight from scene lighting. Natural skin-to-fabric transition.\n"
    "\n"
    "[9. CAMERA / LENS]\n"
    "- Mild chromatic aberration at frame edges, natural vignette. Consistent bokeh.\n"
    "- Film grain / sensor noise (ISO 200-400 DSLR). Kodak Portra / Fuji Pro color. Natural micro-contrast.\n"
    "\n"
    "[10. FINAL QUALITY]\n"
    "- Indistinguishable from real DSLR photo at 100% zoom. No plastic/waxy surfaces, no AI uniformity.\n"
    "- Real photos have micro-imperfections \u2014 embrace them. Same resolution and aspect ratio as input."
)


def _build_realism_prompt() -> str:
    return _REALISM_PROMPT


async def _submit_realism_task(prompt: str, image_url: str, pose_label: str) -> str:
    """Submit a nano-banana-pro realism editing task."""
    logger.info("[%s] Submitting realism pass (%s, %d chars)...", pose_label, settings.REALISM_MODEL, len(prompt))
    payload = json.dumps({
        "model": settings.REALISM_MODEL,
        "input": {
            "prompt":       prompt,
            "image_input":  [image_url],
            "aspect_ratio": settings.REALISM_ASPECT,
            "resolution":   settings.REALISM_QUALITY,
            "output_format": "jpg"
        },
    })
    headers = {
        "Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}",
        "Content-Type":  "application/json",
    }
    for attempt in range(1, int(getattr(settings, "KIE_REQUEST_RETRIES", 3)) + 1):
        try:
            async with httpx.AsyncClient(timeout=settings.KIE_HTTP_TIMEOUT) as client:
                resp = await client.post(_CREATE_URL, headers=headers, content=payload)
                resp.raise_for_status()
            break
        except Exception as exc:
            logger.warning("[%s] Realism submit attempt %d failed: %s", pose_label, attempt, exc)
            if attempt == int(getattr(settings, "KIE_REQUEST_RETRIES", 3)):
                raise RuntimeError(f"Realism submit failed after {attempt} attempts: {exc}") from exc
            await asyncio.sleep(2)
    task_id = resp.json().get("data", {}).get("taskId")
    if not task_id:
        raise RuntimeError(f"No taskId returned (realism): {resp.text}")
    logger.info("[%s] Realism task submitted \u2014 task_id=%s", pose_label, task_id)
    return task_id


async def _poll_realism_task(task_id: str, pose_label: str) -> str:
    """Poll nano-banana-pro task until success/fail."""
    logger.info("[%s] Polling realism task_id=%s ...", pose_label, task_id)
    headers = {"Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}"}
    for attempt in range(1, settings.SEEDDREAM_MAX_RETRIES + 1):
        try:
            resp = None
            for req_try in range(1, int(getattr(settings, "KIE_REQUEST_RETRIES", 3)) + 1):
                try:
                    async with httpx.AsyncClient(timeout=settings.KIE_HTTP_TIMEOUT) as client:
                        resp = await client.get(f"{_STATUS_URL}?taskId={task_id}", headers=headers)
                        resp.raise_for_status()
                    break
                except Exception as req_exc:
                    if req_try == int(getattr(settings, "KIE_REQUEST_RETRIES", 3)):
                        raise req_exc
                    await asyncio.sleep(1)
            data  = resp.json().get("data", {})
            state = data.get("state")
            if state == "success":
                urls = json.loads(data.get("resultJson", "{}")).get("resultUrls", [])
                if urls:
                    logger.info("[%s] Realism pass complete (attempt %d)", pose_label, attempt)
                    return urls[0]
                raise RuntimeError("Realism task succeeded but no resultUrls found.")
            if state == "fail":
                raise RuntimeError("Realism task failed.")
            logger.debug("[%s] Realism poll #%d \u2014 state=%s", pose_label, attempt, state)
            await asyncio.sleep(settings.SEEDDREAM_RETRY_DELAY)
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("[%s] Realism poll #%d error: %s", pose_label, attempt, exc)
            await asyncio.sleep(settings.SEEDDREAM_RETRY_DELAY)
    raise RuntimeError(f"Realism task timed out after {settings.SEEDDREAM_MAX_RETRIES} attempts.")


_SEEDDREAM_PROMPT_LIMIT = 10000   # kie.ai official limit per API docs


async def _submit_seeddream_task(prompt: str, image_urls: List[str], pose_label: str) -> str:
    """Submit a SeedDream generation task (seedream/4.5-edit)."""
    logger.info(
        "[%s] Submitting SeedDream %s (%d chars, %d images)...",
        pose_label, settings.SEEDDREAM_MODEL, len(prompt), len(image_urls),
    )
    payload = json.dumps({
        "model": settings.SEEDDREAM_MODEL,
        "input": {
            "prompt":        prompt,
            "image_urls":    image_urls,
            "aspect_ratio":  settings.SEEDDREAM_ASPECT,
            "quality":       settings.SEEDDREAM_QUALITY,
        },
    })
    headers = {
        "Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}",
        "Content-Type":  "application/json",
    }
    for attempt in range(1, int(getattr(settings, "KIE_REQUEST_RETRIES", 3)) + 1):
        try:
            async with httpx.AsyncClient(timeout=settings.KIE_HTTP_TIMEOUT) as client:
                resp = await client.post(_CREATE_URL, headers=headers, content=payload)
                resp.raise_for_status()
            break
        except Exception as exc:
            logger.warning("[%s] SeedDream submit attempt %d failed: %s", pose_label, attempt, exc)
            if attempt == int(getattr(settings, "KIE_REQUEST_RETRIES", 3)):
                raise RuntimeError(f"SeedDream submit failed after {attempt} attempts: {exc}") from exc
            await asyncio.sleep(2)

    task_id = resp.json().get("data", {}).get("taskId")
    if not task_id:
        raise RuntimeError(f"No taskId returned (SeedDream): {resp.text}")
    logger.info("[%s] SeedDream submitted — task_id=%s", pose_label, task_id)
    return task_id


async def _poll_task(task_id: str, pose_label: str) -> str:
    logger.info("[%s] Polling SeedDream task_id=%s ...", pose_label, task_id)
    headers = {"Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}"}
    for attempt in range(1, settings.SEEDDREAM_MAX_RETRIES + 1):
        try:
            resp = None
            for req_try in range(1, int(getattr(settings, "KIE_REQUEST_RETRIES", 3)) + 1):
                try:
                    async with httpx.AsyncClient(timeout=settings.KIE_HTTP_TIMEOUT) as client:
                        resp = await client.get(f"{_STATUS_URL}?taskId={task_id}", headers=headers)
                        resp.raise_for_status()
                    break
                except Exception as req_exc:
                    if req_try == int(getattr(settings, "KIE_REQUEST_RETRIES", 3)):
                        raise req_exc
                    await asyncio.sleep(1)
            data  = resp.json().get("data", {})
            state = data.get("state")
            if state == "success":
                urls = json.loads(data.get("resultJson", "{}")).get("resultUrls", [])
                if urls:
                    logger.info("[%s] Task complete — result URL obtained (attempt %d)", pose_label, attempt)
                    return urls[0]
                raise RuntimeError("Task succeeded but no resultUrls found.")
            if state == "fail":
                raise RuntimeError("SeedDream task failed.")
            logger.debug("[%s] Poll #%d — state=%s", pose_label, attempt, state)
            await asyncio.sleep(settings.SEEDDREAM_RETRY_DELAY)
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("[%s] Poll #%d error: %s", pose_label, attempt, exc)
            await asyncio.sleep(settings.SEEDDREAM_RETRY_DELAY)
    raise RuntimeError(f"SeedDream task timed out after {settings.SEEDDREAM_MAX_RETRIES} attempts.")


# ---------------------------------------------------------------------------
# Image download + resize helpers
# ---------------------------------------------------------------------------

async def _download_bytes(url: str, label: str = "") -> bytes:
    logger.info("[%s] Downloading image from URL...", label)
    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(timeout=settings.KIE_HTTP_TIMEOUT) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            data = resp.content
            logger.info("[%s] Image downloaded — %d bytes", label, len(data))
            return data
        except Exception as exc:
            logger.warning("[%s] Download attempt %d/3 failed: %s", label, attempt, exc)
            if attempt == 3:
                raise RuntimeError(f"Failed to download image after 3 attempts: {exc}")
            await asyncio.sleep(3)


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
# Single pose worker
# ---------------------------------------------------------------------------

async def _process_one_pose(
    pose_idx:      int,
    pose_data:     dict,
    image_urls:    List[str],
    photoshoot_id: str,
    req_snapshot:  dict,
    *,
    upscaling_col=None,
) -> dict:
    """
    Full async pipeline for one pose. Returns output_image dict or raises on failure.
    Flow: SeedDream generation -> nano-banana-pro realism pass -> configured upscale enhancement.

    ``pose_data`` is ``{"image_url": str, "pose_prompt": str}``.
    When ``image_url`` (mannequin PNG) is present it is appended to the reference images
    and the prompt tells SeedDream to copy only the body posture from it.
    """
    pose_label    = f"pose-{pose_idx:02d}"
    image_id      = str(uuid.uuid4())
    prefix        = f"photoshoots/{photoshoot_id}/{image_id}"
    mannequin_url = (pose_data.get("image_url") or "").strip()
    pose_prompt   = pose_data.get("pose_prompt") or ""
    logger.info("[%s] \u2500\u2500 Starting pose pipeline \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500", pose_label)
    logger.info("[%s] mannequin_image=%s  pose_prompt_len=%d",
                pose_label, bool(mannequin_url), len(pose_prompt))

    # ── Step 1: build prompt and submit to SeedDream ─────────────────────
    has_back      = bool(req_snapshot.get("back_garment_image", ""))
    has_mannequin = bool(mannequin_url)
    prompt = _build_compact_prompt(
        pose_prompt, has_back, req_snapshot,
        has_mannequin_image=has_mannequin,
    )
    logger.info("[%s] SeedDream prompt (%d chars, back=%s, mannequin=%s)",
                pose_label, len(prompt), has_back, has_mannequin)

    urls_for_task = list(image_urls)
    if has_mannequin:
        urls_for_task.append(mannequin_url)
        logger.info("[%s] Mannequin image appended as IMG%d", pose_label, len(urls_for_task))

    counter_seeddream_task = 0
    while counter_seeddream_task < 3:
        try:
            task_id = await _submit_seeddream_task(prompt, urls_for_task, pose_label)
            seeddream_url = await _poll_task(task_id, pose_label)
            logger.info("[%s] SeedDream complete \u2014 URL received", pose_label)
            if seeddream_url:
                break
            counter_seeddream_task += 1
        except Exception as e:
            counter_seeddream_task += 1
            await asyncio.sleep(3)

    # ── Step 2: realism pass (nano-banana-pro) ───────────────────────────
    realism_prompt = _build_realism_prompt()
    counter_realism_task = 0
    while counter_realism_task < 3:
        try:
            realism_task_id = await _submit_realism_task(realism_prompt, seeddream_url, pose_label)
            result_url_4k = await _poll_realism_task(realism_task_id, pose_label)
            if result_url_4k:
                break
            counter_realism_task += 1
        except Exception as e:
            counter_realism_task += 1
            await asyncio.sleep(3)
    
    logger.info("[%s] Realism pass complete \u2014 4K URL received", pose_label)

    bytes_4k = await _download_bytes(result_url_4k, pose_label)

    # ── Step 3: resize realism output 4K \u2192 2K / 1K ──────────────────────
    logger.info("[%s] Resizing 4K \u2192 2K...", pose_label)
    bytes_2k = await asyncio.get_running_loop().run_in_executor(None, _resize_image, bytes_4k, 2048)
    logger.info("[%s] Resizing 4K \u2192 1K...", pose_label)
    bytes_1k = await asyncio.get_running_loop().run_in_executor(None, _resize_image, bytes_4k, 1024)
    logger.info("[%s] Resize complete", pose_label)

    logger.info("[%s] Uploading realism 4K / 2K / 1K to S3 (prefix=%s)...", pose_label, prefix)

    url_4k, url_2k, url_1k = await asyncio.gather(
        upload_bytes_to_s3(bytes_4k, f"{prefix}_4k.png", "image/png"),
        upload_bytes_to_s3(bytes_2k, f"{prefix}_2k.png", "image/png"),
        upload_bytes_to_s3(bytes_1k, f"{prefix}_1k.png", "image/png"),
    )
    logger.info("[%s] Uploaded realism 4K, 2K, 1K", pose_label)

    # ── Step 4: Upscale enhancement (provider from WHICH_UPSCALE) ─────────
    logger.info("[%s] Sending realism output to configured upscale provider...", pose_label)
    upscale_result = await enhance_and_upload(
        image_bytes=bytes_4k,
        photoshoot_id=photoshoot_id,
        image_id=image_id,
        seeddream_4k_url=url_4k,
        seeddream_2k_url=url_2k,
        seeddream_1k_url=url_1k,
        source_image_url=result_url_4k,
        upscaling_col=upscaling_col,
    )
    logger.info("[%s] Upscale enhancement complete \u2014 upscaled 2K: %s", pose_label,
                upscale_result.get("2k_upscaled", "N/A")[:80])

    # Use the upscaled 2K as the primary display image; fall back to realism 2K
    display_image = upscale_result.get("2k_upscaled") or url_2k

    logger.info("[%s] ── Pose pipeline COMPLETE ─────────────────────", pose_label)
    return {
        "image_id":        image_id,
        "pose_prompt":     pose_prompt,
        "pose_image_url":  mannequin_url,
        "image":           display_image,
    }


# ---------------------------------------------------------------------------
# Credit deduction helper
# ---------------------------------------------------------------------------

async def _deduct_photoshoot_credits(
    user_id: str,
    total_credit: float,
    photoshoot_id: str,
    regeneration_type: str = "",
    regenerate_photoshoot_id: str = "",
    image_ids: list = None,
    credit_per_image: float = None,
    users_col=None,
    history_col=None,
) -> None:
    logger.info("[credits] Deducting %.2f credits from user_id=%s for photoshoot=%s",
                total_credit, user_id, photoshoot_id)
    if users_col is None:
        users_col = get_users_collection()
    if history_col is None:
        history_col = get_credit_history_collection()

    user = await users_col.find_one({"user_id": user_id})
    if not user:
        logger.error("[credits] User not found: %s — skipping credit deduction", user_id)
        return

    old_credits = float(user.get("credits", 0))
    new_credits = round(old_credits - total_credit, 4)
    await users_col.update_one(
        {"user_id": user_id},
        {"$set": {"credits": new_credits, "updated_at": datetime.now(timezone.utc)}},
    )
    logger.info("[credits] Credits updated: %.4f → %.4f", old_credits, new_credits)

    feature_name = "photoshoot_regenerate" if regeneration_type == "regenerate" else "photoshoot_generation"
    notes        = f"Photoshoot {photoshoot_id}"
    if regeneration_type:
        notes = f"{regeneration_type} — new photoshoot {photoshoot_id}"
        if regenerate_photoshoot_id:
            notes += f" (from {regenerate_photoshoot_id})"

    history_doc = {
        "history_id":       str(uuid.uuid4()),
        "user_id":          user_id,
        "feature_name":     feature_name,
        "credit":           total_credit,
        "credit_per_image": credit_per_image if credit_per_image is not None else settings.CREDIT_SINGLE_PHOTOSHOOT_PER_IMAGE,
        "image_ids":        image_ids or [],
        "type":             "deduct",
        "thumbnail_image":  "",
        "notes":            notes,
        "photoshoot_id":    photoshoot_id,
        "created_at":       datetime.now(timezone.utc),
    }
    if regeneration_type:
        history_doc["regeneration_type"]        = regeneration_type
        history_doc["regenerate_photoshoot_id"] = regenerate_photoshoot_id

    await history_col.insert_one(history_doc)
    logger.info("[credits] Credit history record inserted")


# ---------------------------------------------------------------------------
# Public background job entry point
# ---------------------------------------------------------------------------

async def run_photoshoot_job(photoshoot_id: str, req: dict, motor_client=None) -> None:
    """
    Runs the full photoshoot pipeline, updates the photoshoot document.
    Credits are deducted only for poses that complete the full pipeline including Modal upscale.

    motor_client: optional AsyncIOMotorClient.  When provided (Celery path),
    collections are derived from it so Motor never touches the closed global
    event loop.  When None (FastAPI BackgroundTask path), the global singleton
    collections are used as before.
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
        _db  = motor_client[_s.MONGO_DB_NAME]
        col  = _db["photoshoots"]
        poses_col = _db["poses_data"]
        upscaling_col = _db["upscaling_data"]
        _get_bg  = lambda: _db["backgrounds"]
        _get_mf  = lambda: _db["model_faces"]
        _get_usr = lambda: _db["users"]
        _get_ch  = lambda: _db["credit_history"]
    else:
        col      = get_photoshoots_collection()
        poses_col = None
        upscaling_col = None
        _get_bg  = get_backgrounds_collection
        _get_mf  = get_model_faces_collection
        _get_usr = get_users_collection
        _get_ch  = get_credit_history_collection

    job_start = time.time()

    try:
        # ── Step 1: resolve pose prompts ──────────────────────────────────
        logger.info("[job] STEP 1 — Resolving pose prompts...")
        pose_data_list = await resolve_poses(req, poses_col=poses_col)

        if not pose_data_list:
            raise ValueError("No poses could be resolved.")

        logger.info("[job] STEP 1 DONE — %d pose(s) ready", len(pose_data_list))

        # ── Step 2: fetch background and model face URLs ──────────────────
        logger.info("[job] STEP 2 — Fetching background and model face from DB...")
        bg_doc = await _get_bg().find_one({"background_id": req["background_id"]})
        if not bg_doc:
            raise ValueError(f"Background not found: {req['background_id']}")
        background_url = bg_doc["background_url"]
        logger.info("[job] Background found: %s", background_url[:80])

        mf_doc = await _get_mf().find_one({"model_id": req["model_id"]})
        if not mf_doc:
            raise ValueError(f"Model face not found: {req['model_id']}")
        model_face_url = mf_doc["face_url"]
        logger.info("[job] Model face found: %s", model_face_url[:80])

        image_urls = [req["front_garment_image"]]
        if req.get("back_garment_image"):
            image_urls.append(req["back_garment_image"])
            logger.info("[job] Back garment image included")
        image_urls.append(model_face_url)
        image_urls.append(background_url)
        logger.info("[job] STEP 2 DONE — %d reference images assembled", len(image_urls))

        # ── Step 3: process all poses concurrently ────────────────────────
        logger.info("[job] STEP 3 — Launching %d pose(s) concurrently...", len(pose_data_list))

        tasks = [
            _process_one_pose(
                idx, pd, image_urls, photoshoot_id, req,
                upscaling_col=upscaling_col,
            )
            for idx, pd in enumerate(pose_data_list, 1)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output_images = []
        failed_poses  = []
        for idx, result in enumerate(results, 1):
            if isinstance(result, Exception):
                failed_poses.append({"pose_index": idx, "error": str(result)})
                logger.error("[job] pose-%02d FAILED: %s", idx, result)
            else:
                output_images.append(result)
                logger.info("[job] pose-%02d SUCCEEDED", idx)

        logger.info("[job] STEP 3 DONE — %d succeeded, %d failed",
                    len(output_images), len(failed_poses))

        # ── Step 4: deduct credits only for fully successful poses ───────
        # Each success includes SeedDream → Modal upscale → S3 (see _process_one_pose).
        successful_count = len(output_images)
        credit_per_image = float(req.get("credit_per_image", settings.CREDIT_SINGLE_PHOTOSHOOT_PER_IMAGE))
        total_credit = round(successful_count * credit_per_image, 4)
        generated_image_ids = [img["image_id"] for img in output_images]

        if total_credit > 0:
            logger.info("[job] STEP 4 — Deducting %.2f credits for %d completed image(s)...",
                        total_credit, successful_count)
            await _deduct_photoshoot_credits(
                req["user_id"],
                total_credit,
                photoshoot_id,
                regeneration_type=req.get("regeneration_type", ""),
                regenerate_photoshoot_id=req.get("regenerate_photoshoot_id", ""),
                image_ids=generated_image_ids,
                credit_per_image=credit_per_image,
                users_col=_get_usr(),
                history_col=_get_ch(),
            )
            logger.info("[job] STEP 4 DONE — %.2f credits deducted", total_credit)
        else:
            logger.info("[job] STEP 4 — No credits deducted (no pose completed full pipeline)")

        # ── Step 5: update photoshoot document ───────────────────────────
        logger.info("[job] STEP 5 — Updating photoshoot document in DB...")
        final_status = "completed" if not failed_poses else "partial"
        await col.update_one(
            {"photoshoot_id": photoshoot_id},
            {"$set": {
                "output_images":       output_images,
                "failed_poses":        failed_poses,
                "total_credit":        total_credit,
                "is_credit_deducted":  bool(total_credit > 0),
                "is_completed":        True,
                "status":              final_status,
                "updated_at":          datetime.now(timezone.utc),
            }},
        )

        elapsed = round(time.time() - job_start, 1)
        logger.info("=" * 70)
        logger.info("[job] Photoshoot job FINISHED — photoshoot_id=%s | status=%s | elapsed=%.1fs",
                    photoshoot_id, final_status, elapsed)
        logger.info("=" * 70)

    except Exception as exc:
        elapsed = round(time.time() - job_start, 1)
        logger.error("[job] Photoshoot job FAILED — photoshoot_id=%s | error=%s | elapsed=%.1fs",
                     photoshoot_id, exc, elapsed)
        await col.update_one(
            {"photoshoot_id": photoshoot_id},
            {"$set": {
                "status":     "failed",
                "error":      str(exc),
                "updated_at": datetime.now(timezone.utc),
            }},
        )


def merge_photoshoot_batch_configs(default_config: dict, list_item: dict) -> dict:
    """
    Merge shared defaults with one row. ``list_item`` keys override ``default_config``.
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
