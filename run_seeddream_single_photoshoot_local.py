#!/usr/bin/env python3
"""
Run SeedDream only (kie.ai) with the same inputs as POST /api/v1/photoshoots/ — no Modal,
no S3, no credits. Results are written under OUTPUT_DIR.

Configure everything via variables below (no CLI args). Loads .env from the project root
for SEEDDREAM_API_KEY, SEEDDREAM_MODEL, SEEDDREAM_QUALITY, SEEDDREAM_ASPECT,
SEEDDREAM_MAX_RETRIES, SEEDDREAM_RETRY_DELAY, MONGO_URI, MONGO_DB_NAME, GEMINI_* when needed.

Set RUN_POSES_IN_PARALLEL to run all resolved poses at once (like production), or False to run
one pose at a time (easier on API rate limits).
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project root on sys.path (run: python run_seeddream_single_photoshoot_local.py)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# =============================================================================
# ——— Same fields as CreatePhotoshootRequest + extras used by the pipeline ———
# =============================================================================

# If True, resolve background_url / face_url from Mongo (background_id / model_id).
# If False, set BACKGROUND_URL and MODEL_FACE_URL manually.
RESOLVE_REF_URLS_FROM_DB = False

BACKGROUND_ID = ""
MODEL_ID = ""

# Used when RESOLVE_REF_URLS_FROM_DB is False (must be reachable HTTP(S) URLs).
BACKGROUND_URL = "https://aavishailabs-uploads-prod.s3.eu-north-1.amazonaws.com/backgrounds/mediterranean_coastal_archway_f9b262e5.jpg"
MODEL_FACE_URL = "https://aavishailabs-uploads-prod.s3.eu-north-1.amazonaws.com/model-faces/adult_female_495c3a0f.png"

FRONT_GARMENT_IMAGE = "https://aavishailabs-uploads-prod.s3.eu-north-1.amazonaws.com/users/5971e90a-2682-4c24-a16e-2eda4162f4e8/606267637a214115968fb5cc4ec2d2a7.jpeg"
BACK_GARMENT_IMAGE = ""  # optional; include back view URL for two-reference garment mode

ETHNICITY = "Indian"
GENDER = "female"
SKIN_TONE = "medium"
AGE = "24"
AGE_GROUP = "adult"

WEIGHT = "regular"
HEIGHT = "regular"

UPPER_GARMENT_TYPE = ""
UPPER_GARMENT_SPECIFICATION = ""
LOWER_GARMENT_TYPE = ""
LOWER_GARMENT_SPECIFICATION = ""
ONE_PIECE_GARMENT_TYPE = ""
ONE_PIECE_GARMENT_SPECIFICATION = ""

FITTING = "regular fit"

# Stored on photoshoot docs; only background_type is read by the SeedDream prompt builder.
LIGHTING_STYLE = "natural"
ORNAMENTS = ""
SKU_ID = "HSJSN"

# Optional; included in the SeedDream prompt when set (indoor / outdoor / studio / general).
BACKGROUND_TYPE = "studio"

# Production API uses ``poses_ids`` only (min 1), resolved from Mongo ``poses_data``.
# For local runs without DB pose rows, set USE_LEGACY_INLINE_PROMPTS = True and fill LEGACY_POSES_PROMPTS.
USE_LEGACY_INLINE_PROMPTS = True

POSES_IDS: list[str] = [
    # Example: paste pose_id values from your DB when USE_LEGACY_INLINE_PROMPTS is False.
]

LEGACY_POSES_PROMPTS: list[str] = [
    "Seated posture on a stool, torso upright with a slight forward lean, shoulders relaxed, right shoulder subtly lower than left. Head facing directly forward, neck straight. Right arm bent, elbow resting on thigh, forearm angled, right hand partially inserted into front pocket. Left arm bent, elbow resting on thigh, forearm angled, left hand resting flat on thigh, fingers gently curled. Right leg bent at knee, foot flat on ground, toes pointing forward. Left leg bent at knee, crossed over the right, left foot resting on a stool rung, angled slightly right. Weight balanced, centered over the seat.",
    "Standing full body profile, torso upright, upper body leaning slightly back, right shoulder elevated and pulled back, left shoulder relaxed and slightly forward, hips level and stable, head turned subtly to the right, neck erect, right arm raised, elbow sharply bent, forearm extended upwards, right hand placed gently behind the head, fingers softly curved, left arm hanging naturally by the side, elbow slightly flexed, left wrist straight, left hand relaxed, fingers lightly curled, palm facing the thigh, both legs straight, knees unbent, feet flat on the ground, parallel and pointing directly forward, weight evenly distributed, stable center of gravity.",
    "Standing body, slightly angled to the right, torso subtly turned right, shoulders relatively level, left shoulder slightly forward. Head turned slightly right, gaze forward, neck upright. Right arm bent at the elbow, forearm crossing the body, hand resting gently on the left upper arm. Left arm bent at the elbow, forearm resting across the front, hand holding the right forearm. Both elbows bent approximately 90-100 degrees. Wrists slightly bent, fingers relaxed and gently curled. Legs mostly straight, close together, knees unbent. Weight evenly distributed, stable centre of gravity, feet flat.",
    "Seated pose, torso upright, slightly leaning forward, shoulders level and relaxed, head facing directly forward, neck straight, both arms bent at the elbows, forearms resting on upper thighs, elbows at approximately 90 degrees, wrists slightly bent, hands relaxed, fingers gently curled, left hand resting on left thigh, right hand resting on right thigh, both legs bent at the knees, thighs parallel to the ground, knees at approximately 90 degrees, both feet flat on the ground, pointing forward, feet slightly wider than hip-width apart, weight evenly distributed, center of gravity low and stable.",
    "Standing, relaxed posture, body turned away, torso rotated approximately 45 degrees left, presenting back and left side, shoulders level, left shoulder slightly advanced, right shoulder slightly retracted, hips level, aligned with torso rotation, head turned sharply left, gaze directed horizontally, no discernible tilt, neck held upright, natural curve, arms extended downwards, close to body, elbows with very slight natural bend, wrists neutral, unbent, hands resting gently against outer upper thighs, fingers relaxed, slightly curved inwards, thumb alongside, legs straight, standing parallel, knees fully extended, weight balanced evenly, center of gravity stable.",
]

OUTPUT_DIR = _ROOT / "seeddream_local_output"
RUN_LABEL = "test-run"

# Multiple poses: True = all poses in parallel (same as production Celery job); False = one after another.
RUN_POSES_IN_PARALLEL = True

# Realism pass: after SeedDream, send output to nano-banana-pro for skin/hair/blending enhancement.
ENABLE_REALISM_PASS = True
REALISM_MODEL = "nano-banana-pro"
REALISM_QUALITY = "4K"
REALISM_ASPECT = "9:16"

_PROMPT_LIMIT = 3000
_REALISM_PROMPT_LIMIT = 10000

_SANITIZE_PATTERNS = [
    re.compile(
        r'\b(wearing|dressed in|outfit|garment|clothing|cloth|clothes|fabric|'
        r'top|bottom|skirt|pants|trousers|shirt|dress|blouse|jacket|coat|suit|'
        r'saree|sari|kurta|lehenga|churidar|dupatta|salwar|kameez|gown|frock|'
        r'shorts|jeans|denim|sweater|hoodie|cardigan|vest|crop|bralette|'
        r'sleeve|collar|neckline|hem|waist|belt)\b', re.IGNORECASE),
    re.compile(r'\b(street|garden|park|beach|office|store|shop)\b', re.IGNORECASE),
    re.compile(r'\b(male|female|man|woman|boy|girl|he|she|his|her|they|them)\b', re.IGNORECASE),
]


def _sanitize_pose(pose: str) -> str:
    result = pose
    for pat in _SANITIZE_PATTERNS:
        result = pat.sub('', result)
    return re.sub(r'\s{2,}', ' ', result).strip()


def _build_compact_prompt(
    pose: str,
    has_back: bool,
    req: dict,
    *,
    has_mannequin_image: bool = False,
) -> str:
    """Build a <=3000-char SeedDream prompt covering all photoshoot cases."""
    fi = 3 if has_back else 2
    bi = 4 if has_back else 3
    mi = bi + 1

    if has_back:
        img_ref = f"IMG1=garment front, IMG2=garment back, IMG{fi}=face, IMG{bi}=background"
        g_imgs = "IMG1+IMG2"
    else:
        img_ref = f"IMG1=garment, IMG{fi}=face, IMG{bi}=background"
        g_imgs = "IMG1"

    if has_mannequin_image:
        img_ref += f", IMG{mi}=POSE MANNEQUIN (body posture reference ONLY)"

    ug = req.get("upper_garment_type", "").strip()
    us = req.get("upper_garment_specification", "").strip()
    lg = req.get("lower_garment_type", "").strip()
    ls = req.get("lower_garment_specification", "").strip()
    op = req.get("one_piece_garment_type", "").strip()
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
            paired = " Add matching lower garment—no bare legs."
        elif lg and not ug:
            paired = " Add matching upper garment—no bare torso."

    # ── garment-specific wearing rules ────────────────────────────────────
    garment_rules = ""
    hands_must_show = False
    if op:
        if "saree" in op_lower or "sari" in op_lower:
            garment_rules = (
                "\n[GARMENT]\n"
                "SAREE—Gujarati drape: pallu over LEFT shoulder to back; neat waist pleats; "
                "fitted blouse + petticoat (chaniya); clear front pleats. "
                "Hands and all fingers visible—pallu must not cover them.\n"
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
                    "Dupatta Gujarati—one shoulder or across chest. "
                    "Hands and all fingers visible—dupatta must not cover them.\n"
                )
                hands_must_show = True
            else:
                garment_rules = (
                    "\n[GARMENT WEARING RULE]\n"
                    "This is a 2-PIECE DRESS. Wear only the top and bottom—"
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
                f"This is a ONE-PIECE garment ({op}). Wear it exactly as a single piece—"
                "do NOT add extra layers, scarves, or separate top/bottom.\n"
            )

    weight = req.get("weight", "regular").strip().lower()
    height = req.get("height", "regular").strip().lower()
    w_desc = {
        "fat": "plus-size/heavy build: broad torso, thick limbs, full waist, wide hips—genuinely heavy silhouette",
        "slim": "slim/lean: narrow shoulders, defined waist, slender limbs",
    }.get(weight, "average build, natural proportions")
    h_desc = {
        "short": "shorter than average",
        "tall": "taller than average",
    }.get(height, "average height")

    bg_type = req.get("background_type", "").strip().lower()
    bg_label = bg_type if bg_type in ("indoor", "outdoor", "studio") else "general"

    clean_pose = _sanitize_pose(pose) if pose else "Natural relaxed full-body fashion pose"

    gender = req.get("gender", "").strip().lower()
    orn = req.get("ornaments", "").strip()
    orn_lower = orn.lower()
    bag = ""
    if gender == "female" and any(kw in orn_lower for kw in ("bag", "purse", "handbag")):
        bag = " Add matching bag/purse held naturally."

    hands_note = ""
    if hands_must_show:
        hands_note = " Do not let dupatta/pallu/fabric hide hands or fingers."

    prompt = (
        f"Hyperrealistic editorial fashion photograph—real person, NOT illustration, NOT AI-looking, NOT plastic skin.\n"
        f"\n"
        f"Refs: {img_ref}\n"
        f"\n"
        f"[GARMENT—EXACT MATCH {g_imgs}]{gt_line}\n"
        f"Copy exact fabric, pattern, color, print, embroidery, buttons, pockets, stitching, hem, neckline, "
        f"sleeves, cuffs, weave, closures, trims, pleats, natural drape and wrinkles. "
        f"No color shift, no simplification. Fitting: {fitting}.{paired}\n"
        + garment_rules
        + f"\n"
        f"[FACE—EXACT COPY IMG{fi}]\n"
        f"Lock all features: face shape, eyes, brows, nose, lips, jaw, chin, cheekbones, ears, hairline, "
        f"hair style/color/length, skin tone/undertone, moles/marks. No beautification. Expression identical to IMG{fi}.\n"
        f"\n"
        f"[SKIN REALISM—NO BEAUTY FILTER]\n"
        f"Visible pores, peach-fuzz, subtle veins/SSS, natural tone variation, knuckle and crease lines, "
        f"realistic elbow texture. No beauty smoothing—raw hyperreal skin.\n"
        f"\n"
        f"[BG + IN-SCENE SUBJECT—IMG{bi} ({bg_label})]\n"
        f"Single real capture—not a composite paste. Light the subject ONLY from the same light field as IMG{bi} "
        f"(key direction, WB, shadow softness); avoid flat frontal studio fill. "
        f"Ambient color bounce from walls, floor, foliage, visible lamps onto skin, garment, and shoes. "
        f"Ambient occlusion plus tight contact shadow at soles and hem where fabric meets floor; cast shadow matches scene key. "
        f"Feet on the floor plane following tile/arch perspective. Rim on hair/shoulders from scene sources; "
        f"subtle light wrap on silhouette edges toward bright areas—no razor cutout. "
        f"Match subject exposure, contrast, and saturation to the environment (not brighter/punchier than the room). "
        f"No floating, no halo. Do not alter background.\n"
        f"\n"
        f"[BODY] {req.get('gender','')}, {req.get('ethnicity','')}, age {req.get('age','')} ({req.get('age_group','')}), "
        f"skin: {req.get('skin_tone','')}.\n"
        f"Weight: {w_desc}. Height: {h_desc}. Body must visibly reflect this build.\n"
        f"\n"
        + (
            f"[POSE — EXACT COPY FROM IMG{mi} MANNEQUIN]\n"
            f"IMG{mi} is a grey featureless mannequin. It is ONLY a body-posture reference.\n"
            f"You MUST replicate the EXACT pose from IMG{mi} with ZERO deviation:\n"
            f"- HANDS: Copy exact hand position—if hand is in pocket, keep it in pocket; if hand is on hip, "
            f"keep it on hip; if hands are clasped, keep them clasped. Reproduce exact finger curl, wrist angle, "
            f"and palm orientation. Do NOT invent a different hand placement.\n"
            f"- BODY/TORSO: Copy exact torso lean angle, shoulder tilt, chest orientation, spine curvature, "
            f"hip rotation, and weight shift. Do NOT straighten or alter the torso.\n"
            f"- LEGS/FEET: Copy exact leg stance—crossed, apart, bent, straight—exactly as shown. "
            f"Replicate knee bend angle, foot placement, foot direction, and weight distribution between legs.\n"
            f"- HEAD: Copy exact head tilt, turn direction, chin angle, and gaze direction.\n"
            f"IGNORE everything else about IMG{mi}—ignore its grey skin, featureless face, bald head, "
            f"white background, and clothing. Only replicate the body posture on the real human model.{hands_note}\n"
            if has_mannequin_image else
            f"[POSE] {clean_pose}\n"
        )
        + f"\n"
        f"Matching footwear, no bare feet.{bag} Ornaments: {orn or 'none'}.\n"
        f"85mm f/2.8, shallow DOF on subject, 4K 9:16, editorial color; natural hair detail; one coherent in-camera exposure."
    )

    if len(prompt) > _PROMPT_LIMIT:
        prompt = prompt[:_PROMPT_LIMIT]

    return prompt


def _build_realism_prompt() -> str:
    """10 000-char editor prompt for nano-banana-pro realism pass.

    Role: photo retoucher / editor — enhance realism on the INPUT image.
    PRESERVE everything (face, body, clothes, background, pose, composition).
    ONLY add micro-level realism details.
    """
    return """You are an expert photo retoucher. You receive a single fashion photoshoot image (IMG1).
Your ONLY job is to make it look like a real, unretouched DSLR photograph — NOT an AI render.
Do NOT change composition, pose, background geometry, or facial identity (bone structure, feature shapes/sizes, expression, gaze, moles/freckles, hair style).
PRIORITY-1 (NON-NEGOTIABLE): If skin looks airbrushed, poreless, waxy, or uniform, you MUST add visible micro-pores, fine grain, subsurface veins, and natural specular micro-highlights—that is required realism, NOT an identity change.

=== HARD CONSTRAINTS — NEVER CHANGE THESE (only add surface/lighting micro-detail ON the existing subject) ===
- FACE IDENTITY: Same person as IMG1. Do NOT change face shape, bone structure, feature size/position/spacing, ethnicity read, age, eye size/openness, nose width/length, lip fullness, jaw/chin line, brow height, ear shape, or expression/gaze beyond rendering existing skin more realistically. Skin edits are SURFACE ONLY (pores, veins, grain, tiny speculars on existing skin)—never cosmetic reshaping or beautification that alters who they are.
- GARMENT: Same exact outfit as IMG1—same garment pieces, colors, print/pattern placement, embroidery, cut, buttons, hem, drape silhouette, accessories. Do NOT swap, add, remove, recolor, or reinterpret the clothing.
- BACKGROUND: Do NOT repaint, replace, extend, crop, or alter the environment. Same architecture, floor, sky, props, colors, and layout as IMG1. All scene edits are limited to how the subject's edges and light response sit against that unchanged background (no new bg elements).
- POSE: IDENTICAL to IMG1—same limb angles, torso orientation, head tilt, hand/finger placement, foot placement, and framing. Do NOT move, rotate, rescale, or reposition the person in the frame; do not change gesture or stance.

=== ABSOLUTE PRESERVATION LOCKS (do NOT alter) ===

[FACE IDENTITY LOCK]
- Preserve every facial feature exactly: face shape, eye shape/size/color/spacing, eyebrow shape/thickness/arch, nose bridge/tip/width, lip shape/fullness/color, jaw angle, chin shape, cheekbone prominence, ear shape, skin tone/undertone, all moles/marks/freckles, facial hair if present.
- Preserve exact expression, head angle, gaze direction.
- Do NOT reshape, slim, or alter facial proportions or feature geometry.
- Do NOT apply beauty blur, skin smoothing, or porcelain/airbrush filters.
- SKIN TEXTURE (additive only): Intensify DSLR-like micro-pores on nose, inner cheeks, forehead, and chin so they read at NORMAL viewing size (not microscopic). Add fine irregular grain, tiny specular hits on nose bridge and cheek peaks from the SAME light direction as the scene, and subtle blue-green venous hints on temples/sides of face and neck where real skin would show them under sidelight. Hands/arms/legs: veins and tendons must read clearly. Do NOT alter facial structure or proportions to do this—only surface micro-detail. Plastic or featureless skin means failure.

[BODY & POSE LOCK]
- Preserve exact body shape, weight, proportions, height, limb positions, posture, hand placement, finger positions, foot placement.
- Do NOT alter body silhouette, muscle definition, or body mass in any way.

[GARMENT LOCK]
- Preserve every detail of clothing: fabric type, pattern, color, print, embroidery, buttons, pockets, stitching, hem, neckline, sleeves, cuffs, weave texture, closures, trims, pleats, wrinkles, drape, fit, tucked/untucked state.
- Do NOT change any garment color, pattern, or styling.
- You MAY add scene-correct lighting RESPONSE only (how light hits existing fabric): directional highlights and micro-shadows along existing weave, embroidery, folds, and lace; subtle bounce tint—without changing base color, pattern, print registration, or garment shape.

[BACKGROUND & COMPOSITION LOCK]
- Preserve the entire background scene: geometry, objects, colors, sky, furniture, walls, floor, props — everything.
- Preserve the exact framing, crop, aspect ratio, and camera angle.

=== REALISM ENHANCEMENTS (apply these ONLY) ===

[1. HANDS — CRITICAL PRIORITY]
Hands are the #1 AI giveaway. Fix them to look like real human hands:
- FINGER ANATOMY: Each finger must have natural slight curvature — never perfectly straight or rigid. Fingers at rest have a gentle curl. Knuckle joints should show bony protrusion and skin bunching when bent. Inter-finger webbing visible between spread fingers.
- SKIN TEXTURE ON HANDS: Deep knuckle wrinkles and creases, finger joint lines, visible tendons on back of hand, dorsal veins (blue-green subcutaneous), bony knuckle ridges. Palm side: visible palm lines (life line, heart line, head line), finger pad skin ridges.
- NAILS: Natural nail shape with visible cuticles, lunula (half-moon at base), slight translucency showing pink nail bed beneath. Nails should not be perfectly uniform — slight variation in shape, minor ridges, natural sheen not plastic gloss.
- NATURAL GRIP: When hand is in pocket, gripping arm, or resting on surface — fingers must wrap naturally with proper pressure dimpling on contacted surface. No "hovering" fingers. Skin compresses where it touches.
- RELAXED HAND: Resting hands should have slightly curled fingers with natural spacing — not stiff mannequin hands. Thumb sits naturally opposed to fingers, not rigidly parallel.
- WRIST: Visible wrist bones (ulnar styloid), skin crease lines, subtle vein detail on inner wrist.

[2. EYES — MICRO-DETAIL]
Eyes must look alive, not CG-rendered:
- IRIS: Complex iris texture — visible radial fibres (collarette pattern), crypts (dark irregular patches), furrows, and pigment spots. Iris is NOT a uniform solid color; it has concentric rings and radial striations of varying shade.
- SCLERA: NOT perfectly white. Add faint pink/yellowish tint, with very subtle micro blood vessels (thin red/pink veins) especially near inner and outer corners. Slight natural discoloration near the limbus (iris border).
- LIMBAL RING: Darker ring at the outer edge of the iris where it meets the sclera — varies in definition by age.
- TEAR FILM / MOISTURE: Wet glossy reflection on the cornea surface — a single bright specular highlight (catchlight) plus subtle secondary reflections. Slight moisture accumulation at inner corner (lacrimal caruncle — the pink fleshy bit). Very faint wet line along lower lid margin.
- PUPIL: True black, perfectly round, with sharp edge. Subtle depth — it's a hole, not a painted circle.
- EYELIDS: Fine skin texture on eyelids, visible eyelid crease, tiny blood vessels on inner eyelid if visible. Lower lid has slight puffiness/texture, not a clean line.

[3. FOOTWEAR — REAL WORN SHOES]
Shoes must NOT look like 3D renders or brand-new pristine objects:
- WEAR MARKS: Subtle scuff marks, minor sole edge dirt, slight creasing at the toe flex point where the shoe bends during walking. Lace/strap wear. These are real shoes on a real person, not showroom display.
- MATERIAL TEXTURE: Canvas/leather grain visible. Stitching detail — individual stitch holes and thread. Sole rubber texture, not smooth plastic.
- GROUND INTERACTION: Shoe sole is FLAT on the ground surface — not floating. Where shoe meets ground, add contact shadow and slight dirt/dust accumulation at the sole-ground junction. Shoe laces (if present) have natural drape and slight twist, not rigid symmetrical bows.
- LIGHT RESPONSE: Shoes respond to scene lighting — proper shadow under shoe tongue, highlight on toe cap, matte vs glossy areas matching real shoe material. Not uniformly lit.

[4. SKIN TEXTURE — PRIORITY-1 (PORES + VEINS + MICRO-CONTRAST)]
Treat the face like a 24–70mm beauty shot at f/4: micro-structure must READ clearly on screen—not optional polish.
Apply across ALL visible skin (face, neck, décolleté if shown, arms, hands, legs, feet):
- MICRO-PORES (MUST BE VISIBLE): Irregular pore field on nose, inner cheeks, forehead, chin. Each pore: tiny shadowed well + soft rim highlight from scene light—not a flat uniform dot grid. T-zone slightly oilier sheen; do NOT blur or melt pores away.
- MICRO-CONTRAST: Add fine high-frequency grain on cheeks/jaw/forehead (sensor+skin texture), not noise blobs. Break up large smooth skin patches.
- VEINS (READABLE, ANATOMICAL): Hands/forearms/wrists: clearly visible branching blue-green veins and tendons. Neck/temples/side of face: subtle but real venous map where sidelight would reveal it—soft, under-skin, tapered; must react to light (not tattoo-like).
- SPECULAR / OIL: Small natural skin highlights on nose bridge, cheekbones, Cupid's bow, and shoulders that MATCH the scene's key light color and direction—breaks the dead matte AI look.
- PERIORBITAL + CHEEK: Fine micro-creases and micro-grain under eyes and smile zones (light, not deep wrinkles unless already in source).
- PEACH FUZZ: Fine vellus on cheeks, jaw, upper lip, temples, forearms; catchlight on fuzz when backlit.
- SKIN TONE VARIATION: Redder knuckles, elbows, knees, nose tip; darker creases, under-eye.
- SSS: Ear-edge redness when backlit, fingertip translucency, nostril light transmission.
- IMPERFECTIONS: Tiny bumps or blemish OK. Zero porcelain/airbrush/wax.

[5. HAIR REALISM + EDGE FIX — PRIORITY-2 LIGHTING]
- Individual strand detail — not a smooth blob. Each strand has its own path with natural irregularity.
- Flyaway hairs and baby hairs at hairline, temples, nape, part line.
- Specular highlights on strands from the scene key + fill (warm/cool matching environment), not flat global sheen. Rim/backlight on hair mass where background is bright; secondary bounce tint from walls/sky onto dark hair.
- HAIR EDGE — CRITICAL: NO compositing halo or bright fringe around hair edges against the background. Hair edges must have natural semi-transparent wisps that blend into the background — not a hard cutout. Individual stray hairs extend beyond the main hair silhouette and are semi-transparent where they overlap the background. Match the background color/brightness through thin hair strands at the edge.
- Eyebrows: individual hairs with natural direction. Eyelashes: natural variation in length/curl.

[6. SPATIAL GROUNDING — ANTI-PASTED FIX (pose & bg unchanged)]
POSE IS LOCKED: do not move, rotate, or rescale the subject; do not change limb or foot placement.
Make the existing placement READ MORE grounded WITHOUT changing pose: refine contact shadows, AO at soles, and edge integration so feet read on the existing floor plane at the SAME pixel position as IMG1.
- If feet already touch the ground convincingly, do NOT change position—only improve shadow realism if needed.
- PERSPECTIVE: Keep model foreshortening and scale as in IMG1; only harmonize shadow/light with the unchanged background.

[7. BACKGROUND BLENDING & LIGHTING — PRIORITY-2 (subject only; bg pixels unchanged)]
Background in IMG1 is FIXED—do not edit background pixels, colors, or objects. Only adjust how the SUBJECT receives light so it matches the existing scene:
- Match key light direction, color temperature, and shadow hardness on face, garment, and hair to the environment (e.g. warm sun from alley opening, cool sky fill, colored walls).
- COLOR BOUNCE / GI: Add believable reflected light from ground, buildings, flowers, sky onto lower face, chin, undersides of arms, garment folds, and hair—same hues as nearby surfaces, low intensity.
- Rim and edge light on hair, shoulders, and ear rims from bright areas behind/side of subject; soft light wrap on silhouette against bright bg.
- Ground contact shadow with soft penumbra at feet; darker ambient occlusion where sole meets ground. Shoes cast shadow onto floor consistent with scene light.
- Atmospheric depth: slight haze match if bg is hazy; reflections on glossy props if present.
- Remove halos, edge glow, bright fringe, compositing artifacts—especially hair and shoulders.
- EDGE BLENDING: Seamless boundary; no razor cutout; no aliasing or color fringe. Natural anti-aliasing matching camera optics.

[8. FABRIC & MATERIAL REALISM — PRIORITY-2 LIGHTING]
- Thread weave and embroidery micro-texture visible; tiny thread-level highlights from scene key light.
- Material-appropriate sheen (linen matte, silk specular, denim rough) with directionally correct micro-shadows in folds.
- Realistic wrinkle shadow/highlight tied to the SAME light as background. Natural skin-to-fabric contact shadow.

[9. CAMERA / LENS]
- Mild chromatic aberration at frame edges, natural vignette. Consistent bokeh.
- Film grain / sensor noise (ISO 200-400 DSLR). Kodak Portra / Fuji Pro color. Natural micro-contrast.

[10. FINAL QUALITY]
- Indistinguishable from real DSLR photo at 100% zoom. No plastic/waxy surfaces, no AI uniformity.
- Real photos have micro-imperfections — embrace them. Same resolution and aspect ratio as input.
- RECHECK: Output must still show the SAME face identity, SAME garment, SAME pose, SAME background as IMG1—only surface realism and scene-consistent light on the subject improved."""


_CREATE_URL = "https://api.kie.ai/api/v1/jobs/createTask"
_STATUS_URL = "https://api.kie.ai/api/v1/jobs/recordInfo"


async def _submit_realism_task(prompt: str, image_url: str, pose_label: str) -> str:
    """Submit a nano-banana-pro realism editing task."""
    from app.config import settings

    print(f"[{pose_label}] Submitting realism pass ({REALISM_MODEL}, {len(prompt)} chars)...")
    payload = json.dumps({
        "model": REALISM_MODEL,
        "input": {
            "prompt": prompt,
            "image_input": [image_url],
            "aspect_ratio": REALISM_ASPECT,
            "resolution": REALISM_QUALITY
        },
    })
    headers = {
        "Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(_CREATE_URL, headers=headers, content=payload)
        resp.raise_for_status()
    task_id = resp.json().get("data", {}).get("taskId")
    if not task_id:
        raise RuntimeError(f"[{pose_label}] No taskId returned (realism): {resp.text}")
    print(f"[{pose_label}] Realism task submitted — task_id={task_id}")
    return task_id


async def _poll_realism_task(task_id: str, pose_label: str) -> str:
    """Poll a nano-banana-pro task until success/fail."""
    from app.config import settings

    headers = {"Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}"}
    for attempt in range(1, settings.SEEDDREAM_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(f"{_STATUS_URL}?taskId={task_id}", headers=headers)
                resp.raise_for_status()
            data = resp.json().get("data", {})
            state = data.get("state")
            if state == "success":
                urls = json.loads(data.get("resultJson", "{}")).get("resultUrls", [])
                if urls:
                    print(f"[{pose_label}] Realism pass complete (attempt {attempt})")
                    return urls[0]
                raise RuntimeError("Realism task succeeded but no resultUrls found.")
            if state == "fail":
                raise RuntimeError("Realism task failed.")
            await asyncio.sleep(settings.SEEDDREAM_RETRY_DELAY)
        except RuntimeError:
            raise
        except Exception as exc:
            print(f"[{pose_label}] Realism poll #{attempt} error: {exc}", file=sys.stderr)
            await asyncio.sleep(settings.SEEDDREAM_RETRY_DELAY)
    raise RuntimeError(f"Realism task timed out after {settings.SEEDDREAM_MAX_RETRIES} attempts.")


# ---------------------------------------------------------------------------
async def _resolve_refs_from_db() -> tuple[str, str]:
    from motor.motor_asyncio import AsyncIOMotorClient

    from app.config import settings

    client = AsyncIOMotorClient(settings.MONGO_URI)
    try:
        db = client[settings.MONGO_DB_NAME]
        bg_doc = await db["backgrounds"].find_one({"background_id": BACKGROUND_ID})
        if not bg_doc:
            raise ValueError(f"Background not found: {BACKGROUND_ID}")
        mf_doc = await db["model_faces"].find_one({"model_id": MODEL_ID})
        if not mf_doc:
            raise ValueError(f"Model face not found: {MODEL_ID}")
        return bg_doc["background_url"], mf_doc["face_url"]
    finally:
        client.close()


async def main() -> None:
    back_garment = BACK_GARMENT_IMAGE.strip()

    if USE_LEGACY_INLINE_PROMPTS:
        if not LEGACY_POSES_PROMPTS:
            raise SystemExit("USE_LEGACY_INLINE_PROMPTS requires non-empty LEGACY_POSES_PROMPTS.")
    elif not POSES_IDS:
        raise SystemExit("Set POSES_IDS (at least one pose_id) or enable USE_LEGACY_INLINE_PROMPTS.")

    if RESOLVE_REF_URLS_FROM_DB:
        if not BACKGROUND_ID or not MODEL_ID:
            raise SystemExit("RESOLVE_REF_URLS_FROM_DB requires BACKGROUND_ID and MODEL_ID.")
        background_url, model_face_url = await _resolve_refs_from_db()
    else:
        background_url = BACKGROUND_URL.strip()
        model_face_url = MODEL_FACE_URL.strip()
        if not background_url or not model_face_url:
            raise SystemExit("Set BACKGROUND_URL and MODEL_FACE_URL or enable RESOLVE_REF_URLS_FROM_DB.")

    image_urls = [FRONT_GARMENT_IMAGE.strip()]
    if back_garment:
        image_urls.append(back_garment)
    image_urls.append(model_face_url)
    image_urls.append(background_url)

    req_snapshot: dict = {
        "front_garment_image": FRONT_GARMENT_IMAGE,
        "back_garment_image": back_garment,
        "ethnicity": ETHNICITY,
        "gender": GENDER,
        "skin_tone": SKIN_TONE,
        "age": AGE,
        "age_group": AGE_GROUP,
        "weight": WEIGHT,
        "height": HEIGHT,
        "upper_garment_type": UPPER_GARMENT_TYPE,
        "upper_garment_specification": UPPER_GARMENT_SPECIFICATION,
        "lower_garment_type": LOWER_GARMENT_TYPE,
        "lower_garment_specification": LOWER_GARMENT_SPECIFICATION,
        "one_piece_garment_type": ONE_PIECE_GARMENT_TYPE,
        "one_piece_garment_specification": ONE_PIECE_GARMENT_SPECIFICATION,
        "fitting": FITTING,
        "background_id": BACKGROUND_ID,
        "poses_ids": list(POSES_IDS),
        "model_id": MODEL_ID,
        "lighting_style": LIGHTING_STYLE,
        "ornaments": ORNAMENTS,
        "sku_id": SKU_ID,
        "background_type": BACKGROUND_TYPE,
    }
    if USE_LEGACY_INLINE_PROMPTS:
        req_snapshot["pose_data"] = [
            {"image_url": "", "pose_prompt": p} for p in LEGACY_POSES_PROMPTS
        ]

    from app.services.photoshoot_service import (
        _download_bytes,
        _poll_task,
        _submit_seeddream_task,
        resolve_poses,
    )

    pose_data_list = await resolve_poses(req_snapshot, poses_col=None)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR.resolve() / f"{RUN_LABEL}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_label": RUN_LABEL,
        "req_snapshot": req_snapshot,
        "image_urls": image_urls,
        "pose_data": pose_data_list,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    has_back = bool(back_garment)

    realism_prompt = _build_realism_prompt() if ENABLE_REALISM_PASS else ""
    if realism_prompt:
        (out_dir / "realism_prompt.txt").write_text(realism_prompt, encoding="utf-8")
        print(f"Realism pass ENABLED ({REALISM_MODEL}, {len(realism_prompt)} / {_REALISM_PROMPT_LIMIT} chars)")

    async def run_one_pose(idx: int, pd: dict) -> None:
        pose_label    = f"pose-{idx:02d}"
        mannequin_url = (pd.get("image_url") or "").strip()
        pose_prompt   = pd.get("pose_prompt") or ""
        has_mannequin = bool(mannequin_url)

        prompt = _build_compact_prompt(
            pose_prompt, has_back, req_snapshot,
            has_mannequin_image=has_mannequin,
        )
        (out_dir / f"{pose_label}_seeddream_prompt.txt").write_text(prompt, encoding="utf-8")
        print(f"[{pose_label}] Prompt length: {len(prompt)} / {_PROMPT_LIMIT} chars  mannequin={has_mannequin}")

        urls = list(image_urls)
        if has_mannequin:
            urls.append(mannequin_url)
            print(f"[{pose_label}] Mannequin image appended as IMG{len(urls)}")

        task_id = await _submit_seeddream_task(prompt, urls, pose_label)
        result_url = await _poll_task(task_id, pose_label)
        image_bytes = await _download_bytes(result_url, pose_label)

        out_path = out_dir / f"{pose_label}_seeddream.png"
        out_path.write_bytes(image_bytes)
        print(f"Wrote {out_path} ({len(image_bytes)} bytes)")

        if ENABLE_REALISM_PASS:
            realism_task_id = await _submit_realism_task(realism_prompt, result_url, pose_label)
            realism_url = await _poll_realism_task(realism_task_id, pose_label)
            realism_bytes = await _download_bytes(realism_url, pose_label)

            realism_path = out_dir / f"{pose_label}_realism.png"
            realism_path.write_bytes(realism_bytes)
            print(f"Wrote {realism_path} ({len(realism_bytes)} bytes)")

    tasks = [run_one_pose(i, pd) for i, pd in enumerate(pose_data_list, start=1)]
    if RUN_POSES_IN_PARALLEL:
        print(f"Running {len(tasks)} pose(s) in parallel…")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            for e in errors:
                print(f"Pose error: {e}", file=sys.stderr)
            raise SystemExit(1)
    else:
        print(f"Running {len(tasks)} pose(s) sequentially…")
        for t in tasks:
            await t

    print(f"Done. Output directory: {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
