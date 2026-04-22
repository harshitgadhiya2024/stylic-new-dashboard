"""Compact SeedDream prompt (<=3000 chars). Used by photoshoot_service and check_seeddream."""

import re
from typing import Any

from app.services.body_prompt_terms import (
    body_height_desc_compact,
    body_weight_desc_compact,
    normalize_body_height,
    normalize_body_weight,
    raw_height_from_req,
    raw_weight_from_req,
)

_COMPACT_PROMPT_LIMIT = 3000

_COMPACT_SANITIZE_PATTERNS = [
    re.compile(
        r'\b(wearing|dressed in|outfit|garment|clothing|cloth|clothes|fabric|'
        r'top|bottom|skirt|pants|trousers|shirt|dress|blouse|jacket|coat|suit|'
        r'saree|sari|kurta|lehenga|churidar|dupatta|salwar|kameez|gown|frock|'
        r'shorts|jeans|denim|sweater|hoodie|cardigan|vest|crop|bralette|'
        r'sleeve|collar|neckline|hem|waist|belt)\b', re.IGNORECASE),
    re.compile(r'\b(street|garden|park|beach|office|store|shop)\b', re.IGNORECASE),
    re.compile(r'\b(male|female|man|woman|boy|girl|he|she|his|her|they|them)\b', re.IGNORECASE),
]


_CURVE_FOCUS_ONE_PIECE_KEYWORDS = (
    "body shaper",
    "full body",
    "bodysuit shaper",
    "thermal innerwear set",
    "teddy lingerie",
    "swimsuit",
    "one-piece swimwear",
    "bikini set",
    "undergarments",
)


def _requires_curve_focus(req: dict) -> bool:
    gender = (req.get("gender") or "").strip().lower()
    weight = normalize_body_weight(raw_weight_from_req(req))
    one_piece_type = (req.get("one_piece_garment_type") or "").strip().lower()
    female_weight_rule = gender == "female" and weight in ("regular", "fat")
    one_piece_rule = any(k in one_piece_type for k in _CURVE_FOCUS_ONE_PIECE_KEYWORDS)
    return female_weight_rule or one_piece_rule


def _curve_focus_instruction(req: dict) -> str:
    if not _requires_curve_focus(req):
        return ""
    return (
        "Keep upper-torso and lower-pelvic contours fuller and naturally rounded per body weight; "
        "ensure transitions read like real skin and soft tissue, with believable garment tension and "
        "compression (never padded, inflated, or synthetic)."
    )


def _sanitize_pose_compact(pose: str) -> str:
    result = pose
    for pat in _COMPACT_SANITIZE_PATTERNS:
        result = pat.sub('', result)
    return re.sub(r'\s{2,}', ' ', result).strip()


def _build_compact_prompt(
    pose: str,
    has_back: bool,
    req: dict,
    *,
    has_mannequin_image: bool = False,
) -> str:
    """Build a <=3000-char SeedDream prompt.

    When ``has_mannequin_image`` is True the last reference image is a grey
    mannequin: the prompt requires a literal match to its posture, framing,
    seated/standing, and limb geometry (placed immediately after Refs for
    emphasis). SeedDream must ignore mannequin surface/clothes/bg only.
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
        img_ref += (
            f", IMG{mi}=POSE MANNEQUIN (exact posture, framing, seated/standing, limb geometry \u2014 "
            f"literal match; not loose inspiration)"
        )

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

    weight = normalize_body_weight(raw_weight_from_req(req))
    height = normalize_body_height(raw_height_from_req(req))
    w_desc = body_weight_desc_compact(weight)
    h_desc = body_height_desc_compact(height)

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

    pose_mannequin_block = (
        f"[P3 \u2014 POSE | EXACT COPY FROM IMG{mi} MANNEQUIN]\n"
        f"IMG{mi} is a grey featureless mannequin. It is ONLY a body-posture reference.\n"
        f"You MUST replicate the EXACT pose from IMG{mi} with ZERO deviation\u2014"
        f"same joint angles and asymmetry; no catalogue substitute, no symmetry clean-up, no prettier re-pose:\n"
        f"- FRAMING/CROP LOCK (CRITICAL): Match mannequin body coverage exactly. "
        f"If IMG{mi} shows only upper half, output only upper half. "
        f"If IMG{mi} shows only lower half, output only lower half. "
        f"If IMG{mi} shows full body, output full body. "
        f"Do NOT expand crop to full body when mannequin is half body. "
        f"Do NOT crop body when mannequin is full body.\n"
        f"- SEATED/SUPPORT LOCK (CRITICAL): If IMG{mi} is seated or uses chair/stool/bench, "
        f"output MUST stay seated with the same hip/knee angles and limb-to-seat/feet-on-rung contact; "
        f"NEVER convert seated mannequin to standing or tighter waist-up for style.\n"
        f"- CLOSE-UP DETAIL LOCK (CRITICAL): If IMG{mi} is a close-up/detail crop "
        f"(neck/shoulder/chest/waist/garment-detail framing), keep the same close-up camera distance "
        f"and same visible garment regions. Do NOT zoom out or invent out-of-frame body parts.\n"
        f"- HANDS: Exact placement as IMG{mi}\u2014pocket, hip, head/temple, thigh, clasped, or props; "
        f"exact finger curl, wrist angle, palm orientation. Do NOT invent a different hand placement.\n"
        f"- BODY/TORSO: Copy exact torso lean, shoulder tilt, chest orientation, spine curvature, "
        f"hip rotation, and weight shift. Do NOT straighten or alter the torso.\n"
        f"- Legs/feet: Only if legs or feet appear in IMG{mi}\u2014match stance and contact; "
        f"do not add or infer lower body for upper-only crops.\n"
        f"- FOOTWEAR (full body or feet visible in output): IGNORE mannequin bare feet or plain studio shoes\u2014"
        f"put believable footwear on the real model coordinated with P2 garment(s) (formality, style, color palette, occasion); "
        f"never leave bare feet when feet/ankles show.\n"
        f"- HEAD: Copy exact head tilt, turn direction, chin angle, and gaze direction.\n"
        f"- Resolve conflicts using the global PRIORITY block: never break P1 face or P2 garment to satisfy pose.\n"
        f"IGNORE everything else about IMG{mi}\u2014ignore its grey skin, featureless face, bald head, "
        f"white background, and clothing. Only replicate the body posture on the real human model.{hands_note}\n"
    )
    p3_pose_label = f"IMG{mi} mannequin" if has_mannequin_image else "text pose below"
    priority_block = (
        f"=====================================================================\n"
        f"PRIORITY ORDER (on conflict, higher P wins \u2014 never violate a higher P for a lower P):\n"
        f"P1 FACE (IMG{fi}) \u2014 HIGHEST: lock identity, expression, facial proportions; never change face for pose, garment, or bg.\n"
        f"P2 GARMENT ({g_imgs}) \u2014 SECOND: exact outfit from refs; never swap, recolor, or simplify garment to satisfy pose.\n"
        f"P3 POSE ({p3_pose_label}) \u2014 THIRD: match as exactly as possible without breaking P1 or P2.\n"
        f"P4 BACKGROUND (IMG{bi}) \u2014 FOURTH: natural blend and scene light; never replace or redraw background geometry.\n"
        f"P5 BODY (weight/height) \u2014 FIFTH: silhouette reflects weight/height only where consistent with P1\u2013P4.\n"
        f"=====================================================================\n\n"
    )
    part_head = (
        f"Hyperrealistic editorial fashion photograph\u2014real person, NOT illustration, NOT AI-looking, NOT plastic skin.\n"
        f"\n"
        f"Refs: {img_ref}\n"
        f"\n"
        f"{priority_block}"
    )
    part_face_skin = (
        f"[P1 \u2014 FACE | EXACT COPY IMG{fi}]\n"
        f"Lock all features: face shape, eyes, brows, nose, lips, jaw, chin, cheekbones, ears, hairline, "
        f"hair style/color/length, skin tone/undertone, moles/marks. No beautification. Expression identical to IMG{fi}.\n"
        f"\n"
        f"[SKIN\u2014IN-CAMERA REAL, NOT AIRBRUSH]\n"
        f"Readable micro-pores on nose/cheeks/forehead/chin; peach-fuzz; visible veins/tendons on hands/forearms; "
        f"SSS, tone variation, knuckle/crease detail. Skin lit ONLY by IMG{bi} light\u2014include micro-specular hits on "
        f"cheek/nose from that light. No plastic or poreless beauty skin.\n"
        f"\n"
    )
    part_garment = (
        f"[P2 \u2014 GARMENT | EXACT MATCH {g_imgs}]{gt_line}\n"
        f"Copy exact fabric, pattern, color, print, embroidery, buttons, pockets, stitching, hem, neckline, "
        f"sleeves, cuffs, weave, closures, trims, pleats, natural drape and wrinkles. "
        f"No color shift, no simplification. Fitting: {fitting}.{paired}\n"
        + garment_rules
        + f"\n"
    )
    part_bg = (
        f"[P4 \u2014 BACKGROUND + IN-SCENE SUBJECT | IMG{bi} ({bg_label})]\n"
        f"Subject is physically present in the same space as IMG{bi}\u2014naturally lit and blended, not pasted. "
        f"Use the same light field as IMG{bi} (key direction, WB, shadow softness); avoid flat frontal studio fill. "
        f"Ambient bounce from the environment onto visible skin and garment (and footwear only if feet are in frame). "
        f"Rim on hair/shoulders from scene sources; subtle light wrap on silhouette edges\u2014no razor cutout. "
        f"When the framing includes full body or visible legs/feet: ground feet on the floor with believable contact shadow "
        f"and cast matching the scene key; hem and soles follow scene perspective; footwear must read as chosen for the garment(s). "
        f"Match subject exposure, contrast, and saturation to the environment (not brighter/punchier than the room). "
        f"No floating, no halo. Do not alter background.\n"
        f"\n"
    )
    part_body = (
        f"[P5 \u2014 BODY | WEIGHT & HEIGHT]\n"
        f"{req.get('gender','')}, {req.get('ethnicity','')}, age {req.get('age','')} ({req.get('age_group','')}), "
        f"skin: {req.get('skin_tone','')}.\n"
        f"Weight: {w_desc}. Height: {h_desc}. Body must visibly reflect this build without overriding P1\u2013P4. "
        f"{_curve_focus_instruction(req)}\n"
        f"\n"
    )
    part_pose_text = f"[P3 \u2014 POSE | TEXT]\n{clean_pose}\n"
    part_footer = (
        f"Footwear: full body or feet visible\u2014REQUIRE shoes coordinated with the garment(s) (formality, style, colors, occasion); "
        f"ignore barefoot mannequin refs; no bare feet when feet show. Upper-body-only / hidden feet: shoes optional.{bag} Ornaments: {orn or 'none'}.\n"
        f"85mm f/2.8, shallow DOF on subject, 4K 9:16, editorial color; natural hair detail; one coherent in-camera exposure."
    )
    if has_mannequin_image:
        prompt = (
            part_head
            + part_face_skin
            + part_garment
            + pose_mannequin_block
            + part_bg
            + part_body
            + part_footer
        )
    else:
        prompt = (
            part_head
            + part_face_skin
            + part_garment
            + part_pose_text
            + part_bg
            + part_body
            + part_footer
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

    pose_mannequin_opt = (
        f"[P3|POSE IMG{mi}] Literal match: crop, joints, seated/standing, stool if present; no catalogue re-pose. "
        f"If feet in frame: garment-matched shoes\u2014not mannequin bare feet. "
        f"Ignore mannequin surface/clothes/bg.{hands_note}\n"
    )
    pose_text_opt = f"[P3|POSE TEXT] {pose}\n"

    p3_lbl = f"IMG{mi}" if has_mannequin_image else "text"
    priority_opt = (
        f"PRIORITY (conflict: P1>P2>P3>P4>P5): P1 FACE IMG{fi}; P2 GARMENT {g_imgs}; P3 POSE {p3_lbl}; "
        f"P4 BG IMG{bi}; P5 body wt/ht.\n"
    )
    mid_p1p2 = (
        f"[P1|FACE IMG{fi}] Same identity/expression; no beautification.\n"
        "[SKIN] Pores; veins/tendons arms/hands; peach fuzz; micro-specular from scene; no airbrush.\n"
        f"[P2|GARMENT {g_imgs}]{gt_line}\n"
        "Keep exact garment color/pattern/fabric/embroidery/details/stitching/fit/drape/wrinkles. "
        f"Fitting: {fitting}.{paired}\n"
        f"{garment_rules}"
        f"{onepiece_hint}\n"
    )
    mid_p4p5 = (
        f"[P4|BG IMG{bi} {bg_label}] In-scene light/blend; bounce; rim; no cutout; match exposure; no bg redraw. "
        f"Feet grounded when full body in frame; garment-matched shoes when feet visible.\n"
        f"[P5|BODY] {req.get('gender','')}, {req.get('ethnicity','')}, age {req.get('age','')} ({req.get('age_group','')}), "
        f"skin {req.get('skin_tone','')}; weight {w_desc}; height {h_desc}. "
        f"{_curve_focus_instruction(req)}\n"
    )
    tail = (
        f"[STYLE] Full body/feet visible: garment-matched footwear required; ignore mannequin bare feet; no bare feet.{bag} Ornaments: {orn or 'none'}.\n"
        "85mm f/2.8, shallow DOF, 4K 9:16, editorial color, natural hair; single coherent in-camera exposure."
    )
    intro = "Hyperrealistic editorial fashion photo (real DSLR look; no AI/plastic skin).\n"
    refs = f"Refs: {img_ref}\n"
    if has_mannequin_image:
        return intro + refs + priority_opt + mid_p1p2 + pose_mannequin_opt + mid_p4p5 + tail
    return intro + refs + priority_opt + mid_p1p2 + pose_text_opt + mid_p4p5 + tail
