"""
Body weight / height phrases for SeedDream prompts.

Supported weight: regular (default), fat, slim.
Supported height: regular (default ~150 cm), short, tall.

Also accepts ``body_weight`` / ``body_height`` on the request dict when present.
"""

from __future__ import annotations

from typing import Any, Mapping


def raw_weight_from_req(req: Mapping[str, Any]) -> Any:
    bw = req.get("body_weight")
    if bw is not None and str(bw).strip() != "":
        return bw
    return req.get("weight")


def raw_height_from_req(req: Mapping[str, Any]) -> Any:
    bh = req.get("body_height")
    if bh is not None and str(bh).strip() != "":
        return bh
    return req.get("height")


def normalize_body_weight(raw: object) -> str:
    """Map to regular | fat | slim. Unknown → regular."""
    if raw is None:
        return "regular"
    w = str(raw).strip().lower()
    if w in ("fat", "fatty", "heavy", "plus", "plus-size", "chubby", "overweight"):
        return "fat"
    if w in ("slim", "thin", "lean", "skinny"):
        return "slim"
    if w == "regular":
        return "regular"
    return "regular"


def normalize_body_height(raw: object) -> str:
    """Map to regular | short | tall. Unknown → regular."""
    if raw is None:
        return "regular"
    h = str(raw).strip().lower()
    if h in ("short", "petite"):
        return "short"
    if h in ("tall", "long"):
        return "tall"
    if h == "regular":
        return "regular"
    return "regular"


def body_weight_instruction_long(weight: str) -> str:
    if weight == "fat":
        return (
            "FATTY / HEAVY BUILD (non-negotiable): visibly larger body mass — "
            "broader torso, thicker upper arms and thighs, fuller waist/abdomen, "
            "wider hips. The silhouette must read as genuinely heavy, not "
            "\"slightly curvy\" or standard fashion-model thin."
        )
    if weight == "slim":
        return (
            "SLIM / LEAN BUILD: narrow shoulders, defined waist, slender limbs. "
            "Visibly leaner than an average model."
        )
    return (
        "REGULAR WEIGHT + MUSCULAR TONE: athletic muscular build (defined shoulders, "
        "chest, arms) with healthy body fat proportional to frame — not skeletal, not heavy."
    )


def body_height_instruction_long(height: str) -> str:
    if height == "short":
        return (
            "SHORT STATURE: noticeably below average adult height — compact limbs and torso "
            "relative to the frame."
        )
    if height == "tall":
        return (
            "TALL STATURE: above-average height — longer limbs and torso proportionate to "
            "a taller frame."
        )
    return (
        "REGULAR HEIGHT (~150 CM): adult stature around 150 cm — typical short-to-average "
        "standing height; scale limb and torso proportions accordingly."
    )


def body_weight_desc_compact(weight: str) -> str:
    if weight == "fat":
        return (
            "fatty/heavy build: fuller softer torso thick limbs wide midsection, "
            "genuinely plus-size silhouette"
        )
    if weight == "slim":
        return "slim/lean: narrow shoulders, defined waist, slender limbs"
    return (
        "regular weight + muscular: athletic definition shoulders/chest/arms, "
        "healthy proportions not heavy not skeletal"
    )


def body_height_desc_compact(height: str) -> str:
    if height == "short":
        return "short: below-average adult height, compact proportions"
    if height == "tall":
        return "tall: above-average height, elongated limbs proportionate to frame"
    return (
        "regular height ~150 cm: short-to-average adult stature, scale proportions to that frame"
    )
