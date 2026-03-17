#!/usr/bin/env python3
"""
Model Face Generator
---------------------
Generates 32 realistic portrait faces for fashion photoshoot models:
  - 8 adult male
  - 8 adult female
  - 8 kid girl  (age 8-12)
  - 8 kid boy   (age 8-12)

Each face is:
  1. Generated via Gemini gemini-2.5-flash-image
  2. Saved locally under ./model_faces/
  3. Uploaded to S3
  4. Recorded in model_faces/face_catalog.json

JSON entry format:
  {
    "face_id": "<uuid>",
    "face_url": "<s3 public url>",
    "is_default": true/false,
    "configuration": { ...face config dict... },
    "is_active": true,
    "created_at": "<ISO datetime>"
  }

Requirements:
    pip install google-genai boto3 Pillow
"""

import asyncio
import datetime
import io
import json
import os
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    print("ERROR: pip install boto3"); sys.exit(1)

try:
    from google import genai
    from google.genai import types as gtypes
except ImportError:
    print("ERROR: pip install google-genai"); sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("ERROR: pip install Pillow"); sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-image")

AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
S3_BUCKET             = os.getenv("AWS_S3_BUCKET_NAME", "")
S3_REGION             = os.getenv("AWS_REGION", "eu-north-1")
S3_PREFIX             = "model-faces"
S3_PUBLIC_ACL         = False   # set True if your bucket ACL allows it

OUTPUT_DIR    = "model_faces"
CATALOG_FILE  = os.path.join(OUTPUT_DIR, "face_catalog.json")

# Max concurrent Gemini calls (keep low to avoid rate limits)
CONCURRENCY   = 4

# =============================================================================
# FACE CONFIGURATIONS — 32 diverse, fashion-photoshoot-ready profiles
# =============================================================================

# ── Adult Female (8) ──────────────────────────────────────────────────────────
ADULT_FEMALE = [
    {
        "face_shape": "oval", "jawline_type": "soft", "cheekbone_height": "high",
        "face_skin_tone": "light", "skin_undertone": "neutral",
        "hair_color": "dark_black", "hair_length": "long", "hair_style": "straight",
        "eye_shape": "almond", "eye_color": "dark_brown",
        "nose_shape": "straight", "lip_shape": "full", "eyebrow_shape": "soft_arch",
        "beard_length": "none", "beard_color": "none",
        "age": "24 years", "ethnicity": "Indian", "gender": "female",
    },
    {
        "face_shape": "heart", "jawline_type": "delicate", "cheekbone_height": "high",
        "face_skin_tone": "medium", "skin_undertone": "warm",
        "hair_color": "dark_brown", "hair_length": "shoulder_length", "hair_style": "soft_waves",
        "eye_shape": "round", "eye_color": "hazel",
        "nose_shape": "button", "lip_shape": "medium", "eyebrow_shape": "arched",
        "beard_length": "none", "beard_color": "none",
        "age": "26 years", "ethnicity": "Indian", "gender": "female",
    },
    {
        "face_shape": "square", "jawline_type": "defined", "cheekbone_height": "medium",
        "face_skin_tone": "dusky", "skin_undertone": "cool",
        "hair_color": "black", "hair_length": "short", "hair_style": "pixie_cut",
        "eye_shape": "hooded", "eye_color": "dark_brown",
        "nose_shape": "wide", "lip_shape": "full", "eyebrow_shape": "straight",
        "beard_length": "none", "beard_color": "none",
        "age": "28 years", "ethnicity": "Indian", "gender": "female",
    },
    {
        "face_shape": "round", "jawline_type": "soft", "cheekbone_height": "medium",
        "face_skin_tone": "fair", "skin_undertone": "pink",
        "hair_color": "light_brown", "hair_length": "medium", "hair_style": "loose_curls",
        "eye_shape": "wide", "eye_color": "light_brown",
        "nose_shape": "snub", "lip_shape": "thin", "eyebrow_shape": "natural",
        "beard_length": "none", "beard_color": "none",
        "age": "22 years", "ethnicity": "Indian", "gender": "female",
    },
    {
        "face_shape": "diamond", "jawline_type": "sharp", "cheekbone_height": "very_high",
        "face_skin_tone": "medium", "skin_undertone": "olive",
        "hair_color": "dark_black", "hair_length": "long", "hair_style": "braided",
        "eye_shape": "cat_eye", "eye_color": "black",
        "nose_shape": "aquiline", "lip_shape": "full", "eyebrow_shape": "bold_arch",
        "beard_length": "none", "beard_color": "none",
        "age": "30 years", "ethnicity": "Indian", "gender": "female",
    },
    {
        "face_shape": "oblong", "jawline_type": "oval", "cheekbone_height": "low",
        "face_skin_tone": "tan", "skin_undertone": "warm",
        "hair_color": "auburn", "hair_length": "medium", "hair_style": "wavy",
        "eye_shape": "almond", "eye_color": "amber",
        "nose_shape": "straight", "lip_shape": "medium", "eyebrow_shape": "feathered",
        "beard_length": "none", "beard_color": "none",
        "age": "27 years", "ethnicity": "Indian", "gender": "female",
    },
    {
        "face_shape": "oval", "jawline_type": "pointed", "cheekbone_height": "high",
        "face_skin_tone": "light_medium", "skin_undertone": "neutral",
        "hair_color": "jet_black", "hair_length": "very_long", "hair_style": "sleek",
        "eye_shape": "upturned", "eye_color": "dark_brown",
        "nose_shape": "button", "lip_shape": "full", "eyebrow_shape": "micro_bladed",
        "beard_length": "none", "beard_color": "none",
        "age": "23 years", "ethnicity": "Indian", "gender": "female",
    },
    {
        "face_shape": "heart", "jawline_type": "soft", "cheekbone_height": "high",
        "face_skin_tone": "medium_dark", "skin_undertone": "warm",
        "hair_color": "dark_brown", "hair_length": "bob", "hair_style": "straight_bob",
        "eye_shape": "almond", "eye_color": "dark_brown",
        "nose_shape": "flat", "lip_shape": "full", "eyebrow_shape": "thick_natural",
        "beard_length": "none", "beard_color": "none",
        "age": "25 years", "ethnicity": "Indian", "gender": "female",
    },
]

# ── Adult Male (8) — 4 with beard, 4 clean-shave ─────────────────────────────
ADULT_MALE = [
    # --- With beard (4) ---
    {
        "face_shape": "square", "jawline_type": "sharp", "cheekbone_height": "high",
        "face_skin_tone": "medium", "skin_undertone": "warm",
        "hair_color": "dark_black", "hair_length": "short", "hair_style": "textured_crop",
        "eye_shape": "almond", "eye_color": "dark_brown",
        "nose_shape": "straight", "lip_shape": "medium", "eyebrow_shape": "thick",
        "beard_length": "stubble", "beard_color": "black",
        "age": "26 years", "ethnicity": "Indian", "gender": "male",
    },
    {
        "face_shape": "oblong", "jawline_type": "angular", "cheekbone_height": "high",
        "face_skin_tone": "tan", "skin_undertone": "olive",
        "hair_color": "dark_brown", "hair_length": "short", "hair_style": "undercut",
        "eye_shape": "hooded", "eye_color": "hazel",
        "nose_shape": "wide", "lip_shape": "full", "eyebrow_shape": "straight",
        "beard_length": "short_beard", "beard_color": "black",
        "age": "28 years", "ethnicity": "Indian", "gender": "male",
    },
    {
        "face_shape": "heart", "jawline_type": "pointed", "cheekbone_height": "high",
        "face_skin_tone": "light", "skin_undertone": "cool",
        "hair_color": "dark_brown", "hair_length": "medium", "hair_style": "side_part",
        "eye_shape": "upturned", "eye_color": "dark_brown",
        "nose_shape": "button", "lip_shape": "thin", "eyebrow_shape": "soft_arch",
        "beard_length": "full_beard", "beard_color": "brown",
        "age": "30 years", "ethnicity": "Indian", "gender": "male",
    },
    {
        "face_shape": "oval", "jawline_type": "lean", "cheekbone_height": "medium",
        "face_skin_tone": "tan", "skin_undertone": "warm",
        "hair_color": "dark_black", "hair_length": "medium", "hair_style": "wavy",
        "eye_shape": "deep_set", "eye_color": "amber",
        "nose_shape": "straight", "lip_shape": "medium", "eyebrow_shape": "natural",
        "beard_length": "goatee", "beard_color": "dark_brown",
        "age": "32 years", "ethnicity": "Indian", "gender": "male",
    },
    # --- Clean shave (4) ---
    {
        "face_shape": "round", "jawline_type": "soft", "cheekbone_height": "medium",
        "face_skin_tone": "fair", "skin_undertone": "neutral",
        "hair_color": "black", "hair_length": "medium", "hair_style": "curly",
        "eye_shape": "round", "eye_color": "light_brown",
        "nose_shape": "snub", "lip_shape": "medium", "eyebrow_shape": "natural",
        "beard_length": "clean_shave", "beard_color": "none",
        "age": "24 years", "ethnicity": "Indian", "gender": "male",
    },
    {
        "face_shape": "diamond", "jawline_type": "sharp", "cheekbone_height": "very_high",
        "face_skin_tone": "medium", "skin_undertone": "warm",
        "hair_color": "jet_black", "hair_length": "short", "hair_style": "buzz_cut",
        "eye_shape": "almond", "eye_color": "dark_brown",
        "nose_shape": "straight", "lip_shape": "full", "eyebrow_shape": "bold",
        "beard_length": "clean_shave", "beard_color": "none",
        "age": "27 years", "ethnicity": "Indian", "gender": "male",
    },
    {
        "face_shape": "square", "jawline_type": "strong", "cheekbone_height": "high",
        "face_skin_tone": "medium_dark", "skin_undertone": "warm",
        "hair_color": "black", "hair_length": "short", "hair_style": "fade",
        "eye_shape": "almond", "eye_color": "black",
        "nose_shape": "broad", "lip_shape": "full", "eyebrow_shape": "thick_straight",
        "beard_length": "clean_shave", "beard_color": "none",
        "age": "29 years", "ethnicity": "Indian", "gender": "male",
    },
    {
        "face_shape": "oval", "jawline_type": "defined", "cheekbone_height": "medium",
        "face_skin_tone": "dusky", "skin_undertone": "cool",
        "hair_color": "black", "hair_length": "medium", "hair_style": "slicked_back",
        "eye_shape": "deep_set", "eye_color": "black",
        "nose_shape": "aquiline", "lip_shape": "thin", "eyebrow_shape": "arched",
        "beard_length": "clean_shave", "beard_color": "none",
        "age": "25 years", "ethnicity": "Indian", "gender": "male",
    },
]

# ── Kid Girl (8) — age 8-12 ───────────────────────────────────────────────────
KID_GIRL = [
    {
        "face_shape": "round", "jawline_type": "soft", "cheekbone_height": "medium",
        "face_skin_tone": "fair", "skin_undertone": "warm",
        "hair_color": "dark_black", "hair_length": "long", "hair_style": "pigtails",
        "eye_shape": "wide", "eye_color": "dark_brown",
        "nose_shape": "button", "lip_shape": "thin", "eyebrow_shape": "natural",
        "beard_length": "none", "beard_color": "none",
        "age": "9 years", "ethnicity": "Indian", "gender": "girl",
    },
    {
        "face_shape": "oval", "jawline_type": "soft", "cheekbone_height": "low",
        "face_skin_tone": "medium", "skin_undertone": "warm",
        "hair_color": "dark_brown", "hair_length": "shoulder_length", "hair_style": "loose_curls",
        "eye_shape": "round", "eye_color": "hazel",
        "nose_shape": "snub", "lip_shape": "thin", "eyebrow_shape": "natural",
        "beard_length": "none", "beard_color": "none",
        "age": "10 years", "ethnicity": "Indian", "gender": "girl",
    },
    {
        "face_shape": "heart", "jawline_type": "delicate", "cheekbone_height": "high",
        "face_skin_tone": "light", "skin_undertone": "cool",
        "hair_color": "black", "hair_length": "medium", "hair_style": "ponytail",
        "eye_shape": "almond", "eye_color": "light_brown",
        "nose_shape": "button", "lip_shape": "thin", "eyebrow_shape": "soft_arch",
        "beard_length": "none", "beard_color": "none",
        "age": "11 years", "ethnicity": "Indian", "gender": "girl",
    },
    {
        "face_shape": "round", "jawline_type": "soft", "cheekbone_height": "medium",
        "face_skin_tone": "dusky", "skin_undertone": "warm",
        "hair_color": "jet_black", "hair_length": "long", "hair_style": "braided",
        "eye_shape": "wide", "eye_color": "dark_brown",
        "nose_shape": "flat", "lip_shape": "full", "eyebrow_shape": "thick_natural",
        "beard_length": "none", "beard_color": "none",
        "age": "8 years", "ethnicity": "Indian", "gender": "girl",
    },
    {
        "face_shape": "oval", "jawline_type": "delicate", "cheekbone_height": "medium",
        "face_skin_tone": "medium", "skin_undertone": "neutral",
        "hair_color": "dark_black", "hair_length": "short", "hair_style": "bob",
        "eye_shape": "round", "eye_color": "black",
        "nose_shape": "button", "lip_shape": "thin", "eyebrow_shape": "natural",
        "beard_length": "none", "beard_color": "none",
        "age": "12 years", "ethnicity": "Indian", "gender": "girl",
    },
    {
        "face_shape": "heart", "jawline_type": "soft", "cheekbone_height": "high",
        "face_skin_tone": "tan", "skin_undertone": "olive",
        "hair_color": "brown", "hair_length": "medium", "hair_style": "wavy",
        "eye_shape": "almond", "eye_color": "amber",
        "nose_shape": "snub", "lip_shape": "thin", "eyebrow_shape": "arched",
        "beard_length": "none", "beard_color": "none",
        "age": "10 years", "ethnicity": "Indian", "gender": "girl",
    },
    {
        "face_shape": "round", "jawline_type": "soft", "cheekbone_height": "low",
        "face_skin_tone": "fair", "skin_undertone": "pink",
        "hair_color": "dark_brown", "hair_length": "long", "hair_style": "straight",
        "eye_shape": "wide", "eye_color": "dark_brown",
        "nose_shape": "button", "lip_shape": "thin", "eyebrow_shape": "natural",
        "beard_length": "none", "beard_color": "none",
        "age": "9 years", "ethnicity": "Indian", "gender": "girl",
    },
    {
        "face_shape": "oval", "jawline_type": "soft", "cheekbone_height": "medium",
        "face_skin_tone": "light_medium", "skin_undertone": "warm",
        "hair_color": "black", "hair_length": "shoulder_length", "hair_style": "half_up",
        "eye_shape": "round", "eye_color": "light_brown",
        "nose_shape": "straight", "lip_shape": "thin", "eyebrow_shape": "feathered",
        "beard_length": "none", "beard_color": "none",
        "age": "11 years", "ethnicity": "Indian", "gender": "girl",
    },
]

# ── Kid Boy (8) — age 8-12 ────────────────────────────────────────────────────
KID_BOY = [
    {
        "face_shape": "round", "jawline_type": "soft", "cheekbone_height": "medium",
        "face_skin_tone": "medium", "skin_undertone": "warm",
        "hair_color": "dark_black", "hair_length": "short", "hair_style": "textured",
        "eye_shape": "round", "eye_color": "dark_brown",
        "nose_shape": "button", "lip_shape": "thin", "eyebrow_shape": "natural",
        "beard_length": "none", "beard_color": "none",
        "age": "9 years", "ethnicity": "Indian", "gender": "boy",
    },
    {
        "face_shape": "oval", "jawline_type": "soft", "cheekbone_height": "low",
        "face_skin_tone": "fair", "skin_undertone": "neutral",
        "hair_color": "black", "hair_length": "short", "hair_style": "side_part",
        "eye_shape": "almond", "eye_color": "hazel",
        "nose_shape": "snub", "lip_shape": "thin", "eyebrow_shape": "natural",
        "beard_length": "none", "beard_color": "none",
        "age": "10 years", "ethnicity": "Indian", "gender": "boy",
    },
    {
        "face_shape": "square", "jawline_type": "defined", "cheekbone_height": "medium",
        "face_skin_tone": "dusky", "skin_undertone": "warm",
        "hair_color": "jet_black", "hair_length": "short", "hair_style": "buzz_cut",
        "eye_shape": "wide", "eye_color": "dark_brown",
        "nose_shape": "straight", "lip_shape": "medium", "eyebrow_shape": "thick",
        "beard_length": "none", "beard_color": "none",
        "age": "11 years", "ethnicity": "Indian", "gender": "boy",
    },
    {
        "face_shape": "heart", "jawline_type": "soft", "cheekbone_height": "high",
        "face_skin_tone": "light", "skin_undertone": "cool",
        "hair_color": "dark_brown", "hair_length": "short", "hair_style": "curly",
        "eye_shape": "round", "eye_color": "light_brown",
        "nose_shape": "button", "lip_shape": "thin", "eyebrow_shape": "arched",
        "beard_length": "none", "beard_color": "none",
        "age": "8 years", "ethnicity": "Indian", "gender": "boy",
    },
    {
        "face_shape": "oval", "jawline_type": "soft", "cheekbone_height": "medium",
        "face_skin_tone": "tan", "skin_undertone": "olive",
        "hair_color": "black", "hair_length": "medium", "hair_style": "shaggy",
        "eye_shape": "almond", "eye_color": "amber",
        "nose_shape": "flat", "lip_shape": "thin", "eyebrow_shape": "straight",
        "beard_length": "none", "beard_color": "none",
        "age": "12 years", "ethnicity": "Indian", "gender": "boy",
    },
    {
        "face_shape": "round", "jawline_type": "soft", "cheekbone_height": "medium",
        "face_skin_tone": "medium", "skin_undertone": "warm",
        "hair_color": "dark_black", "hair_length": "short", "hair_style": "crop",
        "eye_shape": "wide", "eye_color": "black",
        "nose_shape": "broad", "lip_shape": "medium", "eyebrow_shape": "natural",
        "beard_length": "none", "beard_color": "none",
        "age": "10 years", "ethnicity": "Indian", "gender": "boy",
    },
    {
        "face_shape": "oblong", "jawline_type": "angular", "cheekbone_height": "high",
        "face_skin_tone": "fair", "skin_undertone": "pink",
        "hair_color": "brown", "hair_length": "short", "hair_style": "side_swept",
        "eye_shape": "deep_set", "eye_color": "light_brown",
        "nose_shape": "straight", "lip_shape": "thin", "eyebrow_shape": "natural",
        "beard_length": "none", "beard_color": "none",
        "age": "11 years", "ethnicity": "Indian", "gender": "boy",
    },
    {
        "face_shape": "heart", "jawline_type": "delicate", "cheekbone_height": "high",
        "face_skin_tone": "medium_dark", "skin_undertone": "warm",
        "hair_color": "jet_black", "hair_length": "short", "hair_style": "messy",
        "eye_shape": "round", "eye_color": "dark_brown",
        "nose_shape": "snub", "lip_shape": "thin", "eyebrow_shape": "thick_natural",
        "beard_length": "none", "beard_color": "none",
        "age": "9 years", "ethnicity": "Indian", "gender": "boy",
    },
]

# Combine all with category labels and default flags
ALL_FACES: list[dict] = (
    [{"category": "adult_female", "is_default": i == 0, "config": c}
     for i, c in enumerate(ADULT_FEMALE)] +
    [{"category": "adult_male",   "is_default": i == 0, "config": c}
     for i, c in enumerate(ADULT_MALE)] +
    [{"category": "kid_girl",     "is_default": i == 0, "config": c}
     for i, c in enumerate(KID_GIRL)] +
    [{"category": "kid_boy",      "is_default": i == 0, "config": c}
     for i, c in enumerate(KID_BOY)]
)


# =============================================================================
# Prompt builder (from face_generation.py — extended for kids/beard handling)
# =============================================================================

def build_face_prompt(config: dict) -> str:
    gender  = config["gender"]
    age     = config["age"]
    is_kid  = gender in ("girl", "boy")
    is_male = gender in ("male", "boy")

    beard_section = ""
    if is_male and config.get("beard_length") not in ("none", "clean_shave", "", None):
        beard_section = f"""
Beard:
- Beard length: {config['beard_length'].replace('_', ' ')}
- Beard color: {config['beard_color'].replace('_', ' ')}
"""

    kid_note = (
        "This is a child model. Generate an age-appropriate, innocent, natural portrait. "
        "No adult styling or makeup.\n\n"
        if is_kid else ""
    )

    return f"""{kid_note}Generate a highly realistic, professional portrait photograph of a {age} old {config['ethnicity']} {gender}.

Face & Facial Structure:
- Face shape: {config['face_shape']}
- Jawline: {config['jawline_type']}
- Cheekbone height: {config['cheekbone_height']}

Skin:
- Skin tone: {config['face_skin_tone']}
- Skin undertone: {config['skin_undertone']}

Hair:
- Hair color: {config['hair_color'].replace('_', ' ')}
- Hair length: {config['hair_length'].replace('_', ' ')}
- Hair style: {config['hair_style'].replace('_', ' ')}

Eyes:
- Eye shape: {config['eye_shape']}
- Eye color: {config['eye_color'].replace('_', ' ')}

Facial Features:
- Nose shape: {config['nose_shape']}
- Lip shape: {config['lip_shape']}
- Eyebrow shape: {config['eyebrow_shape'].replace('_', ' ')}
{beard_section}
Photography Style:
- Clean, plain white background
- Studio portrait lighting (soft, even, professional)
- Upper body portrait: face, neck, and both bare shoulders clearly visible, framed from roughly mid-chest upward
- Arms cut off at the sides of the frame, shoulders naturally sloping into the edges
- Front-facing pose, looking directly at the camera
- Natural, warm smile
- High resolution, photorealistic quality
- Professional fashion model headshot composition

Do not add any text, watermarks, or overlays on the image."""


# =============================================================================
# S3 helpers
# =============================================================================

def _s3_client():
    return boto3.client(
        "s3",
        region_name=S3_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def upload_to_s3(local_path: str, s3_key: str) -> str:
    """Upload a file to S3 and return its public HTTPS URL."""
    with open(local_path, "rb") as f:
        data = f.read()

    put_kwargs = {
        "Bucket":      S3_BUCKET,
        "Key":         s3_key,
        "Body":        data,
        "ContentType": "image/png",
    }
    if S3_PUBLIC_ACL:
        put_kwargs["ACL"] = "public-read"

    _s3_client().put_object(**put_kwargs)
    url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
    return url


# =============================================================================
# Generation helpers
# =============================================================================

def _gemini_generate_face(config: dict) -> bytes | None:
    """Call Gemini synchronously and return raw PNG bytes, or None on failure."""
    client  = genai.Client(api_key=GEMINI_API_KEY)
    prompt  = build_face_prompt(config)

    contents = [
        gtypes.Content(
            role="user",
            parts=[gtypes.Part.from_text(text=prompt)],
        )
    ]
    cfg = gtypes.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
        image_config=gtypes.ImageConfig(
            aspect_ratio="9:16"
        ),
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=cfg,
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                # Convert to PNG via Pillow to normalise format
                img = Image.open(io.BytesIO(part.inline_data.data))
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                return buf.getvalue()
    except Exception as exc:
        print(f"    [gemini error] {exc}", flush=True)

    return None


async def generate_and_upload(
    entry: dict,
    semaphore: asyncio.Semaphore,
    results: list,
    index: int,
) -> None:
    async with semaphore:
        category   = entry["category"]
        config     = entry["config"]
        is_default = entry["is_default"]
        face_id    = str(uuid.uuid4())
        gender_tag = config["gender"]
        age_tag    = config["age"].replace(" ", "_")

        print(f"  [{index+1:02d}/32] Generating {category} — {gender_tag}, {age_tag}...", flush=True)

        # Run blocking Gemini call in a thread so it doesn't block the event loop
        loop      = asyncio.get_event_loop()
        img_bytes = await loop.run_in_executor(None, _gemini_generate_face, config)

        if not img_bytes:
            print(f"    [{index+1:02d}] FAILED — skipping.", flush=True)
            return

        # Save locally
        filename   = f"{category}_{face_id[:8]}.png"
        local_path = os.path.join(OUTPUT_DIR, filename)
        with open(local_path, "wb") as f:
            f.write(img_bytes)
        print(f"    [{index+1:02d}] Saved locally → {local_path}", flush=True)

        # Upload to S3
        s3_key = f"{S3_PREFIX}/{filename}"
        try:
            face_url = await loop.run_in_executor(None, upload_to_s3, local_path, s3_key)
            print(f"    [{index+1:02d}] Uploaded → {face_url}", flush=True)
        except Exception as exc:
            print(f"    [{index+1:02d}] S3 upload failed: {exc} — using local path.", flush=True)
            face_url = local_path

        record = {
            "face_id":       face_id,
            "category":      category,
            "face_url":      face_url,
            "local_path":    local_path,
            "is_default":    is_default,
            "configuration": config,
            "is_active":     True,
            "created_at":    datetime.datetime.utcnow().isoformat() + "Z",
        }
        results.append(record)
        print(f"    [{index+1:02d}] Done ✓", flush=True)


# =============================================================================
# Main
# =============================================================================

async def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Model Face Generator — {len(ALL_FACES)} faces")
    print(f"  Output dir : {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Concurrency: {CONCURRENCY}")
    print(f"{'='*60}\n")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    results: list = []

    tasks = [
        generate_and_upload(entry, semaphore, results, i)
        for i, entry in enumerate(ALL_FACES)
    ]
    await asyncio.gather(*tasks)

    # Sort by category + created_at for a clean catalog
    results.sort(key=lambda r: (r["category"], r["created_at"]))

    # Strip internal local_path from final JSON (keep it clean)
    catalog = []
    for r in results:
        catalog.append({
            "face_id":       r["face_id"],
            "category":      r["category"],
            "face_url":      r["face_url"],
            "is_default":    r["is_default"],
            "configuration": r["configuration"],
            "is_active":     r["is_active"],
            "created_at":    r["created_at"],
        })

    with open(CATALOG_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  COMPLETE")
    print(f"  Generated : {len(catalog)} / {len(ALL_FACES)} faces")
    print(f"  Catalog   : {os.path.abspath(CATALOG_FILE)}")
    print(f"{'='*60}")

    # Per-category summary
    from collections import Counter
    cats = Counter(r["category"] for r in catalog)
    for cat, count in sorted(cats.items()):
        print(f"    {cat:<20} {count} faces")


if __name__ == "__main__":
    asyncio.run(main())
