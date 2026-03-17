"""
AI face generation service using Google Gemini.
Generates a portrait image from face configuration and uploads it to S3.
"""

import io
import uuid

from fastapi import HTTPException, status

from app.config import settings
from app.services.s3_service import upload_bytes_to_s3

# ── Default configuration values ──────────────────────────────────────────
_DEFAULTS_COMMON = {
    "face_shape":       "oval",
    "jawline_type":     "soft",
    "cheekbone_height": "medium",
    "face_skin_tone":   "medium",
    "skin_undertone":   "warm",
    "hair_color":       "dark_black",
    "hair_length":      "medium",
    "hair_style":       "straight",
    "eye_shape":        "almond",
    "eye_color":        "dark_brown",
    "nose_shape":       "straight",
    "lip_shape":        "medium",
    "eyebrow_shape":    "natural",
    "age":              "25 years",
    "ethnicity":        "Indian",
}

# Beard only for adult_male — all other categories force none
_BEARD_DEFAULTS = {
    "beard_length": "none",
    "beard_color":  "none",
}

_MALE_CATEGORIES = {"adult_male"}


def build_configuration(category: str, overrides: dict) -> dict:
    """
    Merge caller-supplied overrides on top of category-appropriate defaults.
    beard_length / beard_color are stripped and forced to 'none' for non-male categories.
    """
    # Determine gender from category
    if "female" in category or "girl" in category:
        gender = "female" if "adult" in category else "girl"
    elif "boy" in category:
        gender = "boy"
    else:
        gender = "male"

    config = {**_DEFAULTS_COMMON, "gender": gender}

    # Beard support only for adult_male
    if category in _MALE_CATEGORIES:
        config.update(_BEARD_DEFAULTS)
        if overrides.get("beard_length"):
            config["beard_length"] = overrides["beard_length"]
        if overrides.get("beard_color"):
            config["beard_color"] = overrides["beard_color"]
    else:
        config["beard_length"] = "none"
        config["beard_color"]  = "none"

    # Apply all other overrides (skip beard keys — handled above)
    for key, value in overrides.items():
        if key in ("beard_length", "beard_color"):
            continue
        if value is not None:
            config[key] = value

    return config


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


def generate_face_image(config: dict) -> bytes:
    """Call Gemini to generate a portrait image and return raw PNG bytes."""
    try:
        from google import genai
        from google.genai import types as gtypes
        from PIL import Image
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing dependency for AI generation: {exc}. Run: pip install google-genai Pillow",
        )

    client  = genai.Client(api_key=settings.GEMINI_API_KEY)
    prompt  = build_face_prompt(config)

    contents = [
        gtypes.Content(
            role="user",
            parts=[gtypes.Part.from_text(text=prompt)],
        )
    ]
    cfg = gtypes.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
        image_config=gtypes.ImageConfig(aspect_ratio="9:16"),
    )

    try:
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=contents,
            config=cfg,
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                img = Image.open(io.BytesIO(part.inline_data.data))
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                return buf.getvalue()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini image generation failed: {exc}",
        )

    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Gemini returned no image data.",
    )


def generate_and_upload_face(category: str, overrides: dict) -> tuple[str, dict]:
    """
    Full pipeline: build config → generate image → upload to S3.
    Returns (face_url, final_configuration).
    """
    config   = build_configuration(category, overrides)
    img_bytes = generate_face_image(config)

    face_id  = str(uuid.uuid4())
    s3_key   = f"model-faces/{category}_{face_id[:8]}.png"
    face_url = upload_bytes_to_s3(img_bytes, s3_key, content_type="image/png")

    return face_url, config
