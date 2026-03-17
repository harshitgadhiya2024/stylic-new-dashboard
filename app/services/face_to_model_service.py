"""
Face-to-Model service.

Pipeline:
  1. Validate the input image URL with Gemini vision — checks for a clear human face.
  2. Build a generation prompt using the face description.
  3. Submit a SeedDream 4.5-edit task and poll until complete.
  4. Download the generated image bytes.
  5. Upload to S3 and return the public URL.
"""

import io
import json
import time
import uuid

import requests
from fastapi import HTTPException, status

from app.config import settings
from app.services.s3_service import upload_bytes_to_s3


# ---------------------------------------------------------------------------
# Step 1 — Gemini: validate face presence
# ---------------------------------------------------------------------------

def validate_face(image_url: str) -> str:
    """
    Downloads the image and asks Gemini whether it contains a clear human face.
    Returns a brief appearance description on success.
    Raises HTTPException 422 if no valid face is detected.
    """
    try:
        from google import genai
        from google.genai import types as gtypes
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing dependency: {exc}. Run: pip install google-genai",
        )

    try:
        img_resp = requests.get(image_url, timeout=30)
        img_resp.raise_for_status()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not download the provided face_url: {exc}",
        )

    img_bytes = img_resp.content
    content_type = img_resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
    if content_type not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        content_type = "image/jpeg"

    validation_prompt = (
        "You are an image validation assistant for a fashion photoshoot platform. "
        "Check whether this image contains a clear, usable human face.\n\n"
        "Respond with ONLY valid JSON (no markdown fences):\n"
        "{\n"
        '  "has_face": true or false,\n'
        '  "reason": "<if has_face is false: explain why. Leave empty string if has_face is true.>",\n'
        '  "description": "<if has_face is true: concise description of the person\'s '
        "visible appearance — estimated age, gender, ethnicity, skin tone, face shape, "
        "hair color and style, eye color, any distinctive features. "
        'Leave empty string if has_face is false.>"\n'
        "}\n\n"
        "has_face must be false if:\n"
        "- No human face is visible\n"
        "- Face is heavily blurred, masked, or occluded\n"
        "- Image is a cartoon, illustration, or AI art without a real face\n"
        "- Image contains only animals, objects, or scenery\n"
        "- Face is turned away and features are not visible"
    )

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    try:
        response = client.models.generate_content(
            model=settings.GEMINI_VISION_MODEL,
            contents=[
                gtypes.Content(
                    role="user",
                    parts=[
                        gtypes.Part.from_bytes(mime_type=content_type, data=img_bytes),
                        gtypes.Part.from_text(text=validation_prompt),
                    ],
                )
            ],
            config=gtypes.GenerateContentConfig(
                response_modalities=["TEXT"],
                temperature=0,
            ),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini face validation failed: {exc}",
        )

    raw = response.candidates[0].content.parts[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini returned unexpected response: {raw}",
        )

    if not result.get("has_face"):
        reason = result.get("reason", "No valid human face detected in the image.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Input image does not contain a valid face: {reason}",
        )

    return result.get("description", "")


# ---------------------------------------------------------------------------
# Step 2 — Build SeedDream generation prompt
# ---------------------------------------------------------------------------

def _build_generation_prompt(face_description: str) -> str:
    return f"""You are provided with ONE reference image:
  IMAGE 1 — FACE REFERENCE: the exact face and appearance of the person to replicate.

Generate a highly realistic, professional fashion model portrait photograph.

[FACE — CRITICAL: DO NOT CHANGE]
Replicate the EXACT face from the reference image with 100% accuracy:
- Face shape, bone structure, and proportions
- Skin tone and undertone
- Eye shape, color, and expression
- Nose shape and size
- Lip shape and fullness
- Eyebrow shape and thickness
- Hair color, texture, and style
- Any distinctive features (dimples, moles, freckles, etc.)

Reference description for cross-check: {face_description}

FACE VALIDATION: Before finalising, verify the generated face matches the reference exactly. If any feature deviates — regenerate.

[PHOTOGRAPHY STYLE]
- Clean, plain white or very light neutral background
- Professional studio portrait lighting (soft, even, flattering)
- Upper body portrait: face, neck, and both bare shoulders clearly visible
- Framed from roughly mid-chest upward
- Front-facing pose, looking directly at the camera
- Natural, warm, confident expression
- High resolution, photorealistic quality
- Professional fashion model headshot / beauty portrait composition

[QUALITY]
- Ultra-high resolution (2K), photorealistic
- Professional fashion photography lighting with soft shadows
- Sharp focus on face
- Commercial model photography grade

Do not add any text, watermarks, or overlays.
NON-NEGOTIABLE: The generated face must be 100% identical to the reference face image."""


# ---------------------------------------------------------------------------
# Step 3 — SeedDream: submit task and poll
# ---------------------------------------------------------------------------

_CREATE_URL = "https://api.kie.ai/api/v1/jobs/createTask"
_STATUS_URL  = "https://api.kie.ai/api/v1/jobs/recordInfo"


def _submit_task(prompt: str, image_url: str) -> str:
    payload = json.dumps({
        "model": settings.SEEDDREAM_MODEL,
        "input": {
            "prompt":       prompt,
            "image_urls":   [image_url],
            "aspect_ratio": settings.SEEDDREAM_ASPECT,
            "quality":      settings.SEEDDREAM_QUALITY,
        },
    })
    headers = {
        "Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}",
        "Content-Type":  "application/json",
    }

    try:
        resp = requests.post(_CREATE_URL, headers=headers, data=payload, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"SeedDream task submission failed: {exc}",
        )
    print(resp.json())
    task_id = resp.json().get("data", {}).get("taskId")
    if not task_id:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"SeedDream returned no taskId: {resp.text}",
        )
    return task_id


def _poll_task(task_id: str) -> str:
    headers = {"Authorization": f"Bearer {settings.SEEDDREAM_API_KEY}"}

    for attempt in range(1, settings.SEEDDREAM_MAX_RETRIES + 1):
        try:
            resp = requests.get(
                f"{_STATUS_URL}?taskId={task_id}",
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            data   = resp.json().get("data", {})
            state  = data.get("state")

            if state == "success":
                result_urls = json.loads(data.get("resultJson", "{}")).get("resultUrls", [])
                if result_urls:
                    return result_urls[0]
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="SeedDream task succeeded but returned no result URLs.",
                )

            if state == "fail":
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="SeedDream image generation task failed.",
                )

            time.sleep(settings.SEEDDREAM_RETRY_DELAY)

        except HTTPException:
            raise
        except Exception:
            time.sleep(settings.SEEDDREAM_RETRY_DELAY)

    raise HTTPException(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        detail=f"SeedDream task did not complete after {settings.SEEDDREAM_MAX_RETRIES} attempts.",
    )


# ---------------------------------------------------------------------------
# Step 4 — Download generated image bytes
# ---------------------------------------------------------------------------

def _download_image(result_url: str) -> bytes:
    for attempt in range(1, 4):
        try:
            resp = requests.get(result_url, stream=True, timeout=(10, 60))
            resp.raise_for_status()
            buf = io.BytesIO()
            for chunk in resp.iter_content(65536):
                if chunk:
                    buf.write(chunk)
            return buf.getvalue()
        except Exception as exc:
            if attempt == 3:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Failed to download generated image: {exc}",
                )
            time.sleep(3)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_model_face_from_reference(image_url: str, model_category: str) -> str:
    """
    Full pipeline:
      1. Validate face in image_url via Gemini.
      2. Generate a model portrait via SeedDream using the reference face.
      3. Upload the result to S3.

    Returns the public S3 URL of the generated model face image.
    Raises HTTPException on any failure.
    """
    # Step 1 — validate
    face_description = validate_face(image_url)

    # Step 2 — generate
    prompt  = _build_generation_prompt(face_description)
    task_id = _submit_task(prompt, image_url)

    # Step 3 — poll
    result_url = _poll_task(task_id)

    # Step 4 — download + upload to S3
    img_bytes = _download_image(result_url)
    face_id   = str(uuid.uuid4())
    s3_key    = f"model-faces/{model_category}_{face_id[:8]}.png"
    s3_url    = upload_bytes_to_s3(img_bytes, s3_key, content_type="image/png")

    return s3_url
