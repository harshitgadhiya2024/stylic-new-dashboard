#!/usr/bin/env python3
"""
Face-to-Model Generator
------------------------
Input  : A public image URL containing a human face.
Process:
  1. Gemini validates the image — checks if a clear human face is present.
     If not → exits with a descriptive error.
  2. SeedDream 4.5-edit generates a professional fashion model portrait
     that looks exactly like the person in the input image.
Output : 2K, 9:16 portrait saved locally + result URL printed.

Usage:
  python face_to_model.py

Set INPUT_IMAGE_URL at the top before running.

Requirements:
    pip install requests google-genai Pillow
"""

import json
import os
import sys
import time
import datetime

try:
    import requests
except ImportError:
    print("ERROR: pip install requests"); sys.exit(1)

try:
    from google import genai
    from google.genai import types as gtypes
except ImportError:
    print("ERROR: pip install google-genai"); sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# ── Input ─────────────────────────────────────────────────────────────────────
# Public URL of the image containing the face you want to replicate
INPUT_IMAGE_URL = "https://facialrecognition.app/_next/image?url=%2Fimages%2Fcarlos.webp&w=640&q=75"

# ── Gemini — face validation ──────────────────────────────────────────────────
GEMINI_API_KEY   = "AIzaSyBxlCynVylmuVuMnzezVc9HNGtrgjtI5K8"
GEMINI_MODEL     = "gemini-2.5-flash"   # vision-capable, fast, cost-effective

# ── SeedDream / kie.ai ────────────────────────────────────────────────────────
SEEDDREAM_API_KEY    = "aee9b2af00e40357bceaa62011da5879"
SEEDDREAM_CREATE_URL = "https://api.kie.ai/api/v1/jobs/createTask"
SEEDDREAM_STATUS_URL = "https://api.kie.ai/api/v1/jobs/recordInfo"
SEEDDREAM_MODEL      = "seedream/4.5-edit"
SEEDDREAM_QUALITY    = "medium"       # "low" / "medium" / "high"
SEEDDREAM_ASPECT     = "9:16"       # portrait
SEEDDREAM_MAX_RETRIES = 120
SEEDDREAM_RETRY_DELAY = 5           # seconds between polls

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "face_model_output"

# =============================================================================


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# =============================================================================
# Step 1 — Gemini: validate face presence and extract basic description
# =============================================================================

def validate_and_describe_face(image_url: str) -> dict:
    """
    Uses Gemini vision to:
      - Check whether the image contains a clear, visible human face.
      - If yes: return a brief description of the person's appearance.
      - If no: raise ValueError with a user-friendly message.

    Returns dict:
      {
        "has_face": True,
        "description": "25-year-old Indian woman, oval face, dark hair..."
      }
    """
    log("[1/3] Validating image with Gemini...")

    # Download image bytes so we can pass them inline (works for any URL)
    img_resp = requests.get(image_url, timeout=30)
    img_resp.raise_for_status()
    img_bytes = img_resp.content

    # Detect mime type from Content-Type header, fall back to jpeg
    content_type = img_resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
    if content_type not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
        content_type = "image/jpeg"

    validation_prompt = (
        "You are an image validation assistant for a fashion photoshoot platform. "
        "Check whether this image contains a clear, usable human face.\n\n"
        "Respond with ONLY valid JSON (no markdown fences):\n"
        "{\n"
        '  "has_face": true or false,\n'
        '  "reason": "<if has_face is false: explain why — e.g. no face, cartoon, '
        'heavily obscured, animal, scenery, etc. Leave empty string if has_face is true.>",\n'
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

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
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

    raw = response.candidates[0].content.parts[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"Gemini returned unexpected response: {raw}")

    if not result.get("has_face"):
        reason = result.get("reason", "No valid human face detected in the image.")
        raise ValueError(f"Input image does not have a valid face: {reason}")

    log(f"  Face detected ✓")
    log(f"  Description: {result.get('description', '')[:120]}")
    return result


# =============================================================================
# Step 2 — Build SeedDream generation prompt
# =============================================================================

def build_generation_prompt(face_description: str, image_url: str) -> str:
    """
    Build a prompt that instructs SeedDream to generate a professional
    fashion model portrait that is an exact likeness of the reference face.
    """
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


# =============================================================================
# Step 3 — SeedDream: submit task and poll
# =============================================================================

def submit_seeddream_task(prompt: str, image_url: str) -> str:
    log("[2/3] Submitting task to SeedDream...")

    payload = json.dumps({
        "model": SEEDDREAM_MODEL,
        "input": {
            "prompt":       prompt,
            "image_urls":   [image_url],
            "aspect_ratio": SEEDDREAM_ASPECT,
            "quality":      SEEDDREAM_QUALITY,
        }
    })
    headers = {
        "Authorization": f"Bearer {SEEDDREAM_API_KEY}",
        "Content-Type":  "application/json",
    }

    resp = requests.post(SEEDDREAM_CREATE_URL, headers=headers, data=payload, timeout=30)
    resp.raise_for_status()

    data    = resp.json()
    task_id = data.get("data", {}).get("taskId")
    if not task_id:
        raise ValueError(f"No taskId returned: {resp.text}")

    log(f"  Task submitted ✓ — ID: {task_id}")
    return task_id


def poll_seeddream_task(task_id: str) -> str:
    log("  Polling for result...")

    headers = {"Authorization": f"Bearer {SEEDDREAM_API_KEY}"}

    for attempt in range(1, SEEDDREAM_MAX_RETRIES + 1):
        try:
            resp = requests.get(
                f"{SEEDDREAM_STATUS_URL}?taskId={task_id}",
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            data   = resp.json().get("data", {})
            status = data.get("state")

            if status == "success":
                result_urls = json.loads(data.get("resultJson", "{}")).get("resultUrls", [])
                if result_urls:
                    log(f"  Done ✓ — result URL: {result_urls[0][:80]}...")
                    return result_urls[0]
                raise ValueError("Task succeeded but no resultUrls found.")

            elif status == "fail":
                raise RuntimeError("SeedDream task failed.")

            else:
                log(f"  Poll #{attempt:03d} — status: {status}")
                time.sleep(SEEDDREAM_RETRY_DELAY)

        except (RuntimeError, ValueError):
            raise
        except Exception as exc:
            log(f"  Poll #{attempt} error: {exc}")
            time.sleep(SEEDDREAM_RETRY_DELAY)

    raise TimeoutError(f"SeedDream task did not complete after {SEEDDREAM_MAX_RETRIES} attempts.")


# =============================================================================
# Step 4 — Download result
# =============================================================================

def download_result(result_url: str) -> str:
    log("[3/3] Downloading result image...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"model_face_{stamp}.png")

    for attempt in range(1, 4):
        try:
            resp = requests.get(result_url, stream=True, timeout=(10, 60))
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(65536):
                    if chunk:
                        f.write(chunk)
            log(f"  Saved → {os.path.abspath(out_path)}")
            return out_path
        except Exception as exc:
            log(f"  Download attempt {attempt}/3 failed: {exc}")
            if attempt == 3:
                raise
            time.sleep(3)


# =============================================================================
# Main
# =============================================================================

def generate_model_face(image_url: str) -> str:
    """
    Full pipeline: validate → generate → download.
    Returns local path of the saved output image.
    Raises ValueError if no valid face is found in the input image.
    """
    print("\n" + "=" * 55)
    print("  Face-to-Model Generator")
    print("=" * 55)
    log(f"Input URL: {image_url[:80]}{'...' if len(image_url) > 80 else ''}")

    # Step 1 — validate
    face_info   = validate_and_describe_face(image_url)
    description = face_info.get("description", "")

    # Step 2 — build prompt + submit
    prompt  = build_generation_prompt(description, image_url)
    task_id = submit_seeddream_task(prompt, image_url)

    # Step 3 — poll
    result_url = poll_seeddream_task(task_id)

    # Step 4 — download
    out_path = download_result(result_url)

    print("\n" + "=" * 55)
    print("  COMPLETE")
    print(f"  Result URL  : {result_url}")
    print(f"  Saved to    : {os.path.abspath(out_path)}")
    print("=" * 55 + "\n")

    return out_path


if __name__ == "__main__":
    try:
        generate_model_face(INPUT_IMAGE_URL)
    except ValueError as e:
        print(f"\n❌ ERROR: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        sys.exit(1)
