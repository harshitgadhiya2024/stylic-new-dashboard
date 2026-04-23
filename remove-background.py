import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import fal_client
import requests


# Only one input variable as requested.
input_image = "https://pub-51c3a7dccc2448f792c2fb1bacf8e05d.r2.dev/photoshoots/c5f1861f-b3be-429f-8fe0-3a1464ee9f56/df01987d-933c-4e8b-9551-a7a360b22dee_8k_upscaled.png"


# Keep keys in environment variables:
# export KIE_API_KEY="..."
# export FAL_KEY="..."
KIE_API_KEY = "aee9b2af00e40357bceaa62011dasds5879"
FAL_KEY = "3abacfe4-8d89-4b32-a708-7db8ba52ff2a:0d772bfe0d704ee28e87a13d8d35aacc"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FAL_MODEL_ID = "fal-ai/birefnet/v2"
FAL_MODEL_PARAMS = {
    "model": "General Use (Heavy)",
    "operating_resolution": "2048x2048",
    "output_format": "png",
    "refine_foreground": True,
}


def _download_image(url: str) -> Path:
    # Some provider URLs (especially fal.media) may briefly return 404
    # right after job completion, or require auth headers.
    download_headers = [{}]
    if (FAL_KEY or "").strip():
        download_headers.append(_fal_auth_header())
    last_error = None
    response = None
    for attempt in range(1, 6):
        for headers in download_headers:
            try:
                response = requests.get(url, headers=headers, timeout=60)
                response.raise_for_status()
                last_error = None
                break
            except Exception as error:
                last_error = error
        if last_error is None and response is not None:
            break
        time.sleep(1.5)

    if last_error is not None or response is None:
        raise RuntimeError(f"Failed to download output image from {url}") from last_error

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    file_path = OUTPUT_DIR / f"bg_removed_{timestamp}_{unique_id}.png"
    file_path.write_bytes(response.content)
    return file_path


def _extract_output_url(data: dict) -> str:
    # Handles common output shapes from Kie/Fal responses.
    if not isinstance(data, dict):
        return ""

    # Direct keys at root level.
    for key in ("url", "image_url"):
        value = data.get(key)
        if isinstance(value, str) and value.startswith("http"):
            return value

    image_obj = data.get("image")
    if isinstance(image_obj, str) and image_obj.startswith("http"):
        return image_obj
    if isinstance(image_obj, dict):
        value = image_obj.get("url") or image_obj.get("image_url")
        if isinstance(value, str) and value.startswith("http"):
            return value

    candidates = [
        data.get("output"),
        data.get("result"),
        data.get("data"),
    ]

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.startswith("http"):
            return candidate
        if isinstance(candidate, dict):
            for key in ("url", "image", "image_url"):
                value = candidate.get(key)
                if isinstance(value, str) and value.startswith("http"):
                    return value
            images = candidate.get("images")
            if isinstance(images, list):
                for item in images:
                    if isinstance(item, str) and item.startswith("http"):
                        return item
                    if isinstance(item, dict):
                        value = item.get("url") or item.get("image") or item.get("image_url")
                        if isinstance(value, str) and value.startswith("http"):
                            return value
        if isinstance(candidate, list):
            for item in candidate:
                if isinstance(item, str) and item.startswith("http"):
                    return item
                if isinstance(item, dict):
                    value = item.get("url") or item.get("image") or item.get("image_url")
                    if isinstance(value, str) and value.startswith("http"):
                        return value

    return ""


def _fal_auth_header() -> dict[str, str]:
    key = (FAL_KEY or "").strip()
    if not key:
        raise RuntimeError("FAL_KEY is missing.")

    # Accept either raw token or already-prefixed token.
    lowered = key.lower()
    if lowered.startswith("key "):
        auth_value = key
    elif lowered.startswith("bearer "):
        auth_value = key
    else:
        auth_value = f"Key {key}"
    return {"Authorization": auth_value}


def _queue_callback(update):
    status = getattr(update, "status", None)
    if status == "IN_PROGRESS":
        logs = getattr(update, "logs", []) or []
        for log in logs:
            msg = log.get("message", "") if isinstance(log, dict) else str(log)
            if msg:
                print(f"  [fal] {msg}")


def run_kie_remove_bg(image_url: str) -> str:
    if not KIE_API_KEY:
        raise RuntimeError("KIE_API_KEY is missing.")

    create_url = "https://api.kie.ai/api/v1/jobs/createTask"
    result_url = "https://api.kie.ai/api/v1/jobs/recordInfo"
    headers = {
        "Authorization": f"Bearer {KIE_API_KEY}",
        "Content-Type": "application/json",
    }

    create_payload = {
        "model": "recraft/remove-background",
        "input": {"image": image_url},
    }
    create_response = requests.post(create_url, json=create_payload, headers=headers, timeout=60)
    create_response.raise_for_status()
    create_json = create_response.json()

    task_id = (
        create_json.get("data", {}).get("taskId")
        or create_json.get("data", {}).get("id")
        or create_json.get("taskId")
        or create_json.get("id")
    )
    if not task_id:
        raise RuntimeError(f"Kie createTask did not return task id: {create_json}")

    # Poll task result (KIE uses recordInfo with GET + taskId).
    for _ in range(30):
        poll_response = requests.get(
            result_url,
            headers={"Authorization": headers["Authorization"]},
            params={"taskId": task_id},
            timeout=60,
        )
        poll_response.raise_for_status()
        poll_json = poll_response.json()

        data = poll_json.get("data") or {}
        state = str(data.get("state") or "").lower()

        if state == "success":
            raw = data.get("resultJson")
            parsed = raw
            if isinstance(raw, str):
                try:
                    import json

                    parsed = json.loads(raw)
                except Exception:
                    parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}

            urls = parsed.get("resultUrls") or parsed.get("result_urls") or []
            if urls and isinstance(urls[0], str) and urls[0].startswith("http"):
                return urls[0]

            # Fallback parsing in case schema changes.
            output_url = _extract_output_url(parsed) or _extract_output_url(data) or _extract_output_url(poll_json)
            if output_url:
                return output_url

            raise RuntimeError(f"Kie task succeeded but output URL not found: {poll_json}")

        if state in {"fail", "failed", "error"}:
            fail_msg = data.get("failMsg") or data.get("fail_msg") or "Unknown Kie failure"
            fail_code = data.get("failCode") or data.get("fail_code")
            raise RuntimeError(f"Kie task failed: {fail_msg} (code={fail_code})")

        time.sleep(2)

    raise TimeoutError("Kie polling timed out.")


def run_fal_remove_bg(image_url: str) -> str:
    key = (FAL_KEY or "").strip()
    if not key:
        raise RuntimeError("FAL_KEY is missing.")

    client = fal_client.SyncClient(key=key)
    result = client.subscribe(
        FAL_MODEL_ID,
        arguments={"image_url": image_url, **FAL_MODEL_PARAMS},
        with_logs=True,
        on_queue_update=_queue_callback,
    )
    output_url = _extract_output_url(result)
    if output_url:
        return output_url
    raise RuntimeError(f"Fal completed but no output URL found: {result}")


def with_retries(label: str, attempts: int, func, *args):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"[{label}] attempt {attempt}/{attempts}")
            return func(*args)
        except Exception as error:
            last_error = error
            print(f"[{label}] failed attempt {attempt}: {error}")
            time.sleep(1)
    raise RuntimeError(f"{label} failed after {attempts} attempts.") from last_error


def main():
    try:
        # First provider: Kie, retry 2 times.
        output_url = with_retries("kie.ai", 2, run_kie_remove_bg, input_image)
        print(f"[kie.ai] success: {output_url}")
    except Exception as kie_error:
        print(f"[kie.ai] exhausted retries: {kie_error}")
        # Fallback provider: Fal, retry 2 times.
        output_url = with_retries("fal.ai", 2, run_fal_remove_bg, input_image)
        print(f"[fal.ai] success: {output_url}")

    saved_path = _download_image(output_url)
    print(f"Downloaded output image to: {saved_path}")


if __name__ == "__main__":
    main()

