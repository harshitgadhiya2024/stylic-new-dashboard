"""
Upload images from all_backgrounds/ to S3 and build a Mongo-style JSON catalog.

For each image:
  1) Upload to a temporary staging key → public URL
  2) Call Gemini 2.5 Flash (vision) with that URL + image bytes → background_type
     (indoor | outdoor | studio | others) and background_name (snake_case)
  3) Copy object to backgrounds/{background_name}_{uuid8}.ext, delete staging

Output: JSON array like testing.txt (background_id, $date fields, etc.).

Resume: state file skips finished rows; unfinished staging rows retry Gemini + finalize.

Usage:
    python scripts/upload_local_backgrounds_to_s3.py
    python scripts/upload_local_backgrounds_to_s3.py --dry-run
    python scripts/upload_local_backgrounds_to_s3.py --skip-gemini   # filename-based names, empty type

.env: AWS_*, GEMINI_API_KEY, optional GEMINI_VISION_MODEL (default gemini-2.5-flash)
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif"}

VALID_TYPES = frozenset({"indoor", "outdoor", "studio", "others"})

DEFAULT_FOLDER = ROOT / "all_backgrounds"
DEFAULT_OUTPUT = ROOT / "background_catalog_from_upload.json"
DEFAULT_STATE = ROOT / "upload_local_backgrounds_state.json"

GEMINI_CLASSIFY_PROMPT = """You analyze images used as FASHION PHOTOSHOOT BACKGROUNDS (environment / setting only).

The same image is available at this public URL (reference only; analyze the attached image):
{background_url}

Return ONLY one JSON object. No markdown fences, no code blocks, no extra text.
Use exactly this shape:
{{"background_type":"...","background_name":"..."}}

Rules for background_type — MUST be exactly one of these four lowercase words:
- studio — photography studio, cyclorama, seamless paper, plain studio backdrop
- indoor — interior location that is not a studio (home, hotel, office, retail interior, hallway)
- outdoor — exterior (street, garden, beach, skyline, natural outdoor light on location)
- others — if unclear, mixed, or none of the above fit

Rules for background_name:
- snake_case only: lowercase letters, digits, underscores
- 2 to 6 underscore-separated tokens describing mood + setting (e.g. abstract_blur_lounge, marble_floor_natural_light, white_cyclorama_studio)
- no people, no camera jargon; environment description only
"""


def _env(name: str) -> str:
    import os

    v = os.getenv(name, "")
    if not v:
        sys.exit(f"Missing required environment variable: {name}")
    return v


def _optional_env(name: str, default: str) -> str:
    import os

    return os.getenv(name, default) or default


def slugify_stem(stem: str) -> str:
    s = unicodedata.normalize("NFKD", stem)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", "_", s.lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "background"


def iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def discover_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        sys.exit(f"Not a directory: {folder}")
    paths: list[Path] = []
    for p in sorted(folder.rglob("*"), key=lambda x: str(x).lower()):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        paths.append(p)
    return paths


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _s3_client():
    import boto3

    return boto3.client(
        "s3",
        aws_access_key_id=_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=_env("AWS_SECRET_ACCESS_KEY"),
        region_name=_optional_env("AWS_REGION", "us-east-1"),
    )


def public_url(key: str) -> str:
    bucket = _env("AWS_S3_BUCKET_NAME")
    region = _optional_env("AWS_REGION", "us-east-1")
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


def s3_put(body: bytes, key: str, content_type: str) -> None:
    from botocore.exceptions import ClientError

    client = _s3_client()
    try:
        client.put_object(
            Bucket=_env("AWS_S3_BUCKET_NAME"),
            Key=key,
            Body=body,
            ContentType=content_type,
        )
    except ClientError as exc:
        raise RuntimeError(f"S3 upload failed: {exc}") from exc


def s3_copy_then_delete(src_key: str, dst_key: str) -> None:
    from botocore.exceptions import ClientError

    bucket = _env("AWS_S3_BUCKET_NAME")
    client = _s3_client()
    try:
        client.copy_object(
            Bucket=bucket,
            Key=dst_key,
            CopySource={"Bucket": bucket, "Key": src_key},
        )
        client.delete_object(Bucket=bucket, Key=src_key)
    except ClientError as exc:
        raise RuntimeError(f"S3 copy/delete failed: {exc}") from exc


def download_url(url: str) -> tuple[bytes, str]:
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
    ct = r.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    if ct not in ("image/jpeg", "image/png", "image/webp"):
        ct = "image/jpeg"
    return r.content, ct


def parse_json_object(text: str) -> dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object in model output: {t[:200]}")
    return json.loads(t[start : end + 1])


def classify_background_gemini(image_bytes: bytes, mime_type: str, background_url: str) -> tuple[str, str]:
    from google import genai
    from google.genai import types as gtypes

    api_key = _env("GEMINI_API_KEY")
    model = _optional_env("GEMINI_VISION_MODEL", "gemini-2.5-flash")
    prompt = GEMINI_CLASSIFY_PROMPT.format(background_url=background_url)
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[
            gtypes.Content(
                role="user",
                parts=[
                    gtypes.Part.from_bytes(mime_type=mime_type, data=image_bytes),
                    gtypes.Part.from_text(text=prompt),
                ],
            )
        ],
        config=gtypes.GenerateContentConfig(
            response_modalities=["TEXT"],
            temperature=0.2,
        ),
    )
    raw = ""
    for part in response.candidates[0].content.parts:
        if getattr(part, "text", None):
            raw = part.text
            break
    if not raw:
        raise RuntimeError("Gemini returned no text")
    data = parse_json_object(raw)
    btype = str(data.get("background_type", "others")).strip().lower()
    if btype not in VALID_TYPES:
        btype = "others"
    name = str(data.get("background_name", "background")).strip().lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_") or "background"
    return btype, name


def unique_background_name(base: str, used: set[str]) -> str:
    if base not in used:
        return base
    n = 2
    while f"{base}_{n}" in used:
        n += 1
    return f"{base}_{n}"


def build_record(
    *,
    background_id: str,
    background_type: str,
    background_name: str,
    background_url: str,
    now: datetime,
) -> dict:
    ts = {"$date": iso_z(now)}
    return {
        "background_id": background_id,
        "background_type": background_type,
        "background_name": background_name,
        "background_url": background_url,
        "count": 0,
        "tags": [],
        "notes": "",
        "is_default": True,
        "is_active": True,
        "created_at": ts,
        "updated_at": ts,
    }


def resolve_content_type(path: Path, ext: str) -> str:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if content_type not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
        if ext == ".png":
            content_type = "image/png"
        elif ext in (".jpg", ".jpeg"):
            content_type = "image/jpeg"
        elif ext == ".webp":
            content_type = "image/webp"
    return content_type


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload backgrounds to S3 + Gemini labels → catalog JSON")
    ap.add_argument("--folder", type=Path, default=DEFAULT_FOLDER, help="Directory of images")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON array path")
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE, help="Resume cache")
    ap.add_argument("--prefix", default="backgrounds", help="S3 key prefix (no trailing slash)")
    ap.add_argument(
        "--skip-gemini",
        action="store_true",
        help="Do not call Gemini; use filename slug for name and empty background_type",
    )
    ap.add_argument("--dry-run", action="store_true", help="No S3/Gemini/state; preview only")
    args = ap.parse_args()

    out_path = args.output
    if args.dry_run:
        out_path = args.output.with_name(f"{args.output.stem}.dry_run{args.output.suffix}")

    images = discover_images(args.folder)
    if not images:
        sys.exit(f"No images found under {args.folder}")

    state = load_state(args.state)
    rel_key = lambda p: p.relative_to(args.folder).as_posix()

    used_names: set[str] = set()
    for path in images:
        rk0 = rel_key(path)
        rec0 = state.get(rk0, {}).get("record")
        if rec0 and rec0.get("background_name"):
            used_names.add(rec0["background_name"])

    catalog: list[dict] = []
    now = datetime.now(timezone.utc)
    prefix = args.prefix.strip("/")

    for path in images:
        rk = rel_key(path)
        entry = state.get(rk, {})
        if entry.get("record") and entry["record"].get("background_url"):
            catalog.append(entry["record"])
            continue

        pending = (entry.get("pending") or {}) if isinstance(entry.get("pending"), dict) else {}
        ext = path.suffix.lower()
        if ext == ".jpeg":
            ext = ".jpg"
        content_type = resolve_content_type(path, ext)
        data = path.read_bytes()

        if pending.get("staging_key"):
            background_id = pending["background_id"]
            staging_key = pending["staging_key"]
            ext = pending.get("ext", ext)
            content_type = pending.get("content_type", content_type)
        else:
            background_id = str(uuid.uuid4())
            staging_key = f"{prefix}/_staging/{background_id}{ext}"

        short = background_id.split("-", 1)[0]

        if args.dry_run:
            fake_url = public_url(staging_key)
            bg_type = "" if args.skip_gemini else "studio"
            base = slugify_stem(path.stem) if args.skip_gemini else "ai_named_background"
            bg_name = unique_background_name(base, used_names)
            used_names.add(bg_name)
            print(f"[dry-run] {rk} -> gemini_url={fake_url} name={bg_name} type={bg_type}")
            catalog.append(
                build_record(
                    background_id=background_id,
                    background_type=bg_type,
                    background_name=bg_name,
                    background_url=f"(dry-run) {prefix}/{bg_name}_{short}{ext}",
                    now=now,
                )
            )
            continue

        if not pending.get("staging_key"):
            s3_put(data, staging_key, content_type)
            state[rk] = {
                "pending": {
                    "staging_key": staging_key,
                    "background_id": background_id,
                    "ext": ext,
                    "content_type": content_type,
                }
            }
            save_state(args.state, state)

        if args.skip_gemini:
            base = slugify_stem(path.stem)
            bg_name = unique_background_name(base, used_names)
            used_names.add(bg_name)
            bg_type = ""
        else:
            staging_url = public_url(staging_key)
            img_for_gemini, mime_gemini = download_url(staging_url)
            try:
                bg_type, ai_name = classify_background_gemini(img_for_gemini, mime_gemini, staging_url)
            except Exception as exc:
                print(f"[warn] Gemini failed for {rk}, fallback to filename: {exc}", file=sys.stderr)
                bg_type = "others"
                ai_name = slugify_stem(path.stem)
            bg_name = unique_background_name(ai_name, used_names)
            used_names.add(bg_name)

        final_key = f"{prefix}/{bg_name}_{short}{ext}"
        s3_copy_then_delete(staging_key, final_key)
        final_url = public_url(final_key)

        record = build_record(
            background_id=background_id,
            background_type=bg_type,
            background_name=bg_name,
            background_url=final_url,
            now=now,
        )
        catalog.append(record)
        state[rk] = {"record": record}
        save_state(args.state, state)
        print(final_url, file=sys.stderr)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(catalog)} records to {out_path}", file=sys.stderr)
    if args.dry_run:
        print("(dry-run: preview only; run without --dry-run to upload and write final catalog)", file=sys.stderr)


if __name__ == "__main__":
    main()
