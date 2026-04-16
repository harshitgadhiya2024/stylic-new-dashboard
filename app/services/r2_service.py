"""
Cloudflare R2 object storage (S3-compatible API via aioboto3).

See cloudflare-r2-guide.md for bucket setup, API tokens, and R2_PUBLIC_URL
(custom domain or pub-*.r2.dev).
"""

import os
import uuid
from urllib.parse import unquote, urlparse

import aioboto3
from botocore.exceptions import ClientError
from fastapi import HTTPException, UploadFile, status

from app.config import settings


def _r2_endpoint() -> str:
    explicit = (getattr(settings, "R2_ENDPOINT_URL", None) or "").strip()
    if explicit:
        return explicit.rstrip("/")
    account = settings.R2_ACCOUNT_ID.strip()
    if not account:
        raise ValueError("R2_ACCOUNT_ID (or R2_ENDPOINT_URL) is required for Cloudflare R2")
    return f"https://{account}.r2.cloudflarestorage.com"


def _public_object_url(key: str) -> str:
    base = settings.R2_PUBLIC_URL.strip().rstrip("/")
    if not base:
        raise ValueError("R2_PUBLIC_URL is required (custom domain or r2.dev public URL)")
    return f"{base}/{key}"


def public_url_to_object_key(public_url: str) -> str:
    """
    Strip the configured public origin from a full URL and return the R2 object key.

    Raises ValueError if the URL is not under ``R2_PUBLIC_URL`` or the key is unsafe.
    """
    base = settings.R2_PUBLIC_URL.strip().rstrip("/")
    if not base:
        raise ValueError("R2_PUBLIC_URL is not configured")

    raw = (public_url or "").strip()
    if "?" in raw:
        raw = raw.split("?", 1)[0]
    if "#" in raw:
        raw = raw.split("#", 1)[0]

    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("public_url must be a valid http(s) URL")

    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
    base_parsed = urlparse(base)
    base_norm = f"{base_parsed.scheme}://{base_parsed.netloc}{base_parsed.path}".rstrip("/")

    if not normalized.startswith(base_norm + "/") and normalized != base_norm:
        raise ValueError("URL does not match configured R2_PUBLIC_URL")

    prefix = base_norm + "/"
    if not normalized.startswith(prefix):
        raise ValueError("Missing object path under public URL base")

    suffix = normalized[len(prefix) :]
    if not suffix:
        raise ValueError("No object key in URL (path is empty)")
    if ".." in suffix or suffix.startswith("//"):
        raise ValueError("Invalid object key")

    return unquote(suffix)


async def delete_object_by_key(key: str) -> None:
    """Delete one object from R2 by key. No error if the object does not exist (S3-compatible behavior)."""
    try:
        async with _session.client(
            "s3",
            endpoint_url=_r2_endpoint(),
            region_name="auto",
        ) as client:
            await client.delete_object(Bucket=settings.R2_BUCKET_NAME, Key=key)
    except ClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"R2 delete failed: {exc.response['Error']['Message']}",
        ) from exc


_session = aioboto3.Session(
    aws_access_key_id=settings.R2_ACCESS_KEY_ID,
    aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
    region_name="auto",
)


async def upload_file_to_r2(file: UploadFile, folder: str = "uploads") -> str:
    ext = ""
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()

    key = f"{folder}/{uuid.uuid4().hex}{ext}"

    try:
        contents = await file.read()
        async with _session.client(
            "s3",
            endpoint_url=_r2_endpoint(),
            region_name="auto",
        ) as client:
            await client.put_object(
                Bucket=settings.R2_BUCKET_NAME,
                Key=key,
                Body=contents,
                ContentType=file.content_type or "application/octet-stream",
            )
    except ClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"R2 upload failed: {exc.response['Error']['Message']}",
        ) from exc

    return _public_object_url(key)


async def upload_bytes_to_r2(data: bytes, key: str, content_type: str = "image/png") -> str:
    """Upload raw bytes to R2 and return the public URL."""
    try:
        async with _session.client(
            "s3",
            endpoint_url=_r2_endpoint(),
            region_name="auto",
        ) as client:
            await client.put_object(
                Bucket=settings.R2_BUCKET_NAME,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
    except ClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"R2 upload failed: {exc.response['Error']['Message']}",
        ) from exc

    return _public_object_url(key)


# Historical names (same functions) — prefer upload_*_to_r2 in new code.
upload_file_to_s3 = upload_file_to_r2
upload_bytes_to_s3 = upload_bytes_to_r2
