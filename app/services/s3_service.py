import uuid
import os
import boto3
from botocore.exceptions import ClientError
from fastapi import HTTPException, status, UploadFile
from app.config import settings

_s3_client = None


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
    return _s3_client


def upload_file_to_s3(file: UploadFile, folder: str = "uploads") -> str:
    ext = ""
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()

    key = f"{folder}/{uuid.uuid4().hex}{ext}"

    try:
        s3 = get_s3_client()
        s3.upload_fileobj(
            file.file,
            settings.AWS_S3_BUCKET_NAME,
            key,
            ExtraArgs={"ContentType": file.content_type or "application/octet-stream"},
        )
    except ClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {exc.response['Error']['Message']}",
        )

    return f"https://{settings.AWS_S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"


def upload_bytes_to_s3(data: bytes, key: str, content_type: str = "image/png") -> str:
    """Upload raw bytes directly to S3 and return the public URL."""
    try:
        s3 = get_s3_client()
        s3.put_object(
            Bucket=settings.AWS_S3_BUCKET_NAME,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
    except ClientError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"S3 upload failed: {exc.response['Error']['Message']}",
        )

    return f"https://{settings.AWS_S3_BUCKET_NAME}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"
