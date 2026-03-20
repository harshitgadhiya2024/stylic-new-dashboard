import random
import string
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, status
from app.database import get_otps_collection
from app.config import settings


def generate_otp() -> str:
    return "".join(random.choices(string.digits, k=6))


async def save_otp(email: str, otp: str, purpose: str, extra: dict = None) -> None:
    col = get_otps_collection()
    await col.delete_many({"email": email, "purpose": purpose})
    doc = {
        "email": email,
        "otp": otp,
        "purpose": purpose,
        "attempts": 0,
        "is_verified": False,
        "expires_at": datetime.utcnow() + timedelta(minutes=settings.OTP_EXPIRE_MINUTES),
        "created_at": datetime.utcnow(),
    }
    if extra:
        doc.update(extra)
    await col.insert_one(doc)


async def get_pending_otp(email: str, purpose: str) -> dict | None:
    col = get_otps_collection()
    return await col.find_one({"email": email, "purpose": purpose, "is_verified": False})


async def verify_otp(email: str, otp: str, purpose: str) -> dict:
    col = get_otps_collection()
    record = await col.find_one({"email": email, "purpose": purpose, "is_verified": False})

    if not record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OTP not found or already used. Please request a new one.",
        )

    if record["expires_at"] < datetime.utcnow():
        await col.delete_one({"_id": record["_id"]})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OTP has expired. Please request a new one.",
        )

    if record["attempts"] >= 3:
        await col.delete_one({"_id": record["_id"]})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Too many failed attempts. Please request a new OTP.",
        )

    if record["otp"] != otp:
        await col.update_one({"_id": record["_id"]}, {"$inc": {"attempts": 1}})
        remaining = 2 - record["attempts"]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid OTP. {remaining} attempt(s) remaining.",
        )

    await col.update_one({"_id": record["_id"]}, {"$set": {"is_verified": True}})
    return record


async def check_otp_verified(email: str, purpose: str) -> dict | None:
    col = get_otps_collection()
    record = await col.find_one({"email": email, "purpose": purpose, "is_verified": True})
    if record and record["expires_at"] < datetime.utcnow():
        await col.delete_one({"_id": record["_id"]})
        return None
    return record


async def consume_otp(email: str, purpose: str) -> None:
    col = get_otps_collection()
    await col.delete_many({"email": email, "purpose": purpose})
