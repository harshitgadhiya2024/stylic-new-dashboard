"""
Seed script: loads face_catalog.json into the `model_faces` MongoDB collection.

Usage (run from project root):
    python scripts/seed_model_faces.py

Each document stored:
    model_id            – UUID (from face_id in source JSON)
    model_name          – realistic human name based on category
    model_category      – e.g. "adult_female"
    model_configuration – nested configuration dict
    tags                – []
    notes               – ""
    model_used_count    – 0
    face_url            – S3 public URL
    is_default          – bool
    is_active           – bool
    is_favorite         – False
    created_at          – datetime (from source JSON)
    updated_at          – datetime (set to now on insert)
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

# ── Bootstrap ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MONGO_URI    = os.getenv("MONGO_URI")
MONGO_DB     = os.getenv("MONGO_DB_NAME")
CATALOG_PATH = ROOT / "face_catalog.json"
COLLECTION   = "model_faces"

if not MONGO_URI or not MONGO_DB:
    sys.exit("[ERROR] MONGO_URI or MONGO_DB_NAME missing from .env")

if not CATALOG_PATH.exists():
    sys.exit(f"[ERROR] Catalog not found: {CATALOG_PATH}")


# ── Name pools (Indian names, per category) ────────────────────────────────
# Each pool has more names than max possible entries per category
_NAMES: dict[str, list[str]] = {
    "adult_female": [
        "Priya Sharma", "Ananya Patel", "Kavya Nair", "Meera Iyer",
        "Divya Joshi", "Riya Gupta", "Shreya Mehta", "Pooja Verma",
        "Sneha Reddy", "Tanvi Desai", "Aisha Khan", "Swati Kulkarni",
    ],
    "adult_male": [
        "Rohan Patel", "Arjun Mehta", "Vikram Singh", "Aditya Kumar",
        "Kiran Joshi", "Rahul Sharma", "Siddharth Nair", "Nikhil Verma",
        "Dev Gupta", "Manav Desai", "Kabir Khan", "Yash Reddy",
    ],
    "kid_girl": [
        "Aarohi Patel", "Diya Sharma", "Isha Nair", "Pihu Gupta",
        "Navya Mehta", "Riya Joshi", "Aditi Singh", "Kriya Verma",
        "Myra Iyer", "Siya Desai", "Tara Reddy", "Nisha Kulkarni",
    ],
    "kid_boy": [
        "Aryan Patel", "Veer Sharma", "Krish Nair", "Aarav Gupta",
        "Dev Mehta", "Rayan Joshi", "Ishaan Singh", "Kabir Verma",
        "Vivaan Iyer", "Reyansh Desai", "Adi Reddy", "Dhruv Kulkarni",
    ],
}

_FALLBACK_FEMALE = ["Anjali", "Deepika", "Sonal", "Bhavna", "Hetal"]
_FALLBACK_MALE   = ["Amit", "Rajesh", "Suresh", "Mahesh", "Dinesh"]


def _pick_name(category: str, index: int) -> str:
    """Return a unique realistic name for the given category and 0-based index."""
    if "female" in category or "girl" in category:
        pool = _NAMES.get(category, _FALLBACK_FEMALE)
    else:
        pool = _NAMES.get(category, _FALLBACK_MALE)

    if index < len(pool):
        return pool[index]

    # Safety fallback if catalog ever grows beyond the pool
    surname = "Patel" if index % 2 == 0 else "Sharma"
    return f"Model {index + 1} {surname}"


def build_documents(raw: list[dict]) -> list[dict]:
    now = datetime.now(timezone.utc)
    counters: defaultdict[str, int] = defaultdict(int)
    docs = []

    for entry in raw:
        category = entry.get("category", "unknown")
        idx = counters[category]
        counters[category] += 1

        raw_ts = entry.get("created_at", "")
        try:
            created_at = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            created_at = now

        doc = {
            "model_id":            entry["face_id"],
            "model_name":          _pick_name(category, idx),
            "model_category":      category,
            "model_configuration": entry.get("configuration", {}),
            "tags":                [],
            "notes":               "",
            "model_used_count":    0,
            "face_url":            entry.get("face_url", ""),
            "is_default":          entry.get("is_default", False),
            "is_active":           entry.get("is_active", True),
            "is_favorite":         False,
            "created_at":          created_at,
            "updated_at":          now,
        }
        docs.append(doc)

    return docs


def main() -> None:
    with open(CATALOG_PATH, encoding="utf-8") as f:
        raw: list[dict] = json.load(f)

    print(f"[INFO] Loaded {len(raw)} entries from {CATALOG_PATH.name}")

    docs = build_documents(raw)

    client = MongoClient(MONGO_URI)
    db     = client[MONGO_DB]
    col    = db[COLLECTION]

    col.create_index("model_id", unique=True)

    # Drop existing docs and re-insert so names get updated cleanly
    updated  = 0
    inserted = 0
    skipped  = 0

    for doc in docs:
        existing = col.find_one({"model_id": doc["model_id"]})
        if existing:
            col.update_one(
                {"model_id": doc["model_id"]},
                {"$set": {"model_name": doc["model_name"], "updated_at": doc["updated_at"]}},
            )
            updated += 1
        else:
            col.insert_one(doc)
            inserted += 1

    print(f"[DONE] Inserted: {inserted} | Updated: {updated} | Skipped: {skipped}")

    # Print all model names for verification
    print("\n── Model names stored ──────────────────────────────")
    for row in col.find({}, {"model_category": 1, "model_name": 1, "_id": 0}).sort("model_category", 1):
        print(f"  [{row['model_category']:<15}]  {row['model_name']}")

    client.close()


if __name__ == "__main__":
    main()
