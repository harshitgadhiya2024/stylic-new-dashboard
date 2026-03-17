"""
Seed script: loads background_catalog.json into the `backgrounds` MongoDB collection.

Usage (run from project root):
    python scripts/seed_backgrounds.py

Each document stored:
    background_id    – UUID (from source JSON)
    background_type  – e.g. "abstract_blur_background"
    background_name  – same value as background_type
    background_url   – S3 public URL
    count            – 0 (default usage counter)
    tags             – []
    notes            – ""
    is_default       – bool (from source JSON)
    is_active        – bool (from source JSON)
    created_at       – datetime (from source JSON)
    updated_at       – datetime (set to now on insert)
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

# ── Bootstrap ─────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MONGO_URI    = os.getenv("MONGO_URI")
MONGO_DB     = os.getenv("MONGO_DB_NAME")
CATALOG_PATH = ROOT / "background_catalog.json"
COLLECTION   = "backgrounds"

if not MONGO_URI or not MONGO_DB:
    sys.exit("[ERROR] MONGO_URI or MONGO_DB_NAME missing from .env")

if not CATALOG_PATH.exists():
    sys.exit(f"[ERROR] Catalog not found: {CATALOG_PATH}")


def build_documents(raw: list[dict]) -> list[dict]:
    now  = datetime.now(timezone.utc)
    docs = []

    for entry in raw:
        raw_ts = entry.get("created_at", "")
        try:
            created_at = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            created_at = now

        background_type = entry.get("background_type", "")

        doc = {
            "background_id":   entry["background_id"],
            "background_type": background_type,
            "background_name": background_type,
            "background_url":  entry.get("background_url", ""),
            "count":           0,
            "tags":            [],
            "notes":           "",
            "is_default":      entry.get("is_default", True),
            "is_active":       entry.get("is_active", True),
            "created_at":      created_at,
            "updated_at":      now,
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

    col.create_index("background_id", unique=True)

    inserted = 0
    updated  = 0

    for doc in docs:
        existing = col.find_one({"background_id": doc["background_id"]})
        if existing:
            col.update_one(
                {"background_id": doc["background_id"]},
                {
                    "$set": {
                        "background_name": doc["background_name"],
                        "background_url":  doc["background_url"],
                        "is_default":      doc["is_default"],
                        "is_active":       doc["is_active"],
                        "updated_at":      doc["updated_at"],
                    }
                },
            )
            updated += 1
        else:
            col.insert_one(doc)
            inserted += 1

    print(f"[DONE] Inserted: {inserted} | Updated: {updated}")

    print("\n── Backgrounds stored ──────────────────────────────")
    for row in col.find({}, {"background_type": 1, "background_name": 1, "_id": 0}).sort("background_type", 1):
        print(f"  {row['background_type']}")

    client.close()


if __name__ == "__main__":
    main()
