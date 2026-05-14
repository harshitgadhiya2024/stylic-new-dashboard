from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as redis

from app.config import settings

logger = logging.getLogger("photoshoot_events")

PHOTOSHOOT_EVENTS_CHANNEL = "photoshoot:events"


async def publish_photoshoot_event(payload: dict[str, Any]) -> None:
    """
    Publish one photoshoot event to Redis pub/sub so websocket clients can react in real time.
    """
    client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        n = await client.publish(PHOTOSHOOT_EVENTS_CHANNEL, json.dumps(payload))
        logger.info(
            "Photoshoot event published channel=%s receivers=%s event=%s photoshoot_id=%s",
            PHOTOSHOOT_EVENTS_CHANNEL,
            n,
            payload.get("event"),
            payload.get("photoshoot_id"),
        )
    except Exception as exc:
        logger.warning("Failed to publish photoshoot event: %s", exc)
    finally:
        await client.aclose()
