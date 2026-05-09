from __future__ import annotations

import json
import logging

import redis.asyncio as redis
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from app.config import settings
from app.database import get_users_collection
from app.services.jwt_service import decode_token
from app.services.photoshoot_events_service import PHOTOSHOOT_EVENTS_CHANNEL

logger = logging.getLogger("photoshoot_ws")

router = APIRouter(tags=["Photoshoots WebSocket"])


@router.websocket("/api/v1/photoshoots/ws")
async def photoshoot_status_ws(
    websocket: WebSocket,
    token: str = Query(..., description="User access token"),
):
    # Validate access token and active user.
    try:
        payload = decode_token(token, token_type="access")
        user_id = str(payload.get("sub") or "").strip()
        if not user_id:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        user = await get_users_collection().find_one({"user_id": user_id, "is_active": True})
        if not user:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
    except Exception:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    await websocket.send_json({"event": "connected", "message": "photoshoot websocket connected"})

    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(PHOTOSHOOT_EVENTS_CHANNEL)

    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=20.0)
            if message is None:
                # Keepalive so idle proxies don't drop the socket silently.
                await websocket.send_json({"event": "heartbeat"})
                continue
            try:
                event = json.loads(message.get("data") or "{}")
            except Exception:
                continue
            if str(event.get("user_id") or "") != user_id:
                continue
            await websocket.send_json(event)
    except WebSocketDisconnect:
        logger.info("[photoshoot-ws] disconnected user_id=%s", user_id)
    except Exception as exc:
        logger.warning("[photoshoot-ws] loop error user_id=%s: %s", user_id, exc)
    finally:
        try:
            await pubsub.unsubscribe(PHOTOSHOOT_EVENTS_CHANNEL)
            await pubsub.aclose()
        finally:
            await redis_client.aclose()
