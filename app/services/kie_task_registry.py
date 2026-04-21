"""
Redis-backed registry that bridges the KIE webhook (arrives in the FastAPI
process) and the Celery worker that submitted the upscale job.

Flow:
    1. Celery worker calls ``register(task_id, meta)`` right after createTask.
    2. Celery worker calls ``wait_for(task_id, timeout=N)`` which blocks on
       ``BLPOP kie:done:<task_id>``.
    3. When KIE POSTs to /webhooks/kie/upscale, the FastAPI handler calls
       ``publish_result(task_id, payload)`` which stores the result and
       unblocks any waiter via ``LPUSH kie:done:<task_id> 1``.
    4. If the webhook never arrives, the waiter times out and the caller is
       expected to fall back to a direct ``recordInfo`` poll.

Keys (all TTL = 1 hour):
    kie:meta:<task_id>    JSON { photoshoot_id, image_id, submitted_at }
    kie:result:<task_id>  JSON { state, result_url, raw }
    kie:done:<task_id>    list — BLPOP sentinel (value is just "1")
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Optional

import redis as redis_sync          # used by Celery worker (sync BLPOP)
import redis.asyncio as redis_async # used by FastAPI webhook handler

from app.config import settings

logger = logging.getLogger("kie_registry")

_TTL_SECONDS = 3600
_META_KEY = "kie:meta:{}"
_RESULT_KEY = "kie:result:{}"
_DONE_KEY = "kie:done:{}"


# ---------------------------------------------------------------------------
# Connection singletons (lazy)
# ---------------------------------------------------------------------------

_sync_client: Optional[redis_sync.Redis] = None
_async_client: Optional[redis_async.Redis] = None


def _get_sync() -> redis_sync.Redis:
    global _sync_client
    if _sync_client is None:
        _sync_client = redis_sync.from_url(
            settings.REDIS_URL, decode_responses=True,
            socket_timeout=5, socket_connect_timeout=3,
        )
    return _sync_client


def _get_async() -> redis_async.Redis:
    global _async_client
    if _async_client is None:
        _async_client = redis_async.from_url(
            settings.REDIS_URL, decode_responses=True,
            socket_timeout=5, socket_connect_timeout=3,
        )
    return _async_client


# ---------------------------------------------------------------------------
# Writers — called from Celery worker on submit
# ---------------------------------------------------------------------------

def register_sync(task_id: str, photoshoot_id: str, image_id: str) -> None:
    try:
        meta = {
            "photoshoot_id": photoshoot_id,
            "image_id": image_id,
            "submitted_at": time.time(),
        }
        client = _get_sync()
        pipe = client.pipeline()
        pipe.setex(_META_KEY.format(task_id), _TTL_SECONDS, json.dumps(meta))
        # Clear any stale result/done from a prior run with the same id.
        pipe.delete(_RESULT_KEY.format(task_id))
        pipe.delete(_DONE_KEY.format(task_id))
        pipe.execute()
    except Exception as exc:
        logger.warning("[kie-registry] register failed task_id=%s: %s", task_id, exc)


# ---------------------------------------------------------------------------
# Blocking wait — called from Celery worker in a thread
# ---------------------------------------------------------------------------

def wait_for_sync(task_id: str, timeout_s: int) -> Optional[dict]:
    """
    Block up to ``timeout_s`` seconds for a webhook to publish the result.

    Returns the parsed result dict on success, or ``None`` on timeout.
    Safe to call multiple times (the result key has TTL — if result was already
    published, we return it immediately on the second wait).
    """
    try:
        client = _get_sync()

        # Fast path: result already landed before we started waiting.
        raw = client.get(_RESULT_KEY.format(task_id))
        if raw:
            return json.loads(raw)

        timeout_int = max(1, int(timeout_s))
        popped = client.blpop(_DONE_KEY.format(task_id), timeout=timeout_int)
        if not popped:
            return None  # timed out — caller falls back to direct poll

        raw = client.get(_RESULT_KEY.format(task_id))
        return json.loads(raw) if raw else None
    except Exception as exc:
        logger.warning("[kie-registry] wait_for failed task_id=%s: %s", task_id, exc)
        return None


async def wait_for_async(task_id: str, timeout_s: float) -> Optional[dict]:
    """Async wrapper so the celery event loop can await without blocking."""
    return await asyncio.to_thread(wait_for_sync, task_id, int(timeout_s))


# ---------------------------------------------------------------------------
# Publisher — called from the FastAPI webhook handler
# ---------------------------------------------------------------------------

async def publish_result(task_id: str, payload: dict[str, Any]) -> bool:
    """
    Store the webhook payload and unblock any waiting Celery worker.

    Returns True if a meta entry existed (meaning a worker is presumably
    waiting), False otherwise (stray/late webhook — still stored).
    """
    client = _get_async()
    try:
        meta_raw = await client.get(_META_KEY.format(task_id))
    except Exception as exc:
        logger.error("[kie-registry] redis unreachable on publish: %s", exc)
        raise

    had_meta = meta_raw is not None
    serialized = json.dumps(payload, default=str)

    pipe = client.pipeline()
    pipe.setex(_RESULT_KEY.format(task_id), _TTL_SECONDS, serialized)
    pipe.rpush(_DONE_KEY.format(task_id), "1")
    pipe.expire(_DONE_KEY.format(task_id), _TTL_SECONDS)
    await pipe.execute()

    if not had_meta:
        logger.info(
            "[kie-registry] late/unknown webhook task_id=%s — stored for TTL=%ds",
            task_id, _TTL_SECONDS,
        )
    return had_meta


async def get_meta(task_id: str) -> Optional[dict]:
    """For diagnostics / webhook routing."""
    client = _get_async()
    try:
        raw = await client.get(_META_KEY.format(task_id))
        return json.loads(raw) if raw else None
    except Exception:
        return None
