"""
KIE.ai account-wide rate limiter — Redis sliding-window.

KIE enforces "20 new generation requests per 10 seconds" per account.  That
limit is global — it applies across every Celery worker, every droplet, every
FastAPI process that shares the KIE API key.  A per-process or per-job
semaphore is not sufficient; we need a cross-process counter.

This module implements a sliding-window limiter using a Redis sorted set:

    ZADD  <key>  <now_ms>  <uuid>           # record an attempt
    ZREMRANGEBYSCORE <key> -inf <now_ms-W>  # evict entries older than W
    ZCARD <key>                             # current window size

If ZCARD < LIMIT, we've acquired a token.  Otherwise we sleep until the oldest
entry would age out (ZRANGE ... WITHSCORES) and try again.

The whole dance is performed atomically in a Lua script so there's no race
between workers.

Usage (sync, inside Celery tasks which run on a thread)::

    from app.services.kie_rate_limiter import acquire_kie_token_sync
    acquire_kie_token_sync()
    requests.post(kie_create_url, ...)

The limiter is safe to call when Redis is unreachable — it falls back to a
best-effort in-process limiter so a transient Redis outage does not take down
the photoshoot pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from typing import Optional

import redis as redis_lib  # type: ignore

from app.config import settings

logger = logging.getLogger("kie_rate_limiter")


# ---------------------------------------------------------------------------
# Atomic Lua script — try to acquire one token in the sliding window.
#
# Returns:
#   {1, <slots_used>}           — acquired; caller may proceed
#   {0, <wait_ms_until_free>}   — denied; caller must sleep wait_ms then retry
# ---------------------------------------------------------------------------
_ACQUIRE_LUA = """
local key    = KEYS[1]
local now    = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit  = tonumber(ARGV[3])
local member = ARGV[4]

redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
local used = tonumber(redis.call('ZCARD', key))

if used < limit then
    redis.call('ZADD', key, now, member)
    -- Expire the key a little after the window so it self-cleans if the
    -- limiter goes idle.
    redis.call('PEXPIRE', key, window * 2)
    return {1, used + 1}
end

-- Bucket is full. Caller must wait until the oldest entry ages out.
local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
local oldest_score = tonumber(oldest[2])
local wait_ms = (oldest_score + window) - now
if wait_ms < 1 then wait_ms = 1 end
return {0, wait_ms}
"""


# ---------------------------------------------------------------------------
# Redis client (lazy singleton)
# ---------------------------------------------------------------------------

_redis_client: Optional[redis_lib.Redis] = None
_script_sha: Optional[str] = None


def _get_redis() -> Optional[redis_lib.Redis]:
    """Return a shared ``redis.Redis`` client, or None if Redis is unreachable."""
    global _redis_client, _script_sha
    if _redis_client is not None:
        return _redis_client
    try:
        client = redis_lib.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=2.0,
            socket_timeout=2.0,
            decode_responses=False,
        )
        client.ping()
        _redis_client = client
        try:
            _script_sha = client.script_load(_ACQUIRE_LUA)
        except Exception:
            _script_sha = None
        return client
    except Exception as exc:
        logger.warning("[kie-rate] Redis unavailable — falling back to in-process limiter: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Fallback — in-process limiter (used only if Redis is unreachable)
# ---------------------------------------------------------------------------

_local_lock = threading.Lock()
_local_events: list[float] = []


def _acquire_local_sync(window_s: float, limit: int, max_wait_s: float) -> None:
    deadline = time.monotonic() + max_wait_s
    while True:
        now = time.monotonic()
        with _local_lock:
            cutoff = now - window_s
            # drop expired
            while _local_events and _local_events[0] < cutoff:
                _local_events.pop(0)
            if len(_local_events) < limit:
                _local_events.append(now)
                return
            wait = (_local_events[0] + window_s) - now
        if now + wait > deadline:
            logger.warning(
                "[kie-rate][local] wait budget exceeded — proceeding anyway "
                "(wait=%.2fs, budget=%.2fs)", wait, max_wait_s,
            )
            return
        time.sleep(min(wait + 0.05, 2.0))


# ---------------------------------------------------------------------------
# Public API — sync version (for Celery / thread contexts)
# ---------------------------------------------------------------------------

def acquire_kie_token_sync(
    *,
    requests_per_window: Optional[int]  = None,
    window_s:           Optional[float] = None,
    max_wait_s:         Optional[float] = None,
    key:                Optional[str]   = None,
) -> None:
    """Block until a KIE createTask slot is free.

    Call this immediately before ``requests.post(kie_create_url, ...)`` in any
    sync code path (Celery worker, background thread, sync CLI).  The function
    is idempotent per caller — callers pair one ``acquire`` with one HTTP
    request.
    """
    if not getattr(settings, "KIE_RATE_LIMIT_ENABLED", True):
        return

    limit     = requests_per_window if requests_per_window is not None else settings.KIE_RATE_LIMIT_REQUESTS
    window    = window_s            if window_s            is not None else settings.KIE_RATE_LIMIT_WINDOW_S
    budget    = max_wait_s          if max_wait_s          is not None else settings.KIE_RATE_LIMIT_MAX_WAIT_S
    redis_key = key                 if key                 is not None else settings.KIE_RATE_LIMIT_KEY

    client = _get_redis()
    if client is None:
        _acquire_local_sync(window, limit, budget)
        return

    deadline = time.monotonic() + budget
    window_ms = int(window * 1000)
    member_prefix = uuid.uuid4().hex

    while True:
        now_ms = int(time.time() * 1000)
        member = f"{member_prefix}:{now_ms}"
        try:
            if _script_sha is not None:
                try:
                    result = client.evalsha(_script_sha, 1, redis_key, now_ms, window_ms, limit, member)
                except redis_lib.exceptions.NoScriptError:
                    # Redis restarted / failover — re-load once.
                    result = client.eval(_ACQUIRE_LUA, 1, redis_key, now_ms, window_ms, limit, member)
            else:
                result = client.eval(_ACQUIRE_LUA, 1, redis_key, now_ms, window_ms, limit, member)
        except Exception as exc:
            logger.warning("[kie-rate] Redis EVAL failed — falling back to local limiter: %s", exc)
            _acquire_local_sync(window, limit, budget)
            return

        acquired, payload = int(result[0]), int(result[1])
        if acquired == 1:
            if payload >= max(1, int(limit * 0.8)):
                logger.info("[kie-rate] token acquired used=%d/%d", payload, limit)
            return

        wait_s = payload / 1000.0
        remaining_budget = deadline - time.monotonic()
        if remaining_budget <= 0:
            logger.warning(
                "[kie-rate] wait budget exhausted — proceeding without token "
                "(budget=%.1fs, wait_required=%.2fs)", budget, wait_s,
            )
            return
        sleep_for = min(wait_s + 0.05, remaining_budget, 2.0)
        logger.info(
            "[kie-rate] bucket full — sleeping %.2fs (budget_left=%.1fs, limit=%d/%ds)",
            sleep_for, remaining_budget, limit, int(window),
        )
        time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# Public API — async version (for FastAPI / asyncio contexts)
# ---------------------------------------------------------------------------

async def acquire_kie_token_async(**kwargs) -> None:
    """Async wrapper around ``acquire_kie_token_sync`` — runs the blocking
    Redis/time.sleep dance in a thread so the event loop stays free."""
    await asyncio.to_thread(acquire_kie_token_sync, **kwargs)
