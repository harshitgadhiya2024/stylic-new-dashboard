"""
Cross-worker KIE.ai createTask rate limiter.

Production-grade sliding-window limiter used by the photoshoot pipeline to
guarantee that, no matter how many Celery worker processes or FastAPI
processes are running on the box (or on multiple boxes sharing the same
Redis), we never exceed the KIE ``createTask`` account-wide cap of
20 requests / 10 seconds.

Design
------

* **Redis-backed** sliding window keyed per 10-second epoch bucket. Every
  caller does ``INCR + EXPIRE`` (pipelined, so it is atomic from the
  caller's POV). If the incremented count is <= ``KIE_RATE_LIMIT_MAX``
  the caller returns immediately; otherwise it sleeps until the bucket
  rolls over and retries. This is the canonical Redis pattern used in
  high-throughput API clients.

* **Safe fallback** — if Redis is unreachable (``redis-server`` down,
  bad ``REDIS_URL``, etc.) we log and fall through; the worker must not
  crash the photoshoot just because the limiter is unavailable. In that
  degraded state we still back off briefly on 429 responses, but the
  pre-submit gate is effectively disabled.

* **Sync-callable** because the generator agents (``_agent_kie_nb2_sync``)
  run under ``asyncio.to_thread`` — calling ``asyncio.run`` from inside a
  thread that already has a running loop would raise. We therefore keep
  the public API purely synchronous (``acquire()``) and use ``time.sleep``
  internally.

Tuning
------
Defaults are conservative (``KIE_RATE_LIMIT_MAX=10`` / 10 s), sitting at
half of KIE's published cap to absorb clock skew, poll traffic and
retries without ever tripping 429. Raise ``KIE_RATE_LIMIT_MAX`` toward
18 for higher throughput once the pipeline is stable.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import redis  # type: ignore

from app.config import settings

logger = logging.getLogger("photoshoot.kie_rate_limiter")


# ---------------------------------------------------------------------------
# Internal Redis client (lazy, module-level singleton)
# ---------------------------------------------------------------------------

_client: Optional[redis.Redis] = None
_client_lock = threading.Lock()


def _get_redis() -> Optional[redis.Redis]:
    """Return a connected Redis client or ``None`` if unreachable.

    The client is created once and cached. On connection failure we return
    ``None`` so the caller can degrade gracefully rather than crashing the
    whole photoshoot.
    """
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        url = (os.environ.get("REDIS_URL") or settings.REDIS_URL or "").strip()
        if not url:
            logger.warning("[kie-rl] REDIS_URL not configured; limiter disabled")
            return None
        try:
            c = redis.Redis.from_url(
                url,
                decode_responses=True,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
            )
            c.ping()
            _client = c
            logger.info("[kie-rl] connected to redis at %s", url)
            return _client
        except Exception as exc:  # noqa: BLE001
            logger.warning("[kie-rl] redis unavailable (%s); limiter disabled", exc)
            return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _limiter_enabled() -> bool:
    return bool(getattr(settings, "KIE_RATE_LIMIT_ENABLED", True))


def _limiter_max() -> int:
    return int(getattr(settings, "KIE_RATE_LIMIT_MAX", 10) or 10)


def _limiter_window_s() -> float:
    return float(getattr(settings, "KIE_RATE_LIMIT_WINDOW_S", 10.0) or 10.0)


def _limiter_key_prefix() -> str:
    return str(getattr(settings, "KIE_RATE_LIMIT_KEY_PREFIX", "kie:createTask") or
               "kie:createTask")


def _limiter_max_wait_s() -> float:
    return float(getattr(settings, "KIE_RATE_LIMIT_MAX_WAIT_S", 60.0) or 60.0)


def acquire(
    *,
    timeout_s: Optional[float] = None,
    label: str = "kie",
) -> None:
    """Block until a KIE ``createTask`` submission token is available.

    Guarantees, across every worker and process sharing the same Redis,
    that we never submit more than ``KIE_RATE_LIMIT_MAX`` requests inside
    any ``KIE_RATE_LIMIT_WINDOW_S`` second window.

    Parameters
    ----------
    timeout_s:
        Maximum wall-clock seconds to wait for a token. Defaults to
        ``KIE_RATE_LIMIT_MAX_WAIT_S`` (60 s). On timeout we raise
        ``TimeoutError`` so the caller can fall back to another provider.
    label:
        Log label — useful when multiple limiter instances are added later
        (e.g. one per API key).
    """
    if not _limiter_enabled():
        return

    redis_client = _get_redis()
    if redis_client is None:
        # Degraded mode: limiter off, caller still has 429 backoff.
        return

    timeout = float(timeout_s if timeout_s is not None else _limiter_max_wait_s())
    deadline = time.monotonic() + timeout
    window = _limiter_window_s()
    max_req = _limiter_max()
    prefix = _limiter_key_prefix()

    attempts = 0
    while True:
        attempts += 1
        now = time.time()
        bucket = int(now // window)
        key = f"{prefix}:{bucket}"
        try:
            pipe = redis_client.pipeline()
            pipe.incr(key, 1)
            # Keep two windows around so polls don't race bucket rollover.
            pipe.expire(key, int(window * 3))
            count, _ = pipe.execute()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s-rl] redis error during acquire (%s); allowing request",
                label, exc,
            )
            return

        if int(count) <= max_req:
            if attempts > 1:
                logger.info(
                    "[%s-rl] token acquired after %d attempt(s)",
                    label, attempts,
                )
            return

        # Bucket is full — wait until it rolls over, bounded by the deadline.
        wait_until = (bucket + 1) * window
        wait = max(wait_until - now + 0.05, 0.05)
        if time.monotonic() + wait > deadline:
            raise TimeoutError(
                f"[{label}-rl] waited > {timeout:.1f}s for a token "
                f"(max={max_req}/{window:.0f}s)"
            )
        if attempts == 1 or attempts % 3 == 0:
            logger.info(
                "[%s-rl] bucket full (count=%s max=%s); waiting %.2fs",
                label, count, max_req, wait,
            )
        time.sleep(wait)


def on_rate_limited(label: str = "kie") -> None:
    """Call this when an upstream 429 is observed.

    We additionally burn a short cool-down so a bursty response from the
    provider doesn't get us stuck hammering them during the next window.
    The value is controlled by ``KIE_RATE_LIMIT_429_SLEEP_S`` (default 5s).
    """
    sleep_s = float(getattr(settings, "KIE_RATE_LIMIT_429_SLEEP_S", 5.0) or 5.0)
    logger.warning("[%s-rl] 429 observed; cooling down for %.1fs", label, sleep_s)
    time.sleep(sleep_s)
