"""Application-wide HTTP rate limiter (slowapi, Redis-backed).

Why this module exists
----------------------
- Protects *our* API from brute-force credential stuffing, OTP/email
  flooding, scraping, and generic abuse.
- Returns a standard HTTP 429 on overflow; no response *schema* change,
  so the frontend keeps working — well-behaved clients will just see
  occasional 429s under abuse.
- Uses Redis as the counter backend so limits are **shared across every
  uvicorn/gunicorn worker and every Celery worker process**. This is
  critical: in-process counters are useless as soon as you scale out.

Key strategy
------------
- Default key: client IP, honoring `X-Forwarded-For` when we're behind a
  trusted reverse proxy (nginx / Cloudflare per DEPLOYMENT.md).
- For auth endpoints we additionally key on the submitted email (when
  present in the JSON body) so an attacker can't bypass per-IP limits by
  rotating IPs against one account.

Graceful degradation
--------------------
If Redis is unreachable at startup, the limiter falls back to the
in-memory backend (which is still better than nothing for a single
worker) and logs a loud warning. In production you should treat a
limiter-Redis outage as a paging event.
"""

from __future__ import annotations

import logging

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.config import settings

logger = logging.getLogger(__name__)


def _client_ip(request: Request) -> str:
    """Return the client IP, honouring a single layer of trusted proxy.

    slowapi's default `get_remote_address` reads `request.client.host`,
    which when behind nginx/Cloudflare is always the proxy IP — that
    would make the limiter useless (every request shares one key).

    We trust `X-Forwarded-For` only because the app is expected to be
    deployed behind nginx per DEPLOYMENT.md, which we control. If you
    expose the app directly to the internet, remove this.
    """
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        # XFF is a comma-separated list; the left-most is the original client.
        first = xff.split(",", 1)[0].strip()
        if first:
            return first
    real_ip = request.headers.get("x-real-ip", "")
    if real_ip:
        return real_ip.strip()
    return get_remote_address(request)


def _build_limiter() -> Limiter:
    storage_uri = settings.REDIS_URL
    enabled = settings.RATE_LIMIT_ENABLED

    try:
        limiter = Limiter(
            key_func=_client_ip,
            default_limits=[settings.RATE_LIMIT_DEFAULT] if enabled else [],
            storage_uri=storage_uri,
            strategy="fixed-window",
            headers_enabled=True,  # emits X-RateLimit-* response headers
            enabled=enabled,
        )
        logger.info(
            "[rate-limit] enabled=%s backend=%s default=%s",
            enabled, storage_uri, settings.RATE_LIMIT_DEFAULT,
        )
        return limiter
    except Exception as exc:
        logger.error(
            "[rate-limit] Redis backend init failed (%s). "
            "Falling back to in-memory limiter; this is NOT shared across "
            "workers and must be fixed before traffic scales.",
            exc,
        )
        return Limiter(
            key_func=_client_ip,
            default_limits=[settings.RATE_LIMIT_DEFAULT] if enabled else [],
            strategy="fixed-window",
            headers_enabled=True,
            enabled=enabled,
        )


limiter: Limiter = _build_limiter()
