"""
Per-IP and per-email limits for the public contact-sales form.

Prefers ``REDIS_URL`` (shared across workers) when available; otherwise falls back
to an in-process best-effort limiter (single-process dev only).
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Optional, Tuple

from fastapi import Request
from fastapi import HTTPException, status

import redis  # type: ignore

from app.config import settings

logger = logging.getLogger("contact_sales.rate_limit")

_redis: Optional[redis.Redis] = None
_rlock = threading.Lock()

# ip -> list of monotonic second timestamps
_mem_ip: dict[str, list[float]] = {}
# email key -> (count, window_start) simple daily
_mem_email: dict[str, Tuple[int, float]] = {}
_mem_lock = threading.Lock()
_MEM_EMAIL_WINDOW = 86400.0


def _get_redis() -> Optional[redis.Redis]:
    global _redis
    if _redis is not None:
        return _redis
    with _rlock:
        if _redis is not None:
            return _redis
        url = (getattr(settings, "REDIS_URL", "") or "").strip()
        if not url:
            return None
        try:
            c = redis.Redis.from_url(
                url,
                decode_responses=True,
                socket_timeout=1.5,
                socket_connect_timeout=1.5,
            )
            c.ping()
            _redis = c
            logger.info("[contact-rl] Redis connected for contact form rate limit")
            return c
        except Exception as exc:  # noqa: BLE001
            logger.warning("[contact-rl] Redis unavailable (%s); using in-memory only", exc)
            return None


def client_ip_from_request(request: Request) -> str:
    if getattr(settings, "CONTACT_SALES_TRUST_X_FORWARDED_FOR", False):
        h = (request.headers.get("x-forwarded-for") or "").strip()
        if h:
            return h.split(",")[0].strip()[:200]
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _ip_limit_redis(redis_c: redis.Redis, ip: str) -> None:
    w = int(getattr(settings, "CONTACT_SALES_IP_WINDOW_S", 3600) or 3600)
    mx = int(getattr(settings, "CONTACT_SALES_MAX_PER_IP", 5) or 5)
    k = f"contact:sales:ip:{ip}"
    c = int(redis_c.incr(k))
    if c == 1:
        redis_c.expire(k, w)
    if c > mx:
        try:
            redis_c.decr(k)
        except Exception:  # noqa: BLE001
            pass
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please try again later.",
        )


def _ip_limit_mem(ip: str) -> None:
    mx = int(getattr(settings, "CONTACT_SALES_MAX_PER_IP", 5) or 5)
    w = int(getattr(settings, "CONTACT_SALES_IP_WINDOW_S", 3600) or 3600)
    now = time.monotonic()
    with _mem_lock:
        lst = [t for t in _mem_ip.get(ip, []) if now - t < w]
        if len(lst) >= mx:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please try again later.",
            )
        lst.append(now)
        _mem_ip[ip] = lst


def _email_quota_ok_redis(redis_c: redis.Redis, email: str) -> bool:
    mx = int(getattr(settings, "CONTACT_SALES_MAX_PER_EMAIL_PER_24H", 5) or 5)
    h = hashlib.sha256(email.lower().encode("utf-8")).hexdigest()[:32]
    k = f"contact:sales:em24h:{h}"
    n = int(redis_c.get(k) or 0)
    if n >= mx:
        return False
    return True


def _email_increase_redis(redis_c: redis.Redis, email: str) -> None:
    h = hashlib.sha256(email.lower().encode("utf-8")).hexdigest()[:32]
    k = f"contact:sales:em24h:{h}"
    p = int(redis_c.incr(k))
    if p == 1:
        redis_c.expire(k, 86400)


def _email_quota_ok_mem(email: str) -> bool:
    mx = int(getattr(settings, "CONTACT_SALES_MAX_PER_EMAIL_PER_24H", 5) or 5)
    h = email.lower()
    now = time.monotonic()
    with _mem_lock:
        cnt, t0 = _mem_email.get(h, (0, now))
        if now - t0 > _MEM_EMAIL_WINDOW:
            cnt, t0 = 0, now
        if cnt >= mx:
            return False
        return True


def _email_increase_mem(email: str) -> None:
    h = email.lower()
    now = time.monotonic()
    with _mem_lock:
        cnt, t0 = _mem_email.get(h, (0, now))
        if now - t0 > _MEM_EMAIL_WINDOW:
            cnt, t0 = 0, now
        _mem_email[h] = (cnt + 1, t0)


def enforce_rate_limits_for_contact(request: Request, email: str) -> str:
    """
    Enforce per-email 24h quota (read-only) first, then per-IP limit (increments IP).
    Returns client IP.
    """
    ip = client_ip_from_request(request)
    r = _get_redis()
    if r is not None:
        if not _email_quota_ok_redis(r, email):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests from this email address. Please try again later.",
            )
        _ip_limit_redis(r, ip)
    else:
        if not _email_quota_ok_mem(email):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests from this email address. Please try again later.",
            )
        _ip_limit_mem(ip)
    return ip


def record_successful_submission_rl(email: str) -> None:
    """Call after email send + DB store so quota reflects successful contacts only."""
    r = _get_redis()
    if r is not None:
        try:
            _email_increase_redis(r, email)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[contact-rl] failed to bump email count: %s", exc)
    else:
        _email_increase_mem(email)
