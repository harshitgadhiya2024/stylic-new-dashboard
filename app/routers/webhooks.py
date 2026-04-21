"""
KIE webhook receivers.

kie.ai posts the final result of a long-running job to this endpoint as soon
as the task finishes, letting the Celery worker unblock immediately instead
of polling every N seconds.

Register this URL in the kie.ai dashboard (or pass ``callBackUrl`` on each
createTask). The handler verifies the shared-secret header before accepting.

Security:
  - Requires KIE_WEBHOOK_SECRET to match X-KIE-Secret (or ?secret= query).
  - Secret MUST be set in production; if unset, the endpoint returns 503.

Expected payload (kie.ai callback format — tolerant to shape changes):

    {
      "code": 200,
      "data": {
        "taskId": "abc123",
        "state": "success" | "fail",
        "resultJson": "{\"resultUrls\":[\"https://...\"]}"
        "metadata": {"photoshoot_id": "...", "image_id": "...", ...}
      },
      "msg": "success"
    }
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request, status

from app.config import settings
from app.services import kie_task_registry

logger = logging.getLogger("kie_webhook")

router = APIRouter(prefix="/webhooks/kie", tags=["Webhooks"])


def _extract_result_url(data: dict[str, Any]) -> str | None:
    raw = data.get("resultJson")
    parsed: dict[str, Any] = {}
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
    elif isinstance(raw, dict):
        parsed = raw
    urls = parsed.get("resultUrls") or parsed.get("urls") or []
    if isinstance(urls, list) and urls:
        return str(urls[0])
    maybe = parsed.get("url") or parsed.get("image_url")
    return str(maybe) if maybe else None


@router.post("/upscale", status_code=status.HTTP_200_OK)
async def kie_upscale_callback(
    request: Request,
    x_kie_secret: str | None = Header(default=None, alias="X-KIE-Secret"),
):
    expected = (settings.KIE_WEBHOOK_SECRET or "").strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Webhook not configured (KIE_WEBHOOK_SECRET unset).",
        )

    # Accept secret via header OR ?secret= query (KIE dashboards differ).
    provided = x_kie_secret or request.query_params.get("secret") or ""
    if provided != expected:
        logger.warning("[kie-webhook] rejected: bad secret (len=%d)", len(provided))
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="bad secret")

    try:
        body = await request.json()
    except Exception as exc:
        logger.warning("[kie-webhook] invalid JSON body: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid json")

    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, dict):
        logger.warning("[kie-webhook] payload missing 'data' object: %s", body)
        return {"ok": False, "reason": "missing data"}

    task_id = str(data.get("taskId") or data.get("task_id") or "").strip()
    if not task_id:
        logger.warning("[kie-webhook] payload missing taskId: %s", body)
        return {"ok": False, "reason": "missing taskId"}

    state = (data.get("state") or "").lower()
    result_url = _extract_result_url(data) if state == "success" else None

    payload = {
        "state": state,
        "result_url": result_url,
        "raw": data,
    }

    try:
        had_meta = await kie_task_registry.publish_result(task_id, payload)
    except Exception as exc:
        logger.error("[kie-webhook] publish_result failed task_id=%s: %s", task_id, exc)
        # Still ACK to KIE — we don't want them retrying forever into a broken
        # Redis. A safety poll in the worker will catch the result.
        return {"ok": False, "reason": "redis unavailable"}

    logger.info(
        "[kie-webhook] task_id=%s state=%s url=%s had_waiter=%s",
        task_id, state, bool(result_url), had_meta,
    )
    return {"ok": True, "task_id": task_id, "state": state}
