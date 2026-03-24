"""
Celery tasks for photoshoot processing.

Each task wraps the async run_photoshoot_job in a fresh asyncio event loop
so it can run safely inside a synchronous Celery worker process.

The worker is started with --concurrency=1 so only ONE job runs at a time,
preventing simultaneous SeedDream / Modal calls.
"""

import asyncio
import logging

from celery import Task

from app.worker import celery_app

logger = logging.getLogger("photoshoot_tasks")


class _PhotoshootTask(Task):
    """Base task that logs start/end and updates photoshoot status on failure."""

    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        photoshoot_id = args[0] if args else kwargs.get("photoshoot_id", "unknown")
        logger.error(
            "[task] FAILED — photoshoot_id=%s | task_id=%s | error=%s",
            photoshoot_id, task_id, exc,
        )
        # Mark the photoshoot as failed in DB so the client sees a final state
        asyncio.run(_mark_failed(photoshoot_id, str(exc)))


async def _mark_failed(photoshoot_id: str, error: str) -> None:
    from datetime import datetime, timezone
    from app.database import get_photoshoots_collection
    col = get_photoshoots_collection()
    await col.update_one(
        {"photoshoot_id": photoshoot_id},
        {"$set": {
            "status":     "failed",
            "error":      error,
            "updated_at": datetime.now(timezone.utc),
        }},
    )


@celery_app.task(
    bind=True,
    base=_PhotoshootTask,
    name="app.tasks.photoshoot_tasks.run_photoshoot_task",
    queue="photoshoots",
    max_retries=2,
    default_retry_delay=60,
    acks_late=True,
)
def run_photoshoot_task(self, photoshoot_id: str, req: dict) -> dict:
    """
    Executes the full photoshoot pipeline synchronously inside a Celery worker.

    Args:
        photoshoot_id: UUID of the photoshoot document (already saved as 'processing').
        req:           The job payload dict (same as what was passed to run_photoshoot_job).

    Returns:
        {"photoshoot_id": ..., "status": "completed"} on success.
    """
    logger.info("[task] Starting photoshoot task — photoshoot_id=%s", photoshoot_id)
    try:
        asyncio.run(_run_async_job(photoshoot_id, req))
        logger.info("[task] Completed — photoshoot_id=%s", photoshoot_id)
        return {"photoshoot_id": photoshoot_id, "status": "completed"}
    except Exception as exc:
        logger.error("[task] Error — photoshoot_id=%s | %s", photoshoot_id, exc)
        raise self.retry(exc=exc)


async def _run_async_job(photoshoot_id: str, req: dict) -> None:
    """Thin async wrapper so Celery (sync) can call the async pipeline."""
    from app.services.photoshoot_service import run_photoshoot_job
    await run_photoshoot_job(photoshoot_id, req)


def get_queue_length() -> int:
    """Return number of photoshoot jobs currently waiting in the queue."""
    try:
        import redis as redis_lib
        from app.config import settings
        r = redis_lib.from_url(settings.REDIS_URL)
        return r.llen("photoshoots")
    except Exception:
        return -1  # Redis unavailable — return -1 to signal unknown
