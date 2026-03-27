"""
Celery tasks for photoshoot processing.

Each task wraps the async run_photoshoot_job in a fresh asyncio event loop
so it can run safely inside a synchronous Celery worker process.

Root-cause note
---------------
Motor (AsyncIOMotorClient) binds to the event loop that is current at the
time the first async call is made.  When asyncio.run() finishes it closes
that loop, so any Motor client created *before* or *during* that run is
permanently broken for future asyncio.run() calls.

Fix: create a fresh AsyncIOMotorClient at the start of every asyncio.run()
call (via make_motor_client()), pass it into the service layer, and close it
before the loop exits.  Every Mongo read/write in the job (including
``poses_data`` and ``upscaling_data`` via Modal) must use that client — not
``get_*_collection()`` globals.  The global singleton in database.py is for
the FastAPI process only.

The worker is started with --concurrency=1 so only ONE job runs at a time,
preventing simultaneous SeedDream / Modal calls.
"""

import asyncio
import logging

from celery import Task

from app.worker import celery_app

logger = logging.getLogger("photoshoot_tasks")


class _PhotoshootTask(Task):
    """Base task that logs failures and updates photoshoot status in DB."""

    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        photoshoot_id = args[0] if args else kwargs.get("photoshoot_id", "unknown")
        logger.error(
            "[task] FAILED — photoshoot_id=%s | task_id=%s | error=%s",
            photoshoot_id, task_id, exc,
        )
        asyncio.run(_mark_failed(photoshoot_id, str(exc)))


async def _mark_failed(photoshoot_id: str, error: str) -> None:
    from datetime import datetime, timezone
    from app.database import make_motor_client
    from app.config import settings

    client = make_motor_client()
    try:
        col = client[settings.MONGO_DB_NAME]["photoshoots"]
        await col.update_one(
            {"photoshoot_id": photoshoot_id},
            {"$set": {
                "status":     "failed",
                "error":      error,
                "updated_at": datetime.now(timezone.utc),
            }},
        )
    finally:
        client.close()


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
    """Create a fresh Motor client, run the pipeline, then close the client."""
    from app.database import make_motor_client
    from app.services.photoshoot_service import run_photoshoot_job

    client = make_motor_client()
    try:
        await run_photoshoot_job(photoshoot_id, req, motor_client=client)
    finally:
        client.close()


def get_queue_length() -> int:
    """Return number of photoshoot jobs currently waiting in the queue."""
    try:
        import redis as redis_lib
        from app.config import settings
        r = redis_lib.from_url(settings.REDIS_URL)
        return r.llen("photoshoots")
    except Exception:
        return -1  # Redis unavailable — return -1 to signal unknown
