"""
Celery tasks for photoshoot processing — agentic LangGraph runner.

The heavy lifting lives in ``app.services.photoshoot_service``, which compiles
a multi-node LangGraph ``StateGraph``:

    init → gen_kie_nb2 → gen_vertex_nb2 → gen_vertex_nbpro → gen_evolink_nb2 → finalize → END

Each ``gen_*`` node is an *agent* — it runs one generator provider across all
pending poses with its own concurrency + retry policy.  As soon as any pose
finishes generation, a stage-2 task (R2 upload + KIE upscale + Mongo writes)
is fired for that pose immediately; the overall pipeline does NOT block the
other poses.  ``finalize`` awaits any remaining stage-2 tasks and marks the
photoshoot document ``completed`` / ``partial`` / ``failed``.

Celery's role here is:

  1. Own the event loop for the duration of ONE photoshoot job.
  2. Create a fresh Motor (AsyncIOMotorClient) instance per task run (required
     because Motor binds to the event loop that first touches it; a prior
     ``asyncio.run()`` call closes that loop so any Motor client created
     outside this task is unusable).
  3. Compile the LangGraph agent (``build_photoshoot_graph``), wire in the
     per-run context (``PipelineContext``), and ``ainvoke`` the agent.

The worker is started with ``--concurrency=1`` so only ONE photoshoot job
runs per worker, which matches the rate-limit behaviour of SeedDream / KIE
and keeps logs readable.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from celery import Task

from app.worker import celery_app

logger = logging.getLogger("photoshoot_tasks")


# ---------------------------------------------------------------------------
# Task base — hooks failure into the photoshoot document.
# ---------------------------------------------------------------------------

class _PhotoshootTask(Task):
    """Base Celery Task that marks the photoshoot ``failed`` on unhandled errors."""

    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        photoshoot_id = args[0] if args else kwargs.get("photoshoot_id", "unknown")
        logger.error(
            "[task] FAILED — photoshoot_id=%s | task_id=%s | error=%s",
            photoshoot_id, task_id, exc,
        )
        try:
            asyncio.run(_mark_failed(photoshoot_id, str(exc)))
        except Exception as exc2:  # pragma: no cover
            logger.error("[task] on_failure Mongo update failed: %s", exc2)


async def _mark_failed(photoshoot_id: str, error: str) -> None:
    """Set ``status='failed'`` on the photoshoot doc (fresh Motor client)."""
    from app.config import settings
    from app.database import make_motor_client

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


# ---------------------------------------------------------------------------
# Agentic runner — compile the LangGraph and invoke it.
# ---------------------------------------------------------------------------

async def _run_agentic_pipeline(photoshoot_id: str, req: dict) -> None:
    """Run the photoshoot LangGraph agent with a fresh Motor client.

    Delegates to ``run_photoshoot_job`` which internally compiles and
    ``ainvoke``s the ``build_photoshoot_graph()`` LangGraph.
    """
    from app.database import make_motor_client
    from app.services.photoshoot_service import run_photoshoot_job

    logger.info("[task] Compiling + running LangGraph agent — photoshoot_id=%s", photoshoot_id)
    client = make_motor_client()
    try:
        await run_photoshoot_job(photoshoot_id, req, motor_client=client)
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Celery entry point
# ---------------------------------------------------------------------------

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
    """Drive one photoshoot job via the agentic LangGraph pipeline.

    Poses are streamed into Mongo (``photoshoots.output_images``) and the
    user's credits are deducted incrementally as each pose completes its
    Stage-2 (R2 + upscale) work — consumers see progress live and do NOT
    wait for every pose before seeing results.
    """
    logger.info("[task] Starting photoshoot task — photoshoot_id=%s", photoshoot_id)
    try:
        asyncio.run(_run_agentic_pipeline(photoshoot_id, req))
        logger.info("[task] Completed — photoshoot_id=%s", photoshoot_id)
        return {"photoshoot_id": photoshoot_id, "status": "completed"}
    except Exception as exc:
        logger.error("[task] Error — photoshoot_id=%s | %s", photoshoot_id, exc)
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Queue introspection helper (used by diagnostics endpoints).
# ---------------------------------------------------------------------------

def get_queue_length() -> int:
    """Return number of photoshoot jobs currently waiting in the queue."""
    try:
        import redis as redis_lib
        from app.config import settings
        r = redis_lib.from_url(settings.REDIS_URL)
        return r.llen("photoshoots")
    except Exception:
        return -1  # Redis unavailable — return -1 to signal unknown
