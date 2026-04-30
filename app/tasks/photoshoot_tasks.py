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
photoshoot document ``completed`` / ``failed``.

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
import uuid
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


# ---------------------------------------------------------------------------
# Catalogue colorway — KIE recolor + R2 in worker, then same agentic pipeline.
# ---------------------------------------------------------------------------


async def _catalogue_recolor_then_run_pipeline(photoshoot_id: str) -> None:
    """Recolor flat-lays from ``catalogue_recolor``, persist URLs, run ``run_photoshoot_job``."""
    from app.config import settings
    from app.database import make_motor_client
    from app.services.fabric_service import change_garment_upper_lower_colors
    from app.services.photoshoot_service import run_photoshoot_job
    from app.services.r2_service import upload_bytes_to_r2

    client = make_motor_client()
    try:
        col = client[settings.MONGO_DB_NAME]["photoshoots"]
        doc = await col.find_one({"photoshoot_id": photoshoot_id})
        if not doc:
            raise ValueError(f"Photoshoot not found: {photoshoot_id}")

        cr = doc.get("catalogue_recolor")
        if not cr:
            st = doc.get("status")
            if st in ("processing", "completed"):
                logger.info(
                    "[catalogue-recolor] skip duplicate — id=%s status=%s",
                    photoshoot_id,
                    st,
                )
                return
            raise ValueError(
                f"Photoshoot {photoshoot_id} has no catalogue_recolor payload "
                f"(status={doc.get('status')!r})"
            )

        ref_front = (cr.get("ref_front") or "").strip()
        ref_back = (cr.get("ref_back") or "").strip()
        u = cr.get("upper_garment_color")
        l = cr.get("lower_garment_color")
        if l is None:
            l = u
        if not ref_front or u is None:
            raise ValueError("catalogue_recolor requires ref_front and upper_garment_color")

        u_tag = str(u).lstrip("#")
        l_tag = str(l).lstrip("#")
        tag = f"u{u_tag}_l{l_tag}_{uuid.uuid4().hex[:8]}"
        catalogue_id = doc.get("batch_photoshoot_id") or "catalogue"
        idx = int(doc.get("batch_index") if doc.get("batch_index") is not None else 0)

        front_bytes = await change_garment_upper_lower_colors(ref_front, u, l)
        f_key = f"photoshoots/catalogue/{catalogue_id}/{idx}/front_{tag}.png"
        use_front = await upload_bytes_to_r2(front_bytes, f_key, content_type="image/png")

        if ref_back:
            back_bytes = await change_garment_upper_lower_colors(ref_back, u, l)
            b_key = f"photoshoots/catalogue/{catalogue_id}/{idx}/back_{tag}.png"
            use_back = await upload_bytes_to_r2(back_bytes, b_key, content_type="image/png")
        else:
            use_back = ""

        user_id = doc.get("user_id") or ""
        ip = dict(doc.get("input_parameter") or {})
        ip["front_garment_image"] = use_front
        ip["back_garment_image"] = use_back or ""

        await col.update_one(
            {"photoshoot_id": photoshoot_id},
            {
                "$set": {
                    "input_parameter": ip,
                    "updated_at": datetime.now(timezone.utc),
                },
                "$unset": {"catalogue_recolor": ""},
            },
        )

        req = {**ip, "user_id": user_id}
        await run_photoshoot_job(photoshoot_id, req, motor_client=client)
    finally:
        client.close()


@celery_app.task(
    bind=True,
    base=_PhotoshootTask,
    name="app.tasks.photoshoot_tasks.run_catalogue_recolor_then_photoshoot_task",
    queue="photoshoots",
    max_retries=2,
    default_retry_delay=60,
    acks_late=True,
)
def run_catalogue_recolor_then_photoshoot_task(self, photoshoot_id: str) -> dict:
    """KIE upper/lower recolor + R2 upload, then the standard LangGraph photoshoot job."""
    logger.info("[catalogue-recolor] Starting — photoshoot_id=%s", photoshoot_id)
    try:
        asyncio.run(_catalogue_recolor_then_run_pipeline(photoshoot_id))
        logger.info("[catalogue-recolor] Completed — photoshoot_id=%s", photoshoot_id)
        return {"photoshoot_id": photoshoot_id, "status": "completed"}
    except Exception as exc:
        logger.error("[catalogue-recolor] Error — photoshoot_id=%s | %s", photoshoot_id, exc)
        raise self.retry(exc=exc)
