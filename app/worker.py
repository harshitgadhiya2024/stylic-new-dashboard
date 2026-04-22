"""
Celery application — queue management for photoshoot jobs.

Production-safe run command (used by the systemd unit — see DEPLOYMENT.md §10):

    celery -A app.worker worker \
      -Q photoshoots \
      --concurrency=3 \
      --prefetch-multiplier=1 \
      --max-tasks-per-child=50 \
      --max-memory-per-child=1500000 \
      --loglevel=info

Why fixed ``--concurrency=3`` instead of ``--autoscale=3,1``
------------------------------------------------------------
Each photoshoot task fan-outs 8 poses via ``asyncio.gather``. The peak RSS
of a hot worker during Stage-2 PIL 8K → 4K/2K/1K encoding is ~1.2 GB.
Celery's autoscaler decides to spawn more children based on queue depth
WITHOUT knowing current memory use, so on a 16 GB box a burst of 5 big
photoshoots can OOM-kill the worker mid-job and lose progress. Fixed
concurrency + ``--max-memory-per-child`` is the safe production default.

``--max-memory-per-child=1500000`` (KB, ~1.5 GB)
    Celery gracefully RECYCLES a worker child after any task bumps its
    RSS over this threshold. The child finishes its current task first
    (no job is lost), then exits and is replaced. This is the single
    most important knob for keeping the box OOM-free under sustained
    load.

``--prefetch-multiplier=1``
    Each child reserves exactly one task at a time — never hoards the
    queue while other children are idle.

``--max-tasks-per-child=50``
    Belt-and-braces recycle to flush any slow memory leaks from PIL,
    httpx buffer pools, google-genai streams, etc.

KIE rate limiting (20 createTask / 10 s account cap) is enforced
cross-worker by ``app.services.kie_rate_limiter`` — no need to cap
concurrency further for that reason.

Modal GPU auto-scaling is handled on Modal's side.
"""

from celery import Celery

from app.config import settings

celery_app = Celery(
    "photoshoot_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks.photoshoot_tasks"],
)

celery_app.conf.update(
    # Route all photoshoot tasks to a dedicated queue.
    task_routes={
        "app.tasks.photoshoot_tasks.*": {"queue": "photoshoots"},
    },

    # Serialization.
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Keep results for 24 hours so status can be polled.
    result_expires=86400,

    # Acknowledge tasks only after completion (safe retry on worker crash).
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Each child takes one task at a time — prevents one child hoarding
    # the queue while others are idle (critical for long-running jobs).
    worker_prefetch_multiplier=1,

    # Recycle children periodically to cap RSS growth from PIL / httpx.
    worker_max_tasks_per_child=50,
    # 1.5 GB per child — hard ceiling to prevent OOM on a 16 GB box running
    # 3 parallel photoshoots (3 × 1.5 GB = 4.5 GB, with 11+ GB free for the
    # OS/API/Redis/Nginx). Celery recycles the child AFTER the current task
    # completes, so no job is ever lost.
    worker_max_memory_per_child=1_500_000,  # kilobytes

    # Hard wall-clock timeouts so one stuck task can never hang a child.
    # soft < hard so the task gets a SoftTimeLimit and can clean up first.
    task_soft_time_limit=60 * 30,   # 30 min
    task_time_limit=60 * 35,        # 35 min

    # Broker connection resilience.
    broker_connection_retry_on_startup=True,

    # Timezone.
    timezone="UTC",
    enable_utc=True,
)
