"""
Celery application — queue management for photoshoot jobs.

Start the worker with autoscaling (recommended for production):

    celery -A app.worker worker \\
        -Q photoshoots \\
        --autoscale=6,2 \\
        --prefetch-multiplier=1 \\
        --max-tasks-per-child=50 \\
        --max-memory-per-child=900000 \\
        --loglevel=info

    --autoscale=MAX,MIN
      MIN=2   — always keep 2 worker processes warm (fast job pickup)
      MAX=6   — scale up to 6 parallel photoshoot jobs under load

    --max-memory-per-child=900000  (kB, ≈900 MB per process)
      Worker recycles itself AFTER a task if RSS crosses the cap.  Prevents
      the kernel OOM killer from sending SIGKILL (which cascades a
      WorkerLostError on every sibling).

    Each photoshoot task streams up to 8 × 8K PNG bitmaps during Stage-2
    (upscale → encode → upload), so per-task peak RSS is ~700 MB – 1.3 GB.
    This is NOT a pure I/O-bound workload — concurrency must be sized to
    RAM, not just CPU count.

Recommended profile by droplet size:

    droplet         MAX,MIN   --max-memory-per-child (kB)
    ──────────      ───────   ────────────────────────────
    2 vCPU/4 GB     3,1       900000
    4 vCPU/8 GB     6,2       900000   (recommended default)
    8 vCPU/16 GB    12,4      900000
    16 vCPU/32 GB   20,6      1100000

Cross-worker rate limiting:

    KIE.ai enforces a 20-req / 10-s account-level limit on createTask.
    ``app.services.kie_rate_limiter`` implements a Redis sliding-window
    limiter that every Celery worker (and the FastAPI process) consults
    before POSTing /createTask.  Values are env-tunable:
        KIE_RATE_LIMIT_REQUESTS=18
        KIE_RATE_LIMIT_WINDOW_S=10
    Vertex (nano-banana-2, nano-banana-pro) and Evolink have no published
    account-wide limits, so they are bounded only by per-job semaphores
    (``PHOTOSHOOT_VERTEX_*_CONCURRENCY`` / ``PHOTOSHOOT_EVOLINK_CONCURRENCY``)
    — see app/config.py.

For monitoring (optional):
    celery -A app.worker flower --port=5555 --basic_auth=admin:yourpassword

Modal auto-scaling is handled on Modal's side — FashionRealismT4 scales up to
10 containers and FashionRealismL4 scales up to 5 containers automatically
when multiple photoshoot jobs call enhance.remote() concurrently.
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
    # Route all photoshoot tasks to a dedicated queue
    task_routes={
        "app.tasks.photoshoot_tasks.*": {"queue": "photoshoots"},
    },

    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Keep results for 24 hours so status can be polled
    result_expires=86400,

    # Acknowledge task only after it completes (safe retry on worker crash)
    task_acks_late=True,

    # Each worker process takes one task at a time — prevents a single process
    # from hoarding multiple tasks from the queue while others are idle.
    worker_prefetch_multiplier=1,

    # Timezone
    timezone="UTC",
    enable_utc=True,
)
