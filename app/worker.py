"""
Celery application — queue management for photoshoot jobs.

Start the worker with autoscaling (recommended for production):

    celery -A app.worker worker \\
        -Q photoshoots \\
        --autoscale=16,4 \\
        --prefetch-multiplier=1 \\
        --max-tasks-per-child=100 \\
        --loglevel=info

    --autoscale=MAX,MIN
      MIN=4   — always keep 4 worker processes warm (fast job pickup)
      MAX=16  — scale up to 16 parallel photoshoot jobs under load

    Photoshoot tasks are I/O-bound (they spend >90% of wall time waiting for
    KIE / Vertex / Evolink / Modal), so concurrency can far exceed CPU count.

Recommended --autoscale profile by droplet size:

    droplet         MAX,MIN        notes
    ──────────     ─────────       ──────────────────────────────────────
    2 vCPU/4 GB    8,2             safe entry-level production
    4 vCPU/8 GB    16,4            recommended default
    8 vCPU/16 GB   32,6            high-throughput

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
