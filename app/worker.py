"""
Celery application — queue management for photoshoot jobs.

Start the worker (from the project root):
    celery -A app.worker worker --concurrency=1 -Q photoshoots --loglevel=info

concurrency=1 ensures only ONE photoshoot job runs at a time, so SeedDream and
Modal are never hit with multiple parallel requests from this worker.

For monitoring (optional):
    celery -A app.worker flower --port=5555
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

    # One task per worker process at a time (no prefetch)
    worker_prefetch_multiplier=1,

    # Timezone
    timezone="UTC",
    enable_utc=True,
)
