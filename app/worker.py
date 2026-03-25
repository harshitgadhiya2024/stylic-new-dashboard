"""
Celery application — queue management for photoshoot jobs.

Start the worker with autoscaling (recommended for production):
    celery -A app.worker worker -Q photoshoots --autoscale=5,1 --loglevel=info

    --autoscale=MAX,MIN
      MIN=1  — always keep at least 1 worker process alive
      MAX=5  — scale up to 5 parallel photoshoot jobs when queue is deep

    Celery automatically adds worker processes as the queue fills up and
    removes them when the queue drains, so users never wait unnecessarily
    while also not wasting EC2 resources when idle.

Start the worker with fixed concurrency (simpler, good for small instances):
    celery -A app.worker worker -Q photoshoots --concurrency=3 --loglevel=info

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
