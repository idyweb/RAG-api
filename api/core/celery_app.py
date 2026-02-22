"""
Celery application initialization.

Uses Redis as both broker and result backend.
"""

from celery import Celery
from api.config.settings import settings


celery_app = Celery(
    "coragem_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["api.apps.documents.tasks"]
)

# ── Configuration ─────────────────────────────────────────────────────────────

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Retry policy for broker connection
    broker_connection_retry_on_startup=True,
    # Results expire after 24 hours
    result_expires=86400,
    # Worker settings
    worker_prefetch_multiplier=1,  # Fair scheduling for long tasks
    task_acks_late=True,           # Acknowledge after completion (crash safety)
    # Task-level defaults
    task_default_retry_delay=10,
    task_max_retries=3,
)
