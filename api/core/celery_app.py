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


# ── Eager initialization ─────────────────────────────────────────────────────
# Pre-warm VectorStore, CacheManager, and Embeddings when the worker starts,
# not on the first task. Eliminates ~30s cold-start penalty.

from celery.signals import worker_process_init


@worker_process_init.connect
def warmup_worker(**kwargs):
    """Pre-initialize external clients so the first task doesn't pay cold-start cost."""
    from api.utils.logger import get_logger
    _logger = get_logger(__name__)

    try:
        from api.core.dependencies import get_vector_store, get_cache
        from api.core.embeddings import get_embedding_service

        get_vector_store()
        get_cache()
        get_embedding_service()
        _logger.info("[CELERY] Worker warm-up complete: VectorStore, Cache, Embeddings ready")
    except Exception as e:
        _logger.warning(f"[CELERY] Worker warm-up failed (will retry on first task): {e}")
