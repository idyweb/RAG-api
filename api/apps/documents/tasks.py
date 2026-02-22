"""
Celery tasks for document ingestion.

Runs the heavy embedding + Pinecone upsert work in a background worker,
keeping the API responsive.
"""

import asyncio
import time
from celery import Task

from api.core.celery_app import celery_app
from api.apps.documents.schemas import DocumentCreate
from api.utils.logger import get_logger
from api.utils.metrics import ingestion_count, ingestion_latency

logger = get_logger(__name__)


class IngestionTask(Task):
    """
    Custom base task with lazy-loaded async resources.

    Why: Celery workers are sync processes. We create a dedicated
    event loop and DB/vector store sessions per-worker, not per-task.
    """
    _loop = None
    _session_factory = None
    _vector_store = None

    @property
    def loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    @property
    def session_factory(self):
        if self._session_factory is None:
            from api.db.database import async_session_factory
            self._session_factory = async_session_factory
        return self._session_factory

    @property
    def vector_store(self):
        if self._vector_store is None:
            from api.core.dependencies import get_vector_store
            self._vector_store = get_vector_store()
        return self._vector_store


@celery_app.task(
    base=IngestionTask,
    bind=True,
    name="ingest_document_task",
    max_retries=3,
    default_retry_delay=10,
    acks_late=True,
)
def ingest_document_task(
    self: IngestionTask,
    title: str,
    department: str,
    doc_type: str,
    content: str,
    user_department: str,
    source_url: str | None = None,
) -> dict:
    """
    Background task: ingest a document with chunking + embedding + Pinecone upsert.

    Args:
        title: Document title
        department: Target department
        doc_type: Document type (policy, guide, catalog, etc.)
        content: Raw text content (already extracted from PDF)
        user_department: Uploading user's department (for permission check)
        source_url: Optional source URL

    Returns:
        Dict with document ID, title, department, version, chunk_count
    """
    logger.info(f"[CELERY] Starting ingestion: '{title}' for dept={department}")
    start_time = time.time()

    try:
        result = self.loop.run_until_complete(
            _async_ingest(
                self,
                title=title,
                department=department,
                doc_type=doc_type,
                content=content,
                user_department=user_department,
                source_url=source_url,
            )
        )

        latency = time.time() - start_time
        ingestion_count.labels(department=department, status="success").inc()
        ingestion_latency.labels(department=department).observe(latency)

        logger.info(f"[CELERY] Ingestion complete: '{title}' in {latency:.2f}s")
        return result

    except Exception as exc:
        latency = time.time() - start_time
        ingestion_count.labels(department=department, status="failure").inc()
        ingestion_latency.labels(department=department).observe(latency)

        logger.error(f"[CELERY] Ingestion failed: {exc}", exc_info=True)
        raise self.retry(exc=exc)


async def _async_ingest(
    task: IngestionTask,
    title: str,
    department: str,
    doc_type: str,
    content: str,
    user_department: str,
    source_url: str | None,
) -> dict:
    """
    Async wrapper that creates a fresh session per task execution.

    Why a fresh session: Each Celery task retry or execution must
    have its own session to avoid stale connections.
    """
    from api.apps.documents.services import ingest_document

    data = DocumentCreate(
        title=title,
        department=department,
        doc_type=doc_type,
        source_url=source_url,
        content=content,
    )

    async with task.session_factory() as session:
        result = await ingest_document(
            session=session,
            vector_store=task.vector_store,
            data=data,
            user_department=user_department,
        )

    return result.model_dump(mode="json")
