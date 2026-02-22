"""
Documents router.

Handles document ingestion. No logic here — calls services.
PDF ingestion is dispatched as a Celery background task.
"""

from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from celery.result import AsyncResult

from api.apps.auth.services import verify_user
from api.apps.documents.schemas import (
    DocumentCreate,
    DocumentResponse,
    IngestAcceptedResponse,
    TaskStatusResponse,
)
from api.apps.documents.services import ingest_document
from api.apps.documents.tasks import ingest_document_task
from api.core.celery_app import celery_app
from api.core.dependencies import get_vector_store
from api.core.vector_store import VectorStore
from api.db.session import get_session
from api.utils.responses import success_response
from api.utils.logger import get_logger
from api.utils.pdf_parser import extract_text_from_pdf
from fastapi.exceptions import HTTPException

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])


@router.post("/ingest", status_code=201)
async def ingest(
    data: DocumentCreate,
    user: dict = Depends(verify_user),
    session: AsyncSession = Depends(get_session),
    vector_store: VectorStore = Depends(get_vector_store),
):
    """
    Ingest a raw text document (synchronous — for small payloads).

    Security:
    - User can only upload to their own department.
    - IT users can upload to any department.
    """
    result = await ingest_document(
        session=session,
        vector_store=vector_store,
        data=data,
        user_department=user["department"],
    )
    return success_response(
        status_code=201,
        message="Document ingested successfully",
        data=result.model_dump(),
    )


@router.post("/ingest/pdf", status_code=202, response_model=IngestAcceptedResponse)
async def ingest_pdf(
    title: str = Form(..., max_length=500, description="Title of the document"),
    department: str = Form(..., max_length=100, description="Target department"),
    doc_type: str = Form(..., max_length=50, description="Document type (policy, guide, etc)"),
    source_url: str = Form(None, description="Optional source URL"),
    file: UploadFile = File(..., description="PDF file to parse and ingest"),
    user: dict = Depends(verify_user),
):
    """
    Ingest a PDF document as a background task.

    1. Extract text from the PDF immediately (fast)
    2. Dispatch chunking + embedding + Pinecone upsert to Celery worker
    3. Return 202 Accepted with a task_id for status polling

    Poll `GET /api/v1/documents/tasks/{task_id}` to check progress.
    """
    # Guard: file type
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported by this endpoint.")

    # Extract text synchronously (fast, CPU-bound, no network)
    extracted_text = extract_text_from_pdf(file.file)

    if not extracted_text:
        raise HTTPException(status_code=400, detail="No readable text found in the PDF.")

    # Dispatch to Celery worker
    task = ingest_document_task.delay(
        title=title,
        department=department,
        doc_type=doc_type,
        content=extracted_text,
        user_department=user["department"],
        source_url=source_url,
    )

    logger.info(f"PDF ingestion dispatched: task_id={task.id}, title='{title}'")

    return IngestAcceptedResponse(
        task_id=task.id,
        message=f"PDF '{title}' accepted for processing. Poll /tasks/{task.id} for status.",
    )


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    user: dict = Depends(verify_user),
):
    """
    Poll Celery task status.

    Returns:
    - PENDING: Task is waiting in the queue
    - STARTED: Worker picked it up
    - SUCCESS: Done — result contains document metadata
    - FAILURE: Failed — error contains the exception message
    - RETRY: Task is retrying after a transient failure
    """
    result = AsyncResult(task_id, app=celery_app)

    response = TaskStatusResponse(
        task_id=task_id,
        status=result.status,
    )

    if result.successful():
        response.result = result.result
    elif result.failed():
        response.error = str(result.result)

    return response
