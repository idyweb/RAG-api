"""
Documents router.

Handles document ingestion. No logic here — calls services.
PDF ingestion is dispatched as a Celery background task.
"""

from fastapi import APIRouter, Depends, UploadFile, File, Form
from celery.result import AsyncResult
from typing import List

from sqlalchemy.ext.asyncio import AsyncSession

from api.core.dependencies import verify_user, get_vector_store, get_cache
from api.db.session import get_session
from api.apps.documents.schemas import (
    IngestAcceptedResponse,
    TaskStatusResponse,
    UpdatePermissionsRequest,
    UpdatePermissionsResponse,
)
from api.apps.documents.tasks import ingest_document_task
from api.apps.documents.services import update_document_permissions
from api.core.celery_app import celery_app
from api.utils.logger import get_logger
from fastapi.exceptions import HTTPException

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])


@router.post("/ingest/pdf", status_code=202, response_model=IngestAcceptedResponse)
async def ingest_pdf(
    title: str = Form(..., max_length=500, description="Title of the document"),
    department: str = Form(..., max_length=100, description="Owning department"),
    doc_type: str = Form(..., max_length=50, description="Document type (policy, guide, etc)"),
    source_url: str = Form(None, description="Optional source URL"),
    allowed_departments: str = Form(
        None,
        description=(
            "Comma-separated list of departments that can access this document. "
            "Example: 'HR,Sales,Finance'. If omitted, only the owning department can access it."
        ),
    ),
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

    # Guard: file size — read into memory once and check against limit
    from api.config.settings import settings
    file_bytes = await file.read()
    if len(file_bytes) > settings.MAX_DOCUMENT_SIZE:
        max_mb = settings.MAX_DOCUMENT_SIZE // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {max_mb}MB.",
        )

    # Parse allowed_departments from comma-separated string
    dept_list: List[str] | None = None
    if allowed_departments:
        dept_list = [d.strip() for d in allowed_departments.split(",") if d.strip()]
        if not dept_list:
            dept_list = None

    # Save PDF to shared volume so the Celery worker can extract text.
    # This avoids blocking the API for minutes during OCR.
    import uuid as _uuid
    from pathlib import Path

    upload_dir = Path("/app/tmp/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    pdf_filename = f"{_uuid.uuid4().hex}.pdf"
    pdf_path = upload_dir / pdf_filename
    pdf_path.write_bytes(file_bytes)

    # Dispatch to Celery worker — PDF extraction happens in the worker
    task = ingest_document_task.delay(
        title=title,
        department=department,
        doc_type=doc_type,
        pdf_path=str(pdf_path),
        user_department=user["department"],
        source_url=source_url,
        allowed_departments=dept_list,
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


@router.patch("/{document_id}/permissions", response_model=UpdatePermissionsResponse)
async def update_permissions(
    document_id: str,
    payload: UpdatePermissionsRequest,
    user: dict = Depends(verify_user),
    session: AsyncSession = Depends(get_session),
    vector_store=Depends(get_vector_store),
    cache=Depends(get_cache),
):
    """
    Update which departments can access a document.

    Updates permissions across all three layers:
    - PostgreSQL (document record)
    - Pinecone (vector metadata for search filtering)
    - Redis (cache invalidation for old and new departments)

    Only the owning department or IT can update permissions.
    """
    result = await update_document_permissions(
        session=session,
        vector_store=vector_store,
        cache=cache,
        document_id=document_id,
        allowed_departments=payload.allowed_departments,
        user_department=user["department"],
    )
    return UpdatePermissionsResponse(**result)
