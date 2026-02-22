"""
Documents router.

Handles document ingestion. No logic here — calls services.
"""

from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from api.apps.auth.services import verify_user
from api.apps.documents.schemas import DocumentCreate, DocumentResponse
from api.apps.documents.services import ingest_document
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
    Ingest a document (with chunking + embedding) into the RAG system.

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


@router.post("/ingest/pdf", status_code=201)
async def ingest_pdf(
    title: str = Form(..., max_length=500, description="Title of the document"),
    department: str = Form(..., max_length=100, description="Target department"),
    doc_type: str = Form(..., max_length=50, description="Document type (policy, guide, etc)"),
    source_url: str = Form(None, description="Optional source URL"),
    file: UploadFile = File(..., description="PDF file to parse and ingest"),
    user: dict = Depends(verify_user),
    session: AsyncSession = Depends(get_session),
    vector_store: VectorStore = Depends(get_vector_store),
):
    """
    Ingest a PDF document directly.
    Extracts text from the PDF and runs the standard chunking + embedding process.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported by this endpoint.")
    
    # Extract text from the uploaded PDF
    extracted_text = extract_text_from_pdf(file.file)
    
    if not extracted_text:
        raise HTTPException(status_code=400, detail="No readable text found in the PDF.")
        
    # Construct DocumentCreate model
    data = DocumentCreate(
        title=title,
        department=department,
        doc_type=doc_type,
        source_url=source_url,
        content=extracted_text
    )
    
    result = await ingest_document(
        session=session,
        vector_store=vector_store,
        data=data,
        user_department=user["department"],
    )
    
    return success_response(
        status_code=201,
        message="PDF ingested and processed successfully",
        data=result.model_dump(),
    )

