"""
documents schemas.

TODO: Define Pydantic models for input/output
"""

from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, HttpUrl, Field


class DocumentCreate(BaseModel):
    """Input schema for document ingestion."""

    model_config = {"extra": "forbid"}

    title: str = Field(..., max_length=500, description="Title of the document")
    department: str = Field(..., max_length=100, description="Target department")
    doc_type: str = Field(..., max_length=50, description="Document type (policy, guide, etc)")
    source_url: HttpUrl | None = Field(None, description="Optional source URL")
    content: str = Field(..., description="The raw textual content to be ingested and chunked")
    content_format: str = Field(
        default="text",
        pattern="^(text|markdown)$",
        description="Content format: 'text' for plain text, 'markdown' for structured Markdown",
    )


class DocumentResponse(BaseModel):
    """Output schema for document ingestion result."""

    model_config = {"from_attributes": True, "extra": "forbid"}

    id: str
    title: str
    department: str
    version: int
    chunk_count: int
    content_format: str
    parent_chunk_count: int = 0
    child_chunk_count: int = 0
    created_at: datetime


class IngestAcceptedResponse(BaseModel):
    """Returned when a document is accepted for background ingestion."""
    task_id: str = Field(..., description="Celery task ID for status polling")
    message: str = Field(default="Document accepted for processing")


class TaskStatusResponse(BaseModel):
    """Response schema for polling Celery task status."""
    task_id: str
    status: str = Field(..., description="PENDING | STARTED | SUCCESS | FAILURE | RETRY")
    result: Optional[Any] = Field(None, description="Task result on SUCCESS")
    error: Optional[str] = Field(None, description="Error message on FAILURE")
