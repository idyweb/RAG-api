"""
documents schemas.

TODO: Define Pydantic models for input/output
"""

from datetime import datetime
from typing import Optional, Any, List
from pydantic import BaseModel, HttpUrl, Field, field_validator


class DocumentCreate(BaseModel):
    """Input schema for document ingestion."""

    model_config = {"extra": "forbid"}

    title: str = Field(..., max_length=500, description="Title of the document")
    department: str = Field(..., max_length=100, description="Target department")
    doc_type: str = Field(..., max_length=50, description="Document type (policy, guide, etc)")
    source_url: HttpUrl | None = Field(None, description="Optional source URL")
    content: str = Field(..., description="The raw textual content to be ingested and chunked")
    content_format: str = Field(
        default="markdown",
        pattern="^(text|markdown)$",
        description="Content format: 'text' for plain text, 'markdown' for structured Markdown",
    )
    allowed_departments: List[str] | None = Field(
        default=None,
        description=(
            "Departments that can access this document. "
            "If omitted, defaults to [department] (owner only). "
            "Example: ['HR', 'Finance', 'Operations'] for cross-department policies."
        ),
    )

    @field_validator("allowed_departments")
    @classmethod
    def validate_allowed_departments(cls, v: List[str] | None) -> List[str] | None:
        """Ensure no empty strings or duplicates."""
        if v is not None:
            cleaned = list(dict.fromkeys(d.strip() for d in v if d.strip()))
            if not cleaned:
                return None
            return cleaned
        return v


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


class UpdatePermissionsRequest(BaseModel):
    """Input schema for updating document access permissions."""

    model_config = {"extra": "forbid"}

    allowed_departments: List[str] = Field(
        ...,
        min_length=1,
        description=(
            "New list of departments that can access this document. "
            "Example: ['HR', 'Finance', 'All']. Must include at least one department."
        ),
    )

    @field_validator("allowed_departments")
    @classmethod
    def validate_allowed_departments(cls, v: List[str]) -> List[str]:
        """Ensure no empty strings or duplicates."""
        cleaned = list(dict.fromkeys(d.strip() for d in v if d.strip()))
        if not cleaned:
            raise ValueError("allowed_departments must contain at least one valid department")
        return cleaned


class UpdatePermissionsResponse(BaseModel):
    """Output schema after updating document permissions."""
    id: str
    title: str
    department: str
    allowed_departments: List[str]
    vectors_updated: int = Field(description="Number of Pinecone vectors whose metadata was updated")
