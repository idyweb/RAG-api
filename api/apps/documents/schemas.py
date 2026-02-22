"""
documents schemas.

TODO: Define Pydantic models for input/output
"""

from datetime import datetime
from pydantic import BaseModel, HttpUrl, Field


class DocumentCreate(BaseModel):
    title: str = Field(..., max_length=500, description="Title of the document")
    department: str = Field(..., max_length=100, description="Target department")
    doc_type: str = Field(..., max_length=50, description="Document type (policy, guide, etc)")
    source_url: HttpUrl | None = Field(None, description="Optional source URL")
    content: str = Field(..., description="The raw textual content to be ingested and chunked")


class DocumentResponse(BaseModel):
    id: str
    title: str
    department: str
    version: int
    chunk_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True
