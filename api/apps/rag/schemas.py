"""
RAG query schemas.

Input/output contracts for RAG queries.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum


class QueryRequest(BaseModel):
    """Request schema for RAG query."""
    
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversational memory."
    )
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's question"
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of documents to retrieve"
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (hallucination prevention)"
    )


class SourceDocument(BaseModel):
    """Source document metadata."""
    
    document_id: str
    title: str
    department: str
    chunk_index: int
    doc_type: str
    relevance_score: float


class ChatRole(str, Enum):
    """Role for chat messages."""
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Message schema for chat history."""
    role: ChatRole
    content: str


class QueryResponse(BaseModel):
    """Response schema for RAG query."""
    
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceDocument] = Field(
        default_factory=list,
        description="Source documents used"
    )
    confidence: str = Field(
        ...,
        description="Confidence level: 'high' or 'low'"
    )
    latency_ms: float = Field(..., description="Query latency in milliseconds")
    cached: bool = Field(..., description="Whether result was cached")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Cowbell Chocolate 400g costs ₦2,500/tin with a distributor price of ₦2,200/tin.",
                "sources": [
                    {
                        "document_id": "abc-123",
                        "title": "Product Catalog 2024",
                        "department": "Sales",
                        "chunk_index": 0,
                        "doc_type": "catalog",
                        "relevance_score": 0.89
                    }
                ],
                "confidence": "high",
                "latency_ms": 245.3,
                "cached": False
            }
        }