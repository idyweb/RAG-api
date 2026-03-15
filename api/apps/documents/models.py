"""
documents models.

SQLAlchemy ORM models for document storage with hierarchical chunking support.
Parent-child chunk relationships enable retrieval of broad context after
precise child-chunk vector search.
"""

import uuid

from sqlalchemy import String, Text, Integer, Boolean, ForeignKey, Index
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.db.base_model import BaseModel


class Document(BaseModel):
    """
    Master document record.
    Stores metadata about documents with department-based access control.
    """
    __tablename__ = "documents"

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    department: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    doc_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    allowed_departments: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    content_format: Mapped[str] = mapped_column(
        String(20), default="text", nullable=False
    )
    # Values: "text", "markdown"
    content_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    
    # Relationship to chunks
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan",
        foreign_keys="DocumentChunk.document_id"
    )
    
    __table_args__ = (
        Index('idx_dept_active', 'department', 'is_active'),
    )


class DocumentChunk(BaseModel):
    """
    Individual chunk of a document for RAG retrieval.

    Supports hierarchical parent-child relationships:
    - Parent chunks: Large context windows stored in DB, NOT embedded in vector store.
      Used to provide broad context to the LLM after child search.
    - Child chunks: Small, precise chunks embedded in Pinecone for vector search.
      Each child references its parent via parent_chunk_id.
    """
    __tablename__ = "document_chunks"

    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id"), nullable=False, index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    vector_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True, unique=True
    )
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    allowed_departments: Mapped[list[str] | None] = mapped_column(
        ARRAY(String), nullable=True
    )

    # Hierarchical chunking fields
    is_parent: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, index=True
    )
    parent_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_chunks.id"),
        nullable=True,
        index=True,
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document", back_populates="chunks", foreign_keys=[document_id]
    )
    parent_chunk: Mapped["DocumentChunk | None"] = relationship(
        "DocumentChunk",
        remote_side="DocumentChunk.id",
        foreign_keys=[parent_chunk_id],
        back_populates="child_chunks",
    )
    child_chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk",
        back_populates="parent_chunk",
        foreign_keys=[parent_chunk_id],
    )

    __table_args__ = (
        Index('idx_doc_chunk', 'document_id', 'chunk_index'),
    )
