"""
documents models.

TODO: Define SQLAlchemy ORM models
All models inherit from BaseModel
"""

from sqlalchemy import String, Text, Integer, Boolean, ForeignKey, Index
from sqlalchemy.dialects.postgresql import ARRAY
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
    
    # Relationship to chunks
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )
    
    __table_args__ = (
        Index('idx_dept_active', 'department', 'is_active'),
    )


class DocumentChunk(BaseModel):
    """
    Individual chunk of a document for RAG retrieval.
    """
    __tablename__ = "document_chunks"

    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    vector_id: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    allowed_departments: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    
    # Relationship
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    
    __table_args__ = (
        Index('idx_doc_chunk', 'document_id', 'chunk_index'),
    )
