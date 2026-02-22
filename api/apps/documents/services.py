"""
documents services.

Business logic for document ingestion with chunking, embedding, and versioning.
"""

from typing import List
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from api.apps.documents.models import Document, DocumentChunk
from api.apps.documents.schemas import DocumentCreate, DocumentResponse
from api.core.embeddings import generate_embeddings, generate_embeddings_batch
from api.core.vector_store import VectorStore
from api.utils.logger import get_logger
from api.utils.exceptions import InvalidDepartmentError, PermissionDeniedException

logger = get_logger(__name__)

# Valid coragem departments (from JD)
VALID_DEPARTMENTS = ["Sales", "HR", "Finance", "Operations", "Manufacturing", "IT"]


async def ingest_document(
    session: AsyncSession,
    vector_store: VectorStore,
    data: DocumentCreate,
    user_department: str
) -> DocumentResponse:
    """
    Ingest a new document with chunking, embedding, and versioning.
    
    Flow:
    1. Validate department and permissions
    2. Get next version number
    3. Create document record (uncommitted)
    4. Chunk content
    5. Generate embeddings for each chunk
    6. Store in vector DB
    7. Create chunk records (uncommitted)
    8. Deactivate old versions if needed
    9. Commit transaction
    
    Args:
        session: Async DB session
        vector_store: Vector DB client
        data: Document creation request
        user_department: Department of user performing upload
        
    Returns:
        DocumentResponse with created document metadata
        
    Raises:
        InvalidDepartmentError: If department doesn't exist
        PermissionDeniedException: If user can't upload to target dept
    """
    # Guard clause: Validate department
    if data.department not in VALID_DEPARTMENTS:
        raise InvalidDepartmentError(f"Invalid department: {data.department}")
    
    # Guard clause: Check user permission (admins can upload to any dept)
    if user_department != "IT" and user_department != data.department:
        raise PermissionDeniedException(
            detail=f"Cannot upload to department: {data.department}"
        )
    
    logger.info(f"Ingesting document: {data.title} for dept: {data.department}")
    
    # 1. Check for existing document (versioning logic)
    version = await _get_next_version(session, data.title, data.department)
    
    # 2. Create document record (let BaseModel generate UUID)
    document = await Document.create(
        db=session,
        commit=False,  # Don't commit yet (atomic transaction)
        title=data.title,
        department=data.department,
        doc_type=data.doc_type,
        source_url=str(data.source_url) if data.source_url else None,
        version=version,
        is_active=True
    )
    
    # Flush to get the generated document.id
    await session.flush()
    doc_id_str = str(document.id)
    
    # 3. Chunk content
    chunks = _chunk_text(data.content, chunk_size=500, overlap=50)
    logger.info(f"Created {len(chunks)} chunks for document: {doc_id_str}")
    
    # 4. Generate embeddings and store in vector DB
    chunk_records = []
    
    # 5. Batch generate embeddings for 10x speedup
    embeddings = await generate_embeddings_batch(chunks)
    
    for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        # Store in vector DB
        vector_id = f"{doc_id_str}_chunk_{idx}"
        await vector_store.upsert(
            id=vector_id,
            vector=embedding,
            metadata={
                "content": chunk_text,
                "document_id": doc_id_str,
                "department": data.department,  # CRITICAL for filtering
                "chunk_index": idx,
                "title": data.title,
                "doc_type": data.doc_type,
                "version": version,
                "is_active": True
            }
        )
        
        # Prepare chunk record for DB
        chunk_records.append({
            "document_id": document.id,  # UUID object, not string
            "content": chunk_text,
            "chunk_index": idx,
            "vector_id": vector_id,
            "token_count": len(chunk_text.split())
        })
    
    # 5. Bulk create chunks (BaseModel.create_many)
    await DocumentChunk.create_many(
        db=session,
        items=chunk_records,
        commit=False
    )
    
    # 6. Deactivate old versions if this isn't v1
    if version > 1:
        await _deactivate_old_versions(
            session=session,
            vector_store=vector_store,
            title=data.title,
            department=data.department,
            current_doc_id=doc_id_str
        )
    
    # 7. Commit the transaction
    await session.commit()
    await session.refresh(document)  # Get updated timestamp
    
    logger.info(f"Successfully ingested document: {doc_id_str}, version: {version}")
    
    return DocumentResponse(
        id=doc_id_str,
        title=document.title,
        department=document.department,
        version=version,
        chunk_count=len(chunk_records),
        created_at=document.created_at
    )


async def _get_next_version(
    session: AsyncSession,
    title: str,
    department: str
) -> int:
    """
    Get next version number for a document.
    
    Queries for existing documents with same title + department,
    orders by version DESC, and increments highest version.
    
    Args:
        session: DB session
        title: Document title
        department: Department name
        
    Returns:
        Next version number (1 if no existing docs)
        
    Complexity: O(log n) with index on (title, department, version)
    """
    documents = await Document.find_many(
        db=session,
        limit=1,
        filters={"title": title, "department": department},
        order_by="version",
        order_desc=True
    )
    
    return (documents[0].version + 1) if documents else 1


async def _deactivate_old_versions(
    session: AsyncSession,
    vector_store: VectorStore,
    title: str,
    department: str,
    current_doc_id: str
) -> None:
    """
    Soft delete old versions of a document.
    Marks as inactive in both relational DB and vector DB.
    
    Note: This is a simplified implementation. In production, you'd want
    a more efficient bulk update that excludes current_doc_id in the query itself.
    
    Args:
        session: DB session
        vector_store: Vector DB client
        title: Document title
        department: Department name
        current_doc_id: ID of newly created document (don't deactivate this one)
    """
    # Get all old versions (excluding current)
    # Eager load the chunks to get vector IDs
    from sqlalchemy.orm import selectinload
    old_docs = await Document.find_many(
        db=session,
        filters={
            "title": title,
            "department": department,
            "is_active": True
        },
        limit=100,  # Reasonable limit for versions
        options=[selectinload(Document.chunks)]
    )
    
    vector_ids_to_deactivate = []
    
    # Filter out current document and deactivate the rest
    for doc in old_docs:
        if str(doc.id) != current_doc_id:
            doc.is_active = False
            await doc.save(db=session, commit=False)
            
            # Collect the vector IDs from the relation
            for chunk in doc.chunks:
                vector_ids_to_deactivate.append(chunk.vector_id)
            
    # Update vector DB metadata for old chunks by Exact ID
    if vector_ids_to_deactivate:
        await vector_store.update_metadata_by_ids(
            ids=vector_ids_to_deactivate,
            updates={"is_active": False}
        )


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Why overlap:
    - Prevents information loss at boundaries
    - Improves retrieval accuracy
    - Standard RAG practice
    
    Args:
        text: Full document text
        chunk_size: Target chunk size in words (not characters)
        overlap: Overlap between chunks in words
        
    Returns:
        List of text chunks
        
    Complexity: O(n) where n = number of words
    """
    words = text.split()
    chunks = []
    
    # Edge case: Empty text
    if not words:
        return []
    
    # Generate overlapping chunks
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        
        # Move forward by (chunk_size - overlap)
        i += (chunk_size - overlap)
        
        # Stop if we've processed all words
        if i >= len(words):
            break
    
    return chunks
