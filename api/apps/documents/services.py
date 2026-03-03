"""
documents services.

Business logic for document ingestion with hierarchical chunking, embedding,
and versioning.

Hierarchical Chunking Strategy:
- Markdown content → split by headers into parent chunks (broad context)
- Each parent → subdivided into child chunks (precise search targets)
- Only children are embedded in the vector store
- On query, child matches retrieve parent content for LLM context
- Plain text → falls back to fixed-size word-based chunking (flat, no hierarchy)
"""

from typing import List, Tuple
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from api.apps.documents.models import Document, DocumentChunk
from api.apps.documents.schemas import DocumentCreate, DocumentResponse
from api.core.embeddings import generate_embeddings, generate_embeddings_batch
from api.core.vector_store import VectorStore
from api.utils.logger import get_logger
from api.utils.exceptions import InvalidDepartmentError, PermissionDeniedException
from api.utils.markdown_splitter import split_markdown_by_headers

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
    Ingest a new document with hierarchical chunking, embedding, and versioning.

    Flow:
    1. Validate department and permissions
    2. Get next version number
    3. Create document record (uncommitted)
    4. Chunk content (hierarchical for Markdown, flat for plain text)
    5. Store parent chunks in DB (not embedded)
    6. Embed child chunks and store in vector DB + DB
    7. Deactivate old versions if needed
    8. Commit transaction

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

    # 2. Create document record
    document = await Document.create(
        db=session,
        commit=False,
        title=data.title,
        department=data.department,
        doc_type=data.doc_type,
        source_url=str(data.source_url) if data.source_url else None,
        version=version,
        is_active=True,
        content_format=data.content_format,
        allowed_departments=[data.department],
    )

    # Flush to get the generated document.id
    await session.flush()
    doc_id_str = str(document.id)

    # 3. Route to hierarchical or flat chunking
    if data.content_format == "markdown":
        parent_count, child_count = await _ingest_hierarchical(
            session=session,
            vector_store=vector_store,
            document=document,
            content=data.content,
            doc_id_str=doc_id_str,
            data=data,
            version=version,
        )
        total_chunks = parent_count + child_count
    else:
        total_chunks = await _ingest_flat(
            session=session,
            vector_store=vector_store,
            document=document,
            content=data.content,
            doc_id_str=doc_id_str,
            data=data,
            version=version,
        )
        parent_count = 0
        child_count = total_chunks

    # 4. Deactivate old versions if this isn't v1
    if version > 1:
        await _deactivate_old_versions(
            session=session,
            vector_store=vector_store,
            title=data.title,
            department=data.department,
            current_doc_id=doc_id_str,
        )

    # 5. Commit the transaction
    await session.commit()
    await session.refresh(document)

    logger.info(
        f"Successfully ingested document: {doc_id_str}, version: {version}, "
        f"parents: {parent_count}, children: {child_count}"
    )

    return DocumentResponse(
        id=doc_id_str,
        title=document.title,
        department=document.department,
        version=version,
        chunk_count=total_chunks,
        content_format=data.content_format,
        parent_chunk_count=parent_count,
        child_chunk_count=child_count,
        created_at=document.created_at,
    )


# ── Ingestion Strategies ─────────────────────────────────────────────────────


async def _ingest_hierarchical(
    session: AsyncSession,
    vector_store: VectorStore,
    document: Document,
    content: str,
    doc_id_str: str,
    data: DocumentCreate,
    version: int,
) -> Tuple[int, int]:
    """
    Hierarchical ingestion: parent chunks (context) + child chunks (search).

    Parents are stored in DB only (no embedding). Children are embedded
    and stored in both Pinecone and DB with a foreign key to their parent.

    Returns:
        (parent_count, child_count) tuple
    """
    parents, children = _chunk_hierarchical(content)

    logger.info(
        f"Hierarchical chunking: {len(parents)} parents, "
        f"{len(children)} children for document: {doc_id_str}"
    )

    total_children = 0

    for parent_idx, parent_text in enumerate(parents):
        # Store parent chunk in DB (no vector embedding)
        parent_chunk = await DocumentChunk.create(
            db=session,
            commit=False,
            document_id=document.id,
            content=parent_text,
            chunk_index=parent_idx,
            vector_id=None,  # Parents are NOT embedded
            token_count=len(parent_text.split()),
            is_parent=True,
        )
        await session.flush()  # Get parent_chunk.id
        parent_id_str = str(parent_chunk.id)

        # Gather children belonging to this parent
        child_texts = [text for text, pidx in children if pidx == parent_idx]

        if not child_texts:
            continue

        # Batch embed children
        embeddings = await generate_embeddings_batch(child_texts)

        for child_idx, (child_text, embedding) in enumerate(
            zip(child_texts, embeddings)
        ):
            vector_id = f"{doc_id_str}_child_{parent_idx}_{child_idx}"

            await vector_store.upsert(
                id=vector_id,
                vector=embedding,
                metadata={
                    "content": child_text,
                    "document_id": doc_id_str,
                    "department": data.department,
                    "allowed_departments": document.allowed_departments or [data.department],
                    "chunk_index": child_idx,
                    "parent_id": parent_id_str,
                    "title": data.title,
                    "doc_type": data.doc_type,
                    "version": version,
                    "is_active": True,
                },
            )

            await DocumentChunk.create(
                db=session,
                commit=False,
                document_id=document.id,
                content=child_text,
                chunk_index=child_idx,
                vector_id=vector_id,
                token_count=len(child_text.split()),
                is_parent=False,
                parent_chunk_id=parent_chunk.id,
            )

            total_children += 1

    return len(parents), total_children


async def _ingest_flat(
    session: AsyncSession,
    vector_store: VectorStore,
    document: Document,
    content: str,
    doc_id_str: str,
    data: DocumentCreate,
    version: int,
) -> int:
    """
    Flat ingestion: fixed-size word-based chunks, all embedded.

    Used for plain text content without Markdown structure.

    Returns:
        Total chunk count
    """
    chunks = _chunk_fixed_size(content, chunk_size=500, overlap=50)
    logger.info(f"Flat chunking: {len(chunks)} chunks for document: {doc_id_str}")

    # Batch generate embeddings
    embeddings = await generate_embeddings_batch(chunks)

    chunk_records = []
    for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        vector_id = f"{doc_id_str}_chunk_{idx}"

        await vector_store.upsert(
            id=vector_id,
            vector=embedding,
            metadata={
                "content": chunk_text,
                "document_id": doc_id_str,
                "department": data.department,
                "allowed_departments": document.allowed_departments or [data.department],
                "chunk_index": idx,
                "title": data.title,
                "doc_type": data.doc_type,
                "version": version,
                "is_active": True,
            },
        )

        chunk_records.append(
            {
                "document_id": document.id,
                "content": chunk_text,
                "chunk_index": idx,
                "vector_id": vector_id,
                "token_count": len(chunk_text.split()),
                "is_parent": False,
            }
        )

    await DocumentChunk.create_many(
        db=session,
        items=chunk_records,
        commit=False,
    )

    return len(chunk_records)


# ── Chunking Algorithms ──────────────────────────────────────────────────────


def _chunk_hierarchical(
    content: str,
    min_parent_size: int = 2000,
    max_parent_size: int = 4000,
    child_size: int = 500,
    child_overlap: int = 50,
) -> Tuple[List[str], List[Tuple[str, int]]]:
    """
    Hierarchical chunking: parent (large context) + children (small search).

    Algorithm:
    1. Split by Markdown headers (custom splitter, zero deps)
    2. Merge small adjacent sections below min_parent_size
    3. Split parents exceeding max_parent_size
    4. Subdivide each parent into overlapping children

    Args:
        content: Full document text (Markdown)
        min_parent_size: Merge parents smaller than this (chars)
        max_parent_size: Split parents larger than this (chars)
        child_size: Target child chunk size (chars)
        child_overlap: Overlap between child chunks (chars)

    Returns:
        parent_chunks: List of large context strings
        child_chunks: List of (child_text, parent_index) tuples

    Complexity: O(n) where n = content length
    """
    # 1. Split by Markdown headers
    try:
        header_chunks = split_markdown_by_headers(
            content,
            headers=[
                ("#", "H1"), ("##", "H2"), ("###", "H3"),
                ("####", "H4"), ("#####", "H5"), ("######", "H6"),
            ],
            strip_headers=False,
        )
    except Exception:
        # Fallback: split by double newline
        header_chunks = [p.strip() for p in content.split("\n\n") if p.strip()]

    # If no meaningful splits, fall back to fixed-size
    if not header_chunks or len(header_chunks) <= 1:
        header_chunks = [
            content[i : i + max_parent_size]
            for i in range(0, len(content), max_parent_size)
        ]
        header_chunks = [c for c in header_chunks if c.strip()]

    # 2. Merge small adjacent parents
    merged_parents: List[str] = []
    current = ""

    for chunk in header_chunks:
        if len(current) + len(chunk) < min_parent_size:
            current = f"{current}\n\n{chunk}" if current else chunk
        else:
            if current:
                merged_parents.append(current)
            current = chunk

    if current:
        merged_parents.append(current)

    # 3. Split oversized parents
    final_parents: List[str] = []
    for parent in merged_parents:
        if len(parent) > max_parent_size:
            parts = [
                parent[i : i + max_parent_size]
                for i in range(0, len(parent), max_parent_size)
            ]
            final_parents.extend(parts)
        else:
            final_parents.append(parent)

    # 4. Create child chunks from each parent
    child_chunks: List[Tuple[str, int]] = []
    step = max(child_size - child_overlap, 1)

    for parent_idx, parent in enumerate(final_parents):
        for i in range(0, len(parent), step):
            child = parent[i : i + child_size]
            if len(child) > 100:  # Skip tiny fragments
                child_chunks.append((child, parent_idx))

    return final_parents, child_chunks


def _chunk_fixed_size(
    text: str, chunk_size: int = 500, overlap: int = 50
) -> List[str]:
    """
    Split text into overlapping word-based chunks.

    Why overlap:
    - Prevents information loss at boundaries
    - Improves retrieval accuracy
    - Standard RAG practice

    Args:
        text: Full document text
        chunk_size: Target chunk size in words
        overlap: Overlap between chunks in words

    Returns:
        List of text chunks

    Complexity: O(n) where n = number of words
    """
    words = text.split()
    chunks: List[str] = []

    if not words:
        return []

    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

        i += chunk_size - overlap

        if i >= len(words):
            break

    return chunks


# ── Version Management ────────────────────────────────────────────────────────


def calculate_next_version(latest_version: int | None) -> int:
    """
    Pure business logic function to calculate the next document version.
    Separates the calculation from the DB query (LDP Law).
    """
    if latest_version is None:
        return 1
    return latest_version + 1


async def _get_next_version(
    session: AsyncSession,
    title: str,
    department: str,
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
        order_desc=True,
    )

    latest_version = documents[0].version if documents else None
    return calculate_next_version(latest_version)


async def search_active_documents_for_agent(
    session: AsyncSession,
    department: str,
    keyword: str | None = None,
    limit: int = 20,
) -> List[dict]:
    """
    Search active documents for an agent.
    Service layer function to prevent cross-module model queries.

    Args:
        session: DB session
        department: Target department
        keyword: Optional title search keyword
        limit: Max results

    Returns:
        List of dictionaries containing document metadata needed by agents.
    """
    from sqlalchemy import select
    limit = min(limit, 100)

    query = (
        select(Document)
        .where(Document.is_active == True)
        .where(Document.is_deleted == False)
        .limit(limit)
    )
    
    result = await session.execute(query)
    docs = result.scalars().all()

    filtered = []
    for doc in docs:
        if doc.allowed_departments and department in doc.allowed_departments:
            filtered.append(doc)
        elif doc.department == department:
            filtered.append(doc)

    if keyword:
        kw_lower = keyword.lower()
        filtered = [d for d in filtered if kw_lower in d.title.lower()]

    return [
        {
            "id": str(doc.id),
            "title": doc.title,
            "department": doc.department,
            "doc_type": doc.doc_type,
            "version": doc.version,
            "allowed_departments": doc.allowed_departments,
        }
        for doc in filtered
    ]


async def _deactivate_old_versions(
    session: AsyncSession,
    vector_store: VectorStore,
    title: str,
    department: str,
    current_doc_id: str,
) -> None:
    """
    Soft delete old versions of a document.
    Marks as inactive in both relational DB and vector DB.

    Args:
        session: DB session
        vector_store: Vector DB client
        title: Document title
        department: Department name
        current_doc_id: ID of newly created document (don't deactivate this one)
    """
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    query = select(Document).where(
        Document.title == title,
        Document.department == department,
        Document.is_active == True,
        Document.id != current_doc_id
    ).options(selectinload(Document.chunks)).limit(100)

    result = await session.execute(query)
    old_docs = result.scalars().all()

    vector_ids_to_deactivate = []

    for doc in old_docs:
        doc.is_active = False
        await doc.save(db=session, commit=False)

        for chunk in doc.chunks:
            if chunk.vector_id:  # Only children have vector_ids
                vector_ids_to_deactivate.append(chunk.vector_id)

    if vector_ids_to_deactivate:
        await vector_store.update_metadata_by_ids(
            ids=vector_ids_to_deactivate,
            updates={"is_active": False},
        )
