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
import hashlib
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from api.apps.documents.models import Document, DocumentChunk
from api.apps.documents.schemas import DocumentCreate, DocumentResponse
from api.core.embeddings import generate_embeddings, generate_embeddings_batch
from api.core.vector_store import VectorStore
from api.core.cache import CacheManager
from api.utils.logger import get_logger
from api.utils.exceptions import InvalidDepartmentError, PermissionDeniedException
from api.utils.markdown_splitter import split_markdown_by_headers

logger = get_logger(__name__)

# Valid coragem departments
VALID_DEPARTMENTS = ["Sales", "HR", "Finance", "Operations", "Manufacturing", "IT"]


async def ingest_document(
    session: AsyncSession,
    vector_store: VectorStore,
    data: DocumentCreate,
    user_department: str,
    cache: CacheManager | None = None,
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
    9. Invalidate department cache so users see fresh results

    Args:
        session: Async DB session
        vector_store: Vector DB client
        data: Document creation request
        user_department: Department of user performing upload
        cache: Optional cache manager for invalidation after ingest

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

    # 1. Content hash dedup — skip if identical content already exists and is active
    content_hash = hashlib.sha256(data.content.encode()).hexdigest()
    existing = await _find_active_by_hash(session, content_hash, data.department)
    if existing:
        logger.info(
            f"Duplicate content detected (hash={content_hash[:12]}...), "
            f"skipping ingestion. Existing doc: {existing.id}"
        )
        chunk_count = await DocumentChunk.count(
            db=session, filters={"document_id": str(existing.id)}
        )
        parent_count = await DocumentChunk.count(
            db=session, filters={"document_id": str(existing.id), "is_parent": True}
        )
        return DocumentResponse(
            id=str(existing.id),
            title=existing.title,
            department=existing.department,
            version=existing.version,
            chunk_count=chunk_count,
            content_format=existing.content_format,
            parent_chunk_count=parent_count,
            child_chunk_count=chunk_count - parent_count,
            created_at=existing.created_at,
        )

    # 2. Check for existing document (versioning logic)
    version = await _get_next_version(session, data.title, data.department)

    # 3. Create document record
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
        content_hash=content_hash,
        allowed_departments=data.allowed_departments or [data.department],
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

    # 6. Invalidate cache ONLY on re-versioning (same title, updated content).
    # New documents with new titles don't invalidate — TTL handles staleness.
    if cache and version > 1:
        depts_to_invalidate = set(data.allowed_departments or [data.department])
        depts_to_invalidate.add(data.department)
        for dept in depts_to_invalidate:
            await cache.invalidate_department(dept)
        logger.info(f"Cache invalidated for re-versioned doc '{data.title}' v{version}")

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

    Uses batch embedding and batch Pinecone upsert to minimize API calls.
    Complexity: O(n/batch_size) API calls instead of O(n).

    Returns:
        (parent_count, child_count) tuple
    """
    parents, children = _chunk_hierarchical(content)

    logger.info(
        f"Hierarchical chunking: {len(parents)} parents, "
        f"{len(children)} children for document: {doc_id_str}"
    )

    # Collect all vectors for a single batch upsert at the end
    pending_vectors: List[tuple] = []
    total_children = 0
    allowed_depts = document.allowed_departments or [data.department]

    for parent_idx, parent_text in enumerate(parents):
        parent_chunk = await DocumentChunk.create(
            db=session,
            commit=False,
            document_id=document.id,
            content=parent_text,
            chunk_index=parent_idx,
            vector_id=None,
            token_count=len(parent_text.split()),
            is_parent=True,
        )
        await session.flush()
        parent_id_str = str(parent_chunk.id)

        raw_child_texts = [text for text, pidx in children if pidx == parent_idx]
        if not raw_child_texts:
            continue

        # Prepend parent section header to each child so the LLM can cite
        # the exact section regardless of which child chunk is retrieved.
        section_header = _extract_first_header(parent_text)
        if section_header:
            child_texts = [f"[Section: {section_header}]\n{ct}" for ct in raw_child_texts]
        else:
            child_texts = raw_child_texts

        embeddings = await generate_embeddings_batch(child_texts)

        for child_idx, (child_text, embedding) in enumerate(
            zip(child_texts, embeddings)
        ):
            vector_id = f"{doc_id_str}_child_{parent_idx}_{child_idx}"

            pending_vectors.append((
                vector_id,
                embedding,
                {
                    "content": child_text,
                    "document_id": doc_id_str,
                    "department": data.department,
                    "allowed_departments": allowed_depts,
                    "chunk_index": child_idx,
                    "parent_id": parent_id_str,
                    "title": data.title,
                    "doc_type": data.doc_type,
                    "version": version,
                    "is_active": True,
                },
            ))

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

    # Single batch upsert — O(n/100) API calls instead of O(n)
    if pending_vectors:
        await vector_store.upsert_batch(pending_vectors)

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

    Uses batch embedding and batch Pinecone upsert.
    Complexity: O(n/100) Pinecone API calls instead of O(n).

    Returns:
        Total chunk count
    """
    chunks = _chunk_fixed_size(content, chunk_size=500, overlap=50)
    logger.info(f"Flat chunking: {len(chunks)} chunks for document: {doc_id_str}")

    embeddings = await generate_embeddings_batch(chunks)
    allowed_depts = document.allowed_departments or [data.department]

    pending_vectors: List[tuple] = []
    chunk_records = []

    for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        vector_id = f"{doc_id_str}_chunk_{idx}"

        pending_vectors.append((
            vector_id,
            embedding,
            {
                "content": chunk_text,
                "document_id": doc_id_str,
                "department": data.department,
                "allowed_departments": allowed_depts,
                "chunk_index": idx,
                "title": data.title,
                "doc_type": data.doc_type,
                "version": version,
                "is_active": True,
            },
        ))

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

    # Batch upsert all vectors
    if pending_vectors:
        await vector_store.upsert_batch(pending_vectors)

    await DocumentChunk.create_many(
        db=session,
        items=chunk_records,
        commit=False,
    )

    return len(chunk_records)


def _extract_first_header(text: str) -> str | None:
    """
    Extract the first heading from a parent chunk.

    Handles common formats: Markdown headers, numbered sections,
    bold headers. Returns None if no heading found.
    """
    import re

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Markdown header: ## **Title** or ## Title
        match = re.match(r"^#{1,6}\s+\*{0,2}(.+?)\*{0,2}\s*$", line)
        if match:
            return match.group(1).strip()
        # Numbered section: "7.0 How to Deal..." or "3. Title"
        match = re.match(r"^(\d+(?:\.\d+)?\.?\s+\S.{4,80})$", line)
        if match:
            return match.group(1).strip()
        # Stop after first non-empty line if no header found
        break
    return None


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


async def _find_active_by_hash(
    session: AsyncSession,
    content_hash: str,
    department: str,
) -> Document | None:
    """
    Check if an active document with identical content already exists.

    Why hash-based: avoids re-ingesting the same PDF uploaded multiple times,
    which wastes embedding API calls and Pinecone storage.
    """
    docs = await Document.find_many(
        db=session,
        limit=1,
        filters={
            "content_hash": content_hash,
            "department": department,
            "is_active": True,
        },
    )
    return docs[0] if docs else None


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
    Search active documents visible to a department.

    Filters by department authorization in SQL (not Python) to avoid
    loading all documents into memory.

    Args:
        session: DB session
        department: Target department
        keyword: Optional title search keyword (case-insensitive)
        limit: Max results

    Returns:
        List of dicts with document metadata.

    Complexity: O(log n) with proper indexes vs O(n) full-table scan before.
    """
    from sqlalchemy import select, or_, func
    limit = min(limit, 100)

    query = (
        select(Document)
        .where(Document.is_active == True)
        .where(Document.is_deleted == False)
        .where(
            or_(
                Document.department == department,
                Document.allowed_departments.contains([department]),
                Document.allowed_departments.contains(["All"]),
            )
        )
    )

    if keyword:
        query = query.where(func.lower(Document.title).contains(keyword.lower()))

    query = query.limit(limit)
    result = await session.execute(query)
    docs = result.scalars().all()

    return [
        {
            "id": str(doc.id),
            "title": doc.title,
            "department": doc.department,
            "doc_type": doc.doc_type,
            "version": doc.version,
            "allowed_departments": doc.allowed_departments,
        }
        for doc in docs
    ]


async def update_document_permissions(
    session: AsyncSession,
    vector_store: VectorStore,
    cache: CacheManager | None,
    document_id: str,
    allowed_departments: List[str],
    user_department: str,
) -> dict:
    """
    Update which departments can access a document.

    Updates three layers atomically:
    1. Document record in PostgreSQL (allowed_departments column)
    2. All child chunk vectors in Pinecone (allowed_departments metadata)
    3. Redis cache invalidation for affected departments

    Args:
        session: DB session
        vector_store: Vector DB client
        cache: Cache manager for invalidation
        document_id: UUID of the document to update
        allowed_departments: New list of departments
        user_department: Department of the requesting user (for permission check)

    Returns:
        Dict with document metadata and count of updated vectors

    Raises:
        PermissionDeniedException: If user's department doesn't own the document and isn't IT
    """
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    result = await session.execute(
        select(Document)
        .where(Document.id == document_id, Document.is_deleted == False)
        .options(selectinload(Document.chunks))
    )
    document = result.scalar_one_or_none()

    if not document:
        from api.utils.exceptions import BaseAPIException
        raise BaseAPIException(status_code=404, detail="Document not found")

    # Only the owning department or IT can change permissions
    if user_department != "IT" and user_department != document.department:
        raise PermissionDeniedException(
            detail=f"Only the {document.department} department or IT can update permissions for this document"
        )

    old_departments = set(document.allowed_departments or [document.department])

    # 1. Update PostgreSQL
    document.allowed_departments = allowed_departments
    await document.save(db=session, commit=False)

    # 2. Update Pinecone metadata for all child chunk vectors
    vector_ids = [chunk.vector_id for chunk in document.chunks if chunk.vector_id]
    if vector_ids:
        await vector_store.update_metadata_by_ids(
            ids=vector_ids,
            updates={"allowed_departments": allowed_departments},
        )

    await session.commit()

    # 3. Invalidate cache for both old and new departments
    if cache:
        all_affected = old_departments | set(allowed_departments)
        for dept in all_affected:
            await cache.invalidate_department(dept)

    logger.info(
        f"Updated permissions for document {document_id}: "
        f"{list(old_departments)} → {allowed_departments}, "
        f"{len(vector_ids)} vectors updated"
    )

    return {
        "id": str(document.id),
        "title": document.title,
        "department": document.department,
        "allowed_departments": allowed_departments,
        "vectors_updated": len(vector_ids),
    }


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
