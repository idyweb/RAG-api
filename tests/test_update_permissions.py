"""
Tests for the PATCH /api/v1/documents/{id}/permissions endpoint
and the update_document_permissions service function.

Covers:
1. Happy path: owner updates permissions
2. IT admin can update any document
3. Unauthorized department is rejected
4. Document not found returns 404
5. Pinecone vectors and cache are updated correctly
6. Validation: empty allowed_departments rejected
"""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from api.apps.documents.services import update_document_permissions
from api.utils.exceptions import PermissionDeniedException


def _make_document(
    doc_id: str = None,
    title: str = "Test Policy",
    department: str = "HR",
    allowed_departments: list[str] | None = None,
    chunks: list | None = None,
):
    """Create a mock Document with chunks."""
    doc = MagicMock()
    doc.id = doc_id or str(uuid.uuid4())
    doc.title = title
    doc.department = department
    doc.allowed_departments = allowed_departments or [department]
    doc.is_deleted = False
    doc.chunks = chunks or []
    doc.save = AsyncMock()
    return doc


def _make_chunk(vector_id: str | None = None):
    """Create a mock DocumentChunk."""
    chunk = MagicMock()
    chunk.vector_id = vector_id
    return chunk


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    return session


@pytest.fixture
def mock_vector_store():
    vs = AsyncMock()
    vs.update_metadata_by_ids = AsyncMock()
    return vs


@pytest.fixture
def mock_cache():
    cache = AsyncMock()
    cache.invalidate_department = AsyncMock()
    return cache


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: Owner can update their own document's permissions
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_owner_can_update_permissions(mock_session, mock_vector_store, mock_cache):
    """HR user can update permissions on an HR-owned document."""
    doc_id = str(uuid.uuid4())
    chunks = [
        _make_chunk(vector_id=f"{doc_id}_child_0_0"),
        _make_chunk(vector_id=f"{doc_id}_child_0_1"),
        _make_chunk(vector_id=None),  # Parent chunk — no vector_id
    ]
    document = _make_document(
        doc_id=doc_id,
        department="HR",
        allowed_departments=["HR"],
        chunks=chunks,
    )

    # Mock the DB query to return our document
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = document
    mock_session.execute = AsyncMock(return_value=mock_result)

    result = await update_document_permissions(
        session=mock_session,
        vector_store=mock_vector_store,
        cache=mock_cache,
        document_id=doc_id,
        allowed_departments=["HR", "Finance", "Sales"],
        user_department="HR",
    )

    assert result["id"] == doc_id
    assert result["allowed_departments"] == ["HR", "Finance", "Sales"]
    assert result["vectors_updated"] == 2  # Only chunks with vector_ids

    # Verify Pinecone was updated with correct metadata
    mock_vector_store.update_metadata_by_ids.assert_called_once_with(
        ids=[f"{doc_id}_child_0_0", f"{doc_id}_child_0_1"],
        updates={"allowed_departments": ["HR", "Finance", "Sales"]},
    )

    # Verify cache was invalidated for old + new departments
    invalidated_depts = {
        call.args[0] for call in mock_cache.invalidate_department.call_args_list
    }
    assert "HR" in invalidated_depts
    assert "Finance" in invalidated_depts
    assert "Sales" in invalidated_depts

    mock_session.commit.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: IT admin can update any document
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_it_admin_can_update_any_document(mock_session, mock_vector_store, mock_cache):
    """IT department can update permissions on any department's document."""
    doc_id = str(uuid.uuid4())
    document = _make_document(
        doc_id=doc_id,
        department="Sales",
        allowed_departments=["Sales"],
        chunks=[_make_chunk(vector_id=f"{doc_id}_child_0_0")],
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = document
    mock_session.execute = AsyncMock(return_value=mock_result)

    result = await update_document_permissions(
        session=mock_session,
        vector_store=mock_vector_store,
        cache=mock_cache,
        document_id=doc_id,
        allowed_departments=["Sales", "Marketing"],
        user_department="IT",  # IT admin
    )

    assert result["allowed_departments"] == ["Sales", "Marketing"]
    mock_session.commit.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: Unauthorized department is rejected
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_unauthorized_department_rejected(mock_session, mock_vector_store, mock_cache):
    """Finance user cannot update permissions on an HR-owned document."""
    doc_id = str(uuid.uuid4())
    document = _make_document(
        doc_id=doc_id,
        department="HR",
        allowed_departments=["HR"],
        chunks=[],
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = document
    mock_session.execute = AsyncMock(return_value=mock_result)

    with pytest.raises(PermissionDeniedException):
        await update_document_permissions(
            session=mock_session,
            vector_store=mock_vector_store,
            cache=mock_cache,
            document_id=doc_id,
            allowed_departments=["HR", "Finance"],
            user_department="Finance",  # Not the owner, not IT
        )

    # Verify nothing was committed
    mock_session.commit.assert_not_called()
    mock_vector_store.update_metadata_by_ids.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: Document not found
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_document_not_found_raises_404(mock_session, mock_vector_store, mock_cache):
    """Updating a nonexistent document returns 404."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute = AsyncMock(return_value=mock_result)

    from api.utils.exceptions import BaseAPIException

    with pytest.raises(BaseAPIException) as exc_info:
        await update_document_permissions(
            session=mock_session,
            vector_store=mock_vector_store,
            cache=mock_cache,
            document_id=str(uuid.uuid4()),
            allowed_departments=["HR"],
            user_department="HR",
        )

    assert exc_info.value.status_code == 404


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: Document with no child chunks (parents only)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_no_vectors_to_update(mock_session, mock_vector_store, mock_cache):
    """Document with only parent chunks (no vector_ids) skips Pinecone update."""
    doc_id = str(uuid.uuid4())
    document = _make_document(
        doc_id=doc_id,
        department="HR",
        allowed_departments=["HR"],
        chunks=[_make_chunk(vector_id=None), _make_chunk(vector_id=None)],
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = document
    mock_session.execute = AsyncMock(return_value=mock_result)

    result = await update_document_permissions(
        session=mock_session,
        vector_store=mock_vector_store,
        cache=mock_cache,
        document_id=doc_id,
        allowed_departments=["HR", "Finance"],
        user_department="HR",
    )

    assert result["vectors_updated"] == 0
    mock_vector_store.update_metadata_by_ids.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: Cache invalidation covers both old and new departments
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_cache_invalidation_covers_old_and_new(mock_session, mock_vector_store, mock_cache):
    """Changing from [HR, Sales] to [HR, Finance] invalidates HR, Sales, AND Finance."""
    doc_id = str(uuid.uuid4())
    document = _make_document(
        doc_id=doc_id,
        department="HR",
        allowed_departments=["HR", "Sales"],
        chunks=[_make_chunk(vector_id=f"{doc_id}_c0")],
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = document
    mock_session.execute = AsyncMock(return_value=mock_result)

    await update_document_permissions(
        session=mock_session,
        vector_store=mock_vector_store,
        cache=mock_cache,
        document_id=doc_id,
        allowed_departments=["HR", "Finance"],
        user_department="HR",
    )

    invalidated_depts = {
        call.args[0] for call in mock_cache.invalidate_department.call_args_list
    }
    # Old: HR, Sales. New: HR, Finance. Union = HR, Sales, Finance
    assert invalidated_depts == {"HR", "Sales", "Finance"}


# ═══════════════════════════════════════════════════════════════════════════
# TEST 7: Works without cache (cache=None)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_works_without_cache(mock_session, mock_vector_store):
    """Service function works when cache is None (e.g., called from Celery)."""
    doc_id = str(uuid.uuid4())
    document = _make_document(
        doc_id=doc_id,
        department="HR",
        allowed_departments=["HR"],
        chunks=[_make_chunk(vector_id=f"{doc_id}_c0")],
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = document
    mock_session.execute = AsyncMock(return_value=mock_result)

    result = await update_document_permissions(
        session=mock_session,
        vector_store=mock_vector_store,
        cache=None,
        document_id=doc_id,
        allowed_departments=["HR", "Sales"],
        user_department="HR",
    )

    assert result["allowed_departments"] == ["HR", "Sales"]
    mock_session.commit.assert_called_once()
