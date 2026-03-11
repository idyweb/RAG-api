"""
RAG query router.

Entry/exit only - NO business logic here.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.apps.rag.schemas import QueryRequest
from api.apps.rag.services import rag_query_stream
from api.db.session import get_session
from api.core.dependencies import (
    get_vector_store, 
    get_cache,
    get_semantic_router, 
    verify_user
)
from api.utils.logger import get_logger
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])


@router.post("/query/stream")
@limiter.limit("10/minute")
async def query_stream_endpoint(
    payload: QueryRequest,
    request: Request,
    user: dict = Depends(verify_user),
    session: AsyncSession = Depends(get_session),
    vector_store = Depends(get_vector_store),
    cache = Depends(get_cache),
    semantic_router = Depends(get_semantic_router)
) -> StreamingResponse:
    """
    Stream documents and answers with Chat History support (Server-Sent Events).
    """
    logger.info(f"Stream Query from user: {user['email']}, dept: {user['department']}")

    # Guard: reject trivially short queries
    if len(payload.query) < 2:
        raise HTTPException(
            status_code=400,
            detail="Query too short for meaningful RAG.",
        )

    # Custom rate limit key logic per user and department
    request.scope['client'] = (f"{user['email']}:{user['department']}", 0)
    
    try:
        generator = rag_query_stream(
            session=session,
            vector_store=vector_store,
            cache=cache,
            semantic_router=semantic_router,
            request=payload,
            user_department=user["department"],
            user_id=user["id"]
        )
        return StreamingResponse(
            generator, 
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Stream query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Stream query processing failed. Please try again."
        )