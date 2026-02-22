"""
RAG query router.

Entry/exit only - NO business logic here.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api.apps.rag.schemas import QueryRequest, QueryResponse
from api.apps.rag.services import rag_query, rag_query_stream
from api.apps.auth.services import verify_user
from api.db.session import get_session
from api.core.dependencies import get_vector_store, get_cache
from api.utils.logger import get_logger
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])


@router.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")  # 10 queries per minute per user per dept
async def query_endpoint(
    payload: QueryRequest,
    # Need to get raw Request object for limiter
    request: Request,
    user: dict = Depends(verify_user),
    session: AsyncSession = Depends(get_session),
    vector_store = Depends(get_vector_store),
    cache = Depends(get_cache)
) -> QueryResponse:
    """
    Query documents with department-based filtering.
    
    **Security:** User can only retrieve documents from their department.
    **Caching:** Frequently asked questions are cached (dept-specific).
    **Hallucination Prevention:** Confidence threshold filters low-quality results.
    
    Example:
```
        POST /api/v1/rag/query
        Authorization: Bearer token_sales_001
        {
            "query": "What is our Cowbell pricing?",
            "max_results": 5,
            "confidence_threshold": 0.5
        }
```
    """
    logger.info(f"Query from user: {user['email']}, dept: {user['department']}")
    
    # Custom rate limit key logic per user and department
    request.scope['client'] = (f"{user['email']}:{user['department']}", 0)
    
    try:
        return await rag_query(
            session=session,
            vector_store=vector_store,
            cache=cache,
            request=payload,
            user_department=user["department"],
            user_id=user["id"]
        )
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Query processing failed. Please try again."
        )


@router.post("/query/stream")
@limiter.limit("10/minute")
async def query_stream_endpoint(
    payload: QueryRequest,
    request: Request,
    user: dict = Depends(verify_user),
    session: AsyncSession = Depends(get_session),
    vector_store = Depends(get_vector_store),
    cache = Depends(get_cache)
) -> StreamingResponse:
    """
    Stream documents and answers with Chat History support (Server-Sent Events).
    """
    logger.info(f"Stream Query from user: {user['email']}, dept: {user['department']}")
    
    # Custom rate limit key logic per user and department
    request.scope['client'] = (f"{user['email']}:{user['department']}", 0)
    
    try:
        generator = rag_query_stream(
            session=session,
            vector_store=vector_store,
            cache=cache,
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