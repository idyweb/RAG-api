"""
RAG query service with department-based filtering.

This is the CORE of the multi-tenant system.
"""

import time
from typing import List, Dict
from sqlalchemy.ext.asyncio import AsyncSession

from api.apps.rag.schemas import QueryRequest, QueryResponse, SourceDocument, ChatRole
from api.apps.rag.models import QueryLog
from api.core.vector_store import VectorStore
from api.core.cache import CacheManager
from api.core.llm import generate_answer, generate_answer_stream
from api.utils.logger import get_logger
from prometheus_client import Counter, Histogram

logger = get_logger(__name__)

# Prometheus Metrics
query_count = Counter('rag_queries_total', 'Total queries', ['department'])
query_latency = Histogram('rag_query_latency_seconds', 'Query latency', ['department'])

async def rag_query(
    session: AsyncSession,
    vector_store: VectorStore,
    cache: CacheManager,
    request: QueryRequest,
    user_department: str,
    user_id: str
) -> QueryResponse:
    """
    Execute RAG query with department filtering.
    
    Flow:
    1. Check cache (Redis) - dept-specific
    2. If miss: Retrieve from vector DB WITH department filter
    3. Apply confidence threshold (hallucination prevention)
    4. If no high-confidence docs: Return "I don't know"
    5. Generate answer with LLM
    6. Cache result (dept-specific)
    7. Return response
    
    Args:
        session: DB session (for future query logging)
        vector_store: Vector DB client
        cache: Redis cache manager
        request: Query request with question and parameters
        user_department: User's department (CRITICAL for filtering)
        
    Returns:
        QueryResponse with answer and sources
    """
    start_time = time.time()
    
    logger.info(
        f"RAG query: dept={user_department}, "
        f"query='{request.query[:50]}...', "
        f"threshold={request.confidence_threshold}"
    )
    
    # 0. Fetch Chat History if session_id is provided
    chat_history = None
    if request.session_id:
        chat_history = await cache.get_chat_history(request.session_id)
        # Store user's new message
        await cache.append_chat_message(request.session_id, ChatRole.USER, request.query)
        
    # 1. Check cache (dept-specific key)
    cache_key = cache.get_key(request.query, user_department)
    if cached_result := await cache.get(cache_key):
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Cache HIT: {latency_ms:.2f}ms")
        
        return QueryResponse(
            **cached_result,
            latency_ms=latency_ms,
            cached=True
        )
    
    # 2. Retrieve documents with department filter
    docs = await vector_store.search(
        query=request.query,
        filter={
            "department": user_department,  # ← CRITICAL: Dept isolation
            "is_active": True
        },
        limit=request.max_results,
        score_threshold=request.confidence_threshold
    )
    
    # 3. Check if we have high-confidence results
    if not docs:
        latency_ms = (time.time() - start_time) * 1000
        
        logger.warning(
            f"No high-confidence docs found: dept={user_department}, "
            f"query='{request.query[:50]}...'"
        )
        
        response = QueryResponse(
            answer="I don't have enough information to answer this question accurately. "
                   "This might be because:\n"
                   "1. The information isn't available in my knowledge base\n"
                   "2. Your query needs more specific details\n"
                   "3. The information exists in a different department",
            sources=[],
            confidence="low",
            latency_ms=latency_ms,
            cached=False
        )
        
        # Cache "I don't know" responses too (prevent repeated searches)
        await _cache_response(cache, cache_key, response, ttl=1800)  # 30 min TTL
        
        # Save assistant failure response to history
        if request.session_id:
            await cache.append_chat_message(request.session_id, ChatRole.ASSISTANT, response.answer)
            
        return response
    
    # 4. Generate answer with LLM using context and history
    answer = await generate_answer(request.query, docs, user_department, chat_history=chat_history)
    
    # 5. Build source documents list
    sources = [
        SourceDocument(
            document_id=doc["metadata"]["document_id"],
            title=doc["metadata"]["title"],
            department=doc["metadata"]["department"],
            chunk_index=doc["metadata"]["chunk_index"],
            doc_type=doc["metadata"]["doc_type"],
            relevance_score=round(doc["score"], 3)
        )
        for doc in docs
    ]
    
    latency_ms = (time.time() - start_time) * 1000
    
    response = QueryResponse(
        answer=answer,
        sources=sources,
        confidence="high",
        latency_ms=latency_ms,
        cached=False
    )
    
    # Save successful assistant response to history
    if request.session_id:
        await cache.append_chat_message(request.session_id, ChatRole.ASSISTANT, answer)
        
    # 6. Cache result (1 hour TTL for successful queries)
    await _cache_response(cache, cache_key, response, ttl=3600)
    
    # 7. Log query for analytics 
    await _log_query(session, request.query, user_id, user_department, len(docs), latency_ms, False, "high")
    
    # Increment prometheus metrics
    query_count.labels(department=user_department).inc()
    query_latency.labels(department=user_department).observe(latency_ms / 1000.0)

    logger.info(
        f"Query successful: {len(docs)} docs, {latency_ms:.2f}ms, "
        f"answer_len={len(answer)} chars"
    )
    
    return response


import json
from typing import AsyncGenerator

async def rag_query_stream(
    session: AsyncSession,
    vector_store: VectorStore,
    cache: CacheManager,
    request: QueryRequest,
    user_department: str,
    user_id: str
) -> AsyncGenerator[str, None]:
    """
    Execute RAG query with streaming SSE output and conversational memory.
    """
    start_time = time.time()
    
    logger.info(
        f"RAG streaming query: dept={user_department}, "
        f"query='{request.query[:50]}...', "
        f"session_id={request.session_id}"
    )
    
    # Fetch chat history
    chat_history = None
    if request.session_id:
        chat_history = await cache.get_chat_history(request.session_id)
        # Store user's new message
        await cache.append_chat_message(request.session_id, ChatRole.USER, request.query)
        
    # Retrieve documents
    docs = await vector_store.search(
        query=request.query,
        filter={"department": user_department, "is_active": True},
        limit=request.max_results,
        score_threshold=request.confidence_threshold
    )
    
    # Build Sources payload first to stream it immediately
    sources = [
        {
            "document_id": doc["metadata"]["document_id"],
            "title": doc["metadata"]["title"],
            "department": doc["metadata"]["department"],
            "chunk_index": doc["metadata"]["chunk_index"],
            "doc_type": doc["metadata"]["doc_type"],
            "relevance_score": round(doc["score"], 3)
        }
        for doc in docs
    ]
    
    # Send custom events for metadata
    yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
    
    if not docs:
        failure_msg = "I don't have enough information to answer this question accurately."
        yield f"event: message\ndata: {json.dumps({'content': failure_msg})}\n\n"
        
        if request.session_id:
            await cache.append_chat_message(request.session_id, ChatRole.ASSISTANT, failure_msg)
            
        yield "event: end\ndata: {}\n\n"
        return
        
    # Stream the LLM tokens
    full_answer = ""
    try:
        async for chunk in generate_answer_stream(
            query=request.query,
            docs=docs,
            department=user_department,
            chat_history=chat_history
        ):
            full_answer += chunk
            yield f"event: message\ndata: {json.dumps({'content': chunk})}\n\n"
            
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        yield f"event: error\ndata: {json.dumps({'error': 'Stream generation failed'})}\n\n"
        
    finally:
        latency_ms = (time.time() - start_time) * 1000
        
        # Save assistant's completed response to history
        if request.session_id and full_answer:
            await cache.append_chat_message(request.session_id, ChatRole.ASSISTANT, full_answer)
            
        # Log Metrics
        await _log_query(session, request.query, user_id, user_department, len(docs), latency_ms, False, "high")
        query_count.labels(department=user_department).inc()
        query_latency.labels(department=user_department).observe(latency_ms / 1000.0)
        
        yield "event: end\ndata: {}\n\n"




async def _cache_response(
    cache: CacheManager,
    cache_key: str,
    response: QueryResponse,
    ttl: int
) -> None:
    """
    Cache response (without latency/cached fields).
    
    Args:
        cache: Cache manager
        cache_key: Cache key
        response: Response to cache
        ttl: Time-to-live in seconds
    """
    # Convert to dict for caching (exclude latency/cached as they'll change)
    cache_data = {
        "answer": response.answer,
        "sources": [s.model_dump() for s in response.sources],
        "confidence": response.confidence
    }
    
    await cache.set(cache_key, cache_data, ttl=ttl)

async def _log_query(
    session: AsyncSession,
    query: str,
    user_id: str,
    department: str,
    result_count: int,
    latency_ms: float,
    cached: bool,
    confidence: str
) -> None:
    """Log a RAG query to DB for analytics."""
    logger.debug(f"Logging query to DB stats: '{query[:30]}...' from {department}")
    try:
        log_entry = QueryLog(
            query=query,
            user_id=user_id,
            department=department,
            result_count=result_count,
            latency_ms=latency_ms,
            cached=cached,
            confidence=confidence
        )
        session.add(log_entry)
        await session.commit()
    except Exception as e:
        logger.error(f"Failed to save query log: {e}")
        await session.rollback()