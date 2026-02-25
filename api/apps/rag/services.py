"""
RAG query service with department-based filtering.

This is the CORE of the multi-tenant system.
Streaming-only architecture — generate_answer_stream is the single code path.
"""

import json
import time
from typing import List, Dict, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

from api.apps.rag.schemas import QueryRequest, QueryResponse, SourceDocument, ChatRole
from api.apps.rag.models import QueryLog
from api.core.vector_store import VectorStore
from api.core.cache import CacheManager
from api.core.llm import generate_answer_stream
from api.core.semantic_router import SemanticRouter, RoutedAgent
from api.utils.logger import get_logger
from api.utils.metrics import query_count, query_latency
from api.utils.pipeline_timer import PipelineTimer

logger = get_logger(__name__)


# ── Standard JSON Query (Collects Stream) ─────────────────────────────────────


async def rag_query(
    session: AsyncSession,
    vector_store: VectorStore,
    cache: CacheManager,
    semantic_router: SemanticRouter,
    request: QueryRequest,
    user_department: str,
    user_id: str
) -> QueryResponse:
    """
    Execute RAG query with department filtering.

    Internally uses the streaming generator and collects all chunks
    into a single response. This ensures ONE code path for LLM generation.

    Flow:
    1. Check chat history
    2. Check cache (Redis) - dept-specific
    3. Retrieve from vector DB WITH department filter
    4. Stream answer (collecting chunks)
    5. Cache result (dept-specific)
    6. Return response
    """
    timer = PipelineTimer(department=user_department)
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
        await cache.append_chat_message(request.session_id, ChatRole.USER, request.query)

    # 1. Semantic Routing (Intent Classification)
    with timer.stage("intent_classification"):
        decision = await semantic_router.route_query(request.query, user_department)
        
    if decision.routed_to != RoutedAgent.RAG:
        latency_ms = (time.time() - start_time) * 1000
        msg = (
            f"This query requires the '{decision.routed_to.value}' agent "
            f"(Confidence: {decision.confidence_score:.2f}). "
            f"This specialized Copilot integration is coming soon."
        )
        response = QueryResponse(
            answer=msg,
            sources=[],
            confidence="high",
            latency_ms=latency_ms,
            cached=False
        )
        if request.session_id:
            await cache.append_chat_message(request.session_id, ChatRole.ASSISTANT, msg)
            
        await _log_query(
            session, request.query, user_id, user_department,
            0, latency_ms, False, "high",
            stage_timings=timer.as_dict(),
            routed_to=decision.routed_to.value
        )
        return response

    # 2. Check cache (dept-specific key)
    cache_key = cache.get_key(request.query, user_department)
    with timer.stage("cache_check"):
        cached_result = await cache.get(cache_key)
    if cached_result:
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Cache HIT: {latency_ms:.2f}ms")
        return QueryResponse(
            **cached_result,
            latency_ms=latency_ms,
            cached=True
        )

    # 2. Retrieve documents with department filter
    with timer.stage("retrieval"):
        docs = await vector_store.search(
            query=request.query,
            filter={
                "department": user_department,
                "is_active": True
            },
            limit=request.max_results,
            score_threshold=request.confidence_threshold
        )

    # 3. No high-confidence results → bail early
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
        await _cache_response(cache, cache_key, response, ttl=1800)

        if request.session_id:
            await cache.append_chat_message(request.session_id, ChatRole.ASSISTANT, response.answer)

        return response

    # 4. Collect streamed answer into a single string
    answer_parts: List[str] = []
    with timer.stage("generation"):
        async for chunk in generate_answer_stream(
            query=request.query,
            docs=docs,
            department=user_department,
            chat_history=chat_history
        ):
            answer_parts.append(chunk)

    answer = "".join(answer_parts)

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

    # Save to history
    if request.session_id:
        await cache.append_chat_message(request.session_id, ChatRole.ASSISTANT, answer)

    # 6. Cache + Log + Metrics
    await _cache_response(cache, cache_key, response, ttl=3600)
    await _log_query(
        session, request.query, user_id, user_department,
        len(docs), latency_ms, False, "high",
        stage_timings=timer.as_dict(),
        routed_to=RoutedAgent.RAG.value
    )
    query_count.labels(department=user_department, confidence="high").inc()
    query_latency.labels(department=user_department, cached="false").observe(latency_ms / 1000.0)

    logger.info(
        f"Query successful: {len(docs)} docs, {latency_ms:.2f}ms, "
        f"answer_len={len(answer)} chars"
    )

    return response


# ── SSE Streaming Query ───────────────────────────────────────────────────────


async def rag_query_stream(
    session: AsyncSession,
    vector_store: VectorStore,
    cache: CacheManager,
    semantic_router: SemanticRouter,
    request: QueryRequest,
    user_department: str,
    user_id: str
) -> AsyncGenerator[str, None]:
    """
    Execute RAG query with streaming SSE output and conversational memory.
    """
    timer = PipelineTimer(department=user_department)
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
        await cache.append_chat_message(request.session_id, ChatRole.USER, request.query)

    # Semantic Routing (Intent Classification)
    with timer.stage("intent_classification"):
        decision = await semantic_router.route_query(request.query, user_department)
        
    if decision.routed_to != RoutedAgent.RAG:
        latency_ms = (time.time() - start_time) * 1000
        msg = (
            f"This query requires the '{decision.routed_to.value}' agent "
            f"(Confidence: {decision.confidence_score:.2f}). "
            f"This specialized Copilot integration is coming soon."
        )
        yield f"event: sources\ndata: []\n\n"
        yield f"event: message\ndata: {json.dumps({'content': msg})}\n\n"
        
        if request.session_id:
            await cache.append_chat_message(request.session_id, ChatRole.ASSISTANT, msg)
            
        await _log_query(
            session, request.query, user_id, user_department,
            0, latency_ms, False, "high",
            stage_timings=timer.as_dict(),
            routed_to=decision.routed_to.value
        )
        yield "event: end\ndata: {}\n\n"
        return

    # Retrieve documents
    with timer.stage("retrieval"):
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

        if request.session_id and full_answer:
            await cache.append_chat_message(request.session_id, ChatRole.ASSISTANT, full_answer)

        await _log_query(
            session, request.query, user_id, user_department,
            len(docs), latency_ms, False, "high",
            stage_timings=timer.as_dict(),
            routed_to=RoutedAgent.RAG.value
        )
        query_count.labels(department=user_department, confidence="high").inc()
        query_latency.labels(department=user_department, cached="false").observe(latency_ms / 1000.0)

        yield "event: end\ndata: {}\n\n"


# ── Private Helpers ───────────────────────────────────────────────────────────


async def _cache_response(
    cache: CacheManager,
    cache_key: str,
    response: QueryResponse,
    ttl: int
) -> None:
    """Cache response (without latency/cached fields)."""
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
    confidence: str,
    stage_timings: dict | None = None,
    routed_to: str = "rag"
) -> None:
    """Log a RAG query to DB for analytics, including per-stage timings."""
    logger.debug(f"Logging query to DB stats: '{query[:30]}...' from {department}")
    try:
        log_entry = QueryLog(
            query=query,
            user_id=user_id,
            department=department,
            result_count=result_count,
            latency_ms=latency_ms,
            cached=cached,
            confidence=confidence,
            stage_timings=stage_timings,
            routed_to=routed_to
        )
        session.add(log_entry)
        await session.commit()
    except Exception as e:
        logger.error(f"Failed to save query log: {e}")
        await session.rollback()