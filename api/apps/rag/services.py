"""
RAG query service with department-based filtering.

This is the CORE of the multi-tenant system.
Streaming-only architecture — generate_answer_stream is the single code path.
"""

import json
import time
from typing import List, Dict, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

from api.apps.rag.schemas import QueryRequest, ChatRole
from api.apps.rag.models import QueryLog
from api.core.vector_store import VectorStore
from api.core.cache import CacheManager
from api.core.llm import generate_answer_stream
from api.core.semantic_router import SemanticRouter, RoutedAgent
from api.utils.logger import get_logger
from api.utils.metrics import query_count, query_latency
from api.utils.pipeline_timer import PipelineTimer

logger = get_logger(__name__)


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

    # Check Cache First
    cache_key = cache.get_key(request.query, user_department)
    with timer.stage("cache_check"):
        cached_result = await cache.get(cache_key)

    if cached_result:
        # Yield sources from cache immediately
        yield f"event: sources\ndata: {json.dumps(cached_result['sources'])}\n\n"

        yield f"event: message\ndata: {json.dumps({'content': cached_result['answer']})}\n\n"

        yield "event: end\ndata: {}\n\n"

        
        await _log_query(
            session, request.query, user_id, user_department,
            len(cached_result['sources']), (time.time() - start_time) * 1000,
            True, cached_result['confidence'],
            stage_timings=timer.as_dict()
        )
        query_count.labels(department=user_department, confidence=cached_result['confidence']).inc()
        query_latency.labels(department=user_department, cached="true").observe(
            (time.time() - start_time) / 1000.0
        )
        return

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
            f"This query requires the '{decision.routed_to}' agent "
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
            routed_to=decision.routed_to
        )
        yield "event: end\ndata: {}\n\n"
        return

    # 1. First Pass: Fetch ONLY documents the user is authorized to see
    # This guarantees we get the top `max_results` valid documents without being pushed out by unauthorized ones.
    with timer.stage("retrieval_authorized"):
        docs = await vector_store.search(
            query=request.query,
            filter={
                "is_active": True,
                "allowed_departments": {"$in": [user_department, "All"]}
            },
            limit=request.max_results,
            score_threshold=request.confidence_threshold
        )

    # 2. Second Pass: If no docs found, check if it's because of permissions (UX optimization)
    unauthorized_docs_found = False
    if not docs:
        with timer.stage("retrieval_unauthorized"):
            # Fetch 1 document without department filters to see if a match exists elsewhere
            unauthorized_check = await vector_store.search(
                query=request.query,
                filter={"is_active": True},
                limit=1,
                score_threshold=request.confidence_threshold
            )
            if unauthorized_check:
                unauthorized_docs_found = True

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

    # Authorization logic
    if not docs:
        if unauthorized_docs_found:
            failure_msg = "I found some information related to your query, but it appears you do not have the required departmental permissions to access it. Please contact the document owner or your administrator for access."
        else:
            failure_msg = "I don't have enough information to answer this question accurately."
            
        yield f"event: message\ndata: {json.dumps({'content': failure_msg})}\n\n"

        if request.session_id:
            await cache.append_chat_message(request.session_id, ChatRole.ASSISTANT, failure_msg)

        yield "event: end\ndata: {}\n\n"
        return

    # Stream the LLM tokens
    full_answer = ""
    stream_error = False
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
        stream_error = True
        logger.error(f"Streaming failed: {e}")
        yield f"event: error\ndata: {json.dumps({'error': 'Stream generation failed'})}\n\n"

    # Post-stream bookkeeping (outside try/finally to avoid yield-in-finally)
    latency_ms = (time.time() - start_time) * 1000

    if request.session_id and full_answer:
        await cache.append_chat_message(request.session_id, ChatRole.ASSISTANT, full_answer)

    if full_answer and not stream_error:
        await _cache_response(
            cache, cache_key,
            answer=full_answer,
            sources=sources,
            confidence="high"
        )

    await _log_query(
        session, request.query, user_id, user_department,
        len(docs), latency_ms, False, "high",
        stage_timings=timer.as_dict(),
        routed_to=RoutedAgent.RAG
    )
    query_count.labels(department=user_department, confidence="high").inc()
    query_latency.labels(department=user_department, cached="false").observe(latency_ms / 1000.0)

    yield "event: end\ndata: {}\n\n"


# ── Private Helpers ───────────────────────────────────────────────────────────


async def _cache_response(
    cache: CacheManager,
    cache_key: str,
    *,
    answer: str,
    sources: List[Dict],
    confidence: str,
    ttl: int = 3600
) -> None:
    """Populate the Redis cache after a successful LLM generation.

    Accepts raw data types so the streaming path doesn't need to
    construct a throwaway QueryResponse Pydantic model.
    """
    cache_data = {
        "answer": answer,
        "sources": sources,
        "confidence": confidence
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