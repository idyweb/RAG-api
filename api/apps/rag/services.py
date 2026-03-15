"""
RAG query service with department-based filtering.

This is the CORE of the multi-tenant system.
Streaming-only architecture — generate_answer_stream is the single code path.
"""

import json
import time
from typing import List, Dict, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

from api.config.settings import settings
from api.apps.rag.schemas import QueryRequest, ChatRole
from api.apps.rag.models import QueryLog
from api.core.vector_store import VectorStore
from api.core.cache import CacheManager
from api.core.llm import generate_answer_stream
from api.core.semantic_router import SemanticRouter, RoutedAgent
from api.utils.logger import get_logger
from api.utils.metrics import query_count, query_latency
from api.utils.pipeline_timer import PipelineTimer
from api.core.langfuse_client import get_langfuse, flush as langfuse_flush

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

    Langfuse spans wrap each pipeline stage in real time so the trace
    shows actual wall-clock durations, not a single 0.01s blob.
    """

    timer = PipelineTimer(department=user_department)
    start_time = time.time()
    lf = get_langfuse()

    # Create the top-level Langfuse trace for the entire query.
    # Environment tag separates dev/staging/prod traces in the dashboard.
    trace = None
    if lf:
        trace = lf.trace(
            name="rag-query",
            input={"query": request.query, "department": user_department},
            user_id=user_id,
            session_id=str(request.session_id) if request.session_id else None,
            metadata={
                "service": settings.APP_NAME,
                "environment": settings.ENVIRONMENT,
            },
            tags=[f"dept:{user_department}", f"env:{settings.ENVIRONMENT}"],
        )

    # Check Cache First
    cache_key = cache.get_key(request.query, user_department)
    with timer.stage("cache_check"):
        cached_result = await cache.get(cache_key)

    if cached_result:
        yield f"event: sources\ndata: {json.dumps(cached_result['sources'])}\n\n"
        yield f"event: message\ndata: {json.dumps({'content': cached_result['answer']})}\n\n"
        yield "event: end\ndata: {}\n\n"

        latency_ms = (time.time() - start_time) * 1000
        await _log_query(
            session, request.query, user_id, user_department,
            len(cached_result['sources']), latency_ms,
            True, cached_result['confidence'],
            stage_timings=timer.as_dict()
        )
        query_count.labels(department=user_department, confidence=cached_result['confidence']).inc()
        query_latency.labels(department=user_department, cached="true").observe(latency_ms / 1000.0)

        if trace:
            trace.update(
                output={"cached": True, "sources_count": len(cached_result['sources'])},
                metadata={"latency_ms": round(latency_ms, 2)},
            )
            langfuse_flush()
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

    # ── Stage 1: Intent Classification ──────────────────────────────────
    intent_span = trace.span(name="intent-classification", input={"query": request.query}) if trace else None
    with timer.stage("intent_classification"):
        decision = await semantic_router.route_query(request.query, user_department)
    if intent_span:
        intent_span.end(output={
            "routed_to": decision.routed_to,
            "confidence": decision.confidence_score,
            "reasoning": decision.reasoning,
        })

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
        if trace:
            trace.update(
                output={"routed_to": decision.routed_to, "answer": msg},
                metadata={"latency_ms": round(latency_ms, 2)},
            )
            langfuse_flush()
        yield "event: end\ndata: {}\n\n"
        return

    # ── Stage 2: Vector Retrieval ───────────────────────────────────────
    retrieval_span = trace.span(
        name="vector-retrieval",
        input={"query": request.query, "department": user_department, "limit": request.max_results},
    ) if trace else None

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

    unauthorized_docs_found = False
    if not docs:
        with timer.stage("retrieval_unauthorized"):
            unauthorized_check = await vector_store.search(
                query=request.query,
                filter={"is_active": True},
                limit=1,
                score_threshold=request.confidence_threshold
            )
            if unauthorized_check:
                unauthorized_docs_found = True

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

    if retrieval_span:
        retrieval_span.end(output={
            "results_count": len(docs),
            "sources": sources,
            "unauthorized_fallback": unauthorized_docs_found,
        })

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

        if trace:
            trace.update(
                output={"answer": failure_msg, "sources_count": 0},
                metadata={"latency_ms": round((time.time() - start_time) * 1000, 2)},
            )
            langfuse_flush()
        yield "event: end\ndata: {}\n\n"
        return

    # ── Stage 3: LLM Generation ────────────────────────────────────────
    generation_span = trace.generation(
        name="llm-generation",
        model=settings.GEMINI_MODEL,
        input={"query": request.query, "context_chunks": len(docs)},
    ) if trace else None

    full_answer = ""
    stream_error = False
    llm_usage: Dict = {}
    try:
        async for chunk in generate_answer_stream(
            query=request.query,
            docs=docs,
            department=user_department,
            chat_history=chat_history,
            usage_out=llm_usage,
        ):
            full_answer += chunk
            yield f"event: message\ndata: {json.dumps({'content': chunk})}\n\n"

    except Exception as e:
        stream_error = True
        logger.error(f"Streaming failed: {e}")
        yield f"event: error\ndata: {json.dumps({'error': 'Stream generation failed'})}\n\n"

    if generation_span:
        end_kwargs: Dict = {"output": full_answer if not stream_error else "STREAM_ERROR"}
        if llm_usage:
            end_kwargs["usage"] = {
                "input": llm_usage.get("prompt_tokens", 0),
                "output": llm_usage.get("completion_tokens", 0),
                "total": llm_usage.get("total_tokens", 0),
            }
        generation_span.end(**end_kwargs)

    # Post-stream bookkeeping
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

    # Finalize Langfuse trace
    if trace:
        trace.update(
            output={
                "answer_length": len(full_answer),
                "sources_count": len(sources),
                "stream_error": stream_error,
            },
            metadata={
                "latency_ms": round(latency_ms, 2),
                "stage_timings": timer.as_dict(),
            },
        )
        langfuse_flush()

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