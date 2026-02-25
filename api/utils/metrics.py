"""
Centralized Prometheus metrics.

All application metrics are defined here to prevent duplication
and ensure consistent labeling across modules.
"""

from prometheus_client import Counter, Histogram


# ── RAG Query Metrics ─────────────────────────────────────────────────────────

query_count = Counter(
    "rag_queries_total",
    "Total RAG queries processed",
    ["department", "confidence"]
)

query_latency = Histogram(
    "rag_query_latency_seconds",
    "RAG query end-to-end latency",
    ["department", "cached"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)


# ── Document Ingestion Metrics ────────────────────────────────────────────────

ingestion_count = Counter(
    "document_ingestion_total",
    "Total documents ingested",
    ["department", "status"]
)

ingestion_latency = Histogram(
    "document_ingestion_latency_seconds",
    "Document ingestion job latency",
    ["department"],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)


# ── Pipeline Stage Metrics ────────────────────────────────────────────────────

pipeline_stage_latency = Histogram(
    "rag_pipeline_stage_seconds",
    "Per-stage latency within the RAG pipeline",
    ["stage", "department"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

