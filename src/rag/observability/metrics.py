from __future__ import annotations

from prometheus_client import Counter, Histogram, Info

APP_INFO = Info("rag_app", "RAG system application info")

REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "rag_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

INGESTION_CHUNKS = Counter(
    "rag_ingestion_chunks_total",
    "Total chunks ingested",
    ["tenant_id"],
)

ABSTENTION_COUNT = Counter(
    "rag_abstentions_total",
    "Total abstention responses",
)

QUERY_COUNT = Counter(
    "rag_queries_total",
    "Total query requests",
    ["tenant_id"],
)
