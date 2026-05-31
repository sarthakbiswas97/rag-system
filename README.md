# RAG Platform

[![Backend CI](https://github.com/sarthakbiswas97/rag-system/actions/workflows/backend.yml/badge.svg)](https://github.com/sarthakbiswas97/rag-system/actions/workflows/backend.yml)
[![Frontend CI](https://github.com/sarthakbiswas97/rag-system/actions/workflows/frontend.yml/badge.svg)](https://github.com/sarthakbiswas97/rag-system/actions/workflows/frontend.yml)

A production-grade Retrieval-Augmented Generation (RAG) system with multi-tenancy, hybrid search, real-time verification, and zero-hallucination safeguards.

## Overview

This system ingests documents, indexes them into a vector database, and answers user questions by retrieving relevant context and generating cited responses. It features NLI-based verification, abstention when confidence is low, streaming responses, and LLM fallback for resilience.

**Backend:** FastAPI + Python 3.12  
**Frontend:** Next.js 16 + React 19 + Tailwind CSS 4  
**Vector DB:** Qdrant 1.12.6  
**Relational DB:** PostgreSQL 16 (production) / SQLite (development)  
**Cache/Queue:** Redis 7

---

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 22+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Docker & Docker Compose (optional, for infrastructure)

### 1. Clone & Setup

```bash
git clone https://github.com/sarthakbiswas97/rag-system.git
cd rag-system

# Backend dependencies
uv sync --all-extras

# Frontend dependencies
cd dashboard && npm install && cd ..

# Environment
cp .env.example .env
# Edit .env and set OPENAI_API_KEY and ADMIN_API_KEY
```

### 2. Run with Docker Compose (Recommended)

```bash
# Start all services (PostgreSQL, Qdrant, Redis, backend, frontend)
docker compose up -d

# Or use Make
make docker-up
```

Services available:
- Backend API: http://localhost:8000
- Frontend Dashboard: http://localhost:3000
- Qdrant Dashboard: http://localhost:6333

### 3. Run Locally (Development)

```bash
# Start infrastructure only
make dev

# Run database migrations
make db-upgrade

# Start backend server
make run

# In another terminal, start frontend
cd dashboard && npm run dev
```

### 4. Create a Tenant

```bash
curl -X POST http://localhost:8000/v1/register \
  -H "Content-Type: application/json" \
  -d '{"name": "My Team", "email": "team@example.com"}'
```

Use the returned `api_key` in the `X-API-Key` header for all subsequent requests.

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│   Next.js   │────▶│   FastAPI   │────▶│  Query Cache    │
│  Dashboard  │◄────│   Backend   │◄────│     (Redis)     │
└─────────────┘     └──────┬──────┘     └─────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌─────────────────┐
│   Qdrant     │  │  PostgreSQL  │  │  Session Store  │
│ Vector Store │  │   (Tenants,  │  │     (Redis)     │
│(Dense+Sparse)│  │  Documents,  │  └─────────────────┘
└──────────────┘  │   Usage)     │
                  └──────────────┘
```

### Core Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Layer** | FastAPI + Pydantic | REST API, dependency injection, middleware |
| **Embedding** | sentence-transformers (BAAI/bge-small-en, 384-dim) | Dense vector generation; ONNX Runtime opt-in |
| **Sparse Retrieval** | Custom `SparseEmbedder` | TF-based sparse vectors in Qdrant (replaces BM25) |
| **Vector Store** | Qdrant 1.12.6 | HNSW index, INT8 quantization, 6 shards, dot-product distance |
| **LLM** | OpenAI API (gpt-4o-mini) | RAG generation with streaming SSE; fallback model support |
| **Verification** | DeBERTa-v3-large MNLI | Entailment checking, citation validation, abstention decisions |
| **Reranker** | cross-encoder/ms-marco-MiniLM | Optional cross-encoder reranking post-retrieval |
| **Cache** | Redis | Query response cache (TTL 5min), session store, rate limiting |
| **Database** | SQLAlchemy 2.0 async | Tenants, documents, usage events; SQLite dev / PostgreSQL prod |
| **Observability** | structlog + Prometheus | JSON-structured logs, request metrics, ingestion counters |

### Multi-Tenancy

- **Authentication:** `X-API-Key` header → SHA256-hashed API key lookup
- **Data Isolation:** All Qdrant chunks include `tenant_id` payload; every search filters by tenant
- **Document Lifecycle:** Tracks `ACTIVE` / `UPDATING` / `DELETED` status with soft deletes
- **Rate Limiting:** Per-tenant Redis counters (60 queries/min, 10 ingestion/min)
- **Concurrency:** Max 3 concurrent ingestion jobs per tenant via Redis semaphore

---

## Data Flow

### Ingestion Pipeline

```
Upload (PDF/TXT/MD/CSV)
    │
    ▼
Load ──▶ Chunk (512 tokens, 64 overlap)
    │
    ▼
Deduplicate (SHA256 content hash, per-tenant)
    │
    ▼
Embed ──▶ Dense (384-dim) + Sparse (BM25-like)
    │
    ▼
Store ──▶ Qdrant (dense + sparse vectors)
    │
    ▼
Metadata ──▶ PostgreSQL (document record)
    │
    ▼
Cache Invalidation ──▶ Redis (clear tenant query cache)
```

**Optimizations:**
- Batch embedding (max 64 chunks per batch)
- Single DB commit for batched document ingestion
- Content deduplication prevents re-ingesting identical documents

### Query Pipeline

```
User Question
    │
    ▼
[Optional] Conversational Rewrite (session history)
    │
    ▼
Dense Search ──▶ Qdrant (filtered by tenant_id)
Sparse Search ──▶ Qdrant sparse vectors
    │
    ▼
Reciprocal Rank Fusion (RRF, K=60)
    │
    ▼
[Optional] Cross-encoder Reranking
    │
    ▼
Build RAG Prompt (top chunks + citations)
    │
    ▼
LLM Generation ──▶ OpenAI (streaming supported)
    │
    ▼
Citation Parsing + Abstention Detection
    │
    ▼
NLI Verification (entailment per sentence)
    │
    ▼
Abstention Decision ──▶ Return answer or "I don't know"
```

**Verification Signals:**
1. Retrieval score below threshold → abstain
2. Reranker score below threshold → abstain
3. Citation coverage too low → abstain
4. Faithfulness score (NLI) below threshold → abstain

---

## API Reference

All authenticated endpoints require `X-API-Key` header.

### Authentication & Tenant

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/v1/register` | — | Create tenant, returns API key |
| `GET` | `/v1/me` | API Key | Get current tenant |
| `PUT` | `/v1/me` | API Key | Update tenant name |
| `GET` | `/v1/me/stats` | API Key | Chunk count |
| `GET` | `/v1/me/usage` | API Key | Usage events (query, ingest) |

### Documents

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/v1/documents` | API Key | List documents (paginated) |
| `DELETE` | `/v1/documents/{id}` | API Key | Soft delete + remove vectors |
| `PUT` | `/v1/documents/{id}` | API Key | Replace document (re-ingest) |

### Ingestion

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/v1/ingest` | API Key | Upload files (multipart, max 50MB) |
| `POST` | `/v1/ingest/async` | API Key | Async ingestion (returns job ID) |
| `GET` | `/v1/ingest/jobs/{id}` | API Key | Check async job status |

### Query

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/v1/query` | API Key | Standard RAG query |
| `POST` | `/v1/query/stream` | API Key | SSE streaming response |

### Admin

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/admin/tenants` | Admin Key | List all tenants |
| `POST` | `/admin/tenants` | Admin Key | Create tenant manually |
| `DELETE` | `/admin/tenants/{id}` | Admin Key | Delete tenant |

### Health & Metrics

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/v1/health` | — | Health check (DB + Qdrant + Redis) |
| `GET` | `/v1/livez` | — | Liveness probe |
| `GET` | `/metrics` | — | Prometheus metrics |

---

## Configuration

All configuration is via environment variables (see `src/rag/config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key for LLM |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence-transformers model |
| `EMBEDDING_BACKEND` | `pytorch` | `pytorch` or `onnx` |
| `EMBEDDING_ONNX_PROVIDER` | `CPUExecutionProvider` | ONNX execution provider |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_COLLECTION` | `documents` | Collection name |
| `QDRANT_SHARD_NUMBER` | `6` | Number of shards |
| `QDRANT_REPLICATION_FACTOR` | `1` | Replication factor |
| `CHUNK_SIZE` | `512` | Document chunk size (tokens) |
| `CHUNK_OVERLAP` | `64` | Chunk overlap (tokens) |
| `LLM_MODEL` | `gpt-4o-mini` | Primary LLM model |
| `LLM_FALLBACK_MODEL` | *(empty)* | Fallback model name |
| `LLM_FALLBACK_API_KEY` | *(empty)* | Fallback API key (if different) |
| `LLM_TEMPERATURE` | `0.1` | Generation temperature |
| `LLM_MAX_TOKENS` | `1024` | Max tokens per response |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/rag.db` | SQLAlchemy database URL |
| `DB_POOL_SIZE` | `5` | Connection pool size |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `ADMIN_API_KEY` | *(empty)* | Admin API key for management endpoints |
| `ENABLE_QUERY_CACHE` | `true` | Enable Redis query caching |
| `QUERY_CACHE_TTL` | `300` | Query cache TTL (seconds) |
| `ENABLE_RATE_LIMITING` | `true` | Enable per-tenant rate limits |
| `RATE_LIMIT_QUERIES` | `60` | Queries per minute per tenant |
| `RATE_LIMIT_INGESTION` | `10` | Ingestions per minute per tenant |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed CORS origins |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Development

### Backend

```bash
# Run all tests
make test

# Run with coverage
uv run pytest --cov=src/rag --cov-report=term-missing

# Lint and format
make lint
make format

# Run evaluation
make eval DATASET=data/eval/my-dataset.json

# Run locally
make run
```

### Frontend

```bash
cd dashboard

# Dev server
npm run dev

# Lint
npm run lint

# Production build
npm run build
```

### Database Migrations

```bash
# Create migration
make db-migrate MSG="add users table"

# Apply migrations
make db-upgrade

# Rollback one
make db-downgrade
```

### Load Testing

```bash
# Seed benchmark data
make seed-bench DOCS=100

# Run load test
make load-test USERS=50 RATE=10 TIME=60s
```

---

## Deployment

### Docker

```bash
# Build all images
docker compose build

# Start production stack
docker compose up -d
```

The backend Dockerfile uses multi-stage builds with `gunicorn` + `uvicorn.workers.UvicornWorker`. Worker count auto-scales from 2 to 8 based on CPU cores (override with `WEB_CONCURRENCY`).

### CI/CD

GitHub Actions workflows:
- **Backend CI:** lint → test → docker build/push to GHCR
- **Frontend CI:** lint → build → docker build/push to GHCR

Images pushed to `ghcr.io/sarthakbiswas97/rag-system-backend` and `ghcr.io/sarthakbiswas97/rag-system-dashboard`.

### Railway

The project includes `railway.toml` for Railway deployment. Set the required environment variables in the Railway dashboard and link the repository.

---

## Project Structure

```
rag-system/
├── src/rag/                  # Backend source
│   ├── api/                  # FastAPI routers, schemas, middleware
│   ├── ingestion/            # Document loading, chunking, embedding
│   ├── retrieval/            # Vector search, hybrid fusion, reranking
│   ├── generation/           # LLM client, prompt builder, streaming
│   ├── verification/         # NLI entailment, citation validation
│   ├── tenancy/              # Auth, repositories, models
│   ├── session/              # Redis-backed conversation sessions
│   ├── models/               # Core dataclasses
│   ├── observability/        # Logging, metrics
│   ├── config.py             # Pydantic settings
│   └── main.py               # App factory
├── dashboard/                # Next.js frontend
│   ├── src/app/              # App router pages
│   ├── src/lib/api.ts        # API client
│   └── src/components/       # React components
├── tests/                    # Test suite
│   ├── unit/                 # 439 unit tests
│   ├── integration/          # 19 integration tests
│   └── load/                 # Locust load tests
├── scripts/                  # Evaluation, seeding, ingestion
├── migrations/               # Alembic database migrations
├── docker-compose.yml        # Local infrastructure stack
├── Dockerfile                # Backend container
├── pyproject.toml            # Python dependencies
└── Makefile                  # Common commands
```

---

## Key Design Decisions

1. **Hybrid Search:** Dense vectors for semantic similarity + sparse vectors for keyword matching, fused via Reciprocal Rank Fusion (RRF).

2. **Sparse Vectors in Qdrant:** Instead of maintaining an in-memory BM25 index, sparse embeddings are stored natively in Qdrant alongside dense vectors. This eliminates memory pressure and simplifies scaling.

3. **Verification Pipeline:** Every answer is checked with an NLI model. If the answer isn't sufficiently supported by retrieved context, the system abstains rather than hallucinating.

4. **Streaming with Verification:** The `/v1/query/stream` endpoint streams tokens to the client, then runs verification after generation completes. If verification fails, an `abstention` event is sent.

5. **ONNX Opt-in:** Embedding backend defaults to PyTorch (best on GPU/Linux) but can switch to ONNX Runtime via `EMBEDDING_BACKEND=onnx` for CPU-optimized inference.

6. **Sharding Strategy:** Qdrant collections use 6 shards with hash-based routing. No online resharding is supported; migration requires collection recreation (see `MIGRATIONS.md`).

---

## License

MIT
