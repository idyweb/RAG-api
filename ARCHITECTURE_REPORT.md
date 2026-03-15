# Coragem RAG API — Technical Architecture Report

**Date:** 2026-03-11
**Version:** 1.0.0
**Prepared for:** Management Review

---

## 1. Executive Summary

The Coragem RAG (Retrieval-Augmented Generation) API is an enterprise-grade backend that enables employees to query company documents through AI-powered natural language conversations. It is designed to serve as the intelligence layer behind chat platforms such as **Microsoft Teams** and **WhatsApp**.

**Key capabilities:**
- PDF document ingestion with intelligent chunking and versioning
- AI-powered question answering grounded strictly in company documents
- Department-level access isolation — employees only see documents they are authorised to access
- Real-time streaming responses (token-by-token)
- Multi-agent semantic routing — queries are classified and directed to the correct data source
- Extensible architecture ready for Power BI, Business Central ERP, and SAP integration

**Technology stack:** FastAPI · PostgreSQL (async) · Pinecone (vector database) · Redis (caching) · Celery (task queue) · Google Gemini 2.5 Flash (LLM + embeddings)

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│              (Microsoft Teams / WhatsApp / Web)                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTPS / SSE
┌──────────────────────────▼──────────────────────────────────────┐
│                      FastAPI Application                        │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────────┐ │
│  │ Auth API │  │ Documents API│  │       RAG Query API        │ │
│  │ (JWT)    │  │ (Ingest/CRUD)│  │ (Stream answers via SSE)  │ │
│  └──────────┘  └──────┬───────┘  └────────────┬──────────────┘ │
└─────────────────────────┼──────────────────────┼────────────────┘
                          │                      │
          ┌───────────────┘                      │
          ▼                                      ▼
┌──────────────────┐               ┌──────────────────────────┐
│   Celery Worker  │               │    Semantic Router        │
│  (Background     │               │  (Intent Classification)  │
│   Ingestion)     │               └────────────┬─────────────┘
└────────┬─────────┘                            │
         │                        ┌─────────────┼─────────────┐
         │                        ▼             ▼             ▼
         │                    ┌───────┐   ┌──────────┐  ┌──────────┐
         │                    │  RAG  │   │ Power BI │  │ ERP API  │
         │                    │ Agent │   │  Agent   │  │  Agent   │
         │                    └───┬───┘   └──────────┘  └──────────┘
         │                        │        (Future)      (Future)
         ▼                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                 │
│  ┌──────────────┐  ┌───────────────┐  ┌───────────────────────┐ │
│  │  PostgreSQL   │  │   Pinecone    │  │       Redis           │ │
│  │  (Documents,  │  │ (Vectors +    │  │ (Query cache,         │ │
│  │   Chunks,     │  │  Metadata     │  │  Chat history,        │ │
│  │   Users)      │  │  Filtering)   │  │  Dept invalidation)   │ │
│  └──────────────┘  └───────────────┘  └───────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Document Ingestion Pipeline

### 3.1 Upload Flow

When a user uploads a PDF, the system follows this pipeline:

```
PDF Upload (max 10 MB)
    │
    ▼
Text Extraction (pymupdf4llm)
    │  Preserves: headers, tables, lists, reading order
    │  Output: Markdown-formatted text
    │
    ▼
Celery Task Dispatched (HTTP 202 Accepted)
    │  Client receives task_id for polling
    │
    ▼
Background Worker Executes:
    ├─ 1. Validate department & permissions
    ├─ 2. Create Document record in PostgreSQL
    ├─ 3. Determine chunking strategy
    ├─ 4. Generate embeddings (batch)
    ├─ 5. Upsert vectors to Pinecone (batch)
    ├─ 6. Deactivate previous document versions
    └─ 7. Invalidate department cache in Redis
```

The ingestion is fully asynchronous — the API returns immediately with a task ID, and the client polls `GET /tasks/{task_id}` for status updates. This prevents HTTP timeouts on large documents.

### 3.2 Hierarchical Chunking Strategy (Parent–Child Architecture)

This is the core innovation in our document processing. Rather than naively splitting documents into uniform pieces, we use a **two-level hierarchy** that preserves document structure while enabling precise search.

#### How it works:

**Step 1 — Parent Chunks (Context Windows)**

The document's Markdown is split by headers (H1 through H6). Each headed section becomes a candidate parent chunk. Adjacent small sections are merged until they reach 2,000–4,000 characters, ensuring each parent contains a meaningful, self-contained block of content.

```
┌──────────────────────────────────────────────────┐
│ PARENT CHUNK (2,000–4,000 chars)                 │
│                                                  │
│ "## Leave Policy                                 │
│  All employees are entitled to 20 days of        │
│  annual leave per calendar year. Leave must be    │
│  requested at least 2 weeks in advance through   │
│  the HR portal. Unused leave may be carried       │
│  over up to a maximum of 5 days..."              │
│                                                  │
│  → Stored in PostgreSQL (NOT embedded)           │
│  → Used as LLM context after child retrieval     │
└──────────────────────────────────────────────────┘
```

**Parent chunks are stored in the database but are NOT sent to Pinecone.** They serve as the broader context window that gets fed to the LLM once a relevant child chunk is found.

**Step 2 — Child Chunks (Search Targets)**

Each parent is subdivided into overlapping 500-character windows with a 50-character overlap between consecutive children.

```
Parent: [===============================================]
         ▲                                             ▲
         0                                          4000 chars

Child 1: [==========]
Child 2:       [==========]     ← 50-char overlap
Child 3:             [==========]
Child 4:                   [==========]
  ...

Each child:
  → Embedded as a 3,072-dimension vector (Gemini embedding-001)
  → Upserted to Pinecone with full metadata
  → References its parent_chunk_id for context retrieval
```

**Why this matters:**

| Approach | Problem |
|----------|---------|
| Large chunks only | Poor search precision — irrelevant content dilutes the match |
| Small chunks only | Lost context — the LLM sees a sentence fragment without surrounding information |
| **Parent–Child** | **Best of both: precise search on children, rich context from parents** |

When a user asks a question, the system searches against the small, precise child chunks. Once the most relevant children are found, their parent chunks are retrieved from PostgreSQL to give the LLM the full surrounding context needed to generate a complete, accurate answer.

### 3.3 Flat Chunking (Fallback)

For unstructured plain text without Markdown headers, the system falls back to fixed-size chunking:
- **500 words** per chunk with **50-word overlap**
- All chunks are embedded and stored in Pinecone
- No parent–child hierarchy

### 3.4 Batch Processing Performance

Document ingestion uses batch operations to minimise API calls:

| Operation | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Pinecone upserts | 1 API call per vector | 1 call per 100 vectors | **~100x fewer API calls** |
| Version deactivation | Sequential updates | 50 concurrent updates per batch | **~50x faster** |

A 50-page document producing 200 chunks now requires 2 Pinecone API calls instead of 200.

### 3.5 Document Versioning

When a new version of a document is uploaded with the same title and department:
1. The system detects existing versions and increments the version number
2. All previous versions are soft-deactivated (`is_active = False`) in both PostgreSQL and Pinecone
3. Only the latest version appears in search results
4. Old versions remain in the database for audit purposes

---

## 4. Query & Answer Pipeline

### 4.1 End-to-End Query Flow

```
User: "What is the maternity leave policy?"
  │
  ▼
┌─ STEP 1: Cache Check ──────────────────────────────────────────┐
│  Key: rag:{department}:{md5(query + department)}               │
│  Hit? → Return cached answer instantly (~5ms)                  │
│  Miss? → Continue to Step 2                                    │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ STEP 2: Semantic Routing ─────────────────────────────────────┐
│  LLM classifies intent → Routes to correct agent              │
│  "maternity leave policy" → RAG Agent (knowledge question)    │
│  "show me Q3 revenue" → Power BI Agent (analytics)            │
│  "PO-2024-0012 status" → ERP Agent (transaction lookup)       │
│  Latency: ~500ms                                               │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ STEP 3: Two-Pass Vector Search ───────────────────────────────┐
│  Pass 1: Department-scoped search in Pinecone                  │
│    Filter: is_active=true AND allowed_departments ∋ user_dept  │
│    Returns: Top 5 matching child chunks (score ≥ 0.7)         │
│                                                                │
│  Pass 2 (only if Pass 1 returns nothing):                     │
│    Search without department filter                            │
│    If results found → "Document exists but you lack access"   │
│    If still nothing → "No relevant documents found"           │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ STEP 4: LLM Answer Generation (Streaming) ───────────────────┐
│  Context: Retrieved chunks labelled by document title          │
│  Model: Gemini 2.5 Flash (temperature 0.1, max 4096 tokens)  │
│  Output: Token-by-token via Server-Sent Events (SSE)          │
│                                                                │
│  "According to the **Employee Handbook**, all employees are    │
│   entitled to 16 weeks of maternity leave..."                  │
└────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ STEP 5: Post-Response ───────────────────────────────────────┐
│  • Cache the full answer (TTL: 1 hour)                        │
│  • Append to chat history (Redis, max 50 messages/session)    │
│  • Log metrics (latency, confidence, department)              │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 Streaming Response Format

The API uses Server-Sent Events (SSE) to stream responses in real time:

```
event: sources
data: [{"title": "Employee Handbook", "relevance_score": 0.94, ...}]

event: message
data: {"content": "According to the "}

event: message
data: {"content": "**Employee Handbook**"}

event: message
data: {"content": ", all employees are entitled to..."}

event: end
data: {}
```

This delivers a ChatGPT-like streaming experience — the user sees tokens appear as they are generated, with zero buffering delay.

### 4.3 Citation Strategy

The LLM is instructed to cite documents by their **actual title** woven into the natural language response, not by numbered references. This is critical because the primary deployment targets (Teams, WhatsApp) are plain-text chat platforms that do not support clickable UI elements.

| Platform | Citation Format |
|----------|----------------|
| WhatsApp/Teams | *"According to the **Employee Handbook**..."* |
| Web Dashboard (future) | Can toggle to `[1]` format via API parameter |

### 4.4 Conversational Memory

Each chat session maintains a rolling history in Redis:
- **Max 50 messages** per session (bounded by `LTRIM`)
- **24-hour TTL** — sessions expire after inactivity
- **Last 10 messages** included in the LLM context for multi-turn conversations
- Atomic writes via Redis pipeline (`RPUSH` + `LTRIM` + `EXPIRE` in a single round trip)

---

## 5. Department Isolation & Access Control

### 5.1 Multi-Tenancy Model

Every document has an `allowed_departments` array that controls which departments can access it. This is enforced at **three independent layers** — a defence-in-depth approach where no single layer failure exposes data.

```
┌───────────────────────────────────────────────────────────────┐
│                    DOCUMENT: Employee Handbook                 │
│                    allowed_departments: [HR, Sales, Finance]  │
│                                                               │
│   ┌─────────────────────────────────────────────────────────┐ │
│   │ LAYER 1: PostgreSQL                                     │ │
│   │ WHERE allowed_departments @> ARRAY['Sales']             │ │
│   │ → SQL-level filter before data leaves the database      │ │
│   └─────────────────────────────────────────────────────────┘ │
│                                                               │
│   ┌─────────────────────────────────────────────────────────┐ │
│   │ LAYER 2: Pinecone (Vector Database)                     │ │
│   │ filter: {"allowed_departments": {"$in": ["Sales"]}}     │ │
│   │ → Vector search only returns permitted documents        │ │
│   └─────────────────────────────────────────────────────────┘ │
│                                                               │
│   ┌─────────────────────────────────────────────────────────┐ │
│   │ LAYER 3: Redis Cache                                    │ │
│   │ key: rag:Sales:{query_hash}                             │ │
│   │ → Department in cache key prevents cross-dept leaks     │ │
│   └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

### 5.2 Access Scenarios

| Scenario | Behaviour |
|----------|-----------|
| HR uploads a document for HR only | `allowed_departments: ["HR"]` — only HR employees see it |
| HR uploads a company-wide policy | `allowed_departments: ["HR", "Sales", "Finance", ...]` — visible to all listed departments |
| Sales user queries an HR-only doc | Pinecone filter excludes it; two-pass search detects it exists and returns "You don't have access" |
| Permissions updated post-upload | `PATCH /documents/{id}/permissions` updates PostgreSQL, Pinecone metadata, and invalidates Redis cache atomically |
| IT department | Has admin-level access — can read and manage documents from any department |

### 5.3 Permission Updates

The `PATCH /api/v1/documents/{id}/permissions` endpoint allows document owners (or IT admins) to modify access after upload. This triggers a three-system atomic update:

1. **PostgreSQL** — Updates `allowed_departments` on the Document and all its chunks
2. **Pinecone** — Updates metadata on all associated vectors (concurrent batch of 50)
3. **Redis** — Invalidates cached queries for both the old and new department sets

---

## 6. Semantic Router (Multi-Agent Architecture)

### 6.1 Design Pattern: Microkernel

The semantic router uses a **microkernel pattern** — a lightweight core that classifies intent and dispatches to registered agents. New data sources are added by registering an agent; the core routing logic never changes.

```
┌────────────────────────────────────────────────────┐
│              SEMANTIC ROUTER (Core)                 │
│                                                    │
│  Input: "What is our maternity leave policy?"     │
│                                                    │
│  Decision Tree:                                    │
│    1. Finding a specific document? → doc_search    │
│    2. Live data with reference #?  → erp/bi/gtm   │
│    3. Knowledge/policy/how-to?     → rag ✓        │
│    4. Out of scope?                → unknown       │
│                                                    │
│  Output: {agent: "rag", confidence: 0.95}         │
└────────────────────────────────────────────────────┘
```

### 6.2 Registered Agents

| Agent | Purpose | Status |
|-------|---------|--------|
| **rag** | Knowledge, policies, how-to questions | **Active** |
| **document_search** | Finding specific files/documents | **Active** |
| **power_bi** | Analytics, dashboards, KPI trends | Planned |
| **erp_api** | Purchase orders, invoices, stock levels (Business Central) | Planned |
| **gtm_api** | Sales routes, pipeline, distributor data | Planned |
| **qms** | Quality management, equipment OEE | Planned |
| **unknown** | Out-of-scope / nonsensical queries | **Active** (fallback) |

### 6.3 Routing Performance

- **Model:** Gemini 2.5 Flash (fast, low-cost classification)
- **Latency:** ~500ms per classification
- **LRU Cache:** 256 entries — identical queries return cached routing decisions in O(1) time
- **Retry:** 3 attempts with exponential backoff (2–10 seconds)
- **Accuracy target:** >95% correct routing (validated by golden dataset tests)

---

## 7. Infrastructure & Performance

### 7.1 Connection Pooling

| Resource | Pool Size | Max | Recycle |
|----------|-----------|-----|---------|
| PostgreSQL | 20 connections | 60 (20 + 40 overflow) | 30 minutes |
| Redis | 50 connections | — | Keep-alive enabled |
| HTTP (LLM) | 20 keep-alive | 100 max | — |

### 7.2 Caching Strategy

| Cache | TTL | Invalidation |
|-------|-----|-------------|
| RAG query results | 1 hour | On document ingest or permission change |
| Chat history | 24 hours | Automatic expiry |
| Semantic route decisions | In-memory LRU (256 entries) | Eviction on capacity |

### 7.3 Key Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `LLM_MAX_TOKENS` | 4,096 | Prevents truncated responses; ~3,000 words |
| `LLM_TEMPERATURE` | 0.1 | Low variance for factual accuracy |
| `EMBEDDING_DIMENSION` | 3,072 | Gemini embedding-001 model output |
| `SEARCH_TOP_K` | 5 (max 10) | Balance between relevance and context window |
| `SCORE_THRESHOLD` | 0.7 | Minimum cosine similarity for retrieval |
| `MAX_DOCUMENT_SIZE` | 10 MB | PDF upload limit |
| `MAX_CHAT_MESSAGES` | 50 | Bounded memory per session |
| `RATE_LIMIT` | 10 req/min | Per-user query throttling |

### 7.4 Embedding Provider Support

The system supports two embedding providers, switchable via environment variable:

| Provider | Model | Dimensions | Use Case |
|----------|-------|-----------|----------|
| **Google Gemini** | gemini-embedding-001 | 3,072 | Development / free tier |
| **Azure OpenAI** | text-embedding-ada-002 | 1,536 | Production / enterprise |

### 7.5 Background Processing (Celery)

Document ingestion runs in Celery workers to avoid blocking the API:
- **Max retries:** 3 (with 10-second delay between attempts)
- **Acknowledgement:** `acks_late=True` — task only marked complete after successful processing
- **Resource sharing:** Each worker maintains a single event loop, DB session factory, and vector store client (lazy-loaded)

---

## 8. API Endpoints Summary

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Register new user |
| POST | `/api/v1/auth/login` | Authenticate and receive JWT |
| POST | `/api/v1/auth/refresh` | Refresh access token |

### Document Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/documents/ingest/pdf` | Upload and ingest PDF (async) |
| GET | `/api/v1/documents/tasks/{id}` | Poll ingestion task status |
| PATCH | `/api/v1/documents/{id}/permissions` | Update department access |

### RAG Query
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/rag/query/stream` | Stream AI-powered answer (SSE) |

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness probe (checks DB + Redis) |

---

## 9. Security Summary

| Control | Implementation |
|---------|---------------|
| **Authentication** | JWT (HS256, 15-minute access token, 7/30-day refresh) |
| **Authorisation** | Department-based access on every query and document operation |
| **Data isolation** | Three-layer enforcement (PostgreSQL, Pinecone, Redis) |
| **Input validation** | Pydantic schemas with sanitisation (strip, deduplicate, reject empty) |
| **Rate limiting** | 10 queries/minute per user |
| **File validation** | PDF-only, 10 MB max, empty-content rejection |
| **Grounding** | LLM answers strictly from provided context — no external knowledge |

---

## 10. Future Integration Points

The multi-agent architecture is specifically designed to accommodate future data sources without modifying the core system:

| Integration | Agent | How It Connects |
|-------------|-------|-----------------|
| **Power BI** | `power_bi` | Register agent that calls Power BI REST API for dashboards/KPIs |
| **Business Central ERP** | `erp_api` | Register agent that queries ERP endpoints for POs, invoices, stock |
| **SAP** | `erp_api` | Same agent pattern, different API backend |
| **GTM/Sales Tools** | `gtm_api` | Register agent for distributor routes and sales pipeline |
| **QMS** | `qms` | Register agent for quality metrics and equipment OEE |

Each integration requires only:
1. Implement the `BaseAgent` protocol (a `name`, `description`, and `execute()` method)
2. Register it with the semantic router at startup
3. Update the router's system prompt with disambiguation examples

The core routing, caching, streaming, and authentication infrastructure is reused across all agents.

---

*This report describes the system as of version 1.0.0 (2026-03-11). The RAG agent is fully operational; Power BI, ERP, and GTM agents are architecturally prepared and pending API discovery with the respective platform teams.*
