# Coragem Enterprise AI Platform

A production-grade, multi-tenant AI Platform and Retrieval-Augmented Generation (RAG) API. The system empowers employees across diverse organizational departments to query internal documents securely using advanced AI. It features strict data isolation, specialized routing, conversational memory, and real-time streaming capabilities.

---

## 1. System Overview

Coragem Enterprise AI operates as a **routing intelligence layer** rather than a standard RAG chatbot. When a user submits a query, the Semantic Router evaluates the intent and directs the query to the appropriate agent:
- **Intelligent RAG**: For policy documents, FAQs, and unstructured data.
- **Specialized Copilots**: (e.g., Sales Intelligence Assistant, ERP Support Copilot) for formatted data retrieval directly from integrated systems like Power BI or Dynamics Business Central. (Phase 2 feature)

### Core Architecture Capabilities
- **Multi-Department Access Controls**: Documents are secured via an `allowed_departments` matrix, allowing policies to span multiple departments while strictly guarding isolated unit data.
- **FastMCP Integration**: The platform acts as an MCP Server, exposing dynamic tools for downstream agent consumption. 
- **Provider-Agnostic Embeddings**: Seamlessly switch between Google Gemini (`gemini-embedding-001`) and Azure OpenAI  using a dynamic Factory Pattern.
- **Conversational Memory**: Redis-backed session history (`session_id`) allows the LLM to remember multi-turn conversations naturally.
- **Streaming Generation**: The `/query/stream` endpoint yields tokens dynamically for low-latency frontend experiences.
- **Observability**: Integrated Prometheus metrics track pipeline latency, token consumption, and routing decisions.

---

## 2. Technology Stack

- **Framework:** FastAPI (Python 3.12)
- **Agent Protocol:** FastMCP (Model Context Protocol)
- **Vector Database:** Pinecone (Serverless)
- **LLM/Embeddings:** Google Gemini / Azure OpenAI
- **Relational DB:** PostgreSQL (Async SQLAlchemy)
- **Caching & Memory:** Redis
- **Infra/Deployment:** Docker Compose, Uvicorn

---

## 3. Getting Started (Local Development)

### Prerequisites
- Docker and Docker Compose
- `uv` (Python Package Manager if running natively)

### Environment Configuration
Create a `.env` file in the root directory based on `.env.example`:

```env
# Application
ENVIRONMENT=development
CORS_ORIGINS=["http://localhost:3000"]
ALLOWED_HOSTS=["*"]

# Security
SECRET_KEY=your_super_secret_jwt_key
ACCESS_TOKEN_EXPIRE_MINUTES=60
COMPANY_NAME=Coragem

# Databases
DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/rag_db
REDIS_URL=redis://redis:6379/0

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=documents

# LLM Providers (Gemini is enabled by default)
EMBEDDING_PROVIDER=gemini  # Or `azure_openai`
GEMINI_API_KEY=your_gemini_api_key

# Azure OpenAI (If enabled)
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_DEPLOYMENT_NAME=text-embedding-3-small
```

### Running the Stack
The stack (API, PostgreSQL, Redis) is managed via Docker Compose:

```bash
docker compose up --build -d
```

### Database Migrations
Initialize the PostgreSQL schema dynamically:

```bash
docker exec -it api alembic upgrade head
```

The API will be available at: `http://localhost:8000`
Interactive Swagger Documentation: `http://localhost:8000/docs`

---

## 4. API Endpoints

### Authentication
- `POST /api/v1/auth/login`: Authenticate and receive a JWT access token.
- `POST /api/v1/auth/register`: Register a new employee with department assignment.

### Document Management (Ingestion)
- `POST /api/v1/documents/ingest/text`: Ingest raw text directly into Pinecone and Postgres.
- `POST /api/v1/documents/ingest/pdf`: Upload and parse a PDF, splitting it into semantic chunks for vector storage.

### Core RAG & Querying
*(Requires `Authorization: Bearer <token>`)*

**Standard JSON Query:**
```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
-H "Authorization: Bearer YOUR_TOKEN" \
-H "Content-Type: application/json" \
-d '{
  "query": "What are the rules for maternal leave?",
  "session_id": "hr-session-01",
  "confidence_threshold": 0.5
}'
```

**Real-Time Streaming Query (SSE):**
```bash
curl -N -X POST "http://localhost:8000/api/v1/rag/query/stream" \
-H "Authorization: Bearer YOUR_TOKEN" \
-H "Content-Type: application/json" \
-d '{
  "query": "Summarize the Q3 product catalog.",
  "session_id": "sales-session-99",
  "confidence_threshold": 0.5
}'
```

### Protocol Servers
- `GET /mcp/sse`: FastMCP Server-Sent Events endpoint for dynamic agent tool discovery.

---

## 5. Project Architecture structure

```text
api/
├── apps/               # Business Logic Domains
│   ├── agents/         # FastMCP Server and Tool Definitions
│   ├── auth/           # JWT, Users, RBAC, Channel Identities
│   ├── documents/      # Parsing, Chunking, Multi-Department auth
│   ├── rag/            # Pipeline execution
│   └── health/         # System statuses
├── core/               # App Infrastructure
│   ├── cache.py        # Redis Memory & caching
│   ├── embeddings.py   # Factory pattern for multi-vendor Embeddings
│   ├── llm.py          # LLM connection (Gemini)
│   ├── semantic_router.py # Intent classification engine
│   ├── vector_store.py # Pinecone search/upsert operations
│   └── config.py       # Pydantic Settings
├── db/                 # PostgreSQL Async Session Management
├── utils/              # Obsevability (Metrics, Pipeline Timers, Logs)
└── tests/              # Golden Dataset and CI/CD Pytest suite
```
