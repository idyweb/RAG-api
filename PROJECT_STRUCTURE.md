# Project Structure

```
coragem-rag-api/
│
├── 📄 README.md                    ← Start here!
├── 📄 BUILD_PLAN.md                ← Step-by-step build guide
├── 📄 requirements.txt             ← Python dependencies
├── 📄 docker-compose.yml           ← Postgres + Redis + Qdrant
├── 📄 .env.example                 ← Config template
│
├── 📁 api/                         ← Main application
│   ├── 📄 main.py                  ← FastAPI app (Phase 5)
│   │
│   ├── 📁 config/
│   │   └── 📄 settings.py          ← Environment config (Phase 1)
│   │
│   ├── 📁 database/
│   │   ├── 📄 base.py              ← BaseModel CRUD (Phase 1) ⭐ START HERE
│   │   └── 📄 session.py           ← AsyncSession factory (Phase 1)
│   │
│   ├── 📁 apps/                    ← Feature modules (self-contained)
│   │   ├── 📁 auth/                ← Authentication
│   │   │   ├── models.py           ← User, Department (Phase 2)
│   │   │   ├── schemas.py          ← LoginRequest, TokenResponse
│   │   │   ├── services.py         ← login(), verify_token() (Phase 4)
│   │   │   └── routers.py          ← POST /auth/login (Phase 5)
│   │   │
│   │   ├── 📁 documents/           ← Document management
│   │   │   ├── models.py           ← Document, DocumentChunk (Phase 2)
│   │   │   ├── schemas.py          ← DocumentCreate, DocumentResponse
│   │   │   ├── services.py         ← ingest_document() (Phase 4)
│   │   │   └── routers.py          ← POST /documents/ingest (Phase 5)
│   │   │
│   │   ├── 📁 rag/                 ← RAG query module ⭐ THE CORE
│   │   │   ├── models.py           ← QueryLog (analytics)
│   │   │   ├── schemas.py          ← QueryRequest, QueryResponse
│   │   │   ├── services.py         ← rag_query() with dept filtering (Phase 4)
│   │   │   └── routers.py          ← POST /rag/query (Phase 5)
│   │   │
│   │   └── 📁 health/              ← Health checks
│   │       ├── schemas.py          ← HealthResponse
│   │       ├── services.py         ← check_db(), check_redis()
│   │       └── routers.py          ← GET /health (Phase 5)
│   │
│   ├── 📁 core/                    ← Shared infrastructure
│   │   ├── 📄 embeddings.py        ← Embedding generation (Phase 3)
│   │   ├── 📄 vector_store.py      ← Qdrant operations (Phase 3)
│   │   ├── 📄 llm.py               ← LLM generation (Phase 3)
│   │   └── 📄 cache.py             ← Redis caching (Phase 3)
│   │
│   └── 📁 utils/                   ← Helper functions
│       ├── 📄 logger.py            ← Structured logging (Phase 1)
│       ├── 📄 responses.py         ← Standardized responses (Phase 1)
│       ├── 📄 exceptions.py        ← Custom exceptions (Phase 1)
│       └── 📄 security.py          ← JWT, hashing (Phase 1)
│
├── 📁 data/                        ← Mock data for testing
│   ├── 📁 mock_departments/        ← Sample docs (5 departments)
│   │   ├── 📁 sales/               ← Product catalogs, pricing
│   │   ├── 📁 hr/                  ← Leave policies, benefits
│   │   ├── 📁 finance/             ← Expense policies
│   │   ├── 📁 operations/          ← Safety protocols
│   │   └── 📁 manufacturing/       ← Quality standards
│   │
│   └── 📁 scripts/
│       └── 📄 seed_data.py         ← Populate DB (Phase 7)
│
└── 📁 tests/                       ← Test suite
    ├── 📁 unit/                    ← Unit tests (per module)
    └── 📁 integration/             ← Integration tests
        └── 📄 test_dept_isolation.py  ← ⭐ CRITICAL: Test security (Phase 6)
```

## Key Files Explained

### ⭐ Priority 1 (Build First)
1. `api/database/base.py` - BaseModel with CRUD operations
2. `api/config/settings.py` - Environment configuration
3. `api/database/session.py` - Database session management

### 🔐 Security-Critical Files
- `api/apps/rag/services.py` - Department filtering logic
- `api/core/vector_store.py` - Metadata filter enforcement
- `tests/integration/test_dept_isolation.py` - Verify no data leakage

### 🎯 Business Logic Files (Services)
- `api/apps/auth/services.py` - Authentication
- `api/apps/documents/services.py` - Document ingestion + versioning
- `api/apps/rag/services.py` - RAG query with dept filtering

### 🌐 API Layer (Routers)
- Thin wrappers around services
- No logic here (just route definitions)
- Call service functions

## Architecture Rules

**✅ DO:**
- Inherit all models from BaseModel
- Put ALL logic in services
- Use BaseModel methods (no direct DB queries)
- Use async everywhere
- Enforce LIMIT on all queries

**❌ DON'T:**
- Put logic in routers
- Use sync database calls
- Query without limits
- Hardcode configuration
- Skip soft deletes

## Build Sequence

**Phase 1** → Foundation (base.py, settings.py, session.py)
**Phase 2** → Models (User, Department, Document, DocumentChunk)
**Phase 3** → Core (embeddings, vector_store, cache, llm)
**Phase 4** → Services (auth, documents, **rag** ⭐)
**Phase 5** → API (routers, main.py, responses)
**Phase 6** → Testing (dept isolation ⭐)
**Phase 7** → Data & Deployment

## Next Steps

1. **Read**: `README.md` - Project overview
2. **Read**: `BUILD_PLAN.md` - Detailed build guide
3. **Start**: `docker-compose up -d` - Start infrastructure
4. **Build**: Open `api/database/base.py` and let's code together!

---

**Ready to build? Start with `api/database/base.py`. 🚀**
