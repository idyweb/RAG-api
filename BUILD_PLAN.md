# BUILD PLAN - Step-by-Step Guide

**We'll build this together, file by file. NO rushing.**

## Phase 1: Foundation (Files 1-3)

### 1. `api/database/base.py` - BaseModel CRUD
**What we'll implement:**
- [ ] Common fields (id, created_at, updated_at, is_deleted)
- [ ] `create()` - Create new record
- [ ] `get_by_id()` - Retrieve single record
- [ ] `get_all()` - List with MANDATORY limit (default 100, max 1000)
- [ ] `update_by_id()` - Update record
- [ ] `soft_delete()` - Mark as deleted (preferred)
- [ ] `hard_delete()` - Permanent delete (use sparingly)

**Why this is critical:**
- All models inherit from this
- Enforces consistent CRUD across entire app
- Prevents N+1 queries, unbounded selects
- Type-safe (Generic[T] pattern)

---

### 2. `api/config/settings.py` - Environment Config
**What we'll implement:**
- [ ] Pydantic BaseSettings class
- [ ] Database URL (Postgres)
- [ ] Redis URL
- [ ] Qdrant URL + API key
- [ ] OpenAI API key + model
- [ ] JWT secret + expiry
- [ ] CORS origins
- [ ] Log level

**Why this matters:**
- Zero hardcoded secrets
- 12-Factor App compliance
- Easy environment switching (dev/staging/prod)

---

### 3. `api/database/session.py` - DB Session Management
**What we'll implement:**
- [ ] AsyncEngine with connection pooling
- [ ] AsyncSession factory
- [ ] `get_session()` dependency for FastAPI
- [ ] Pool configuration (size=10, max_overflow=20)

**Why async:**
- 10x throughput vs sync
- Non-blocking I/O
- Required for FastAPI async endpoints

---

## Phase 2: Models (Files 4-5)

### 4. `api/apps/auth/models.py` - User & Department Models
**What we'll implement:**
- [ ] `User` model (inherits BaseModel)
  - email (unique, indexed)
  - hashed_password
  - department (FK to Department)
  - role (user, admin)
  - is_active
- [ ] `Department` model
  - name (Sales, HR, Finance, etc.)
  - description

**Why separate Department table:**
- Can add department-level settings later
- Referential integrity
- Easy to extend (dept admins, custom permissions)

---

### 5. `api/apps/documents/models.py` - Document Models
**What we'll implement:**
- [ ] `Document` model (master record)
  - title, department (indexed!)
  - doc_type, source_url
  - version, is_active (for versioning)
- [ ] `DocumentChunk` model (for RAG)
  - document_id (FK)
  - content (text)
  - chunk_index (position)
  - vector_id (link to Qdrant)
  - token_count (for cost tracking)

**Why split Document + Chunk:**
- Large docs need chunking for embedding
- Different chunks have different relevance
- Enables versioning at document level
- Vector DB stores chunks, not full docs

---

## Phase 3: Core Infrastructure (Files 6-8)

### 6. `api/core/embeddings.py` - Embedding Generation
**What we'll implement:**
- [ ] `generate_embedding()` - single text to vector
- [ ] `generate_embeddings_batch()` - batch processing
- [ ] Support both OpenAI and local models (sentence-transformers)
- [ ] Caching (don't re-embed same text)

**Models:**
- Local: `sentence-transformers/all-MiniLM-L6-v2` (fast, free)
- OpenAI: `text-embedding-3-small` (better quality, costs money)

---

### 7. `api/core/vector_store.py` - Qdrant Integration
**What we'll implement:**
- [ ] `VectorStore` class wrapping Qdrant client
- [ ] `upsert()` - Insert/update vectors with metadata
- [ ] `search()` - Query with metadata filters (THE KEY)
- [ ] `delete()` - Remove vectors
- [ ] `update_metadata()` - Bulk metadata updates

**Critical method signature:**
```python
async def search(
    self,
    query_vector: List[float],
    filter: Dict[str, Any],  # {"department": "Sales", "is_active": True}
    limit: int = 5
) -> List[SearchResult]:
    ...
```

---

### 8. `api/core/cache.py` - Redis Caching
**What we'll implement:**
- [ ] `CacheManager` class
- [ ] `get()` - Retrieve cached result
- [ ] `set()` - Store with TTL
- [ ] `delete()` - Invalidate cache
- [ ] `get_key()` - Generate cache key (query + department)

**Cache strategy:**
- Key: `hash(query + department)`
- TTL: 1 hour (configurable)
- Invalidate on document updates

---

## Phase 4: Services (Files 9-11) - THE CORE LOGIC

### 9. `api/apps/auth/services.py` - Authentication
**What we'll implement:**
- [ ] `login()` - Email/password → JWT token
- [ ] `verify_token()` - JWT → User object
- [ ] `get_user_department()` - Extract department from token
- [ ] Password hashing (bcrypt)

**JWT payload:**
```json
{
  "sub": "user_id",
  "email": "john@coragem.com",
  "department": "Sales",
  "role": "user",
  "exp": 1234567890
}
```

---

### 10. `api/apps/documents/services.py` - Document Management
**What we'll implement:**
- [ ] `ingest_document()` - **THE BIG ONE**
  - Validate department
  - Check permissions
  - Chunk text (500 chars, 50 overlap)
  - Generate embeddings
  - Store in Qdrant + Postgres
  - Handle versioning
  - Atomic transaction (rollback if fails)
- [ ] `get_document_versions()` - List all versions
- [ ] `deactivate_old_versions()` - Mark old as inactive

**Flow:**
```
1. User uploads doc
2. Check: Can user upload to this dept?
3. Chunk content (500 chars, 50 overlap)
4. Generate embeddings (batch)
5. Store in Qdrant (with dept metadata)
6. Store in Postgres (chunks + metadata)
7. Deactivate old versions
8. Invalidate cache
```

---

### 11. `api/apps/rag/services.py` - RAG Query (THE STAR)
**What we'll implement:**
- [ ] `rag_query()` - **DEPARTMENT-FILTERED RETRIEVAL**

**The algorithm (this is THE core):**
```python
async def rag_query(
    query: str,
    user_department: str,
    confidence_threshold: float = 0.7
) -> QueryResponse:
    # 1. Check cache
    cache_key = hash(query + user_department)
    if cached := cache.get(cache_key):
        return cached  # Instant, free
    
    # 2. Generate query embedding
    query_embedding = await generate_embedding(query)
    
    # 3. Search vector DB with DEPT FILTER (CRITICAL!)
    results = await vector_store.search(
        query_vector=query_embedding,
        filter={
            "department": user_department,  # ← THE SECURITY BOUNDARY
            "is_active": True
        },
        limit=10
    )
    
    # 4. Apply confidence threshold (hallucination prevention)
    relevant_docs = [r for r in results if r.score >= confidence_threshold]
    
    # 5. If no high-confidence docs, return "I don't know"
    if not relevant_docs:
        return QueryResponse(
            answer="I don't have enough information to answer accurately.",
            confidence="low",
            sources=[]
        )
    
    # 6. Generate answer with LLM
    context = "\n\n".join([doc.content for doc in relevant_docs[:5]])
    prompt = f"""Based ONLY on this context, answer the question.
If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {query}

Answer:"""
    
    answer = await llm.generate(prompt)
    
    # 7. Cache result
    cache.set(cache_key, answer, ttl=3600)
    
    return QueryResponse(
        answer=answer,
        confidence="high",
        sources=[doc.metadata for doc in relevant_docs]
    )
```

**Why this is secure:**
- Vector DB enforces filter BEFORE retrieval
- No post-filtering (can't leak via processing)
- Admin (IT dept) can override filter for support

---

## Phase 5: API Layer (Files 12-14)

### 12. Routers (All `routers.py` files)
**What we'll implement:**
- Thin wrappers around services
- Extract auth from headers
- Call service functions
- Return standardized responses

**Example pattern:**
```python
@router.post("/query")
async def query(
    request: QueryRequest,
    token: str = Depends(verify_token),
    session: AsyncSession = Depends(get_session)
):
    user_dept = extract_department(token)
    result = await rag_service.rag_query(
        query=request.query,
        user_department=user_dept,
        session=session
    )
    return success_response(200, "Query successful", result)
```

---

### 13. `api/utils/responses.py` - Standardized Responses
**What we'll implement:**
- `success_response(code, message, data)`
- `error_response(code, message, context)`
- `validation_error_response(errors)`
- Consistent JSON structure

---

### 14. `api/main.py` - FastAPI App
**What we'll implement:**
- App initialization
- CORS middleware
- Router registration
- Global exception handler
- Startup/shutdown events
- Health check endpoints

---

## Phase 6: Testing (Files 15-16)

### 15. `tests/integration/test_dept_isolation.py`
**THE MOST CRITICAL TEST:**

```python
async def test_sales_cannot_see_hr_docs():
    """Sales user must NOT see HR documents"""
    
    # 1. Ingest HR doc
    await ingest_document(
        title="Leave Policy",
        content="Maternity leave: 16 weeks",
        department="HR"
    )
    
    # 2. Login as Sales user
    sales_token = await login("sales@coragem.com")
    
    # 3. Try to query HR info
    response = await query_rag(
        token=sales_token,
        query="What is the maternity leave policy?"
    )
    
    # 4. MUST return "don't have information" (not the HR policy)
    assert "don't have enough information" in response["answer"]
    assert len(response["sources"]) == 0
```

**Other tests:**
- [ ] HR user can see HR docs
- [ ] IT user (admin) can see all docs
- [ ] Document versioning works
- [ ] Cache invalidation works
- [ ] Concurrent queries don't leak data

---

## Phase 7: Data & Deployment (Files 17-18)

### 17. `data/scripts/seed_data.py`
**What we'll implement:**
- Create 5 departments (Sales, HR, Finance, Ops, Manufacturing)
- Create test users (1 per department + 1 admin)
- Ingest mock documents (5 docs from `/data/mock_departments/`)
- Verify embeddings created

---

### 18. Docker & Deployment
**What we'll implement:**
- Multi-stage Dockerfile
- Alembic migrations
- Health checks
- Deployment guide (Azure/AWS)

---

## BUILD SEQUENCE (The Order Matters)

**Week 1: Foundation**
1. BaseModel
2. Settings
3. Session
4. Logger + Responses

**Week 2: Models & DB**
5. User/Department models
6. Document models
7. Alembic migrations
8. Seed data

**Week 3: Core**
9. Embeddings
10. Vector Store
11. Cache
12. LLM

**Week 4: Services**
13. Auth service
14. Document service
15. RAG service (THE BIG ONE)

**Week 5: API**
16. Routers
17. Main app
18. Exception handling

**Week 6: Testing & Polish**
19. Integration tests
20. Performance testing
21. Documentation
22. Deployment

---

## TESTING STRATEGY

After each phase, test manually:

**After Phase 2 (Models):**
```python
# Test BaseModel CRUD
user = await User.create(session, email="test@coragem.com", department="Sales")
users = await User.get_all(session, filters={"department": "Sales"})
```

**After Phase 3 (Core):**
```python
# Test embedding generation
embedding = await generate_embedding("test query")
assert len(embedding) == 384  # sentence-transformers dimension
```

**After Phase 4 (Services):**
```python
# Test document ingestion
doc = await ingest_document(
    title="Test",
    content="Lorem ipsum" * 100,
    department="Sales"
)
# Check: Chunks created? Embeddings stored?
```

**After Phase 5 (API):**
```bash
# Test end-to-end
curl -X POST localhost:8000/api/v1/rag/query \
  -H "Authorization: Bearer <token>" \
  -d '{"query":"What are product prices?"}'
```

---

## READY TO START?

**First file: `api/database/base.py`**

Open it and we'll implement the BaseModel CRUD pattern together.

Then we move to settings, session, and so on.

**No rushing. Build it right. Test each piece.** 🔥
