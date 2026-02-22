# Quick Start Guide

**Get the system running in 5 minutes, then start building.**

## Prerequisites

- Python 3.12+
- Docker & Docker Compose
- Git (optional)

## Step 1: Get the Project

```bash
# If you downloaded as zip, extract it
# cd coragem-rag-api

# Or if cloning
git clone <repo-url>
cd coragem-rag-api
```

## Step 2: Start Infrastructure

```bash
# Start Postgres + Redis + Qdrant
docker-compose up -d

# Verify all running
docker ps
# Should see: postgres, redis, qdrant
```

## Step 3: Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## Step 4: Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env
nano .env  # or your editor

# MUST SET:
OPENAI_API_KEY=sk-your-actual-key-here

# Generate JWT secret:
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Copy output to JWT_SECRET_KEY in .env
```

## Step 5: Initialize Database

```bash
# Create migrations folder (first time only)
alembic init alembic

# Run migrations (once we build them)
# alembic upgrade head
```

## Step 6: Seed Mock Data

```bash
# Populate with sample docs (once we build seed script)
# python data/scripts/seed_data.py
```

## Step 7: Run the API

```bash
# Start FastAPI (once we build main.py)
# uvicorn api.main:app --reload --port 8000

# Check health
# curl http://localhost:8000/health
```

## What We'll Build (Step-by-Step)

Now that infrastructure is ready, we'll build:

1. **BaseModel** (`api/database/base.py`) ← **START HERE**
2. **Settings** (`api/config/settings.py`)
3. **Session** (`api/database/session.py`)
4. **Models** (User, Document, etc.)
5. **Services** (Auth, RAG with dept filtering)
6. **API** (Routers, main app)
7. **Tests** (Department isolation)

## Verify Infrastructure

```bash
# Check Postgres
docker exec -it coragem_postgres psql -U coragem -d rag_db -c "SELECT version();"

# Check Redis
docker exec -it coragem_redis redis-cli ping
# Should return: PONG

# Check Qdrant
curl http://localhost:6333/health
# Should return: {"title":"qdrant - vector search engine","version":"..."}
```

## Troubleshooting

### Ports already in use
```bash
# Check what's using ports
lsof -i :5432  # Postgres
lsof -i :6379  # Redis
lsof -i :6333  # Qdrant

# Stop conflicting services or change ports in docker-compose.yml
```

### Docker issues
```bash
# Reset everything
docker-compose down -v
docker-compose up -d
```

### Python version
```bash
# Check version
python --version
# Must be 3.12+

# If wrong version, use pyenv or conda
```

---

## Next: Start Building

**Open `api/database/base.py` and let's write the BaseModel together!**

Follow `BUILD_PLAN.md` for step-by-step guidance.

**No rushing. Build it right. Test each piece.** 🚀
