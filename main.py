"""
FastAPI application entry point.

Registers routers, middleware, and exception handlers.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone

from api.config.settings import settings
from api.utils.exceptions import BaseAPIException
from api.utils.exception_handlers import (
    base_api_exception_handler,
    request_validation_exception_handler,
    http_exception_handler,
    general_exception_handler,
)
from api.utils.logger import get_logger
from api.apps.auth.routers import router as auth_router
from api.apps.documents.routers import router as documents_router
from api.apps.rag.routers import router as rag_router
from api.apps.agents.mcp_server import mcp
from api.db.session import get_session
from api.core.dependencies import get_cache
from fastmcp.utilities.lifespan import combine_lifespans

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown lifecycle events."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} [{settings.ENVIRONMENT}]")
    yield
    logger.info("Shutdown complete")


# ── MCP ────────────────────────────────────────────────────────────────────
mcp_app = mcp.http_app(path="/")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Multi-Department RAG System with Role-Based Access Control",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=combine_lifespans(lifespan, mcp_app.lifespan),
)


# ── Middleware ────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception Handlers ────────────────────────────────────────────────────────

app.add_exception_handler(BaseAPIException, base_api_exception_handler)          # type: ignore[arg-type]
app.add_exception_handler(RequestValidationError, request_validation_exception_handler)  # type: ignore[arg-type]
app.add_exception_handler(HTTPException, http_exception_handler)                  # type: ignore[arg-type]
app.add_exception_handler(Exception, general_exception_handler)


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(rag_router)

# ── MCP Mount ─────────────────────────────────────────────────────────────────
app.mount("/mcp", mcp_app)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Infra"])
async def health():
    """Liveness probe — must respond < 200ms."""
    return {"status": "ok", "version": settings.APP_VERSION}


@app.get("/ready", tags=["Infra"])
async def ready(
    session: AsyncSession = Depends(get_session),
    cache = Depends(get_cache)
):
    """
    Readiness probe — verifies DB and Redis are reachable.
    Returns 503 if any dependency is down.
    """
    checks = {}
    healthy = True

    # Check PostgreSQL
    try:
        await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        logger.error(f"DB health check failed: {e}")
        checks["database"] = f"error: {str(e)[:80]}"
        healthy = False

    # Check Redis
    try:
        await cache.redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        checks["redis"] = f"error: {str(e)[:80]}"
        healthy = False

    payload = {
        "status": "ready" if healthy else "degraded",
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    if not healthy:
        return JSONResponse(status_code=503, content=payload)

    return payload
