"""
Langfuse LLM observability integration (SDK v3 — OpenTelemetry-based).

Provides a singleton Langfuse client for tracing RAG pipeline stages.
Uses context managers for nested spans: trace → routing, retrieval, generation.

Gracefully no-ops when LANGFUSE_ENABLED=False.
"""

from api.config.settings import settings
from api.utils.logger import get_logger

logger = get_logger(__name__)

_langfuse = None


def get_langfuse():
    """Get singleton Langfuse client. Returns None if disabled."""
    global _langfuse
    if _langfuse is not None:
        return _langfuse

    if not settings.LANGFUSE_ENABLED:
        return None

    try:
        from langfuse import Langfuse

        _langfuse = Langfuse(
            secret_key=settings.LANGFUSE_SECRET_KEY,
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            host=settings.LANGFUSE_BASE_URL,
        )
        # Verify connection
        _langfuse.auth_check()
        logger.info(f"Langfuse initialized: host={settings.LANGFUSE_BASE_URL}")
        return _langfuse
    except Exception as e:
        logger.warning(f"Langfuse init failed (tracing disabled): {e}")
        return None


def flush():
    """Flush pending events to Langfuse."""
    lf = get_langfuse()
    if lf:
        try:
            lf.flush()
        except Exception:
            pass


def shutdown():
    """Shutdown Langfuse client cleanly."""
    lf = get_langfuse()
    if lf:
        try:
            lf.shutdown()
        except Exception:
            pass
