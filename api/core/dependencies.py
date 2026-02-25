"""
Dependency injection for FastAPI.

Provides singleton instances of services.
"""

from functools import lru_cache
from api.core.vector_store import VectorStore
from api.core.cache import CacheManager
from api.core.semantic_router import SemanticRouter
from api.config.settings import settings


@lru_cache()
def get_vector_store() -> VectorStore:
    """Get vector store singleton."""
    return VectorStore(
        api_key=settings.PINECONE_API_KEY,
        index_name=settings.PINECONE_INDEX_NAME
    )


@lru_cache()
def get_cache() -> CacheManager:
    """Get cache manager singleton."""
    return CacheManager(url=settings.REDIS_URL)


@lru_cache()
def get_semantic_router() -> SemanticRouter:
    """Get Semantic Router singleton."""
    return SemanticRouter()