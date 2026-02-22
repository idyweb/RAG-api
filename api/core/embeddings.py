"""
Embedding service with provider abstraction.

Supports:
- Google Gemini (dev/free tier)
- Azure OpenAI (production)
"""

from typing import List, Literal
from abc import ABC, abstractmethod
import asyncio

from api.config.settings import settings
from api.utils.logger import get_logger

logger = get_logger(__name__)

# Provider type
EmbeddingProvider = Literal["gemini", "azure_openai"]


class EmbeddingService(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    async def generate(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        pass
    
    @abstractmethod
    async def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class GeminiEmbeddingService(EmbeddingService):
    """Google Gemini embedding service."""
    
    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self._dimension = 3072 if "001" in model else 768
        logger.info(f"Initialized Gemini embeddings: {model}, dim={self._dimension}")
    
    async def generate(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        if not text or not text.strip():
            return [0.0] * self._dimension
        
        try:
            response = await self.client.aio.models.embed_content(
                model=self.model,
                contents=[text]
            )
            return response.embeddings[0].values
        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            return [0.0] * self._dimension
    
    async def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        
        try:
            response = await self.client.aio.models.embed_content(
                model=self.model,
                contents=texts
            )
            return [emb.values for emb in response.embeddings]
        except Exception as e:
            logger.error(f"Gemini batch embedding failed: {e}")
            return [[0.0] * self._dimension for _ in texts]
    
    @property
    def dimension(self) -> int:
        return self._dimension


class AzureOpenAIEmbeddingService(EmbeddingService):
    """Azure OpenAI embedding service."""
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment_name: str,
        api_version: str = "2024-02-15-preview"
    ):
        from openai import AsyncAzureOpenAI
        
        self.client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        self.deployment_name = deployment_name
        self._dimension = 1536  # text-embedding-ada-002 or text-embedding-3-small
        
        logger.info(
            f"Initialized Azure OpenAI embeddings: "
            f"deployment={deployment_name}, dim={self._dimension}"
        )
    
    async def generate(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        if not text or not text.strip():
            return [0.0] * self._dimension
        
        try:
            response = await self.client.embeddings.create(
                model=self.deployment_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Azure OpenAI embedding failed: {e}")
            return [0.0] * self._dimension
    
    async def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batched API call)."""
        if not texts:
            return []
        
        try:
            response = await self.client.embeddings.create(
                model=self.deployment_name,
                input=texts  # Azure OpenAI handles batching
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Azure OpenAI batch embedding failed: {e}")
            return [[0.0] * self._dimension for _ in texts]
    
    @property
    def dimension(self) -> int:
        return self._dimension


# ─────────────────────────────────────────────────────────────────────────────
# Factory function to get the right provider
# ─────────────────────────────────────────────────────────────────────────────

def get_embedding_service() -> EmbeddingService:
    """
    Factory function to get embedding service based on config.
    
    Returns appropriate provider (Gemini or Azure) based on settings.
    """
    provider: EmbeddingProvider = settings.EMBEDDING_PROVIDER
    
    if provider == "gemini":
        return GeminiEmbeddingService(
            api_key=settings.GEMINI_API_KEY,
            model=settings.EMBEDDING_MODEL
        )
    
    elif provider == "azure_openai":
        return AzureOpenAIEmbeddingService(
            endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            deployment_name=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_version=settings.AZURE_OPENAI_API_VERSION
        )
    
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience functions (backward compatible with your existing code)
# ─────────────────────────────────────────────────────────────────────────────

_embedding_service: EmbeddingService | None = None

def _get_service() -> EmbeddingService:
    """Get singleton embedding service."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = get_embedding_service()
    return _embedding_service


async def generate_embeddings(text: str) -> List[float]:
    """
    Generate embedding for single text.
    Provider-agnostic wrapper.
    """
    service = _get_service()
    return await service.generate(text)


async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts.
    Provider-agnostic wrapper.
    """
    service = _get_service()
    return await service.generate_batch(texts)


def get_embedding_dimension() -> int:
    """Get current embedding dimension."""
    service = _get_service()
    return service.dimension
