"""
Application settings.

Uses Pydantic BaseSettings for type-safe config.
All values loaded from .env — zero hardcoded secrets.
Follows 12-Factor App principles.
"""

from typing import List
from pydantic import Field, AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.

    Sections:
    - App
    - Database (PostgreSQL)
    - Redis
    - Qdrant (Vector DB)
    - Gemini (LLM)
    - JWT (Auth)
    - CORS
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────
    APP_NAME: str = "coragem RAG API"
    COMPANY_NAME: str = "Coragem"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = Field("development", description="development | staging | production")
    LOG_LEVEL: str = "INFO"
    MAX_QUERY_LENGTH: int = 1000
    MAX_DOCUMENT_SIZE: int = 10_000_000  # 10MB

    # ── Database ─────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        ...,
        description="Async PostgreSQL URL: postgresql+asyncpg://user:pass@host/db"
    )

    # ── Redis ────────────────────────────────────────────────────────────
    REDIS_URL: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )

    # ── Pinecone ─────────────────────────────────────────────────────────
    PINECONE_API_KEY: str = Field(..., description="Pinecone API key")
    PINECONE_INDEX_NAME: str = "documents"

    # ── Gemini ───────────────────────────────────────────────────────────
    GEMINI_API_KEY: str = Field(..., description="Google Gemini API key")
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # ── JWT ──────────────────────────────────────────────────────────────
    JWT_SECRET_KEY: str = Field(..., description="Secret key for JWT signing (min 32 chars)")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS_WEB: int = 7
    REFRESH_TOKEN_EXPIRE_DAYS_MOBILE: int = 30

    # ── CORS ─────────────────────────────────────────────────────────────
    ALLOWED_HOSTS: List[str] = ["*"]
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    ALLOWED_ORIGINS: List[str] = ["*"]

    # ── Embedding ────────────────────────────────────────────────────────
    EMBEDDING_PROVIDER: str = Field("gemini", description="gemini | azure_openai")
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIMENSION: int = 3072

    # ── Azure OpenAI ─────────────────────────────────────────────────────
    AZURE_OPENAI_ENDPOINT: str = Field(default="", description="Azure OpenAI Endpoint")
    AZURE_OPENAI_API_KEY: str = Field(default="", description="Azure OpenAI API Key")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = Field(default="text-embedding-ada-002", description="Azure OpenAI Embedding Deployment Name")
    AZURE_OPENAI_API_VERSION: str = Field(default="2024-02-15-preview", description="Azure OpenAI API Version")

# Singleton — import this everywhere
settings = Settings()  # type: ignore[call-arg]
