"""
Dependency injection for FastAPI.

Provides singleton instances of services and cross-cutting auth dependencies.
"""

from dataclasses import dataclass
from functools import lru_cache

from api.core.vector_store import VectorStore
from api.core.cache import CacheManager
from api.core.semantic_router import SemanticRouter
from api.config.settings import settings


@dataclass
class DefaultAgent:
    """
    Lightweight agent definition for built-in routing targets.

    Satisfies the BaseAgent protocol. The `execute` method is a
    placeholder — real implementations will override this when
    live API integrations (Power BI, ERP, etc.) are available.
    """

    name: str
    description: str

    async def execute(self, query: str, context: dict) -> dict:
        """Placeholder — returns a 'coming soon' message."""
        return {
            "agent": self.name,
            "message": f"The '{self.name}' agent integration is coming soon.",
            "query": query,
        }

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from api.apps.auth.models import User
from api.db.session import get_session
from api.utils.security import verify_token_type
from api.utils.logger import get_logger

logger = get_logger(__name__)
_bearer = HTTPBearer()


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
    """
    Get Semantic Router singleton with all default agents registered.

    Why here: agent registration happens once at startup. New agents
    are added by appending to this list — zero changes to router logic.
    """
    router = SemanticRouter()

    _DEFAULT_AGENTS = [
        DefaultAgent(
            name="rag",
            description=(
                "DEFAULT agent for all knowledge/policy/how-to questions. "
                "Searches company documents (HR policies, handbooks, guidelines, product catalogs, SOPs, FAQs). "
                "Use for: 'What is the policy for...', 'How do I...', 'What does [document] say about...', "
                "workplace concerns, process explanations, static product info, any question answerable from documents."
            ),
        ),
        DefaultAgent(
            name="power_bi",
            description=(
                "Analytics dashboards and aggregate business metrics. "
                "Use ONLY for: revenue vs target, sales trends over time, KPI dashboards, "
                "top/bottom performing SKUs/regions, charts, comparisons across periods. "
                "Signals: 'show me', 'compare', 'trend', 'target vs actual', 'top selling', 'dashboard'."
            ),
        ),
        DefaultAgent(
            name="gtm_api",
            description=(
                "Live CRM and field sales operations. "
                "Use ONLY for: sales rep daily routes, active pipeline deals, field visit logs, "
                "rep performance tracking. "
                "Signals: specific rep names + 'route/schedule/visits', 'pipeline status', 'field performance'."
            ),
        ),
        DefaultAgent(
            name="erp_api",
            description=(
                "Live ERP transactional data (Business Central, SAP, etc). "
                "Use ONLY for: specific PO/invoice/order status, current stock levels, "
                "live account balances, remaining leave balance, payment status. "
                "Signals: specific reference numbers (PO #, invoice #), 'approved yet', 'current stock', "
                "'how much do I have left', 'status of my order'. "
                "NOT for: policy questions, process explanations, or 'how does X work' — those go to rag."
            ),
        ),
        DefaultAgent(
            name="qms",
            description=(
                "Quality Management System and production equipment data. "
                "Use ONLY for: OEE metrics, equipment maintenance logs/history, "
                "QMS incidents, batch quality records, raw material traceability. "
                "Signals: batch numbers, equipment/line names, 'maintenance history', 'quality incident', 'OEE'."
            ),
        ),
        DefaultAgent(
            name="document_search",
            description=(
                "File/document discovery — finding WHERE a document is, not what's IN it. "
                "Use ONLY when the user wants to locate/find/retrieve a specific file, presentation, or email. "
                "Signals: 'find the document', 'where is the file', 'locate the presentation that [person] sent'."
            ),
        ),
        DefaultAgent(
            name="unknown",
            description=(
                "Fallback for nonsensical, malicious, or completely out-of-scope requests "
                "that no other agent can handle."
            ),
        ),
    ]

    for agent in _DEFAULT_AGENTS:
        router.register_agent(agent)

    return router


async def verify_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """
    FastAPI dependency: validates Bearer token and returns the active user dict.

    Used by all secured endpoints. Raises HTTP 401 if token is invalid or expired.
    Guards against inactive accounts.

    Returns:
        dict with: id, email, full_name, department, role
    """
    # Decode JWT — raises 401 on failure
    payload = verify_token_type(credentials.credentials, expected_type="access")
    user_id: str = payload.get("sub", "")

    # Fetch user from DB
    result = await session.execute(select(User).where(User.id == user_id))  # type: ignore
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    logger.debug(f"Authenticated user: {user.email} dept={user.department}")
    return {
        "id": str(user.id),
        "email": user.email,
        "full_name": user.full_name,
        "department": user.department,
        "role": user.role,
    }