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
                "(HR Policy Assistant, General Knowledge)\n"
                "   - Handles queries about corporate policies, HR, leave "
                "(e.g., maternity leave), IT guidelines, FAQs, facilities, "
                "general admin, complaints, and static manuals.\n"
                "   - If the user asks \"What is...\", \"How do I...\", "
                "\"What is the policy for...\", or expresses a general "
                "workplace concern or complaint (e.g., food, cafeteria, "
                "office), route here."
            ),
        ),
        DefaultAgent(
            name="power_bi",
            description=(
                "(Sales Intelligence - Analytics)\n"
                "   - Handles queries about dashboards, historical charts, "
                "revenue targets, aggregate sales figures.\n"
                "   - Example: \"Show me the top selling SKUs in Angola "
                "this month compared to target.\""
            ),
        ),
        DefaultAgent(
            name="gtm_api",
            description=(
                "(Sales Intelligence - CRM/Operations)\n"
                "   - Handles route planning, sales rep performance, "
                "active pipeline status.\n"
                "   - Example: \"What is John's route today?\" or "
                "\"Show me the pipeline for Q3.\""
            ),
        ),
        DefaultAgent(
            name="erp_api",
            description=(
                "(ERP Support Copilot)\n"
                "   - Handles live transactional data: Microsoft Dynamics "
                "Business Central, inventory limits, PO status, financial "
                "closes, stock levels.\n"
                "   - Example: \"Has Purchase Order 12345 been approved?\" "
                "or \"How much Cowbell 400g is in the Lagos warehouse?\""
            ),
        ),
        DefaultAgent(
            name="qms",
            description=(
                "(Production Quality Assistant)\n"
                "   - Handles IS-OEE (Overall Equipment Effectiveness) "
                "systems, Quality Management System (QMS), equipment "
                "maintenance logs, raw materials batch info, incidence "
                "reports.\n"
                "   - Example: \"Show the maintenance history for "
                "packaging line 3\" or \"Are there any QMS incidents "
                "for batch A1?\""
            ),
        ),
        DefaultAgent(
            name="document_search",
            description=(
                "(Intelligent Document Search)\n"
                "   - The user is explicitly asking to *find* a specific "
                "file, presentation, or email, rather than asking for "
                "the answer inside it.\n"
                "   - Example: \"Find the Q3 marketing presentation that "
                "John sent last week.\""
            ),
        ),
        DefaultAgent(
            name="unknown",
            description=(
                "(Fallback)\n"
                "   - The request is completely nonsensical, malicious, or "
                "clearly falls outside any corporate AI capabilities."
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