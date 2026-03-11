"""
Semantic Router for multi-agent intent classification.

Microkernel architecture: agents register themselves via `register_agent()`,
removing hardcoded Enums. New capabilities (Power BI, ERP, etc.) are added
by registering a new agent — zero changes to the core routing logic.

Uses a fast LLM call (Gemini) with structured JSON output to classify
the user's intent into one of the dynamically registered agents.
"""

import json
import hashlib
from collections import OrderedDict
from typing import Dict, Optional, Protocol, runtime_checkable

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from api.config.settings import settings
from api.utils.logger import get_logger

logger = get_logger(__name__)


# ── Agent Plugin Interface ────────────────────────────────────────────────────


@runtime_checkable
class BaseAgent(Protocol):
    """
    Protocol that every pluggable agent must satisfy.

    To add a new capability to the platform:
    1. Create a class that fulfills this protocol.
    2. Call `semantic_router.register_agent(your_agent)` at startup.
    3. The router will automatically include it in LLM classification.
    """

    name: str
    """Unique identifier used in routing decisions (e.g. 'power_bi')."""

    description: str
    """
    Human-readable description shown to the LLM so it understands
    when to route queries to this agent. Be specific about triggers.
    """

    async def execute(self, query: str, context: dict) -> dict:
        """
        Execute the agent's core capability.

        Args:
            query: The user's natural language query.
            context: Runtime context (department, user_id, session, etc.).

        Returns:
            Agent-specific result dict.
        """
        ...


class RoutedAgent:
    """String constants for built-in agent names (backward compat)."""

    RAG = "rag"
    POWER_BI = "power_bi"
    GTM_API = "gtm_api"
    ERP_API = "erp_api"
    QMS = "qms"
    DOCUMENT_SEARCH = "document_search"
    UNKNOWN = "unknown"


# ── Router Response Schema ────────────────────────────────────────────────────


class RouterResponse(BaseModel):
    """Structured output expected from the LLM Router."""

    routed_to: str = Field(
        description="The target agent name chosen to handle the query"
    )
    confidence_score: float = Field(
        description="Confidence score between 0.0 and 1.0 that the chosen agent is correct"
    )
    reasoning: str = Field(
        description="Brief reasoning (1 sentence) for why this agent was chosen"
    )


# ── Semantic Router (Microkernel Core) ────────────────────────────────────────


class SemanticRouter:
    """
    Microkernel Semantic Router.

    Agents register themselves, removing hardcoded Enums.
    The LLM prompt is built dynamically from the registry.
    """

    _ROUTE_CACHE_SIZE = 256

    def __init__(self) -> None:
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model = settings.GEMINI_MODEL
        self._registry: Dict[str, BaseAgent] = {}
        self._route_cache: OrderedDict[str, RouterResponse] = OrderedDict()

    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register a new agent capability.

        Why this matters: adding a new agent (e.g. Power BI connector)
        requires zero changes to the core router logic — just register it.
        """
        if agent.name in self._registry:
            logger.warning(f"Agent '{agent.name}' already registered — overwriting.")
        self._registry[agent.name] = agent
        logger.info(f"Registered agent: '{agent.name}'")

    @property
    def registered_agents(self) -> list[str]:
        """Return the names of all registered agents."""
        return list(self._registry.keys())

    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def _execute_routing(self, user_prompt: str, system_prompt: str) -> RouterResponse:
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=RouterResponse,
            ),
        )

        if not response.text:
            logger.warning("Empty response from LLM Router. Defaulting to UNKNOWN.")
            return RouterResponse(
                routed_to=RoutedAgent.UNKNOWN,
                confidence_score=0.0,
                reasoning="Empty LLM response",
            )

        decision_dict = json.loads(response.text)
        return RouterResponse(**decision_dict)

    def _cache_key(self, query: str, department: str) -> str:
        """Normalize + hash for O(1) cache lookup."""
        normalized = f"{query.lower().strip()}:{department}"
        return hashlib.md5(normalized.encode()).hexdigest()

    async def route_query(
        self,
        query: str,
        department: str,
        chat_history: Optional[list] = None,
    ) -> RouterResponse:
        """
        Analyze the query and determine which registered agent should handle it.

        Includes an in-memory LRU cache to avoid redundant LLM calls for
        identical or near-identical queries. Cache is bounded to
        _ROUTE_CACHE_SIZE entries.
        """
        if not self._registry:
            logger.error("No agents registered. Cannot route.")
            return RouterResponse(
                routed_to=RoutedAgent.UNKNOWN,
                confidence_score=0.0,
                reasoning="No agents registered in the router",
            )

        # Check in-memory route cache
        ck = self._cache_key(query, department)
        if ck in self._route_cache:
            self._route_cache.move_to_end(ck)
            logger.debug(f"Route cache HIT for dept={department}: '{query[:50]}...'")
            return self._route_cache[ck]

        system_prompt = self._build_system_prompt(department)
        user_prompt = (
            f"User Request: {query}\n\n"
            "Analyze the request and return the JSON routing decision."
        )

        try:
            logger.debug(f"Routing query from {department}: '{query[:50]}...'")
            decision = await self._execute_routing(user_prompt, system_prompt)

            if decision.routed_to not in self._registry:
                logger.warning(
                    f"LLM returned unregistered agent '{decision.routed_to}'. "
                    "Falling back to 'unknown'."
                )
                decision.routed_to = RoutedAgent.UNKNOWN

            # Store in cache (evict oldest if full)
            self._route_cache[ck] = decision
            if len(self._route_cache) > self._ROUTE_CACHE_SIZE:
                self._route_cache.popitem(last=False)

            logger.info(
                f"Router Decision: {decision.routed_to} "
                f"(conf={decision.confidence_score}): {decision.reasoning}"
            )
            return decision

        except json.JSONDecodeError as e:
            logger.error(f"Router failed to produce valid JSON: {e}")
            return RouterResponse(
                routed_to=RoutedAgent.UNKNOWN,
                confidence_score=0.0,
                reasoning="JSON parse error",
            )
        except Exception as e:
            logger.error(f"Semantic Router execution failed: {e}")
            return RouterResponse(
                routed_to=RoutedAgent.UNKNOWN,
                confidence_score=0.0,
                reasoning=str(e),
            )

    def _build_system_prompt(self, department: str) -> str:
        """
        Build the classification prompt dynamically from the agent registry.

        Why dynamic: adding an agent automatically updates the LLM's
        decision space — no prompt editing required.
        """
        agent_lines: list[str] = []
        for idx, (name, agent) in enumerate(self._registry.items(), start=1):
            agent_lines.append(f"{idx}. `{name}`: {agent.description}")

        agent_block = "\n\n".join(agent_lines)

        return (
            f"You are the Semantic Router for the Coragem Enterprise AI Platform.\n"
            f"Your ONLY job is to classify the user's request and route it to the correct specialized agent.\n"
            f"The user belongs to the '{department}' department. Use this for context, but route based on the query intent.\n\n"
            f"ROUTING DECISION TREE (follow in order):\n\n"
            f"Step 1 — Is the user asking to FIND a specific file/document/presentation (not its contents)?\n"
            f"  → YES: route to `document_search`\n\n"
            f"Step 2 — Is the query about a LIVE transaction or real-time system state?\n"
            f"  Look for signals: specific PO/invoice numbers, 'current stock', 'approved yet', 'status of order', 'balance right now'\n"
            f"  → YES and it's about supply chain/inventory/invoices/purchase orders: route to `erp_api`\n"
            f"  → YES and it's about sales dashboards/charts/KPIs/revenue targets/aggregate figures: route to `power_bi`\n"
            f"  → YES and it's about sales rep routes/CRM pipeline/field performance: route to `gtm_api`\n"
            f"  → YES and it's about equipment OEE/maintenance logs/QMS incidents/batch quality: route to `qms`\n\n"
            f"Step 3 — Otherwise, it's a knowledge/policy/how-to/general question:\n"
            f"  → route to `rag` (this is the DEFAULT)\n\n"
            f"Step 4 — If the query is nonsensical, malicious, or completely outside corporate scope:\n"
            f"  → route to `unknown`\n\n"
            f"DISAMBIGUATION EXAMPLES:\n"
            f"- 'What is the maternity leave policy?' → rag (policy question)\n"
            f"- 'How much maternity leave do I have left?' → erp_api (live balance check)\n"
            f"- 'What are our top products this quarter?' → power_bi (aggregate analytics)\n"
            f"- 'What is the price of Cowbell 400g?' → rag (static product info from docs)\n"
            f"- 'Is PO 54321 approved?' → erp_api (specific transaction status)\n"
            f"- 'What is our procurement process?' → rag (policy/process question)\n"
            f"- 'Show John's route for today' → gtm_api (live field operations)\n"
            f"- 'Find the Q3 marketing deck' → document_search (file lookup)\n"
            f"- 'What does the Q3 marketing deck say about pricing?' → rag (content question)\n"
            f"- 'Any quality incidents for batch A1?' → qms (quality system query)\n"
            f"- 'What is our quality control procedure?' → rag (policy question)\n\n"
            f"Available Agents:\n\n"
            f"{agent_block}\n\n"
            f"You must reply with a valid JSON object matching the requested schema. "
            f"Do not include markdown formatting or extra text outside the JSON."
        )
