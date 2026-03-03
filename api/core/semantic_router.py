"""
Semantic Router for multi-agent intent classification.

Microkernel architecture: agents register themselves via `register_agent()`,
removing hardcoded Enums. New capabilities (Power BI, ERP, etc.) are added
by registering a new agent — zero changes to the core routing logic.

Uses a fast LLM call (Gemini) with structured JSON output to classify
the user's intent into one of the dynamically registered agents.
"""

import json
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


# ── Backward-Compatible String Constants ──────────────────────────────────────
# Consumers that previously imported `RoutedAgent.RAG` can now use
# `RoutedAgent.RAG` as a plain string constant instead.


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

    def __init__(self) -> None:
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model = settings.GEMINI_MODEL
        self._registry: Dict[str, BaseAgent] = {}

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

    async def route_query(
        self,
        query: str,
        department: str,
        chat_history: Optional[list] = None,
    ) -> RouterResponse:
        """
        Analyze the query and determine which registered agent should handle it.

        The LLM prompt is built dynamically from whatever agents are
        currently registered — no hardcoded list.
        """
        if not self._registry:
            logger.error("No agents registered. Cannot route.")
            return RouterResponse(
                routed_to=RoutedAgent.UNKNOWN,
                confidence_score=0.0,
                reasoning="No agents registered in the router",
            )

        system_prompt = self._build_system_prompt(department)
        user_prompt = (
            f"User Request: {query}\n\n"
            "Analyze the request and return the JSON routing decision."
        )

        try:
            logger.debug(f"Routing query from {department}: '{query[:50]}...'")
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
            decision = RouterResponse(**decision_dict)

            # Post-hoc validation: if the LLM hallucinated an agent name
            # that isn't registered, fall back to unknown.
            if decision.routed_to not in self._registry:
                logger.warning(
                    f"LLM returned unregistered agent '{decision.routed_to}'. "
                    "Falling back to 'unknown'."
                )
                decision.routed_to = RoutedAgent.UNKNOWN

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
            f"Available Agents:\n\n"
            f"{agent_block}\n\n"
            f"You must reply with a valid JSON object matching the requested schema. "
            f"Do not include markdown formatting or extra text outside the JSON."
        )
