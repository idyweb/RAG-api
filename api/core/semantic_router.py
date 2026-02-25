"""
Semantic Router for multi-agent intent classification.

Uses a fast LLM call (Gemini) with structured JSON output to classify
the user's intent into one of the designated Copilots/Agents.
"""

import json
from enum import Enum
from typing import Dict, Any, Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from api.config.settings import settings
from api.utils.logger import get_logger

logger = get_logger(__name__)


class RoutedAgent(str, Enum):
    """Available target agents for routing."""
    RAG = "rag"
    POWER_BI = "power_bi"
    GTM_API = "gtm_api"
    ERP_API = "erp_api"
    QMS = "qms"
    DOCUMENT_SEARCH = "document_search"
    UNKNOWN = "unknown"


class RouterResponse(BaseModel):
    """Structured output expected from the LLM Router."""
    routed_to: RoutedAgent = Field(
        description="The target agent strictly chosen to handle the query"
    )
    confidence_score: float = Field(
        description="Confidence score between 0.0 and 1.0 that the chosen agent is correct"
    )
    reasoning: str = Field(
        description="Brief reasoning (1 sentence) for why this agent was chosen"
    )


class SemanticRouter:
    """
    Evaluates user intent to route the query to the correct Coragem Copilot.
    """

    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        # We use a fast, cost-effective model for the high-volume routing layer
        self.model = settings.GEMINI_MODEL

    async def route_query(
        self, 
        query: str, 
        department: str, 
        chat_history: Optional[list] = None
    ) -> RouterResponse:
        """
        Analyze the query and determine which specialized agent should handle it.
        """
        system_prompt = self._build_system_prompt(department)
        
        # Build contents structure 
        # (For accurate routing, we just need the system prompt and the latest query.
        # We omit chat_history to keep routing fast and highly focused on the immediate intent,
        # unless strictly necessary. For now, we only pass the query.)
        
        user_prompt = f"User Request: {query}\n\nAnalyze the request and return the JSON routing decision."

        try:
            logger.debug(f"Routing query from {department}: '{query[:50]}...'")
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,  # Zero temperature for deterministic routing
                    response_mime_type="application/json",
                    response_schema=RouterResponse,
                )
            )

            if not response.text:
                logger.warning("Empty response from LLM Router. Defaulting to UNKNOWN.")
                return RouterResponse(
                    routed_to=RoutedAgent.UNKNOWN, 
                    confidence_score=0.0, 
                    reasoning="Empty LLM response"
                )

            # Gemini SDK will guarantee JSON output matches Schema when response_schema is set
            decision_dict = json.loads(response.text)
            decision = RouterResponse(**decision_dict)
            
            logger.info(
                f"Router Decision: {decision.routed_to.value} "
                f"(conf={decision.confidence_score}): {decision.reasoning}"
            )
            return decision

        except json.JSONDecodeError as e:
            logger.error(f"Router failed to produce valid JSON: {e}")
            return RouterResponse(
                routed_to=RoutedAgent.UNKNOWN, 
                confidence_score=0.0, 
                reasoning="JSON parse error"
            )
        except Exception as e:
            logger.error(f"Semantic Router execution failed: {e}")
            return RouterResponse(
                routed_to=RoutedAgent.UNKNOWN, 
                confidence_score=0.0, 
                reasoning=str(e)
            )

    def _build_system_prompt(self, department: str) -> str:
        """
        Builds the strict classification rules for the Semantic Router.
        """
        return f"""You are the Semantic Router for the Coragem Enterprise AI Platform.
Your ONLY job is to classify the user's request and route it to the correct specialized agent.
The user belongs to the '{department}' department. Use this for context, but route based on the query intent.

Available Agents:

1. `rag`: (HR Policy Assistant, General Knowledge)
   - Handles queries about corporate policies, HR, leave (e.g., maternity leave), IT guidelines, FAQs, facilities, general admin, complaints, and static manuals.
   - If the user asks "What is...", "How do I...", "What is the policy for...", or expresses a general workplace concern or complaint (e.g., food, cafeteria, office), route here.

2. `power_bi`: (Sales Intelligence - Analytics)
   - Handles queries about dashboards, historical charts, revenue targets, aggregate sales figures.
   - Example: "Show me the top selling SKUs in Angola this month compared to target."

3. `gtm_api`: (Sales Intelligence - CRM/Operations)
   - Handles route planning, sales rep performance, active pipeline status.
   - Example: "What is John's route today?" or "Show me the pipeline for Q3."

4. `erp_api`: (ERP Support Copilot)
   - Handles live transactional data: Microsoft Dynamics Business Central, inventory limits, PO status, financial closes, stock levels.
   - Example: "Has Purchase Order 12345 been approved?" or "How much Cowbell 400g is in the Lagos warehouse?"

5. `qms`: (Production Quality Assistant)
   - Handles IS-OEE (Overall Equipment Effectiveness) systems, Quality Management System (QMS), equipment maintenance logs, raw materials batch info, incidence reports.
   - Example: "Show the maintenance history for packaging line 3" or "Are there any QMS incidents for batch A1?"

6. `document_search`: (Intelligent Document Search)
   - The user is explicitly asking to *find* a specific file, presentation, or email, rather than asking for the answer inside it.
   - Example: "Find the Q3 marketing presentation that John sent last week."

7. `unknown`: 
   - The request is completely nonsensical, malicious, or clearly falls outside any corporate AI capabilities.

You must reply with a valid JSON object matching the requested schema. Do not include markdown formatting or extra text outside the JSON.
"""
