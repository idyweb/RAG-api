"""
Query clarification service.

Analyzes user queries for clarity and specificity before RAG retrieval.
Rewrites ambiguous queries and requests clarification when needed.

Model-agnostic design: Uses the configured LLM provider (Gemini/Azure)
via a simple JSON prompt pattern — no provider-specific structured output.
"""

import json
from typing import List, Dict, Optional

from pydantic import BaseModel, Field

from api.config.settings import settings
from api.utils.logger import get_logger

logger = get_logger(__name__)


class QueryAnalysis(BaseModel):
    """Result of query clarity analysis."""

    model_config = {"extra": "forbid"}

    is_clear: bool = Field(
        description="Whether the query is clear and specific enough to search documents"
    )
    rewritten_query: str = Field(
        description="Search-optimized rewrite of the original query"
    )
    clarification_needed: str = Field(
        default="",
        description="What to ask the user if the query is unclear (empty if clear)",
    )
    sub_queries: List[str] = Field(
        default_factory=list,
        description="Optional decomposition into sub-questions for complex queries",
    )


_CLARIFIER_PROMPT = """You are a query analysis engine for an enterprise knowledge base.
Analyze the user's query for clarity and specificity.

Your tasks:
1. Determine if the query is clear enough to search internal documents
2. If unclear, explain what clarification is needed
3. Rewrite the query to be optimized for semantic search
4. For complex queries, decompose into sub-questions

Rules:
- A query is UNCLEAR if it uses ambiguous pronouns ("it", "that", "this") without context
- A query is UNCLEAR if it's too vague ("tell me about the policy", "how do I update it?")
- A query is CLEAR if it specifies the subject clearly ("What is the annual leave policy?")
- When conversation history is available, resolve pronouns using context
- The rewritten query should be a standalone search query (no pronouns, no ambiguity)
- Sub-queries are optional — only generate them for multi-part questions

Examples of UNCLEAR queries:
- "How do I update it?" → What is "it"?
- "Tell me about that" → About what?
- "What's the policy?" → Which specific policy?
- "Can you help?" → With what?

Examples of CLEAR queries:
- "What is the annual leave policy for Nigeria?"
- "How do I update my employee profile in Business Central?"
- "What are the safety procedures for the manufacturing floor?"

Return ONLY valid JSON matching this exact schema:
{
    "is_clear": true/false,
    "rewritten_query": "optimized search query string",
    "clarification_needed": "question to ask user (empty string if clear)",
    "sub_queries": ["optional", "sub-questions"]
}"""


async def analyze_query(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> QueryAnalysis:
    """
    Analyze a user query for clarity before RAG retrieval.

    Uses the configured LLM to determine if the query is specific enough
    to search documents, and rewrites it for optimal retrieval.

    Args:
        query: Raw user query
        conversation_history: Recent chat messages for pronoun resolution

    Returns:
        QueryAnalysis with clarity assessment and rewritten query

    Complexity: O(1) — single LLM call
    """
    # Build context from conversation history if available
    context = ""
    if conversation_history:
        recent = conversation_history[-2:]  # Last 2 messages for context
        context_parts = [
            f"{msg['role']}: {msg['content']}" for msg in recent
        ]
        context = "\n".join(context_parts)

    user_prompt = f"Query: \"{query}\""
    if context:
        user_prompt += f"\n\nRecent conversation context:\n{context}"
    user_prompt += "\n\nAnalyze this query and return the JSON result."

    try:
        result_text = await _call_llm(
            system_prompt=_CLARIFIER_PROMPT,
            user_prompt=user_prompt,
        )

        # Parse JSON response
        parsed = json.loads(result_text)
        analysis = QueryAnalysis(**parsed)

        logger.info(
            f"Query analysis: clear={analysis.is_clear}, "
            f"rewritten='{analysis.rewritten_query[:50]}...'"
        )
        return analysis

    except json.JSONDecodeError as e:
        logger.warning(f"Query clarifier returned invalid JSON: {e}")
        # Fail-open: treat as clear, use original query
        return QueryAnalysis(
            is_clear=True,
            rewritten_query=query,
            clarification_needed="",
        )
    except Exception as e:
        logger.error(f"Query clarifier failed: {e}")
        # Fail-open: don't block the user on clarifier errors
        return QueryAnalysis(
            is_clear=True,
            rewritten_query=query,
            clarification_needed="",
        )


async def _call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Call the configured LLM provider for text generation.

    Currently routes to Gemini or Azure OpenAI based on settings.
    Returns raw text response.

    Why separate: Isolates LLM provider coupling to one function.
    Swapping providers only requires changing this function.
    """
    provider = settings.EMBEDDING_PROVIDER  # Reuse provider config

    if provider == "gemini":
        return await _call_gemini(system_prompt, user_prompt)
    elif provider == "azure_openai":
        return await _call_azure_openai(system_prompt, user_prompt)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


async def _call_gemini(system_prompt: str, user_prompt: str) -> str:
    """Generate text using Google Gemini."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    response = await client.aio.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    if not response.text:
        raise ValueError("Empty response from Gemini")

    return response.text


async def _call_azure_openai(system_prompt: str, user_prompt: str) -> str:
    """Generate text using Azure OpenAI."""
    from openai import AsyncAzureOpenAI

    client = AsyncAzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
    )

    response = await client.chat.completions.create(
        model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from Azure OpenAI")

    return content
