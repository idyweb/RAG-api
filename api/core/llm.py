"""
LLM generation service.

Uses Google Gemini for answer generation.
Streaming-only architecture — all responses are yielded as async generators.
"""

from typing import List, Dict, AsyncGenerator, Optional
from api.config.settings import settings
from api.utils.logger import get_logger

logger = get_logger(__name__)


# ── Private Helpers ───────────────────────────────────────────────────────────


def _build_context(docs: List[Dict]) -> str:
    """
    Build context string from retrieved documents.

    Uses only the document title as the label — no numeric index.
    This forces the LLM to cite by title (e.g. "According to the **Code of Conduct**")
    instead of generic references like "Document 3".
    """
    context_parts = []
    seen_titles: dict[str, int] = {}
    for doc in docs:
        title = doc["metadata"]["title"]
        content = doc["content"]
        # Deduplicate labels when same doc appears multiple times (different chunks)
        seen_titles[title] = seen_titles.get(title, 0) + 1
        if seen_titles[title] > 1:
            label = f"[{title} — section {seen_titles[title]}]"
        else:
            label = f"[{title}]"
        context_parts.append(f"{label}\n{content}")
    return "\n\n".join(context_parts)


import json
import httpx

# Shared client with connection pooling — avoids TCP+TLS handshake per request.
# Limits set to handle enterprise concurrency without exhausting OS sockets.
_http_client = httpx.AsyncClient(
    timeout=60.0,
    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
)


def _build_chat_contents(
    system_prompt: str,
    user_prompt: str,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> List[Dict]:
    """
    Build standard OpenAI-compatible messages list.
    """
    messages = [{"role": "system", "content": system_prompt}]

    if chat_history:
        for msg in chat_history:
            # chat_history roles are 'user' and 'assistant'
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    # Current query always goes last
    messages.append({
        "role": "user",
        "content": user_prompt
    })
    
    return messages


# ── Public API (Streaming Only) ──────────────────────────────────────────────


async def generate_answer_stream(
    query: str,
    docs: List[Dict],
    department: str,
    language: str = "en",
    chat_history: Optional[List[Dict[str, str]]] = None
) -> AsyncGenerator[str, None]:
    """
    Stream answer tokens using retrieved documents as context.

    This uses httpx to bypass python SDK issues, hitting Gemini's
    OpenAI-compatible REST SSE endpoint directly.
    """
    if not docs:
        yield "I don't have enough information to answer this question."
        return

    context = _build_context(docs)
    system_prompt = _build_system_prompt(department, language)

    user_prompt = f"""Context documents:

{context}

Question: {query}

Answer (based only on the context above):"""

    messages = _build_chat_contents(system_prompt, user_prompt, chat_history)

    payload = {
        "model": settings.GEMINI_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "stream": True,
    }
    
    # Use Gemini's OpenAI-compatible endpoint
    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        logger.info(f"Connecting to Gemini REST API: {settings.GEMINI_MODEL}")

        async with _http_client.stream("POST", url, headers=headers, json=payload) as response:

            if response.status_code != 200:
                error_msg = await response.aread()
                logger.error(f"Gemini API returned {response.status_code}: {error_msg.decode('utf-8')}")
                yield "I encountered an error connecting to the AI provider."
                return

            chunk_count = 0
            finish_reason = None
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()

                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        choice = data.get("choices", [{}])[0]
                        content = choice.get("delta", {}).get("content")
                        # Capture finish_reason from the final chunk
                        fr = choice.get("finish_reason")
                        if fr:
                            finish_reason = fr

                        if content:
                            chunk_count += 1
                            yield content

                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse stream chunk: {data_str}")
                        continue

            if finish_reason == "length":
                logger.warning(
                    f"Gemini response TRUNCATED (hit max_tokens={settings.LLM_MAX_TOKENS}). "
                    f"Consider increasing LLM_MAX_TOKENS."
                )
            logger.info(f"Gemini streaming completed. chunks={chunk_count}, finish_reason={finish_reason}")

    except httpx.RequestError as e:
        logger.error(f"HTTP request error with Gemini API: {str(e)}")
        yield "I encountered a network error while connecting to the AI."
    except Exception as e:
        logger.error(f"LLM streaming generation failed with error: {e}")
        yield "I encountered an unexpected error generating the answer."


# ── Prompt Engineering ────────────────────────────────────────────────────────


def _build_system_prompt(department: str, language: str = "en") -> str:
    """
    Build department-specific system prompt.

    Args:
        department: User's department (for context)
        language: Response language (for multinational support)
    """

    base_prompt = f"""You are an AI assistant for {settings.COMPANY_NAME} employees.

## Your Role
You provide accurate, professional information based strictly on company documents. You help employees across our operations in over 30 countries find answers quickly.

## Response Guidelines

### Accuracy (CRITICAL)
- Answer ONLY using the provided context documents
- If information is not in the context, respond: "I don't have this information in my current knowledge base. Please contact [relevant department] for assistance."
- Never speculate, guess, or use external knowledge
- When uncertain, err on the side of directing to human support

### Citations (CRITICAL)
- ALWAYS cite the document by its exact title in bold. The title appears in square brackets at the start of each context block (e.g. [Code of Conduct]).
- NEVER use numbered references like "Document 1", "Document 3", "Doc 1", or "[1]". Always use the real title.
- Format: "According to the **Code of Conduct**..." or "Based on the **Employee Handbook**..."
- Multiple sources: "According to the **Travel Policy** and the **Expense Guidelines**..."

### Completeness (CRITICAL)
- Always finish your answer. Do not stop mid-sentence or mid-list.
- If listing items, list ALL of them — do not truncate.

### Tone and Formatting
- Professional but approachable
- Use markdown: bold key info, bullet points for lists
- Include relevant numbers, dates, amounts exactly as stated in documents
- Aim for completeness over brevity — it's better to give a full answer than a cut-off one

### Department Context
You are currently assisting a {department} department employee. Tailor your language and examples to their context when appropriate.

### Multilingual Support
If the query is in a language other than English, respond in that language when possible, but always maintain accuracy over fluency.

## Error Handling
If the query is:
- Vague: Ask for clarification with specific options
- Outside your scope: Direct to appropriate human contact
- Urgent/safety-related: Flag immediately and provide emergency contact info

## Example

Good: "According to the **Product Catalog 2024**, Cowbell Chocolate 400g is priced at:
- Retail: ₦2,500/tin
- Distributor: ₦2,200/tin
- Minimum order: 24 tins per carton

For current promotions, please check with your Regional Sales Manager."

Bad: "I think the price is around 2500 naira but I'm not completely sure." (speculating, no citation)
Bad: "According to **Document 3**..." (using a number instead of the real document title)
"""

    # Add language-specific instructions
    if language != "en":
        base_prompt += f"\n\n**Language Note:** Respond in {_get_language_name(language)} while maintaining accuracy."

    return base_prompt


def _get_language_name(code: str) -> str:
    """Map language codes to names."""
    lang_map = {
        "en": "English",
        "fr": "French",
        "pt": "Portuguese",
        "ar": "Arabic",
        "sw": "Swahili",
        # Add more as needed for Coragem's markets
    }
    return lang_map.get(code, "English")