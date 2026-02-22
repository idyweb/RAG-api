"""
LLM generation service.

Uses Google Gemini for answer generation.
"""

from typing import List, Dict, AsyncGenerator, Optional
from google import genai
from google.genai import types
from api.config.settings import settings
from api.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize Gemini client
client = genai.Client(api_key=settings.GEMINI_API_KEY)


async def generate_answer(
    query: str, 
    docs: List[Dict], 
    department: str, 
    language: str = "en",
    chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Generate answer using retrieved documents as context.
    
    Args:
        query: User's question
        docs: List of retrieved documents with 'content' and 'metadata'
        department: User's department for enhanced prompt context
        language: User's intended language for response
        chat_history: Previous conversational messages

        
    Returns:
        Generated answer
        
    Note: Prompt engineering is critical here to prevent hallucinations.
    """
    if not docs:
        return "I don't have enough information to answer this question."
    
    # Build context from retrieved documents
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(
            f"[Document {i} - {doc['metadata']['title']}]\n{doc['content']}"
        )
    
    context = "\n\n".join(context_parts)
    
    # Prompt engineering for accuracy
    system_prompt = _build_system_prompt(department, language)

    user_prompt = f"""Context documents:

{context}

Question: {query}

Answer (based only on the context above):"""

    # Build history contents list for Gemini
    contents = []
    if chat_history:
        for msg in chat_history:
            # map 'user'/'assistant' to Gemini's 'user'/'model'
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
            )
            
    # Append the current query
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]))
    
    try:
        response = await client.aio.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,  # Low temperature for consistency
                max_output_tokens=500
            )
        )
        
        answer = response.text.strip()
        
        logger.info(
            f"Generated answer: {len(answer)} chars, model={settings.GEMINI_MODEL}"
        )
        
        return answer
        
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return "I encountered an error generating the answer. Please try again."


async def generate_answer_stream(
    query: str, 
    docs: List[Dict], 
    department: str, 
    language: str = "en",
    chat_history: Optional[List[Dict[str, str]]] = None
) -> AsyncGenerator[str, None]:
    """
    Generate streaming answer using retrieved documents as context.
    Yields chunks of text as they arrive from the LLM.
    """
    if not docs:
        yield "I don't have enough information to answer this question."
        return
        
    # Build context from retrieved documents
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(
            f"[Document {i} - {doc['metadata']['title']}]\n{doc['content']}"
        )
    
    context = "\n\n".join(context_parts)
    
    # Prompt engineering for accuracy
    system_prompt = _build_system_prompt(department, language)

    user_prompt = f"""Context documents:

{context}

Question: {query}

Answer (based only on the context above):"""

    # Build history contents list for Gemini
    contents = []
    if chat_history:
        for msg in chat_history:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
            )
            
    # Append the current query
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]))
    
    try:
        response_stream = await client.aio.models.generate_content_stream(
            model=settings.GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
                max_output_tokens=500
            )
        )
        
        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text
                
    except Exception as e:
        logger.error(f"LLM streaming generation failed: {e}")
        yield "I encountered an error generating the answer."

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

### Citations
- Reference documents explicitly: "According to the [Document Name]..."
- Include document numbers when multiple sources used: "[Doc 1], [Doc 2]"
- For policies, cite version and effective date if available

### Tone
- Professional but approachable
- Clear and concise (aim for 2-3 paragraphs maximum)
- Use bullet points for lists or multiple items
- Avoid jargon unless it's department-specific terminology

### Formatting
- Use markdown formatting for readability
- Bold key information
- Use bullet points for clarity
- Include relevant numbers, dates, amounts exactly as stated in documents

### Department Context
You are currently assisting a {department} department employee. Tailor your language and examples to their context when appropriate.

### Multilingual Support
If the query is in a language other than English, respond in that language when possible, but always maintain accuracy over fluency.

## Error Handling
If the query is:
- Vague: Ask for clarification with specific options
- Outside your scope: Direct to appropriate human contact
- Urgent/safety-related: Flag immediately and provide emergency contact info

## Examples

Good Response:
"According to the **Product Catalog 2024** [Doc 1], Cowbell Chocolate 400g is priced at:
- Retail: ₦2,500/tin
- Distributor: ₦2,200/tin
- Minimum order: 24 tins per carton

For current promotions, please check with your Regional Sales Manager."

Bad Response:
"I think the price is around 2500 naira but I'm not completely sure. You might want to double check with someone."
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