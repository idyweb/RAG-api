"""
FastMCP server for the Coragem Enterprise AI Platform.

Exposes internal tools and resources via the Model Context Protocol,
allowing LLMs to natively discover and invoke enterprise capabilities.

Why MCP over REST: MCP provides zero-shot tool discovery for AI agents.
Instead of hardcoding API schemas into prompts, agents connect once and
introspect available tools/resources dynamically.
"""

from typing import Optional

from fastmcp import FastMCP
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from api.config.settings import settings
from api.apps.agents.models import AgentTool
from api.apps.documents.models import Document

# ── Engine (read-only queries, shares the same DB as FastAPI) ─────────────────
_engine = create_async_engine(settings.DATABASE_URL, pool_size=5, max_overflow=2)
_async_session = async_sessionmaker(_engine, expire_on_commit=False)

# ── MCP Server Instance ──────────────────────────────────────────────────────
mcp = FastMCP("Coragem Enterprise MCP")


@mcp.tool()
async def list_agent_tools(department: Optional[str] = None) -> list[dict]:
    """List active agent tools, optionally filtered by department.

    Args:
        department: Filter tools accessible to this department. If None,
                    returns all active tools.

    Returns:
        List of tool dicts with name, description, endpoint_url, and
        allowed_departments.
    """
    async with _async_session() as session:
        query = (
            select(AgentTool)
            .where(AgentTool.is_active == True)
            .where(AgentTool.is_deleted == False)
            .limit(100)
        )

        result = await session.execute(query)
        tools = result.scalars().all()

        if department:
            tools = [
                t for t in tools
                if t.allowed_departments is None
                or department in t.allowed_departments
            ]

        return [
            {
                "name": t.name,
                "description": t.description,
                "endpoint_url": t.endpoint_url,
                "allowed_departments": t.allowed_departments,
            }
            for t in tools
        ]


@mcp.tool()
async def search_documents(
    department: str,
    keyword: Optional[str] = None,
    limit: int = 20,
) -> list[dict]:
    """Search documents accessible to a department.

    Args:
        department: The requesting department. Documents are filtered by
                    allowed_departments or the legacy department field.
        keyword: Optional keyword to filter document titles (case-insensitive).
        limit: Max results to return (default 20, max 100).

    Returns:
        List of document metadata dicts.
    """
    limit = min(limit, 100)

    async with _async_session() as session:
        query = (
            select(Document)
            .where(Document.is_active == True)
            .where(Document.is_deleted == False)
            .limit(limit)
        )

        result = await session.execute(query)
        docs = result.scalars().all()

        # Filter by department access
        filtered = []
        for doc in docs:
            if doc.allowed_departments and department in doc.allowed_departments:
                filtered.append(doc)
            elif doc.department == department:
                filtered.append(doc)

        # Optional keyword filter on title
        if keyword:
            kw_lower = keyword.lower()
            filtered = [d for d in filtered if kw_lower in d.title.lower()]

        return [
            {
                "id": str(doc.id),
                "title": doc.title,
                "department": doc.department,
                "doc_type": doc.doc_type,
                "version": doc.version,
                "allowed_departments": doc.allowed_departments,
            }
            for doc in filtered
        ]


@mcp.resource("agent-tools://list")
async def agent_tools_resource() -> str:
    """Provides a text summary of all active agent tools as a resource.

    This allows LLMs to passively read the tool catalogue without
    invoking a tool call.
    """
    tools = await list_agent_tools()
    if not tools:
        return "No active agent tools configured."

    lines = ["# Active Agent Tools\n"]
    for t in tools:
        depts = ", ".join(t["allowed_departments"]) if t["allowed_departments"] else "All"
        lines.append(f"- **{t['name']}**: {t['description']} (Departments: {depts})")

    return "\n".join(lines)
