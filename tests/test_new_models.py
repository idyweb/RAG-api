"""
Tests for database schema expansion and FastMCP integration.

Run with: PYTHONPATH=. uv run pytest tests/test_new_models.py -v
"""

import pytest
import uuid
import asyncio

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from httpx import AsyncClient, ASGITransport

from api.apps.agents.models import AgentTool
from api.apps.agents.mcp_server import mcp
from api.config.settings import settings

# ── DB Setup ──────────────────────────────────────────────────────────────────
engine = create_async_engine(settings.DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False)


# ── Schema Tests ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_agent_tool_insertion():
    """Verify AgentTool CRUD and Postgres ARRAY serialization."""
    async with async_session() as session:
        test_tool = AgentTool(
            name=f"test_tool_{uuid.uuid4().hex[:6]}",
            description="A test tool to verify schema update.",
            endpoint_url="http://internal/test",
            allowed_departments=["Finance", "Operations"],
        )

        session.add(test_tool)
        await session.commit()
        await session.refresh(test_tool)

        assert test_tool.id is not None
        assert test_tool.allowed_departments == ["Finance", "Operations"]

        # Clean up
        await session.delete(test_tool)
        await session.commit()


# ── MCP Server Tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mcp_server_tools_registered():
    """Verify the MCP instance has the expected tools registered."""
    tools = await mcp.list_tools()
    tool_names = [t.name for t in tools]
    assert "list_agent_tools" in tool_names, f"Expected 'list_agent_tools' in {tool_names}"
    assert "search_documents" in tool_names, f"Expected 'search_documents' in {tool_names}"


@pytest.mark.asyncio
async def test_mcp_endpoint_reachable():
    """Verify the /mcp endpoint is mounted and does not return 404."""
    from main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/mcp")
        # MCP endpoint should NOT be 404 — any other status is acceptable
        assert response.status_code != 404, (
            f"MCP endpoint returned 404. Expected it to be mounted."
        )


@pytest.mark.asyncio
async def test_existing_health_endpoint():
    """Verify existing /health endpoint still works after MCP mount."""
    from main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
