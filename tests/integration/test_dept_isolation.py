import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio

@pytest.mark.skip(reason="Missing conftest.py fixtures configuration")
async def test_sales_user_cannot_see_hr_docs(async_client: AsyncClient, token_hr: str, token_sales: str):
    """CRITICAL: Verify dept isolation works"""
    
    # 1. Ingest HR doc as HR user
    doc_payload = {
        "title": "HR Confidential Policy",
        "doc_type": "Policy",
        "department": "HR",
        "content": "This is a strictly confidential HR policy.",
        "source_url": "https://hr.promasidor.com/policy.pdf"
    }
    
    ingest_resp = await async_client.post(
        "/api/v1/documents",
        json=doc_payload,
        headers={"Authorization": f"Bearer {token_hr}"}
    )
    assert ingest_resp.status_code == 201
    
    # 2. Query as Sales user
    query_payload = {
        "query": "HR Confidential Policy",
        "max_results": 5,
        "confidence_threshold": 0.5
    }
    
    query_resp = await async_client.post(
        "/api/v1/rag/query",
        json=query_payload,
        headers={"Authorization": f"Bearer {token_sales}"}
    )
    assert query_resp.status_code == 200
    
    response_data = query_resp.json()
    
    # 3. Should get "I don't know" / low confidence
    assert response_data["confidence"] == "low"
    assert len(response_data["sources"]) == 0
