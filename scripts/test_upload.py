"""
End-to-End Test Script: Login and PDF Ingestion

This script uses the FastAPI TestClient to:
1. Log in as the Sales Rep (created by the seed script).
2. Create a dummy PDF file in memory.
3. Call the /ingest/pdf endpoint with the Bearer token.
"""

import asyncio
import httpx
from urllib.parse import urlparse
from main import app
import os

async def test_login_and_upload():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        print("1. Logging in as Sales Rep...")
        login_response = await client.post(
            "/api/v1/auth/login",
            json={
                "email": "sales.rep@coragem.com",
                "password": "Password123!"
            }
        )
        
        if login_response.status_code != 200:
            print(f"❌ Login Failed! {login_response.text}")
            return
            
        token = login_response.json()["data"]["access_token"]
        print("✅ Login Successful! Acquired Access Token.")
        
        print("\n2. Loading 'product_catalog.pdf' from disk for upload...")
        
        # Path to our freshly created PDF
        pdf_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "departments", "sales", "product_catalog.pdf"
        )
        
        if not os.path.exists(pdf_path):
            print(f"❌ Could not find {pdf_path}. Did you run the conversion script?")
            return
            
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        
        print("3. Calling /ingest/pdf endpoint...")
        headers = {
            "Authorization": f"Bearer {token}"
        }
        
        data = {
            "title": "Q3 Sales Playbook",
            "department": "Sales",
            "doc_type": "Guide",
            "source_url": "https://coragem.com/sales/q3.pdf"
        }
        
        files = {
            "file": ("q3_sales_playbook.pdf", pdf_bytes, "application/pdf")
        }
        
        ingest_response = await client.post(
            "/api/v1/documents/ingest/pdf",
            headers=headers,
            data=data,
            files=files
        )
        
        if ingest_response.status_code == 201:
            print("✅ Document Ingested Successfully!")
            print(ingest_response.json())
        else:
            print(f"❌ Ingestion Failed: Response Code {ingest_response.status_code}")
            print(ingest_response.text)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_login_and_upload())
