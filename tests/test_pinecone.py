import asyncio
from api.config.settings import settings
from api.core.vector_store import VectorStore
import logging

logging.basicConfig(level=logging.INFO)

async def test():
    print("Testing Pinecone retrieve...")
    try:
        vs = VectorStore(settings.PINECONE_API_KEY, settings.PINECONE_INDEX_NAME)
        results = await vs.search(
            query="cowbell and miksi", 
            filter={"department": "Sales", "is_active": True}, 
            limit=5, 
            score_threshold=0.0
        )
        print("RESULTS:", len(results))
        for r in results:
            print("Score:", r['score'])
            print("Title:", r['metadata'].get('title'))
    except Exception as e:
        print("ERROR:", e)

asyncio.run(test())
