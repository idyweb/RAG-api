import asyncio
from api.core.embeddings import get_embedding_dimension, generate_embeddings
import logging

logging.basicConfig(level=logging.INFO)

async def test():
    dim = get_embedding_dimension()
    vec = await generate_embeddings("cowbell and miksi")
    print(f"Dimension configured: {dim}, Actual vector length: {len(vec)}")

asyncio.run(test())
