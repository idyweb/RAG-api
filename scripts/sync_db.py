import asyncio
from api.db.database import engine
from api.db.base_model import BaseModel

# Import all models to ensure they are registered with BaseModel
from api.apps.auth.models import User, ChannelIdentity
from api.apps.documents.models import Document, DocumentChunk
from api.apps.rag.models import QueryLog
from api.apps.agents.models import AgentTool

async def create_all():
    async with engine.begin() as conn:
        await conn.run_sync(BaseModel.metadata.drop_all)
        await conn.run_sync(BaseModel.metadata.create_all)

if __name__ == "__main__":
    asyncio.run(create_all())
