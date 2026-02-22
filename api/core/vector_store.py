"""
vector_store module.

Pinecone vector database operations with metadata filtering.
CRITICAL: Department isolation happens here via metadata filters.
"""

from typing import List, Dict, Optional
import time
import asyncio
from pinecone import Pinecone, ServerlessSpec

from api.core.embeddings import generate_embeddings, get_embedding_dimension
from api.utils.logger import get_logger
from api.config.settings import settings

logger = get_logger(__name__)


class VectorStore:
    """
    Wrapper around Pinecone for document embeddings.
    
    Key feature: Metadata filtering for department isolation.
    """
    
    def __init__(self, api_key: str, index_name: str = "coragem-documents"):
        """
        Initialize Pinecone client.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of index to use
        """
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        
        # We don't connect to the index yet; we check/create it in create_collection
        self.index = None
        logger.info(f"Initialized VectorStore for Pinecone index: {index_name}")

    def _get_index(self):
        """Helper to lazy-load the index connection."""
        if self.index is None:
            self.index = self.pc.Index(self.index_name)
        return self.index
        
    async def create_collection(self, vector_size: int = None) -> None:
        """
        Create Pinecone index if not exists.
        
        Args:
            vector_size: Embedding dimension 
        """
        if vector_size is None:
            vector_size = get_embedding_dimension()
            
        try:
            active_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in active_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                await asyncio.to_thread(
                    self.pc.create_index,
                    name=self.index_name,
                    dimension=vector_size,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to be ready
                while not (await asyncio.to_thread(self.pc.describe_index, self.index_name)).status["ready"]:
                    await asyncio.sleep(1)
                logger.info(f"Created index: {self.index_name}")
            else:
                logger.info(f"Index already exists: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    async def upsert(
        self,
        id: str,
        vector: List[float],
        metadata: Dict
    ) -> None:
        """
        Insert or update document embedding.
        
        Args:
            id: Unique document chunk ID
            vector: Embedding vector
            metadata: MUST include 'department', 'is_active', 'title', etc.
            
        CRITICAL: metadata['department'] is used for filtering
        """
        index = self._get_index()
        
        # Pinecone upsert format: [(id, vector, metadata)]
        await asyncio.to_thread(
            index.upsert,
            vectors=[(id, vector, metadata)]
        )
        
        logger.debug(f"Upserted vector: {id}, dept: {metadata.get('department')}")
    
    async def search(
        self,
        query: str,
        filter: Dict,
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Search for similar documents with metadata filtering.
        
        Args:
            query: Query text (will be embedded)
            filter: Metadata filters (MUST include department)
            limit: Max results
            score_threshold: Minimum similarity score
            
        Returns:
            List of dicts with 'content', 'score', 'metadata'
            
        CRITICAL: This is where department isolation happens
        
        Example filter:
            {"department": "Sales", "is_active": True}
        """
        index = self._get_index()
        
        # Generate query embedding
        query_embedding = await generate_embeddings(query)
        
        # Search using Pinecone syntax
        response = await asyncio.to_thread(
            index.query,
            vector=query_embedding,
            filter=filter,
            top_k=limit,
            include_metadata=True
        )
        
        # Format results and apply score threshold
        documents = []
        for match in response.get("matches", []):
            if match["score"] >= score_threshold:
                metadata = match.get("metadata", {})
                documents.append({
                    "content": metadata.get("content", ""),
                    "score": match["score"],
                    "metadata": {
                        "document_id": metadata.get("document_id"),
                        "title": metadata.get("title"),
                        "department": metadata.get("department"),
                        "chunk_index": metadata.get("chunk_index"),
                        "doc_type": metadata.get("doc_type"),
                    }
                })
        
        logger.info(
            f"Search: query='{query[:50]}...', "
            f"dept={filter.get('department')}, "
            f"results={len(documents)}"
        )
        
        return documents
    
    async def delete_by_id(self, id: str) -> None:
        """
        Delete document chunk by ID.
        """
        index = self._get_index()
        await asyncio.to_thread(index.delete, ids=[id])
        logger.debug(f"Deleted vector: {id}")
    
    async def update_metadata_by_ids(
        self,
        ids: List[str],
        updates: Dict
    ) -> None:
        """
        Update metadata for specific document chunks by ID.
        """
        index = self._get_index()
        
        updated_count = 0
        # Pinecone recommend batching updates depending on tier, but typical sync is fine for few
        for doc_id in ids:
            await asyncio.to_thread(
                index.update,
                id=doc_id,
                set_metadata=updates
            )
            updated_count += 1
            
        logger.info(f"Updated metadata for {updated_count} specific vectors.")