"""
Redis cache manager.

Caches RAG query results by (query + department).
"""

import hashlib
import json
from typing import Optional, Dict, Any, List
import redis.asyncio as redis
from api.config.settings import settings
from api.utils.logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """
    Redis-based cache for RAG queries.
    
    Key insight: Cache must be department-specific.
    Same query from different departments = different results.
    """
    
    def __init__(self, url: str):
        """
        Initialize Redis client.
        
        Args:
            url: Redis connection URL (e.g. redis://localhost:6379)
        """
        self.redis = redis.from_url(url,
        decode_responses=True,
        max_connections=50,
        socket_keepalive=True,
        socket_timeout=5.0,
        retry_on_timeout=True)
        
        logger.info(f"Initialized CacheManager: {url}")
    
    def get_key(self, query: str, department: str) -> str:
        """
        Generate cache key from query + department.
        
        CRITICAL: Department MUST be in the key.
        Otherwise Sales cache could serve HR user (data leak).
        
        Args:
            query: User's question
            department: User's department
            
        Returns:
            Cache key pattern 'rag:{department}:{md5_hash}' for wildcard invalidate support
        """
        combined = f"{query.lower().strip()}:{department}"
        hash_value = hashlib.md5(combined.encode()).hexdigest()
        return f"rag:{department}:{hash_value}"
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result dict or None if not found
        """
        try:
            value = await self.redis.get(key)
            if value:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(value)
            else:
                logger.debug(f"Cache MISS: {key}")
                return None
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: int = 3600
    ) -> None:
        """
        Cache result for ttl seconds.
        
        Args:
            key: Cache key
            value: Result to cache
            ttl: Time-to-live in seconds (default 1 hour)
        """
        try:
            await self.redis.setex(
                key,
                ttl,
                json.dumps(value)
            )
            logger.debug(f"Cached: {key}, ttl={ttl}s")
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
    
    async def delete(self, key: str) -> None:
        """
        Delete cached result.
        
        Args:
            key: Cache key
        """
        try:
            await self.redis.delete(key)
            logger.debug(f"Deleted cache: {key}")
        except Exception as e:
            logger.error(f"Cache delete failed: {e}")
    
    async def invalidate_department(self, department: str) -> int:
        """
        Invalidate all cached queries for a department.
        
        Use case: When new documents are added to a department,
        invalidate cache so users see updated results.
        
        Args:
            department: Department to invalidate
            
        Returns:
            Number of keys deleted
        """
        try:
            # Find all keys matching new pattern: rag:{department}:*
            pattern = f"rag:{department}:*"
            cursor = 0
            deleted = 0
            
            while True:
                cursor, keys = await self.redis.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100
                )
                
                if keys:
                    await self.redis.delete(*keys)
                    deleted += len(keys)
                
                if cursor == 0:
                    break
            
            logger.info(f"Invalidated {deleted} cached queries for dept: {department}")
            return deleted
            
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
            return 0
            
    async def get_chat_history(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Get previous chat messages for a given session.
        Returns up to `limit` most recent messages (default 10).
        """
        try:
            key = f"chat:history:{session_id}"
            # LTRIM keeps the list size bounded, but we also just fetch the latest N.
            # LRANGE returns elements in order they were pushed (0 to -1 is all).
            raw_messages = await self.redis.lrange(key, 0, -1)
            
            messages = []
            for msg in raw_messages:
                messages.append(json.loads(msg))
                
            return messages[-limit:]
            
        except Exception as e:
            logger.error(f"Failed to fetch chat history: {e}")
            return []
            
    async def append_chat_message(self, session_id: str, role: str, content: str, ttl: int = 86400) -> None:
        """
        Append a new message to the chat history for a session.
        TTL resets on every push (default 24 hours).
        """
        try:
            key = f"chat:history:{session_id}"
            msg_data = {
                "role": role,
                "content": content
            }
            
            # Use pipeline to execute LPUSH and EXPIRE atomically
            pipe = self.redis.pipeline()
            pipe.rpush(key, json.dumps(msg_data))
            pipe.expire(key, ttl)
            await pipe.execute()
            
            logger.debug(f"Appended {role} message to session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to append chat message: {e}")
    
    async def close(self) -> None:
        """Close Redis connection."""
        await self.redis.close()