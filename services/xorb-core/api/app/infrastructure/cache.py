"""Caching infrastructure with Redis backend."""
import json
import pickle
from typing import Any, Optional, Union

import redis.asyncio as redis
from pydantic import BaseModel


class CacheBackend:
    """Async cache backend interface."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = await self.redis.get(key)
            if value is None:
                return None

            # Try JSON first, fallback to pickle
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return pickle.loads(value)
        except Exception:
            return None

    async def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None
    ) -> bool:
        """Set value in cache with optional expiration."""
        try:
            # Serialize value
            if isinstance(value, (dict, list, str, int, float, bool)):
                serialized = json.dumps(value)
            else:
                serialized = pickle.dumps(value)

            if expire:
                return await self.redis.setex(key, expire, serialized)
            else:
                return await self.redis.set(key, serialized)
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            result = await self.redis.delete(key)
            return result > 0
        except Exception:
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return await self.redis.exists(key) > 0
        except Exception:
            return False

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on existing key."""
        try:
            return await self.redis.expire(key, seconds)
        except Exception:
            return False

    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment key value."""
        try:
            return await self.redis.incrby(key, amount)
        except Exception:
            return None

    async def decr(self, key: str, amount: int = 1) -> Optional[int]:
        """Decrement key value."""
        try:
            return await self.redis.decrby(key, amount)
        except Exception:
            return None


# Global cache instance
_cache: Optional[CacheBackend] = None


def init_cache(redis_client: redis.Redis) -> None:
    """Initialize global cache backend."""
    global _cache
    _cache = CacheBackend(redis_client)


def get_cache() -> CacheBackend:
    """Get global cache backend."""
    if _cache is None:
        raise RuntimeError("Cache not initialized")
    return _cache
