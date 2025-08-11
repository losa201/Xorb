"""
Production Redis Client with Cross-Version Compatibility
Handles Redis connection management, fallbacks, and graceful degradation
"""

import asyncio
import logging
from typing import Optional, Any, Dict, List, Union
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Redis compatibility layer
try:
    import redis.asyncio as redis_async
    REDIS_ASYNCIO_AVAILABLE = True
except ImportError:
    REDIS_ASYNCIO_AVAILABLE = False
    redis_async = None

try:
    import aioredis
    AIOREDIS_AVAILABLE = True
except (ImportError, TypeError):
    # TypeError can occur with Python 3.12 compatibility issues
    AIOREDIS_AVAILABLE = False
    aioredis = None

try:
    import redis
    REDIS_SYNC_AVAILABLE = True
except ImportError:
    REDIS_SYNC_AVAILABLE = False
    redis = None


class InMemoryRedisClient:
    """In-memory Redis-compatible client for fallback"""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._expiry: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            await self._cleanup_expired()
            return self._data.get(key)
    
    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        async with self._lock:
            self._data[key] = value
            if ex:
                self._expiry[key] = datetime.utcnow() + timedelta(seconds=ex)
            return True
    
    async def setex(self, key: str, time: int, value: str) -> bool:
        return await self.set(key, value, ex=time)
    
    async def delete(self, key: str) -> int:
        async with self._lock:
            if key in self._data:
                del self._data[key]
                if key in self._expiry:
                    del self._expiry[key]
                return 1
            return 0
    
    async def exists(self, key: str) -> int:
        async with self._lock:
            await self._cleanup_expired()
            return 1 if key in self._data else 0
    
    async def incr(self, key: str) -> int:
        return await self.incrby(key, 1)
    
    async def incrby(self, key: str, amount: int) -> int:
        async with self._lock:
            current = int(self._data.get(key, 0))
            new_value = current + amount
            self._data[key] = str(new_value)
            return new_value
    
    async def decr(self, key: str) -> int:
        return await self.decrby(key, 1)
    
    async def decrby(self, key: str, amount: int) -> int:
        async with self._lock:
            current = int(self._data.get(key, 0))
            new_value = max(0, current - amount)
            self._data[key] = str(new_value)
            return new_value
    
    async def expire(self, key: str, time: int) -> bool:
        async with self._lock:
            if key in self._data:
                self._expiry[key] = datetime.utcnow() + timedelta(seconds=time)
                return True
            return False
    
    async def ttl(self, key: str) -> int:
        async with self._lock:
            if key in self._expiry:
                remaining = (self._expiry[key] - datetime.utcnow()).total_seconds()
                return int(max(0, remaining))
            return -1 if key in self._data else -2
    
    async def hget(self, key: str, field: str) -> Optional[str]:
        async with self._lock:
            hash_data = self._data.get(key, {})
            if isinstance(hash_data, dict):
                return hash_data.get(field)
            return None
    
    async def hset(self, key: str, field: str, value: str) -> int:
        async with self._lock:
            if key not in self._data or not isinstance(self._data[key], dict):
                self._data[key] = {}
            self._data[key][field] = value
            return 1
    
    async def hgetall(self, key: str) -> Dict[str, str]:
        async with self._lock:
            hash_data = self._data.get(key, {})
            return hash_data if isinstance(hash_data, dict) else {}
    
    async def lpush(self, key: str, *values: str) -> int:
        async with self._lock:
            if key not in self._data or not isinstance(self._data[key], list):
                self._data[key] = []
            for value in reversed(values):
                self._data[key].insert(0, value)
            return len(self._data[key])
    
    async def rpop(self, key: str) -> Optional[str]:
        async with self._lock:
            if key in self._data and isinstance(self._data[key], list) and self._data[key]:
                return self._data[key].pop()
            return None
    
    async def llen(self, key: str) -> int:
        async with self._lock:
            if key in self._data and isinstance(self._data[key], list):
                return len(self._data[key])
            return 0
    
    async def _cleanup_expired(self):
        """Remove expired keys"""
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, expiry in self._expiry.items() 
            if expiry <= current_time
        ]
        for key in expired_keys:
            if key in self._data:
                del self._data[key]
            del self._expiry[key]


class UniversalRedisClient:
    """Universal Redis client that handles multiple Redis library versions"""
    
    def __init__(self, url: str = "redis://localhost:6379/0", **kwargs):
        self.url = url
        self.kwargs = kwargs
        self.client = None
        self._initialized = False
        self._fallback_client = None
    
    async def initialize(self) -> bool:
        """Initialize Redis connection with fallback handling"""
        if self._initialized:
            return True
        
        try:
            # Try redis.asyncio first (Python 3.12+ preferred)
            if REDIS_ASYNCIO_AVAILABLE:
                try:
                    self.client = redis_async.from_url(
                        self.url, 
                        decode_responses=True,
                        **self.kwargs
                    )
                    # Test connection
                    await self.client.ping()
                    logger.info("Connected to Redis using redis.asyncio")
                    self._initialized = True
                    return True
                except Exception as e:
                    logger.warning(f"redis.asyncio connection failed: {e}")
            
            # Try aioredis as fallback
            if AIOREDIS_AVAILABLE:
                try:
                    if hasattr(aioredis, 'from_url'):
                        self.client = await aioredis.from_url(self.url, **self.kwargs)
                    else:
                        # Older aioredis API
                        self.client = await aioredis.create_redis_pool(self.url, **self.kwargs)
                    
                    # Test connection
                    await self.client.ping()
                    logger.info("Connected to Redis using aioredis")
                    self._initialized = True
                    return True
                except Exception as e:
                    logger.warning(f"aioredis connection failed: {e}")
            
            # Fall back to in-memory client
            logger.warning("Redis not available, using in-memory fallback")
            self._fallback_client = InMemoryRedisClient()
            self.client = self._fallback_client
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Redis initialization failed: {e}")
            self._fallback_client = InMemoryRedisClient()
            self.client = self._fallback_client
            self._initialized = True
            return False
    
    async def _ensure_connection(self):
        """Ensure Redis connection is established"""
        if not self._initialized:
            await self.initialize()
    
    async def get(self, key: str) -> Optional[str]:
        """Get value by key"""
        await self._ensure_connection()
        try:
            result = await self.client.get(key)
            return result.decode() if isinstance(result, bytes) else result
        except Exception as e:
            logger.error(f"Redis GET failed: {e}")
            return None
    
    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set key-value pair with optional expiration"""
        await self._ensure_connection()
        try:
            if ex:
                result = await self.client.setex(key, ex, value)
            else:
                result = await self.client.set(key, value)
            return bool(result)
        except Exception as e:
            logger.error(f"Redis SET failed: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key"""
        await self._ensure_connection()
        try:
            result = await self.client.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Redis DELETE failed: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        await self._ensure_connection()
        try:
            result = await self.client.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Redis EXISTS failed: {e}")
            return False
    
    async def incr(self, key: str) -> int:
        """Increment counter"""
        await self._ensure_connection()
        try:
            return await self.client.incr(key)
        except Exception as e:
            logger.error(f"Redis INCR failed: {e}")
            return 0
    
    async def incrby(self, key: str, amount: int) -> int:
        """Increment counter by amount"""
        await self._ensure_connection()
        try:
            return await self.client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Redis INCRBY failed: {e}")
            return 0
    
    async def expire(self, key: str, time: int) -> bool:
        """Set key expiration"""
        await self._ensure_connection()
        try:
            result = await self.client.expire(key, time)
            return bool(result)
        except Exception as e:
            logger.error(f"Redis EXPIRE failed: {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """Get time to live for key"""
        await self._ensure_connection()
        try:
            return await self.client.ttl(key)
        except Exception as e:
            logger.error(f"Redis TTL failed: {e}")
            return -2
    
    async def hget(self, key: str, field: str) -> Optional[str]:
        """Get hash field value"""
        await self._ensure_connection()
        try:
            result = await self.client.hget(key, field)
            return result.decode() if isinstance(result, bytes) else result
        except Exception as e:
            logger.error(f"Redis HGET failed: {e}")
            return None
    
    async def hset(self, key: str, field: str, value: str) -> bool:
        """Set hash field value"""
        await self._ensure_connection()
        try:
            result = await self.client.hset(key, field, value)
            return bool(result)
        except Exception as e:
            logger.error(f"Redis HSET failed: {e}")
            return False
    
    async def hgetall(self, key: str) -> Dict[str, str]:
        """Get all hash fields"""
        await self._ensure_connection()
        try:
            result = await self.client.hgetall(key)
            # Handle bytes decoding
            if isinstance(result, dict):
                return {
                    k.decode() if isinstance(k, bytes) else k: 
                    v.decode() if isinstance(v, bytes) else v 
                    for k, v in result.items()
                }
            return result or {}
        except Exception as e:
            logger.error(f"Redis HGETALL failed: {e}")
            return {}
    
    async def lpush(self, key: str, *values: str) -> int:
        """Push values to list"""
        await self._ensure_connection()
        try:
            return await self.client.lpush(key, *values)
        except Exception as e:
            logger.error(f"Redis LPUSH failed: {e}")
            return 0
    
    async def rpop(self, key: str) -> Optional[str]:
        """Pop value from list"""
        await self._ensure_connection()
        try:
            result = await self.client.rpop(key)
            return result.decode() if isinstance(result, bytes) else result
        except Exception as e:
            logger.error(f"Redis RPOP failed: {e}")
            return None
    
    async def llen(self, key: str) -> int:
        """Get list length"""
        await self._ensure_connection()
        try:
            return await self.client.llen(key)
        except Exception as e:
            logger.error(f"Redis LLEN failed: {e}")
            return 0
    
    async def ping(self) -> bool:
        """Test Redis connection"""
        await self._ensure_connection()
        try:
            if hasattr(self.client, 'ping'):
                await self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis PING failed: {e}")
            return False
    
    async def close(self):
        """Close Redis connection"""
        if self.client and hasattr(self.client, 'close'):
            try:
                await self.client.close()
            except Exception as e:
                logger.error(f"Redis close failed: {e}")
        self._initialized = False
    
    def is_fallback(self) -> bool:
        """Check if using in-memory fallback"""
        return self._fallback_client is not None


# Global Redis client instance
_redis_client: Optional[UniversalRedisClient] = None


def get_redis_client(url: str = "redis://localhost:6379/0") -> UniversalRedisClient:
    """Get global Redis client instance"""
    global _redis_client
    if _redis_client is None:
        _redis_client = UniversalRedisClient(url)
    return _redis_client


async def init_redis(url: str = "redis://localhost:6379/0") -> bool:
    """Initialize global Redis client"""
    client = get_redis_client(url)
    return await client.initialize()


def redis_health_check() -> Dict[str, Any]:
    """Get Redis health status"""
    global _redis_client
    if _redis_client is None:
        return {"status": "not_initialized", "fallback": False}
    
    return {
        "status": "initialized" if _redis_client._initialized else "not_initialized",
        "fallback": _redis_client.is_fallback(),
        "available_backends": {
            "redis_asyncio": REDIS_ASYNCIO_AVAILABLE,
            "aioredis": AIOREDIS_AVAILABLE,
            "redis_sync": REDIS_SYNC_AVAILABLE
        }
    }