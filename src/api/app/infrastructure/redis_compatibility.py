"""
Redis Compatibility Layer for Python 3.12+
Provides a unified interface for different Redis client versions
"""

import logging
from typing import Optional, Any, Dict, List, Union
import asyncio
import time

logger = logging.getLogger(__name__)

# Try different Redis client options with fallbacks
REDIS_CLIENT = None
REDIS_CLIENT_TYPE = None

# Option 1: redis.asyncio (preferred for Python 3.12+)
try:
    import redis.asyncio as redis_async
    REDIS_CLIENT = redis_async
    REDIS_CLIENT_TYPE = "redis.asyncio"
    logger.info("Using redis.asyncio client")
except ImportError:
    pass

# Option 2: aioredis (legacy fallback)
if REDIS_CLIENT is None:
    try:
        import aioredis
        REDIS_CLIENT = aioredis
        REDIS_CLIENT_TYPE = "aioredis"
        logger.info("Using aioredis client")
    except ImportError:
        pass

# Option 3: In-memory fallback
if REDIS_CLIENT is None:
    REDIS_CLIENT_TYPE = "memory"
    logger.warning("No Redis client available, using in-memory fallback")


class CompatibleRedisClient:
    """Redis client compatibility wrapper"""
    
    def __init__(self, url: str = "redis://localhost:6379/0", **kwargs):
        self.url = url
        self.kwargs = kwargs
        self._client = None
        self._memory_store = {}
        
    async def initialize(self):
        """Initialize Redis client with compatibility handling"""
        if REDIS_CLIENT_TYPE == "redis.asyncio":
            try:
                self._client = REDIS_CLIENT.from_url(
                    self.url, 
                    decode_responses=True,
                    **self.kwargs
                )
                # Test connection
                await self._client.ping()
                logger.info("Redis connection established (redis.asyncio)")
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                self._client = None
                
        elif REDIS_CLIENT_TYPE == "aioredis":
            try:
                # Handle different aioredis API versions
                if hasattr(REDIS_CLIENT, 'from_url'):
                    self._client = REDIS_CLIENT.from_url(self.url, **self.kwargs)
                else:
                    # Legacy aioredis API
                    self._client = await REDIS_CLIENT.create_redis_pool(self.url, **self.kwargs)
                
                # Test connection
                await self.ping()
                logger.info("Redis connection established (aioredis)")
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                self._client = None
        
        if self._client is None:
            logger.warning("Using in-memory Redis fallback")
    
    async def ping(self) -> bool:
        """Ping Redis server"""
        try:
            if self._client:
                if REDIS_CLIENT_TYPE == "redis.asyncio":
                    await self._client.ping()
                elif REDIS_CLIENT_TYPE == "aioredis":
                    if hasattr(self._client, 'ping'):
                        await self._client.ping()
                    else:
                        # Legacy aioredis might not have ping
                        await self._client.get("__ping_test__")
                return True
            return False
        except Exception:
            return False
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis"""
        try:
            if self._client:
                result = await self._client.get(key)
                if isinstance(result, bytes):
                    return result.decode('utf-8')
                return result
            else:
                data = self._memory_store.get(key)
                if isinstance(data, dict):
                    # Check if expired
                    if data.get("expires") and time.time() > data["expires"]:
                        del self._memory_store[key]
                        return None
                    return data.get("value")
                return data
        except Exception as e:
            logger.error(f"Redis GET failed for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set value in Redis"""
        try:
            if self._client:
                if ex:
                    await self._client.setex(key, ex, value)
                else:
                    await self._client.set(key, value)
                return True
            else:
                # Implement TTL for memory store using scheduled cleanup
                if ex:
                    expiry_time = time.time() + ex
                    self._memory_store[key] = {"value": value, "expires": expiry_time}
                else:
                    self._memory_store[key] = {"value": value, "expires": None}
                
                # Schedule cleanup of expired keys
                asyncio.create_task(self._cleanup_expired_memory_keys())
                return True
        except Exception as e:
            logger.error(f"Redis SET failed for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> int:
        """Delete key from Redis"""
        try:
            if self._client:
                if REDIS_CLIENT_TYPE == "redis.asyncio":
                    return await self._client.delete(key)
                elif REDIS_CLIENT_TYPE == "aioredis":
                    return await self._client.delete(key)
            else:
                if key in self._memory_store:
                    del self._memory_store[key]
                    return 1
                return 0
        except Exception as e:
            logger.error(f"Redis DELETE failed for key {key}: {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        try:
            if self._client:
                result = await self._client.exists(key)
                return bool(result)
            else:
                return key in self._memory_store
        except Exception as e:
            logger.error(f"Redis EXISTS failed for key {key}: {e}")
            return False
    
    async def incr(self, key: str) -> int:
        """Increment value in Redis"""
        try:
            if self._client:
                return await self._client.incr(key)
            else:
                current = int(self._memory_store.get(key, 0))
                current += 1
                self._memory_store[key] = str(current)
                return current
        except Exception as e:
            logger.error(f"Redis INCR failed for key {key}: {e}")
            return 0
    
    async def incrby(self, key: str, amount: int) -> int:
        """Increment value by amount in Redis"""
        try:
            if self._client:
                return await self._client.incrby(key, amount)
            else:
                current = int(self._memory_store.get(key, 0))
                current += amount
                self._memory_store[key] = str(current)
                return current
        except Exception as e:
            logger.error(f"Redis INCRBY failed for key {key}: {e}")
            return 0
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set TTL for key"""
        try:
            if self._client:
                return await self._client.expire(key, seconds)
            else:
                # Memory store doesn't support TTL properly
                return True
        except Exception as e:
            logger.error(f"Redis EXPIRE failed for key {key}: {e}")
            return False
    
    async def hget(self, key: str, field: str) -> Optional[str]:
        """Get hash field value"""
        try:
            if self._client:
                result = await self._client.hget(key, field)
                if isinstance(result, bytes):
                    return result.decode('utf-8')
                return result
            else:
                hash_data = self._memory_store.get(key, {})
                if isinstance(hash_data, dict):
                    return hash_data.get(field)
                return None
        except Exception as e:
            logger.error(f"Redis HGET failed for key {key}, field {field}: {e}")
            return None
    
    async def hset(self, key: str, field: str, value: str) -> bool:
        """Set hash field value"""
        try:
            if self._client:
                await self._client.hset(key, field, value)
                return True
            else:
                if key not in self._memory_store:
                    self._memory_store[key] = {}
                if isinstance(self._memory_store[key], dict):
                    self._memory_store[key][field] = value
                return True
        except Exception as e:
            logger.error(f"Redis HSET failed for key {key}, field {field}: {e}")
            return False
    
    async def _cleanup_expired_memory_keys(self):
        """Clean up expired keys from memory store"""
        try:
            current_time = time.time()
            expired_keys = []
            
            for key, data in self._memory_store.items():
                if isinstance(data, dict) and data.get("expires"):
                    if current_time > data["expires"]:
                        expired_keys.append(key)
            
            for key in expired_keys:
                del self._memory_store[key]
                
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired keys from memory store")
                
        except Exception as e:
            logger.error(f"Memory store cleanup failed: {e}")
    
    async def close(self):
        """Close Redis connection"""
        try:
            if self._client:
                if hasattr(self._client, 'close'):
                    await self._client.close()
                elif hasattr(self._client, 'wait_closed'):
                    self._client.close()
                    await self._client.wait_closed()
        except Exception as e:
            logger.error(f"Redis close failed: {e}")


# Global instance for easy importing
redis_client: Optional[CompatibleRedisClient] = None


async def get_redis_client(url: str = "redis://localhost:6379/0") -> CompatibleRedisClient:
    """Get Redis client instance"""
    global redis_client
    if redis_client is None:
        redis_client = CompatibleRedisClient(url)
        await redis_client.initialize()
    return redis_client


async def close_redis_client():
    """Close global Redis client"""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None