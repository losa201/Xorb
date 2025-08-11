"""
Cache Manager - Clean Architecture Infrastructure
Unified caching interface with multiple backend support
"""

import logging
import json
import pickle
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class CacheBackend(Enum):
    """Supported cache backends"""
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"


class SerializationMethod(Enum):
    """Serialization methods for cache values"""
    JSON = "json"
    PICKLE = "pickle"
    STRING = "string"


class CacheConfig:
    """Cache configuration"""
    
    def __init__(
        self,
        backend: CacheBackend = CacheBackend.MEMORY,
        default_ttl: int = 3600,
        max_memory_size: int = 1000,
        serialization: SerializationMethod = SerializationMethod.JSON,
        redis_url: Optional[str] = None,
        key_prefix: str = "xorb:",
        compression_threshold: int = 1024
    ):
        self.backend = backend
        self.default_ttl = default_ttl
        self.max_memory_size = max_memory_size
        self.serialization = serialization
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.compression_threshold = compression_threshold


class CacheEntry:
    """Cache entry with metadata"""
    
    def __init__(self, value: Any, ttl: Optional[int] = None):
        self.value = value
        self.created_at = datetime.utcnow()
        self.expires_at = datetime.utcnow() + timedelta(seconds=ttl) if ttl else None
        self.access_count = 0
        self.last_accessed = self.created_at
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def touch(self):
        """Update access metadata"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


class CacheBackendInterface(ABC):
    """Abstract interface for cache backends"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all keys"""
        pass
    
    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check backend health"""
        pass


class MemoryCacheBackend(CacheBackendInterface):
    """In-memory cache backend with LRU eviction"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        async with self._lock:
            entry = self.cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired:
                del self.cache[key]
                return None
            
            entry.touch()
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL"""
        async with self._lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_lru()
            
            self.cache[key] = CacheEntry(value, ttl)
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete key"""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        async with self._lock:
            entry = self.cache.get(key)
            if entry is None:
                return False
            
            if entry.is_expired:
                del self.cache[key]
                return False
            
            return True
    
    async def clear(self) -> bool:
        """Clear all keys"""
        async with self._lock:
            self.cache.clear()
            return True
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        async with self._lock:
            if pattern == "*":
                return list(self.cache.keys())
            
            # Simple pattern matching
            import fnmatch
            return [key for key in self.cache.keys() if fnmatch.fnmatch(key, pattern)]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check backend health"""
        async with self._lock:
            # Clean expired entries
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            return {
                'backend': 'memory',
                'status': 'healthy',
                'entries': len(self.cache),
                'max_size': self.max_size,
                'expired_cleaned': len(expired_keys)
            }
    
    async def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.cache:
            return
        
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_accessed
        )
        
        del self.cache[lru_key]


class RedisCacheBackend(CacheBackendInterface):
    """Redis cache backend"""
    
    def __init__(self, redis_url: str, key_prefix: str = "xorb:"):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._redis = None
        self._lock = asyncio.Lock()
    
    async def _get_redis(self):
        """Get Redis connection"""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(self.redis_url)
                # Test connection
                await self._redis.ping()
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        
        return self._redis
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key"""
        return f"{self.key_prefix}{key}"
    
    def _serialize(self, value: Any) -> str:
        """Serialize value for storage"""
        try:
            return json.dumps(value)
        except (TypeError, ValueError):
            # Fallback to pickle for complex objects
            import base64
            return base64.b64encode(pickle.dumps(value)).decode('utf-8')
    
    def _deserialize(self, value: str) -> Any:
        """Deserialize value from storage"""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            # Try pickle fallback
            try:
                import base64
                return pickle.loads(base64.b64decode(value.encode('utf-8')))
            except Exception:
                # Return as string if all else fails
                return value
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        try:
            redis = await self._get_redis()
            value = await redis.get(self._make_key(key))
            
            if value is None:
                return None
            
            return self._deserialize(value.decode('utf-8'))
            
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value with optional TTL"""
        try:
            redis = await self._get_redis()
            serialized_value = self._serialize(value)
            
            if ttl:
                result = await redis.setex(self._make_key(key), ttl, serialized_value)
            else:
                result = await redis.set(self._make_key(key), serialized_value)
            
            return result is True
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key"""
        try:
            redis = await self._get_redis()
            result = await redis.delete(self._make_key(key))
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            redis = await self._get_redis()
            result = await redis.exists(self._make_key(key))
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all keys with prefix"""
        try:
            redis = await self._get_redis()
            keys = await redis.keys(f"{self.key_prefix}*")
            
            if keys:
                await redis.delete(*keys)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        try:
            redis = await self._get_redis()
            keys = await redis.keys(f"{self.key_prefix}{pattern}")
            
            # Remove prefix from returned keys
            prefix_len = len(self.key_prefix)
            return [key.decode('utf-8')[prefix_len:] for key in keys]
            
        except Exception as e:
            logger.error(f"Redis keys error: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check backend health"""
        try:
            redis = await self._get_redis()
            
            # Test ping
            pong = await redis.ping()
            
            # Get info
            info = await redis.info()
            
            return {
                'backend': 'redis',
                'status': 'healthy' if pong else 'unhealthy',
                'ping': pong,
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', 'unknown'),
                'uptime': info.get('uptime_in_seconds', 0)
            }
            
        except Exception as e:
            logger.error(f"Redis health check error: {e}")
            return {
                'backend': 'redis',
                'status': 'unhealthy',
                'error': str(e)
            }


class CacheManager:
    """
    Unified cache manager with multiple backend support.
    Provides clean abstraction over different caching strategies.
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.backend = self._create_backend()
    
    def _create_backend(self) -> CacheBackendInterface:
        """Create appropriate cache backend"""
        if self.config.backend == CacheBackend.MEMORY:
            return MemoryCacheBackend(self.config.max_memory_size)
        
        elif self.config.backend == CacheBackend.REDIS:
            if not self.config.redis_url:
                raise ValueError("Redis URL required for Redis backend")
            return RedisCacheBackend(self.config.redis_url, self.config.key_prefix)
        
        elif self.config.backend == CacheBackend.HYBRID:
            # TODO: Implement hybrid backend (L1 memory + L2 Redis)
            raise NotImplementedError("Hybrid backend not yet implemented")
        
        else:
            raise ValueError(f"Unsupported cache backend: {self.config.backend}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        return await self.backend.get(key)
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """Set value with optional TTL"""
        if ttl is None:
            ttl = self.config.default_ttl
        
        return await self.backend.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete key"""
        return await self.backend.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        return await self.backend.exists(key)
    
    async def clear(self) -> bool:
        """Clear all keys"""
        return await self.backend.clear()
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        return await self.backend.keys(pattern)
    
    async def get_or_set(
        self,
        key: str,
        factory_func: callable,
        ttl: Optional[int] = None
    ) -> Any:
        """Get value or set it using factory function"""
        value = await self.get(key)
        
        if value is None:
            value = await factory_func() if asyncio.iscoroutinefunction(factory_func) else factory_func()
            await self.set(key, value, ttl)
        
        return value
    
    async def health_check(self) -> Dict[str, Any]:
        """Check cache health"""
        return await self.backend.health_check()


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        # Default to memory cache if not initialized
        config = CacheConfig(backend=CacheBackend.MEMORY)
        _cache_manager = CacheManager(config)
    return _cache_manager


def initialize_cache_manager(config: CacheConfig) -> CacheManager:
    """Initialize global cache manager"""
    global _cache_manager
    _cache_manager = CacheManager(config)
    return _cache_manager