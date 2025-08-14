"""
Advanced caching system with multiple backends and intelligent strategies
"""

import json
import pickle
import hashlib
import time
import asyncio
from typing import Any, Optional, Dict, List, Union, Callable, TypeVar, Generic
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import weakref
from abc import ABC, abstractmethod

import redis.asyncio as redis
from redis.asyncio.client import Redis
import structlog

from .logging import get_logger
from .metrics import get_metrics_service

logger = get_logger(__name__)
T = TypeVar('T')


class CacheBackend(Enum):
    """Available cache backends"""
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"


class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live only


@dataclass
class CacheConfig:
    """Cache configuration settings"""
    backend: CacheBackend = CacheBackend.HYBRID
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    redis_max_connections: int = 20
    redis_socket_timeout: int = 5
    redis_socket_connect_timeout: int = 5
    redis_retry_on_timeout: bool = True
    
    # Memory cache settings
    memory_max_size: int = 1000
    memory_ttl_seconds: int = 3600
    memory_strategy: CacheStrategy = CacheStrategy.LRU
    
    # General settings
    default_ttl_seconds: int = 3600
    compression_threshold: int = 1024
    enable_metrics: bool = True
    key_prefix: str = "xorb:cache:"
    
    # Circuit breaker settings
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60


@dataclass
class CacheItem(Generic[T]):
    """Cache item with metadata"""
    value: T
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if item is expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update access metadata"""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheBackendInterface(ABC):
    """Abstract interface for cache backends"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass


class MemoryCacheBackend(CacheBackendInterface):
    """In-memory cache backend with LRU/LFU eviction"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: Dict[str, CacheItem] = {}
        self.lock = asyncio.Lock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0
        }
    
    async def get(self, key: str) -> Optional[Any]:
        async with self.lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None
            
            item = self.cache[key]
            
            # Check expiration
            if item.is_expired():
                del self.cache[key]
                self.stats["misses"] += 1
                return None
            
            # Update access metadata
            item.touch()
            self.stats["hits"] += 1
            
            return item.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        async with self.lock:
            try:
                # Check if we need to evict items
                if len(self.cache) >= self.config.memory_max_size and key not in self.cache:
                    await self._evict_items()
                
                # Create cache item
                cache_item = CacheItem(
                    value=value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    access_count=1,
                    ttl=ttl or self.config.memory_ttl_seconds
                )
                
                self.cache[key] = cache_item
                self.stats["sets"] += 1
                return True
                
            except Exception as e:
                logger.error("Memory cache set failed", key=key, error=str(e))
                return False
    
    async def delete(self, key: str) -> bool:
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats["deletes"] += 1
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        async with self.lock:
            if key not in self.cache:
                return False
            
            item = self.cache[key]
            if item.is_expired():
                del self.cache[key]
                return False
            
            return True
    
    async def clear(self) -> bool:
        async with self.lock:
            self.cache.clear()
            return True
    
    async def get_stats(self) -> Dict[str, Any]:
        async with self.lock:
            return {
                **self.stats,
                "size": len(self.cache),
                "max_size": self.config.memory_max_size,
                "hit_ratio": self.stats["hits"] / max(1, self.stats["hits"] + self.stats["misses"])
            }
    
    async def _evict_items(self):
        """Evict items based on strategy"""
        if not self.cache:
            return
        
        evict_count = max(1, len(self.cache) // 10)  # Evict 10%
        
        if self.config.memory_strategy == CacheStrategy.LRU:
            # Sort by last_accessed
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1].last_accessed
            )
        elif self.config.memory_strategy == CacheStrategy.LFU:
            # Sort by access_count
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1].access_count
            )
        elif self.config.memory_strategy == CacheStrategy.FIFO:
            # Sort by created_at
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1].created_at
            )
        else:  # TTL
            # Sort by expiration time
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1].created_at + (x[1].ttl or 0)
            )
        
        # Remove items
        for i in range(evict_count):
            if i < len(sorted_items):
                key = sorted_items[i][0]
                del self.cache[key]
                self.stats["evictions"] += 1


class RedisCacheBackend(CacheBackendInterface):
    """Redis cache backend with connection pooling"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.redis_client: Optional[Redis] = None
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = 0
        self.circuit_breaker_open = False
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
    
    async def _get_client(self) -> Optional[Redis]:
        """Get Redis client with circuit breaker"""
        if self.config.circuit_breaker_enabled and self.circuit_breaker_open:
            # Check if we should try to recover
            if time.time() - self.circuit_breaker_last_failure > self.config.circuit_breaker_recovery_timeout:
                self.circuit_breaker_open = False
                self.circuit_breaker_failures = 0
                logger.info("Redis circuit breaker reset")
            else:
                return None
        
        if not self.redis_client:
            try:
                self.redis_client = redis.from_url(
                    self.config.redis_url,
                    db=self.config.redis_db,
                    max_connections=self.config.redis_max_connections,
                    socket_timeout=self.config.redis_socket_timeout,
                    socket_connect_timeout=self.config.redis_socket_connect_timeout,
                    retry_on_timeout=self.config.redis_retry_on_timeout,
                    decode_responses=False
                )
                # Test connection
                await self.redis_client.ping()
                logger.info("Redis connection established")
                
            except Exception as e:
                logger.error("Redis connection failed", error=str(e))
                await self._handle_redis_error()
                return None
        
        return self.redis_client
    
    async def _handle_redis_error(self):
        """Handle Redis connection errors with circuit breaker"""
        if self.config.circuit_breaker_enabled:
            self.circuit_breaker_failures += 1
            self.circuit_breaker_last_failure = time.time()
            
            if self.circuit_breaker_failures >= self.config.circuit_breaker_failure_threshold:
                self.circuit_breaker_open = True
                logger.warning("Redis circuit breaker opened", failures=self.circuit_breaker_failures)
        
        self.stats["errors"] += 1
        
        # Close existing connection
        if self.redis_client:
            try:
                await self.redis_client.close()
            except:
                pass
            self.redis_client = None
    
    def _make_key(self, key: str) -> str:
        """Create full cache key with prefix"""
        return f"{self.config.key_prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        client = await self._get_client()
        if not client:
            self.stats["misses"] += 1
            return None
        
        try:
            full_key = self._make_key(key)
            data = await client.get(full_key)
            
            if data is None:
                self.stats["misses"] += 1
                return None
            
            # Deserialize data
            value = pickle.loads(data)
            self.stats["hits"] += 1
            return value
            
        except Exception as e:
            logger.error("Redis get failed", key=key, error=str(e))
            await self._handle_redis_error()
            self.stats["misses"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        client = await self._get_client()
        if not client:
            return False
        
        try:
            full_key = self._make_key(key)
            
            # Serialize data
            data = pickle.dumps(value)
            
            # Set with TTL
            ttl_value = ttl or self.config.default_ttl_seconds
            await client.setex(full_key, ttl_value, data)
            
            self.stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.error("Redis set failed", key=key, error=str(e))
            await self._handle_redis_error()
            return False
    
    async def delete(self, key: str) -> bool:
        client = await self._get_client()
        if not client:
            return False
        
        try:
            full_key = self._make_key(key)
            result = await client.delete(full_key)
            
            self.stats["deletes"] += 1
            return result > 0
            
        except Exception as e:
            logger.error("Redis delete failed", key=key, error=str(e))
            await self._handle_redis_error()
            return False
    
    async def exists(self, key: str) -> bool:
        client = await self._get_client()
        if not client:
            return False
        
        try:
            full_key = self._make_key(key)
            result = await client.exists(full_key)
            return result > 0
            
        except Exception as e:
            logger.error("Redis exists failed", key=key, error=str(e))
            await self._handle_redis_error()
            return False
    
    async def clear(self) -> bool:
        client = await self._get_client()
        if not client:
            return False
        
        try:
            # Delete all keys with our prefix
            pattern = f"{self.config.key_prefix}*"
            keys = await client.keys(pattern)
            
            if keys:
                await client.delete(*keys)
            
            return True
            
        except Exception as e:
            logger.error("Redis clear failed", error=str(e))
            await self._handle_redis_error()
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "circuit_breaker_open": self.circuit_breaker_open,
            "circuit_breaker_failures": self.circuit_breaker_failures,
            "hit_ratio": self.stats["hits"] / max(1, self.stats["hits"] + self.stats["misses"])
        }
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()


class HybridCacheBackend(CacheBackendInterface):
    """Hybrid cache using memory as L1 and Redis as L2"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.memory_cache = MemoryCacheBackend(config)
        self.redis_cache = RedisCacheBackend(config)
    
    async def get(self, key: str) -> Optional[Any]:
        # Try L1 cache first
        value = await self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try L2 cache
        value = await self.redis_cache.get(key)
        if value is not None:
            # Populate L1 cache
            await self.memory_cache.set(key, value)
            return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        # Set in both caches
        l1_success = await self.memory_cache.set(key, value, ttl)
        l2_success = await self.redis_cache.set(key, value, ttl)
        
        # Return success if at least one succeeded
        return l1_success or l2_success
    
    async def delete(self, key: str) -> bool:
        # Delete from both caches
        l1_success = await self.memory_cache.delete(key)
        l2_success = await self.redis_cache.delete(key)
        
        return l1_success or l2_success
    
    async def exists(self, key: str) -> bool:
        # Check L1 first, then L2
        return await self.memory_cache.exists(key) or await self.redis_cache.exists(key)
    
    async def clear(self) -> bool:
        # Clear both caches
        l1_success = await self.memory_cache.clear()
        l2_success = await self.redis_cache.clear()
        
        return l1_success and l2_success
    
    async def get_stats(self) -> Dict[str, Any]:
        memory_stats = await self.memory_cache.get_stats()
        redis_stats = await self.redis_cache.get_stats()
        
        return {
            "memory": memory_stats,
            "redis": redis_stats,
            "combined_hit_ratio": (
                (memory_stats["hits"] + redis_stats["hits"]) /
                max(1, memory_stats["hits"] + memory_stats["misses"] + 
                    redis_stats["hits"] + redis_stats["misses"])
            )
        }
    
    async def close(self):
        """Close connections"""
        await self.redis_cache.close()


class CacheService:
    """Main cache service with intelligent caching strategies"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        
        # Choose backend
        if config.backend == CacheBackend.MEMORY:
            self.backend = MemoryCacheBackend(config)
        elif config.backend == CacheBackend.REDIS:
            self.backend = RedisCacheBackend(config)
        else:  # HYBRID
            self.backend = HybridCacheBackend(config)
        
        self.metrics_service = get_metrics_service() if config.enable_metrics else None
        backend_name = getattr(config.backend, 'value', config.backend)
        logger.info("Cache service initialized", backend=backend_name)
    
    def _make_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        # Create deterministic key from arguments
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items()) if kwargs else {}
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        result = await self.backend.get(key)
        
        # Record metrics
        if self.metrics_service:
            if result is not None:
                self.metrics_service.custom_metrics.increment_counter("cache_hits", 1, {"backend": self.config.backend.value})
            else:
                self.metrics_service.custom_metrics.increment_counter("cache_misses", 1, {"backend": self.config.backend.value})
        
        return result
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        success = await self.backend.set(key, value, ttl)
        
        # Record metrics
        if self.metrics_service:
            if success:
                self.metrics_service.custom_metrics.increment_counter("cache_sets", 1, {"backend": self.config.backend.value})
            else:
                self.metrics_service.custom_metrics.increment_counter("cache_set_errors", 1, {"backend": self.config.backend.value})
        
        return success
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        return await self.backend.delete(key)
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        return await self.backend.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return await self.backend.get_stats()
    
    async def close(self):
        """Close cache connections"""
        if hasattr(self.backend, 'close'):
            await self.backend.close()
    
    # Decorator for caching function results
    def cached(
        self,
        ttl: Optional[int] = None,
        key_prefix: str = "",
        skip_cache: Callable[..., bool] = None
    ):
        """Decorator to cache function results"""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                # Check if we should skip cache
                if skip_cache and skip_cache(*args, **kwargs):
                    return await func(*args, **kwargs)
                
                # Generate cache key
                cache_key = f"{key_prefix}{func.__name__}:{self._make_cache_key(*args, **kwargs)}"
                
                # Try to get from cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    logger.debug("Cache hit", function=func.__name__, key=cache_key)
                    return cached_result
                
                # Execute function
                logger.debug("Cache miss", function=func.__name__, key=cache_key)
                result = await func(*args, **kwargs)
                
                # Cache the result
                await self.set(cache_key, result, ttl)
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                # For sync functions, we can't use async cache
                return func(*args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator


# Global cache service instance
_cache_service: Optional[CacheService] = None


def get_cache_service() -> Optional[CacheService]:
    """Get global cache service instance"""
    return _cache_service


def setup_cache(config: CacheConfig) -> CacheService:
    """Setup global cache service"""
    global _cache_service
    _cache_service = CacheService(config)
    return _cache_service


# Convenience decorator using global cache service
def cached(
    ttl: Optional[int] = None,
    key_prefix: str = "",
    skip_cache: Callable[..., bool] = None
):
    """Convenience decorator using global cache service"""
    cache_service = get_cache_service()
    if not cache_service:
        # If no cache service, return function unchanged
        def no_cache_decorator(func):
            return func
        return no_cache_decorator
    
    return cache_service.cached(ttl, key_prefix, skip_cache)