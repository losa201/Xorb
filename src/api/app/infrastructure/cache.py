"""
Production Caching Infrastructure for XORB Platform
Advanced Redis-based caching with intelligent cache management
"""

import asyncio
import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import zlib
from functools import wraps
import inspect

import redis.asyncio as redis
from redis.asyncio.client import Pipeline

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheStrategy(Enum):
    """Cache strategies for different use cases"""
    WRITE_THROUGH = "write_through"       # Write to cache and database simultaneously
    WRITE_BEHIND = "write_behind"         # Write to cache first, database later
    CACHE_ASIDE = "cache_aside"           # Application manages cache
    READ_THROUGH = "read_through"         # Cache loads from database on miss
    REFRESH_AHEAD = "refresh_ahead"       # Proactively refresh before expiry


class CacheLevel(Enum):
    """Cache levels with different TTL and eviction policies"""
    L1_MEMORY = "l1_memory"              # In-memory, ultra-fast
    L2_REDIS = "l2_redis"                # Redis, fast distributed
    L3_PERSISTENT = "l3_persistent"       # Persistent storage, slower


@dataclass
class CacheConfig:
    """Cache configuration for different data types"""
    default_ttl: int = 3600              # 1 hour
    max_value_size: int = 1024 * 1024    # 1MB
    compression_threshold: int = 1024     # Compress values > 1KB
    enable_compression: bool = True
    enable_serialization: bool = True
    enable_metrics: bool = True
    key_prefix: str = "xorb"
    namespace_separator: str = ":"


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    compressed: bool = False
    serialized: bool = False
    size_bytes: int = 0


class CacheMetrics:
    """Cache performance metrics"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0
        self.total_size = 0
        self.evictions = 0
        
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "total_size": self.total_size,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate()
        }


class IntelligentCacheManager:
    """Intelligent cache manager with adaptive policies"""
    
    def __init__(self, redis_client: redis.Redis, config: CacheConfig = None):
        self.redis_client = redis_client
        self.config = config or CacheConfig()
        self.metrics = CacheMetrics()
        
        # In-memory cache for frequently accessed data
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.memory_cache_max_size = 1000
        
        # Access patterns for cache optimization
        self.access_patterns: Dict[str, List[float]] = {}
        
        # Cache policies
        self.cache_policies = {
            "user_sessions": {
                "ttl": 86400,  # 24 hours
                "strategy": CacheStrategy.WRITE_THROUGH,
                "level": CacheLevel.L2_REDIS,
                "compress": False
            },
            "threat_intelligence": {
                "ttl": 3600,   # 1 hour
                "strategy": CacheStrategy.REFRESH_AHEAD,
                "level": CacheLevel.L2_REDIS,
                "compress": True
            },
            "scan_results": {
                "ttl": 7200,   # 2 hours
                "strategy": CacheStrategy.CACHE_ASIDE,
                "level": CacheLevel.L2_REDIS,
                "compress": True
            },
            "api_responses": {
                "ttl": 300,    # 5 minutes
                "strategy": CacheStrategy.CACHE_ASIDE,
                "level": CacheLevel.L1_MEMORY,
                "compress": False
            },
            "database_queries": {
                "ttl": 1800,   # 30 minutes
                "strategy": CacheStrategy.READ_THROUGH,
                "level": CacheLevel.L2_REDIS,
                "compress": True
            }
        }
    
    async def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get value from cache with intelligent retrieval"""
        full_key = self._build_key(namespace, key)
        
        try:
            # Check L1 cache first (memory)
            if full_key in self.memory_cache:
                entry = self.memory_cache[full_key]
                if not self._is_expired(entry):
                    self._update_access_metrics(entry)
                    self.metrics.hits += 1
                    return entry.value
                else:
                    # Remove expired entry
                    del self.memory_cache[full_key]
            
            # Check L2 cache (Redis)
            cached_data = await self.redis_client.get(full_key)
            if cached_data:
                try:
                    value = self._deserialize(cached_data)
                    
                    # Update access patterns
                    self._record_access(full_key)
                    
                    # Store in L1 cache if frequently accessed
                    if self._should_promote_to_l1(full_key):
                        await self._store_in_memory_cache(full_key, value)
                    
                    self.metrics.hits += 1
                    return value
                    
                except Exception as e:
                    logger.error(f"Cache deserialization error for {full_key}: {e}")
                    await self.redis_client.delete(full_key)
            
            self.metrics.misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for {full_key}: {e}")
            self.metrics.errors += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                 namespace: str = "default", compress: Optional[bool] = None) -> bool:
        """Set value in cache with intelligent storage"""
        full_key = self._build_key(namespace, key)
        
        try:
            # Get cache policy
            policy = self._get_cache_policy(namespace)
            effective_ttl = ttl or policy.get("ttl", self.config.default_ttl)
            should_compress = compress if compress is not None else policy.get("compress", self.config.enable_compression)
            
            # Serialize and optionally compress the value
            serialized_value = self._serialize(value, should_compress)
            
            # Check size limits
            if len(serialized_value) > self.config.max_value_size:
                logger.warning(f"Value too large for cache: {len(serialized_value)} bytes")
                return False
            
            # Store in Redis
            success = await self.redis_client.setex(full_key, effective_ttl, serialized_value)
            
            if success:
                # Also store in memory cache if appropriate
                cache_level = policy.get("level", CacheLevel.L2_REDIS)
                if cache_level == CacheLevel.L1_MEMORY or self._should_store_in_l1(full_key):
                    await self._store_in_memory_cache(full_key, value, effective_ttl)
                
                self.metrics.sets += 1
                self.metrics.total_size += len(serialized_value)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Cache set error for {full_key}: {e}")
            self.metrics.errors += 1
            return False
    
    async def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete value from cache"""
        full_key = self._build_key(namespace, key)
        
        try:
            # Remove from memory cache
            if full_key in self.memory_cache:
                del self.memory_cache[full_key]
            
            # Remove from Redis
            result = await self.redis_client.delete(full_key)
            
            if result:
                self.metrics.deletes += 1
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Cache delete error for {full_key}: {e}")
            self.metrics.errors += 1
            return False
    
    async def exists(self, key: str, namespace: str = "default") -> bool:
        """Check if key exists in cache"""
        full_key = self._build_key(namespace, key)
        
        try:
            # Check memory cache first
            if full_key in self.memory_cache:
                entry = self.memory_cache[full_key]
                if not self._is_expired(entry):
                    return True
                else:
                    del self.memory_cache[full_key]
            
            # Check Redis
            return bool(await self.redis_client.exists(full_key))
            
        except Exception as e:
            logger.error(f"Cache exists error for {full_key}: {e}")
            return False
    
    async def get_multi(self, keys: List[str], namespace: str = "default") -> Dict[str, Any]:
        """Get multiple values from cache efficiently"""
        full_keys = [self._build_key(namespace, key) for key in keys]
        results = {}
        
        try:
            # Check memory cache first
            memory_results = {}
            redis_keys = []
            
            for i, full_key in enumerate(full_keys):
                if full_key in self.memory_cache:
                    entry = self.memory_cache[full_key]
                    if not self._is_expired(entry):
                        memory_results[keys[i]] = entry.value
                        self.metrics.hits += 1
                    else:
                        del self.memory_cache[full_key]
                        redis_keys.append((keys[i], full_key))
                else:
                    redis_keys.append((keys[i], full_key))
            
            # Get remaining from Redis
            if redis_keys:
                redis_values = await self.redis_client.mget([fk for _, fk in redis_keys])
                
                for (original_key, full_key), redis_value in zip(redis_keys, redis_values):
                    if redis_value:
                        try:
                            value = self._deserialize(redis_value)
                            memory_results[original_key] = value
                            self.metrics.hits += 1
                            
                            # Record access for promotion consideration
                            self._record_access(full_key)
                            
                        except Exception as e:
                            logger.error(f"Deserialization error for {full_key}: {e}")
                            self.metrics.errors += 1
                    else:
                        self.metrics.misses += 1
            
            return memory_results
            
        except Exception as e:
            logger.error(f"Cache get_multi error: {e}")
            self.metrics.errors += 1
            return {}
    
    async def set_multi(self, data: Dict[str, Any], ttl: Optional[int] = None, 
                       namespace: str = "default") -> bool:
        """Set multiple values in cache efficiently"""
        try:
            policy = self._get_cache_policy(namespace)
            effective_ttl = ttl or policy.get("ttl", self.config.default_ttl)
            should_compress = policy.get("compress", self.config.enable_compression)
            
            # Prepare data for batch operation
            pipe = self.redis_client.pipeline()
            
            for key, value in data.items():
                full_key = self._build_key(namespace, key)
                serialized_value = self._serialize(value, should_compress)
                
                if len(serialized_value) <= self.config.max_value_size:
                    pipe.setex(full_key, effective_ttl, serialized_value)
                    self.metrics.total_size += len(serialized_value)
            
            # Execute batch operation
            results = await pipe.execute()
            
            success_count = sum(1 for result in results if result)
            self.metrics.sets += success_count
            
            return success_count == len(data)
            
        except Exception as e:
            logger.error(f"Cache set_multi error: {e}")
            self.metrics.errors += 1
            return False
    
    async def invalidate_pattern(self, pattern: str, namespace: str = "default") -> int:
        """Invalidate cache entries matching a pattern"""
        try:
            full_pattern = self._build_key(namespace, pattern)
            
            # Find matching keys
            keys = []
            cursor = 0
            while True:
                cursor, batch_keys = await self.redis_client.scan(
                    cursor=cursor, 
                    match=full_pattern, 
                    count=100
                )
                keys.extend(batch_keys)
                if cursor == 0:
                    break
            
            if keys:
                # Remove from memory cache
                for key in keys:
                    if key in self.memory_cache:
                        del self.memory_cache[key]
                
                # Remove from Redis
                deleted_count = await self.redis_client.delete(*keys)
                self.metrics.deletes += deleted_count
                self.metrics.evictions += deleted_count
                
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.error(f"Cache pattern invalidation error: {e}")
            self.metrics.errors += 1
            return 0
    
    async def cache_warmup(self, warmup_data: Dict[str, Any], namespace: str = "default"):
        """Warm up cache with frequently accessed data"""
        try:
            logger.info(f"Starting cache warmup for namespace '{namespace}' with {len(warmup_data)} items")
            
            # Set data in batches to avoid overwhelming Redis
            batch_size = 100
            items = list(warmup_data.items())
            
            for i in range(0, len(items), batch_size):
                batch = dict(items[i:i + batch_size])
                await self.set_multi(batch, namespace=namespace)
                
                # Small delay between batches
                await asyncio.sleep(0.1)
            
            logger.info(f"Cache warmup completed for namespace '{namespace}'")
            
        except Exception as e:
            logger.error(f"Cache warmup error: {e}")
    
    def _build_key(self, namespace: str, key: str) -> str:
        """Build cache key with namespace and prefix"""
        return f"{self.config.key_prefix}{self.config.namespace_separator}{namespace}{self.config.namespace_separator}{key}"
    
    def _serialize(self, value: Any, compress: bool = False) -> bytes:
        """Serialize and optionally compress value"""
        if self.config.enable_serialization:
            # Use pickle for Python objects
            serialized = pickle.dumps(value)
        else:
            # Use JSON for simple types
            serialized = json.dumps(value).encode('utf-8')
        
        if compress and self.config.enable_compression and len(serialized) > self.config.compression_threshold:
            serialized = zlib.compress(serialized)
        
        return serialized
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize and decompress value"""
        try:
            # Try decompression first
            try:
                decompressed = zlib.decompress(data)
                data = decompressed
            except zlib.error:
                # Not compressed
                pass
            
            # Try pickle first
            try:
                return pickle.loads(data)
            except (pickle.PickleError, EOFError):
                # Fall back to JSON
                return json.loads(data.decode('utf-8'))
                
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise
    
    def _get_cache_policy(self, namespace: str) -> Dict[str, Any]:
        """Get cache policy for namespace"""
        return self.cache_policies.get(namespace, {
            "ttl": self.config.default_ttl,
            "strategy": CacheStrategy.CACHE_ASIDE,
            "level": CacheLevel.L2_REDIS,
            "compress": self.config.enable_compression
        })
    
    def _record_access(self, key: str):
        """Record access pattern for intelligent caching"""
        current_time = time.time()
        
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(current_time)
        
        # Keep only recent accesses (last hour)
        cutoff_time = current_time - 3600
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t > cutoff_time
        ]
    
    def _should_promote_to_l1(self, key: str) -> bool:
        """Determine if key should be promoted to L1 cache"""
        if key not in self.access_patterns:
            return False
        
        # Promote if accessed more than 5 times in last hour
        return len(self.access_patterns[key]) > 5
    
    def _should_store_in_l1(self, key: str) -> bool:
        """Determine if key should be stored in L1 cache initially"""
        # Store in L1 if memory cache has space and it's a frequent pattern
        return (len(self.memory_cache) < self.memory_cache_max_size and 
                self._should_promote_to_l1(key))
    
    async def _store_in_memory_cache(self, key: str, value: Any, ttl: int = None):
        """Store value in memory cache"""
        try:
            # Evict old entries if at capacity
            if len(self.memory_cache) >= self.memory_cache_max_size:
                self._evict_lru_memory_cache()
            
            expires_at = datetime.utcnow() + timedelta(seconds=ttl) if ttl else None
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                access_count=1,
                last_accessed=datetime.utcnow(),
                size_bytes=len(str(value))  # Rough estimate
            )
            
            self.memory_cache[key] = entry
            
        except Exception as e:
            logger.error(f"Memory cache store error: {e}")
    
    def _evict_lru_memory_cache(self):
        """Evict least recently used items from memory cache"""
        if not self.memory_cache:
            return
        
        # Sort by last accessed time
        sorted_entries = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1].last_accessed or x[1].created_at
        )
        
        # Remove oldest 10% of entries
        evict_count = max(1, len(sorted_entries) // 10)
        
        for i in range(evict_count):
            key, _ = sorted_entries[i]
            del self.memory_cache[key]
            self.metrics.evictions += 1
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        if entry.expires_at is None:
            return False
        return datetime.utcnow() > entry.expires_at
    
    def _update_access_metrics(self, entry: CacheEntry):
        """Update access metrics for cache entry"""
        entry.access_count += 1
        entry.last_accessed = datetime.utcnow()
    
    async def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information"""
        try:
            # Redis info
            redis_info = await self.redis_client.info()
            redis_memory = redis_info.get('used_memory_human', 'unknown')
            redis_hits = redis_info.get('keyspace_hits', 0)
            redis_misses = redis_info.get('keyspace_misses', 0)
            
            # Memory cache info
            memory_cache_size = len(self.memory_cache)
            memory_cache_bytes = sum(entry.size_bytes for entry in self.memory_cache.values())
            
            return {
                "metrics": self.metrics.to_dict(),
                "redis": {
                    "memory_usage": redis_memory,
                    "hits": redis_hits,
                    "misses": redis_misses,
                    "hit_rate": redis_hits / (redis_hits + redis_misses) if (redis_hits + redis_misses) > 0 else 0.0
                },
                "memory_cache": {
                    "entries": memory_cache_size,
                    "size_bytes": memory_cache_bytes,
                    "max_entries": self.memory_cache_max_size
                },
                "access_patterns": {
                    "tracked_keys": len(self.access_patterns),
                    "total_accesses": sum(len(accesses) for accesses in self.access_patterns.values())
                },
                "config": asdict(self.config)
            }
            
        except Exception as e:
            logger.error(f"Cache info error: {e}")
            return {"error": str(e)}


# Cache decorators for automatic caching
def cached(ttl: int = 3600, namespace: str = "default", key_func: Optional[Callable] = None):
    """Decorator for automatic function result caching"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache manager from global instance
            cache_manager = await get_cache_manager()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _generate_cache_key(func, args, kwargs)
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key, namespace)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, ttl, namespace)
            
            return result
        
        return wrapper
    return decorator


def cache_invalidate(pattern: str, namespace: str = "default"):
    """Decorator to invalidate cache on function execution"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Invalidate cache pattern
            cache_manager = await get_cache_manager()
            await cache_manager.invalidate_pattern(pattern, namespace)
            
            return result
        
        return wrapper
    return decorator


def _generate_cache_key(func: Callable, args: tuple, kwargs: dict) -> str:
    """Generate cache key from function signature"""
    # Create a deterministic key from function name and arguments
    func_name = f"{func.__module__}.{func.__name__}"
    
    # Include arguments in key
    key_parts = [func_name]
    
    if args:
        key_parts.append(str(hash(args)))
    
    if kwargs:
        sorted_kwargs = sorted(kwargs.items())
        key_parts.append(str(hash(tuple(sorted_kwargs))))
    
    return ":".join(key_parts)


# Global cache manager instance
_cache_manager: Optional[IntelligentCacheManager] = None


async def init_cache(redis_url: str = "redis://localhost:6379"):
    """Initialize global cache manager"""
    global _cache_manager
    
    try:
        # Create Redis client
        redis_client = await redis.from_url(redis_url)
        
        # Test connection
        await redis_client.ping()
        
        # Create cache manager
        config = CacheConfig()
        _cache_manager = IntelligentCacheManager(redis_client, config)
        
        logger.info("Cache infrastructure initialized successfully")
        
    except Exception as e:
        logger.error(f"Cache initialization failed: {e}")
        raise


async def get_cache_manager() -> IntelligentCacheManager:
    """Get global cache manager instance"""
    if _cache_manager is None:
        await init_cache()
    
    return _cache_manager


async def close_cache():
    """Close cache connections"""
    global _cache_manager
    
    if _cache_manager and _cache_manager.redis_client:
        await _cache_manager.redis_client.close()
        _cache_manager = None
        logger.info("Cache connections closed")


# Convenience functions
async def cache_get(key: str, namespace: str = "default") -> Optional[Any]:
    """Convenience function for cache get"""
    cache_manager = await get_cache_manager()
    return await cache_manager.get(key, namespace)


async def cache_set(key: str, value: Any, ttl: int = 3600, namespace: str = "default") -> bool:
    """Convenience function for cache set"""
    cache_manager = await get_cache_manager()
    return await cache_manager.set(key, value, ttl, namespace)


async def cache_delete(key: str, namespace: str = "default") -> bool:
    """Convenience function for cache delete"""
    cache_manager = await get_cache_manager()
    return await cache_manager.delete(key, namespace)