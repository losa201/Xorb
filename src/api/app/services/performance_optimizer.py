"""
Performance Optimization and Caching Service
Advanced caching, connection pooling, and performance monitoring
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from functools import wraps
import hashlib
import json
import gc
from collections import defaultdict, deque
import weakref

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: int
    size_bytes: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.utcnow() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    @property
    def age_seconds(self) -> int:
        """Get age in seconds"""
        return int((datetime.utcnow() - self.created_at).total_seconds())

@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics"""
    request_count: int = 0
    average_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    active_connections: int = 0
    error_rate: float = 0.0
    requests_per_second: float = 0.0
    
class AdvancedCache:
    """High-performance caching system with LRU eviction and statistics"""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = deque()  # For LRU tracking
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "total_size_bytes": 0
        }
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            self.stats["misses"] += 1
            return None
        
        entry = self.cache[key]
        
        # Check expiration
        if entry.is_expired:
            await self.delete(key)
            self.stats["misses"] += 1
            return None
        
        # Update access metadata
        entry.last_accessed = datetime.utcnow()
        entry.access_count += 1
        
        # Move to end of access order (most recently used)
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        self.stats["hits"] += 1
        return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        
        # Calculate size
        try:
            size_bytes = len(json.dumps(value, default=str).encode('utf-8'))
        except:
            size_bytes = 1024  # Default estimate
        
        # Check if we need to evict entries
        if len(self.cache) >= self.max_size:
            await self._evict_lru()
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            access_count=1,
            ttl_seconds=ttl,
            size_bytes=size_bytes
        )
        
        # Remove old entry if exists
        if key in self.cache:
            old_entry = self.cache[key]
            self.stats["total_size_bytes"] -= old_entry.size_bytes
        
        # Add new entry
        self.cache[key] = entry
        self.access_order.append(key)
        self.stats["sets"] += 1
        self.stats["total_size_bytes"] += size_bytes
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        if key not in self.cache:
            return False
        
        entry = self.cache[key]
        self.stats["total_size_bytes"] -= entry.size_bytes
        
        del self.cache[key]
        if key in self.access_order:
            self.access_order.remove(key)
        
        return True
    
    async def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.access_order:
            return
        
        lru_key = self.access_order.popleft()
        if lru_key in self.cache:
            entry = self.cache[lru_key]
            self.stats["total_size_bytes"] -= entry.size_bytes
            del self.cache[lru_key]
            self.stats["evictions"] += 1
    
    async def cleanup_expired(self):
        """Remove expired entries"""
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired
        ]
        
        for key in expired_keys:
            await self.delete(key)
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": round(hit_rate, 3),
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "sets": self.stats["sets"],
            "evictions": self.stats["evictions"],
            "total_size_mb": round(self.stats["total_size_bytes"] / 1024 / 1024, 2),
            "avg_entry_size_kb": round(
                self.stats["total_size_bytes"] / len(self.cache) / 1024, 2
            ) if self.cache else 0
        }

class ConnectionPool:
    """Database connection pool with health monitoring"""
    
    def __init__(self, max_connections: int = 20, min_connections: int = 5):
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.pool = deque()
        self.active_connections = set()
        self.stats = {
            "created": 0,
            "destroyed": 0,
            "borrowed": 0,
            "returned": 0
        }
    
    async def get_connection(self):
        """Get connection from pool"""
        if self.pool:
            conn = self.pool.popleft()
            self.active_connections.add(conn)
            self.stats["borrowed"] += 1
            return conn
        
        if len(self.active_connections) < self.max_connections:
            # Create new connection (mock)
            conn = f"conn_{self.stats['created']}"
            self.active_connections.add(conn)
            self.stats["created"] += 1
            self.stats["borrowed"] += 1
            return conn
        
        # Wait for connection to become available
        await asyncio.sleep(0.1)
        return await self.get_connection()
    
    async def return_connection(self, conn):
        """Return connection to pool"""
        if conn in self.active_connections:
            self.active_connections.remove(conn)
            
            if len(self.pool) < self.min_connections:
                self.pool.append(conn)
                self.stats["returned"] += 1
            else:
                # Destroy excess connection
                self.stats["destroyed"] += 1
    
    def get_stats(self) -> Dict[str, int]:
        """Get connection pool statistics"""
        return {
            "pool_size": len(self.pool),
            "active_connections": len(self.active_connections),
            "max_connections": self.max_connections,
            **self.stats
        }

class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self, window_size: int = 300):  # 5 minutes
        self.window_size = window_size
        self.request_times = deque(maxlen=1000)
        self.error_count = deque(maxlen=1000)
        self.response_times = defaultdict(list)
        self.endpoint_stats = defaultdict(lambda: {"count": 0, "total_time": 0, "errors": 0})
        
    def record_request(self, endpoint: str, response_time: float, status_code: int):
        """Record request metrics"""
        timestamp = time.time()
        
        self.request_times.append((timestamp, response_time))
        
        if status_code >= 400:
            self.error_count.append(timestamp)
        
        # Update endpoint stats
        stats = self.endpoint_stats[endpoint]
        stats["count"] += 1
        stats["total_time"] += response_time
        if status_code >= 400:
            stats["errors"] += 1
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        now = time.time()
        cutoff = now - self.window_size
        
        # Filter recent requests
        recent_requests = [(t, rt) for t, rt in self.request_times if t >= cutoff]
        recent_errors = [t for t in self.error_count if t >= cutoff]
        
        if not recent_requests:
            return PerformanceMetrics()
        
        # Calculate metrics
        request_count = len(recent_requests)
        avg_response_time = sum(rt for _, rt in recent_requests) / request_count
        error_rate = len(recent_errors) / request_count if request_count > 0 else 0
        rps = request_count / self.window_size
        
        return PerformanceMetrics(
            request_count=request_count,
            average_response_time=round(avg_response_time, 3),
            error_rate=round(error_rate, 3),
            requests_per_second=round(rps, 2),
            memory_usage_mb=round(self._get_memory_usage(), 2)
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback using gc
            return len(gc.get_objects()) * 0.001  # Rough estimate
    
    def get_endpoint_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get per-endpoint statistics"""
        stats = {}
        
        for endpoint, data in self.endpoint_stats.items():
            if data["count"] > 0:
                stats[endpoint] = {
                    "request_count": data["count"],
                    "average_response_time": round(data["total_time"] / data["count"], 3),
                    "error_count": data["errors"],
                    "error_rate": round(data["errors"] / data["count"], 3)
                }
        
        return stats

class PerformanceOptimizer:
    """Main performance optimization service"""
    
    def __init__(self):
        self.cache = AdvancedCache()
        self.connection_pool = ConnectionPool()
        self.monitor = PerformanceMonitor()
        self.optimization_rules = {}
        self._background_tasks_started = False
        
    async def start_background_tasks(self):
        """Start background tasks (called when event loop is available)"""
        if not self._background_tasks_started:
            try:
                asyncio.create_task(self._cleanup_task())
                asyncio.create_task(self._optimization_task())
                self._background_tasks_started = True
                logger.info("Performance optimizer background tasks started")
            except RuntimeError:
                # No event loop available yet - tasks will be started later
                pass
    
    async def _cleanup_task(self):
        """Background cleanup task"""
        while True:
            try:
                # Cleanup expired cache entries
                expired_count = await self.cache.cleanup_expired()
                if expired_count > 0:
                    logger.debug(f"Cleaned up {expired_count} expired cache entries")
                
                # Garbage collection
                collected = gc.collect()
                if collected > 0:
                    logger.debug(f"Garbage collected {collected} objects")
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)
    
    async def _optimization_task(self):
        """Background optimization task"""
        while True:
            try:
                await self._optimize_performance()
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                logger.error(f"Error in optimization task: {e}")
                await asyncio.sleep(60)
    
    async def _optimize_performance(self):
        """Perform automatic performance optimizations"""
        
        metrics = self.monitor.get_metrics()
        cache_stats = self.cache.get_stats()
        
        # Auto-adjust cache size based on hit rate
        if cache_stats["hit_rate"] < 0.7 and cache_stats["cache_size"] < 50000:
            self.cache.max_size = min(self.cache.max_size * 1.2, 50000)
            logger.info(f"Increased cache size to {self.cache.max_size}")
        
        # Warn about performance issues
        if metrics.average_response_time > 5.0:
            logger.warning(f"High average response time: {metrics.average_response_time}s")
        
        if metrics.error_rate > 0.05:
            logger.warning(f"High error rate: {metrics.error_rate:.1%}")
        
        if metrics.memory_usage_mb > 1000:
            logger.warning(f"High memory usage: {metrics.memory_usage_mb}MB")
    
    def cached(self, ttl: int = 3600, key_prefix: str = ""):
        """Caching decorator"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                key_data = {
                    "func": func.__name__,
                    "args": str(args),
                    "kwargs": str(sorted(kwargs.items()))
                }
                cache_key = f"{key_prefix}:{hashlib.md5(str(key_data).encode()).hexdigest()}"
                
                # Try to get from cache
                cached_result = await self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                await self.cache.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def monitored(self, endpoint_name: str = None):
        """Performance monitoring decorator"""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                endpoint = endpoint_name or func.__name__
                start_time = time.time()
                status_code = 200
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    status_code = 500
                    raise
                finally:
                    response_time = time.time() - start_time
                    self.monitor.record_request(endpoint, response_time, status_code)
            
            return wrapper
        return decorator
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        return {
            "performance_metrics": self.monitor.get_metrics().__dict__,
            "cache_stats": self.cache.get_stats(),
            "connection_pool_stats": self.connection_pool.get_stats(),
            "endpoint_stats": self.monitor.get_endpoint_stats(),
            "system_info": {
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": time.time() - getattr(self, '_start_time', time.time()),
                "optimization_active": True
            }
        }

# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()
performance_optimizer._start_time = time.time()

async def get_performance_optimizer() -> PerformanceOptimizer:
    """Get performance optimizer instance"""
    return performance_optimizer

# Convenience decorators for easy use
def cached(ttl: int = 3600, key_prefix: str = ""):
    """Caching decorator"""
    return performance_optimizer.cached(ttl, key_prefix)

def monitored(endpoint_name: str = None):
    """Performance monitoring decorator"""
    return performance_optimizer.monitored(endpoint_name)