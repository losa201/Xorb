"""Performance optimizations and monitoring."""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import sys
import os

# Import performance libraries with fallbacks
try:
    import uvloop
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False
    import json

from prometheus_client import Counter, Histogram, Gauge, Info
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


logger = logging.getLogger(__name__)


# Prometheus metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

active_connections = Gauge(
    'active_connections',
    'Active database connections'
)

database_query_duration = Histogram(
    'database_query_duration_seconds',
    'Database query duration',
    ['query_type']
)

memory_usage = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes'
)

app_info = Info(
    'app_info',
    'Application information'
)


class PerformanceConfig:
    """Performance configuration settings."""
    
    def __init__(self):
        self.enable_uvloop = os.getenv("ENABLE_UVLOOP", "true").lower() == "true"
        self.enable_orjson = os.getenv("ENABLE_ORJSON", "true").lower() == "true"
        self.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        self.enable_request_logging = os.getenv("ENABLE_REQUEST_LOGGING", "false").lower() == "true"
        
        # HTTP optimizations
        self.http_keepalive = int(os.getenv("HTTP_KEEPALIVE", "60"))
        self.http_max_requests = int(os.getenv("HTTP_MAX_REQUESTS", "1000"))
        
        # AsyncIO optimizations
        self.asyncio_debug = os.getenv("ASYNCIO_DEBUG", "false").lower() == "true"


def setup_uvloop() -> None:
    """Setup uvloop for better async performance."""
    if UVLOOP_AVAILABLE and PerformanceConfig().enable_uvloop:
        try:
            uvloop.install()
            logger.info("uvloop installed for improved async performance")
        except Exception as e:
            logger.warning(f"Failed to install uvloop: {e}")
    else:
        logger.info("uvloop not available or disabled, using default asyncio")


def setup_json_encoder():
    """Setup fast JSON encoder."""
    if ORJSON_AVAILABLE and PerformanceConfig().enable_orjson:
        logger.info("Using orjson for improved JSON performance")
        return orjson
    else:
        logger.info("Using standard json library")
        return json


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and optimization."""
    
    def __init__(self, app, enable_metrics: bool = True):
        super().__init__(app)
        self.enable_metrics = enable_metrics
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with performance monitoring."""
        start_time = time.time()
        
        # Extract endpoint info
        method = request.method
        endpoint = self._get_endpoint_name(request)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Record metrics
            if self.enable_metrics:
                duration = time.time() - start_time
                
                request_count.labels(
                    method=method,
                    endpoint=endpoint,
                    status_code=response.status_code
                ).inc()
                
                request_duration.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
            
            # Add performance headers
            response.headers["X-Process-Time"] = str(time.time() - start_time)
            
            return response
        
        except Exception as e:
            # Record error metrics
            if self.enable_metrics:
                request_count.labels(
                    method=method,
                    endpoint=endpoint,
                    status_code=500
                ).inc()
            
            raise
    
    def _get_endpoint_name(self, request: Request) -> str:
        """Extract endpoint name from request."""
        if hasattr(request, 'url') and request.url.path:
            # Normalize path patterns
            path = request.url.path
            
            # Replace UUIDs with placeholder
            import re
            path = re.sub(
                r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
                '/{uuid}',
                path
            )
            
            # Replace other dynamic segments
            path = re.sub(r'/\d+', '/{id}', path)
            
            return path
        
        return "unknown"


class DatabaseMetrics:
    """Database performance metrics collector."""
    
    @staticmethod
    async def record_query_time(query_type: str, duration: float):
        """Record database query execution time."""
        database_query_duration.labels(query_type=query_type).observe(duration)
    
    @staticmethod
    async def update_connection_count(count: int):
        """Update active database connection count."""
        active_connections.set(count)


class MemoryMonitor:
    """Memory usage monitoring."""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, Any]:
        """Get current memory usage statistics."""
        import psutil
        import gc
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss,  # Resident Set Size
            "vms": memory_info.vms,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "available": psutil.virtual_memory().available,
            "gc_objects": len(gc.get_objects())
        }
    
    @staticmethod
    async def update_memory_metrics():
        """Update memory usage metrics."""
        try:
            stats = MemoryMonitor.get_memory_usage()
            memory_usage.set(stats["rss"])
        except Exception as e:
            logger.warning(f"Failed to update memory metrics: {e}")


class CacheManager:
    """Application-level caching for performance."""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        if key in self._cache:
            # Check TTL
            if key in self._cache_ttl:
                if time.time() > self._cache_ttl[key]:
                    self.delete(key)
                    return None
            
            return self._cache[key]
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set cached value with optional TTL."""
        self._cache[key] = value
        
        if ttl:
            self._cache_ttl[key] = time.time() + ttl
    
    def delete(self, key: str):
        """Delete cached value."""
        self._cache.pop(key, None)
        self._cache_ttl.pop(key, None)
    
    def clear(self):
        """Clear all cached values."""
        self._cache.clear()
        self._cache_ttl.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self._cache)


# Global instances
_cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    return _cache_manager


@asynccontextmanager
async def performance_monitor():
    """Performance monitoring context manager."""
    
    # Setup performance optimizations
    setup_uvloop()
    json_encoder = setup_json_encoder()
    
    # Set application info
    app_info.info({
        'version': '1.0.0',
        'uvloop_enabled': str(UVLOOP_AVAILABLE and PerformanceConfig().enable_uvloop),
        'orjson_enabled': str(ORJSON_AVAILABLE and PerformanceConfig().enable_orjson),
        'python_version': sys.version
    })
    
    # Start background monitoring
    monitor_task = asyncio.create_task(_background_monitor())
    
    try:
        yield
    finally:
        # Cleanup
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass


async def _background_monitor():
    """Background task for performance monitoring."""
    while True:
        try:
            # Update memory metrics
            await MemoryMonitor.update_memory_metrics()
            
            # Update database connection metrics
            try:
                from .database import get_database_stats
                db_stats = await get_database_stats()
                await DatabaseMetrics.update_connection_count(db_stats["pool_size"])
            except Exception as e:
                logger.debug(f"Could not update database metrics: {e}")
            
            # Sleep for monitoring interval
            await asyncio.sleep(30)  # 30 seconds
        
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Background monitor error: {e}")
            await asyncio.sleep(60)  # Longer sleep on error


class AsyncProfiler:
    """Async function profiler for performance analysis."""
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.profiles: Dict[str, list] = {}
    
    def profile(self, func_name: str):
        """Decorator to profile async function execution time."""
        def decorator(func):
            if not self.enabled:
                return func
            
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    
                    if func_name not in self.profiles:
                        self.profiles[func_name] = []
                    
                    self.profiles[func_name].append(duration)
                    
                    # Keep only last 100 measurements
                    if len(self.profiles[func_name]) > 100:
                        self.profiles[func_name] = self.profiles[func_name][-100:]
            
            return wrapper
        return decorator
    
    def get_stats(self, func_name: str) -> Optional[Dict[str, float]]:
        """Get performance statistics for a function."""
        if func_name not in self.profiles:
            return None
        
        durations = self.profiles[func_name]
        if not durations:
            return None
        
        return {
            "count": len(durations),
            "min": min(durations),
            "max": max(durations),
            "avg": sum(durations) / len(durations),
            "total": sum(durations)
        }


# Global profiler instance
_profiler = AsyncProfiler(enabled=os.getenv("ENABLE_PROFILING", "false").lower() == "true")


def get_profiler() -> AsyncProfiler:
    """Get global profiler instance."""
    return _profiler