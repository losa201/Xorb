"""
Advanced Performance Monitoring and Optimization Middleware
Implements comprehensive performance tracking and optimization for the XORB API
"""

import time
import asyncio
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from prometheus_client import Counter, Histogram, Gauge, Summary

from ..infrastructure.redis_manager import get_redis_manager

logger = logging.getLogger(__name__)

# Performance Metrics
request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint', 'status_code']
)

request_size = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint']
)

response_size = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint', 'status_code']
)

active_requests = Gauge(
    'http_active_requests',
    'Number of active HTTP requests',
    ['method', 'endpoint']
)

slow_requests = Counter(
    'http_slow_requests_total',
    'Total number of slow HTTP requests',
    ['method', 'endpoint']
)

error_requests = Counter(
    'http_error_requests_total',
    'Total number of HTTP error requests',
    ['method', 'endpoint', 'status_code']
)

cache_hits = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

cache_misses = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type']
)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a request"""
    request_id: str
    method: str
    path: str
    status_code: int
    duration_ms: float
    request_size_bytes: int
    response_size_bytes: int
    cache_hit: bool
    db_queries: int
    db_time_ms: float
    redis_operations: int
    redis_time_ms: float
    timestamp: datetime


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Advanced performance monitoring middleware"""
    
    def __init__(self, app, slow_request_threshold: float = 1.0):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold  # seconds
        self.performance_cache = {}
        
    async def dispatch(self, request: Request, call_next):
        # Skip metrics for internal endpoints
        if request.url.path in ['/metrics', '/health', '/favicon.ico']:
            return await call_next(request)
        
        start_time = time.time()
        method = request.method
        path = self._normalize_path(request.url.path)
        request_id = request.headers.get('x-request-id', 'unknown')
        
        # Get request size
        request_size_bytes = int(request.headers.get('content-length', 0))
        
        # Track active requests
        active_requests.labels(method=method, endpoint=path).inc()
        
        # Record request size
        request_size.labels(method=method, endpoint=path).observe(request_size_bytes)
        
        # Initialize performance context
        performance_context = {
            'start_time': start_time,
            'db_queries': 0,
            'db_time': 0.0,
            'redis_operations': 0,
            'redis_time': 0.0,
            'cache_hit': False
        }
        
        # Add performance context to request state
        request.state.performance = performance_context
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate metrics
            end_time = time.time()
            duration = end_time - start_time
            duration_ms = duration * 1000
            
            # Get response size
            response_size_bytes = 0
            if hasattr(response, 'body'):
                response_size_bytes = len(response.body)
            elif 'content-length' in response.headers:
                response_size_bytes = int(response.headers['content-length'])
            
            # Record metrics
            request_duration.labels(
                method=method, 
                endpoint=path, 
                status_code=response.status_code
            ).observe(duration)
            
            response_size.labels(
                method=method,
                endpoint=path,
                status_code=response.status_code
            ).observe(response_size_bytes)
            
            # Check for slow requests
            if duration > self.slow_request_threshold:
                slow_requests.labels(method=method, endpoint=path).inc()
                logger.warning(
                    f"Slow request detected: {method} {path} took {duration:.2f}s",
                    extra={
                        'request_id': request_id,
                        'duration': duration,
                        'method': method,
                        'path': path
                    }
                )
            
            # Check for errors
            if response.status_code >= 400:
                error_requests.labels(
                    method=method,
                    endpoint=path,
                    status_code=response.status_code
                ).inc()
            
            # Record detailed performance metrics
            await self._record_performance_metrics(
                request_id, method, path, response.status_code,
                duration_ms, request_size_bytes, response_size_bytes,
                performance_context
            )
            
            # Add performance headers
            response.headers['X-Response-Time'] = f"{duration_ms:.2f}ms"
            response.headers['X-Request-ID'] = request_id
            
            return response
            
        except Exception as e:
            # Record error metrics
            end_time = time.time()
            duration = end_time - start_time
            
            error_requests.labels(
                method=method,
                endpoint=path,
                status_code=500
            ).inc()
            
            logger.error(
                f"Request error: {method} {path} failed after {duration:.2f}s",
                extra={
                    'request_id': request_id,
                    'error': str(e),
                    'duration': duration
                }
            )
            
            raise
            
        finally:
            # Always decrement active requests
            active_requests.labels(method=method, endpoint=path).dec()
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path for metrics (remove IDs, etc.)"""
        # Replace UUIDs and numeric IDs with placeholders
        import re
        
        # Replace UUIDs
        path = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '/{uuid}',
            path,
            flags=re.IGNORECASE
        )
        
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        # Replace session IDs
        path = re.sub(r'/session_[a-zA-Z0-9]+', '/session_{id}', path)
        
        return path
    
    async def _record_performance_metrics(
        self,
        request_id: str,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        request_size_bytes: int,
        response_size_bytes: int,
        context: Dict[str, Any]
    ):
        """Record detailed performance metrics"""
        
        metrics = PerformanceMetrics(
            request_id=request_id,
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            request_size_bytes=request_size_bytes,
            response_size_bytes=response_size_bytes,
            cache_hit=context.get('cache_hit', False),
            db_queries=context.get('db_queries', 0),
            db_time_ms=context.get('db_time', 0.0) * 1000,
            redis_operations=context.get('redis_operations', 0),
            redis_time_ms=context.get('redis_time', 0.0) * 1000,
            timestamp=datetime.utcnow()
        )
        
        # Store in Redis for analysis (with TTL)
        try:
            redis_manager = await get_redis_manager()
            if redis_manager and redis_manager.is_healthy:
                key = f"performance_metrics:{request_id}"
                value = json.dumps(asdict(metrics), default=str)
                await redis_manager.set(key, value, ex=3600)  # 1 hour TTL
        except Exception as e:
            logger.error(f"Failed to store performance metrics: {e}")
        
        # Log performance data for slow requests
        if duration_ms > self.slow_request_threshold * 1000:
            logger.info(
                "Performance metrics for slow request",
                extra={
                    'metrics': asdict(metrics),
                    'performance_analysis': self._analyze_performance(metrics)
                }
            )
    
    def _analyze_performance(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Analyze performance metrics and provide insights"""
        analysis = {
            'overall_rating': 'good',
            'bottlenecks': [],
            'recommendations': []
        }
        
        # Analyze response time
        if metrics.duration_ms > 5000:  # 5 seconds
            analysis['overall_rating'] = 'poor'
            analysis['bottlenecks'].append('very_slow_response')
            analysis['recommendations'].append('Investigate application logic and database queries')
        elif metrics.duration_ms > 1000:  # 1 second
            analysis['overall_rating'] = 'slow'
            analysis['bottlenecks'].append('slow_response')
            analysis['recommendations'].append('Consider optimizing critical path')
        
        # Analyze database performance
        if metrics.db_time_ms > metrics.duration_ms * 0.5:  # DB time > 50% of total
            analysis['bottlenecks'].append('database_bottleneck')
            analysis['recommendations'].append('Optimize database queries and indexing')
        
        # Analyze cache effectiveness
        if not metrics.cache_hit and metrics.method == 'GET':
            analysis['bottlenecks'].append('cache_miss')
            analysis['recommendations'].append('Consider implementing caching for this endpoint')
        
        # Analyze response size
        if metrics.response_size_bytes > 1024 * 1024:  # 1MB
            analysis['bottlenecks'].append('large_response')
            analysis['recommendations'].append('Consider pagination or response compression')
        
        return analysis


class RequestTracker:
    """Track request performance over time"""
    
    def __init__(self):
        self.request_history = []
        self.max_history = 1000
    
    async def track_request(self, metrics: PerformanceMetrics):
        """Track a request in history"""
        self.request_history.append(metrics)
        
        # Keep only recent requests
        if len(self.request_history) > self.max_history:
            self.request_history = self.request_history[-self.max_history:]
    
    def get_performance_summary(self, time_window: int = 300) -> Dict[str, Any]:
        """Get performance summary for last N seconds"""
        cutoff_time = datetime.utcnow() - timedelta(seconds=time_window)
        recent_requests = [
            r for r in self.request_history 
            if r.timestamp > cutoff_time
        ]
        
        if not recent_requests:
            return {"message": "No requests in time window"}
        
        durations = [r.duration_ms for r in recent_requests]
        
        return {
            "time_window_seconds": time_window,
            "total_requests": len(recent_requests),
            "avg_response_time_ms": sum(durations) / len(durations),
            "min_response_time_ms": min(durations),
            "max_response_time_ms": max(durations),
            "p95_response_time_ms": self._percentile(durations, 0.95),
            "p99_response_time_ms": self._percentile(durations, 0.99),
            "error_rate": len([r for r in recent_requests if r.status_code >= 400]) / len(recent_requests),
            "cache_hit_rate": len([r for r in recent_requests if r.cache_hit]) / len(recent_requests) if recent_requests else 0
        }
    
    def _percentile(self, data: list, percentile: float) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


# Global request tracker
request_tracker = RequestTracker()


def get_request_tracker() -> RequestTracker:
    """Get global request tracker"""
    return request_tracker


# Utility functions for performance monitoring
def track_db_query(request: Request, duration: float):
    """Track database query performance"""
    if hasattr(request.state, 'performance'):
        request.state.performance['db_queries'] += 1
        request.state.performance['db_time'] += duration


def track_redis_operation(request: Request, duration: float):
    """Track Redis operation performance"""
    if hasattr(request.state, 'performance'):
        request.state.performance['redis_operations'] += 1
        request.state.performance['redis_time'] += duration


def mark_cache_hit(request: Request):
    """Mark that a cache hit occurred"""
    if hasattr(request.state, 'performance'):
        request.state.performance['cache_hit'] = True
    
    cache_hits.labels(cache_type='application').inc()


def mark_cache_miss(request: Request):
    """Mark that a cache miss occurred"""
    cache_misses.labels(cache_type='application').inc()