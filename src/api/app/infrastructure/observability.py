"""
Observability infrastructure with fallback implementations
"""

import logging
from typing import Any, Dict, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Fallback metrics collector"""
    
    def increment(self, metric_name: str, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        logger.debug(f"Metrics: {metric_name} incremented", extra={"tags": tags})
    
    def histogram(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value"""
        logger.debug(f"Metrics: {metric_name} = {value}", extra={"tags": tags})
    
    def gauge(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge value"""
        logger.debug(f"Metrics: {metric_name} = {value}", extra={"tags": tags})

def add_trace_context(func: Callable) -> Callable:
    """Add tracing context (fallback implementation)"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Add trace context in a real implementation
        return await func(*args, **kwargs)
    return wrapper

def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector instance"""
    return MetricsCollector()