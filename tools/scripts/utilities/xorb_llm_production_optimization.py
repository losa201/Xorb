#!/usr/bin/env python3
"""
XORB LLM Cognitive Cortex Production Optimization

Production-grade optimizations for the LLM cognitive cortex including
performance tuning, resource management, monitoring, and error handling.
"""

import os
import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import aiohttp
from concurrent.futures import ThreadPoolExecutor

# Production configuration optimizations
PRODUCTION_CONFIG = {
    "performance": {
        "max_concurrent_requests": 50,
        "request_timeout_seconds": 30,
        "circuit_breaker_threshold": 5,
        "circuit_breaker_timeout": 60,
        "connection_pool_size": 100,
        "keepalive_timeout": 30,
        "retry_attempts": 3,
        "retry_backoff_factor": 2.0
    },
    "caching": {
        "response_cache_ttl": 3600,  # 1 hour
        "model_cache_ttl": 86400,   # 24 hours
        "performance_cache_ttl": 43200,  # 12 hours
        "max_cache_size": 10000,
        "cache_cleanup_interval": 1800  # 30 minutes
    },
    "monitoring": {
        "metrics_collection_interval": 30,
        "health_check_interval": 60,
        "alert_thresholds": {
            "error_rate": 0.05,  # 5%
            "response_time_p99": 15.0,  # 15 seconds
            "memory_usage": 0.85,  # 85%
            "cpu_usage": 0.80  # 80%
        }
    },
    "security": {
        "api_key_rotation_hours": 12,
        "audit_log_retention_days": 90,
        "max_request_size": 1048576,  # 1MB
        "rate_limit_per_minute": 100,
        "enable_request_validation": True
    },
    "resilience": {
        "graceful_shutdown_timeout": 30,
        "health_check_retries": 3,
        "fallback_model_enabled": True,
        "auto_recovery_enabled": True,
        "circuit_breaker_enabled": True
    }
}


@dataclass
class ProductionMetrics:
    """Production metrics tracking"""
    requests_total: int = 0
    requests_successful: int = 0
    requests_failed: int = 0
    average_response_time: float = 0.0
    p99_response_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_connections: int = 0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0
    last_updated: datetime = None


class ProductionLLMOptimizer:
    """Production optimization manager for LLM cortex"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = {**PRODUCTION_CONFIG, **(config or {})}
        self.metrics = ProductionMetrics()
        self.start_time = time.time()
        
        # Circuit breaker state
        self.circuit_breaker = {
            "state": "CLOSED",  # CLOSED, OPEN, HALF_OPEN
            "failure_count": 0,
            "last_failure": None,
            "next_attempt": None
        }
        
        # Performance monitoring
        self.response_times: List[float] = []
        self.error_counts: Dict[str, int] = {}
        
        # Resource management
        self.connection_pool = None
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        self.logger = logging.getLogger("xorb.llm.production")
    
    async def initialize(self):
        """Initialize production optimizations"""
        self.logger.info("Initializing production optimizations...")
        
        # Setup connection pool
        connector = aiohttp.TCPConnector(
            limit=self.config["performance"]["connection_pool_size"],
            limit_per_host=30,
            keepalive_timeout=self.config["performance"]["keepalive_timeout"],
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=self.config["performance"]["request_timeout_seconds"]
        )
        
        self.connection_pool = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        # Start background tasks
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._health_monitoring_loop())
        asyncio.create_task(self._cache_cleanup_loop())
        
        self.logger.info("Production optimizations initialized")
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Starting graceful shutdown...")
        
        if self.connection_pool:
            await self.connection_pool.close()
        
        self.thread_pool.shutdown(wait=True)
        
        self.logger.info("Graceful shutdown completed")
    
    async def optimize_request(self, model: str, prompt: str, task_type: str) -> Dict[str, Any]:
        """Optimize LLM request with production features"""
        
        # Circuit breaker check
        if not self._circuit_breaker_allow_request():
            raise Exception("Circuit breaker is OPEN - service temporarily unavailable")
        
        # Request validation
        if self.config["security"]["enable_request_validation"]:
            self._validate_request(prompt)
        
        start_time = time.time()
        
        try:
            # Execute optimized request
            result = await self._execute_optimized_request(model, prompt, task_type)
            
            # Record success metrics
            response_time = time.time() - start_time
            self._record_success(response_time)
            
            return result
            
        except Exception as e:
            # Record failure metrics
            response_time = time.time() - start_time
            self._record_failure(str(e), response_time)
            
            # Circuit breaker logic
            self._circuit_breaker_record_failure()
            
            raise
    
    async def _execute_optimized_request(self, model: str, prompt: str, task_type: str) -> Dict[str, Any]:
        """Execute request with production optimizations"""
        
        # Retry logic with exponential backoff
        for attempt in range(self.config["performance"]["retry_attempts"]):
            try:
                # Use connection pool for request
                async with self.connection_pool.post(
                    "http://localhost:8009/llm/request",
                    json={
                        "prompt": prompt,
                        "task_type": task_type,
                        "agent_id": "production-optimizer"
                    }
                ) as response:
                    
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"Request failed: {response.status} - {error_text}")
                        
            except Exception as e:
                if attempt < self.config["performance"]["retry_attempts"] - 1:
                    # Exponential backoff
                    delay = self.config["performance"]["retry_backoff_factor"] ** attempt
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise
    
    def _circuit_breaker_allow_request(self) -> bool:
        """Check if circuit breaker allows request"""
        if not self.config["resilience"]["circuit_breaker_enabled"]:
            return True
        
        now = time.time()
        cb = self.circuit_breaker
        
        if cb["state"] == "CLOSED":
            return True
        elif cb["state"] == "OPEN":
            if cb["next_attempt"] and now >= cb["next_attempt"]:
                cb["state"] = "HALF_OPEN"
                return True
            return False
        elif cb["state"] == "HALF_OPEN":
            return True
        
        return True
    
    def _circuit_breaker_record_failure(self):
        """Record circuit breaker failure"""
        if not self.config["resilience"]["circuit_breaker_enabled"]:
            return
        
        cb = self.circuit_breaker
        cb["failure_count"] += 1
        cb["last_failure"] = time.time()
        
        threshold = self.config["performance"]["circuit_breaker_threshold"]
        timeout = self.config["performance"]["circuit_breaker_timeout"]
        
        if cb["failure_count"] >= threshold and cb["state"] == "CLOSED":
            cb["state"] = "OPEN"
            cb["next_attempt"] = time.time() + timeout
            self.logger.warning(f"Circuit breaker opened after {cb['failure_count']} failures")
    
    def _circuit_breaker_record_success(self):
        """Record circuit breaker success"""
        cb = self.circuit_breaker
        
        if cb["state"] == "HALF_OPEN":
            cb["state"] = "CLOSED"
            cb["failure_count"] = 0
            self.logger.info("Circuit breaker closed after successful request")
    
    def _validate_request(self, prompt: str):
        """Validate request for security"""
        max_size = self.config["security"]["max_request_size"]
        
        if len(prompt.encode('utf-8')) > max_size:
            raise ValueError(f"Request size exceeds limit: {len(prompt)} > {max_size}")
        
        # Additional validation could be added here
        # - Content filtering
        # - Input sanitization
        # - Rate limiting per user
    
    def _record_success(self, response_time: float):
        """Record successful request metrics"""
        self.metrics.requests_total += 1
        self.metrics.requests_successful += 1
        
        # Update response times
        self.response_times.append(response_time)
        if len(self.response_times) > 1000:  # Keep last 1000 requests
            self.response_times = self.response_times[-1000:]
        
        # Update averages
        self.metrics.average_response_time = sum(self.response_times) / len(self.response_times)
        
        if len(self.response_times) >= 100:
            sorted_times = sorted(self.response_times)
            p99_index = int(len(sorted_times) * 0.99)
            self.metrics.p99_response_time = sorted_times[p99_index]
        
        # Circuit breaker success
        self._circuit_breaker_record_success()
    
    def _record_failure(self, error: str, response_time: float):
        """Record failed request metrics"""
        self.metrics.requests_total += 1
        self.metrics.requests_failed += 1
        
        # Track error types
        error_type = error.split(':')[0] if ':' in error else error
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Update error rate
        if self.metrics.requests_total > 0:
            self.metrics.error_rate = self.metrics.requests_failed / self.metrics.requests_total
    
    async def _metrics_collection_loop(self):
        """Background metrics collection"""
        interval = self.config["monitoring"]["metrics_collection_interval"]
        
        while True:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        import psutil
        
        # System metrics
        self.metrics.memory_usage_mb = psutil.virtual_memory().used / (1024 * 1024)
        self.metrics.cpu_usage_percent = psutil.cpu_percent()
        
        # Uptime
        self.metrics.uptime_seconds = time.time() - self.start_time
        
        # Update timestamp
        self.metrics.last_updated = datetime.utcnow()
        
        # Check alert thresholds
        await self._check_alert_thresholds()
    
    async def _check_alert_thresholds(self):
        """Check if metrics exceed alert thresholds"""
        thresholds = self.config["monitoring"]["alert_thresholds"]
        
        alerts = []
        
        if self.metrics.error_rate > thresholds["error_rate"]:
            alerts.append(f"Error rate too high: {self.metrics.error_rate:.3f} > {thresholds['error_rate']}")
        
        if self.metrics.p99_response_time > thresholds["response_time_p99"]:
            alerts.append(f"P99 response time too high: {self.metrics.p99_response_time:.2f}s > {thresholds['response_time_p99']}s")
        
        if self.metrics.memory_usage_mb > 0:  # Only if we have memory data
            memory_percent = self.metrics.memory_usage_mb / (psutil.virtual_memory().total / (1024 * 1024))
            if memory_percent > thresholds["memory_usage"]:
                alerts.append(f"Memory usage too high: {memory_percent:.3f} > {thresholds['memory_usage']}")
        
        if self.metrics.cpu_usage_percent > thresholds["cpu_usage"] * 100:
            alerts.append(f"CPU usage too high: {self.metrics.cpu_usage_percent:.1f}% > {thresholds['cpu_usage'] * 100}%")
        
        if alerts:
            for alert in alerts:
                self.logger.warning(f"ALERT: {alert}")
    
    async def _health_monitoring_loop(self):
        """Background health monitoring"""
        interval = self.config["monitoring"]["health_check_interval"]
        
        while True:
            try:
                await self._perform_health_check()
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(interval)
    
    async def _perform_health_check(self):
        """Perform comprehensive health check"""
        try:
            # Check LLM cortex service
            async with self.connection_pool.get("http://localhost:8009/llm/health") as response:
                if response.status != 200:
                    self.logger.error(f"LLM cortex health check failed: {response.status}")
                    return False
            
            # Check circuit breaker state
            if self.circuit_breaker["state"] == "OPEN":
                self.logger.warning("Circuit breaker is OPEN")
            
            # All checks passed
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def _cache_cleanup_loop(self):
        """Background cache cleanup"""
        interval = self.config["caching"]["cache_cleanup_interval"]
        
        while True:
            try:
                await self._cleanup_caches()
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(interval)
    
    async def _cleanup_caches(self):
        """Clean up expired cache entries"""
        # This would implement cache cleanup logic
        # For now, just log the cleanup
        self.logger.debug("Performing cache cleanup")
    
    def get_production_metrics(self) -> Dict[str, Any]:
        """Get comprehensive production metrics"""
        return {
            "performance": {
                "requests_total": self.metrics.requests_total,
                "requests_successful": self.metrics.requests_successful,
                "requests_failed": self.metrics.requests_failed,
                "success_rate": (
                    self.metrics.requests_successful / self.metrics.requests_total
                    if self.metrics.requests_total > 0 else 0.0
                ),
                "error_rate": self.metrics.error_rate,
                "average_response_time": self.metrics.average_response_time,
                "p99_response_time": self.metrics.p99_response_time
            },
            "system": {
                "memory_usage_mb": self.metrics.memory_usage_mb,
                "cpu_usage_percent": self.metrics.cpu_usage_percent,
                "uptime_seconds": self.metrics.uptime_seconds,
                "active_connections": self.metrics.active_connections
            },
            "circuit_breaker": {
                "state": self.circuit_breaker["state"],
                "failure_count": self.circuit_breaker["failure_count"],
                "last_failure": self.circuit_breaker["last_failure"]
            },
            "errors": dict(self.error_counts),
            "config": self.config,
            "last_updated": self.metrics.last_updated.isoformat() if self.metrics.last_updated else None
        }
    
    def generate_production_report(self) -> str:
        """Generate production status report"""
        metrics = self.get_production_metrics()
        
        report = f"""
# XORB LLM Cognitive Cortex Production Report

**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

## Performance Metrics
- **Total Requests:** {metrics['performance']['requests_total']:,}
- **Success Rate:** {metrics['performance']['success_rate']:.3f} ({metrics['performance']['success_rate']*100:.1f}%)
- **Error Rate:** {metrics['performance']['error_rate']:.3f} ({metrics['performance']['error_rate']*100:.1f}%)
- **Average Response Time:** {metrics['performance']['average_response_time']:.3f}s
- **P99 Response Time:** {metrics['performance']['p99_response_time']:.3f}s

## System Health
- **Memory Usage:** {metrics['system']['memory_usage_mb']:.1f} MB
- **CPU Usage:** {metrics['system']['cpu_usage_percent']:.1f}%
- **Uptime:** {metrics['system']['uptime_seconds']:.0f}s ({metrics['system']['uptime_seconds']/3600:.1f} hours)

## Circuit Breaker Status
- **State:** {metrics['circuit_breaker']['state']}
- **Failure Count:** {metrics['circuit_breaker']['failure_count']}

## Error Summary
"""
        
        if metrics['errors']:
            for error_type, count in metrics['errors'].items():
                report += f"- **{error_type}:** {count} occurrences\n"
        else:
            report += "- No errors recorded\n"
        
        return report


# Global production optimizer instance
_production_optimizer = None

async def get_production_optimizer() -> ProductionLLMOptimizer:
    """Get global production optimizer instance"""
    global _production_optimizer
    if _production_optimizer is None:
        _production_optimizer = ProductionLLMOptimizer()
        await _production_optimizer.initialize()
    return _production_optimizer


async def optimize_llm_request(model: str, prompt: str, task_type: str) -> Dict[str, Any]:
    """Optimize LLM request with production features"""
    optimizer = await get_production_optimizer()
    return await optimizer.optimize_request(model, prompt, task_type)


def get_production_status() -> Dict[str, Any]:
    """Get current production status"""
    if _production_optimizer:
        return _production_optimizer.get_production_metrics()
    return {"status": "not_initialized"}


async def main():
    """Test production optimization features"""
    print("🚀 Testing XORB LLM Production Optimization")
    print("=" * 50)
    
    optimizer = await get_production_optimizer()
    
    # Test metrics collection
    print("📊 Collecting initial metrics...")
    await optimizer._collect_system_metrics()
    
    metrics = optimizer.get_production_metrics()
    print(f"Memory usage: {metrics['system']['memory_usage_mb']:.1f} MB")
    print(f"CPU usage: {metrics['system']['cpu_usage_percent']:.1f}%")
    print(f"Circuit breaker state: {metrics['circuit_breaker']['state']}")
    
    # Generate report
    print("\n📄 Production Report:")
    print(optimizer.generate_production_report())
    
    await optimizer.shutdown()
    print("\n✅ Production optimization test completed")


if __name__ == "__main__":
    asyncio.run(main())