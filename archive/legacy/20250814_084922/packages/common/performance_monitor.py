"""
Performance Monitoring and Benchmarking System
Provides comprehensive performance metrics, benchmarking, and monitoring capabilities
"""

import time
import asyncio
import psutil
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from contextlib import asynccontextmanager
import statistics
import json
from enum import Enum

import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest


class MetricType(Enum):
    """Types of performance metrics"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    DATABASE_CONNECTIONS = "database_connections"
    CACHE_HIT_RATE = "cache_hit_rate"
    QUEUE_SIZE = "queue_size"


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark"""
    name: str
    duration: float
    operations_per_second: float
    min_time: float
    max_time: float
    avg_time: float
    median_time: float
    p95_time: float
    p99_time: float
    success_rate: float
    error_count: int
    total_operations: int
    memory_usage_mb: float
    cpu_usage_percent: float


@dataclass
class SystemMetrics:
    """System-level performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    open_file_descriptors: int
    active_connections: int
    timestamp: datetime


class PerformanceTimer:
    """Context manager for timing operations"""

    def __init__(self, name: str, monitor: 'PerformanceMonitor' = None):
        self.name = name
        self.monitor = monitor
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time

        if self.monitor:
            self.monitor.record_metric(
                self.name,
                MetricType.RESPONSE_TIME,
                duration * 1000,  # Convert to milliseconds
                unit="ms"
            )

        return False  # Don't suppress exceptions

    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


class PerformanceMonitor:
    """Comprehensive performance monitoring system"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.metrics: List[PerformanceMetric] = []
        self.benchmarks: Dict[str, List[BenchmarkResult]] = {}
        self.logger = logging.getLogger(__name__)

        # Prometheus metrics
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()

        # Performance tracking
        self.operation_counters: Dict[str, int] = {}
        self.response_times: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}

        # System monitoring
        self.system_metrics_history: List[SystemMetrics] = []
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        self.request_count = Counter(
            'xorb_requests_total',
            'Total requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )

        self.request_duration = Histogram(
            'xorb_request_duration_seconds',
            'Request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )

        self.memory_usage = Gauge(
            'xorb_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )

        self.cpu_usage = Gauge(
            'xorb_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )

        self.active_connections = Gauge(
            'xorb_active_connections',
            'Number of active connections',
            registry=self.registry
        )

    def record_metric(
        self,
        name: str,
        metric_type: MetricType,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        unit: str = ""
    ):
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            type=metric_type,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            unit=unit
        )

        self.metrics.append(metric)

        # Update Prometheus metrics
        if metric_type == MetricType.MEMORY_USAGE:
            self.memory_usage.set(value)
        elif metric_type == MetricType.CPU_USAGE:
            self.cpu_usage.set(value)

        # Store in Redis if available
        if self.redis_client:
            asyncio.create_task(self._store_metric_async(metric))

    async def _store_metric_async(self, metric: PerformanceMetric):
        """Store metric in Redis asynchronously"""
        try:
            key = f"metrics:{metric.name}:{metric.type.value}"
            data = {
                "value": metric.value,
                "timestamp": metric.timestamp.isoformat(),
                "tags": metric.tags,
                "unit": metric.unit
            }
            await self.redis_client.lpush(key, json.dumps(data))
            await self.redis_client.ltrim(key, 0, 999)  # Keep last 1000 metrics
        except Exception as e:
            self.logger.warning(f"Failed to store metric in Redis: {e}")

    def timer(self, name: str) -> PerformanceTimer:
        """Create a performance timer"""
        return PerformanceTimer(name, self)

    def track_operation(self, operation_name: str):
        """Decorator to track operation performance"""
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    with self.timer(f"{operation_name}_duration"):
                        try:
                            result = await func(*args, **kwargs)
                            self._increment_success(operation_name)
                            return result
                        except Exception as e:
                            self._increment_error(operation_name)
                            raise
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.timer(f"{operation_name}_duration"):
                        try:
                            result = func(*args, **kwargs)
                            self._increment_success(operation_name)
                            return result
                        except Exception as e:
                            self._increment_error(operation_name)
                            raise
                return sync_wrapper
        return decorator

    def _increment_success(self, operation_name: str):
        """Increment success counter"""
        self.operation_counters[operation_name] = self.operation_counters.get(operation_name, 0) + 1

    def _increment_error(self, operation_name: str):
        """Increment error counter"""
        self.error_counts[operation_name] = self.error_counts.get(operation_name, 0) + 1

    async def benchmark_function(
        self,
        func: Callable,
        name: str,
        iterations: int = 1000,
        concurrent: bool = False,
        concurrency_level: int = 10
    ) -> BenchmarkResult:
        """Benchmark a function's performance"""
        self.logger.info(f"Starting benchmark: {name} ({iterations} iterations)")

        times = []
        errors = 0
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_cpu = psutil.cpu_percent()

        start_time = time.perf_counter()

        if concurrent:
            # Concurrent execution
            semaphore = asyncio.Semaphore(concurrency_level)

            async def run_with_semaphore():
                async with semaphore:
                    try:
                        op_start = time.perf_counter()
                        if asyncio.iscoroutinefunction(func):
                            await func()
                        else:
                            func()
                        op_end = time.perf_counter()
                        return op_end - op_start
                    except Exception as e:
                        self.logger.warning(f"Benchmark error: {e}")
                        return None

            tasks = [run_with_semaphore() for _ in range(iterations)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    errors += 1
                elif result is not None:
                    times.append(result)
                else:
                    errors += 1
        else:
            # Sequential execution
            for _ in range(iterations):
                try:
                    op_start = time.perf_counter()
                    if asyncio.iscoroutinefunction(func):
                        await func()
                    else:
                        func()
                    op_end = time.perf_counter()
                    times.append(op_end - op_start)
                except Exception as e:
                    errors += 1
                    self.logger.warning(f"Benchmark error: {e}")

        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu = psutil.cpu_percent()

        total_duration = end_time - start_time
        successful_operations = len(times)

        if not times:
            raise ValueError("No successful operations in benchmark")

        result = BenchmarkResult(
            name=name,
            duration=total_duration,
            operations_per_second=successful_operations / total_duration,
            min_time=min(times) * 1000,  # Convert to ms
            max_time=max(times) * 1000,
            avg_time=statistics.mean(times) * 1000,
            median_time=statistics.median(times) * 1000,
            p95_time=statistics.quantiles(times, n=20)[18] * 1000,  # 95th percentile
            p99_time=statistics.quantiles(times, n=100)[98] * 1000,  # 99th percentile
            success_rate=(successful_operations / iterations) * 100,
            error_count=errors,
            total_operations=iterations,
            memory_usage_mb=end_memory - start_memory,
            cpu_usage_percent=end_cpu - start_cpu
        )

        # Store benchmark result
        if name not in self.benchmarks:
            self.benchmarks[name] = []
        self.benchmarks[name].append(result)

        self.logger.info(f"Benchmark completed: {name} - {result.operations_per_second:.2f} ops/sec")
        return result

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        process = psutil.Process()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()

        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024 / 1024,
            memory_available_mb=memory.available / 1024 / 1024,
            disk_usage_percent=(disk.used / disk.total) * 100,
            network_io_sent_mb=network.bytes_sent / 1024 / 1024,
            network_io_recv_mb=network.bytes_recv / 1024 / 1024,
            open_file_descriptors=process.num_fds() if hasattr(process, 'num_fds') else 0,
            active_connections=len(process.connections()),
            timestamp=datetime.utcnow()
        )

    async def start_monitoring(self, interval: int = 30):
        """Start continuous system monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitor_loop(interval))
        self.logger.info(f"Started performance monitoring (interval: {interval}s)")

    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped performance monitoring")

    async def _monitor_loop(self, interval: int):
        """Continuous monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = self.get_system_metrics()
                self.system_metrics_history.append(metrics)

                # Keep only last 24 hours of metrics (assuming 30s interval)
                max_metrics = (24 * 60 * 60) // interval
                if len(self.system_metrics_history) > max_metrics:
                    self.system_metrics_history = self.system_metrics_history[-max_metrics:]

                # Record individual metrics
                self.record_metric("system_cpu", MetricType.CPU_USAGE, metrics.cpu_percent, unit="%")
                self.record_metric("system_memory", MetricType.MEMORY_USAGE, metrics.memory_used_mb, unit="MB")

                await asyncio.sleep(interval)

            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(interval)

    def get_performance_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for the specified time window"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]

        summary = {
            "window_minutes": window_minutes,
            "total_metrics": len(recent_metrics),
            "metric_types": {},
            "operations": {},
            "system": {}
        }

        # Group metrics by type
        by_type = {}
        for metric in recent_metrics:
            if metric.type not in by_type:
                by_type[metric.type] = []
            by_type[metric.type].append(metric.value)

        # Calculate statistics for each type
        for metric_type, values in by_type.items():
            if values:
                summary["metric_types"][metric_type.value] = {
                    "count": len(values),
                    "avg": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "median": statistics.median(values)
                }

        # Operation statistics
        for operation, count in self.operation_counters.items():
            error_count = self.error_counts.get(operation, 0)
            summary["operations"][operation] = {
                "total_operations": count,
                "errors": error_count,
                "success_rate": ((count - error_count) / count * 100) if count > 0 else 0
            }

        # Recent system metrics
        recent_system = [m for m in self.system_metrics_history if m.timestamp >= cutoff_time]
        if recent_system:
            summary["system"] = {
                "avg_cpu": statistics.mean([m.cpu_percent for m in recent_system]),
                "avg_memory_mb": statistics.mean([m.memory_used_mb for m in recent_system]),
                "avg_memory_percent": statistics.mean([m.memory_percent for m in recent_system]),
                "samples": len(recent_system)
            }

        return summary

    def get_benchmark_history(self, name: Optional[str] = None) -> Dict[str, List[BenchmarkResult]]:
        """Get benchmark history"""
        if name:
            return {name: self.benchmarks.get(name, [])}
        return self.benchmarks.copy()

    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')

    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        metrics = self.get_system_metrics()

        health_status = {
            "status": "healthy",
            "timestamp": metrics.timestamp.isoformat(),
            "checks": {}
        }

        # CPU check
        if metrics.cpu_percent > 90:
            health_status["checks"]["cpu"] = {"status": "warning", "value": metrics.cpu_percent}
        else:
            health_status["checks"]["cpu"] = {"status": "ok", "value": metrics.cpu_percent}

        # Memory check
        if metrics.memory_percent > 90:
            health_status["checks"]["memory"] = {"status": "critical", "value": metrics.memory_percent}
        elif metrics.memory_percent > 80:
            health_status["checks"]["memory"] = {"status": "warning", "value": metrics.memory_percent}
        else:
            health_status["checks"]["memory"] = {"status": "ok", "value": metrics.memory_percent}

        # Disk check
        if metrics.disk_usage_percent > 95:
            health_status["checks"]["disk"] = {"status": "critical", "value": metrics.disk_usage_percent}
        elif metrics.disk_usage_percent > 85:
            health_status["checks"]["disk"] = {"status": "warning", "value": metrics.disk_usage_percent}
        else:
            health_status["checks"]["disk"] = {"status": "ok", "value": metrics.disk_usage_percent}

        # Overall status
        critical_checks = [c for c in health_status["checks"].values() if c["status"] == "critical"]
        warning_checks = [c for c in health_status["checks"].values() if c["status"] == "warning"]

        if critical_checks:
            health_status["status"] = "critical"
        elif warning_checks:
            health_status["status"] = "warning"

        return health_status


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def setup_performance_monitoring(redis_client: Optional[redis.Redis] = None):
    """Setup global performance monitoring"""
    global _global_monitor
    _global_monitor = PerformanceMonitor(redis_client)
    return _global_monitor


# Convenience decorators
def track_performance(operation_name: str):
    """Decorator to track operation performance"""
    monitor = get_performance_monitor()
    return monitor.track_operation(operation_name)


@asynccontextmanager
async def benchmark_context(name: str):
    """Async context manager for benchmarking"""
    monitor = get_performance_monitor()
    timer = monitor.timer(name)

    with timer:
        yield timer
