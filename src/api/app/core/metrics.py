"""
Comprehensive metrics and monitoring for production deployment
"""

import time
import psutil
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
import asyncio

from prometheus_client import (
    Counter, Histogram, Gauge, Info, CollectorRegistry,
    start_http_server, generate_latest
)
import structlog

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricConfig:
    """Configuration for metrics collection"""
    enable_prometheus: bool = True
    enable_custom_metrics: bool = True
    prometheus_port: int = 9090
    collection_interval: int = 60
    retention_days: int = 7
    enable_detailed_metrics: bool = True


class CustomMetricsCollector:
    """Custom metrics collector for application-specific metrics"""
    
    def __init__(self, config: MetricConfig):
        self.config = config
        self.metrics: Dict[str, Any] = {}
        self.counters: Dict[str, int] = defaultdict(int)
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.gauges: Dict[str, float] = {}
        self.info: Dict[str, Dict[str, str]] = {}
        self.lock = Lock()
        self.start_time = datetime.utcnow()
    
    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        with self.lock:
            key = f"{name}:{':'.join(f'{k}={v}' for k, v in (labels or {}).items())}"
            self.counters[key] += value
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram value"""
        with self.lock:
            key = f"{name}:{':'.join(f'{k}={v}' for k, v in (labels or {}).items())}"
            self.histograms[key].append((time.time(), value))
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge value"""
        with self.lock:
            key = f"{name}:{':'.join(f'{k}={v}' for k, v in (labels or {}).items())}"
            self.gauges[key] = value
    
    def set_info(self, name: str, info_dict: Dict[str, str]):
        """Set info metric"""
        with self.lock:
            self.info[name] = info_dict
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        with self.lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "info": dict(self.info),
                "histogram_counts": {k: len(v) for k, v in self.histograms.items()},
                "collection_time": datetime.utcnow().isoformat(),
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
            }


class PrometheusMetrics:
    """Prometheus metrics for production monitoring"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        
        # HTTP Request metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.http_request_duration_seconds = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.http_request_size_bytes = Histogram(
            'http_request_size_bytes',
            'HTTP request size in bytes',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.http_response_size_bytes = Histogram(
            'http_response_size_bytes',
            'HTTP response size in bytes',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Database metrics
        self.database_connections = Gauge(
            'database_connections',
            'Number of database connections',
            ['database', 'state'],
            registry=self.registry
        )
        
        self.database_query_duration_seconds = Histogram(
            'database_query_duration_seconds',
            'Database query duration in seconds',
            ['database', 'operation'],
            registry=self.registry
        )
        
        self.database_errors_total = Counter(
            'database_errors_total',
            'Total database errors',
            ['database', 'error_type'],
            registry=self.registry
        )
        
        # Application metrics
        self.scan_sessions_total = Counter(
            'scan_sessions_total',
            'Total scan sessions',
            ['scan_type', 'status'],
            registry=self.registry
        )
        
        self.scan_duration_seconds = Histogram(
            'scan_duration_seconds',
            'Scan duration in seconds',
            ['scan_type'],
            registry=self.registry
        )
        
        self.threats_detected_total = Counter(
            'threats_detected_total',
            'Total threats detected',
            ['severity', 'type'],
            registry=self.registry
        )
        
        self.active_users = Gauge(
            'active_users',
            'Number of active users',
            ['time_window'],
            registry=self.registry
        )
        
        # System metrics
        self.process_cpu_seconds_total = Counter(
            'process_cpu_seconds_total',
            'Total CPU time spent',
            registry=self.registry
        )
        
        self.process_memory_bytes = Gauge(
            'process_memory_bytes',
            'Process memory usage in bytes',
            ['type'],
            registry=self.registry
        )
        
        self.process_open_fds = Gauge(
            'process_open_fds',
            'Number of open file descriptors',
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_operations_total = Counter(
            'cache_operations_total',
            'Total cache operations',
            ['operation', 'cache'],
            registry=self.registry
        )
        
        self.cache_hit_ratio = Gauge(
            'cache_hit_ratio',
            'Cache hit ratio',
            ['cache'],
            registry=self.registry
        )
        
        # Security metrics
        self.auth_attempts_total = Counter(
            'auth_attempts_total',
            'Total authentication attempts',
            ['result'],
            registry=self.registry
        )
        
        self.rate_limit_violations_total = Counter(
            'rate_limit_violations_total',
            'Total rate limit violations',
            ['endpoint'],
            registry=self.registry
        )
        
        # Application info
        self.app_info = Info(
            'app_info',
            'Application information',
            registry=self.registry
        )


class SystemMetricsCollector:
    """Collect system-level metrics"""
    
    def __init__(self, prometheus_metrics: PrometheusMetrics):
        self.prometheus = prometheus_metrics
        self.process = psutil.Process()
        logger.info("System metrics collector initialized")
    
    def collect_system_metrics(self):
        """Collect and update system metrics"""
        try:
            # CPU metrics
            cpu_times = self.process.cpu_times()
            self.prometheus.process_cpu_seconds_total._value.set(cpu_times.user + cpu_times.system)
            
            # Memory metrics
            memory_info = self.process.memory_info()
            self.prometheus.process_memory_bytes.labels(type='rss').set(memory_info.rss)
            self.prometheus.process_memory_bytes.labels(type='vms').set(memory_info.vms)
            
            # File descriptor metrics
            try:
                num_fds = self.process.num_fds()
                self.prometheus.process_open_fds.set(num_fds)
            except AttributeError:
                # Windows doesn't have num_fds
                pass
            
            logger.debug("System metrics collected successfully")
            
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))


class MetricsService:
    """Main metrics service for the application"""
    
    def __init__(self, config: MetricConfig):
        self.config = config
        self.prometheus_metrics = PrometheusMetrics()
        self.custom_metrics = CustomMetricsCollector(config)
        self.system_metrics = SystemMetricsCollector(self.prometheus_metrics)
        self.running = False
        self.collection_task: Optional[asyncio.Task] = None
        
        # Set application info
        self.prometheus_metrics.app_info.info({
            'version': '3.0.0',
            'component': 'xorb-api',
            'environment': 'production'
        })
        
        logger.info("Metrics service initialized", config=config.__dict__)
    
    async def start(self):
        """Start metrics collection"""
        if self.running:
            logger.warning("Metrics service already running")
            return
        
        self.running = True
        
        # Start Prometheus HTTP server if enabled
        if self.config.enable_prometheus:
            try:
                start_http_server(self.config.prometheus_port, registry=self.prometheus_metrics.registry)
                logger.info("Prometheus metrics server started", port=self.config.prometheus_port)
            except Exception as e:
                logger.error("Failed to start Prometheus server", error=str(e))
        
        # Start metrics collection task
        if self.config.enable_custom_metrics:
            self.collection_task = asyncio.create_task(self._collection_loop())
            logger.info("Metrics collection started")
    
    async def stop(self):
        """Stop metrics collection"""
        self.running = False
        
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Metrics service stopped")
    
    async def _collection_loop(self):
        """Main metrics collection loop"""
        while self.running:
            try:
                self.system_metrics.collect_system_metrics()
                await asyncio.sleep(self.config.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in metrics collection loop", error=str(e))
                await asyncio.sleep(5)  # Short delay before retry
    
    def record_http_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        request_size: int = 0,
        response_size: int = 0
    ):
        """Record HTTP request metrics"""
        self.prometheus_metrics.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code)
        ).inc()
        
        self.prometheus_metrics.http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        if request_size > 0:
            self.prometheus_metrics.http_request_size_bytes.labels(
                method=method,
                endpoint=endpoint
            ).observe(request_size)
        
        if response_size > 0:
            self.prometheus_metrics.http_response_size_bytes.labels(
                method=method,
                endpoint=endpoint
            ).observe(response_size)
    
    def record_database_operation(
        self,
        database: str,
        operation: str,
        duration: float,
        success: bool = True,
        error_type: Optional[str] = None
    ):
        """Record database operation metrics"""
        self.prometheus_metrics.database_query_duration_seconds.labels(
            database=database,
            operation=operation
        ).observe(duration)
        
        if not success and error_type:
            self.prometheus_metrics.database_errors_total.labels(
                database=database,
                error_type=error_type
            ).inc()
    
    def record_scan_session(self, scan_type: str, status: str, duration: Optional[float] = None):
        """Record scan session metrics"""
        self.prometheus_metrics.scan_sessions_total.labels(
            scan_type=scan_type,
            status=status
        ).inc()
        
        if duration is not None:
            self.prometheus_metrics.scan_duration_seconds.labels(
                scan_type=scan_type
            ).observe(duration)
    
    def record_threat_detection(self, severity: str, threat_type: str):
        """Record threat detection metrics"""
        self.prometheus_metrics.threats_detected_total.labels(
            severity=severity,
            type=threat_type
        ).inc()
    
    def record_auth_attempt(self, success: bool):
        """Record authentication attempt"""
        result = "success" if success else "failure"
        self.prometheus_metrics.auth_attempts_total.labels(result=result).inc()
    
    def record_rate_limit_violation(self, endpoint: str):
        """Record rate limit violation"""
        self.prometheus_metrics.rate_limit_violations_total.labels(endpoint=endpoint).inc()
    
    def get_metrics_export(self) -> str:
        """Get Prometheus metrics export"""
        return generate_latest(self.prometheus_metrics.registry).decode()
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health-related metrics"""
        return {
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
            },
            "application": self.custom_metrics.get_metrics_summary(),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global metrics service instance
_metrics_service: Optional[MetricsService] = None


def get_metrics_service() -> Optional[MetricsService]:
    """Get the global metrics service instance"""
    return _metrics_service


def setup_metrics(config: MetricConfig) -> MetricsService:
    """Setup global metrics service"""
    global _metrics_service
    _metrics_service = MetricsService(config)
    return _metrics_service


# Decorator for timing functions
def timed_operation(operation_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to time operations"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                if _metrics_service:
                    _metrics_service.custom_metrics.record_histogram(
                        f"{operation_name}_duration_seconds",
                        duration,
                        labels
                    )
                return result
            except Exception as e:
                duration = time.time() - start_time
                if _metrics_service:
                    _metrics_service.custom_metrics.record_histogram(
                        f"{operation_name}_duration_seconds",
                        duration,
                        {**(labels or {}), "status": "error"}
                    )
                raise
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                if _metrics_service:
                    _metrics_service.custom_metrics.record_histogram(
                        f"{operation_name}_duration_seconds",
                        duration,
                        labels
                    )
                return result
            except Exception as e:
                duration = time.time() - start_time
                if _metrics_service:
                    _metrics_service.custom_metrics.record_histogram(
                        f"{operation_name}_duration_seconds",
                        duration,
                        {**(labels or {}), "status": "error"}
                    )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator