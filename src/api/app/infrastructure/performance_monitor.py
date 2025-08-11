"""
Advanced Performance Monitoring for XORB Enterprise Platform
Real-time performance analytics, optimization, and alerting
"""

import asyncio
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import threading
from contextlib import asynccontextmanager
import aioredis
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, 
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    active_connections: int
    request_rate: float
    response_time_p95: float
    response_time_p99: float
    error_rate: float
    database_connections: int
    cache_hit_rate: float


@dataclass
class ServiceHealthMetrics:
    """Service health metrics"""
    service_name: str
    uptime_seconds: float
    request_count: int
    error_count: int
    avg_response_time: float
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None


@dataclass
class AlertCondition:
    """Alert condition configuration"""
    metric_name: str
    threshold: float
    comparison: str  # gt, lt, eq
    duration_seconds: int
    severity: str    # critical, warning, info
    message_template: str


class PerformanceMonitor:
    """
    Advanced performance monitoring system with real-time analytics
    - Real-time system and application metrics collection
    - Prometheus integration for metrics export
    - Intelligent alerting based on configurable thresholds
    - Performance optimization recommendations
    - Historical trend analysis
    """
    
    def __init__(
        self,
        redis_client: Optional[aioredis.Redis] = None,
        collection_interval: int = 10,
        retention_period: int = 86400,  # 24 hours
        enable_prometheus: bool = True
    ):
        self.redis_client = redis_client
        self.collection_interval = collection_interval
        self.retention_period = retention_period
        self.enable_prometheus = enable_prometheus
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=1000)
        self.service_metrics: Dict[str, ServiceHealthMetrics] = {}
        self.alert_conditions: List[AlertCondition] = []
        
        # Performance tracking
        self.request_timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.last_network_stats = None
        self.last_disk_stats = None
        
        # Prometheus metrics
        if self.enable_prometheus:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Alert handlers
        self.alert_handlers: List[Callable] = []
        
        # Setup default alert conditions
        self._setup_default_alerts()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics collectors"""
        
        # System metrics
        self.cpu_usage_gauge = Gauge(
            'system_cpu_usage_percent', 
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage_gauge = Gauge(
            'system_memory_usage_percent',
            'Memory usage percentage', 
            registry=self.registry
        )
        
        self.memory_used_gauge = Gauge(
            'system_memory_used_mb',
            'Memory used in MB',
            registry=self.registry
        )
        
        # Application metrics
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.active_connections_gauge = Gauge(
            'active_connections_total',
            'Total active connections',
            registry=self.registry
        )
        
        self.database_connections_gauge = Gauge(
            'database_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        self.cache_hit_rate_gauge = Gauge(
            'cache_hit_rate_percent',
            'Cache hit rate percentage',
            registry=self.registry
        )
        
        # Business metrics
        self.scan_duration = Histogram(
            'ptaas_scan_duration_seconds',
            'PTaaS scan duration',
            ['scan_profile', 'status'],
            registry=self.registry
        )
        
        self.scan_count = Counter(
            'ptaas_scans_total',
            'Total PTaaS scans',
            ['scan_profile', 'status'],
            registry=self.registry
        )
        
        self.vulnerability_count = Counter(
            'vulnerabilities_found_total',
            'Total vulnerabilities found',
            ['severity', 'scanner'],
            registry=self.registry
        )
    
    def _setup_default_alerts(self):
        """Setup default alert conditions"""
        
        default_alerts = [
            AlertCondition(
                metric_name="cpu_percent",
                threshold=80.0,
                comparison="gt",
                duration_seconds=300,  # 5 minutes
                severity="warning",
                message_template="High CPU usage detected: {value}%"
            ),
            AlertCondition(
                metric_name="memory_percent", 
                threshold=85.0,
                comparison="gt",
                duration_seconds=300,
                severity="warning",
                message_template="High memory usage detected: {value}%"
            ),
            AlertCondition(
                metric_name="response_time_p95",
                threshold=2.0,
                comparison="gt", 
                duration_seconds=180,  # 3 minutes
                severity="warning",
                message_template="High response time detected: {value}s"
            ),
            AlertCondition(
                metric_name="error_rate",
                threshold=5.0,
                comparison="gt",
                duration_seconds=120,  # 2 minutes
                severity="critical",
                message_template="High error rate detected: {value}%"
            ),
            AlertCondition(
                metric_name="cache_hit_rate",
                threshold=80.0,
                comparison="lt",
                duration_seconds=600,  # 10 minutes
                severity="warning",
                message_template="Low cache hit rate detected: {value}%"
            )
        ]
        
        self.alert_conditions.extend(default_alerts)
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        
        if self.is_monitoring:
            logger.warning("Performance monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        
        self.is_monitoring = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        
        try:
            while self.is_monitoring:
                # Collect metrics
                metrics = await self._collect_metrics()
                
                # Store metrics
                self.metrics_history.append(metrics)
                
                # Update Prometheus metrics
                if self.enable_prometheus:
                    self._update_prometheus_metrics(metrics)
                
                # Store in Redis for persistence
                if self.redis_client:
                    await self._store_metrics_redis(metrics)
                
                # Check alert conditions
                await self._check_alerts(metrics)
                
                # Sleep until next collection
                await asyncio.sleep(self.collection_interval)
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
    
    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics"""
        
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if self.last_disk_stats:
                disk_read_mb = (disk_io.read_bytes - self.last_disk_stats.read_bytes) / 1024 / 1024
                disk_write_mb = (disk_io.write_bytes - self.last_disk_stats.write_bytes) / 1024 / 1024
            else:
                disk_read_mb = 0
                disk_write_mb = 0
            self.last_disk_stats = disk_io
            
            # Network I/O
            network_io = psutil.net_io_counters()
            if self.last_network_stats:
                network_sent_mb = (network_io.bytes_sent - self.last_network_stats.bytes_sent) / 1024 / 1024
                network_recv_mb = (network_io.bytes_recv - self.last_network_stats.bytes_recv) / 1024 / 1024
            else:
                network_sent_mb = 0
                network_recv_mb = 0
            self.last_network_stats = network_io
            
            # Network connections
            active_connections = len(psutil.net_connections())
            
            # Application metrics
            request_rate = self._calculate_request_rate()
            response_time_p95, response_time_p99 = self._calculate_response_times()
            error_rate = self._calculate_error_rate()
            database_connections = await self._get_database_connections()
            cache_hit_rate = await self._get_cache_hit_rate()
            
            return PerformanceMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                active_connections=active_connections,
                request_rate=request_rate,
                response_time_p95=response_time_p95,
                response_time_p99=response_time_p99,
                error_rate=error_rate,
                database_connections=database_connections,
                cache_hit_rate=cache_hit_rate
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            # Return empty metrics on error
            return PerformanceMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=0, memory_percent=0, memory_used_mb=0,
                disk_io_read_mb=0, disk_io_write_mb=0,
                network_sent_mb=0, network_recv_mb=0,
                active_connections=0, request_rate=0,
                response_time_p95=0, response_time_p99=0,
                error_rate=0, database_connections=0, cache_hit_rate=0
            )
    
    def _calculate_request_rate(self) -> float:
        """Calculate requests per second"""
        
        if not self.request_timings:
            return 0.0
        
        # Count requests in last 60 seconds
        now = time.time()
        total_requests = 0
        
        for endpoint_timings in self.request_timings.values():
            recent_requests = sum(1 for timing in endpoint_timings if now - timing < 60)
            total_requests += recent_requests
        
        return total_requests / 60.0
    
    def _calculate_response_times(self) -> tuple[float, float]:
        """Calculate 95th and 99th percentile response times"""
        
        if not self.request_timings:
            return 0.0, 0.0
        
        # Collect all recent response times
        now = time.time()
        all_times = []
        
        for endpoint_timings in self.request_timings.values():
            recent_times = [timing for timing in endpoint_timings if now - timing < 300]  # Last 5 minutes
            all_times.extend(recent_times)
        
        if not all_times:
            return 0.0, 0.0
        
        all_times.sort()
        n = len(all_times)
        
        p95_index = int(0.95 * n)
        p99_index = int(0.99 * n)
        
        p95 = all_times[p95_index] if p95_index < n else all_times[-1]
        p99 = all_times[p99_index] if p99_index < n else all_times[-1]
        
        return p95, p99
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate percentage"""
        
        total_requests = sum(len(timings) for timings in self.request_timings.values())
        total_errors = sum(self.error_counts.values())
        
        if total_requests == 0:
            return 0.0
        
        return (total_errors / total_requests) * 100.0
    
    async def _get_database_connections(self) -> int:
        """Get active database connections count"""
        # This would integrate with your database monitoring
        # For now, return a placeholder
        return 10
    
    async def _get_cache_hit_rate(self) -> float:
        """Get cache hit rate percentage"""
        
        if not self.redis_client:
            return 0.0
        
        try:
            info = await self.redis_client.info()
            hits = info.get('keyspace_hits', 0)
            misses = info.get('keyspace_misses', 0)
            
            if hits + misses == 0:
                return 0.0
            
            return (hits / (hits + misses)) * 100.0
            
        except Exception as e:
            logger.error(f"Error getting cache hit rate: {e}")
            return 0.0
    
    def _update_prometheus_metrics(self, metrics: PerformanceMetrics):
        """Update Prometheus metrics"""
        
        if not self.enable_prometheus:
            return
        
        try:
            # Update system metrics
            self.cpu_usage_gauge.set(metrics.cpu_percent)
            self.memory_usage_gauge.set(metrics.memory_percent)
            self.memory_used_gauge.set(metrics.memory_used_mb)
            self.active_connections_gauge.set(metrics.active_connections)
            self.database_connections_gauge.set(metrics.database_connections)
            self.cache_hit_rate_gauge.set(metrics.cache_hit_rate)
            
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")
    
    async def _store_metrics_redis(self, metrics: PerformanceMetrics):
        """Store metrics in Redis for persistence"""
        
        if not self.redis_client:
            return
        
        try:
            # Store with timestamp as key
            key = f"metrics:{metrics.timestamp.isoformat()}"
            value = json.dumps(asdict(metrics), default=str)
            
            # Set with expiration
            await self.redis_client.setex(key, self.retention_period, value)
            
            # Also store in a sorted set for time-based queries
            score = metrics.timestamp.timestamp()
            await self.redis_client.zadd("metrics:timeline", {key: score})
            
            # Clean up old metrics
            cutoff = datetime.utcnow() - timedelta(seconds=self.retention_period)
            await self.redis_client.zremrangebyscore("metrics:timeline", 0, cutoff.timestamp())
            
        except Exception as e:
            logger.error(f"Error storing metrics in Redis: {e}")
    
    async def _check_alerts(self, metrics: PerformanceMetrics):
        """Check alert conditions and trigger alerts"""
        
        for condition in self.alert_conditions:
            try:
                # Get metric value
                metric_value = getattr(metrics, condition.metric_name, None)
                if metric_value is None:
                    continue
                
                # Check condition
                should_alert = False
                if condition.comparison == "gt" and metric_value > condition.threshold:
                    should_alert = True
                elif condition.comparison == "lt" and metric_value < condition.threshold:
                    should_alert = True
                elif condition.comparison == "eq" and metric_value == condition.threshold:
                    should_alert = True
                
                if should_alert:
                    await self._trigger_alert(condition, metric_value, metrics.timestamp)
                    
            except Exception as e:
                logger.error(f"Error checking alert condition {condition.metric_name}: {e}")
    
    async def _trigger_alert(self, condition: AlertCondition, value: float, timestamp: datetime):
        """Trigger an alert"""
        
        alert_data = {
            "metric": condition.metric_name,
            "value": value,
            "threshold": condition.threshold,
            "severity": condition.severity,
            "message": condition.message_template.format(value=value),
            "timestamp": timestamp.isoformat()
        }
        
        logger.warning(f"ALERT: {alert_data['message']}")
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert_data)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    @asynccontextmanager
    async def track_request(self, endpoint: str, method: str = "GET"):
        """Context manager to track request performance"""
        
        start_time = time.time()
        
        try:
            yield
            
            # Record successful request
            duration = time.time() - start_time
            self.request_timings[f"{method}:{endpoint}"].append(duration)
            
            # Update Prometheus metrics
            if self.enable_prometheus:
                self.request_count.labels(method=method, endpoint=endpoint, status="200").inc()
                self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
                
        except Exception as e:
            # Record error
            self.error_counts[f"{method}:{endpoint}"] += 1
            
            # Update Prometheus metrics
            if self.enable_prometheus:
                self.request_count.labels(method=method, endpoint=endpoint, status="500").inc()
            
            raise
    
    def track_scan_performance(self, scan_profile: str, duration: float, status: str):
        """Track PTaaS scan performance"""
        
        if self.enable_prometheus:
            self.scan_duration.labels(scan_profile=scan_profile, status=status).observe(duration)
            self.scan_count.labels(scan_profile=scan_profile, status=status).inc()
    
    def track_vulnerability(self, severity: str, scanner: str):
        """Track vulnerability detection"""
        
        if self.enable_prometheus:
            self.vulnerability_count.labels(severity=severity, scanner=scanner).inc()
    
    def add_alert_handler(self, handler: Callable):
        """Add custom alert handler"""
        self.alert_handlers.append(handler)
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        
        if not self.enable_prometheus:
            return ""
        
        return generate_latest(self.registry).decode('utf-8')
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get most recent metrics"""
        
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self, minutes: int = 60) -> List[PerformanceMetrics]:
        """Get metrics history for specified time period"""
        
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        return [m for m in self.metrics_history if m.timestamp >= cutoff]
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        current = self.get_current_metrics()
        history = self.get_metrics_history(60)  # Last hour
        
        if not current or not history:
            return {"status": "no_data"}
        
        # Calculate averages and trends
        avg_cpu = sum(m.cpu_percent for m in history) / len(history)
        avg_memory = sum(m.memory_percent for m in history) / len(history)
        avg_response_time = sum(m.response_time_p95 for m in history) / len(history)
        
        # Calculate trends (simple slope)
        cpu_trend = "stable"
        if len(history) > 1:
            cpu_slope = (history[-1].cpu_percent - history[0].cpu_percent) / len(history)
            if cpu_slope > 1:
                cpu_trend = "increasing"
            elif cpu_slope < -1:
                cpu_trend = "decreasing"
        
        return {
            "timestamp": current.timestamp.isoformat(),
            "current": asdict(current),
            "averages": {
                "cpu_percent": round(avg_cpu, 2),
                "memory_percent": round(avg_memory, 2),
                "response_time_p95": round(avg_response_time, 3)
            },
            "trends": {
                "cpu": cpu_trend
            },
            "recommendations": self._generate_recommendations(current, history)
        }
    
    def _generate_recommendations(
        self, 
        current: PerformanceMetrics, 
        history: List[PerformanceMetrics]
    ) -> List[str]:
        """Generate performance optimization recommendations"""
        
        recommendations = []
        
        # High CPU recommendations
        if current.cpu_percent > 80:
            recommendations.append("Consider horizontal scaling or CPU optimization")
        
        # High memory recommendations  
        if current.memory_percent > 85:
            recommendations.append("Memory usage is high - check for memory leaks or increase memory allocation")
        
        # Slow response time recommendations
        if current.response_time_p95 > 1.0:
            recommendations.append("Response times are slow - consider database query optimization or caching")
        
        # Low cache hit rate recommendations
        if current.cache_hit_rate < 80:
            recommendations.append("Low cache hit rate - review caching strategy and TTL settings")
        
        # High error rate recommendations
        if current.error_rate > 2:
            recommendations.append("High error rate detected - check application logs and error handling")
        
        return recommendations