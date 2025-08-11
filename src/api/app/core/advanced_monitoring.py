"""
Advanced Monitoring and Observability System - Principal Auditor Enhanced
Comprehensive monitoring with predictive analytics and automated alerting
"""

import asyncio
import json
import logging
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import statistics

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    prometheus_client = None
    PROMETHEUS_AVAILABLE = False

try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    aioredis = None
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Alert:
    """Alert definition"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    metric: str
    threshold: float
    condition: str  # "gt", "lt", "eq", "gte", "lte"
    duration: int = 60  # seconds
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class MetricDefinition:
    """Metric definition"""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: List[float] = field(default_factory=list)  # For histograms
    quantiles: List[float] = field(default_factory=list)  # For summaries


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    load_average: List[float]
    open_files: int = 0
    tcp_connections: int = 0


@dataclass
class ServiceMetrics:
    """Service-specific metrics"""
    service_name: str
    timestamp: datetime
    request_count: int
    error_count: int
    response_time_avg: float
    response_time_p95: float
    response_time_p99: float
    active_connections: int
    queue_size: int = 0
    health_score: float = 100.0


class MetricsCollector:
    """
    Advanced Metrics Collector with Prometheus Integration
    """
    
    def __init__(self, registry: Optional[prometheus_client.CollectorRegistry] = None):
        self.registry = registry or prometheus_client.REGISTRY
        self.metrics: Dict[str, Any] = {}
        self.custom_metrics: Dict[str, MetricDefinition] = {}
        self.system_metrics_history: deque = deque(maxlen=1000)
        self.service_metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._initialize_default_metrics()
    
    def _initialize_default_metrics(self):
        """Initialize default application metrics"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, using fallback metrics")
            return
        
        # Request metrics
        self.metrics["http_requests_total"] = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
            registry=self.registry
        )
        
        self.metrics["http_request_duration_seconds"] = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            registry=self.registry
        )
        
        # System metrics
        self.metrics["system_cpu_usage"] = Gauge(
            "system_cpu_usage",
            "System CPU usage percentage",
            registry=self.registry
        )
        
        self.metrics["system_memory_usage"] = Gauge(
            "system_memory_usage",
            "System memory usage percentage",
            registry=self.registry
        )
        
        # PTaaS metrics
        self.metrics["ptaas_scans_total"] = Counter(
            "ptaas_scans_total",
            "Total PTaaS scans executed",
            ["scan_type", "status"],
            registry=self.registry
        )
        
        self.metrics["ptaas_scan_duration_seconds"] = Histogram(
            "ptaas_scan_duration_seconds",
            "PTaaS scan duration in seconds",
            ["scan_type"],
            registry=self.registry
        )
        
        self.metrics["ptaas_vulnerabilities_found"] = Counter(
            "ptaas_vulnerabilities_found",
            "Total vulnerabilities found",
            ["severity", "scan_type"],
            registry=self.registry
        )
        
        # Security metrics
        self.metrics["security_events_total"] = Counter(
            "security_events_total",
            "Total security events",
            ["event_type", "severity"],
            registry=self.registry
        )
        
        self.metrics["authentication_attempts_total"] = Counter(
            "authentication_attempts_total",
            "Total authentication attempts",
            ["result", "method"],
            registry=self.registry
        )
        
        # Database metrics
        self.metrics["database_connections_active"] = Gauge(
            "database_connections_active",
            "Active database connections",
            registry=self.registry
        )
        
        self.metrics["database_query_duration_seconds"] = Histogram(
            "database_query_duration_seconds",
            "Database query duration in seconds",
            ["operation"],
            registry=self.registry
        )
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        if PROMETHEUS_AVAILABLE and "http_requests_total" in self.metrics:
            self.metrics["http_requests_total"].labels(
                method=method, endpoint=endpoint, status_code=str(status_code)
            ).inc()
            
            self.metrics["http_request_duration_seconds"].labels(
                method=method, endpoint=endpoint
            ).observe(duration)
    
    def record_ptaas_scan(self, scan_type: str, status: str, duration: float, vulnerabilities: Dict[str, int]):
        """Record PTaaS scan metrics"""
        if PROMETHEUS_AVAILABLE:
            # Record scan completion
            self.metrics["ptaas_scans_total"].labels(
                scan_type=scan_type, status=status
            ).inc()
            
            # Record scan duration
            self.metrics["ptaas_scan_duration_seconds"].labels(
                scan_type=scan_type
            ).observe(duration)
            
            # Record vulnerabilities found
            for severity, count in vulnerabilities.items():
                self.metrics["ptaas_vulnerabilities_found"].labels(
                    severity=severity, scan_type=scan_type
                ).inc(count)
    
    def record_security_event(self, event_type: str, severity: str):
        """Record security event"""
        if PROMETHEUS_AVAILABLE and "security_events_total" in self.metrics:
            self.metrics["security_events_total"].labels(
                event_type=event_type, severity=severity
            ).inc()
    
    def record_authentication_attempt(self, result: str, method: str):
        """Record authentication attempt"""
        if PROMETHEUS_AVAILABLE and "authentication_attempts_total" in self.metrics:
            self.metrics["authentication_attempts_total"].labels(
                result=result, method=method
            ).inc()
    
    def update_system_metrics(self):
        """Update system performance metrics"""
        try:
            # Collect system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            system_metrics = SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=(disk.used / disk.total) * 100,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                process_count=len(psutil.pids()),
                load_average=list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            )
            
            # Store in history
            self.system_metrics_history.append(system_metrics)
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                self.metrics["system_cpu_usage"].set(cpu_percent)
                self.metrics["system_memory_usage"].set(memory.percent)
            
            return system_metrics
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
            return None
    
    def get_metric_value(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current metric value"""
        if metric_name in self.metrics:
            metric = self.metrics[metric_name]
            if hasattr(metric, '_value'):
                return metric._value._value
            elif hasattr(metric, 'collect'):
                # For more complex metrics, collect samples
                samples = list(metric.collect())[0].samples
                if samples:
                    return samples[0].value
        return None


class AlertManager:
    """
    Advanced Alert Management System
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.notification_handlers: Dict[str, Callable] = {}
        self.evaluation_interval = 30  # seconds
        self.running = False
        self.evaluation_task: Optional[asyncio.Task] = None
        self._initialize_default_alerts()
    
    def _initialize_default_alerts(self):
        """Initialize default system alerts"""
        default_alerts = [
            Alert(
                id="high_cpu_usage",
                name="High CPU Usage",
                description="System CPU usage is above 80%",
                severity=AlertSeverity.WARNING,
                metric="system_cpu_usage",
                threshold=80.0,
                condition="gte",
                duration=120,
                notification_channels=["email", "slack"]
            ),
            Alert(
                id="high_memory_usage",
                name="High Memory Usage",
                description="System memory usage is above 90%",
                severity=AlertSeverity.CRITICAL,
                metric="system_memory_usage",
                threshold=90.0,
                condition="gte",
                duration=60,
                notification_channels=["email", "slack", "webhook"]
            ),
            Alert(
                id="high_error_rate",
                name="High Error Rate",
                description="HTTP error rate is above 5%",
                severity=AlertSeverity.ERROR,
                metric="http_error_rate",
                threshold=5.0,
                condition="gte",
                duration=300,
                notification_channels=["email", "slack"]
            ),
            Alert(
                id="security_incident",
                name="Security Incident Detected",
                description="Multiple failed authentication attempts",
                severity=AlertSeverity.CRITICAL,
                metric="failed_auth_rate",
                threshold=10.0,
                condition="gte",
                duration=60,
                notification_channels=["email", "slack", "webhook", "sms"]
            )
        ]
        
        for alert in default_alerts:
            self.alerts[alert.id] = alert
    
    def add_alert(self, alert: Alert):
        """Add a new alert"""
        self.alerts[alert.id] = alert
        logger.info(f"Added alert: {alert.name}")
    
    def remove_alert(self, alert_id: str):
        """Remove an alert"""
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            logger.info(f"Removed alert: {alert_id}")
    
    def add_notification_handler(self, channel: str, handler: Callable):
        """Add notification handler for a channel"""
        self.notification_handlers[channel] = handler
        logger.info(f"Added notification handler for channel: {channel}")
    
    async def start_evaluation(self):
        """Start alert evaluation loop"""
        if self.running:
            return
        
        self.running = True
        self.evaluation_task = asyncio.create_task(self._evaluation_loop())
        logger.info("Alert evaluation started")
    
    async def stop_evaluation(self):
        """Stop alert evaluation loop"""
        self.running = False
        if self.evaluation_task:
            self.evaluation_task.cancel()
            try:
                await self.evaluation_task
            except asyncio.CancelledError:
                pass
        logger.info("Alert evaluation stopped")
    
    async def _evaluation_loop(self):
        """Main alert evaluation loop"""
        while self.running:
            try:
                await self._evaluate_alerts()
                await asyncio.sleep(self.evaluation_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert evaluation: {e}")
                await asyncio.sleep(self.evaluation_interval)
    
    async def _evaluate_alerts(self):
        """Evaluate all alerts"""
        for alert in self.alerts.values():
            if not alert.enabled:
                continue
            
            try:
                await self._evaluate_alert(alert)
            except Exception as e:
                logger.error(f"Error evaluating alert {alert.id}: {e}")
    
    async def _evaluate_alert(self, alert: Alert):
        """Evaluate a single alert"""
        # Get metric value
        metric_value = self._get_metric_value_for_alert(alert)
        if metric_value is None:
            return
        
        # Check condition
        triggered = self._check_alert_condition(alert, metric_value)
        
        if triggered:
            # Check if alert should fire (duration check)
            now = datetime.utcnow()
            if (alert.last_triggered is None or 
                (now - alert.last_triggered).total_seconds() >= alert.duration):
                
                await self._fire_alert(alert, metric_value)
    
    def _get_metric_value_for_alert(self, alert: Alert) -> Optional[float]:
        """Get metric value for alert evaluation"""
        if alert.metric == "http_error_rate":
            # Calculate error rate from request metrics
            return self._calculate_error_rate()
        elif alert.metric == "failed_auth_rate":
            # Calculate failed authentication rate
            return self._calculate_failed_auth_rate()
        else:
            # Get direct metric value
            return self.metrics_collector.get_metric_value(alert.metric)
    
    def _calculate_error_rate(self) -> float:
        """Calculate HTTP error rate"""
        # This would calculate error rate based on request metrics
        # For now, return a mock value
        return 0.0
    
    def _calculate_failed_auth_rate(self) -> float:
        """Calculate failed authentication rate"""
        # This would calculate failed auth rate
        # For now, return a mock value
        return 0.0
    
    def _check_alert_condition(self, alert: Alert, value: float) -> bool:
        """Check if alert condition is met"""
        if alert.condition == "gt":
            return value > alert.threshold
        elif alert.condition == "gte":
            return value >= alert.threshold
        elif alert.condition == "lt":
            return value < alert.threshold
        elif alert.condition == "lte":
            return value <= alert.threshold
        elif alert.condition == "eq":
            return value == alert.threshold
        return False
    
    async def _fire_alert(self, alert: Alert, value: float):
        """Fire an alert"""
        alert.last_triggered = datetime.utcnow()
        alert.trigger_count += 1
        
        alert_event = {
            "alert_id": alert.id,
            "alert_name": alert.name,
            "severity": alert.severity.value,
            "description": alert.description,
            "metric": alert.metric,
            "threshold": alert.threshold,
            "current_value": value,
            "timestamp": alert.last_triggered.isoformat(),
            "trigger_count": alert.trigger_count
        }
        
        # Store in history
        self.alert_history.append(alert_event)
        
        # Send notifications
        for channel in alert.notification_channels:
            if channel in self.notification_handlers:
                try:
                    await self.notification_handlers[channel](alert_event)
                except Exception as e:
                    logger.error(f"Failed to send alert to {channel}: {e}")
        
        logger.warning(f"Alert fired: {alert.name} (value: {value}, threshold: {alert.threshold})")


class AdvancedMonitoringSystem:
    """
    Comprehensive Monitoring System - Principal Auditor Enhanced
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.redis_client: Optional[aioredis.Redis] = None
        self.monitoring_tasks: List[asyncio.Task] = []
        self.running = False
        
        # Performance tracking
        self.performance_baselines: Dict[str, float] = {}
        self.anomaly_detection_enabled = True
        
        # Setup notification handlers
        self._setup_notification_handlers()
    
    def _setup_notification_handlers(self):
        """Setup notification handlers"""
        async def log_notification(alert_event: Dict[str, Any]):
            logger.warning(f"ALERT: {alert_event['alert_name']} - {alert_event['description']}")
        
        self.alert_manager.add_notification_handler("log", log_notification)
    
    async def initialize(self, redis_url: Optional[str] = None):
        """Initialize monitoring system"""
        try:
            # Initialize Redis connection if available
            if redis_url and REDIS_AVAILABLE:
                self.redis_client = aioredis.from_url(redis_url)
                logger.info("Redis connection established for monitoring")
            
            # Start monitoring tasks
            await self.start_monitoring()
            
            logger.info("Advanced monitoring system initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring system: {e}")
            raise
    
    async def start_monitoring(self):
        """Start all monitoring tasks"""
        if self.running:
            return
        
        self.running = True
        
        # Start alert evaluation
        await self.alert_manager.start_evaluation()
        
        # Start system metrics collection
        self.monitoring_tasks.append(
            asyncio.create_task(self._system_metrics_loop())
        )
        
        # Start performance baseline calculation
        self.monitoring_tasks.append(
            asyncio.create_task(self._performance_baseline_loop())
        )
        
        logger.info("Monitoring tasks started")
    
    async def stop_monitoring(self):
        """Stop all monitoring tasks"""
        self.running = False
        
        # Stop alert evaluation
        await self.alert_manager.stop_evaluation()
        
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Monitoring stopped")
    
    async def _system_metrics_loop(self):
        """System metrics collection loop"""
        while self.running:
            try:
                # Update system metrics
                system_metrics = self.metrics_collector.update_system_metrics()
                
                # Store metrics in Redis if available
                if self.redis_client and system_metrics:
                    await self._store_metrics_in_redis(system_metrics)
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system metrics loop: {e}")
                await asyncio.sleep(30)
    
    async def _performance_baseline_loop(self):
        """Performance baseline calculation loop"""
        while self.running:
            try:
                await self._calculate_performance_baselines()
                await asyncio.sleep(3600)  # Calculate every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance baseline loop: {e}")
                await asyncio.sleep(3600)
    
    async def _store_metrics_in_redis(self, metrics: SystemMetrics):
        """Store metrics in Redis for persistence"""
        try:
            metrics_data = asdict(metrics)
            metrics_data["timestamp"] = metrics.timestamp.isoformat()
            
            # Store current metrics
            await self.redis_client.set(
                "xorb:metrics:current",
                json.dumps(metrics_data),
                ex=3600  # Expire after 1 hour
            )
            
            # Store in time series (keep last 24 hours)
            timestamp_key = int(metrics.timestamp.timestamp())
            await self.redis_client.zadd(
                "xorb:metrics:timeseries",
                {json.dumps(metrics_data): timestamp_key}
            )
            
            # Clean old entries (older than 24 hours)
            cutoff_time = timestamp_key - 86400
            await self.redis_client.zremrangebyscore(
                "xorb:metrics:timeseries", 0, cutoff_time
            )
            
        except Exception as e:
            logger.error(f"Failed to store metrics in Redis: {e}")
    
    async def _calculate_performance_baselines(self):
        """Calculate performance baselines for anomaly detection"""
        try:
            if len(self.metrics_collector.system_metrics_history) < 10:
                return  # Need more data points
            
            # Calculate baselines from recent metrics
            recent_metrics = list(self.metrics_collector.system_metrics_history)[-100:]
            
            self.performance_baselines = {
                "cpu_percent": statistics.mean([m.cpu_percent for m in recent_metrics]),
                "memory_percent": statistics.mean([m.memory_percent for m in recent_metrics]),
                "disk_percent": statistics.mean([m.disk_percent for m in recent_metrics]),
                "process_count": statistics.mean([m.process_count for m in recent_metrics])
            }
            
            logger.debug(f"Updated performance baselines: {self.performance_baselines}")
            
        except Exception as e:
            logger.error(f"Failed to calculate performance baselines: {e}")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        current_metrics = None
        if self.metrics_collector.system_metrics_history:
            current_metrics = self.metrics_collector.system_metrics_history[-1]
        
        return {
            "monitoring_status": "active" if self.running else "inactive",
            "current_metrics": asdict(current_metrics) if current_metrics else None,
            "alert_count": len(self.alert_manager.alerts),
            "active_alerts": len([a for a in self.alert_manager.alerts.values() if a.enabled]),
            "recent_alert_count": len([
                a for a in self.alert_manager.alert_history 
                if datetime.fromisoformat(a["timestamp"]) > datetime.utcnow() - timedelta(hours=1)
            ]),
            "metrics_history_size": len(self.metrics_collector.system_metrics_history),
            "performance_baselines": self.performance_baselines,
            "redis_connected": self.redis_client is not None
        }
    
    async def export_metrics(self, format: str = "prometheus") -> str:
        """Export metrics in specified format"""
        if format == "prometheus" and PROMETHEUS_AVAILABLE:
            return prometheus_client.generate_latest(self.metrics_collector.registry).decode()
        elif format == "json":
            health_summary = self.get_health_summary()
            return json.dumps(health_summary, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global monitoring system instance
_monitoring_system = None


def get_monitoring_system() -> AdvancedMonitoringSystem:
    """Get the global monitoring system instance"""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = AdvancedMonitoringSystem()
    return _monitoring_system


async def initialize_monitoring(config: Optional[Dict[str, Any]] = None) -> AdvancedMonitoringSystem:
    """Initialize the global monitoring system"""
    global _monitoring_system
    _monitoring_system = AdvancedMonitoringSystem(config)
    await _monitoring_system.initialize()
    return _monitoring_system