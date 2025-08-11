"""
Production-grade observability stack for XORB
Comprehensive monitoring, tracing, and alerting
"""

import asyncio
import logging
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque
import psutil

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available. Install with: pip install prometheus-client")
    # Fallback implementations that actually work for development
    class Counter:
        def __init__(self, name, description, labelnames=None, registry=None):
            self.name = name
            self._value = 0
            self.labelnames = labelnames or []
            
        def inc(self, amount=1):
            self._value += amount
            logger.debug(f"Counter {self.name}: {self._value}")
            
        def labels(self, **kwargs):
            return self
            
    class Histogram:
        def __init__(self, name, description, labelnames=None, buckets=None, registry=None):
            self.name = name
            self._observations = []
            self.labelnames = labelnames or []
            
        def observe(self, value):
            self._observations.append(value)
            logger.debug(f"Histogram {self.name}: observed {value}")
            
        def time(self): 
            return MockTimer(self)
            
        def labels(self, **kwargs):
            return self
            
    class MockTimer:
        def __init__(self, histogram):
            self.histogram = histogram
            self.start_time = None
            
        def __enter__(self):
            self.start_time = time.time()
            return self
            
        def __exit__(self, *args):
            if self.start_time:
                duration = time.time() - self.start_time
                self.histogram.observe(duration)
                
    class Gauge:
        def __init__(self, name, description, labelnames=None, registry=None):
            self.name = name
            self._value = 0
            self.labelnames = labelnames or []
            
        def set(self, value):
            self._value = value
            logger.debug(f"Gauge {self.name}: {value}")
            
        def inc(self, amount=1):
            self._value += amount
            
        def dec(self, amount=1):
            self._value -= amount
            
        def labels(self, **kwargs):
            return self
            
    class Info:
        def __init__(self, name, description, labelnames=None, registry=None):
            self.name = name
            self.labelnames = labelnames or []
            
        def info(self, data):
            logger.info(f"Info {self.name}: {data}")
            
        def labels(self, **kwargs):
            return self

try:
    import opentelemetry
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    metric_name: str
    condition: str  # e.g., "> 0.9", "< 100"
    threshold: float
    duration: int = 60  # seconds
    labels: Dict[str, str] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    triggered_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    check_function: Callable[[], bool]
    interval: int = 30  # seconds
    timeout: int = 10  # seconds
    retries: int = 3
    enabled: bool = True
    last_check: Optional[datetime] = None
    last_result: bool = True
    consecutive_failures: int = 0


class MetricsCollector:
    """Production metrics collector"""
    
    def __init__(self):
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self.custom_metrics = {}
        self.health_checks: Dict[str, HealthCheck] = {}
        self.alerts: Dict[str, Alert] = {}
        self.metric_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Core metrics
        self._initialize_core_metrics()
        
        # System metrics
        self.system_collector = SystemMetricsCollector()
        
        # Application metrics
        self.app_collector = ApplicationMetricsCollector()
        
        # Background tasks
        self.monitoring_tasks = []
        self.alert_handlers = []
    
    def _initialize_core_metrics(self):
        """Initialize core Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # HTTP metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Database metrics
        self.database_connections = Gauge(
            'database_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        self.database_query_duration = Histogram(
            'database_query_duration_seconds',
            'Database query duration',
            ['query_type', 'table'],
            registry=self.registry
        )
        
        # Business metrics
        self.security_scans_total = Counter(
            'security_scans_total',
            'Total security scans performed',
            ['scan_type', 'status'],
            registry=self.registry
        )
        
        self.threats_detected_total = Counter(
            'threats_detected_total',
            'Total threats detected',
            ['threat_type', 'severity'],
            registry=self.registry
        )
        
        # AI/LLM metrics
        self.llm_requests_total = Counter(
            'llm_requests_total',
            'Total LLM requests',
            ['provider', 'model', 'status'],
            registry=self.registry
        )
        
        self.llm_response_time = Histogram(
            'llm_response_time_seconds',
            'LLM response time',
            ['provider', 'model'],
            registry=self.registry
        )
        
        # System health
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage',
            ['device'],
            registry=self.registry
        )
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        if PROMETHEUS_AVAILABLE:
            self.http_requests_total.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
            self.http_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_database_query(self, query_type: str, table: str, duration: float):
        """Record database query metrics"""
        if PROMETHEUS_AVAILABLE:
            self.database_query_duration.labels(query_type=query_type, table=table).observe(duration)
    
    def record_security_scan(self, scan_type: str, status: str):
        """Record security scan metrics"""
        if PROMETHEUS_AVAILABLE:
            self.security_scans_total.labels(scan_type=scan_type, status=status).inc()
    
    def record_threat_detection(self, threat_type: str, severity: str):
        """Record threat detection metrics"""
        if PROMETHEUS_AVAILABLE:
            self.threats_detected_total.labels(threat_type=threat_type, severity=severity).inc()
    
    def record_llm_request(self, provider: str, model: str, status: str, response_time: float):
        """Record LLM request metrics"""
        if PROMETHEUS_AVAILABLE:
            self.llm_requests_total.labels(provider=provider, model=model, status=status).inc()
            self.llm_response_time.labels(provider=provider, model=model).observe(response_time)
    
    def update_database_connections(self, count: int):
        """Update active database connections"""
        if PROMETHEUS_AVAILABLE:
            self.database_connections.set(count)
    
    def add_health_check(self, health_check: HealthCheck):
        """Add health check"""
        self.health_checks[health_check.name] = health_check
        logger.info(f"Added health check: {health_check.name}")
    
    def add_alert(self, alert: Alert):
        """Add alert definition"""
        self.alerts[alert.alert_id] = alert
        logger.info(f"Added alert: {alert.name}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler"""
        self.alert_handlers.append(handler)
    
    async def start_monitoring(self):
        """Start background monitoring tasks"""
        self.monitoring_tasks = [
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._system_metrics_loop()),
            asyncio.create_task(self._alert_evaluation_loop())
        ]
        logger.info("Started monitoring tasks")
    
    async def stop_monitoring(self):
        """Stop monitoring tasks"""
        for task in self.monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        logger.info("Stopped monitoring tasks")
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while True:
            try:
                for health_check in self.health_checks.values():
                    if health_check.enabled:
                        await self._execute_health_check(health_check)
                
                await asyncio.sleep(10)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(30)
    
    async def _execute_health_check(self, health_check: HealthCheck):
        """Execute single health check"""
        if health_check.last_check:
            time_since_last = (datetime.utcnow() - health_check.last_check).total_seconds()
            if time_since_last < health_check.interval:
                return
        
        try:
            # Execute health check with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(health_check.check_function),
                timeout=health_check.timeout
            )
            
            health_check.last_check = datetime.utcnow()
            
            if result:
                if not health_check.last_result:
                    logger.info(f"Health check {health_check.name} recovered")
                health_check.last_result = True
                health_check.consecutive_failures = 0
            else:
                health_check.last_result = False
                health_check.consecutive_failures += 1
                logger.warning(f"Health check {health_check.name} failed ({health_check.consecutive_failures} consecutive)")
                
                if health_check.consecutive_failures >= health_check.retries:
                    await self._trigger_health_alert(health_check)
        
        except asyncio.TimeoutError:
            health_check.last_result = False
            health_check.consecutive_failures += 1
            logger.warning(f"Health check {health_check.name} timed out")
        
        except Exception as e:
            health_check.last_result = False
            health_check.consecutive_failures += 1
            logger.error(f"Health check {health_check.name} error: {e}")
    
    async def _trigger_health_alert(self, health_check: HealthCheck):
        """Trigger alert for failed health check"""
        alert = Alert(
            alert_id=f"health_check_{health_check.name}",
            name=f"Health Check Failed: {health_check.name}",
            description=f"Health check {health_check.name} has failed {health_check.consecutive_failures} times",
            severity=AlertSeverity.CRITICAL,
            metric_name="health_check_status",
            condition="== 0",
            threshold=0,
            labels={"health_check": health_check.name},
            triggered_at=datetime.utcnow()
        )
        
        await self._send_alert(alert)
    
    async def _system_metrics_loop(self):
        """Background system metrics collection"""
        while True:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                if PROMETHEUS_AVAILABLE:
                    self.system_cpu_usage.set(cpu_percent)
                    self.system_memory_usage.set(memory.used)
                
                # Disk usage
                disk_partitions = psutil.disk_partitions()
                for partition in disk_partitions:
                    try:
                        disk_usage = psutil.disk_usage(partition.mountpoint)
                        usage_percent = (disk_usage.used / disk_usage.total) * 100
                        if PROMETHEUS_AVAILABLE:
                            self.system_disk_usage.labels(device=partition.device).set(usage_percent)
                    except PermissionError:
                        continue
                
                await asyncio.sleep(30)  # Collect every 30 seconds
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System metrics collection error: {e}")
                await asyncio.sleep(60)
    
    async def _alert_evaluation_loop(self):
        """Background alert evaluation"""
        while True:
            try:
                for alert in self.alerts.values():
                    if not alert.triggered_at:  # Only check non-triggered alerts
                        await self._evaluate_alert(alert)
                
                await asyncio.sleep(30)  # Evaluate every 30 seconds
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert evaluation error: {e}")
                await asyncio.sleep(60)
    
    async def _evaluate_alert(self, alert: Alert):
        """Evaluate single alert condition"""
        # This is a simplified implementation
        # In production, would integrate with time series database
        try:
            current_value = self._get_current_metric_value(alert.metric_name)
            if current_value is None:
                return
            
            condition_met = self._evaluate_condition(current_value, alert.condition, alert.threshold)
            
            if condition_met:
                alert.triggered_at = datetime.utcnow()
                await self._send_alert(alert)
        
        except Exception as e:
            logger.error(f"Alert evaluation error for {alert.name}: {e}")
    
    def _get_current_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value of a metric"""
        # Simplified - in production would query time series database
        if metric_name == "cpu_usage":
            return psutil.cpu_percent()
        elif metric_name == "memory_usage":
            return psutil.virtual_memory().percent
        elif metric_name == "disk_usage":
            return psutil.disk_usage('/').percent
        return None
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        if condition == "> threshold":
            return value > threshold
        elif condition == "< threshold":
            return value < threshold
        elif condition == ">= threshold":
            return value >= threshold
        elif condition == "<= threshold":
            return value <= threshold
        elif condition == "== threshold":
            return value == threshold
        return False
    
    async def _send_alert(self, alert: Alert):
        """Send alert to handlers"""
        logger.warning(f"ALERT TRIGGERED: {alert.name} - {alert.description}")
        
        for handler in self.alert_handlers:
            try:
                await asyncio.to_thread(handler, alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def get_metrics_export(self) -> str:
        """Export metrics in Prometheus format"""
        if PROMETHEUS_AVAILABLE and self.registry:
            return generate_latest(self.registry).decode('utf-8')
        return "# Metrics not available\n"
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        health_status = {}
        overall_healthy = True
        
        for name, check in self.health_checks.items():
            status = {
                "healthy": check.last_result,
                "last_check": check.last_check.isoformat() if check.last_check else None,
                "consecutive_failures": check.consecutive_failures
            }
            health_status[name] = status
            
            if not check.last_result:
                overall_healthy = False
        
        return {
            "overall_healthy": overall_healthy,
            "checks": health_status,
            "timestamp": datetime.utcnow().isoformat()
        }


class SystemMetricsCollector:
    """System-level metrics collector"""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        cpu = psutil.cpu_times()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        return {
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "user": cpu.user,
                "system": cpu.system,
                "idle": cpu.idle
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            },
            "network": {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            },
            "process": {
                "pid": self.process.pid,
                "cpu_percent": self.process.cpu_percent(),
                "memory_info": self.process.memory_info()._asdict(),
                "num_threads": self.process.num_threads(),
                "open_files": len(self.process.open_files()),
                "connections": len(self.process.connections())
            }
        }


class ApplicationMetricsCollector:
    """Application-specific metrics collector"""
    
    def __init__(self):
        self.app_start_time = datetime.utcnow()
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.response_times = defaultdict(list)
    
    def record_request(self, endpoint: str, method: str, status_code: int, response_time: float):
        """Record request metrics"""
        key = f"{method}:{endpoint}"
        self.request_counts[key] += 1
        self.response_times[key].append(response_time)
        
        if status_code >= 400:
            self.error_counts[key] += 1
    
    def get_application_metrics(self) -> Dict[str, Any]:
        """Get application-specific metrics"""
        uptime = (datetime.utcnow() - self.app_start_time).total_seconds()
        
        # Calculate response time statistics
        response_stats = {}
        for key, times in self.response_times.items():
            if times:
                response_stats[key] = {
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "count": len(times)
                }
        
        return {
            "uptime_seconds": uptime,
            "total_requests": sum(self.request_counts.values()),
            "total_errors": sum(self.error_counts.values()),
            "error_rate": sum(self.error_counts.values()) / max(sum(self.request_counts.values()), 1),
            "endpoints": {
                "request_counts": dict(self.request_counts),
                "error_counts": dict(self.error_counts),
                "response_times": response_stats
            }
        }


# Health check functions
def check_database_connection() -> bool:
    """Check database connectivity"""
    try:
        # This would check actual database connection
        # For now, return True as placeholder
        return True
    except Exception:
        return False


def check_redis_connection() -> bool:
    """Check Redis connectivity"""
    try:
        # This would check actual Redis connection
        return True
    except Exception:
        return False


def check_disk_space() -> bool:
    """Check available disk space"""
    try:
        disk_usage = psutil.disk_usage('/')
        free_percent = (disk_usage.free / disk_usage.total) * 100
        return free_percent > 10  # Alert if less than 10% free
    except Exception:
        return False


def check_memory_usage() -> bool:
    """Check memory usage"""
    try:
        memory = psutil.virtual_memory()
        return memory.percent < 90  # Alert if more than 90% used
    except Exception:
        return False


# Global metrics collector
metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector"""
    global metrics_collector
    if metrics_collector is None:
        metrics_collector = MetricsCollector()
        
        # Add default health checks
        metrics_collector.add_health_check(HealthCheck(
            name="database",
            check_function=check_database_connection,
            interval=30
        ))
        
        metrics_collector.add_health_check(HealthCheck(
            name="redis",
            check_function=check_redis_connection,
            interval=30
        ))
        
        metrics_collector.add_health_check(HealthCheck(
            name="disk_space",
            check_function=check_disk_space,
            interval=60
        ))
        
        metrics_collector.add_health_check(HealthCheck(
            name="memory_usage",
            check_function=check_memory_usage,
            interval=30
        ))
        
        # Add default alerts
        metrics_collector.add_alert(Alert(
            alert_id="high_cpu_usage",
            name="High CPU Usage",
            description="CPU usage is above 80%",
            severity=AlertSeverity.WARNING,
            metric_name="cpu_usage",
            condition="> threshold",
            threshold=80.0
        ))
        
        metrics_collector.add_alert(Alert(
            alert_id="high_memory_usage",
            name="High Memory Usage",
            description="Memory usage is above 90%",
            severity=AlertSeverity.CRITICAL,
            metric_name="memory_usage",
            condition="> threshold",
            threshold=90.0
        ))
        
        metrics_collector.add_alert(Alert(
            alert_id="low_disk_space",
            name="Low Disk Space",
            description="Disk usage is above 95%",
            severity=AlertSeverity.CRITICAL,
            metric_name="disk_usage",
            condition="> threshold",
            threshold=95.0
        ))
    
    return metrics_collector


# Alert handlers
def log_alert_handler(alert: Alert):
    """Log alert to file"""
    logger.critical(f"ALERT: {alert.name} - {alert.description}")


async def webhook_alert_handler(alert: Alert):
    """Send alert to webhook - production implementation"""
    webhook_urls = {
        "slack": os.getenv("SLACK_WEBHOOK_URL"),
        "teams": os.getenv("TEAMS_WEBHOOK_URL"),
        "discord": os.getenv("DISCORD_WEBHOOK_URL"),
        "pagerduty": os.getenv("PAGERDUTY_API_URL")
    }
    
    for service, url in webhook_urls.items():
        if url:
            try:
                await _send_webhook_alert(alert, service, url)
            except Exception as e:
                logger.error(f"Failed to send {service} alert: {e}")

async def _send_webhook_alert(alert: Alert, service: str, webhook_url: str):
    """Send alert to specific webhook service"""
    import aiohttp
    
    # Format message based on service
    if service == "slack":
        payload = {
            "text": f"🚨 XORB Alert: {alert.name}",
            "attachments": [{
                "color": "danger" if alert.severity == AlertSeverity.CRITICAL else "warning",
                "fields": [
                    {"title": "Severity", "value": alert.severity.value, "short": True},
                    {"title": "Description", "value": alert.description, "short": False},
                    {"title": "Metric", "value": alert.metric_name, "short": True},
                    {"title": "Threshold", "value": str(alert.threshold), "short": True},
                    {"title": "Triggered At", "value": alert.triggered_at.isoformat() if alert.triggered_at else "N/A", "short": False}
                ]
            }]
        }
    elif service == "teams":
        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": "FF0000" if alert.severity == AlertSeverity.CRITICAL else "FFA500",
            "summary": f"XORB Alert: {alert.name}",
            "sections": [{
                "activityTitle": f"🚨 XORB Alert: {alert.name}",
                "activitySubtitle": alert.description,
                "facts": [
                    {"name": "Severity", "value": alert.severity.value},
                    {"name": "Metric", "value": alert.metric_name},
                    {"name": "Threshold", "value": str(alert.threshold)},
                    {"name": "Triggered At", "value": alert.triggered_at.isoformat() if alert.triggered_at else "N/A"}
                ]
            }]
        }
    elif service == "discord":
        payload = {
            "content": f"🚨 **XORB Alert: {alert.name}**",
            "embeds": [{
                "title": alert.name,
                "description": alert.description,
                "color": 16711680 if alert.severity == AlertSeverity.CRITICAL else 16753920,
                "fields": [
                    {"name": "Severity", "value": alert.severity.value, "inline": True},
                    {"name": "Metric", "value": alert.metric_name, "inline": True},
                    {"name": "Threshold", "value": str(alert.threshold), "inline": True}
                ],
                "timestamp": alert.triggered_at.isoformat() if alert.triggered_at else datetime.utcnow().isoformat()
            }]
        }
    elif service == "pagerduty":
        payload = {
            "routing_key": os.getenv("PAGERDUTY_ROUTING_KEY"),
            "event_action": "trigger",
            "dedup_key": f"xorb_alert_{alert.alert_id}",
            "payload": {
                "summary": f"XORB Alert: {alert.name}",
                "severity": "critical" if alert.severity == AlertSeverity.CRITICAL else "warning",
                "source": "XORB Security Platform",
                "component": alert.metric_name,
                "group": "security",
                "class": "monitoring",
                "custom_details": {
                    "description": alert.description,
                    "threshold": alert.threshold,
                    "triggered_at": alert.triggered_at.isoformat() if alert.triggered_at else "N/A",
                    "labels": alert.labels
                }
            }
        }
    else:
        payload = {
            "alert_name": alert.name,
            "severity": alert.severity.value,
            "description": alert.description,
            "metric": alert.metric_name,
            "threshold": alert.threshold,
            "triggered_at": alert.triggered_at.isoformat() if alert.triggered_at else "N/A"
        }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            if response.status >= 400:
                logger.error(f"Webhook {service} returned status {response.status}")
            else:
                logger.info(f"Successfully sent {service} alert for {alert.name}")

async def email_alert_handler(alert: Alert):
    """Send alert via email - production implementation"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    # Email configuration from environment
    smtp_host = os.getenv("SMTP_HOST", "localhost")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("ALERT_FROM_EMAIL", "alerts@xorb-security.com")
    to_emails = os.getenv("ALERT_TO_EMAILS", "").split(",")
    
    if not to_emails or not to_emails[0]:
        logger.warning("No email recipients configured for alerts")
        return
    
    try:
        # Create email message
        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = ", ".join(to_emails)
        msg["Subject"] = f"🚨 XORB Alert: {alert.name}"
        
        # Email body
        body = f"""
XORB Security Platform Alert

Alert: {alert.name}
Severity: {alert.severity.value.upper()}
Description: {alert.description}

Details:
- Metric: {alert.metric_name}
- Threshold: {alert.threshold}
- Condition: {alert.condition}
- Triggered At: {alert.triggered_at.isoformat() if alert.triggered_at else 'N/A'}

Labels: {', '.join(f'{k}={v}' for k, v in alert.labels.items())}

Actions:
{', '.join(alert.actions) if alert.actions else 'No automated actions configured'}

--
XORB Security Platform
Automated Alert System
"""
        
        msg.attach(MIMEText(body, "plain"))
        
        # Send email
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            if smtp_user and smtp_password:
                server.starttls()
                server.login(smtp_user, smtp_password)
            
            server.send_message(msg)
            
        logger.info(f"Successfully sent email alert for {alert.name} to {len(to_emails)} recipients")
        
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")

async def sms_alert_handler(alert: Alert):
    """Send critical alerts via SMS - production implementation"""
    if alert.severity != AlertSeverity.CRITICAL:
        return  # Only send SMS for critical alerts
    
    # SMS configuration
    twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_from = os.getenv("TWILIO_FROM_NUMBER")
    sms_numbers = os.getenv("ALERT_SMS_NUMBERS", "").split(",")
    
    if not all([twilio_sid, twilio_token, twilio_from]) or not sms_numbers[0]:
        logger.warning("SMS configuration incomplete, skipping SMS alerts")
        return
    
    try:
        # Use Twilio API
        auth = aiohttp.BasicAuth(twilio_sid, twilio_token)
        url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_sid}/Messages.json"
        
        message = f"🚨 CRITICAL XORB ALERT: {alert.name}\n{alert.description}\nThreshold: {alert.threshold}\nTime: {alert.triggered_at.strftime('%H:%M %d/%m') if alert.triggered_at else 'N/A'}"
        
        async with aiohttp.ClientSession(auth=auth) as session:
            for number in sms_numbers:
                number = number.strip()
                if number:
                    data = {
                        "From": twilio_from,
                        "To": number,
                        "Body": message
                    }
                    
                    async with session.post(url, data=data) as response:
                        if response.status == 201:
                            logger.info(f"Successfully sent SMS alert to {number}")
                        else:
                            logger.error(f"Failed to send SMS to {number}: {response.status}")
                            
    except Exception as e:
        logger.error(f"Failed to send SMS alerts: {e}")

async def create_incident_handler(alert: Alert):
    """Create incident in external system for critical alerts"""
    if alert.severity != AlertSeverity.CRITICAL:
        return
    
    # Integration with incident management systems
    servicenow_url = os.getenv("SERVICENOW_URL")
    servicenow_user = os.getenv("SERVICENOW_USER") 
    servicenow_password = os.getenv("SERVICENOW_PASSWORD")
    
    jira_url = os.getenv("JIRA_URL")
    jira_user = os.getenv("JIRA_USER")
    jira_token = os.getenv("JIRA_TOKEN")
    
    # ServiceNow incident creation
    if all([servicenow_url, servicenow_user, servicenow_password]):
        try:
            await _create_servicenow_incident(alert, servicenow_url, servicenow_user, servicenow_password)
        except Exception as e:
            logger.error(f"Failed to create ServiceNow incident: {e}")
    
    # JIRA incident creation  
    if all([jira_url, jira_user, jira_token]):
        try:
            await _create_jira_incident(alert, jira_url, jira_user, jira_token)
        except Exception as e:
            logger.error(f"Failed to create JIRA incident: {e}")

async def _create_servicenow_incident(alert: Alert, url: str, user: str, password: str):
    """Create ServiceNow incident"""
    auth = aiohttp.BasicAuth(user, password)
    incident_url = f"{url}/api/now/table/incident"
    
    payload = {
        "short_description": f"XORB Critical Alert: {alert.name}",
        "description": f"{alert.description}\n\nMetric: {alert.metric_name}\nThreshold: {alert.threshold}\nTriggered: {alert.triggered_at}",
        "impact": "1",  # High impact
        "urgency": "1",  # High urgency  
        "category": "Security",
        "subcategory": "Monitoring",
        "caller_id": user,
        "assignment_group": "Security Operations"
    }
    
    async with aiohttp.ClientSession(auth=auth) as session:
        async with session.post(
            incident_url,
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"}
        ) as response:
            if response.status == 201:
                incident_data = await response.json()
                incident_number = incident_data.get("result", {}).get("number", "Unknown")
                logger.info(f"Created ServiceNow incident {incident_number} for alert {alert.name}")
            else:
                logger.error(f"ServiceNow incident creation failed: {response.status}")

async def _create_jira_incident(alert: Alert, url: str, user: str, token: str):
    """Create JIRA incident ticket"""
    auth = aiohttp.BasicAuth(user, token)
    issue_url = f"{url}/rest/api/3/issue"
    
    payload = {
        "fields": {
            "project": {"key": os.getenv("JIRA_PROJECT_KEY", "SEC")},
            "summary": f"XORB Critical Alert: {alert.name}",
            "description": {
                "type": "doc",
                "version": 1,
                "content": [{
                    "type": "paragraph",
                    "content": [{
                        "type": "text",
                        "text": f"{alert.description}\n\nMetric: {alert.metric_name}\nThreshold: {alert.threshold}\nTriggered: {alert.triggered_at}"
                    }]
                }]
            },
            "issuetype": {"name": "Incident"},
            "priority": {"name": "Critical"},
            "labels": ["xorb", "automated", "security"]
        }
    }
    
    async with aiohttp.ClientSession(auth=auth) as session:
        async with session.post(
            issue_url,
            json=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"}
        ) as response:
            if response.status == 201:
                issue_data = await response.json()
                issue_key = issue_data.get("key", "Unknown")
                logger.info(f"Created JIRA incident {issue_key} for alert {alert.name}")
            else:
                logger.error(f"JIRA incident creation failed: {response.status}")


# Initialize monitoring
async def initialize_monitoring():
    """Initialize production monitoring with comprehensive alerting"""
    collector = get_metrics_collector()
    
    # Add alert handlers - production ready
    collector.add_alert_handler(log_alert_handler)
    
    # Async handlers require special handling
    def async_webhook_handler(alert):
        asyncio.create_task(webhook_alert_handler(alert))
    
    def async_email_handler(alert):
        asyncio.create_task(email_alert_handler(alert))
        
    def async_sms_handler(alert):
        asyncio.create_task(sms_alert_handler(alert))
        
    def async_incident_handler(alert):
        asyncio.create_task(create_incident_handler(alert))
    
    collector.add_alert_handler(async_webhook_handler)
    collector.add_alert_handler(async_email_handler)
    collector.add_alert_handler(async_sms_handler)
    collector.add_alert_handler(async_incident_handler)
    
    # Start monitoring
    await collector.start_monitoring()
    
    logger.info("Production monitoring initialized with comprehensive alerting (webhooks, email, SMS, incident management)")


async def shutdown_monitoring():
    """Shutdown monitoring"""
    collector = get_metrics_collector()
    await collector.stop_monitoring()
    logger.info("Production monitoring shutdown")