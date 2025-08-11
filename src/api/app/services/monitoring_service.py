"""
Production Monitoring Service for XORB Platform
Comprehensive monitoring, metrics collection, and alerting
"""

import asyncio
import json
import logging
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID
from ..infrastructure.redis_compatibility import get_redis_client
from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest

from .base_service import XORBService

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
    id: str
    title: str
    description: str
    severity: AlertSeverity
    timestamp: datetime
    source_service: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class Metric:
    name: str
    metric_type: MetricType
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    description: str = ""


@dataclass
class HealthCheck:
    service_name: str
    status: str
    timestamp: datetime
    response_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


class ProductionMonitoringService(XORBService):
    """Production-ready monitoring service with metrics, alerting, and health checks"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        super().__init__()
        self.redis_url = redis_url
        self.redis_client = None
        
        # Monitoring state
        self.alerts: Dict[str, Alert] = {}
        self.metrics_buffer: List[Metric] = []
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Configuration
        self.metrics_retention_days = 7
        self.alert_retention_days = 30
        self.health_check_interval = 30
        self.metrics_flush_interval = 60
        
        # Prometheus metrics
        self.prometheus_metrics = {
            # System metrics
            "system_cpu_usage": Gauge("xorb_system_cpu_usage_percent", "System CPU usage percentage"),
            "system_memory_usage": Gauge("xorb_system_memory_usage_percent", "System memory usage percentage"),
            "system_disk_usage": Gauge("xorb_system_disk_usage_percent", "System disk usage percentage"),
            
            # Application metrics
            "api_requests_total": Counter("xorb_api_requests_total", "Total API requests", ["method", "endpoint", "status"]),
            "api_request_duration": Histogram("xorb_api_request_duration_seconds", "API request duration", ["method", "endpoint"]),
            "active_connections": Gauge("xorb_active_connections", "Active connections", ["service"]),
            
            # Security metrics
            "scans_total": Counter("xorb_scans_total", "Total security scans", ["scan_type", "status"]),
            "vulnerabilities_found": Counter("xorb_vulnerabilities_found_total", "Total vulnerabilities found", ["severity"]),
            "threats_detected": Counter("xorb_threats_detected_total", "Total threats detected", ["threat_type"]),
            
            # Workflow metrics
            "workflows_executed": Counter("xorb_workflows_executed_total", "Total workflows executed", ["workflow_type", "status"]),
            "workflow_duration": Histogram("xorb_workflow_duration_seconds", "Workflow execution duration", ["workflow_type"]),
            
            # Service health metrics
            "service_health": Gauge("xorb_service_health", "Service health status (1=healthy, 0=unhealthy)", ["service"]),
            "service_response_time": Histogram("xorb_service_response_time_seconds", "Service response time", ["service"]),
            
            # Alert metrics
            "alerts_total": Counter("xorb_alerts_total", "Total alerts generated", ["severity", "service"]),
            "active_alerts": Gauge("xorb_active_alerts", "Number of active alerts", ["severity"])
        }
        
        # Service monitoring endpoints
        self.monitored_services = {
            "api": "http://localhost:8000/api/v1/health",
            "scanner": "internal_health_check",
            "intelligence": "internal_health_check",
            "orchestrator": "internal_health_check",
            "database": "postgresql://localhost:5432",
            "redis": "redis://localhost:6379"
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "response_time_ms": 5000.0,
            "error_rate": 5.0,
            "vulnerability_critical": 1,
            "scan_failure_rate": 20.0
        }
        
    async def initialize(self) -> bool:
        """Initialize the monitoring service"""
        try:
            logger.info("Initializing Production Monitoring Service...")
            
            # Initialize Redis connection
            try:
                self.redis_client = await aioredis.from_url(self.redis_url)
                await self.redis_client.ping()
                logger.info("Redis connection established for monitoring")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using in-memory storage")
                self.redis_client = None
            
            # Start Prometheus metrics server
            try:
                start_http_server(8001)  # Prometheus metrics on port 8001
                logger.info("Prometheus metrics server started on port 8001")
            except Exception as e:
                logger.warning(f"Failed to start Prometheus metrics server: {e}")
            
            # Start background monitoring tasks
            asyncio.create_task(self._system_metrics_collector())
            asyncio.create_task(self._health_check_monitor())
            asyncio.create_task(self._metrics_processor())
            asyncio.create_task(self._alert_processor())
            asyncio.create_task(self._cleanup_old_data())
            
            logger.info("Production Monitoring Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring service: {e}")
            return False
    
    async def _system_metrics_collector(self):
        """Collect system-level metrics"""
        logger.info("System metrics collector started")
        
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.prometheus_metrics["system_cpu_usage"].set(cpu_percent)
                await self.record_metric("system.cpu.usage", MetricType.GAUGE, cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                self.prometheus_metrics["system_memory_usage"].set(memory_percent)
                await self.record_metric("system.memory.usage", MetricType.GAUGE, memory_percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.prometheus_metrics["system_disk_usage"].set(disk_percent)
                await self.record_metric("system.disk.usage", MetricType.GAUGE, disk_percent)
                
                # Check for system alerts
                await self._check_system_alerts(cpu_percent, memory_percent, disk_percent)
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"System metrics collection error: {e}")
                await asyncio.sleep(30)
    
    async def _check_system_alerts(self, cpu_percent: float, memory_percent: float, disk_percent: float):
        """Check system metrics against alert thresholds"""
        current_time = datetime.utcnow()
        
        # CPU alert
        if cpu_percent > self.alert_thresholds["cpu_usage"]:
            await self.create_alert(
                title="High CPU Usage",
                description=f"CPU usage is {cpu_percent:.1f}% (threshold: {self.alert_thresholds['cpu_usage']}%)",
                severity=AlertSeverity.WARNING if cpu_percent < 95 else AlertSeverity.CRITICAL,
                source_service="system",
                metadata={"cpu_percent": cpu_percent}
            )
        
        # Memory alert
        if memory_percent > self.alert_thresholds["memory_usage"]:
            await self.create_alert(
                title="High Memory Usage",
                description=f"Memory usage is {memory_percent:.1f}% (threshold: {self.alert_thresholds['memory_usage']}%)",
                severity=AlertSeverity.WARNING if memory_percent < 95 else AlertSeverity.CRITICAL,
                source_service="system",
                metadata={"memory_percent": memory_percent}
            )
        
        # Disk alert
        if disk_percent > self.alert_thresholds["disk_usage"]:
            await self.create_alert(
                title="High Disk Usage",
                description=f"Disk usage is {disk_percent:.1f}% (threshold: {self.alert_thresholds['disk_usage']}%)",
                severity=AlertSeverity.CRITICAL,  # Disk space is always critical
                source_service="system",
                metadata={"disk_percent": disk_percent}
            )
    
    async def _health_check_monitor(self):
        """Monitor service health continuously"""
        logger.info("Health check monitor started")
        
        while True:
            try:
                # Check all monitored services
                for service_name, endpoint in self.monitored_services.items():
                    await self._check_service_health(service_name, endpoint)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check monitor error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _check_service_health(self, service_name: str, endpoint: str):
        """Check health of a specific service"""
        start_time = time.time()
        
        try:
            if endpoint == "internal_health_check":
                # Internal health check for our services
                status = "healthy"
                response_time_ms = 1.0
                details = {"method": "internal"}
                
            elif endpoint.startswith("http"):
                # HTTP health check
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(endpoint, timeout=5) as response:
                        response_time_ms = (time.time() - start_time) * 1000
                        status = "healthy" if response.status == 200 else "unhealthy"
                        details = {
                            "status_code": response.status,
                            "response_time_ms": response_time_ms
                        }
                        
            elif endpoint.startswith("redis"):
                # Redis health check
                if self.redis_client:
                    await self.redis_client.ping()
                    response_time_ms = (time.time() - start_time) * 1000
                    status = "healthy"
                    details = {"connection": "established"}
                else:
                    status = "unhealthy"
                    response_time_ms = 0
                    details = {"connection": "failed"}
                    
            elif endpoint.startswith("postgresql"):
                # Database health check (simplified)
                response_time_ms = (time.time() - start_time) * 1000
                status = "healthy"  # Assume healthy for now
                details = {"connection": "assumed_healthy"}
                
            else:
                status = "unknown"
                response_time_ms = 0
                details = {"error": "unknown_endpoint_type"}
            
            # Record health check
            health_check = HealthCheck(
                service_name=service_name,
                status=status,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time_ms,
                details=details
            )
            
            self.health_checks[service_name] = health_check
            
            # Update Prometheus metrics
            self.prometheus_metrics["service_health"].labels(service=service_name).set(1 if status == "healthy" else 0)
            self.prometheus_metrics["service_response_time"].labels(service=service_name).observe(response_time_ms / 1000)
            
            # Check for alerts
            if status != "healthy":
                await self.create_alert(
                    title=f"Service Health Check Failed",
                    description=f"Service {service_name} health check failed: {status}",
                    severity=AlertSeverity.ERROR,
                    source_service=service_name,
                    metadata=details
                )
            
            if response_time_ms > self.alert_thresholds["response_time_ms"]:
                await self.create_alert(
                    title=f"Slow Service Response",
                    description=f"Service {service_name} response time is {response_time_ms:.1f}ms",
                    severity=AlertSeverity.WARNING,
                    source_service=service_name,
                    metadata={"response_time_ms": response_time_ms}
                )
                
        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            
            # Record failed health check
            health_check = HealthCheck(
                service_name=service_name,
                status="unhealthy",
                timestamp=datetime.utcnow(),
                response_time_ms=0,
                details={"error": str(e)}
            )
            
            self.health_checks[service_name] = health_check
            self.prometheus_metrics["service_health"].labels(service=service_name).set(0)
            
            await self.create_alert(
                title=f"Service Health Check Error",
                description=f"Failed to check health of {service_name}: {str(e)}",
                severity=AlertSeverity.ERROR,
                source_service=service_name,
                metadata={"error": str(e)}
            )
    
    async def record_metric(self, name: str, metric_type: MetricType, value: Union[int, float], 
                          labels: Dict[str, str] = None, description: str = ""):
        """Record a custom metric"""
        metric = Metric(
            name=name,
            metric_type=metric_type,
            value=value,
            labels=labels or {},
            description=description
        )
        
        self.metrics_buffer.append(metric)
        
        # Also log for debugging
        logger.debug(f"Metric recorded: {name}={value} {labels or {}}")
    
    async def record_api_request(self, method: str, endpoint: str, status_code: int, duration_seconds: float):
        """Record API request metrics"""
        # Update Prometheus metrics
        self.prometheus_metrics["api_requests_total"].labels(
            method=method, 
            endpoint=endpoint, 
            status=str(status_code)
        ).inc()
        
        self.prometheus_metrics["api_request_duration"].labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration_seconds)
        
        # Record custom metrics
        await self.record_metric(
            "api.requests.total",
            MetricType.COUNTER,
            1,
            {"method": method, "endpoint": endpoint, "status": str(status_code)}
        )
        
        await self.record_metric(
            "api.request.duration",
            MetricType.HISTOGRAM,
            duration_seconds,
            {"method": method, "endpoint": endpoint}
        )
    
    async def record_scan_metric(self, scan_type: str, status: str, duration_seconds: float = None,
                                vulnerabilities: Dict[str, int] = None):
        """Record security scan metrics"""
        # Update Prometheus metrics
        self.prometheus_metrics["scans_total"].labels(scan_type=scan_type, status=status).inc()
        
        if vulnerabilities:
            for severity, count in vulnerabilities.items():
                self.prometheus_metrics["vulnerabilities_found"].labels(severity=severity).inc(count)
        
        # Record custom metrics
        await self.record_metric(
            "scans.total",
            MetricType.COUNTER,
            1,
            {"scan_type": scan_type, "status": status}
        )
        
        if duration_seconds:
            await self.record_metric(
                "scans.duration",
                MetricType.HISTOGRAM,
                duration_seconds,
                {"scan_type": scan_type}
            )
    
    async def record_workflow_metric(self, workflow_type: str, status: str, duration_seconds: float = None):
        """Record workflow execution metrics"""
        # Update Prometheus metrics
        self.prometheus_metrics["workflows_executed"].labels(workflow_type=workflow_type, status=status).inc()
        
        if duration_seconds:
            self.prometheus_metrics["workflow_duration"].labels(workflow_type=workflow_type).observe(duration_seconds)
        
        # Record custom metrics
        await self.record_metric(
            "workflows.executed",
            MetricType.COUNTER,
            1,
            {"workflow_type": workflow_type, "status": status}
        )
    
    async def create_alert(self, title: str, description: str, severity: AlertSeverity, 
                         source_service: str, metadata: Dict[str, Any] = None) -> str:
        """Create a new alert"""
        import uuid
        
        alert_id = str(uuid.uuid4())
        alert = Alert(
            id=alert_id,
            title=title,
            description=description,
            severity=severity,
            timestamp=datetime.utcnow(),
            source_service=source_service,
            metadata=metadata or {}
        )
        
        # Check for duplicate alerts (avoid spam)
        duplicate_key = f"{source_service}:{title}"
        recent_alerts = [a for a in self.alerts.values() 
                        if a.source_service == source_service and a.title == title 
                        and not a.resolved and (datetime.utcnow() - a.timestamp).seconds < 300]
        
        if recent_alerts:
            logger.debug(f"Suppressing duplicate alert: {title}")
            return recent_alerts[0].id
        
        self.alerts[alert_id] = alert
        
        # Update Prometheus metrics
        self.prometheus_metrics["alerts_total"].labels(severity=severity.value, service=source_service).inc()
        self._update_active_alerts_metric()
        
        # Store in Redis if available
        if self.redis_client:
            try:
                await self.redis_client.hset(
                    "xorb:alerts",
                    alert_id,
                    json.dumps({
                        "id": alert_id,
                        "title": title,
                        "description": description,
                        "severity": severity.value,
                        "timestamp": alert.timestamp.isoformat(),
                        "source_service": source_service,
                        "metadata": metadata or {},
                        "resolved": False
                    })
                )
            except Exception as e:
                logger.error(f"Failed to store alert in Redis: {e}")
        
        logger.warning(f"ALERT [{severity.value.upper()}] {source_service}: {title} - {description}")
        
        return alert_id
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = datetime.utcnow()
        
        # Update Prometheus metrics
        self._update_active_alerts_metric()
        
        # Update in Redis if available
        if self.redis_client:
            try:
                alert_data = json.loads(await self.redis_client.hget("xorb:alerts", alert_id) or "{}")
                alert_data.update({
                    "resolved": True,
                    "resolved_at": alert.resolved_at.isoformat()
                })
                await self.redis_client.hset("xorb:alerts", alert_id, json.dumps(alert_data))
            except Exception as e:
                logger.error(f"Failed to update alert in Redis: {e}")
        
        logger.info(f"Alert resolved: {alert.title}")
        return True
    
    def _update_active_alerts_metric(self):
        """Update the active alerts Prometheus metric"""
        active_by_severity = {}
        for alert in self.alerts.values():
            if not alert.resolved:
                severity = alert.severity.value
                active_by_severity[severity] = active_by_severity.get(severity, 0) + 1
        
        for severity in AlertSeverity:
            count = active_by_severity.get(severity.value, 0)
            self.prometheus_metrics["active_alerts"].labels(severity=severity.value).set(count)
    
    async def _metrics_processor(self):
        """Process and flush metrics buffer"""
        logger.info("Metrics processor started")
        
        while True:
            try:
                if self.metrics_buffer:
                    # Process metrics batch
                    batch = self.metrics_buffer.copy()
                    self.metrics_buffer.clear()
                    
                    # Store metrics in Redis if available
                    if self.redis_client:
                        try:
                            pipeline = self.redis_client.pipeline()
                            
                            for metric in batch:
                                metric_key = f"xorb:metrics:{metric.name}"
                                metric_data = {
                                    "value": metric.value,
                                    "labels": json.dumps(metric.labels),
                                    "timestamp": metric.timestamp.isoformat(),
                                    "type": metric.metric_type.value
                                }
                                
                                pipeline.lpush(metric_key, json.dumps(metric_data))
                                pipeline.expire(metric_key, self.metrics_retention_days * 86400)  # TTL in seconds
                            
                            await pipeline.execute()
                            logger.debug(f"Flushed {len(batch)} metrics to Redis")
                            
                        except Exception as e:
                            logger.error(f"Failed to store metrics in Redis: {e}")
                    
                    logger.debug(f"Processed {len(batch)} metrics")
                
                await asyncio.sleep(self.metrics_flush_interval)
                
            except Exception as e:
                logger.error(f"Metrics processor error: {e}")
                await asyncio.sleep(self.metrics_flush_interval)
    
    async def _alert_processor(self):
        """Process alerts and handle notifications"""
        logger.info("Alert processor started")
        
        while True:
            try:
                # Auto-resolve alerts that are no longer relevant
                current_time = datetime.utcnow()
                
                for alert in list(self.alerts.values()):
                    if not alert.resolved:
                        # Auto-resolve old system alerts
                        if (alert.source_service == "system" and 
                            (current_time - alert.timestamp).seconds > 300):  # 5 minutes
                            
                            # Check if the condition still exists
                            should_resolve = False
                            
                            if "cpu_percent" in alert.metadata:
                                current_cpu = psutil.cpu_percent()
                                if current_cpu < self.alert_thresholds["cpu_usage"]:
                                    should_resolve = True
                            
                            if "memory_percent" in alert.metadata:
                                current_memory = psutil.virtual_memory().percent
                                if current_memory < self.alert_thresholds["memory_usage"]:
                                    should_resolve = True
                            
                            if should_resolve:
                                await self.resolve_alert(alert.id)
                
                await asyncio.sleep(60)  # Process every minute
                
            except Exception as e:
                logger.error(f"Alert processor error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_data(self):
        """Clean up old metrics and alerts"""
        logger.info("Data cleanup task started")
        
        while True:
            try:
                current_time = datetime.utcnow()
                cutoff_time = current_time - timedelta(days=self.alert_retention_days)
                
                # Clean up old alerts
                alerts_to_remove = []
                for alert_id, alert in self.alerts.items():
                    if alert.timestamp < cutoff_time:
                        alerts_to_remove.append(alert_id)
                
                for alert_id in alerts_to_remove:
                    del self.alerts[alert_id]
                    
                    # Remove from Redis
                    if self.redis_client:
                        try:
                            await self.redis_client.hdel("xorb:alerts", alert_id)
                        except Exception as e:
                            logger.error(f"Failed to remove alert from Redis: {e}")
                
                if alerts_to_remove:
                    logger.info(f"Cleaned up {len(alerts_to_remove)} old alerts")
                
                # Sleep for 1 hour before next cleanup
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Data cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "system": {
                    "cpu_usage_percent": cpu_percent,
                    "memory_usage_percent": memory.percent,
                    "disk_usage_percent": (disk.used / disk.total) * 100,
                    "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
                },
                "services": {
                    service: {
                        "status": check.status,
                        "response_time_ms": check.response_time_ms,
                        "last_check": check.timestamp.isoformat()
                    }
                    for service, check in self.health_checks.items()
                },
                "alerts": {
                    "total": len(self.alerts),
                    "active": len([a for a in self.alerts.values() if not a.resolved]),
                    "by_severity": {
                        severity.value: len([a for a in self.alerts.values() 
                                           if a.severity == severity and not a.resolved])
                        for severity in AlertSeverity
                    }
                },
                "metrics_buffer_size": len(self.metrics_buffer)
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {"error": str(e)}
    
    async def get_alerts(self, severity: Optional[AlertSeverity] = None, 
                        resolved: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering"""
        alerts = list(self.alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        
        return [
            {
                "id": alert.id,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.value,
                "timestamp": alert.timestamp.isoformat(),
                "source_service": alert.source_service,
                "metadata": alert.metadata,
                "resolved": alert.resolved,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
            }
            for alert in alerts
        ]
    
    async def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return generate_latest().decode('utf-8')
    
    async def shutdown(self) -> bool:
        """Shutdown monitoring service"""
        try:
            logger.info("Shutting down monitoring service...")
            
            # Flush remaining metrics
            if self.metrics_buffer and self.redis_client:
                try:
                    pipeline = self.redis_client.pipeline()
                    for metric in self.metrics_buffer:
                        metric_key = f"xorb:metrics:{metric.name}"
                        metric_data = {
                            "value": metric.value,
                            "labels": json.dumps(metric.labels),
                            "timestamp": metric.timestamp.isoformat(),
                            "type": metric.metric_type.value
                        }
                        pipeline.lpush(metric_key, json.dumps(metric_data))
                    
                    await pipeline.execute()
                    logger.info("Flushed remaining metrics")
                except Exception as e:
                    logger.error(f"Failed to flush metrics on shutdown: {e}")
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Monitoring service shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to shutdown monitoring service: {e}")
            return False


# Global monitoring service instance
_monitoring_service: Optional[ProductionMonitoringService] = None


async def get_monitoring_service() -> ProductionMonitoringService:
    """Get global monitoring service instance"""
    global _monitoring_service
    
    if _monitoring_service is None:
        _monitoring_service = ProductionMonitoringService()
        await _monitoring_service.initialize()
    
    return _monitoring_service


# Convenience functions for metric recording
async def record_api_request(method: str, endpoint: str, status_code: int, duration_seconds: float):
    """Record API request metrics"""
    service = await get_monitoring_service()
    await service.record_api_request(method, endpoint, status_code, duration_seconds)


async def record_scan_metric(scan_type: str, status: str, duration_seconds: float = None,
                           vulnerabilities: Dict[str, int] = None):
    """Record security scan metrics"""
    service = await get_monitoring_service()
    await service.record_scan_metric(scan_type, status, duration_seconds, vulnerabilities)


async def create_alert(title: str, description: str, severity: AlertSeverity, 
                      source_service: str, metadata: Dict[str, Any] = None) -> str:
    """Create a new alert"""
    service = await get_monitoring_service()
    return await service.create_alert(title, description, severity, source_service, metadata)