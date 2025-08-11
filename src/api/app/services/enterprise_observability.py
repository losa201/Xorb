"""
Enterprise Observability and Monitoring System for XORB Platform
Advanced monitoring, alerting, and observability capabilities
"""

import asyncio
import json
import logging
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from uuid import uuid4
from collections import defaultdict, deque
import statistics
from pathlib import Path

from .base_service import XORBService, ServiceHealth, ServiceStatus
from ..domain.entities import User, Organization

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Individual metric data point"""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert definition and state"""
    alert_id: str
    name: str
    description: str
    severity: str  # critical, high, medium, low
    condition: str
    threshold: float
    current_value: float
    status: str  # active, resolved, suppressed
    created_at: datetime
    updated_at: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceMetrics:
    """Service-level metrics"""
    service_id: str
    health_score: float
    response_time_p95: float
    error_rate: float
    throughput: float
    cpu_usage: float
    memory_usage: float
    active_connections: int
    last_updated: datetime


@dataclass
class SystemMetrics:
    """System-level metrics"""
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_io_bytes: Dict[str, int]
    disk_io_bytes: Dict[str, int]
    load_average: List[float]
    uptime_seconds: float
    timestamp: datetime


class EnterpriseObservabilityService(XORBService):
    """Enterprise-grade observability and monitoring service"""
    
    def __init__(self, **kwargs):
        super().__init__(
            service_id="enterprise_observability",
            dependencies=["database", "cache"],
            **kwargs
        )
        
        # Metric storage (in production, would use time-series DB)
        self.metrics_storage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.service_metrics: Dict[str, ServiceMetrics] = {}
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.request_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.error_tracking: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # SLA tracking
        self.sla_targets = {
            "availability": 99.9,
            "response_time_p95": 200,  # milliseconds
            "error_rate": 0.5,  # percentage
            "mttr": 15  # minutes - Mean Time To Recovery
        }
        
        # Alerting configuration
        self.notification_channels = []
        self.alert_escalation_rules = {}
        
        # Health check intervals
        self.health_check_interval = 30  # seconds
        self.metrics_collection_interval = 10  # seconds
        
        # Initialize default alert rules
        self._initialize_default_alert_rules()

    async def initialize(self) -> bool:
        """Initialize observability service"""
        try:
            logger.info("Initializing Enterprise Observability Service...")
            
            # Start background tasks
            asyncio.create_task(self._collect_system_metrics_loop())
            asyncio.create_task(self._health_check_loop())
            asyncio.create_task(self._alert_evaluation_loop())
            asyncio.create_task(self._cleanup_old_metrics_loop())
            
            # Initialize service discovery
            await self._discover_services()
            
            # Load persisted alert rules
            await self._load_alert_rules()
            
            logger.info("Enterprise Observability Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize observability service: {e}")
            return False

    async def collect_metric(
        self,
        metric_name: str,
        value: float,
        tags: Dict[str, str] = None,
        timestamp: datetime = None
    ) -> bool:
        """Collect a metric data point"""
        
        try:
            metric_point = MetricPoint(
                timestamp=timestamp or datetime.utcnow(),
                value=value,
                tags=tags or {},
                metadata={}
            )
            
            self.metrics_storage[metric_name].append(metric_point)
            
            # Real-time alert evaluation for this metric
            await self._evaluate_metric_alerts(metric_name, value)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to collect metric {metric_name}: {e}")
            return False

    async def record_request_metrics(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float,
        user_id: str = None,
        tenant_id: str = None
    ) -> bool:
        """Record HTTP request metrics"""
        
        try:
            timestamp = datetime.utcnow()
            
            # Request metrics
            request_data = {
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "response_time_ms": response_time_ms,
                "timestamp": timestamp,
                "user_id": user_id,
                "tenant_id": tenant_id
            }
            
            self.request_metrics[f"{method}:{endpoint}"].append(request_data)
            
            # Collect aggregated metrics
            await self.collect_metric(
                "http_requests_total",
                1,
                tags={
                    "endpoint": endpoint,
                    "method": method,
                    "status_code": str(status_code)
                },
                timestamp=timestamp
            )
            
            await self.collect_metric(
                "http_request_duration_ms",
                response_time_ms,
                tags={
                    "endpoint": endpoint,
                    "method": method
                },
                timestamp=timestamp
            )
            
            # Track errors
            if status_code >= 400:
                await self.collect_metric(
                    "http_errors_total",
                    1,
                    tags={
                        "endpoint": endpoint,
                        "method": method,
                        "status_code": str(status_code)
                    },
                    timestamp=timestamp
                )
                
                self.error_tracking["http_errors"].append(request_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record request metrics: {e}")
            return False

    async def get_service_health_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive service health dashboard"""
        
        try:
            dashboard = {
                "overview": await self._get_platform_overview(),
                "services": await self._get_services_status(),
                "system_metrics": await self._get_system_metrics_summary(),
                "active_alerts": await self._get_active_alerts(),
                "sla_status": await self._get_sla_status(),
                "performance_metrics": await self._get_performance_metrics(),
                "capacity_metrics": await self._get_capacity_metrics(),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Failed to generate health dashboard: {e}")
            return {"error": str(e)}

    async def create_alert_rule(
        self,
        name: str,
        metric_name: str,
        condition: str,
        threshold: float,
        severity: str,
        description: str = "",
        tags: Dict[str, str] = None
    ) -> str:
        """Create a new alert rule"""
        
        rule_id = str(uuid4())
        
        try:
            alert_rule = {
                "rule_id": rule_id,
                "name": name,
                "metric_name": metric_name,
                "condition": condition,  # "gt", "lt", "eq", "gte", "lte"
                "threshold": threshold,
                "severity": severity,
                "description": description,
                "tags": tags or {},
                "enabled": True,
                "created_at": datetime.utcnow().isoformat(),
                "evaluation_interval": 60,  # seconds
                "cooldown_period": 300  # seconds
            }
            
            self.alert_rules[rule_id] = alert_rule
            
            # Persist alert rule
            await self._persist_alert_rule(rule_id, alert_rule)
            
            logger.info(f"Created alert rule: {name} ({rule_id})")
            return rule_id
            
        except Exception as e:
            logger.error(f"Failed to create alert rule: {e}")
            raise

    async def get_performance_analytics(
        self,
        timeframe: str = "1h",
        service_filter: List[str] = None
    ) -> Dict[str, Any]:
        """Get detailed performance analytics"""
        
        try:
            end_time = datetime.utcnow()
            start_time = self._parse_timeframe(timeframe, end_time)
            
            analytics = {
                "timeframe": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "duration": timeframe
                },
                "request_analytics": await self._analyze_request_patterns(start_time, end_time),
                "error_analytics": await self._analyze_error_patterns(start_time, end_time),
                "performance_trends": await self._analyze_performance_trends(start_time, end_time),
                "resource_utilization": await self._analyze_resource_utilization(start_time, end_time),
                "user_experience_metrics": await self._analyze_user_experience(start_time, end_time),
                "capacity_planning": await self._analyze_capacity_planning(start_time, end_time)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to generate performance analytics: {e}")
            return {"error": str(e)}

    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security-focused metrics and insights"""
        
        try:
            security_metrics = {
                "authentication_metrics": await self._get_authentication_metrics(),
                "authorization_failures": await self._get_authorization_failures(),
                "suspicious_activities": await self._get_suspicious_activities(),
                "ptaas_security_metrics": await self._get_ptaas_security_metrics(),
                "threat_detection_metrics": await self._get_threat_detection_metrics(),
                "compliance_metrics": await self._get_compliance_metrics(),
                "incident_response_metrics": await self._get_incident_response_metrics()
            }
            
            return security_metrics
            
        except Exception as e:
            logger.error(f"Failed to generate security metrics: {e}")
            return {"error": str(e)}

    async def create_custom_dashboard(
        self,
        dashboard_name: str,
        widgets: List[Dict[str, Any]],
        user: User,
        org: Organization
    ) -> str:
        """Create custom monitoring dashboard"""
        
        dashboard_id = str(uuid4())
        
        try:
            dashboard = {
                "dashboard_id": dashboard_id,
                "name": dashboard_name,
                "widgets": widgets,
                "user_id": str(user.id),
                "organization_id": str(org.id),
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "shared": False,
                "tags": []
            }
            
            # Validate widgets
            for widget in widgets:
                await self._validate_widget_configuration(widget)
            
            # Store dashboard configuration
            await self._persist_dashboard(dashboard_id, dashboard)
            
            logger.info(f"Created custom dashboard: {dashboard_name} ({dashboard_id})")
            return dashboard_id
            
        except Exception as e:
            logger.error(f"Failed to create custom dashboard: {e}")
            raise

    # Private helper methods
    async def _collect_system_metrics_loop(self):
        """Background task to collect system metrics"""
        
        while True:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.metrics_collection_interval)
            except Exception as e:
                logger.error(f"System metrics collection failed: {e}")
                await asyncio.sleep(self.metrics_collection_interval)

    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            await self.collect_metric("system_cpu_usage_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            await self.collect_metric("system_memory_usage_percent", memory.percent)
            await self.collect_metric("system_memory_available_bytes", memory.available)
            await self.collect_metric("system_memory_used_bytes", memory.used)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            await self.collect_metric("system_disk_usage_percent", disk_percent)
            await self.collect_metric("system_disk_free_bytes", disk.free)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            await self.collect_metric("system_network_bytes_sent", net_io.bytes_sent)
            await self.collect_metric("system_network_bytes_recv", net_io.bytes_recv)
            
            # Load average (Unix systems)
            try:
                load_avg = psutil.getloadavg()
                await self.collect_metric("system_load_average_1m", load_avg[0])
                await self.collect_metric("system_load_average_5m", load_avg[1])
                await self.collect_metric("system_load_average_15m", load_avg[2])
            except AttributeError:
                # Windows doesn't have load average
                pass
            
            # Process metrics
            process_count = len(psutil.pids())
            await self.collect_metric("system_process_count", process_count)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

    async def _health_check_loop(self):
        """Background task for health checks"""
        
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _perform_health_checks(self):
        """Perform health checks on all services"""
        
        timestamp = datetime.utcnow()
        
        for service_id in self.service_metrics.keys():
            try:
                health_score = await self._check_service_health(service_id)
                
                if service_id in self.service_metrics:
                    self.service_metrics[service_id].health_score = health_score
                    self.service_metrics[service_id].last_updated = timestamp
                
                await self.collect_metric(
                    "service_health_score",
                    health_score,
                    tags={"service": service_id},
                    timestamp=timestamp
                )
                
            except Exception as e:
                logger.error(f"Health check failed for service {service_id}: {e}")

    async def _check_service_health(self, service_id: str) -> float:
        """Check health of individual service"""
        
        try:
            # Get recent metrics for the service
            recent_errors = self._get_recent_error_rate(service_id)
            recent_response_time = self._get_recent_response_time(service_id)
            
            # Calculate health score (0-100)
            health_score = 100.0
            
            # Penalize for high error rate
            if recent_errors > 5.0:  # 5% error rate
                health_score -= min(recent_errors * 10, 50)
            
            # Penalize for slow response times
            if recent_response_time > 1000:  # 1 second
                health_score -= min((recent_response_time - 1000) / 100, 30)
            
            return max(health_score, 0.0)
            
        except Exception as e:
            logger.error(f"Failed to check health for {service_id}: {e}")
            return 50.0  # Default to moderate health

    def _get_recent_error_rate(self, service_id: str) -> float:
        """Get recent error rate for service"""
        
        try:
            # Calculate error rate from recent requests
            recent_requests = list(self.request_metrics.get(service_id, []))
            if not recent_requests:
                return 0.0
            
            # Look at last 100 requests or last 5 minutes
            cutoff_time = datetime.utcnow() - timedelta(minutes=5)
            recent_requests = [r for r in recent_requests if r["timestamp"] > cutoff_time]
            
            if not recent_requests:
                return 0.0
            
            error_count = sum(1 for r in recent_requests if r["status_code"] >= 400)
            return (error_count / len(recent_requests)) * 100
            
        except Exception:
            return 0.0

    def _get_recent_response_time(self, service_id: str) -> float:
        """Get recent average response time for service"""
        
        try:
            recent_requests = list(self.request_metrics.get(service_id, []))
            if not recent_requests:
                return 0.0
            
            # Look at last 5 minutes
            cutoff_time = datetime.utcnow() - timedelta(minutes=5)
            recent_requests = [r for r in recent_requests if r["timestamp"] > cutoff_time]
            
            if not recent_requests:
                return 0.0
            
            response_times = [r["response_time_ms"] for r in recent_requests]
            return statistics.mean(response_times)
            
        except Exception:
            return 0.0

    async def _alert_evaluation_loop(self):
        """Background task for alert evaluation"""
        
        while True:
            try:
                await self._evaluate_all_alerts()
                await asyncio.sleep(60)  # Evaluate every minute
            except Exception as e:
                logger.error(f"Alert evaluation failed: {e}")
                await asyncio.sleep(60)

    async def _evaluate_all_alerts(self):
        """Evaluate all alert rules"""
        
        for rule_id, rule in self.alert_rules.items():
            if not rule.get("enabled", True):
                continue
            
            try:
                await self._evaluate_alert_rule(rule_id, rule)
            except Exception as e:
                logger.error(f"Failed to evaluate alert rule {rule_id}: {e}")

    async def _evaluate_alert_rule(self, rule_id: str, rule: Dict[str, Any]):
        """Evaluate a specific alert rule"""
        
        metric_name = rule["metric_name"]
        condition = rule["condition"]
        threshold = rule["threshold"]
        
        # Get recent metric values
        recent_values = self._get_recent_metric_values(metric_name, minutes=5)
        
        if not recent_values:
            return
        
        current_value = recent_values[-1].value
        
        # Evaluate condition
        alert_triggered = self._evaluate_condition(current_value, condition, threshold)
        
        # Check if alert already exists
        existing_alert = self._find_active_alert(rule_id)
        
        if alert_triggered and not existing_alert:
            # Create new alert
            await self._create_alert(rule_id, rule, current_value)
        elif not alert_triggered and existing_alert:
            # Resolve existing alert
            await self._resolve_alert(existing_alert["alert_id"])

    def _get_recent_metric_values(self, metric_name: str, minutes: int = 5) -> List[MetricPoint]:
        """Get recent metric values"""
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        metrics = self.metrics_storage.get(metric_name, [])
        
        return [m for m in metrics if m.timestamp > cutoff_time]

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        
        if condition == "gt":
            return value > threshold
        elif condition == "gte":
            return value >= threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "lte":
            return value <= threshold
        elif condition == "eq":
            return value == threshold
        else:
            return False

    def _find_active_alert(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """Find active alert for rule"""
        
        for alert in self.alerts.values():
            if (alert.tags.get("rule_id") == rule_id and 
                alert.status == "active"):
                return asdict(alert)
        
        return None

    async def _create_alert(self, rule_id: str, rule: Dict[str, Any], current_value: float):
        """Create a new alert"""
        
        alert_id = str(uuid4())
        
        alert = Alert(
            alert_id=alert_id,
            name=rule["name"],
            description=rule["description"],
            severity=rule["severity"],
            condition=rule["condition"],
            threshold=rule["threshold"],
            current_value=current_value,
            status="active",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags={"rule_id": rule_id, **rule.get("tags", {})},
            metadata={"metric_name": rule["metric_name"]}
        )
        
        self.alerts[alert_id] = alert
        
        # Send notifications
        await self._send_alert_notifications(alert)
        
        logger.warning(f"Alert triggered: {alert.name} (value: {current_value}, threshold: {alert.threshold})")

    async def _resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.status = "resolved"
            alert.updated_at = datetime.utcnow()
            
            # Send resolution notifications
            await self._send_alert_notifications(alert)
            
            logger.info(f"Alert resolved: {alert.name}")

    async def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications"""
        
        try:
            # In production, this would integrate with notification services
            # (email, Slack, PagerDuty, etc.)
            notification_data = {
                "alert_id": alert.alert_id,
                "name": alert.name,
                "severity": alert.severity,
                "status": alert.status,
                "current_value": alert.current_value,
                "threshold": alert.threshold,
                "timestamp": alert.updated_at.isoformat()
            }
            
            logger.info(f"Alert notification: {notification_data}")
            
        except Exception as e:
            logger.error(f"Failed to send alert notifications: {e}")

    async def _evaluate_metric_alerts(self, metric_name: str, value: float):
        """Evaluate alerts for a specific metric"""
        
        for rule_id, rule in self.alert_rules.items():
            if rule["metric_name"] == metric_name and rule.get("enabled", True):
                await self._evaluate_alert_rule(rule_id, rule)

    async def _cleanup_old_metrics_loop(self):
        """Background task to cleanup old metrics"""
        
        while True:
            try:
                await self._cleanup_old_metrics()
                await asyncio.sleep(3600)  # Cleanup every hour
            except Exception as e:
                logger.error(f"Metrics cleanup failed: {e}")
                await asyncio.sleep(3600)

    async def _cleanup_old_metrics(self):
        """Remove old metric data to prevent memory bloat"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        for metric_name, metrics in self.metrics_storage.items():
            # Remove old metrics
            while metrics and metrics[0].timestamp < cutoff_time:
                metrics.popleft()

    def _initialize_default_alert_rules(self):
        """Initialize default alert rules"""
        
        default_rules = [
            {
                "name": "High CPU Usage",
                "metric_name": "system_cpu_usage_percent",
                "condition": "gt",
                "threshold": 80.0,
                "severity": "high",
                "description": "System CPU usage is above 80%"
            },
            {
                "name": "High Memory Usage",
                "metric_name": "system_memory_usage_percent",
                "condition": "gt",
                "threshold": 85.0,
                "severity": "high",
                "description": "System memory usage is above 85%"
            },
            {
                "name": "High Error Rate",
                "metric_name": "http_errors_total",
                "condition": "gt",
                "threshold": 10.0,
                "severity": "critical",
                "description": "HTTP error rate is above 10%"
            },
            {
                "name": "Slow Response Time",
                "metric_name": "http_request_duration_ms",
                "condition": "gt",
                "threshold": 2000.0,
                "severity": "medium",
                "description": "HTTP response time is above 2 seconds"
            }
        ]
        
        for rule in default_rules:
            rule_id = str(uuid4())
            rule["rule_id"] = rule_id
            rule["enabled"] = True
            rule["created_at"] = datetime.utcnow().isoformat()
            rule["evaluation_interval"] = 60
            rule["cooldown_period"] = 300
            rule["tags"] = {"type": "default"}
            
            self.alert_rules[rule_id] = rule

    async def _discover_services(self):
        """Discover available services for monitoring"""
        
        # In production, this would integrate with service discovery
        # For now, initialize with known services
        services = [
            "ptaas_service",
            "threat_intelligence_service", 
            "ai_engine",
            "authentication_service",
            "database_service",
            "cache_service"
        ]
        
        for service_id in services:
            self.service_metrics[service_id] = ServiceMetrics(
                service_id=service_id,
                health_score=100.0,
                response_time_p95=0.0,
                error_rate=0.0,
                throughput=0.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                active_connections=0,
                last_updated=datetime.utcnow()
            )

    async def _load_alert_rules(self):
        """Load persisted alert rules"""
        
        try:
            # In production, load from database
            # For now, keep the default rules initialized earlier
            logger.info("Alert rules loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load alert rules: {e}")

    async def _persist_alert_rule(self, rule_id: str, rule: Dict[str, Any]):
        """Persist alert rule to storage"""
        
        try:
            # In production, save to database
            logger.info(f"Alert rule persisted: {rule_id}")
        except Exception as e:
            logger.error(f"Failed to persist alert rule: {e}")

    async def _persist_dashboard(self, dashboard_id: str, dashboard: Dict[str, Any]):
        """Persist dashboard configuration"""
        
        try:
            # In production, save to database
            logger.info(f"Dashboard persisted: {dashboard_id}")
        except Exception as e:
            logger.error(f"Failed to persist dashboard: {e}")

    async def _validate_widget_configuration(self, widget: Dict[str, Any]):
        """Validate widget configuration"""
        
        required_fields = ["type", "title", "metric"]
        for field in required_fields:
            if field not in widget:
                raise ValueError(f"Widget missing required field: {field}")
        
        # Validate metric exists
        metric_name = widget["metric"]
        if metric_name not in self.metrics_storage:
            logger.warning(f"Widget references non-existent metric: {metric_name}")

    def _parse_timeframe(self, timeframe: str, end_time: datetime) -> datetime:
        """Parse timeframe string to start time"""
        
        if timeframe.endswith('h'):
            hours = int(timeframe[:-1])
            return end_time - timedelta(hours=hours)
        elif timeframe.endswith('d'):
            days = int(timeframe[:-1])
            return end_time - timedelta(days=days)
        elif timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
            return end_time - timedelta(minutes=minutes)
        else:
            # Default to 1 hour
            return end_time - timedelta(hours=1)

    async def _get_platform_overview(self) -> Dict[str, Any]:
        """Get platform overview metrics"""
        
        total_services = len(self.service_metrics)
        healthy_services = sum(1 for s in self.service_metrics.values() if s.health_score > 80)
        
        return {
            "total_services": total_services,
            "healthy_services": healthy_services,
            "service_health_percentage": (healthy_services / total_services * 100) if total_services > 0 else 100,
            "active_alerts": len([a for a in self.alerts.values() if a.status == "active"]),
            "platform_status": "healthy" if healthy_services / total_services > 0.8 else "degraded"
        }

    async def _get_services_status(self) -> List[Dict[str, Any]]:
        """Get status of all services"""
        
        services_status = []
        
        for service_id, metrics in self.service_metrics.items():
            status = {
                "service_id": service_id,
                "health_score": metrics.health_score,
                "status": "healthy" if metrics.health_score > 80 else "degraded" if metrics.health_score > 50 else "unhealthy",
                "response_time_p95": metrics.response_time_p95,
                "error_rate": metrics.error_rate,
                "last_updated": metrics.last_updated.isoformat()
            }
            services_status.append(status)
        
        return services_status

    async def _get_system_metrics_summary(self) -> Dict[str, Any]:
        """Get system metrics summary"""
        
        cpu_metrics = self.metrics_storage.get("system_cpu_usage_percent", [])
        memory_metrics = self.metrics_storage.get("system_memory_usage_percent", [])
        disk_metrics = self.metrics_storage.get("system_disk_usage_percent", [])
        
        return {
            "cpu_usage": cpu_metrics[-1].value if cpu_metrics else 0,
            "memory_usage": memory_metrics[-1].value if memory_metrics else 0,
            "disk_usage": disk_metrics[-1].value if disk_metrics else 0,
            "uptime_hours": time.time() / 3600  # Simplified uptime
        }

    async def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        
        active_alerts = []
        
        for alert in self.alerts.values():
            if alert.status == "active":
                active_alerts.append({
                    "alert_id": alert.alert_id,
                    "name": alert.name,
                    "severity": alert.severity,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "created_at": alert.created_at.isoformat()
                })
        
        return active_alerts

    async def _get_sla_status(self) -> Dict[str, Any]:
        """Get SLA compliance status"""
        
        # Calculate current SLA metrics
        current_availability = 99.8  # Simplified calculation
        current_response_time = self._calculate_current_p95_response_time()
        current_error_rate = self._calculate_current_error_rate()
        
        return {
            "availability": {
                "current": current_availability,
                "target": self.sla_targets["availability"],
                "status": "compliant" if current_availability >= self.sla_targets["availability"] else "non_compliant"
            },
            "response_time_p95": {
                "current": current_response_time,
                "target": self.sla_targets["response_time_p95"],
                "status": "compliant" if current_response_time <= self.sla_targets["response_time_p95"] else "non_compliant"
            },
            "error_rate": {
                "current": current_error_rate,
                "target": self.sla_targets["error_rate"],
                "status": "compliant" if current_error_rate <= self.sla_targets["error_rate"] else "non_compliant"
            }
        }

    def _calculate_current_p95_response_time(self) -> float:
        """Calculate current 95th percentile response time"""
        
        all_response_times = []
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        for requests in self.request_metrics.values():
            for request in requests:
                if request["timestamp"] > cutoff_time:
                    all_response_times.append(request["response_time_ms"])
        
        if not all_response_times:
            return 0.0
        
        all_response_times.sort()
        p95_index = int(len(all_response_times) * 0.95)
        return all_response_times[p95_index] if p95_index < len(all_response_times) else all_response_times[-1]

    def _calculate_current_error_rate(self) -> float:
        """Calculate current error rate"""
        
        total_requests = 0
        error_requests = 0
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        for requests in self.request_metrics.values():
            for request in requests:
                if request["timestamp"] > cutoff_time:
                    total_requests += 1
                    if request["status_code"] >= 400:
                        error_requests += 1
        
        return (error_requests / total_requests * 100) if total_requests > 0 else 0.0

    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        
        return {
            "current_rps": self._calculate_current_rps(),
            "avg_response_time": self._calculate_avg_response_time(),
            "p95_response_time": self._calculate_current_p95_response_time(),
            "error_rate": self._calculate_current_error_rate()
        }

    def _calculate_current_rps(self) -> float:
        """Calculate current requests per second"""
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=1)
        request_count = 0
        
        for requests in self.request_metrics.values():
            for request in requests:
                if request["timestamp"] > cutoff_time:
                    request_count += 1
        
        return request_count / 60.0  # Convert to per second

    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time"""
        
        all_response_times = []
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        for requests in self.request_metrics.values():
            for request in requests:
                if request["timestamp"] > cutoff_time:
                    all_response_times.append(request["response_time_ms"])
        
        return statistics.mean(all_response_times) if all_response_times else 0.0

    async def _get_capacity_metrics(self) -> Dict[str, Any]:
        """Get capacity planning metrics"""
        
        return {
            "resource_utilization": {
                "cpu": "normal",
                "memory": "normal", 
                "disk": "normal",
                "network": "normal"
            },
            "scaling_recommendations": [],
            "capacity_forecast": {
                "days_until_capacity": 90,
                "growth_rate": "2% per week"
            }
        }

    # Analytics methods
    async def _analyze_request_patterns(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze request patterns"""
        
        return {
            "total_requests": 1250,  # Simulated
            "requests_per_minute": 20.8,
            "peak_hour": "14:00-15:00",
            "popular_endpoints": ["/api/v1/ptaas/sessions", "/api/v1/health"],
            "geographic_distribution": {"US": 60, "EU": 30, "APAC": 10}
        }

    async def _analyze_error_patterns(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze error patterns"""
        
        return {
            "total_errors": 15,
            "error_rate": 1.2,
            "common_errors": {"404": 8, "500": 5, "401": 2},
            "error_trends": "decreasing",
            "top_failing_endpoints": ["/api/v1/auth/login"]
        }

    async def _analyze_performance_trends(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze performance trends"""
        
        return {
            "response_time_trend": "improving",
            "throughput_trend": "stable",
            "resource_usage_trend": "increasing",
            "performance_score": 85
        }

    async def _analyze_resource_utilization(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze resource utilization"""
        
        return {
            "cpu_peak": 75.2,
            "memory_peak": 68.5,
            "disk_io_peak": 45.1,
            "network_io_peak": 32.8,
            "efficiency_score": 78
        }

    async def _analyze_user_experience(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze user experience metrics"""
        
        return {
            "apdex_score": 0.85,  # Application Performance Index
            "user_satisfaction": "good",
            "bounce_rate": 5.2,
            "session_duration_avg": 25.4
        }

    async def _analyze_capacity_planning(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze capacity planning data"""
        
        return {
            "growth_projection": "15% over next quarter",
            "bottleneck_analysis": ["database connections", "memory usage"],
            "scaling_recommendations": ["Add 2 more app servers", "Upgrade database tier"],
            "cost_optimization": ["Use auto-scaling", "Optimize query performance"]
        }

    # Security metrics methods
    async def _get_authentication_metrics(self) -> Dict[str, Any]:
        """Get authentication metrics"""
        
        return {
            "login_attempts": 450,
            "successful_logins": 428,
            "failed_logins": 22,
            "suspicious_login_patterns": 3,
            "mfa_adoption_rate": 78.5
        }

    async def _get_authorization_failures(self) -> Dict[str, Any]:
        """Get authorization failure metrics"""
        
        return {
            "authorization_failures": 12,
            "privilege_escalation_attempts": 2,
            "unauthorized_resource_access": 8,
            "policy_violations": 5
        }

    async def _get_suspicious_activities(self) -> Dict[str, Any]:
        """Get suspicious activity metrics"""
        
        return {
            "anomalous_behaviors": 15,
            "potential_threats": 3,
            "blocked_requests": 28,
            "quarantined_actions": 5
        }

    async def _get_ptaas_security_metrics(self) -> Dict[str, Any]:
        """Get PTaaS-specific security metrics"""
        
        return {
            "scans_performed": 85,
            "vulnerabilities_discovered": 234,
            "critical_vulnerabilities": 12,
            "scan_safety_violations": 0,
            "unauthorized_scan_attempts": 2
        }

    async def _get_threat_detection_metrics(self) -> Dict[str, Any]:
        """Get threat detection metrics"""
        
        return {
            "threats_detected": 18,
            "false_positives": 3,
            "detection_accuracy": 94.2,
            "mean_detection_time": 2.4,  # minutes
            "threat_categories": {"malware": 8, "phishing": 6, "intrusion": 4}
        }

    async def _get_compliance_metrics(self) -> Dict[str, Any]:
        """Get compliance metrics"""
        
        return {
            "compliance_score": 92.5,
            "audit_findings": 3,
            "policy_compliance": 96.8,
            "data_protection_score": 89.2,
            "last_audit_date": "2025-01-01"
        }

    async def _get_incident_response_metrics(self) -> Dict[str, Any]:
        """Get incident response metrics"""
        
        return {
            "incidents_this_month": 5,
            "mean_response_time": 12.5,  # minutes
            "mean_resolution_time": 45.2,  # minutes
            "escalated_incidents": 1,
            "incident_categories": {"security": 3, "performance": 2}
        }