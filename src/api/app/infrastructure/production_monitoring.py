"""
Production Monitoring Infrastructure
Enterprise-grade monitoring, metrics, and observability for XORB Platform
"""

import asyncio
import json
import logging
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque

import httpx
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class MetricType(Enum):
    """Monitoring metric types"""
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
    timestamp: datetime
    source: str
    metadata: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class MonitoringRule:
    """Monitoring rule configuration"""
    rule_id: str
    name: str
    metric_name: str
    condition: str  # e.g., "> 0.8", "< 100"
    threshold: float
    severity: AlertSeverity
    cooldown_minutes: int = 5
    enabled: bool = True

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io_bytes: Dict[str, int]
    load_average: List[float]
    process_count: int
    timestamp: datetime

class ProductionMonitoring:
    """
    Production-grade monitoring and observability system
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        self.alerts: List[Alert] = []
        self.monitoring_rules: List[MonitoringRule] = []
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_handlers: List[Callable] = []
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Initialize core metrics
        self._initialize_core_metrics()
        self._load_monitoring_rules()
        
        # Background monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self.running = False
    
    def _initialize_core_metrics(self):
        """Initialize core system metrics"""
        
        # API Performance Metrics
        self.metrics['api_requests_total'] = Counter(
            'xorb_api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.metrics['api_request_duration'] = Histogram(
            'xorb_api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.metrics['api_active_connections'] = Gauge(
            'xorb_api_active_connections',
            'Active API connections',
            registry=self.registry
        )
        
        # PTaaS Metrics
        self.metrics['ptaas_scans_total'] = Counter(
            'xorb_ptaas_scans_total',
            'Total PTaaS scans',
            ['scan_type', 'status'],
            registry=self.registry
        )
        
        self.metrics['ptaas_scan_duration'] = Histogram(
            'xorb_ptaas_scan_duration_seconds',
            'PTaaS scan duration',
            ['scan_type'],
            registry=self.registry
        )
        
        self.metrics['ptaas_vulnerabilities_found'] = Counter(
            'xorb_ptaas_vulnerabilities_found_total',
            'Vulnerabilities found by PTaaS',
            ['severity', 'scanner'],
            registry=self.registry
        )
        
        # System Metrics
        self.metrics['system_cpu_percent'] = Gauge(
            'xorb_system_cpu_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.metrics['system_memory_percent'] = Gauge(
            'xorb_system_memory_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.metrics['system_disk_percent'] = Gauge(
            'xorb_system_disk_percent',
            'System disk usage percentage',
            registry=self.registry
        )
        
        # Security Metrics
        self.metrics['auth_attempts_total'] = Counter(
            'xorb_auth_attempts_total',
            'Authentication attempts',
            ['result', 'provider'],
            registry=self.registry
        )
        
        self.metrics['security_incidents_total'] = Counter(
            'xorb_security_incidents_total',
            'Security incidents detected',
            ['severity', 'type'],
            registry=self.registry
        )
        
        # Business Metrics
        self.metrics['active_users'] = Gauge(
            'xorb_active_users',
            'Currently active users',
            registry=self.registry
        )
        
        self.metrics['tenant_count'] = Gauge(
            'xorb_tenant_count',
            'Total number of tenants',
            registry=self.registry
        )
    
    def _load_monitoring_rules(self):
        """Load monitoring and alerting rules"""
        
        self.monitoring_rules = [
            # System Performance Rules
            MonitoringRule(
                rule_id="high_cpu_usage",
                name="High CPU Usage",
                metric_name="system_cpu_percent",
                condition=">",
                threshold=80.0,
                severity=AlertSeverity.HIGH,
                cooldown_minutes=5
            ),
            MonitoringRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                metric_name="system_memory_percent",
                condition=">",
                threshold=90.0,
                severity=AlertSeverity.CRITICAL,
                cooldown_minutes=5
            ),
            MonitoringRule(
                rule_id="high_disk_usage",
                name="High Disk Usage",
                metric_name="system_disk_percent",
                condition=">",
                threshold=85.0,
                severity=AlertSeverity.HIGH,
                cooldown_minutes=15
            ),
            
            # API Performance Rules
            MonitoringRule(
                rule_id="high_api_response_time",
                name="High API Response Time",
                metric_name="api_request_duration",
                condition=">",
                threshold=2.0,  # 2 seconds
                severity=AlertSeverity.MEDIUM,
                cooldown_minutes=3
            ),
            
            # Security Rules
            MonitoringRule(
                rule_id="high_failed_auth_rate",
                name="High Failed Authentication Rate",
                metric_name="auth_failures_rate",
                condition=">",
                threshold=10.0,  # 10 failures per minute
                severity=AlertSeverity.HIGH,
                cooldown_minutes=2
            ),
            
            # PTaaS Rules
            MonitoringRule(
                rule_id="critical_vulnerabilities_found",
                name="Critical Vulnerabilities Detected",
                metric_name="ptaas_critical_vulns_rate",
                condition=">",
                threshold=5.0,  # 5 critical vulns in scan
                severity=AlertSeverity.CRITICAL,
                cooldown_minutes=1
            )
        ]
    
    # ========================================================================
    # METRICS COLLECTION
    # ========================================================================
    
    async def start_monitoring(self):
        """Start background monitoring"""
        
        if self.running:
            return
        
        self.running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Production monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        
        self.running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Production monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self.running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Evaluate monitoring rules
                await self._evaluate_monitoring_rules()
                
                # Wait before next collection
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _collect_system_metrics(self):
        """Collect comprehensive system metrics"""
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics['system_cpu_percent'].set(cpu_percent)
            self._record_metric_history('system_cpu_percent', cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.metrics['system_memory_percent'].set(memory_percent)
            self._record_metric_history('system_memory_percent', memory_percent)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.metrics['system_disk_percent'].set(disk_percent)
            self._record_metric_history('system_disk_percent', disk_percent)
            
            # Network I/O
            network_io = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            # Create system metrics snapshot
            system_metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_io_bytes={
                    'bytes_sent': network_io.bytes_sent,
                    'bytes_recv': network_io.bytes_recv
                },
                load_average=list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0, 0, 0],
                process_count=process_count,
                timestamp=datetime.utcnow()
            )
            
            # Store in history
            self._record_metric_history('system_metrics', asdict(system_metrics))
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _record_metric_history(self, metric_name: str, value: Any):
        """Record metric value in history"""
        
        self.metric_history[metric_name].append({
            'timestamp': datetime.utcnow(),
            'value': value
        })
    
    # ========================================================================
    # METRIC RECORDING
    # ========================================================================
    
    def record_api_request(
        self, 
        method: str, 
        endpoint: str, 
        status_code: int, 
        duration: float
    ):
        """Record API request metrics"""
        
        # Request counter
        self.metrics['api_requests_total'].labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code)
        ).inc()
        
        # Duration histogram
        self.metrics['api_request_duration'].labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
        
        # Check for slow requests
        if duration > 2.0:
            asyncio.create_task(self._trigger_alert(
                "slow_api_request",
                f"Slow API request detected: {method} {endpoint} took {duration:.2f}s",
                AlertSeverity.MEDIUM,
                {
                    "method": method,
                    "endpoint": endpoint,
                    "duration": duration,
                    "status_code": status_code
                }
            ))
    
    def record_ptaas_scan(
        self, 
        scan_type: str, 
        status: str, 
        duration: float,
        vulnerabilities: Dict[str, int] = None
    ):
        """Record PTaaS scan metrics"""
        
        # Scan counter
        self.metrics['ptaas_scans_total'].labels(
            scan_type=scan_type,
            status=status
        ).inc()
        
        # Duration histogram
        self.metrics['ptaas_scan_duration'].labels(
            scan_type=scan_type
        ).observe(duration)
        
        # Vulnerability metrics
        if vulnerabilities:
            for severity, count in vulnerabilities.items():
                self.metrics['ptaas_vulnerabilities_found'].labels(
                    severity=severity,
                    scanner=scan_type
                ).inc(count)
                
                # Alert on critical vulnerabilities
                if severity == "critical" and count > 0:
                    asyncio.create_task(self._trigger_alert(
                        "critical_vulnerabilities",
                        f"Critical vulnerabilities found: {count} in {scan_type} scan",
                        AlertSeverity.CRITICAL,
                        {
                            "scan_type": scan_type,
                            "critical_count": count,
                            "total_vulnerabilities": vulnerabilities
                        }
                    ))
    
    def record_auth_attempt(self, result: str, provider: str):
        """Record authentication attempt"""
        
        self.metrics['auth_attempts_total'].labels(
            result=result,
            provider=provider
        ).inc()
    
    def record_security_incident(self, severity: str, incident_type: str):
        """Record security incident"""
        
        self.metrics['security_incidents_total'].labels(
            severity=severity,
            type=incident_type
        ).inc()
        
        # Auto-alert on security incidents
        asyncio.create_task(self._trigger_alert(
            "security_incident",
            f"Security incident detected: {incident_type}",
            AlertSeverity.HIGH if severity == "high" else AlertSeverity.CRITICAL,
            {
                "incident_type": incident_type,
                "severity": severity
            }
        ))
    
    def update_business_metrics(self, active_users: int, tenant_count: int):
        """Update business metrics"""
        
        self.metrics['active_users'].set(active_users)
        self.metrics['tenant_count'].set(tenant_count)
    
    # ========================================================================
    # ALERTING
    # ========================================================================
    
    async def _evaluate_monitoring_rules(self):
        """Evaluate all monitoring rules for alerts"""
        
        for rule in self.monitoring_rules:
            if not rule.enabled:
                continue
            
            try:
                # Check cooldown
                last_alert = self.last_alert_times.get(rule.rule_id)
                if last_alert:
                    cooldown_end = last_alert + timedelta(minutes=rule.cooldown_minutes)
                    if datetime.utcnow() < cooldown_end:
                        continue
                
                # Get current metric value
                current_value = await self._get_metric_value(rule.metric_name)
                if current_value is None:
                    continue
                
                # Evaluate condition
                triggered = self._evaluate_condition(
                    current_value, rule.condition, rule.threshold
                )
                
                if triggered:
                    await self._trigger_rule_alert(rule, current_value)
                
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
    
    async def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value for a metric"""
        
        try:
            if metric_name in self.metric_history:
                history = self.metric_history[metric_name]
                if history:
                    latest = history[-1]
                    value = latest.get('value')
                    
                    # Handle different value types
                    if isinstance(value, (int, float)):
                        return float(value)
                    elif isinstance(value, dict) and 'cpu_percent' in value:
                        return float(value['cpu_percent'])
                    
            return None
            
        except Exception as e:
            logger.error(f"Error getting metric value for {metric_name}: {e}")
            return None
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate monitoring condition"""
        
        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return abs(value - threshold) < 0.001
        
        return False
    
    async def _trigger_rule_alert(self, rule: MonitoringRule, current_value: float):
        """Trigger alert for monitoring rule"""
        
        await self._trigger_alert(
            rule.rule_id,
            f"{rule.name}: {rule.metric_name} is {current_value:.2f} (threshold: {rule.threshold})",
            rule.severity,
            {
                "rule_id": rule.rule_id,
                "metric_name": rule.metric_name,
                "current_value": current_value,
                "threshold": rule.threshold,
                "condition": rule.condition
            }
        )
        
        # Update last alert time
        self.last_alert_times[rule.rule_id] = datetime.utcnow()
    
    async def _trigger_alert(
        self, 
        alert_id: str, 
        description: str, 
        severity: AlertSeverity,
        metadata: Dict[str, Any] = None
    ):
        """Trigger alert"""
        
        alert = Alert(
            alert_id=alert_id,
            name=alert_id.replace("_", " ").title(),
            description=description,
            severity=severity,
            timestamp=datetime.utcnow(),
            source="xorb_monitoring",
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        logger.warning(f"Alert triggered: {alert.name} - {alert.description}")
    
    def add_alert_handler(self, handler: Callable):
        """Add alert handler"""
        self.alert_handlers.append(handler)
    
    # ========================================================================
    # METRICS EXPORT
    # ========================================================================
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "system": {},
            "api": {},
            "ptaas": {},
            "security": {},
            "business": {}
        }
        
        # Get latest system metrics
        if 'system_metrics' in self.metric_history:
            latest_system = self.metric_history['system_metrics']
            if latest_system:
                summary["system"] = latest_system[-1]['value']
        
        # Calculate API metrics
        summary["api"] = {
            "total_requests": self._get_counter_value('api_requests_total'),
            "avg_response_time": self._get_histogram_avg('api_request_duration')
        }
        
        # Calculate PTaaS metrics
        summary["ptaas"] = {
            "total_scans": self._get_counter_value('ptaas_scans_total'),
            "avg_scan_duration": self._get_histogram_avg('ptaas_scan_duration'),
            "vulnerabilities_found": self._get_counter_value('ptaas_vulnerabilities_found')
        }
        
        return summary
    
    def _get_counter_value(self, metric_name: str) -> float:
        """Get counter metric value"""
        try:
            if metric_name in self.metrics:
                metric = self.metrics[metric_name]
                return float(metric._value._value if hasattr(metric, '_value') else 0)
        except:
            pass
        return 0.0
    
    def _get_histogram_avg(self, metric_name: str) -> float:
        """Get histogram average value"""
        try:
            if metric_name in self.metrics:
                metric = self.metrics[metric_name]
                if hasattr(metric, '_sum') and hasattr(metric, '_count'):
                    sum_val = float(metric._sum._value)
                    count_val = float(metric._count._value)
                    return sum_val / count_val if count_val > 0 else 0
        except:
            pass
        return 0.0
    
    # ========================================================================
    # ALERT MANAGEMENT
    # ========================================================================
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        
        active_alerts = []
        for alert in self.alerts:
            if not alert.resolved:
                active_alerts.append({
                    "alert_id": alert.alert_id,
                    "name": alert.name,
                    "description": alert.description,
                    "severity": alert.severity.value,
                    "timestamp": alert.timestamp.isoformat(),
                    "acknowledged": alert.acknowledged,
                    "metadata": alert.metadata
                })
        
        return active_alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
        
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                logger.info(f"Alert resolved: {alert_id}")
                return True
        
        return False
    
    # ========================================================================
    # HEALTH CHECK
    # ========================================================================
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get monitoring system health status"""
        
        return {
            "monitoring_active": self.running,
            "rules_count": len(self.monitoring_rules),
            "active_alerts": len([a for a in self.alerts if not a.resolved]),
            "metrics_collected": len(self.metric_history),
            "last_collection": max(
                [h[-1]['timestamp'] for h in self.metric_history.values() if h],
                default=datetime.utcnow()
            ).isoformat()
        }


# ========================================================================
# ALERT HANDLERS
# ========================================================================

async def webhook_alert_handler(alert: Alert):
    """Send alert via webhook"""
    
    webhook_urls = {
        "slack": os.getenv("SLACK_WEBHOOK_URL"),
        "teams": os.getenv("TEAMS_WEBHOOK_URL"),
        "discord": os.getenv("DISCORD_WEBHOOK_URL"),
        "pagerduty": os.getenv("PAGERDUTY_API_URL")
    }
    
    for service, url in webhook_urls.items():
        if url:
            await _send_webhook_alert(alert, service, url)

async def _send_webhook_alert(alert: Alert, service: str, webhook_url: str):
    """Send alert to specific webhook service"""
    
    try:
        payload = _format_alert_payload(alert, service)
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(webhook_url, json=payload)
            response.raise_for_status()
            
        logger.info(f"Alert sent to {service}: {alert.alert_id}")
        
    except Exception as e:
        logger.error(f"Failed to send alert to {service}: {e}")

def _format_alert_payload(alert: Alert, service: str) -> Dict[str, Any]:
    """Format alert for specific service"""
    
    if service == "slack":
        return {
            "text": f"🚨 XORB Alert: {alert.name}",
            "attachments": [{
                "color": _get_alert_color(alert.severity),
                "fields": [
                    {"title": "Description", "value": alert.description, "short": False},
                    {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                    {"title": "Source", "value": alert.source, "short": True},
                    {"title": "Time", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"), "short": True}
                ]
            }]
        }
    
    elif service == "teams":
        return {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "summary": f"XORB Alert: {alert.name}",
            "themeColor": _get_alert_color(alert.severity),
            "sections": [{
                "activityTitle": f"🚨 {alert.name}",
                "activitySubtitle": alert.description,
                "facts": [
                    {"name": "Severity", "value": alert.severity.value.upper()},
                    {"name": "Source", "value": alert.source},
                    {"name": "Time", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}
                ]
            }]
        }
    
    # Default generic format
    return {
        "alert_id": alert.alert_id,
        "name": alert.name,
        "description": alert.description,
        "severity": alert.severity.value,
        "timestamp": alert.timestamp.isoformat(),
        "source": alert.source,
        "metadata": alert.metadata
    }

def _get_alert_color(severity: AlertSeverity) -> str:
    """Get color for alert severity"""
    
    colors = {
        AlertSeverity.CRITICAL: "#FF0000",  # Red
        AlertSeverity.HIGH: "#FF8C00",      # Orange
        AlertSeverity.MEDIUM: "#FFD700",    # Yellow
        AlertSeverity.LOW: "#00CED1",       # Light Blue
        AlertSeverity.INFO: "#808080"       # Gray
    }
    
    return colors.get(severity, "#808080")