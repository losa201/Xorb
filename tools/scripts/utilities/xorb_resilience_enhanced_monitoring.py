#!/usr/bin/env python3
"""
XORB Resilience & Scalability Layer - Enhanced Monitoring & Alerting
Advanced monitoring with Prometheus metrics, Grafana dashboards, and anomaly detection
"""

import asyncio
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics
import numpy as np
from collections import defaultdict, deque
import uuid
import yaml
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class MetricType(Enum):
    """Prometheus metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AnomalyType(Enum):
    """Types of anomalies detected"""
    SPIKE = "spike"
    DROP = "drop"
    TREND_CHANGE = "trend_change"
    SEASONAL_DEVIATION = "seasonal_deviation"
    OUTLIER = "outlier"

@dataclass
class PrometheusMetric:
    """Prometheus metric definition"""
    name: str
    metric_type: MetricType
    description: str
    labels: Dict[str, str] = field(default_factory=dict)
    value: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SLATarget:
    """Service Level Agreement target"""
    service_name: str
    metric_name: str
    target_value: float
    comparison: str  # ">=", "<=", "==", "!="
    measurement_window: int  # seconds
    alert_threshold: float  # percentage deviation before alert
    description: str = ""

@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    service_name: str
    alert_name: str
    severity: AlertSeverity
    description: str
    condition: str
    value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    escalation_level: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    anomaly_id: str
    service_name: str
    metric_name: str
    anomaly_type: AnomalyType
    severity: float  # 0.0-1.0
    description: str
    detected_at: datetime
    expected_value: float
    actual_value: float
    confidence: float  # 0.0-1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class XORBEnhancedMonitoring:
    """Enhanced monitoring system with Prometheus integration"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.monitoring_id = str(uuid.uuid4())
        
        # Prometheus configuration
        self.prometheus_url = self.config.get('prometheus_url', 'http://localhost:9090')
        self.prometheus_gateway_url = self.config.get('prometheus_gateway_url', 'http://localhost:9091')
        self.metrics_push_interval = self.config.get('metrics_push_interval', 15)
        
        # Grafana configuration
        self.grafana_url = self.config.get('grafana_url', 'http://localhost:3000')
        self.grafana_api_key = self.config.get('grafana_api_key', '')
        
        # Metrics storage
        self.metrics: Dict[str, PrometheusMetric] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # SLA monitoring
        self.sla_targets: Dict[str, SLATarget] = {}
        self.sla_violations: List[Dict[str, Any]] = []
        
        # Alerting
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_rules: List[Dict[str, Any]] = []
        
        # Anomaly detection
        self.anomaly_detectors: Dict[str, Dict[str, Any]] = {}
        self.detected_anomalies: List[AnomalyDetection] = []
        
        # Background task control
        self.monitoring_active = False
        
        # Initialize components
        self._initialize_default_metrics()
        self._initialize_default_sla_targets()
        self._initialize_alert_rules()
        
        logger.info(f"Enhanced Monitoring System initialized: {self.monitoring_id}")
    
    def _initialize_default_metrics(self):
        """Initialize default XORB platform metrics"""
        default_metrics = {
            # System metrics
            'xorb_system_cpu_usage': PrometheusMetric(
                name='xorb_system_cpu_usage',
                metric_type=MetricType.GAUGE,
                description='System CPU usage percentage'
            ),
            'xorb_system_memory_usage': PrometheusMetric(
                name='xorb_system_memory_usage',
                metric_type=MetricType.GAUGE,
                description='System memory usage percentage'
            ),
            'xorb_system_disk_usage': PrometheusMetric(
                name='xorb_system_disk_usage',
                metric_type=MetricType.GAUGE,
                description='System disk usage percentage'
            ),
            
            # Service metrics
            'xorb_service_requests_total': PrometheusMetric(
                name='xorb_service_requests_total',
                metric_type=MetricType.COUNTER,
                description='Total number of service requests'
            ),
            'xorb_service_request_duration_seconds': PrometheusMetric(
                name='xorb_service_request_duration_seconds',
                metric_type=MetricType.HISTOGRAM,
                description='Service request duration in seconds'
            ),
            'xorb_service_errors_total': PrometheusMetric(
                name='xorb_service_errors_total',
                metric_type=MetricType.COUNTER,
                description='Total number of service errors'
            ),
            'xorb_service_availability': PrometheusMetric(
                name='xorb_service_availability',
                metric_type=MetricType.GAUGE,
                description='Service availability percentage'
            ),
            
            # Load balancer metrics
            'xorb_loadbalancer_requests_total': PrometheusMetric(
                name='xorb_loadbalancer_requests_total',
                metric_type=MetricType.COUNTER,
                description='Total load balancer requests'
            ),
            'xorb_loadbalancer_response_time_ms': PrometheusMetric(
                name='xorb_loadbalancer_response_time_ms',
                metric_type=MetricType.GAUGE,
                description='Load balancer response time in milliseconds'
            ),
            
            # Circuit breaker metrics
            'xorb_circuit_breaker_state': PrometheusMetric(
                name='xorb_circuit_breaker_state',
                metric_type=MetricType.GAUGE,
                description='Circuit breaker state (0=closed, 1=half-open, 2=open)'
            ),
            'xorb_circuit_breaker_failures_total': PrometheusMetric(
                name='xorb_circuit_breaker_failures_total',
                metric_type=MetricType.COUNTER,
                description='Total circuit breaker failures'
            ),
            
            # Replication metrics
            'xorb_replication_lag_ms': PrometheusMetric(
                name='xorb_replication_lag_ms',
                metric_type=MetricType.GAUGE,
                description='Data replication lag in milliseconds'
            ),
            'xorb_replication_success_total': PrometheusMetric(
                name='xorb_replication_success_total',
                metric_type=MetricType.COUNTER,
                description='Total successful replications'
            ),
            
            # AI/ML metrics
            'xorb_agent_performance_score': PrometheusMetric(
                name='xorb_agent_performance_score',
                metric_type=MetricType.GAUGE,
                description='AI agent performance score'
            ),
            'xorb_evolution_events_total': PrometheusMetric(
                name='xorb_evolution_events_total',
                metric_type=MetricType.COUNTER,
                description='Total agent evolution events'
            ),
            'xorb_threat_detections_total': PrometheusMetric(
                name='xorb_threat_detections_total',
                metric_type=MetricType.COUNTER,
                description='Total threat detections'
            )
        }
        
        self.metrics.update(default_metrics)
    
    def _initialize_default_sla_targets(self):
        """Initialize default SLA targets"""
        default_slas = [
            SLATarget(
                service_name="neural_orchestrator",
                metric_name="response_time_ms",
                target_value=200.0,
                comparison="<=",
                measurement_window=300,
                alert_threshold=20.0,
                description="Neural orchestrator response time SLA"
            ),
            SLATarget(
                service_name="threat_detection",
                metric_name="availability",
                target_value=99.9,
                comparison=">=",
                measurement_window=3600,
                alert_threshold=1.0,
                description="Threat detection availability SLA"
            ),
            SLATarget(
                service_name="evolution_accelerator",
                metric_name="success_rate",
                target_value=95.0,
                comparison=">=",
                measurement_window=1800,
                alert_threshold=5.0,
                description="Evolution accelerator success rate SLA"
            )
        ]
        
        for sla in default_slas:
            self.sla_targets[f"{sla.service_name}_{sla.metric_name}"] = sla
    
    def _initialize_alert_rules(self):
        """Initialize default alert rules"""
        self.alert_rules = [
            {
                'name': 'HighCPUUsage',
                'condition': 'xorb_system_cpu_usage > 90',
                'severity': AlertSeverity.HIGH,
                'description': 'System CPU usage is critically high',
                'for': '5m'
            },
            {
                'name': 'HighMemoryUsage',
                'condition': 'xorb_system_memory_usage > 85',
                'severity': AlertSeverity.HIGH,
                'description': 'System memory usage is critically high',
                'for': '5m'
            },
            {
                'name': 'ServiceDown',
                'condition': 'xorb_service_availability < 50',
                'severity': AlertSeverity.CRITICAL,
                'description': 'Service is down or severely degraded',
                'for': '1m'
            },
            {
                'name': 'HighErrorRate',
                'condition': 'rate(xorb_service_errors_total[5m]) > 0.1',
                'severity': AlertSeverity.MEDIUM,
                'description': 'Service error rate is elevated',
                'for': '3m'
            },
            {
                'name': 'CircuitBreakerOpen',
                'condition': 'xorb_circuit_breaker_state == 2',
                'severity': AlertSeverity.HIGH,
                'description': 'Circuit breaker is in open state',
                'for': '1m'
            },
            {
                'name': 'ReplicationLagHigh',
                'condition': 'xorb_replication_lag_ms > 5000',
                'severity': AlertSeverity.MEDIUM,
                'description': 'Data replication lag is high',
                'for': '2m'
            }
        ]
    
    async def update_metric(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Update a metric value"""
        try:
            if metric_name in self.metrics:
                metric = self.metrics[metric_name]
                metric.value = value
                metric.timestamp = datetime.now()
                if labels:
                    metric.labels.update(labels)
                
                # Store in history
                self.metric_history[metric_name].append({
                    'timestamp': metric.timestamp,
                    'value': value,
                    'labels': labels or {}
                })
                
                # Check for anomalies
                await self._check_anomalies(metric_name, value)
                
                # Check SLA violations
                await self._check_sla_violations(metric_name, value)
                
            else:
                logger.warning(f"Unknown metric: {metric_name}")
                
        except Exception as e:
            logger.error(f"Failed to update metric {metric_name}: {e}")
    
    async def add_custom_metric(self, service_name: str, metric_name: str, value: float, 
                              metadata: Optional[Dict[str, Any]] = None):
        """Add custom service metric"""
        try:
            custom_key = f"{service_name}_{metric_name}"
            
            metric_data = {
                'timestamp': datetime.now(),
                'service_name': service_name,
                'metric_name': metric_name,
                'value': value,
                'metadata': metadata or {}
            }
            
            self.custom_metrics[custom_key].append(metric_data)
            
            # Also update in main metrics if it exists
            full_metric_name = f"xorb_{custom_key}"
            if full_metric_name in self.metrics:
                await self.update_metric(full_metric_name, value)
            
            logger.debug(f"Added custom metric: {custom_key} = {value}")
            
        except Exception as e:
            logger.error(f"Failed to add custom metric {service_name}.{metric_name}: {e}")
    
    async def _check_anomalies(self, metric_name: str, value: float):
        """Check for anomalies in metric values"""
        try:
            if metric_name not in self.metric_history:
                return
            
            history = list(self.metric_history[metric_name])
            if len(history) < 10:  # Need sufficient history
                return
            
            # Get recent values
            recent_values = [h['value'] for h in history[-30:]]  # Last 30 measurements
            
            if len(recent_values) < 10:
                return
            
            # Calculate statistical measures
            mean = statistics.mean(recent_values)
            std_dev = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
            
            # Define anomaly thresholds
            spike_threshold = mean + (3 * std_dev)
            drop_threshold = mean - (3 * std_dev)
            
            anomaly_detected = False
            anomaly_type = None
            severity = 0.0
            
            # Check for spike
            if value > spike_threshold and std_dev > 0:
                anomaly_type = AnomalyType.SPIKE
                severity = min(1.0, (value - spike_threshold) / (spike_threshold * 0.5))
                anomaly_detected = True
            
            # Check for drop
            elif value < drop_threshold and std_dev > 0:
                anomaly_type = AnomalyType.DROP
                severity = min(1.0, (drop_threshold - value) / (drop_threshold * 0.5))
                anomaly_detected = True
            
            # Check for trend change (simplified)
            if len(recent_values) >= 20:
                first_half = recent_values[:10]
                second_half = recent_values[-10:]
                
                first_mean = statistics.mean(first_half)
                second_mean = statistics.mean(second_half)
                
                # Significant trend change detection
                if abs(second_mean - first_mean) > (std_dev * 2) and std_dev > 0:
                    anomaly_type = AnomalyType.TREND_CHANGE
                    severity = min(1.0, abs(second_mean - first_mean) / (mean * 0.3))
                    anomaly_detected = True
            
            # Create anomaly if detected
            if anomaly_detected and severity > 0.3:  # Only report significant anomalies
                anomaly = AnomalyDetection(
                    anomaly_id=str(uuid.uuid4()),
                    service_name=metric_name.split('_')[1] if '_' in metric_name else 'system',
                    metric_name=metric_name,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    description=f"{anomaly_type.value.title()} detected in {metric_name}",
                    detected_at=datetime.now(),
                    expected_value=mean,
                    actual_value=value,
                    confidence=min(1.0, severity * 1.2)
                )
                
                self.detected_anomalies.append(anomaly)
                
                # Keep only recent anomalies
                if len(self.detected_anomalies) > 100:
                    self.detected_anomalies = self.detected_anomalies[-100:]
                
                # Generate alert for high-severity anomalies
                if severity > 0.7:
                    await self._create_anomaly_alert(anomaly)
                
                logger.warning(f"Anomaly detected in {metric_name}: {anomaly_type.value} (severity: {severity:.2f})")
                
        except Exception as e:
            logger.error(f"Failed to check anomalies for {metric_name}: {e}")
    
    async def _check_sla_violations(self, metric_name: str, value: float):
        """Check for SLA violations"""
        try:
            for sla_key, sla in self.sla_targets.items():
                if sla.metric_name in metric_name.lower():
                    # Check if value violates SLA
                    violation = False
                    
                    if sla.comparison == ">=" and value < sla.target_value:
                        violation = True
                    elif sla.comparison == "<=" and value > sla.target_value:
                        violation = True
                    elif sla.comparison == "==" and abs(value - sla.target_value) > (sla.target_value * 0.01):
                        violation = True
                    elif sla.comparison == "!=" and abs(value - sla.target_value) < (sla.target_value * 0.01):
                        violation = True
                    
                    if violation:
                        # Calculate deviation percentage
                        deviation = abs(value - sla.target_value) / sla.target_value * 100
                        
                        if deviation > sla.alert_threshold:
                            violation_record = {
                                'sla_key': sla_key,
                                'service_name': sla.service_name,
                                'metric_name': sla.metric_name,
                                'target_value': sla.target_value,
                                'actual_value': value,
                                'deviation_percent': deviation,
                                'timestamp': datetime.now(),
                                'description': sla.description
                            }
                            
                            self.sla_violations.append(violation_record)
                            
                            # Keep only recent violations
                            if len(self.sla_violations) > 100:
                                self.sla_violations = self.sla_violations[-100:]
                            
                            # Create alert
                            await self._create_sla_violation_alert(violation_record)
                            
                            logger.warning(f"SLA violation detected: {sla.service_name}.{sla.metric_name} = {value} (target: {sla.target_value})")
                            
        except Exception as e:
            logger.error(f"Failed to check SLA violations: {e}")
    
    async def _create_anomaly_alert(self, anomaly: AnomalyDetection):
        """Create alert for anomaly detection"""
        try:
            alert_id = f"anomaly_{anomaly.anomaly_id}"
            
            # Determine severity based on anomaly severity
            if anomaly.severity >= 0.9:
                severity = AlertSeverity.CRITICAL
            elif anomaly.severity >= 0.7:
                severity = AlertSeverity.HIGH
            elif anomaly.severity >= 0.5:
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW
            
            alert = Alert(
                alert_id=alert_id,
                service_name=anomaly.service_name,
                alert_name=f"Anomaly: {anomaly.anomaly_type.value}",
                severity=severity,
                description=f"{anomaly.description} (confidence: {anomaly.confidence:.1%})",
                condition=f"{anomaly.metric_name} {anomaly.anomaly_type.value}",
                value=anomaly.actual_value,
                threshold=anomaly.expected_value,
                timestamp=anomaly.detected_at,
                metadata={'anomaly_id': anomaly.anomaly_id, 'confidence': anomaly.confidence}
            )
            
            await self._process_alert(alert)
            
        except Exception as e:
            logger.error(f"Failed to create anomaly alert: {e}")
    
    async def _create_sla_violation_alert(self, violation: Dict[str, Any]):
        """Create alert for SLA violation"""
        try:
            alert_id = f"sla_{violation['sla_key']}_{int(time.time())}"
            
            # Determine severity based on deviation
            deviation = violation['deviation_percent']
            if deviation >= 50:
                severity = AlertSeverity.CRITICAL
            elif deviation >= 25:
                severity = AlertSeverity.HIGH
            elif deviation >= 10:
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW
            
            alert = Alert(
                alert_id=alert_id,
                service_name=violation['service_name'],
                alert_name="SLA Violation",
                severity=severity,
                description=f"SLA violated: {violation['description']} (deviation: {deviation:.1f}%)",
                condition=f"{violation['metric_name']} SLA violation",
                value=violation['actual_value'],
                threshold=violation['target_value'],
                timestamp=violation['timestamp'],
                metadata={'sla_key': violation['sla_key'], 'deviation_percent': deviation}
            )
            
            await self._process_alert(alert)
            
        except Exception as e:
            logger.error(f"Failed to create SLA violation alert: {e}")
    
    async def _process_alert(self, alert: Alert):
        """Process and store alert"""
        try:
            # Check if similar alert already exists
            existing_alert = None
            for existing_id, existing in self.active_alerts.items():
                if (existing.service_name == alert.service_name and 
                    existing.alert_name == alert.alert_name and 
                    not existing.resolved):
                    existing_alert = existing
                    break
            
            if existing_alert:
                # Update existing alert
                existing_alert.escalation_level += 1
                existing_alert.timestamp = alert.timestamp
                existing_alert.value = alert.value
                logger.info(f"Updated existing alert: {existing_alert.alert_id} (escalation level: {existing_alert.escalation_level})")
            else:
                # Create new alert
                self.active_alerts[alert.alert_id] = alert
                self.alert_history.append(asdict(alert))
                
                # Send notification (placeholder)
                await self._send_alert_notification(alert)
                
                logger.warning(f"New alert created: {alert.alert_id} - {alert.description}")
            
        except Exception as e:
            logger.error(f"Failed to process alert: {e}")
    
    async def _send_alert_notification(self, alert: Alert):
        """Send alert notification (placeholder implementation)"""
        try:
            # In a real implementation, this would send notifications via:
            # - Email
            # - Slack/Teams
            # - PagerDuty
            # - SMS
            
            notification_data = {
                'alert_id': alert.alert_id,
                'service_name': alert.service_name,
                'severity': alert.severity.value,
                'description': alert.description,
                'timestamp': alert.timestamp.isoformat(),
                'value': alert.value,
                'threshold': alert.threshold
            }
            
            logger.info(f"Alert notification sent: {alert.alert_id}")
            # print(f"📢 ALERT: {alert.severity.value.upper()} - {alert.description}")
            
        except Exception as e:
            logger.error(f"Failed to send alert notification: {e}")
    
    async def resolve_alert(self, alert_id: str, resolution_note: Optional[str] = None) -> bool:
        """Resolve an active alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                if resolution_note:
                    alert.metadata['resolution_note'] = resolution_note
                
                logger.info(f"Alert resolved: {alert_id}")
                return True
            else:
                logger.warning(f"Alert not found for resolution: {alert_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False
    
    async def push_metrics_to_prometheus(self):
        """Push metrics to Prometheus Push Gateway"""
        try:
            if not self.prometheus_gateway_url:
                return
            
            # Prepare metrics for push
            metrics_data = []
            
            for metric_name, metric in self.metrics.items():
                metric_line = f"{metric_name}"
                
                # Add labels
                if metric.labels:
                    label_pairs = [f'{k}="{v}"' for k, v in metric.labels.items()]
                    metric_line += "{" + ",".join(label_pairs) + "}"
                
                metric_line += f" {metric.value} {int(metric.timestamp.timestamp() * 1000)}"
                metrics_data.append(metric_line)
            
            # Push to gateway
            if metrics_data:
                payload = "\n".join(metrics_data)
                
                response = requests.post(
                    f"{self.prometheus_gateway_url}/metrics/job/xorb_platform",
                    data=payload,
                    headers={'Content-Type': 'text/plain'},
                    timeout=10
                )
                
                if response.status_code == 200:
                    logger.debug("Metrics pushed to Prometheus successfully")
                else:
                    logger.error(f"Failed to push metrics to Prometheus: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Failed to push metrics to Prometheus: {e}")
    
    async def create_grafana_dashboard(self, dashboard_name: str, service_names: List[str]) -> bool:
        """Create Grafana dashboard for XORB services"""
        try:
            if not self.grafana_api_key:
                logger.warning("Grafana API key not configured")
                return False
            
            # Dashboard configuration
            dashboard_config = {
                "dashboard": {
                    "title": dashboard_name,
                    "tags": ["xorb", "monitoring"],
                    "timezone": "browser",
                    "panels": [],
                    "time": {
                        "from": "now-1h",
                        "to": "now"
                    },
                    "refresh": "30s"
                }
            }
            
            # Add panels for each service
            panel_id = 1
            
            for service_name in service_names:
                # Response time panel
                response_time_panel = {
                    "id": panel_id,
                    "title": f"{service_name} - Response Time",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": (panel_id - 1) * 8},
                    "targets": [{
                        "expr": f'xorb_service_request_duration_seconds{{service="{service_name}"}}',
                        "legendFormat": "Response Time"
                    }],
                    "yAxes": [{
                        "label": "Seconds",
                        "min": 0
                    }, {
                        "show": False
                    }]
                }
                
                # Error rate panel
                error_rate_panel = {
                    "id": panel_id + 1,
                    "title": f"{service_name} - Error Rate",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": (panel_id - 1) * 8},
                    "targets": [{
                        "expr": f'rate(xorb_service_errors_total{{service="{service_name}"}}[5m])',
                        "legendFormat": "Error Rate"
                    }],
                    "yAxes": [{
                        "label": "Errors/sec",
                        "min": 0
                    }, {
                        "show": False
                    }]
                }
                
                dashboard_config["dashboard"]["panels"].extend([response_time_panel, error_rate_panel])
                panel_id += 2
            
            # Create dashboard via Grafana API
            headers = {
                'Authorization': f'Bearer {self.grafana_api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f"{self.grafana_url}/api/dashboards/db",
                json=dashboard_config,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"Grafana dashboard created: {dashboard_name}")
                return True
            else:
                logger.error(f"Failed to create Grafana dashboard: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to create Grafana dashboard: {e}")
            return False
    
    async def export_prometheus_config(self, output_path: str) -> bool:
        """Export Prometheus configuration"""
        try:
            config = {
                'global': {
                    'scrape_interval': '15s',
                    'evaluation_interval': '15s'
                },
                'rule_files': [
                    'xorb_alerts.yml'
                ],
                'scrape_configs': [
                    {
                        'job_name': 'xorb-platform',
                        'static_configs': [{
                            'targets': [
                                'localhost:8003',  # neural_orchestrator
                                'localhost:8004',  # learning_service
                                'localhost:8005',  # threat_detection
                                'localhost:8006',  # agent_cluster
                                'localhost:8007',  # intelligence_fusion
                                'localhost:8008'   # evolution_accelerator
                            ]
                        }],
                        'scrape_interval': '15s',
                        'metrics_path': '/metrics'
                    },
                    {
                        'job_name': 'pushgateway',
                        'static_configs': [{
                            'targets': ['localhost:9091']
                        }]
                    }
                ],
                'alerting': {
                    'alertmanagers': [{
                        'static_configs': [{
                            'targets': ['localhost:9093']
                        }]
                    }]
                }
            }
            
            # Write configuration file
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Prometheus configuration exported: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export Prometheus config: {e}")
            return False
    
    async def export_alert_rules(self, output_path: str) -> bool:
        """Export Prometheus alert rules"""
        try:
            alert_rules = {
                'groups': [{
                    'name': 'xorb_alerts',
                    'rules': []
                }]
            }
            
            for rule in self.alert_rules:
                alert_rule = {
                    'alert': rule['name'],
                    'expr': rule['condition'],
                    'for': rule.get('for', '1m'),
                    'labels': {
                        'severity': rule['severity'].value
                    },
                    'annotations': {
                        'summary': rule['description'],
                        'description': f"{{ $labels.instance }} {rule['description']}"
                    }
                }
                alert_rules['groups'][0]['rules'].append(alert_rule)
            
            # Write alert rules file
            with open(output_path, 'w') as f:
                yaml.dump(alert_rules, f, default_flow_style=False)
            
            logger.info(f"Alert rules exported: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export alert rules: {e}")
            return False
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        try:
            # Active alerts by severity
            alert_counts = defaultdict(int)
            for alert in self.active_alerts.values():
                if not alert.resolved:
                    alert_counts[alert.severity.value] += 1
            
            # Recent anomalies
            recent_anomalies = [
                anomaly for anomaly in self.detected_anomalies
                if (datetime.now() - anomaly.detected_at).total_seconds() < 3600
            ]
            
            # SLA violations
            recent_sla_violations = [
                violation for violation in self.sla_violations
                if (datetime.now() - violation['timestamp']).total_seconds() < 3600
            ]
            
            # Metric statistics
            metric_stats = {}
            for metric_name, metric in self.metrics.items():
                history = list(self.metric_history[metric_name])
                if history:
                    values = [h['value'] for h in history[-20:]]  # Last 20 measurements
                    metric_stats[metric_name] = {
                        'current_value': metric.value,
                        'avg_value': statistics.mean(values) if values else 0,
                        'min_value': min(values) if values else 0,
                        'max_value': max(values) if values else 0,
                        'measurement_count': len(history)
                    }
            
            return {
                'monitoring_id': self.monitoring_id,
                'monitoring_active': self.monitoring_active,
                'configuration': {
                    'prometheus_url': self.prometheus_url,
                    'grafana_url': self.grafana_url,
                    'metrics_push_interval': self.metrics_push_interval
                },
                'metrics': {
                    'total_metrics': len(self.metrics),
                    'custom_metrics': len(self.custom_metrics),
                    'metric_statistics': metric_stats
                },
                'alerting': {
                    'active_alerts': dict(alert_counts),
                    'total_active_alerts': len([a for a in self.active_alerts.values() if not a.resolved]),
                    'alert_rules_count': len(self.alert_rules),
                    'recent_alerts_1h': len([a for a in self.alert_history if (datetime.now() - datetime.fromisoformat(a['timestamp'])).total_seconds() < 3600])
                },
                'sla_monitoring': {
                    'sla_targets_count': len(self.sla_targets),
                    'recent_violations_1h': len(recent_sla_violations),
                    'violation_services': list(set([v['service_name'] for v in recent_sla_violations]))
                },
                'anomaly_detection': {
                    'recent_anomalies_1h': len(recent_anomalies),
                    'anomaly_types': list(set([a.anomaly_type.value for a in recent_anomalies])),
                    'high_severity_anomalies': len([a for a in recent_anomalies if a.severity > 0.7])
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get monitoring status: {e}")
            return {'error': str(e)}
    
    async def start_monitoring(self):
        """Start background monitoring tasks"""
        try:
            self.monitoring_active = True
            
            # Metrics push task
            async def metrics_push_loop():
                while self.monitoring_active:
                    await self.push_metrics_to_prometheus()
                    await asyncio.sleep(self.metrics_push_interval)
            
            # Start background tasks
            asyncio.create_task(metrics_push_loop())
            
            logger.info("Enhanced monitoring system started")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
    
    async def stop_monitoring(self):
        """Stop monitoring tasks"""
        self.monitoring_active = False
        logger.info("Enhanced monitoring system stopped")

# Example usage and testing
async def main():
    """Example usage of XORB Enhanced Monitoring"""
    try:
        print("📊 XORB Enhanced Monitoring System initializing...")
        
        # Initialize monitoring system
        monitoring = XORBEnhancedMonitoring({
            'prometheus_url': 'http://localhost:9090',
            'grafana_url': 'http://localhost:3000',
            'metrics_push_interval': 15
        })
        
        # Start monitoring
        await monitoring.start_monitoring()
        
        print("✅ Enhanced monitoring system started")
        
        # Simulate metric updates
        print("\n📈 Simulating metric updates...")
        
        # System metrics
        await monitoring.update_metric('xorb_system_cpu_usage', 45.5)
        await monitoring.update_metric('xorb_system_memory_usage', 62.3)
        await monitoring.update_metric('xorb_system_disk_usage', 78.9)
        
        # Service metrics
        await monitoring.update_metric('xorb_service_requests_total', 1234)
        await monitoring.update_metric('xorb_service_request_duration_seconds', 0.145)
        await monitoring.update_metric('xorb_service_availability', 99.8)
        
        # Custom metrics
        await monitoring.add_custom_metric('neural_orchestrator', 'decisions_per_second', 25.4)
        await monitoring.add_custom_metric('threat_detection', 'threats_detected', 3)
        await monitoring.add_custom_metric('evolution_accelerator', 'evolutions_completed', 12)
        
        print("✅ Metrics updated successfully")
        
        # Simulate anomaly (high CPU)
        print("\n🚨 Simulating anomaly detection...")
        await monitoring.update_metric('xorb_system_cpu_usage', 95.5)  # Trigger high CPU alert
        
        # Wait a bit for processing
        await asyncio.sleep(1)
        
        # Export configurations
        print("\n📄 Exporting monitoring configurations...")
        await monitoring.export_prometheus_config('/tmp/prometheus.yml')
        await monitoring.export_alert_rules('/tmp/xorb_alerts.yml')
        
        # Get monitoring status
        status = await monitoring.get_monitoring_status()
        print(f"\n📊 Monitoring Status:")
        print(f"- Total Metrics: {status['metrics']['total_metrics']}")
        print(f"- Custom Metrics: {status['metrics']['custom_metrics']}")
        print(f"- Active Alerts: {status['alerting']['total_active_alerts']}")
        print(f"- SLA Targets: {status['sla_monitoring']['sla_targets_count']}")
        print(f"- Recent Anomalies: {status['anomaly_detection']['recent_anomalies_1h']}")
        
        # Show alert details
        if status['alerting']['total_active_alerts'] > 0:
            print(f"\n🚨 Active Alerts:")
            for severity, count in status['alerting']['active_alerts'].items():
                if count > 0:
                    print(f"  - {severity.upper()}: {count}")
        
        print(f"\n✅ XORB Enhanced Monitoring System demonstration completed!")
        
        # Stop monitoring
        await monitoring.stop_monitoring()
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())