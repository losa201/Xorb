"""
Advanced Metrics Collection and Export System

This module provides comprehensive metrics collection, custom metric definitions,
and advanced monitoring capabilities for the XORB ecosystem with support for
Prometheus, OpenTelemetry, and custom metric pipelines.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager

import structlog
from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, generate_latest

logger = structlog.get_logger(__name__)


class MetricType(Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


@dataclass
class MetricDefinition:
    """Custom metric definition."""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    unit: Optional[str] = None
    help_text: Optional[str] = None


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    metric_query: str
    threshold: float
    severity: AlertSeverity
    description: str
    duration: int = 300  # seconds
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSample:
    """Individual metric sample."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[float] = None


class IMetricExporter(ABC):
    """Interface for metric exporters."""
    
    @abstractmethod
    async def export_metrics(self, metrics: List[MetricSample]) -> bool:
        """Export metrics to external system."""
        pass
    
    @abstractmethod
    def get_exporter_name(self) -> str:
        """Get exporter name."""
        pass


class PrometheusExporter(IMetricExporter):
    """Prometheus metrics exporter."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.metrics_cache = {}
    
    async def export_metrics(self, metrics: List[MetricSample]) -> bool:
        """Export metrics to Prometheus format."""
        try:
            for metric in metrics:
                prom_metric = self._get_or_create_metric(metric)
                if prom_metric:
                    self._update_metric(prom_metric, metric)
            return True
        except Exception as e:
            logger.error("Failed to export to Prometheus", error=str(e))
            return False
    
    def _get_or_create_metric(self, sample: MetricSample):
        """Get or create Prometheus metric."""
        if sample.name not in self.metrics_cache:
            # Default to Counter for unknown metrics
            self.metrics_cache[sample.name] = Counter(
                sample.name,
                f"XORB metric: {sample.name}",
                labelnames=list(sample.labels.keys()),
                registry=self.registry
            )
        return self.metrics_cache[sample.name]
    
    def _update_metric(self, prom_metric, sample: MetricSample):
        """Update Prometheus metric with sample."""
        if hasattr(prom_metric, 'labels'):
            if sample.labels:
                prom_metric.labels(**sample.labels).inc(sample.value)
            else:
                prom_metric.inc(sample.value)
        else:
            prom_metric.set(sample.value)
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_exporter_name(self) -> str:
        return "prometheus"


class OpenTelemetryExporter(IMetricExporter):
    """OpenTelemetry metrics exporter."""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.otel_metrics = None
        self._initialize_otel()
    
    def _initialize_otel(self):
        """Initialize OpenTelemetry."""
        try:
            from opentelemetry import metrics
            from opentelemetry.exporter.prometheus import PrometheusMetricReader
            from opentelemetry.sdk.metrics import MeterProvider
            
            self.otel_metrics = metrics.get_meter("xorb_ecosystem")
            logger.info("OpenTelemetry initialized")
        except ImportError:
            logger.warning("OpenTelemetry not available")
    
    async def export_metrics(self, metrics: List[MetricSample]) -> bool:
        """Export metrics via OpenTelemetry."""
        if not self.otel_metrics:
            return False
        
        try:
            for metric in metrics:
                # Create and update OTEL metrics
                pass  # Implementation would depend on specific OTEL setup
            return True
        except Exception as e:
            logger.error("Failed to export to OpenTelemetry", error=str(e))
            return False
    
    def get_exporter_name(self) -> str:
        return "opentelemetry"


class CustomMetricCollector:
    """Collector for custom XORB metrics."""
    
    def __init__(self):
        self.metrics_buffer = deque(maxlen=10000)
        self.custom_metrics = {}
        self.metric_definitions = {}
        self._setup_xorb_metrics()
    
    def _setup_xorb_metrics(self):
        """Setup XORB-specific metrics."""
        xorb_metrics = [
            MetricDefinition(
                name="xorb_agent_health_score",
                metric_type=MetricType.GAUGE,
                description="Health score of agents (0-100)",
                labels=["agent_name", "capability"],
                unit="score"
            ),
            MetricDefinition(
                name="xorb_campaign_success_rate",
                metric_type=MetricType.GAUGE,
                description="Campaign success rate percentage",
                labels=["campaign_type", "target_type"],
                unit="percent"
            ),
            MetricDefinition(
                name="xorb_vulnerability_detection_rate",
                metric_type=MetricType.COUNTER,
                description="Rate of vulnerability detections",
                labels=["severity", "category", "agent"]
            ),
            MetricDefinition(
                name="xorb_knowledge_graph_nodes",
                metric_type=MetricType.GAUGE,
                description="Number of nodes in knowledge graph",
                labels=["node_type"]
            ),
            MetricDefinition(
                name="xorb_ml_model_accuracy",
                metric_type=MetricType.GAUGE,
                description="ML model accuracy scores",
                labels=["model_name", "model_version"],
                unit="accuracy"
            ),
            MetricDefinition(
                name="xorb_stealth_detection_rate",
                metric_type=MetricType.GAUGE,
                description="Stealth operation detection rate",
                labels=["stealth_type"],
                unit="percent"
            ),
            MetricDefinition(
                name="xorb_payload_success_rate",
                metric_type=MetricType.HISTOGRAM,
                description="Payload execution success rate",
                labels=["payload_type", "target_system"],
                buckets=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
            ),
            MetricDefinition(
                name="xorb_false_positive_rate",
                metric_type=MetricType.GAUGE,
                description="False positive rate in detections",
                labels=["detection_type", "agent"],
                unit="percent"
            ),
            MetricDefinition(
                name="xorb_coverage_percentage",
                metric_type=MetricType.GAUGE,
                description="Coverage percentage of target assessment",
                labels=["coverage_type", "target_id"],
                unit="percent"
            ),
            MetricDefinition(
                name="xorb_compliance_score",
                metric_type=MetricType.GAUGE,
                description="Compliance score for security frameworks",
                labels=["framework", "category"],
                unit="score"
            )
        ]
        
        for metric_def in xorb_metrics:
            self.register_metric(metric_def)
    
    def register_metric(self, metric_def: MetricDefinition):
        """Register a custom metric definition."""
        self.metric_definitions[metric_def.name] = metric_def
        logger.info("Registered custom metric", name=metric_def.name, type=metric_def.metric_type.value)
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric sample."""
        sample = MetricSample(
            name=name,
            value=value,
            labels=labels or {},
            timestamp=time.time()
        )
        self.metrics_buffer.append(sample)
    
    def get_metrics(self) -> List[MetricSample]:
        """Get all buffered metrics."""
        return list(self.metrics_buffer)
    
    def clear_buffer(self):
        """Clear metrics buffer."""
        self.metrics_buffer.clear()


class AlertManager:
    """Alert management system."""
    
    def __init__(self):
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default XORB alert rules."""
        default_alerts = [
            AlertRule(
                name="high_cpu_utilization",
                metric_query="xorb_cpu_utilization > 90",
                threshold=90.0,
                severity=AlertSeverity.WARNING,
                description="High CPU utilization detected",
                duration=300
            ),
            AlertRule(
                name="agent_failure_rate",
                metric_query="rate(xorb_agent_executions_total{status='failed'}[5m]) > 0.1",
                threshold=0.1,
                severity=AlertSeverity.CRITICAL,
                description="High agent failure rate",
                duration=180
            ),
            AlertRule(
                name="low_agent_health_score",
                metric_query="xorb_agent_health_score < 70",
                threshold=70.0,
                severity=AlertSeverity.WARNING,
                description="Low agent health score",
                duration=600
            ),
            AlertRule(
                name="campaign_failure_spike",
                metric_query="increase(xorb_campaign_operations_total{status='failed'}[10m]) > 5",
                threshold=5.0,
                severity=AlertSeverity.CRITICAL,
                description="Campaign failure spike detected",
                duration=300
            ),
            AlertRule(
                name="knowledge_graph_stale",
                metric_query="time() - xorb_knowledge_graph_last_update > 3600",
                threshold=3600.0,
                severity=AlertSeverity.WARNING,
                description="Knowledge graph not updated recently",
                duration=900
            )
        ]
        
        for alert in default_alerts:
            self.add_alert_rule(alert)
    
    def add_alert_rule(self, alert_rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[alert_rule.name] = alert_rule
        logger.info("Added alert rule", name=alert_rule.name, severity=alert_rule.severity.value)
    
    def evaluate_alerts(self, metrics: List[MetricSample]):
        """Evaluate alert conditions against metrics."""
        for rule_name, rule in self.alert_rules.items():
            try:
                if self._evaluate_rule(rule, metrics):
                    self._trigger_alert(rule)
                else:
                    self._resolve_alert(rule_name)
            except Exception as e:
                logger.error("Error evaluating alert rule", rule=rule_name, error=str(e))
    
    def _evaluate_rule(self, rule: AlertRule, metrics: List[MetricSample]) -> bool:
        """Evaluate a single alert rule."""
        # Simplified evaluation - in production would use proper query engine
        relevant_metrics = [m for m in metrics if rule.name.replace('_', '') in m.name.replace('_', '')]
        
        if not relevant_metrics:
            return False
        
        # Simple threshold check
        latest_value = relevant_metrics[-1].value if relevant_metrics else 0
        return latest_value > rule.threshold
    
    def _trigger_alert(self, rule: AlertRule):
        """Trigger an alert."""
        if rule.name not in self.active_alerts:
            alert_data = {
                'rule': rule,
                'triggered_at': time.time(),
                'status': 'firing'
            }
            self.active_alerts[rule.name] = alert_data
            self.alert_history.append(alert_data.copy())
            
            logger.warning("Alert triggered",
                          alert=rule.name,
                          severity=rule.severity.value,
                          description=rule.description)
    
    def _resolve_alert(self, rule_name: str):
        """Resolve an alert."""
        if rule_name in self.active_alerts:
            alert_data = self.active_alerts[rule_name]
            alert_data['status'] = 'resolved'
            alert_data['resolved_at'] = time.time()
            
            del self.active_alerts[rule_name]
            
            logger.info("Alert resolved", alert=rule_name)
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return list(self.active_alerts.values())


class AdvancedMetricsManager:
    """Advanced metrics management system."""
    
    def __init__(self):
        self.collectors = [CustomMetricCollector()]
        self.exporters = []
        self.alert_manager = AlertManager()
        self.running = False
        
        # Performance tracking
        self.export_duration_histogram = Histogram(
            'xorb_metrics_export_duration_seconds',
            'Time spent exporting metrics'
        )
        self.metrics_processed_counter = Counter(
            'xorb_metrics_processed_total',
            'Total metrics processed'
        )
    
    def add_exporter(self, exporter: IMetricExporter):
        """Add a metrics exporter."""
        self.exporters.append(exporter)
        logger.info("Added metrics exporter", exporter=exporter.get_exporter_name())
    
    def add_collector(self, collector: CustomMetricCollector):
        """Add a metrics collector."""
        self.collectors.append(collector)
        logger.info("Added metrics collector")
    
    async def start_metrics_collection(self):
        """Start metrics collection and export."""
        self.running = True
        
        # Setup default exporters
        prometheus_exporter = PrometheusExporter()
        self.add_exporter(prometheus_exporter)
        
        # Start collection and export tasks
        collection_task = asyncio.create_task(self._collection_loop())
        export_task = asyncio.create_task(self._export_loop())
        alert_task = asyncio.create_task(self._alert_loop())
        
        logger.info("Advanced metrics collection started")
        
        try:
            await asyncio.gather(collection_task, export_task, alert_task)
        except asyncio.CancelledError:
            logger.info("Metrics collection stopped")
    
    async def stop_metrics_collection(self):
        """Stop metrics collection."""
        self.running = False
    
    async def _collection_loop(self):
        """Main metrics collection loop."""
        while self.running:
            try:
                await self._collect_system_metrics()
                await self._collect_application_metrics()
                await asyncio.sleep(30)  # Collect every 30 seconds
            except Exception as e:
                logger.error("Error in metrics collection loop", error=str(e))
                await asyncio.sleep(30)
    
    async def _export_loop(self):
        """Metrics export loop."""
        while self.running:
            try:
                with self.export_duration_histogram.time():
                    await self._export_all_metrics()
                await asyncio.sleep(60)  # Export every minute
            except Exception as e:
                logger.error("Error in metrics export loop", error=str(e))
                await asyncio.sleep(60)
    
    async def _alert_loop(self):
        """Alert evaluation loop."""
        while self.running:
            try:
                all_metrics = []
                for collector in self.collectors:
                    all_metrics.extend(collector.get_metrics())
                
                self.alert_manager.evaluate_alerts(all_metrics)
                await asyncio.sleep(60)  # Check alerts every minute
            except Exception as e:
                logger.error("Error in alert loop", error=str(e))
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            import psutil
            
            # System resource metrics
            collector = self.collectors[0]  # Main collector
            
            collector.record_metric("xorb_system_cpu_percent", psutil.cpu_percent())
            collector.record_metric("xorb_system_memory_percent", psutil.virtual_memory().percent)
            collector.record_metric("xorb_system_disk_percent", psutil.disk_usage('/').percent)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            collector.record_metric("xorb_system_network_bytes_sent", net_io.bytes_sent)
            collector.record_metric("xorb_system_network_bytes_recv", net_io.bytes_recv)
            
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
    
    async def _collect_application_metrics(self):
        """Collect application-specific metrics."""
        try:
            # These would be populated by actual application components
            collector = self.collectors[0]
            
            # Simulate some application metrics
            collector.record_metric("xorb_agent_health_score", 95.0, {"agent_name": "discovery", "capability": "scanning"})
            collector.record_metric("xorb_campaign_success_rate", 87.5, {"campaign_type": "vulnerability_scan", "target_type": "web_app"})
            collector.record_metric("xorb_knowledge_graph_nodes", 1250, {"node_type": "vulnerability"})
            
        except Exception as e:
            logger.error("Failed to collect application metrics", error=str(e))
    
    async def _export_all_metrics(self):
        """Export metrics via all exporters."""
        all_metrics = []
        
        # Collect from all collectors
        for collector in self.collectors:
            metrics = collector.get_metrics()
            all_metrics.extend(metrics)
            collector.clear_buffer()  # Clear after collection
        
        if not all_metrics:
            return
        
        # Export via all exporters
        for exporter in self.exporters:
            try:
                success = await exporter.export_metrics(all_metrics)
                if success:
                    self.metrics_processed_counter.inc(len(all_metrics))
            except Exception as e:
                logger.error("Exporter failed", exporter=exporter.get_exporter_name(), error=str(e))
    
    @contextmanager
    def measure_duration(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for measuring operation duration."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.collectors[0].record_metric(f"{metric_name}_duration_seconds", duration, labels)
    
    def record_business_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a business-specific metric."""
        if self.collectors:
            self.collectors[0].record_metric(name, value, labels)
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        for exporter in self.exporters:
            if isinstance(exporter, PrometheusExporter):
                return exporter.get_metrics_text()
        return ""
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        total_metrics = sum(len(c.get_metrics()) for c in self.collectors)
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            'collectors': len(self.collectors),
            'exporters': len(self.exporters),
            'total_metrics_buffered': total_metrics,
            'active_alerts': len(active_alerts),
            'alert_details': active_alerts,
            'metric_definitions': len(self.collectors[0].metric_definitions) if self.collectors else 0
        }


# Global metrics manager instance
metrics_manager = AdvancedMetricsManager()


async def initialize_metrics():
    """Initialize the advanced metrics system."""
    await metrics_manager.start_metrics_collection()


async def shutdown_metrics():
    """Shutdown the metrics system."""
    await metrics_manager.stop_metrics_collection()


def get_metrics_manager() -> AdvancedMetricsManager:
    """Get the global metrics manager."""
    return metrics_manager