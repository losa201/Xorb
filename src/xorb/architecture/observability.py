#!/usr/bin/env python3
"""
XORB Advanced Monitoring and Observability Stack
Comprehensive telemetry integration with EPYC optimization and security focus
"""

import asyncio
import logging
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
import json
import uuid
import inspect
from functools import wraps
from contextlib import asynccontextmanager

# Simplified imports to avoid dependency issues
try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    # Skip aioredis for now due to compatibility issue
    aioredis = None
except ImportError:
    aioredis = None

# Functional mock implementations for unavailable dependencies
class MockCounter:
    def __init__(self, name, description, labelnames=None, registry=None):
        self.name = name
        self._value = 0

    def inc(self, amount=1):
        self._value += amount
        logger.debug(f"Counter {self.name}: {self._value}")

    def labels(self, **kwargs):
        return self

class MockHistogram:
    def __init__(self, name, description, labelnames=None, buckets=None, registry=None):
        self.name = name
        self._observations = []

    def observe(self, value):
        self._observations.append(value)
        logger.debug(f"Histogram {self.name}: observed {value}")

    def time(self):
        return MockContextManager(self)

    def labels(self, **kwargs):
        return self

class MockGauge:
    def __init__(self, name, description, labelnames=None, registry=None):
        self.name = name
        self._value = 0

    def set(self, value):
        self._value = value
        logger.debug(f"Gauge {self.name}: {value}")

    def inc(self, amount=1):
        self._value += amount

    def dec(self, amount=1):
        self._value -= amount

    def labels(self, **kwargs):
        return self

class MockContextManager:
    def __init__(self, histogram=None):
        self.histogram = histogram
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        if self.histogram and self.start_time:
            duration = time.time() - self.start_time
            self.histogram.observe(duration)

# Use mock implementations if real ones aren't available
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
except ImportError:
    Counter = MockCounter
    Histogram = MockHistogram
    Gauge = MockGauge
    Summary = MockHistogram
    CollectorRegistry = object
    def generate_latest(*args): return b"# Mock metrics"

# Mock OpenTelemetry if not available
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
    from opentelemetry.propagate import extract, inject
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.semantic_conventions.trace import SpanAttributes
except ImportError:
    class MockTracer:
        def start_span(self, name, **kwargs):
            return MockSpan()
        def get_current_span(self):
            return MockSpan()

    class MockSpan:
        def __init__(self):
            self.attributes = {}
            self.status = None
            self.start_time = time.time()
            self.end_time = None

        def set_attribute(self, key, value):
            self.attributes[key] = value
            logger.debug(f"Span attribute: {key} = {value}")

        def set_status(self, status):
            self.status = status
            logger.debug(f"Span status: {status}")

        def end(self):
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            logger.debug(f"Span completed in {duration:.3f}s")

        def __enter__(self): return self
        def __exit__(self, *args):
            self.end()

    class MockTrace:
        def get_tracer(self, name): return MockTracer()

    trace = MockTrace()
    SpanAttributes = type('SpanAttributes', (), {})()
    TracerProvider = object
    BatchSpanProcessor = object
    JaegerExporter = object
    MeterProvider = object
    PrometheusMetricReader = object

logger = logging.getLogger(__name__)

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class TraceLevel(Enum):
    MINIMAL = "minimal"        # Basic request/response tracking
    STANDARD = "standard"      # Include key operations
    DETAILED = "detailed"      # Include database queries, cache operations
    COMPREHENSIVE = "comprehensive"  # Include all operations and spans

@dataclass
class MetricDefinition:
    """Definition of a custom metric."""
    name: str
    type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    unit: str = "1"

@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    metric_query: str
    threshold: float
    severity: AlertSeverity
    duration: timedelta = timedelta(minutes=5)
    description: str = ""
    runbook_url: str = ""
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class TraceContext:
    """Trace context for request correlation."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    service_name: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)

class XORBTracer:
    """XORB-specific distributed tracing implementation."""

    def __init__(self, service_name: str, jaeger_endpoint: str = "http://jaeger:14268/api/traces"):
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint

        # Setup OpenTelemetry tracing
        trace.set_tracer_provider(TracerProvider())
        self.tracer = trace.get_tracer(__name__)

        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name="jaeger",
            agent_port=6831,
        )

        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)

        # Instrument async libraries
        AsyncioInstrumentor().instrument()
        AioHttpClientInstrumentor().instrument()

        # Active spans tracking
        self.active_spans: Dict[str, Any] = {}

        # Metrics for tracing
        self.traces_created = Counter(
            'xorb_traces_created_total',
            'Total traces created',
            ['service', 'operation']
        )
        self.span_duration = Histogram(
            'xorb_span_duration_seconds',
            'Span duration in seconds',
            ['service', 'operation', 'status']
        )

    @asynccontextmanager
    async def start_span(
        self,
        operation_name: str,
        parent_context: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, Any]] = None
    ):
        """Start a new trace span with context management."""
        span = self.tracer.start_span(operation_name)

        # Set span attributes
        span.set_attribute(SpanAttributes.SERVICE_NAME, self.service_name)
        span.set_attribute("operation.name", operation_name)

        if tags:
            for key, value in tags.items():
                span.set_attribute(f"tag.{key}", str(value))

        # Store span for access
        span_id = str(id(span))
        self.active_spans[span_id] = span

        start_time = time.time()

        try:
            self.traces_created.labels(
                service=self.service_name,
                operation=operation_name
            ).inc()

            yield span

            # Mark span as successful
            span.set_attribute("status", "success")

        except Exception as e:
            # Record exception in span
            span.record_exception(e)
            span.set_attribute("status", "error")
            span.set_attribute("error.message", str(e))
            raise
        finally:
            # Complete span
            duration = time.time() - start_time
            span.end()

            # Remove from active spans
            self.active_spans.pop(span_id, None)

            # Record metrics
            status = "error" if span.status.status_code.name == "ERROR" else "success"
            self.span_duration.labels(
                service=self.service_name,
                operation=operation_name,
                status=status
            ).observe(duration)

    def trace_function(self, operation_name: Optional[str] = None, tags: Optional[Dict[str, Any]] = None):
        """Decorator for automatic function tracing."""
        def decorator(func):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    async with self.start_span(op_name, tags=tags) as span:
                        # Add function parameters as tags
                        sig = inspect.signature(func)
                        bound_args = sig.bind(*args, **kwargs)
                        bound_args.apply_defaults()

                        for param_name, param_value in bound_args.arguments.items():
                            if param_name not in ['self', 'cls']:
                                span.set_attribute(f"param.{param_name}", str(param_value)[:100])

                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.tracer.start_as_current_span(op_name) as span:
                        # Add function parameters as tags
                        sig = inspect.signature(func)
                        bound_args = sig.bind(*args, **kwargs)
                        bound_args.apply_defaults()

                        for param_name, param_value in bound_args.arguments.items():
                            if param_name not in ['self', 'cls']:
                                span.set_attribute(f"param.{param_name}", str(param_value)[:100])

                        return func(*args, **kwargs)
                return sync_wrapper
        return decorator

class XORBMetrics:
    """XORB-specific metrics collection and management."""

    def __init__(self, service_name: str, registry: Optional[CollectorRegistry] = None):
        self.service_name = service_name
        self.registry = registry or CollectorRegistry()

        # Core business metrics
        self.campaigns_total = Counter(
            'xorb_campaigns_total',
            'Total campaigns executed',
            ['service', 'status', 'target_type'],
            registry=self.registry
        )

        self.vulnerabilities_discovered = Counter(
            'xorb_vulnerabilities_discovered_total',
            'Total vulnerabilities discovered',
            ['service', 'severity', 'target_type'],
            registry=self.registry
        )

        self.exploits_successful = Counter(
            'xorb_exploits_successful_total',
            'Total successful exploits',
            ['service', 'exploit_type', 'target_type'],
            registry=self.registry
        )

        self.stealth_score = Gauge(
            'xorb_stealth_score',
            'Current stealth effectiveness score',
            ['service', 'campaign_id'],
            registry=self.registry
        )

        # Technical performance metrics
        self.request_duration = Histogram(
            'xorb_request_duration_seconds',
            'Request duration in seconds',
            ['service', 'endpoint', 'method', 'status'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=self.registry
        )

        self.active_connections = Gauge(
            'xorb_active_connections',
            'Current active connections',
            ['service', 'connection_type'],
            registry=self.registry
        )

        self.memory_usage = Gauge(
            'xorb_memory_usage_bytes',
            'Memory usage in bytes',
            ['service', 'type'],
            registry=self.registry
        )

        self.cpu_usage = Gauge(
            'xorb_cpu_usage_percent',
            'CPU usage percentage',
            ['service', 'core'],
            registry=self.registry
        )

        # EPYC-specific metrics
        self.numa_utilization = Gauge(
            'xorb_numa_utilization_percent',
            'NUMA node utilization percentage',
            ['service', 'numa_node'],
            registry=self.registry
        )

        self.ccx_temperature = Gauge(
            'xorb_ccx_temperature_celsius',
            'Core Complex temperature in Celsius',
            ['service', 'ccx'],
            registry=self.registry
        )

        self.cache_hit_rate = Gauge(
            'xorb_cache_hit_rate_percent',
            'Cache hit rate percentage',
            ['service', 'cache_level'],
            registry=self.registry
        )

        # AI/ML metrics
        self.ai_inference_duration = Histogram(
            'xorb_ai_inference_duration_seconds',
            'AI inference duration in seconds',
            ['service', 'model', 'provider'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0],
            registry=self.registry
        )

        self.ai_model_accuracy = Gauge(
            'xorb_ai_model_accuracy_percent',
            'AI model accuracy percentage',
            ['service', 'model', 'task_type'],
            registry=self.registry
        )

        self.ai_cost_total = Counter(
            'xorb_ai_cost_usd_total',
            'Total AI API costs in USD',
            ['service', 'provider', 'model'],
            registry=self.registry
        )

        # Security metrics
        self.security_events = Counter(
            'xorb_security_events_total',
            'Total security events',
            ['service', 'event_type', 'severity'],
            registry=self.registry
        )

        self.compliance_score = Gauge(
            'xorb_compliance_score_percent',
            'Compliance score percentage',
            ['service', 'framework'],
            registry=self.registry
        )

        # Custom metrics storage
        self.custom_metrics: Dict[str, Any] = {}

    def create_custom_metric(self, definition: MetricDefinition) -> Any:
        """Create a custom metric based on definition."""
        if definition.type == MetricType.COUNTER:
            metric = Counter(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        elif definition.type == MetricType.GAUGE:
            metric = Gauge(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        elif definition.type == MetricType.HISTOGRAM:
            metric = Histogram(
                definition.name,
                definition.description,
                definition.labels,
                buckets=definition.buckets,
                registry=self.registry
            )
        elif definition.type == MetricType.SUMMARY:
            metric = Summary(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        else:
            raise ValueError(f"Unsupported metric type: {definition.type}")

        self.custom_metrics[definition.name] = metric
        return metric

    def get_metric(self, name: str) -> Optional[Any]:
        """Get a metric by name."""
        return self.custom_metrics.get(name)

    def export_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest(self.registry)

class EPYCResourceMonitor:
    """EPYC-specific resource monitoring for optimal performance."""

    def __init__(self, metrics: XORBMetrics):
        self.metrics = metrics
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None

        # EPYC topology information
        self.numa_nodes = 2  # Typical EPYC configuration
        self.ccx_per_node = 4  # Core Complexes per NUMA node
        self.cores_per_ccx = 4  # Cores per Core Complex

    async def start_monitoring(self, interval: float = 5.0):
        """Start EPYC resource monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info("EPYC resource monitoring started")

    async def stop_monitoring(self):
        """Stop EPYC resource monitoring."""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("EPYC resource monitoring stopped")

    async def _monitoring_loop(self, interval: float):
        """Main monitoring loop for EPYC resources."""
        while self.monitoring_active:
            try:
                await self._collect_numa_metrics()
                await self._collect_thermal_metrics()
                await self._collect_cache_metrics()
                await self._collect_memory_metrics()

                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in EPYC monitoring loop: {e}")
                await asyncio.sleep(interval)

    async def _collect_numa_metrics(self):
        """Collect NUMA node utilization metrics."""
        try:
            import psutil

            for numa_node in range(self.numa_nodes):
                # Calculate CPU utilization per NUMA node
                # This is simplified - real implementation would use libnuma
                cpu_count = psutil.cpu_count(logical=True) // self.numa_nodes
                start_cpu = numa_node * cpu_count
                end_cpu = start_cpu + cpu_count

                cpu_percentages = psutil.cpu_percent(percpu=True)
                numa_utilization = sum(cpu_percentages[start_cpu:end_cpu]) / cpu_count

                self.metrics.numa_utilization.labels(
                    service=self.metrics.service_name,
                    numa_node=str(numa_node)
                ).set(numa_utilization)

        except ImportError:
            logger.warning("psutil not available for NUMA monitoring")
        except Exception as e:
            logger.error(f"Error collecting NUMA metrics: {e}")

    async def _collect_thermal_metrics(self):
        """Collect thermal metrics for CCX monitoring."""
        try:
            import psutil

            # Get CPU temperatures (if available)
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()

                if 'k10temp' in temps:  # AMD temperature sensor
                    for i, temp in enumerate(temps['k10temp']):
                        if i < self.numa_nodes * self.ccx_per_node:
                            ccx_id = i // self.ccx_per_node
                            self.metrics.ccx_temperature.labels(
                                service=self.metrics.service_name,
                                ccx=str(ccx_id)
                            ).set(temp.current)

        except Exception as e:
            logger.error(f"Error collecting thermal metrics: {e}")

    async def _collect_cache_metrics(self):
        """Collect cache performance metrics."""
        try:
            # This would require specialized tools like perf or Intel PCM
            # For now, we'll simulate cache hit rates
            import random

            for level in ['l1', 'l2', 'l3']:
                # Simulate cache hit rate (would be real data in production)
                hit_rate = 85 + random.uniform(-5, 10)  # 80-95% typical range

                self.metrics.cache_hit_rate.labels(
                    service=self.metrics.service_name,
                    cache_level=level
                ).set(max(0, min(100, hit_rate)))

        except Exception as e:
            logger.error(f"Error collecting cache metrics: {e}")

    async def _collect_memory_metrics(self):
        """Collect memory usage metrics."""
        try:
            import psutil

            memory = psutil.virtual_memory()

            self.metrics.memory_usage.labels(
                service=self.metrics.service_name,
                type="total"
            ).set(memory.total)

            self.metrics.memory_usage.labels(
                service=self.metrics.service_name,
                type="used"
            ).set(memory.used)

            self.metrics.memory_usage.labels(
                service=self.metrics.service_name,
                type="available"
            ).set(memory.available)

        except Exception as e:
            logger.error(f"Error collecting memory metrics: {e}")

class AlertManager:
    """Advanced alerting system with multi-channel notifications."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.notification_channels: Dict[str, Callable] = {}

        # Alert metrics
        self.alerts_fired = Counter(
            'xorb_alerts_fired_total',
            'Total alerts fired',
            ['severity', 'rule_name']
        )

        self.alerts_resolved = Counter(
            'xorb_alerts_resolved_total',
            'Total alerts resolved',
            ['severity', 'rule_name']
        )

    def register_alert_rule(self, rule: AlertRule):
        """Register a new alert rule."""
        self.alert_rules[rule.name] = rule
        logger.info(f"Registered alert rule: {rule.name}")

    def register_notification_channel(self, name: str, handler: Callable):
        """Register a notification channel."""
        self.notification_channels[name] = handler
        logger.info(f"Registered notification channel: {name}")

    async def evaluate_alerts(self, metrics_data: Dict[str, float]):
        """Evaluate all alert rules against current metrics."""
        for rule_name, rule in self.alert_rules.items():
            try:
                await self._evaluate_rule(rule, metrics_data)
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule_name}: {e}")

    async def _evaluate_rule(self, rule: AlertRule, metrics_data: Dict[str, float]):
        """Evaluate a specific alert rule."""
        metric_value = metrics_data.get(rule.metric_query, 0.0)

        # Check if threshold is crossed
        alert_active = metric_value >= rule.threshold
        alert_key = f"{rule.name}_{hash(rule.metric_query)}"

        if alert_active and alert_key not in self.active_alerts:
            # New alert
            alert_data = {
                'rule': rule,
                'value': metric_value,
                'started_at': datetime.utcnow(),
                'notified': False
            }
            self.active_alerts[alert_key] = alert_data

            # Wait for duration before firing
            await asyncio.sleep(rule.duration.total_seconds())

            # Check if still active
            if alert_key in self.active_alerts:
                await self._fire_alert(alert_key, alert_data)

        elif not alert_active and alert_key in self.active_alerts:
            # Alert resolved
            alert_data = self.active_alerts.pop(alert_key)
            await self._resolve_alert(rule, alert_data)

    async def _fire_alert(self, alert_key: str, alert_data: Dict[str, Any]):
        """Fire an alert and send notifications."""
        rule = alert_data['rule']

        self.alerts_fired.labels(
            severity=rule.severity.value,
            rule_name=rule.name
        ).inc()

        # Create alert message
        alert_message = {
            'alert_name': rule.name,
            'severity': rule.severity.value,
            'description': rule.description,
            'metric_query': rule.metric_query,
            'threshold': rule.threshold,
            'current_value': alert_data['value'],
            'started_at': alert_data['started_at'].isoformat(),
            'runbook_url': rule.runbook_url,
            'labels': rule.labels
        }

        # Send notifications
        for channel_name, handler in self.notification_channels.items():
            try:
                await handler(alert_message)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel_name}: {e}")

        alert_data['notified'] = True
        logger.warning(f"Alert fired: {rule.name} (value: {alert_data['value']}, threshold: {rule.threshold})")

    async def _resolve_alert(self, rule: AlertRule, alert_data: Dict[str, Any]):
        """Resolve an alert."""
        self.alerts_resolved.labels(
            severity=rule.severity.value,
            rule_name=rule.name
        ).inc()

        logger.info(f"Alert resolved: {rule.name}")

class XORBObservabilityStack:
    """Comprehensive observability stack for XORB platform."""

    def __init__(self, service_name: str, config: Dict[str, Any]):
        self.service_name = service_name
        self.config = config

        # Initialize components
        self.tracer = XORBTracer(service_name, config.get('jaeger_endpoint', 'http://jaeger:14268/api/traces'))
        self.metrics = XORBMetrics(service_name)
        self.resource_monitor = EPYCResourceMonitor(self.metrics)
        self.alert_manager = AlertManager(config.get('alerting', {}))

        # HTTP server for metrics exposition
        self.metrics_server: Optional[aiohttp.web.Application] = None
        self.metrics_server_runner: Optional[aiohttp.web.AppRunner] = None

        self._setup_default_alerts()

    async def initialize(self):
        """Initialize the observability stack."""
        # Start EPYC resource monitoring
        await self.resource_monitor.start_monitoring()

        # Setup metrics HTTP server
        await self._setup_metrics_server()

        # Register default notification channels
        self._setup_notification_channels()

        logger.info(f"Observability stack initialized for service: {self.service_name}")

    async def close(self):
        """Clean shutdown of observability stack."""
        await self.resource_monitor.stop_monitoring()

        if self.metrics_server_runner:
            await self.metrics_server_runner.cleanup()

        logger.info("Observability stack shutdown complete")

    async def _setup_metrics_server(self):
        """Setup HTTP server for Prometheus metrics."""
        self.metrics_server = aiohttp.web.Application()

        async def metrics_handler(request):
            return aiohttp.web.Response(
                text=self.metrics.export_metrics(),
                content_type='text/plain; version=0.0.4; charset=utf-8'
            )

        async def health_handler(request):
            return aiohttp.web.json_response({'status': 'healthy', 'service': self.service_name})

        self.metrics_server.router.add_get('/metrics', metrics_handler)
        self.metrics_server.router.add_get('/health', health_handler)

        self.metrics_server_runner = aiohttp.web.AppRunner(self.metrics_server)
        await self.metrics_server_runner.setup()

        site = aiohttp.web.TCPSite(
            self.metrics_server_runner,
            '0.0.0.0',
            self.config.get('metrics_port', 9090)
        )
        await site.start()

        logger.info(f"Metrics server started on port {self.config.get('metrics_port', 9090)}")

    def _setup_notification_channels(self):
        """Setup default notification channels."""
        async def log_notification(alert_message: Dict[str, Any]):
            """Log-based notification channel."""
            severity = alert_message['severity']
            if severity == 'critical':
                logger.critical(f"ALERT: {alert_message}")
            elif severity == 'warning':
                logger.warning(f"ALERT: {alert_message}")
            else:
                logger.info(f"ALERT: {alert_message}")

        async def webhook_notification(alert_message: Dict[str, Any]):
            """Webhook-based notification channel."""
            webhook_url = self.config.get('webhook_url')
            if webhook_url:
                async with aiohttp.ClientSession() as session:
                    await session.post(webhook_url, json=alert_message)

        self.alert_manager.register_notification_channel('log', log_notification)
        self.alert_manager.register_notification_channel('webhook', webhook_notification)

    def _setup_default_alerts(self):
        """Setup default alert rules for XORB platform."""
        default_alerts = [
            AlertRule(
                name="high_cpu_usage",
                metric_query="xorb_cpu_usage_percent",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
                duration=timedelta(minutes=5),
                description="CPU usage is above 80%",
                labels={"component": "infrastructure"}
            ),
            AlertRule(
                name="high_memory_usage",
                metric_query="xorb_memory_usage_percent",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                duration=timedelta(minutes=3),
                description="Memory usage is above 85%",
                labels={"component": "infrastructure"}
            ),
            AlertRule(
                name="high_epyc_temperature",
                metric_query="xorb_ccx_temperature_celsius",
                threshold=75.0,
                severity=AlertSeverity.CRITICAL,
                duration=timedelta(minutes=1),
                description="EPYC CCX temperature is above 75°C",
                labels={"component": "hardware"}
            ),
            AlertRule(
                name="high_request_latency",
                metric_query="xorb_request_duration_seconds",
                threshold=5.0,
                severity=AlertSeverity.WARNING,
                duration=timedelta(minutes=2),
                description="Request latency is above 5 seconds",
                labels={"component": "application"}
            ),
            AlertRule(
                name="ai_cost_threshold",
                metric_query="xorb_ai_cost_usd_total",
                threshold=100.0,
                severity=AlertSeverity.WARNING,
                duration=timedelta(hours=1),
                description="AI costs exceeded $100 per hour",
                labels={"component": "ai"}
            )
        ]

        for alert in default_alerts:
            self.alert_manager.register_alert_rule(alert)

# Global observability stack
observability_stack: Optional[XORBObservabilityStack] = None

async def initialize_observability(service_name: str, config: Dict[str, Any]) -> XORBObservabilityStack:
    """Initialize global observability stack."""
    global observability_stack
    observability_stack = XORBObservabilityStack(service_name, config)
    await observability_stack.initialize()
    return observability_stack

async def get_observability() -> Optional[XORBObservabilityStack]:
    """Get global observability stack."""
    return observability_stack

def trace(operation_name: Optional[str] = None, tags: Optional[Dict[str, Any]] = None):
    """Decorator for automatic tracing."""
    def decorator(func):
        if observability_stack:
            return observability_stack.tracer.trace_function(operation_name, tags)(func)
        return func
    return decorator
