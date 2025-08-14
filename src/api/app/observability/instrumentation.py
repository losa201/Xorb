"""
XORB Phase G5 OpenTelemetry Instrumentation
Comprehensive observability for customer-facing reliability signals
"""

import os
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager

from opentelemetry import trace, metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from prometheus_client import start_http_server

# Global tracer and meter instances
_tracer: Optional[trace.Tracer] = None
_meter: Optional[metrics.Meter] = None


def setup_instrumentation(
    app_name: str = "xorb-api",
    version: str = "1.0.0",
    environment: str = "development",
    prometheus_port: int = 8080,
    enable_otlp: bool = True,
    otlp_endpoint: Optional[str] = None
) -> None:
    """
    Setup comprehensive OpenTelemetry instrumentation for XORB platform.

    Args:
        app_name: Application name for telemetry
        version: Application version
        environment: Deployment environment (dev/staging/prod)
        prometheus_port: Port for Prometheus metrics endpoint
        enable_otlp: Enable OTLP export to external systems
        otlp_endpoint: OTLP collector endpoint (optional)
    """
    global _tracer, _meter

    # Create resource with comprehensive metadata
    resource = Resource.create({
        "service.name": app_name,
        "service.version": version,
        "deployment.environment": environment,
        "service.namespace": "xorb",
        "service.instance.id": os.getenv("HOSTNAME", "unknown"),
        "cloud.platform": os.getenv("CLOUD_PLATFORM", "kubernetes"),
        "k8s.cluster.name": os.getenv("K8S_CLUSTER", "xorb-cluster"),
        "k8s.namespace.name": os.getenv("K8S_NAMESPACE", "xorb-system")
    })

    # Setup distributed tracing
    trace_exporters = []

    if enable_otlp:
        otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        trace_exporters.append(OTLPSpanExporter(endpoint=otlp_endpoint))

    tracer_provider = TracerProvider(resource=resource)

    for exporter in trace_exporters:
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(tracer_provider)
    _tracer = trace.get_tracer(app_name, version)

    # Setup metrics collection
    metric_readers = []

    # Prometheus metrics reader (pull-based)
    prometheus_reader = PrometheusMetricReader()
    metric_readers.append(prometheus_reader)

    # OTLP metrics export (push-based)
    if enable_otlp:
        otlp_metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint)
        metric_readers.append(
            PeriodicExportingMetricReader(otlp_metric_exporter, export_interval_millis=30000)
        )

    meter_provider = MeterProvider(resource=resource, metric_readers=metric_readers)
    metrics.set_meter_provider(meter_provider)
    _meter = metrics.get_meter(app_name, version)

    # Start Prometheus HTTP server
    try:
        start_http_server(prometheus_port)
        print(f"✅ Prometheus metrics server started on port {prometheus_port}")
    except OSError as e:
        print(f"⚠️ Prometheus server port {prometheus_port} already in use: {e}")

    # Auto-instrument common libraries
    _setup_auto_instrumentation()

    print(f"✅ OpenTelemetry instrumentation initialized for {app_name} v{version}")


def _setup_auto_instrumentation() -> None:
    """Setup automatic instrumentation for common libraries."""
    try:
        # Auto-instrument FastAPI
        FastAPIInstrumentor.instrument()

        # Auto-instrument database connections
        AsyncPGInstrumentor().instrument()

        # Auto-instrument Redis
        RedisInstrumentor().instrument()

        # Auto-instrument HTTP requests
        RequestsInstrumentor().instrument()

        print("✅ Auto-instrumentation enabled for FastAPI, AsyncPG, Redis, Requests")
    except Exception as e:
        print(f"⚠️ Auto-instrumentation setup failed: {e}")


def get_tracer() -> trace.Tracer:
    """Get the global tracer instance."""
    if _tracer is None:
        raise RuntimeError("OpenTelemetry not initialized. Call setup_instrumentation() first.")
    return _tracer


def get_meter() -> metrics.Meter:
    """Get the global meter instance."""
    if _meter is None:
        raise RuntimeError("OpenTelemetry not initialized. Call setup_instrumentation() first.")
    return _meter


@contextmanager
def trace_operation(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Context manager for tracing operations with automatic timing.

    Args:
        name: Span name
        attributes: Additional span attributes
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        start_time = time.time()

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            span.set_attribute("duration_ms", duration_ms)


def record_custom_metric(
    name: str,
    value: float,
    metric_type: str = "counter",
    attributes: Optional[Dict[str, str]] = None,
    description: str = ""
) -> None:
    """
    Record a custom metric value.

    Args:
        name: Metric name
        value: Metric value
        metric_type: Type of metric (counter, histogram, gauge)
        attributes: Metric labels/attributes
        description: Metric description
    """
    meter = get_meter()
    attributes = attributes or {}

    if metric_type == "counter":
        counter = meter.create_counter(name, description=description)
        counter.add(value, attributes)
    elif metric_type == "histogram":
        histogram = meter.create_histogram(name, description=description)
        histogram.record(value, attributes)
    elif metric_type == "gauge":
        gauge = meter.create_up_down_counter(name, description=description)
        gauge.add(value, attributes)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


class ObservabilityMiddleware:
    """FastAPI middleware for comprehensive request observability."""

    def __init__(self, app):
        self.app = app
        self.meter = get_meter()

        # Create core metrics
        self.request_counter = self.meter.create_counter(
            "http_requests_total",
            description="Total HTTP requests"
        )
        self.request_duration = self.meter.create_histogram(
            "http_request_duration_seconds",
            description="HTTP request duration in seconds"
        )
        self.request_size = self.meter.create_histogram(
            "http_request_size_bytes",
            description="HTTP request size in bytes"
        )
        self.response_size = self.meter.create_histogram(
            "http_response_size_bytes",
            description="HTTP response size in bytes"
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        request = scope

        # Extract request metadata
        method = request["method"]
        path = request["path"]

        # Prepare labels
        labels = {
            "method": method,
            "path": path,
            "handler": "unknown"
        }

        # Measure request size
        content_length = 0
        for header_name, header_value in request.get("headers", []):
            if header_name == b"content-length":
                content_length = int(header_value.decode())
                break

        if content_length > 0:
            self.request_size.record(content_length, labels)

        # Process request
        response_data = []
        status_code = 200

        async def send_wrapper(message):
            nonlocal status_code, response_data
            if message["type"] == "http.response.start":
                status_code = message["status"]
                labels["status_code"] = str(status_code)
                labels["status_class"] = f"{status_code // 100}xx"
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                if body:
                    response_data.append(len(body))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            labels["status_code"] = "500"
            labels["status_class"] = "5xx"
            labels["error"] = type(e).__name__
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time

            self.request_counter.add(1, labels)
            self.request_duration.record(duration, labels)

            if response_data:
                self.response_size.record(sum(response_data), labels)


# Health check for observability system
async def observability_health_check() -> Dict[str, Any]:
    """Health check for observability components."""
    try:
        tracer = get_tracer()
        meter = get_meter()

        # Test trace creation
        with tracer.start_as_current_span("health_check_trace") as span:
            span.set_attribute("component", "observability")

        # Test metric recording
        health_counter = meter.create_counter("observability_health_checks_total")
        health_counter.add(1, {"status": "healthy"})

        return {
            "status": "healthy",
            "tracing": "enabled",
            "metrics": "enabled",
            "prometheus_endpoint": "/metrics",
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }
