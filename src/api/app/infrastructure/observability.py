"""Observability infrastructure with OpenTelemetry and structured logging."""
import logging
import os
import sys
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

import structlog

# Optional OpenTelemetry imports
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    TELEMETRY_AVAILABLE = True
    
    # Optional instrumentation imports (may fail if packages not installed)
    try:
        from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
        ASYNCPG_INSTRUMENTATION = True
    except ImportError:
        ASYNCPG_INSTRUMENTATION = False
        
    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor
        REDIS_INSTRUMENTATION = True
    except ImportError:
        REDIS_INSTRUMENTATION = False
        
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPX_INSTRUMENTATION = True
    except ImportError:
        HTTPX_INSTRUMENTATION = False
        
except ImportError:
    TELEMETRY_AVAILABLE = False
    ASYNCPG_INSTRUMENTATION = False
    REDIS_INSTRUMENTATION = False
    HTTPX_INSTRUMENTATION = False
    
    # Provide fallback classes
    class MockTracer:
        def start_span(self, *args, **kwargs):
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    
    trace = type('trace', (), {'get_tracer': lambda *args: MockTracer()})()
    SERVICE_NAME = "service.name"
    SERVICE_VERSION = "service.version"


class ObservabilityConfig:
    """Observability configuration."""
    
    def __init__(self):
        self.service_name = os.getenv("SERVICE_NAME", "xorb-api")
        self.service_version = os.getenv("SERVICE_VERSION", "1.0.0")
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # OpenTelemetry configuration
        self.otlp_endpoint = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")
        self.enable_tracing = os.getenv("ENABLE_TRACING", "true").lower() == "true"
        self.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        
        # Logging configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.log_format = os.getenv("LOG_FORMAT", "json")  # json or text
        self.enable_request_logging = os.getenv("ENABLE_REQUEST_LOGGING", "true").lower() == "true"
        
        # Sampling configuration
        self.trace_sample_rate = float(os.getenv("TRACE_SAMPLE_RATE", "1.0"))
        self.metrics_export_interval = int(os.getenv("METRICS_EXPORT_INTERVAL", "30"))


def setup_tracing(config: ObservabilityConfig) -> None:
    """Setup OpenTelemetry tracing."""
    if not config.enable_tracing or not TELEMETRY_AVAILABLE:
        logging.info("Tracing disabled or OpenTelemetry not available")
        return
    
    # Create resource
    resource = Resource.create({
        SERVICE_NAME: config.service_name,
        SERVICE_VERSION: config.service_version,
        "environment": config.environment,
        "deployment.environment": config.environment
    })
    
    # Setup tracer provider
    tracer_provider = TracerProvider(resource=resource)
    
    # Add OTLP exporter
    if config.otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint)
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)
    
    # Set global tracer provider
    trace.set_tracer_provider(tracer_provider)
    
    logging.info(f"OpenTelemetry tracing enabled for {config.service_name}")


def setup_metrics(config: ObservabilityConfig) -> None:
    """Setup OpenTelemetry metrics."""
    if not config.enable_metrics or not TELEMETRY_AVAILABLE:
        logging.info("Metrics disabled or OpenTelemetry not available")
        return
    
    # Create resource
    resource = Resource.create({
        SERVICE_NAME: config.service_name,
        SERVICE_VERSION: config.service_version,
        "environment": config.environment
    })
    
    # Setup metric exporter
    if config.otlp_endpoint:
        metric_exporter = OTLPMetricExporter(endpoint=config.otlp_endpoint)
        metric_reader = PeriodicExportingMetricReader(
            exporter=metric_exporter,
            export_interval_millis=config.metrics_export_interval * 1000
        )
    else:
        # Console exporter for development
        from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
        metric_exporter = ConsoleMetricExporter()
        metric_reader = PeriodicExportingMetricReader(
            exporter=metric_exporter,
            export_interval_millis=config.metrics_export_interval * 1000
        )
    
    # Setup meter provider
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)
    
    logging.info(f"OpenTelemetry metrics enabled for {config.service_name}")


def setup_instrumentation(app) -> None:
    """Setup automatic instrumentation."""
    if not TELEMETRY_AVAILABLE:
        logging.info("Instrumentation disabled - OpenTelemetry not available")
        return
    
    # Instrument FastAPI
    try:
        FastAPIInstrumentor.instrument_app(app)
        logging.info("FastAPI instrumentation enabled")
    except Exception as e:
        logging.warning(f"FastAPI instrumentation failed: {e}")
    
    # Instrument database
    if ASYNCPG_INSTRUMENTATION:
        try:
            AsyncPGInstrumentor().instrument()
            logging.info("AsyncPG instrumentation enabled")
        except Exception as e:
            logging.warning(f"AsyncPG instrumentation failed: {e}")
    
    # Instrument Redis
    if REDIS_INSTRUMENTATION:
        try:
            RedisInstrumentor().instrument()
            logging.info("Redis instrumentation enabled")
        except Exception as e:
            logging.warning(f"Redis instrumentation failed: {e}")
    
    # Instrument HTTP client
    if HTTPX_INSTRUMENTATION:
        try:
            HTTPXClientInstrumentor().instrument()
            logging.info("HTTPX instrumentation enabled")
        except Exception as e:
            logging.warning(f"HTTPX instrumentation failed: {e}")
    
    logging.info("Automatic instrumentation setup completed")


def setup_structured_logging(config: ObservabilityConfig) -> None:
    """Setup structured logging with structlog."""
    
    # Configure structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
    ]
    
    if config.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True),
        ])
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.log_level)
    )
    
    # Silence noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    logging.info(f"Structured logging configured: {config.log_format} format, {config.log_level} level")


class RequestLoggingMiddleware:
    """Middleware for structured request logging."""
    
    def __init__(self, app, config: ObservabilityConfig):
        self.app = app
        self.config = config
        self.logger = structlog.get_logger("http_requests")
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.config.enable_request_logging:
            await self.app(scope, receive, send)
            return
        
        # Extract request info
        request_info = {
            "method": scope["method"],
            "path": scope["path"],
            "query_string": scope.get("query_string", b"").decode(),
            "client": scope.get("client"),
            "user_agent": dict(scope.get("headers", [])).get(b"user-agent", b"").decode()
        }
        
        # Add trace context if available
        span = trace.get_current_span()
        if span.is_recording():
            span_context = span.get_span_context()
            request_info.update({
                "trace_id": format(span_context.trace_id, "032x"),
                "span_id": format(span_context.span_id, "016x")
            })
        
        import time
        start_time = time.time()
        
        # Capture response
        response_info = {}
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                response_info["status_code"] = message["status"]
                response_info["headers"] = dict(message.get("headers", []))
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
            
            # Log successful request
            duration = time.time() - start_time
            self.logger.info(
                "HTTP request completed",
                **request_info,
                **response_info,
                duration_ms=round(duration * 1000, 2)
            )
        
        except Exception as e:
            # Log failed request
            duration = time.time() - start_time
            self.logger.error(
                "HTTP request failed",
                **request_info,
                error=str(e),
                duration_ms=round(duration * 1000, 2)
            )
            raise


class MetricsCollector:
    """Custom metrics collector for business logic."""
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        if config.enable_metrics and TELEMETRY_AVAILABLE:
            self.meter = metrics.get_meter("xorb.custom")
            self._setup_metrics()
        else:
            self.meter = None
    
    def _setup_metrics(self):
        """Setup custom metrics."""
        # Counters
        self.evidence_uploads = self.meter.create_counter(
            "evidence_uploads_total",
            description="Total number of evidence uploads"
        )
        
        self.auth_attempts = self.meter.create_counter(
            "auth_attempts_total", 
            description="Total authentication attempts"
        )
        
        self.job_executions = self.meter.create_counter(
            "job_executions_total",
            description="Total job executions"
        )
        
        # Histograms  
        self.vector_search_duration = self.meter.create_histogram(
            "vector_search_duration_seconds",
            description="Vector search execution time"
        )
        
        self.file_upload_size = self.meter.create_histogram(
            "file_upload_size_bytes",
            description="Uploaded file sizes"
        )
        
        # Gauges
        self.active_tenants = self.meter.create_up_down_counter(
            "active_tenants",
            description="Number of active tenants"
        )
    
    def record_evidence_upload(self, tenant_id: str, file_size: int, success: bool):
        """Record evidence upload metrics."""
        if not self.config.enable_metrics or not self.meter:
            return
        
        self.evidence_uploads.add(1, {
            "tenant_id": tenant_id,
            "status": "success" if success else "failure"
        })
        
        if success:
            self.file_upload_size.record(file_size, {"tenant_id": tenant_id})
    
    def record_auth_attempt(self, method: str, success: bool, tenant_id: Optional[str] = None):
        """Record authentication attempt."""
        if not self.config.enable_metrics or not self.meter:
            return
        
        attributes = {
            "method": method,
            "status": "success" if success else "failure"
        }
        if tenant_id:
            attributes["tenant_id"] = tenant_id
        
        self.auth_attempts.add(1, attributes)
    
    def record_job_execution(self, job_type: str, duration: float, success: bool):
        """Record job execution metrics."""
        if not self.config.enable_metrics or not self.meter:
            return
        
        self.job_executions.add(1, {
            "job_type": job_type,
            "status": "success" if success else "failure"
        })
    
    def record_vector_search(self, duration: float, result_count: int):
        """Record vector search metrics."""
        if not self.config.enable_metrics:
            return
        
        self.vector_search_duration.record(duration, {
            "result_count_bucket": self._get_count_bucket(result_count)
        })
    
    def _get_count_bucket(self, count: int) -> str:
        """Get bucket label for result count."""
        if count == 0:
            return "zero"
        elif count <= 10:
            return "1-10"
        elif count <= 50:
            return "11-50"
        elif count <= 100:
            return "51-100"
        else:
            return "100+"


# Global instances
_config: Optional[ObservabilityConfig] = None
_metrics_collector: Optional[MetricsCollector] = None


def get_config() -> ObservabilityConfig:
    """Get observability configuration."""
    global _config
    if _config is None:
        _config = ObservabilityConfig()
    return _config


def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(get_config())
    return _metrics_collector


def get_logger(name: str) -> structlog.BoundLogger:
    """Get structured logger."""
    return structlog.get_logger(name)


@asynccontextmanager
async def observability_lifespan(app):
    """Observability lifespan manager for FastAPI."""
    config = get_config()
    
    # Setup observability
    setup_structured_logging(config)
    setup_tracing(config)
    setup_metrics(config)
    setup_instrumentation(app)
    
    # Initialize metrics collector
    get_metrics_collector()
    
    logger = get_logger("observability")
    logger.info("Observability stack initialized", 
                service_name=config.service_name,
                tracing_enabled=config.enable_tracing,
                metrics_enabled=config.enable_metrics)
    
    try:
        yield
    finally:
        logger.info("Observability stack shutting down")


def add_trace_context(**kwargs) -> None:
    """Add context to current trace span."""
    span = trace.get_current_span()
    if span.is_recording():
        for key, value in kwargs.items():
            span.set_attribute(key, str(value))


def trace_function(operation_name: str):
    """Decorator to trace function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(operation_name) as span:
                # Add function info to span
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("function.success", True)
                    return result
                except Exception as e:
                    span.set_attribute("function.success", False)
                    span.set_attribute("function.error", str(e))
                    raise
        
        return wrapper
    return decorator


async def trace_async_function(operation_name: str):
    """Decorator to trace async function execution."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(operation_name) as span:
                # Add function info to span
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("function.success", True)
                    return result
                except Exception as e:
                    span.set_attribute("function.success", False)
                    span.set_attribute("function.error", str(e))
                    raise
        
        return wrapper
    return decorator