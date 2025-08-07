from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Initialize tracer
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)

def create_span(name: str):
    """Create a new span with the given name"""
    return trace.get_tracer(__name__).start_as_current_span(name)

def get_tracer():
    """Get the configured tracer"""
    return trace.get_tracer_provider().get_tracer(__name__)

def start_trace(context: dict = None):
    """Start a new trace with optional context"""
    tracer = get_tracer()
    return tracer.start_span("request", attributes=context or {})

def end_trace(span):
    """End the given trace span"""
    span.end()

def add_span_event(span, name: str, attributes: dict = None):
    """Add an event to the given span"""
    span.add_event(name, attributes or {})

def set_span_attribute(span, key: str, value: str):
    """Set an attribute on the given span"""
    span.set_attribute(key, value)

def record_exception(span, exception: Exception):
    """Record an exception on the given span"""
    span.record_exception(exception)

def set_span_status(span, status_code: int, description: str = None):
    """Set the status of the given span"""
    from opentelemetry.trace import Status, StatusCode
    
    if status_code >= 500:
        span.set_status(Status(StatusCode.ERROR, description or "Internal Server Error"))
    elif status_code >= 400:
        span.set_status(Status(StatusCode.ERROR, description or "Client Error"))
    else:
        span.set_status(Status(StatusCode.OK))

def get_trace_id(span):
    """Get the trace ID from the span"""
    return span.get_span_context().trace_id

def get_span_id(span):
    """Get the span ID"""
    return span.get_span_context().span_id

def get_traceparent(span):
    """Get the traceparent header value"""
    span_context = span.get_span_context()
    return f"00-{span_context.trace_id:032x}-{span_context.span_id:016x}-01"

# Initialize the tracer provider
tracer_provider = trace.get_tracer_provider()