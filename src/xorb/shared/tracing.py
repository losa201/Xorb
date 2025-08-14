from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
import os

def init_tracing():
    """Initialize distributed tracing with OpenTelemetry"""

    # Create a resource with service name
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: os.getenv("SERVICE_NAME", "xorb-service"),
        ResourceAttributes.SERVICE_VERSION: os.getenv("SERVICE_VERSION", "1.0.0")
    })

    # Create tracer provider with resource
    provider = TracerProvider(resource=resource)

    # Create OTLP exporter (for sending traces to collector)
    exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTLP_ENDPOINT", "http://localhost:4317"),
        insecure=True if os.getenv("OTLP_INSECURE", "true").lower() == "true" else False
    )

    # Create batch span processor
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    # Set the global tracer provider
    trace.set_tracer_provider(provider)

    return provider.get_tracer(__name__)

# Initialize the tracer
tracer = init_tracing()

# Context manager for creating spans
def start_span(name, context=None):
    """Create a new span with the given name"""
    return tracer.start_as_current_span(name, context=context)
