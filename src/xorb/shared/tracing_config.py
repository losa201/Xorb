from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
import os

def configure_tracing():
    """Configure distributed tracing for the Xorb platform"""
    
    # Create tracer provider with service name
    resource = Resource.create({
        "service.name": os.getenv("SERVICE_NAME", "xorb-service"),
        "environment": os.getenv("ENVIRONMENT", "development")
    })
    
    provider = TracerProvider(resource=resource)
    
    # Configure OTLP exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTLP_ENDPOINT", "http://localhost:4317"),
        insecure=True if os.getenv("OTLP_INSECURE", "true").lower() == "true" else False
    )
    
    # Add span processor
    provider.add_span_processor(
        BatchSpanProcessor(otlp_exporter)
    )
    
    # Set as global tracer provider
    trace.set_tracer_provider(provider)
    
    # Instrument common libraries
    RequestsInstrumentor().instrument()
    LoggingInstrumentor().instrument()

    return provider

# Initialize tracing when module is loaded
configure_tracing()