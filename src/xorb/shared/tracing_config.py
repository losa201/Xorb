from typing import Dict, Any, Optional
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
import os

class TracingConfig:
    """
    Configuration class for distributed tracing with OpenTelemetry.
    Provides a standardized way to configure tracing across services.
    """

    def __init__(self, service_name: str, config: Optional[Dict[str, Any]] = None):
        self.service_name = service_name
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.exporter_type = self.config.get('exporter', 'console')
        self.endpoint = self.config.get('endpoint', 'http://jaeger:14268/api/traces')
        self.headers = self.config.get('headers', {})
        self.batch_size = self.config.get('batch_size', 512)
        self.max_queue_size = self.config.get('max_queue_size', 2048)
        self.export_timeout = self.config.get('export_timeout', 30)
        self.service_version = self.config.get('version', '1.0.0')
        self.environment = self.config.get('environment', 'development')
        self.log_spans = self.config.get('log_spans', False)

        # Initialize tracing components
        self.tracer_provider = None
        self.tracer = None

    def configure_tracing(self) -> None:
        """
        Configure the tracing system based on the configuration.
        """
        if not self.enabled:
            trace.set_tracer_provider(trace.NoOpTracerProvider())
            return

        # Create resource with service information
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: self.service_name,
            ResourceAttributes.SERVICE_VERSION: self.service_version,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.environment
        })

        # Create tracer provider with resource
        self.tracer_provider = TracerProvider(resource=resource)

        # Configure exporter based on type
        if self.exporter_type == 'otlp':
            exporter = OTLPSpanExporter(
                endpoint=self.endpoint,
                headers=self.headers,
                timeout=self.export_timeout
            )
            span_processor = BatchSpanProcessor(
                exporter,
                max_queue_size=self.max_queue_size,
                scheduled_delay_millis=self.batch_size * 10,  # Rough estimate based on batch size
                max_export_batch_size=self.batch_size,
                export_timeout_millis=self.export_timeout * 1000
            )
            self.tracer_provider.add_span_processor(span_processor)

        elif self.exporter_type == 'console' or self.log_spans:
            # Always add console exporter if log_spans is enabled
            console_exporter = ConsoleSpanExporter()
            console_processor = BatchSpanProcessor(
                console_exporter,
                max_queue_size=self.max_queue_size,
                scheduled_delay_millis=self.batch_size * 10,
                max_export_batch_size=self.batch_size,
                export_timeout_millis=self.export_timeout * 1000
            )
            self.tracer_provider.add_span_processor(console_processor)

        # Set as global tracer provider
        trace.set_tracer_provider(self.tracer_provider)

        # Create tracer instance
        self.tracer = self.tracer_provider.get_tracer(self.service_name, self.service_version)

    def get_tracer(self):
        """
        Get the configured tracer instance.

        Returns:
            Configured tracer instance
        """
        if not self.tracer:
            self.configure_tracing()
        return self.tracer

    def shutdown(self):
        """
        Shutdown the tracing system gracefully.
        """
        if self.tracer_provider:
            self.tracer_provider.shutdown()

    @classmethod
    def from_env(cls, service_name: str) -> 'TracingConfig':
        """
        Create a tracing configuration from environment variables.

        Args:
            service_name: Name of the service

        Returns:
            Configured TracingConfig instance
        """
        config = {
            'enabled': os.getenv('TRACING_ENABLED', 'true').lower() == 'true',
            'exporter': os.getenv('TRACING_EXPORTER', 'console'),
            'endpoint': os.getenv('TRACING_ENDPOINT', 'http://jaeger:14268/api/traces'),
            'headers': cls._parse_headers(os.getenv('TRACING_HEADERS', '')),
            'batch_size': int(os.getenv('TRACING_BATCH_SIZE', '512')),
            'max_queue_size': int(os.getenv('TRACING_MAX_QUEUE_SIZE', '2048')),
            'export_timeout': int(os.getenv('TRACING_EXPORT_TIMEOUT', '30')),
            'version': os.getenv('SERVICE_VERSION', '1.0.0'),
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'log_spans': os.getenv('TRACING_LOG_SPANS', 'false').lower() == 'true'
        }

        return cls(service_name, config)

    @staticmethod
    def _parse_headers(header_str: str) -> Dict[str, str]:
        """
        Parse headers from a string format.

        Args:
            header_str: String containing headers in format "key1=value1,key2=value2"

        Returns:
            Dictionary of headers
        """
        headers = {}
        if header_str:
            pairs = header_str.split(',')
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    headers[key.strip()] = value.strip()
        return headers

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the tracing configuration.

        Args:
            new_config: Dictionary with new configuration values
        """
        self.config.update(new_config)
        self.__init__(self.service_name, self.config)  # Re-initialize with new config

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current tracing configuration.

        Returns:
            Dictionary with current configuration
        """
        return {
            'enabled': self.enabled,
            'exporter': self.exporter_type,
            'endpoint': self.endpoint,
            'headers': self.headers,
            'batch_size': self.batch_size,
            'max_queue_size': self.max_queue_size,
            'export_timeout': self.export_timeout,
            'version': self.service_version,
            'environment': self.environment,
            'log_spans': self.log_spans
        }
