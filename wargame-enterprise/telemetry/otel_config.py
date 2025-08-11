#!/usr/bin/env python3
"""
OpenTelemetry Configuration for Cyber Range Telemetry Spine
Comprehensive observability for episode tracing and analysis
"""

import os
import socket
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# OpenTelemetry imports
from opentelemetry import trace, metrics, baggage
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositeHTTPPropagator
from opentelemetry.propagators.b3 import B3MultiFormat, B3SingleFormat
from opentelemetry.propagators.jaeger import JaegerPropagator
from opentelemetry.propagators.baggage import BaggagePropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Custom cyber range attributes
CYBER_RANGE_ATTRIBUTES = {
    "service.name": "xorb-cyber-range",
    "service.namespace": "xorb",
    "service.version": "1.0.0",
    "deployment.environment": os.getenv("ENVIRONMENT", "development"),
    "cyber_range.version": "2025.1",
    "cyber_range.platform": "xorb-enterprise",
}

@dataclass
class TelemetryConfig:
    """Configuration for telemetry spine"""
    # Service identification
    service_name: str = "xorb-cyber-range"
    service_version: str = "1.0.0"
    environment: str = "development"
    
    # Tracing configuration
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    trace_sample_rate: float = 1.0
    batch_span_timeout: int = 5000
    max_queue_size: int = 2048
    
    # Metrics configuration
    prometheus_port: int = 8889
    metrics_export_interval: int = 10
    
    # Custom attributes
    additional_attributes: Dict[str, str] = None
    
    def __post_init__(self):
        if self.additional_attributes is None:
            self.additional_attributes = {}

class CyberRangeTelemetry:
    """Centralized telemetry management for cyber range"""
    
    def __init__(self, config: TelemetryConfig):
        self.config = config
        self.tracer = None
        self.meter = None
        self._initialized = False
        
        # Custom metrics
        self.episode_counter = None
        self.attack_action_counter = None
        self.defense_action_counter = None
        self.detection_latency_histogram = None
        self.agent_decision_time_histogram = None
        self.resource_utilization_gauge = None
    
    def initialize(self):
        """Initialize OpenTelemetry with cyber range specific configuration"""
        if self._initialized:
            return
        
        # Create resource with cyber range attributes
        resource_attributes = {**CYBER_RANGE_ATTRIBUTES}
        resource_attributes.update(self.config.additional_attributes)
        resource_attributes.update({
            "service.name": self.config.service_name,
            "service.version": self.config.service_version,
            "deployment.environment": self.config.environment,
            "host.name": socket.gethostname(),
            "process.pid": os.getpid(),
        })
        
        resource = Resource.create(resource_attributes)
        
        # Initialize tracing
        self._setup_tracing(resource)
        
        # Initialize metrics
        self._setup_metrics(resource)
        
        # Setup propagators for distributed tracing
        self._setup_propagators()
        
        # Instrument libraries
        self._instrument_libraries()
        
        # Initialize custom metrics
        self._initialize_custom_metrics()
        
        self._initialized = True
    
    def _setup_tracing(self, resource: Resource):
        """Setup distributed tracing with Jaeger"""
        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        
        # Add Jaeger exporter
        jaeger_exporter = JaegerExporter(
            endpoint=self.config.jaeger_endpoint,
        )
        
        span_processor = BatchSpanProcessor(
            jaeger_exporter,
            max_queue_size=self.config.max_queue_size,
            schedule_delay_millis=self.config.batch_span_timeout,
        )
        tracer_provider.add_span_processor(span_processor)
        
        # Add console exporter for development
        if self.config.environment == "development":
            console_processor = BatchSpanProcessor(ConsoleSpanExporter())
            tracer_provider.add_span_processor(console_processor)
        
        # Set global tracer provider
        trace.set_tracer_provider(tracer_provider)
        self.tracer = trace.get_tracer(__name__)
    
    def _setup_metrics(self, resource: Resource):
        """Setup metrics collection with Prometheus"""
        # Prometheus metric reader
        prometheus_reader = PrometheusMetricReader(port=self.config.prometheus_port)
        
        # Console metric reader for development
        readers = [prometheus_reader]
        if self.config.environment == "development":
            console_reader = PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=self.config.metrics_export_interval * 1000
            )
            readers.append(console_reader)
        
        # Create meter provider
        meter_provider = MeterProvider(resource=resource, metric_readers=readers)
        metrics.set_meter_provider(meter_provider)
        self.meter = metrics.get_meter(__name__)
    
    def _setup_propagators(self):
        """Setup trace context propagation for distributed tracing"""
        propagators = [
            TraceContextTextMapPropagator(),
            BaggagePropagator(),
            B3MultiFormat(),
            B3SingleFormat(),
            JaegerPropagator(),
        ]
        
        set_global_textmap(CompositeHTTPPropagator(propagators))
    
    def _instrument_libraries(self):
        """Instrument common libraries for automatic telemetry"""
        try:
            RequestsInstrumentor().instrument()
            URLLib3Instrumentor().instrument()
            RedisInstrumentor().instrument()
            Psycopg2Instrumentor().instrument()
            
            # FastAPI instrumentation (if FastAPI is used)
            try:
                FastAPIInstrumentor().instrument()
            except Exception:
                pass  # FastAPI might not be available in all components
                
        except Exception as e:
            print(f"Warning: Could not instrument some libraries: {e}")
    
    def _initialize_custom_metrics(self):
        """Initialize cyber range specific metrics"""
        # Episode metrics
        self.episode_counter = self.meter.create_counter(
            name="cyber_range_episodes_total",
            description="Total number of cyber range episodes",
            unit="1"
        )
        
        self.episode_duration_histogram = self.meter.create_histogram(
            name="cyber_range_episode_duration_seconds",
            description="Duration of cyber range episodes",
            unit="s"
        )
        
        # Attack metrics
        self.attack_action_counter = self.meter.create_counter(
            name="cyber_range_attack_actions_total",
            description="Total number of attack actions",
            unit="1"
        )
        
        self.attack_success_rate = self.meter.create_histogram(
            name="cyber_range_attack_success_rate",
            description="Attack success rate per episode",
            unit="1"
        )
        
        # Defense metrics
        self.defense_action_counter = self.meter.create_counter(
            name="cyber_range_defense_actions_total",
            description="Total number of defense actions",
            unit="1"
        )
        
        self.detection_latency_histogram = self.meter.create_histogram(
            name="cyber_range_detection_latency_seconds",
            description="Time from attack to detection",
            unit="s"
        )
        
        # AI/ML metrics
        self.agent_decision_time_histogram = self.meter.create_histogram(
            name="cyber_range_agent_decision_time_seconds",
            description="Time for AI agent to make decisions",
            unit="s"
        )
        
        self.model_inference_counter = self.meter.create_counter(
            name="cyber_range_model_inferences_total",
            description="Total number of model inferences",
            unit="1"
        )
        
        # Resource metrics
        self.resource_utilization_gauge = self.meter.create_up_down_counter(
            name="cyber_range_resource_utilization",
            description="Resource utilization during episodes",
            unit="1"
        )
        
        # Compliance metrics
        self.compliance_violations_counter = self.meter.create_counter(
            name="cyber_range_compliance_violations_total",
            description="Total compliance violations detected",
            unit="1"
        )
        
        # Safety metrics
        self.safety_violations_counter = self.meter.create_counter(
            name="cyber_range_safety_violations_total",
            description="Total safety violations detected",
            unit="1"
        )
    
    def start_episode_span(self, episode_id: str, episode_type: str, 
                          agents: List[str], environment_id: str) -> trace.Span:
        """Start a new episode span with comprehensive attributes"""
        if not self.tracer:
            raise RuntimeError("Telemetry not initialized")
        
        span = self.tracer.start_span(
            name=f"cyber_range.episode.{episode_type}",
            attributes={
                "cyber_range.episode.id": episode_id,
                "cyber_range.episode.type": episode_type,
                "cyber_range.episode.start_time": datetime.utcnow().isoformat(),
                "cyber_range.agents.count": len(agents),
                "cyber_range.agents.types": ",".join(agents),
                "cyber_range.environment.id": environment_id,
            }
        )
        
        # Add episode context to baggage
        baggage.set_baggage("episode_id", episode_id)
        baggage.set_baggage("episode_type", episode_type)
        
        # Record episode start metric
        self.episode_counter.add(1, {
            "episode_type": episode_type,
            "environment_id": environment_id
        })
        
        return span
    
    def start_action_span(self, action_type: str, agent_id: str, 
                         technique: str, target: str, parent_span: Optional[trace.Span] = None) -> trace.Span:
        """Start a span for an individual action (attack/defense)"""
        if not self.tracer:
            raise RuntimeError("Telemetry not initialized")
        
        span_name = f"cyber_range.action.{action_type}"
        
        # Use parent span context if provided
        context = trace.set_span_in_context(parent_span) if parent_span else None
        
        span = self.tracer.start_span(
            name=span_name,
            context=context,
            attributes={
                "cyber_range.action.type": action_type,
                "cyber_range.action.agent_id": agent_id,
                "cyber_range.action.technique": technique,
                "cyber_range.action.target": target,
                "cyber_range.action.timestamp": datetime.utcnow().isoformat(),
            }
        )
        
        return span
    
    def record_attack_action(self, agent_id: str, technique: str, success: bool, 
                           detected: bool, duration_ms: float):
        """Record attack action metrics"""
        attributes = {
            "agent_id": agent_id,
            "technique": technique,
            "success": str(success),
            "detected": str(detected)
        }
        
        self.attack_action_counter.add(1, attributes)
    
    def record_defense_action(self, agent_id: str, category: str, effectiveness: str, 
                            cost: int, duration_ms: float):
        """Record defense action metrics"""
        attributes = {
            "agent_id": agent_id,
            "category": category,
            "effectiveness": effectiveness,
            "cost_tier": str(cost)
        }
        
        self.defense_action_counter.add(1, attributes)
    
    def record_detection_latency(self, latency_seconds: float, detection_source: str, 
                               confidence: float):
        """Record detection latency metrics"""
        attributes = {
            "detection_source": detection_source,
            "confidence_tier": "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"
        }
        
        self.detection_latency_histogram.record(latency_seconds, attributes)
    
    def record_agent_decision_time(self, agent_id: str, decision_type: str, 
                                 duration_seconds: float, model_name: str):
        """Record AI agent decision time"""
        attributes = {
            "agent_id": agent_id,
            "decision_type": decision_type,
            "model_name": model_name
        }
        
        self.agent_decision_time_histogram.record(duration_seconds, attributes)
        self.model_inference_counter.add(1, attributes)
    
    def record_resource_utilization(self, cpu_percent: float, memory_percent: float, 
                                  disk_percent: float):
        """Record resource utilization"""
        self.resource_utilization_gauge.add(cpu_percent, {"resource_type": "cpu"})
        self.resource_utilization_gauge.add(memory_percent, {"resource_type": "memory"})
        self.resource_utilization_gauge.add(disk_percent, {"resource_type": "disk"})
    
    def record_compliance_violation(self, framework: str, violation_type: str, 
                                  severity: str):
        """Record compliance violations"""
        attributes = {
            "framework": framework,
            "violation_type": violation_type,
            "severity": severity
        }
        
        self.compliance_violations_counter.add(1, attributes)
    
    def record_safety_violation(self, violation_type: str, severity: str, 
                              auto_mitigated: bool):
        """Record safety violations"""
        attributes = {
            "violation_type": violation_type,
            "severity": severity,
            "auto_mitigated": str(auto_mitigated)
        }
        
        self.safety_violations_counter.add(1, attributes)
    
    def create_custom_span(self, name: str, attributes: Dict[str, Any] = None) -> trace.Span:
        """Create a custom span with cyber range context"""
        if not self.tracer:
            raise RuntimeError("Telemetry not initialized")
        
        span_attributes = {"cyber_range.custom.span": "true"}
        if attributes:
            span_attributes.update(attributes)
        
        return self.tracer.start_span(name, attributes=span_attributes)
    
    def add_span_event(self, span: trace.Span, event_name: str, 
                      attributes: Dict[str, Any] = None):
        """Add an event to a span"""
        span.add_event(event_name, attributes or {})
    
    def set_span_error(self, span: trace.Span, error: Exception):
        """Mark span as error and record exception details"""
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(error)))
        span.record_exception(error)
    
    def shutdown(self):
        """Shutdown telemetry and flush pending data"""
        if self.tracer and hasattr(self.tracer, 'provider'):
            self.tracer.provider.shutdown()

# Global telemetry instance
_telemetry_instance: Optional[CyberRangeTelemetry] = None

def get_telemetry() -> CyberRangeTelemetry:
    """Get the global telemetry instance"""
    global _telemetry_instance
    if _telemetry_instance is None:
        config = TelemetryConfig(
            environment=os.getenv("ENVIRONMENT", "development"),
            jaeger_endpoint=os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces"),
            prometheus_port=int(os.getenv("PROMETHEUS_PORT", "8889"))
        )
        _telemetry_instance = CyberRangeTelemetry(config)
        _telemetry_instance.initialize()
    return _telemetry_instance

def initialize_telemetry(config: Optional[TelemetryConfig] = None):
    """Initialize global telemetry with custom config"""
    global _telemetry_instance
    if config is None:
        config = TelemetryConfig()
    _telemetry_instance = CyberRangeTelemetry(config)
    _telemetry_instance.initialize()

# Decorator for automatic span creation
def traced_operation(operation_name: str, attributes: Dict[str, Any] = None):
    """Decorator to automatically trace operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            telemetry = get_telemetry()
            with telemetry.create_custom_span(operation_name, attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("operation.success", True)
                    return result
                except Exception as e:
                    telemetry.set_span_error(span, e)
                    raise
        return wrapper
    return decorator

if __name__ == "__main__":
    # Example usage and testing
    print("Testing OpenTelemetry configuration...")
    
    # Initialize telemetry
    config = TelemetryConfig(
        service_name="test-cyber-range",
        environment="test"
    )
    telemetry = CyberRangeTelemetry(config)
    telemetry.initialize()
    
    # Test episode span
    episode_span = telemetry.start_episode_span(
        episode_id="test_ep_001",
        episode_type="training",
        agents=["red_agent", "blue_agent"],
        environment_id="test_env"
    )
    
    # Test action spans
    with telemetry.start_action_span(
        action_type="attack",
        agent_id="red_agent_001",
        technique="T1190",
        target="web_server",
        parent_span=episode_span
    ) as attack_span:
        telemetry.add_span_event(attack_span, "payload_executed", {"payload_type": "exploit"})
        telemetry.record_attack_action("red_agent_001", "T1190", True, False, 250.5)
    
    # Test custom metrics
    telemetry.record_detection_latency(15.5, "IDS", 0.85)
    telemetry.record_agent_decision_time("blue_agent_001", "response", 2.3, "claude-3-sonnet")
    telemetry.record_resource_utilization(75.2, 68.1, 45.3)
    
    # End episode span
    episode_span.end()
    
    print("OpenTelemetry test completed successfully!")
    print(f"Telemetry initialized: {telemetry._initialized}")
    print(f"Service name: {telemetry.config.service_name}")
    
    # Shutdown
    telemetry.shutdown()