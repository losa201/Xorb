//! Distributed tracing implementation for the scanner service
//!
//! Provides OpenTelemetry integration with Jaeger export,
//! span utilities, and context propagation.

use anyhow::Result;
use opentelemetry::global;
use opentelemetry::trace::{TraceError, Tracer};
use opentelemetry_jaeger::new_agent_pipeline;
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Initialize distributed tracing with OpenTelemetry and Jaeger
pub async fn init_tracing(service_name: &str) -> Result<()> {
    // Initialize Jaeger tracer
    let tracer = new_agent_pipeline()
        .with_service_name(service_name)
        .with_auto_split_batch(true)
        .install_batch(opentelemetry::runtime::Tokio)?;

    // Create tracing subscriber with OpenTelemetry layer
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_opentelemetry::layer().with_tracer(tracer))
        .init();

    tracing::info!(service = %service_name, "Distributed tracing initialized");
    Ok(())
}

/// Create a new span for tool execution
pub fn create_tool_span(tool_name: &str, target: &str) -> Span {
    tracing::info_span!(
        "tool_execution",
        tool.name = %tool_name,
        tool.target = %target,
        tool.status = tracing::field::Empty,
        tool.duration_ms = tracing::field::Empty,
        tool.findings_count = tracing::field::Empty,
    )
}

/// Create a new span for fingerprinting operations
pub fn create_fingerprint_span(operation: &str, asset_id: &str) -> Span {
    tracing::info_span!(
        "fingerprint_operation",
        fp.operation = %operation,
        fp.asset_id = %asset_id,
        fp.confidence = tracing::field::Empty,
        fp.status = tracing::field::Empty,
        fp.duration_ms = tracing::field::Empty,
    )
}

/// Create a new span for risk assessment
pub fn create_risk_assessment_span(asset_id: &str) -> Span {
    tracing::info_span!(
        "risk_assessment",
        risk.asset_id = %asset_id,
        risk.score = tracing::field::Empty,
        risk.level = tracing::field::Empty,
        risk.tags_count = tracing::field::Empty,
        risk.duration_ms = tracing::field::Empty,
    )
}

/// Create a new span for bus message processing
pub fn create_bus_message_span(message_type: &str, direction: &str) -> Span {
    tracing::info_span!(
        "bus_message",
        bus.message_type = %message_type,
        bus.direction = %direction,
        bus.status = tracing::field::Empty,
        bus.processing_duration_ms = tracing::field::Empty,
    )
}

/// Record an error in the current span
pub fn record_error(error: &dyn std::error::Error) {
    let current_span = Span::current();
    current_span.record_error(error);
    tracing::error!(error = %error, "Operation failed");
}

/// Record tool execution completion
pub fn record_tool_completion(
    span: &Span,
    status: &str,
    duration_ms: u64,
    findings_count: usize,
) {
    span.record("tool.status", status);
    span.record("tool.duration_ms", duration_ms);
    span.record("tool.findings_count", findings_count);
    
    if status == "success" {
        tracing::info!(
            status = %status,
            duration_ms = %duration_ms,
            findings_count = %findings_count,
            "Tool execution completed successfully"
        );
    } else {
        tracing::warn!(
            status = %status,
            duration_ms = %duration_ms,
            "Tool execution failed"
        );
    }
}

/// Record fingerprint operation completion
pub fn record_fingerprint_completion(
    span: &Span,
    status: &str,
    confidence: f64,
    duration_ms: u64,
) {
    span.record("fp.status", status);
    span.record("fp.confidence", confidence);
    span.record("fp.duration_ms", duration_ms);
    
    tracing::info!(
        status = %status,
        confidence = %confidence,
        duration_ms = %duration_ms,
        "Fingerprint operation completed"
    );
}

/// Record risk assessment completion
pub fn record_risk_assessment_completion(
    span: &Span,
    risk_score: f64,
    risk_level: &str,
    tags_count: usize,
    duration_ms: u64,
) {
    span.record("risk.score", risk_score);
    span.record("risk.level", risk_level);
    span.record("risk.tags_count", tags_count);
    span.record("risk.duration_ms", duration_ms);
    
    tracing::info!(
        score = %risk_score,
        level = %risk_level,
        tags_count = %tags_count,
        duration_ms = %duration_ms,
        "Risk assessment completed"
    );
}

/// Record bus message processing completion
pub fn record_bus_message_completion(
    span: &Span,
    status: &str,
    processing_duration_ms: u64,
) {
    span.record("bus.status", status);
    span.record("bus.processing_duration_ms", processing_duration_ms);
    
    tracing::info!(
        status = %status,
        processing_duration_ms = %processing_duration_ms,
        "Bus message processed"
    );
}

/// Shutdown tracing and flush any pending spans
pub async fn shutdown_tracing() {
    global::shutdown_tracer_provider();
    tracing::info!("Distributed tracing shutdown complete");
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing::instrument;
    
    #[tokio::test]
    async fn test_tracing_initialization() {
        // Test should not panic
        let result = init_tracing("test-scanner").await;
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_span_creation() {
        let span = create_tool_span("nmap", "192.168.1.1");
        assert_eq!(span.metadata().unwrap().name(), "tool_execution");
        
        let fp_span = create_fingerprint_span("create", "asset-001");
        assert_eq!(fp_span.metadata().unwrap().name(), "fingerprint_operation");
        
        let risk_span = create_risk_assessment_span("asset-001");
        assert_eq!(risk_span.metadata().unwrap().name(), "risk_assessment");
        
        let bus_span = create_bus_message_span("discovery_job", "inbound");
        assert_eq!(bus_span.metadata().unwrap().name(), "bus_message");
    }
    
    #[test]
    fn test_span_recording() {
        let span = create_tool_span("nmap", "192.168.1.1");
        
        // Test recording without panicking
        record_tool_completion(&span, "success", 1500, 5);
        
        let fp_span = create_fingerprint_span("create", "asset-001");
        record_fingerprint_completion(&fp_span, "success", 0.95, 200);
        
        let risk_span = create_risk_assessment_span("asset-001");
        record_risk_assessment_completion(&risk_span, 7.5, "HIGH", 3, 100);
        
        let bus_span = create_bus_message_span("discovery_job", "inbound");
        record_bus_message_completion(&bus_span, "success", 50);
    }
    
    #[instrument(skip(test_value))]
    fn instrumented_function(test_value: i32) -> i32 {
        tracing::info!(value = %test_value, "Processing test value");
        test_value * 2
    }
    
    #[test]
    fn test_instrumented_function() {
        let result = instrumented_function(42);
        assert_eq!(result, 84);
    }
}