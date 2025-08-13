//! Observability and metrics for the Rust scanner service
//! 
//! Implements comprehensive observability with Prometheus metrics,
//! distributed tracing, and structured logging.

use anyhow::Result;
use prometheus::{
    register_counter_vec, register_gauge_vec, register_histogram_vec,
    CounterVec, GaugeVec, HistogramVec, Encoder, TextEncoder
};
use std::time::{Duration, Instant};
use tracing::{info, warn, error, instrument};
use once_cell::sync::Lazy;

// Prometheus metrics registry
static SCAN_JOBS_TOTAL: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        "xorb_scanner_jobs_total",
        "Total number of scan jobs processed",
        &["tenant_id", "job_type", "status"]
    ).expect("Failed to register scan_jobs_total metric")
});

static SCAN_JOB_DURATION: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "xorb_scanner_job_duration_seconds",
        "Duration of scan job execution",
        &["tenant_id", "job_type", "tool"],
        vec![0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0]
    ).expect("Failed to register scan_job_duration metric")
});

static TOOL_EXECUTIONS_TOTAL: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        "xorb_scanner_tool_executions_total",
        "Total number of tool executions",
        &["tool_name", "status"]
    ).expect("Failed to register tool_executions_total metric")
});

static TOOL_EXECUTION_DURATION: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "xorb_scanner_tool_execution_duration_seconds",
        "Duration of tool execution",
        &["tool_name"],
        vec![0.1, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0]
    ).expect("Failed to register tool_execution_duration metric")
});

static ASSETS_DISCOVERED_TOTAL: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        "xorb_scanner_assets_discovered_total",
        "Total number of assets discovered",
        &["tenant_id", "asset_type"]
    ).expect("Failed to register assets_discovered_total metric")
});

static VULNERABILITIES_FOUND_TOTAL: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        "xorb_scanner_vulnerabilities_found_total",
        "Total number of vulnerabilities found",
        &["tenant_id", "severity", "tool"]
    ).expect("Failed to register vulnerabilities_found_total metric")
});

static FINGERPRINT_OPERATIONS_TOTAL: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        "xorb_scanner_fingerprint_operations_total",
        "Total number of fingerprinting operations",
        &["tenant_id", "operation", "status"]
    ).expect("Failed to register fingerprint_operations_total metric")
});

static FINGERPRINT_CONFIDENCE: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "xorb_scanner_fingerprint_confidence_score",
        "Confidence score of fingerprinting operations",
        &["tenant_id"],
        vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ).expect("Failed to register fingerprint_confidence metric")
});

static RISK_ASSESSMENTS_TOTAL: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        "xorb_scanner_risk_assessments_total",
        "Total number of risk assessments",
        &["tenant_id", "risk_level"]
    ).expect("Failed to register risk_assessments_total metric")
});

static RISK_SCORE_DISTRIBUTION: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "xorb_scanner_risk_score_distribution",
        "Distribution of risk scores",
        &["tenant_id"],
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ).expect("Failed to register risk_score_distribution metric")
});

static ACTIVE_SCAN_JOBS: Lazy<GaugeVec> = Lazy::new(|| {
    register_gauge_vec!(
        "xorb_scanner_active_jobs",
        "Number of currently active scan jobs",
        &["tenant_id", "job_type"]
    ).expect("Failed to register active_scan_jobs metric")
});

static QUEUE_SIZE: Lazy<GaugeVec> = Lazy::new(|| {
    register_gauge_vec!(
        "xorb_scanner_queue_size",
        "Size of the job queue",
        &["tenant_id", "queue_type"]
    ).expect("Failed to register queue_size metric")
});

static BUS_MESSAGES_TOTAL: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        "xorb_scanner_bus_messages_total",
        "Total number of bus messages",
        &["direction", "message_type", "status"]
    ).expect("Failed to register bus_messages_total metric")
});

static BUS_MESSAGE_PROCESSING_DURATION: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "xorb_scanner_bus_message_processing_duration_seconds",
        "Duration of bus message processing",
        &["message_type"],
        vec![0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    ).expect("Failed to register bus_message_processing_duration metric")
});

/// Metrics collector for the scanner service
#[derive(Debug, Clone)]
pub struct ScannerMetrics {
    service_name: String,
}

impl ScannerMetrics {
    pub fn new(service_name: String) -> Self {
        Self { service_name }
    }
    
    /// Record a scan job completion
    #[instrument(skip(self), fields(service = %self.service_name))]
    pub fn record_scan_job(&self, tenant_id: &str, job_type: &str, status: &str, duration: Duration) {
        SCAN_JOBS_TOTAL
            .with_label_values(&[tenant_id, job_type, status])
            .inc();
        
        SCAN_JOB_DURATION
            .with_label_values(&[tenant_id, job_type, "overall"])
            .observe(duration.as_secs_f64());
        
        info!(
            tenant_id = %tenant_id,
            job_type = %job_type,
            status = %status,
            duration_seconds = %duration.as_secs_f64(),
            "Scan job completed"
        );
    }
    
    /// Get metrics in Prometheus format
    pub fn get_metrics(&self) -> Result<String> {
        let encoder = TextEncoder::new();
        let metric_families = prometheus::gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
}

/// Timer utility for measuring operation duration
pub struct Timer {
    start: Instant,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }
    
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize metrics subsystem
pub fn init_metrics() {
    // Lazy statics will be initialized on first access
    Lazy::force(&SCAN_JOBS_TOTAL);
    Lazy::force(&SCAN_JOB_DURATION);
    Lazy::force(&TOOL_EXECUTIONS_TOTAL);
    Lazy::force(&TOOL_EXECUTION_DURATION);
    Lazy::force(&ASSETS_DISCOVERED_TOTAL);
    Lazy::force(&VULNERABILITIES_FOUND_TOTAL);
    Lazy::force(&FINGERPRINT_OPERATIONS_TOTAL);
    Lazy::force(&FINGERPRINT_CONFIDENCE);
    Lazy::force(&RISK_ASSESSMENTS_TOTAL);
    Lazy::force(&RISK_SCORE_DISTRIBUTION);
    Lazy::force(&ACTIVE_SCAN_JOBS);
    Lazy::force(&QUEUE_SIZE);
    Lazy::force(&BUS_MESSAGES_TOTAL);
    Lazy::force(&BUS_MESSAGE_PROCESSING_DURATION);
    
    info!("Scanner metrics initialized");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[test]
    fn test_metrics_initialization() {
        init_metrics();
        // Should not panic
    }
    
    #[test]
    fn test_scanner_metrics() {
        let metrics = ScannerMetrics::new("test-scanner".to_string());
        
        // Test scan job recording
        metrics.record_scan_job("test-tenant", "network-scan", "success", Duration::from_secs(30));
        
        // Test metrics export
        let exported = metrics.get_metrics().expect("Failed to export metrics");
        assert!(exported.contains("xorb_scanner_jobs_total"));
    }
    
    #[test]
    fn test_timer() {
        let timer = Timer::new();
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.elapsed();
        assert!(elapsed >= Duration::from_millis(10));
    }
}