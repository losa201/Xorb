//! XORB Scanner Core
//! 
//! Core orchestration and infrastructure for the Rust scanner service.
//! Provides gRPC services, NATS integration, metrics, and configuration.

pub mod config;
pub mod metrics;
pub mod tracing;
pub mod bus;
pub mod services;

// Re-export main types
pub use config::ScannerConfig;
pub use metrics::ScannerMetrics;

// Generated protobuf code
pub mod proto {
    tonic::include_proto!("xorb.discovery.v1");
    tonic::include_proto!("xorb.fingerprint.v1");
    tonic::include_proto!("xorb.risktag.v1");
    tonic::include_proto!("xorb.audit.v1");
}

use anyhow::Result;
use tracing::info;

/// Initialize the scanner core with configuration
pub async fn init(config: ScannerConfig) -> Result<()> {
    // Initialize tracing
    crate::tracing::init_tracing(&config.tenant_id).await?;
    
    // Initialize metrics
    crate::metrics::init_metrics();
    
    // Validate configuration
    config.validate()?;
    
    info!(
        tenant_id = %config.tenant_id,
        port = %config.port,
        metrics_port = %config.metrics_port,
        "Scanner core initialized successfully"
    );
    
    Ok(())
}

/// Health check function
pub async fn health_check() -> Result<String> {
    Ok("OK".to_string())
}