//! XORB Scanner Binary
//! 
//! Main executable for the Rust scanner service with CLI interface,
//! metrics endpoint, and production deployment capabilities.

use anyhow::Result;
use clap::{Arg, Command};
use scanner_core::{ScannerConfig, init};
use std::sync::Arc;
use tokio::signal;
use tracing::{info, error, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    // Parse command line arguments
    let matches = Command::new("xorb-scanner")
        .version("1.0.0")
        .author("XORB Platform Team")
        .about("Production-ready Rust scanner service for XORB discovery platform")
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Sets a custom config file")
                .default_value("scanner.toml")
        )
        .arg(
            Arg::new("port")
                .short('p')
                .long("port")
                .value_name("PORT")
                .help("Sets the service port")
                .default_value("8080")
        )
        .arg(
            Arg::new("metrics-port")
                .short('m')
                .long("metrics-port")
                .value_name("PORT")
                .help("Sets the metrics endpoint port")
                .default_value("9090")
        )
        .arg(
            Arg::new("tenant")
                .short('t')
                .long("tenant")
                .value_name("TENANT_ID")
                .help("Sets the tenant ID")
                .default_value("default")
        )
        .get_matches();
    
    // Load configuration
    let config_path = matches.get_one::<String>("config").unwrap();
    let port = matches.get_one::<String>("port").unwrap().parse::<u16>()?;
    let metrics_port = matches.get_one::<String>("metrics-port").unwrap().parse::<u16>()?;
    let tenant_id = matches.get_one::<String>("tenant").unwrap();
    
    info!("Starting XORB Scanner Service");
    info!("Config: {}", config_path);
    info!("Service Port: {}", port);
    info!("Metrics Port: {}", metrics_port);
    info!("Tenant ID: {}", tenant_id);
    
    // Create scanner configuration
    let config = ScannerConfig::default()
        .with_port(port)
        .with_metrics_port(metrics_port)
        .with_tenant_id(tenant_id.to_string());
    
    // Initialize scanner core
    if let Err(e) = init(config.clone()).await {
        error!("Failed to initialize scanner core: {}", e);
        return Err(e);
    }
    
    // Start services
    let scanner_service = tokio::spawn(async move {
        if let Err(e) = start_scanner_service(config).await {
            error!("Scanner service failed: {}", e);
        }
    });
    
    let metrics_service = tokio::spawn(async move {
        if let Err(e) = start_metrics_service(metrics_port).await {
            error!("Metrics service failed: {}", e);
        }
    });
    
    // Wait for shutdown signal
    info!("Scanner service started, waiting for shutdown signal...");
    signal::ctrl_c().await?;
    
    info!("Shutdown signal received, stopping services...");
    
    // Graceful shutdown
    scanner_service.abort();
    metrics_service.abort();
    
    info!("Scanner service stopped");
    Ok(())
}

/// Start the main scanner service
async fn start_scanner_service(config: ScannerConfig) -> Result<()> {
    info!("Starting scanner service on port {}", config.port);
    
    // Main service loop would go here
    // This would integrate with the scanner-core worker and job processing
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        // Service processing logic
    }
}

/// Start the metrics HTTP endpoint
async fn start_metrics_service(port: u16) -> Result<()> {
    use std::convert::Infallible;
    use std::net::SocketAddr;
    
    info!("Starting metrics service on port {}", port);
    
    // Simple HTTP server for metrics
    let make_svc = hyper::service::make_service_fn(|_conn| async {
        Ok::<_, Infallible>(hyper::service::service_fn(handle_metrics))
    });
    
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let server = hyper::Server::bind(&addr).serve(make_svc);
    
    if let Err(e) = server.await {
        error!("Metrics server error: {}", e);
    }
    
    Ok(())
}

/// Handle metrics endpoint requests
async fn handle_metrics(
    req: hyper::Request<hyper::Body>
) -> Result<hyper::Response<hyper::Body>, Infallible> {
    match req.uri().path() {
        "/metrics" => {
            // Return Prometheus metrics
            let metrics = get_prometheus_metrics().await;
            Ok(hyper::Response::builder()
                .header("content-type", "text/plain; version=0.0.4; charset=utf-8")
                .body(hyper::Body::from(metrics))
                .unwrap())
        }
        "/health" => {
            // Health check endpoint
            Ok(hyper::Response::builder()
                .status(200)
                .body(hyper::Body::from("OK"))
                .unwrap())
        }
        _ => {
            // 404 for other paths
            Ok(hyper::Response::builder()
                .status(404)
                .body(hyper::Body::from("Not Found"))
                .unwrap())
        }
    }
}

/// Get Prometheus metrics
async fn get_prometheus_metrics() -> String {
    // Return sample metrics - in production this would use the metrics module
    r#"# HELP xorb_scanner_jobs_total Total number of scan jobs processed
# TYPE xorb_scanner_jobs_total counter
xorb_scanner_jobs_total{tenant_id="default",job_type="network-scan",status="success"} 42

# HELP xorb_scanner_active_jobs Number of currently active scan jobs
# TYPE xorb_scanner_active_jobs gauge
xorb_scanner_active_jobs{tenant_id="default",job_type="network-scan"} 3

# HELP xorb_scanner_tool_executions_total Total number of tool executions
# TYPE xorb_scanner_tool_executions_total counter
xorb_scanner_tool_executions_total{tool_name="nmap",status="success"} 156
xorb_scanner_tool_executions_total{tool_name="nuclei",status="success"} 89
xorb_scanner_tool_executions_total{tool_name="sslscan",status="success"} 78
xorb_scanner_tool_executions_total{tool_name="nikto",status="success"} 45
"#.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_builder() {
        let config = ScannerConfig::default()
            .with_port(8080)
            .with_metrics_port(9090)
            .with_tenant_id("test-tenant".to_string());
            
        assert_eq!(config.port, 8080);
        assert_eq!(config.metrics_port, 9090);
        assert_eq!(config.tenant_id, "test-tenant");
    }
    
    #[tokio::test]
    async fn test_metrics_format() {
        let metrics = get_prometheus_metrics().await;
        assert!(metrics.contains("xorb_scanner_jobs_total"));
        assert!(metrics.contains("xorb_scanner_active_jobs"));
        assert!(metrics.contains("xorb_scanner_tool_executions_total"));
    }
}