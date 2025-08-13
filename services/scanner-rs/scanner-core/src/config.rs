//! Scanner configuration management

use serde::{Deserialize, Serialize};
use std::time::Duration;
use anyhow::{Result, anyhow};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScannerConfig {
    /// NATS JetStream connection URL
    pub nats_url: String,
    
    /// Redis connection URL for idempotency
    pub redis_url: String,
    
    /// Consumer group name for this scanner instance
    pub consumer_group: String,
    
    /// Maximum concurrent jobs
    pub max_concurrent_jobs: usize,
    
    /// Job timeout duration
    pub job_timeout: Duration,
    
    /// Service port
    pub port: u16,
    
    /// Metrics port
    pub metrics_port: u16,
    
    /// Tenant ID
    pub tenant_id: String,
    
    /// Tool configurations
    pub tools: ToolsConfig,
    
    /// Fingerprinting configuration
    pub fingerprinting: FingerprintConfig,
    
    /// Risk tagging configuration  
    pub risk_tagging: RiskTagConfig,
    
    /// Observability configuration
    pub observability: ObservabilityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsConfig {
    /// Path to nmap binary
    pub nmap_path: String,
    
    /// Path to nuclei binary
    pub nuclei_path: String,
    
    /// Path to sslscan binary
    pub sslscan_path: String,
    
    /// Path to nikto binary
    pub nikto_path: String,
    
    /// Tool timeout per scan
    pub tool_timeout: Duration,
    
    /// Maximum retries per tool
    pub max_retries: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintConfig {
    /// Enable OS fingerprinting
    pub enable_os_fingerprinting: bool,
    
    /// Enable technology stack detection
    pub enable_tech_stack: bool,
    
    /// Enable behavioral fingerprinting
    pub enable_behavioral: bool,
    
    /// Confidence threshold for fingerprints
    pub confidence_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskTagConfig {
    /// Enable automatic risk tagging
    pub enable_auto_tagging: bool,
    
    /// Risk scoring model to use
    pub scoring_model: String,
    
    /// Minimum risk score for tagging
    pub min_risk_score: f64,
    
    /// Compliance frameworks to check
    pub compliance_frameworks: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Enable Prometheus metrics
    pub enable_metrics: bool,
    
    /// Metrics bind address
    pub metrics_addr: String,
    
    /// Enable distributed tracing
    pub enable_tracing: bool,
    
    /// Jaeger endpoint for traces
    pub jaeger_endpoint: Option<String>,
}

impl Default for ScannerConfig {
    fn default() -> Self {
        Self {
            nats_url: "nats://localhost:4222".to_string(),
            redis_url: "redis://localhost:6379".to_string(),
            consumer_group: "scanner-workers".to_string(),
            max_concurrent_jobs: 10,
            job_timeout: Duration::from_secs(3600), // 1 hour
            port: 8080,
            metrics_port: 9090,
            tenant_id: "default".to_string(),
            tools: ToolsConfig::default(),
            fingerprinting: FingerprintConfig::default(),
            risk_tagging: RiskTagConfig::default(),
            observability: ObservabilityConfig::default(),
        }
    }
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            nmap_path: "nmap".to_string(),
            nuclei_path: "nuclei".to_string(),
            sslscan_path: "sslscan".to_string(),
            nikto_path: "nikto".to_string(),
            tool_timeout: Duration::from_secs(300), // 5 minutes
            max_retries: 3,
        }
    }
}

impl Default for FingerprintConfig {
    fn default() -> Self {
        Self {
            enable_os_fingerprinting: true,
            enable_tech_stack: true,
            enable_behavioral: false, // Disabled by default for performance
            confidence_threshold: 0.85,
        }
    }
}

impl Default for RiskTagConfig {
    fn default() -> Self {
        Self {
            enable_auto_tagging: true,
            scoring_model: "cvss_v3".to_string(),
            min_risk_score: 4.0, // Medium severity
            compliance_frameworks: vec![
                "PCI-DSS".to_string(),
                "HIPAA".to_string(),
                "SOX".to_string(),
            ],
        }
    }
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            metrics_addr: "0.0.0.0:9090".to_string(),
            enable_tracing: true,
            jaeger_endpoint: Some("http://localhost:14268/api/traces".to_string()),
        }
    }
}

impl ScannerConfig {
    /// Load configuration from environment variables and files
    pub fn load() -> Result<Self> {
        let mut config = config::Config::builder()
            .add_source(config::Environment::with_prefix("SCANNER"))
            .build()?;
            
        // Try to load from config file if present
        if let Ok(config_file) = std::env::var("SCANNER_CONFIG_FILE") {
            config = config::Config::builder()
                .add_source(config::File::with_name(&config_file))
                .add_source(config::Environment::with_prefix("SCANNER"))
                .build()?;
        }
        
        let scanner_config: ScannerConfig = config.try_deserialize()?;
        Ok(scanner_config)
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.max_concurrent_jobs == 0 {
            return Err(anyhow!("max_concurrent_jobs must be greater than 0"));
        }
        
        if self.job_timeout.as_secs() == 0 {
            return Err(anyhow!("job_timeout must be greater than 0"));
        }
        
        if self.fingerprinting.confidence_threshold < 0.0 || self.fingerprinting.confidence_threshold > 1.0 {
            return Err(anyhow!("fingerprinting confidence_threshold must be between 0.0 and 1.0"));
        }
        
        if self.risk_tagging.min_risk_score < 0.0 || self.risk_tagging.min_risk_score > 10.0 {
            return Err(anyhow!("risk_tagging min_risk_score must be between 0.0 and 10.0"));
        }
        
        Ok(())
    }
    
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }
    
    pub fn with_metrics_port(mut self, metrics_port: u16) -> Self {
        self.metrics_port = metrics_port;
        self
    }
    
    pub fn with_tenant_id(mut self, tenant_id: String) -> Self {
        self.tenant_id = tenant_id;
        self
    }
}