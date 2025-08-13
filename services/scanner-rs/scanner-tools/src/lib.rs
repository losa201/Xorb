//! Security Tool Integrations
//! 
//! Production-ready wrappers for security scanning tools:
//! - Nmap: Network discovery and port scanning
//! - Nuclei: Modern vulnerability scanner
//! - SSLScan: SSL/TLS configuration analysis
//! - Nikto: Web application security scanner

pub mod nmap;
pub mod nuclei;
pub mod sslscan;
pub mod nikto;

// Re-export tool implementations
pub use nmap::NmapTool;
pub use nuclei::NucleiTool;
pub use sslscan::SslScanTool;
pub use nikto::NiktoTool;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Common interface for all security tools
#[async_trait::async_trait]
pub trait SecurityTool {
    /// Tool name identifier
    fn name(&self) -> &str;
    
    /// Execute tool scan with target and options
    async fn scan(&self, target: &str, options: &ToolOptions) -> Result<ToolResult>;
    
    /// Check if tool is available on system
    async fn is_available(&self) -> bool;
    
    /// Get tool version information
    async fn version(&self) -> Result<String>;
}

/// Common tool execution options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOptions {
    /// Maximum execution timeout
    pub timeout: Duration,
    
    /// Stealth mode (slower, less detectable)
    pub stealth: bool,
    
    /// Aggressive scanning (faster, more detectable)
    pub aggressive: bool,
    
    /// Additional custom arguments
    pub extra_args: Vec<String>,
    
    /// Output format preference
    pub output_format: OutputFormat,
}

/// Tool output format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    Xml,
    Text,
    Csv,
}

/// Tool execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Tool that generated this result
    pub tool_name: String,
    
    /// Target that was scanned
    pub target: String,
    
    /// Execution status
    pub status: ExecutionStatus,
    
    /// Raw tool output
    pub raw_output: String,
    
    /// Parsed structured findings
    pub findings: Vec<Finding>,
    
    /// Execution metadata
    pub metadata: ExecutionMetadata,
}

/// Tool execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Success,
    Failed(String),
    Timeout,
    NotFound,
}

/// Security finding from tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    /// Finding identifier
    pub id: String,
    
    /// Finding title/summary
    pub title: String,
    
    /// Detailed description
    pub description: String,
    
    /// Severity level
    pub severity: Severity,
    
    /// Affected target/port/service
    pub target: String,
    
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
}

/// Finding severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

/// Tool execution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetadata {
    /// Start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    
    /// End time
    pub end_time: chrono::DateTime<chrono::Utc>,
    
    /// Duration in milliseconds
    pub duration_ms: u64,
    
    /// Exit code
    pub exit_code: Option<i32>,
    
    /// Tool version used
    pub tool_version: String,
    
    /// Command line executed
    pub command_line: String,
}

impl Default for ToolOptions {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(300), // 5 minutes
            stealth: false,
            aggressive: false,
            extra_args: Vec::new(),
            output_format: OutputFormat::Json,
        }
    }
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Info => write!(f, "INFO"),
            Severity::Low => write!(f, "LOW"),
            Severity::Medium => write!(f, "MEDIUM"),
            Severity::High => write!(f, "HIGH"),
            Severity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Tool factory for creating tool instances
pub struct ToolFactory;

impl ToolFactory {
    /// Create all available tools
    pub fn create_all() -> Vec<Box<dyn SecurityTool + Send + Sync>> {
        vec![
            Box::new(NmapTool::new()),
            Box::new(NucleiTool::new()),
            Box::new(SslScanTool::new()),
            Box::new(NiktoTool::new()),
        ]
    }
    
    /// Create tool by name
    pub fn create_by_name(name: &str) -> Option<Box<dyn SecurityTool + Send + Sync>> {
        match name.to_lowercase().as_str() {
            "nmap" => Some(Box::new(NmapTool::new())),
            "nuclei" => Some(Box::new(NucleiTool::new())),
            "sslscan" => Some(Box::new(SslScanTool::new())),
            "nikto" => Some(Box::new(NiktoTool::new())),
            _ => None,
        }
    }
}