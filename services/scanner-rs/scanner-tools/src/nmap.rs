//! Nmap integration for network discovery and port scanning
//!
//! Provides production-ready Nmap wrapper with XML output parsing,
//! service detection, and OS fingerprinting capabilities.

use crate::{SecurityTool, ToolOptions, ToolResult, ExecutionStatus, Finding, Severity, ExecutionMetadata};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use quick_xml::de::from_str;
use serde::{Deserialize, Serialize};
use std::process::Stdio;
use tokio::process::Command;
use tokio::time::{timeout, Duration};
use tracing::{info, warn, error, instrument};
use uuid::Uuid;

/// Nmap tool wrapper
#[derive(Debug, Clone)]
pub struct NmapTool {
    binary_path: String,
}

/// Nmap XML output structure
#[derive(Debug, Deserialize)]
struct NmapRun {
    #[serde(rename = "host")]
    hosts: Vec<Host>,
}

#[derive(Debug, Deserialize)]
struct Host {
    #[serde(rename = "address")]
    addresses: Vec<Address>,
    #[serde(rename = "ports")]
    ports: Option<Ports>,
    #[serde(rename = "os")]
    os: Option<Os>,
}

#[derive(Debug, Deserialize)]
struct Address {
    #[serde(rename = "addr")]
    addr: String,
    #[serde(rename = "addrtype")]
    addrtype: String,
}

#[derive(Debug, Deserialize)]
struct Ports {
    #[serde(rename = "port")]
    port: Vec<Port>,
}

#[derive(Debug, Deserialize)]
struct Port {
    #[serde(rename = "portid")]
    portid: u16,
    #[serde(rename = "protocol")]
    protocol: String,
    #[serde(rename = "state")]
    state: PortState,
    #[serde(rename = "service")]
    service: Option<Service>,
}

#[derive(Debug, Deserialize)]
struct PortState {
    #[serde(rename = "state")]
    state: String,
}

#[derive(Debug, Deserialize)]
struct Service {
    #[serde(rename = "name")]
    name: Option<String>,
    #[serde(rename = "version")]
    version: Option<String>,
    #[serde(rename = "product")]
    product: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Os {
    #[serde(rename = "osmatch")]
    osmatch: Vec<OsMatch>,
}

#[derive(Debug, Deserialize)]
struct OsMatch {
    #[serde(rename = "name")]
    name: String,
    #[serde(rename = "accuracy")]
    accuracy: u8,
}

impl NmapTool {
    /// Create new Nmap tool instance
    pub fn new() -> Self {
        Self {
            binary_path: "nmap".to_string(),
        }
    }
    
    /// Create with custom binary path
    pub fn with_binary_path(path: &str) -> Self {
        Self {
            binary_path: path.to_string(),
        }
    }
    
    /// Build nmap command arguments
    fn build_args(&self, target: &str, options: &ToolOptions) -> Vec<String> {
        let mut args = vec![
            "-oX".to_string(), // XML output
            "-".to_string(),   // Output to stdout
        ];
        
        // Scan mode based on options
        if options.stealth {
            args.extend(vec!["-sS".to_string(), "-T2".to_string()]);
        } else if options.aggressive {
            args.extend(vec!["-A".to_string(), "-T4".to_string()]);
        } else {
            args.extend(vec!["-sV".to_string(), "-T3".to_string()]);
        }
        
        // Service detection
        args.push("-sV".to_string());
        
        // OS detection (if not stealth)
        if !options.stealth {
            args.push("-O".to_string());
        }
        
        // Add extra arguments
        args.extend(options.extra_args.clone());
        
        // Add target
        args.push(target.to_string());
        
        args
    }
    
    /// Parse Nmap XML output
    fn parse_xml_output(&self, output: &str, target: &str) -> Result<Vec<Finding>> {
        let mut findings = Vec::new();
        
        match from_str::<NmapRun>(output) {
            Ok(nmap_run) => {
                for host in nmap_run.hosts {
                    // Get host IP
                    let host_ip = host.addresses
                        .iter()
                        .find(|addr| addr.addrtype == "ipv4")
                        .map(|addr| addr.addr.clone())
                        .unwrap_or_else(|| target.to_string());
                    
                    // Process ports
                    if let Some(ports) = host.ports {
                        for port in ports.port {
                            if port.state.state == "open" {
                                let mut metadata = std::collections::HashMap::new();
                                metadata.insert("protocol".to_string(), port.protocol.clone());
                                metadata.insert("state".to_string(), port.state.state.clone());
                                
                                let (service_name, service_version) = if let Some(service) = &port.service {
                                    let name = service.name.clone().unwrap_or("unknown".to_string());
                                    let version = service.version.clone().unwrap_or_else(|| {
                                        service.product.clone().unwrap_or("unknown".to_string())
                                    });
                                    metadata.insert("service".to_string(), name.clone());
                                    metadata.insert("version".to_string(), version.clone());
                                    (name, version)
                                } else {
                                    ("unknown".to_string(), "unknown".to_string())
                                };
                                
                                findings.push(Finding {
                                    id: Uuid::new_v4().to_string(),
                                    title: format!("Open Port: {}/{}", port.portid, port.protocol),
                                    description: format!(
                                        "Port {}/{} is open on {} running {} ({})",
                                        port.portid, port.protocol, host_ip, service_name, service_version
                                    ),
                                    severity: Severity::Info,
                                    target: format!("{}:{}", host_ip, port.portid),
                                    metadata,
                                });
                            }
                        }
                    }
                    
                    // Process OS detection
                    if let Some(os) = host.os {
                        for os_match in os.osmatch {
                            if os_match.accuracy >= 80 {
                                let mut metadata = std::collections::HashMap::new();
                                metadata.insert("accuracy".to_string(), os_match.accuracy.to_string());
                                metadata.insert("os_type".to_string(), "operating_system".to_string());
                                
                                findings.push(Finding {
                                    id: Uuid::new_v4().to_string(),
                                    title: "OS Detection".to_string(),
                                    description: format!(
                                        "Detected OS: {} ({}% accuracy)",
                                        os_match.name, os_match.accuracy
                                    ),
                                    severity: Severity::Info,
                                    target: host_ip.clone(),
                                    metadata,
                                });
                            }
                        }
                    }
                }
            }
            Err(e) => {
                warn!(error = %e, "Failed to parse Nmap XML output, using text parsing");
                // Fallback to simple text parsing
                findings.extend(self.parse_text_output(output, target)?);
            }
        }
        
        Ok(findings)
    }
    
    /// Fallback text parsing for Nmap output
    fn parse_text_output(&self, output: &str, target: &str) -> Result<Vec<Finding>> {
        let mut findings = Vec::new();
        
        for line in output.lines() {
            if line.contains("/tcp") && line.contains("open") {
                // Parse line like "80/tcp open  http    nginx 1.18.0"
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let port_proto = parts[0];
                    let port = port_proto.split('/').next().unwrap_or("unknown");
                    let service = if parts.len() > 3 { parts[3] } else { "unknown" };
                    
                    let mut metadata = std::collections::HashMap::new();
                    metadata.insert("protocol".to_string(), "tcp".to_string());
                    metadata.insert("service".to_string(), service.to_string());
                    
                    findings.push(Finding {
                        id: Uuid::new_v4().to_string(),
                        title: format!("Open Port: {}", port_proto),
                        description: format!("Port {} is open on {} running {}", port, target, service),
                        severity: Severity::Info,
                        target: format!("{}:{}", target, port),
                        metadata,
                    });
                }
            }
        }
        
        Ok(findings)
    }
}

impl Default for NmapTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SecurityTool for NmapTool {
    fn name(&self) -> &str {
        "nmap"
    }
    
    #[instrument(skip(self, options), fields(tool = "nmap", target = %target))]
    async fn scan(&self, target: &str, options: &ToolOptions) -> Result<ToolResult> {
        let start_time = chrono::Utc::now();
        let start_instant = std::time::Instant::now();
        
        info!(target = %target, "Starting Nmap scan");
        
        // Build command arguments
        let args = self.build_args(target, options);
        let command_line = format!("{} {}", self.binary_path, args.join(" "));
        
        // Execute nmap with timeout
        let result = timeout(
            options.timeout,
            Command::new(&self.binary_path)
                .args(&args)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
        ).await;
        
        let execution_time = start_instant.elapsed();
        let end_time = chrono::Utc::now();
        
        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                
                if output.status.success() {
                    info!(
                        target = %target,
                        duration_ms = %execution_time.as_millis(),
                        "Nmap scan completed successfully"
                    );
                    
                    let findings = self.parse_xml_output(&stdout, target)
                        .unwrap_or_else(|e| {
                            warn!(error = %e, "Failed to parse output, returning empty findings");
                            Vec::new()
                        });
                    
                    Ok(ToolResult {
                        tool_name: self.name().to_string(),
                        target: target.to_string(),
                        status: ExecutionStatus::Success,
                        raw_output: stdout.to_string(),
                        findings,
                        metadata: ExecutionMetadata {
                            start_time,
                            end_time,
                            duration_ms: execution_time.as_millis() as u64,
                            exit_code: output.status.code(),
                            tool_version: self.version().await.unwrap_or_else(|_| "unknown".to_string()),
                            command_line,
                        },
                    })
                } else {
                    let error_msg = if stderr.is_empty() {
                        format!("Nmap exited with code: {:?}", output.status.code())
                    } else {
                        stderr.to_string()
                    };
                    
                    error!(
                        target = %target,
                        error = %error_msg,
                        "Nmap scan failed"
                    );
                    
                    Ok(ToolResult {
                        tool_name: self.name().to_string(),
                        target: target.to_string(),
                        status: ExecutionStatus::Failed(error_msg),
                        raw_output: format!("STDOUT:\n{}\nSTDERR:\n{}", stdout, stderr),
                        findings: Vec::new(),
                        metadata: ExecutionMetadata {
                            start_time,
                            end_time,
                            duration_ms: execution_time.as_millis() as u64,
                            exit_code: output.status.code(),
                            tool_version: self.version().await.unwrap_or_else(|_| "unknown".to_string()),
                            command_line,
                        },
                    })
                }
            }
            Ok(Err(e)) => {
                error!(target = %target, error = %e, "Failed to execute Nmap");
                Ok(ToolResult {
                    tool_name: self.name().to_string(),
                    target: target.to_string(),
                    status: ExecutionStatus::Failed(e.to_string()),
                    raw_output: String::new(),
                    findings: Vec::new(),
                    metadata: ExecutionMetadata {
                        start_time,
                        end_time,
                        duration_ms: execution_time.as_millis() as u64,
                        exit_code: None,
                        tool_version: "unknown".to_string(),
                        command_line,
                    },
                })
            }
            Err(_) => {
                warn!(target = %target, timeout_ms = %options.timeout.as_millis(), "Nmap scan timed out");
                Ok(ToolResult {
                    tool_name: self.name().to_string(),
                    target: target.to_string(),
                    status: ExecutionStatus::Timeout,
                    raw_output: String::new(),
                    findings: Vec::new(),
                    metadata: ExecutionMetadata {
                        start_time,
                        end_time,
                        duration_ms: execution_time.as_millis() as u64,
                        exit_code: None,
                        tool_version: "unknown".to_string(),
                        command_line,
                    },
                })
            }
        }
    }
    
    async fn is_available(&self) -> bool {
        Command::new(&self.binary_path)
            .arg("--version")
            .output()
            .await
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
    
    async fn version(&self) -> Result<String> {
        let output = Command::new(&self.binary_path)
            .arg("--version")
            .output()
            .await
            .map_err(|e| anyhow!("Failed to get Nmap version: {}", e))?;
            
        if output.status.success() {
            let version_output = String::from_utf8_lossy(&output.stdout);
            // Parse version from output like "Nmap version 7.80"
            for line in version_output.lines() {
                if line.contains("Nmap version") {
                    return Ok(line.to_string());
                }
            }
            Ok("Nmap (unknown version)".to_string())
        } else {
            Err(anyhow!("Failed to get Nmap version"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToolOptions, OutputFormat};
    
    #[test]
    fn test_nmap_tool_creation() {
        let tool = NmapTool::new();
        assert_eq!(tool.name(), "nmap");
        assert_eq!(tool.binary_path, "nmap");
        
        let tool_custom = NmapTool::with_binary_path("/usr/local/bin/nmap");
        assert_eq!(tool_custom.binary_path, "/usr/local/bin/nmap");
    }
    
    #[test]
    fn test_build_args() {
        let tool = NmapTool::new();
        let options = ToolOptions {
            stealth: false,
            aggressive: false,
            extra_args: vec!["-p".to_string(), "80,443".to_string()],
            ..Default::default()
        };
        
        let args = tool.build_args("192.168.1.1", &options);
        assert!(args.contains(&"-oX".to_string()));
        assert!(args.contains(&"-sV".to_string()));
        assert!(args.contains(&"-p".to_string()));
        assert!(args.contains(&"192.168.1.1".to_string()));
    }
    
    #[test]
    fn test_parse_text_output() {
        let tool = NmapTool::new();
        let output = r#"
22/tcp open  ssh     OpenSSH 8.0
80/tcp open  http    nginx 1.18.0
443/tcp open  https   nginx 1.18.0
        "#;
        
        let findings = tool.parse_text_output(output, "192.168.1.1").unwrap();
        assert_eq!(findings.len(), 3);
        
        let ssh_finding = &findings[0];
        assert!(ssh_finding.title.contains("22/tcp"));
        assert!(ssh_finding.description.contains("ssh"));
        assert_eq!(ssh_finding.severity, Severity::Info);
    }
    
    #[tokio::test]
    async fn test_tool_availability() {
        let tool = NmapTool::new();
        // This test will pass if nmap is installed, otherwise it will be false
        let _available = tool.is_available().await;
        // We don't assert here since nmap might not be installed in test environment
    }
}