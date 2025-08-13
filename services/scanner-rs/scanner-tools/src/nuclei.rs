//! Nuclei vulnerability scanner integration
//!
//! Modern vulnerability scanner with extensive template support.
//! Provides JSON output parsing and vulnerability classification.

use crate::{SecurityTool, ToolOptions, ToolResult, ExecutionStatus, Finding, Severity, ExecutionMetadata};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::process::Stdio;
use tokio::process::Command;
use tokio::time::{timeout, Duration};
use tracing::{info, warn, error, instrument};
use uuid::Uuid;

/// Nuclei tool wrapper
#[derive(Debug, Clone)]
pub struct NucleiTool {
    binary_path: String,
}

/// Nuclei JSON output structure
#[derive(Debug, Deserialize)]
struct NucleiResult {
    #[serde(rename = "template-id")]
    template_id: String,
    #[serde(rename = "template-path")]
    template_path: Option<String>,
    info: NucleiInfo,
    #[serde(rename = "matched-at")]
    matched_at: String,
    #[serde(rename = "extracted-results")]
    extracted_results: Option<Vec<String>>,
    #[serde(rename = "curl-command")]
    curl_command: Option<String>,
}

#[derive(Debug, Deserialize)]
struct NucleiInfo {
    name: String,
    author: Option<Vec<String>>,
    tags: Option<Vec<String>>,
    description: Option<String>,
    severity: String,
    #[serde(rename = "cvss-score")]
    cvss_score: Option<f64>,
    reference: Option<Vec<String>>,
    classification: Option<NucleiClassification>,
}

#[derive(Debug, Deserialize)]
struct NucleiClassification {
    #[serde(rename = "cve-id")]
    cve_id: Option<Vec<String>>,
    #[serde(rename = "cwe-id")]
    cwe_id: Option<Vec<String>>,
    #[serde(rename = "cvss-score")]
    cvss_score: Option<f64>,
}

impl NucleiTool {
    /// Create new Nuclei tool instance
    pub fn new() -> Self {
        Self {
            binary_path: "nuclei".to_string(),
        }
    }
    
    /// Create with custom binary path
    pub fn with_binary_path(path: &str) -> Self {
        Self {
            binary_path: path.to_string(),
        }
    }
    
    /// Build nuclei command arguments
    fn build_args(&self, target: &str, options: &ToolOptions) -> Vec<String> {
        let mut args = vec![
            "-u".to_string(),
            target.to_string(),
            "-json".to_string(), // JSON output
            "-silent".to_string(), // Reduce noise
        ];
        
        // Scan mode based on options
        if options.aggressive {
            args.extend(vec![
                "-severity".to_string(),
                "critical,high,medium,low,info".to_string(),
                "-rate-limit".to_string(),
                "150".to_string(),
            ]);
        } else if options.stealth {
            args.extend(vec![
                "-severity".to_string(),
                "critical,high".to_string(),
                "-rate-limit".to_string(),
                "10".to_string(),
            ]);
        } else {
            args.extend(vec![
                "-severity".to_string(),
                "critical,high,medium".to_string(),
                "-rate-limit".to_string(),
                "50".to_string(),
            ]);
        }
        
        // Add timeout
        args.extend(vec![
            "-timeout".to_string(),
            "10".to_string(), // 10 seconds per request
        ]);
        
        // Add extra arguments
        args.extend(options.extra_args.clone());
        
        args
    }
    
    /// Parse Nuclei JSON output
    fn parse_json_output(&self, output: &str, target: &str) -> Result<Vec<Finding>> {
        let mut findings = Vec::new();
        
        // Nuclei outputs one JSON object per line
        for line in output.lines() {
            if line.trim().is_empty() {
                continue;
            }
            
            match serde_json::from_str::<NucleiResult>(line) {
                Ok(result) => {
                    let mut metadata = std::collections::HashMap::new();
                    metadata.insert("template_id".to_string(), result.template_id.clone());
                    metadata.insert("matched_at".to_string(), result.matched_at.clone());
                    
                    if let Some(path) = &result.template_path {
                        metadata.insert("template_path".to_string(), path.clone());
                    }
                    
                    if let Some(tags) = &result.info.tags {
                        metadata.insert("tags".to_string(), tags.join(","));
                    }
                    
                    if let Some(authors) = &result.info.author {
                        metadata.insert("author".to_string(), authors.join(","));
                    }
                    
                    if let Some(references) = &result.info.reference {
                        metadata.insert("references".to_string(), references.join(","));
                    }
                    
                    // Add CVE information if available
                    if let Some(classification) = &result.info.classification {
                        if let Some(cve_ids) = &classification.cve_id {
                            metadata.insert("cve_id".to_string(), cve_ids.join(","));
                        }
                        if let Some(cwe_ids) = &classification.cwe_id {
                            metadata.insert("cwe_id".to_string(), cwe_ids.join(","));
                        }
                    }
                    
                    let severity = match result.info.severity.to_lowercase().as_str() {
                        "critical" => Severity::Critical,
                        "high" => Severity::High,
                        "medium" => Severity::Medium,
                        "low" => Severity::Low,
                        _ => Severity::Info,
                    };
                    
                    let description = result.info.description
                        .unwrap_or_else(|| format!("Vulnerability detected by template: {}", result.template_id));
                    
                    findings.push(Finding {
                        id: Uuid::new_v4().to_string(),
                        title: result.info.name,
                        description,
                        severity,
                        target: result.matched_at,
                        metadata,
                    });
                }
                Err(e) => {
                    warn!(error = %e, line = %line, "Failed to parse Nuclei JSON line");
                }
            }
        }
        
        Ok(findings)
    }
    
    /// Convert Nuclei severity to our severity enum
    fn map_severity(nuclei_severity: &str) -> Severity {
        match nuclei_severity.to_lowercase().as_str() {
            "critical" => Severity::Critical,
            "high" => Severity::High,
            "medium" => Severity::Medium,
            "low" => Severity::Low,
            "info" => Severity::Info,
            _ => Severity::Info,
        }
    }
}

impl Default for NucleiTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SecurityTool for NucleiTool {
    fn name(&self) -> &str {
        "nuclei"
    }
    
    #[instrument(skip(self, options), fields(tool = "nuclei", target = %target))]
    async fn scan(&self, target: &str, options: &ToolOptions) -> Result<ToolResult> {
        let start_time = chrono::Utc::now();
        let start_instant = std::time::Instant::now();
        
        info!(target = %target, "Starting Nuclei vulnerability scan");
        
        // Build command arguments
        let args = self.build_args(target, options);
        let command_line = format!("{} {}", self.binary_path, args.join(" "));
        
        // Execute nuclei with timeout
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
                    let findings = self.parse_json_output(&stdout, target)
                        .unwrap_or_else(|e| {
                            warn!(error = %e, "Failed to parse Nuclei output, returning empty findings");
                            Vec::new()
                        });
                    
                    info!(
                        target = %target,
                        duration_ms = %execution_time.as_millis(),
                        findings_count = %findings.len(),
                        "Nuclei scan completed successfully"
                    );
                    
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
                        format!("Nuclei exited with code: {:?}", output.status.code())
                    } else {
                        stderr.to_string()
                    };
                    
                    error!(
                        target = %target,
                        error = %error_msg,
                        "Nuclei scan failed"
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
                error!(target = %target, error = %e, "Failed to execute Nuclei");
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
                warn!(target = %target, timeout_ms = %options.timeout.as_millis(), "Nuclei scan timed out");
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
            .arg("-version")
            .output()
            .await
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
    
    async fn version(&self) -> Result<String> {
        let output = Command::new(&self.binary_path)
            .arg("-version")
            .output()
            .await
            .map_err(|e| anyhow!("Failed to get Nuclei version: {}", e))?;
            
        if output.status.success() {
            let version_output = String::from_utf8_lossy(&output.stdout);
            // Parse version from output
            for line in version_output.lines() {
                if line.contains("Nuclei") || line.contains("Current Version:") {
                    return Ok(line.trim().to_string());
                }
            }
            Ok("Nuclei (unknown version)".to_string())
        } else {
            Err(anyhow!("Failed to get Nuclei version"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToolOptions, OutputFormat};
    
    #[test]
    fn test_nuclei_tool_creation() {
        let tool = NucleiTool::new();
        assert_eq!(tool.name(), "nuclei");
        assert_eq!(tool.binary_path, "nuclei");
        
        let tool_custom = NucleiTool::with_binary_path("/usr/local/bin/nuclei");
        assert_eq!(tool_custom.binary_path, "/usr/local/bin/nuclei");
    }
    
    #[test]
    fn test_build_args() {
        let tool = NucleiTool::new();
        let options = ToolOptions {
            aggressive: true,
            extra_args: vec!["-tags".to_string(), "cve".to_string()],
            ..Default::default()
        };
        
        let args = tool.build_args("https://example.com", &options);
        assert!(args.contains(&"-u".to_string()));
        assert!(args.contains(&"https://example.com".to_string()));
        assert!(args.contains(&"-json".to_string()));
        assert!(args.contains(&"-severity".to_string()));
        assert!(args.contains(&"-tags".to_string()));
    }
    
    #[test]
    fn test_parse_json_output() {
        let tool = NucleiTool::new();
        let json_output = r#"{"template-id":"CVE-2021-44228","info":{"name":"Apache Log4j RCE","author":["daffainfo"],"tags":["cve","rce","log4j"],"description":"Apache Log4j2 <=2.14.1 JNDI features used in configuration, log messages, and parameters do not protect against attacker controlled LDAP and other JNDI related endpoints.","severity":"critical","cvss-score":10.0},"matched-at":"https://example.com:443"}"#;
        
        let findings = tool.parse_json_output(json_output, "https://example.com").unwrap();
        assert_eq!(findings.len(), 1);
        
        let finding = &findings[0];
        assert_eq!(finding.title, "Apache Log4j RCE");
        assert_eq!(finding.severity, Severity::Critical);
        assert!(finding.description.contains("Log4j"));
        assert_eq!(finding.target, "https://example.com:443");
        assert!(finding.metadata.contains_key("template_id"));
        assert!(finding.metadata.contains_key("tags"));
    }
    
    #[test]
    fn test_severity_mapping() {
        assert_eq!(NucleiTool::map_severity("critical"), Severity::Critical);
        assert_eq!(NucleiTool::map_severity("HIGH"), Severity::High);
        assert_eq!(NucleiTool::map_severity("Medium"), Severity::Medium);
        assert_eq!(NucleiTool::map_severity("low"), Severity::Low);
        assert_eq!(NucleiTool::map_severity("info"), Severity::Info);
        assert_eq!(NucleiTool::map_severity("unknown"), Severity::Info);
    }
    
    #[tokio::test]
    async fn test_tool_availability() {
        let tool = NucleiTool::new();
        // This test will pass if nuclei is installed, otherwise it will be false
        let _available = tool.is_available().await;
        // We don't assert here since nuclei might not be installed in test environment
    }
}