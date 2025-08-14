//! Nikto web application scanner integration
//!
//! Comprehensive web application security scanner that tests for
//! thousands of potentially dangerous files, outdated software, and server misconfigurations.

use crate::{SecurityTool, ToolOptions, ToolResult, ExecutionStatus, Finding, Severity, ExecutionMetadata};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use csv::ReaderBuilder;
use serde::{Deserialize, Serialize};
use std::process::Stdio;
use tokio::process::Command;
use tokio::time::{timeout, Duration};
use tracing::{info, warn, error, instrument};
use uuid::Uuid;

/// Nikto tool wrapper
#[derive(Debug, Clone)]
pub struct NiktoTool {
    binary_path: String,
}

/// Nikto CSV output record
#[derive(Debug, Deserialize)]
struct NiktoRecord {
    #[serde(rename = "Host IP")]
    host_ip: String,
    #[serde(rename = "Host Port")]
    host_port: String,
    #[serde(rename = "Test ID")]
    test_id: String,
    #[serde(rename = "Test Description")]
    test_description: String,
    #[serde(rename = "HTTP Method")]
    http_method: String,
    #[serde(rename = "Test URI")]
    test_uri: String,
    #[serde(rename = "Test Data")]
    test_data: Option<String>,
    #[serde(rename = "Test Result")]
    test_result: String,
}

impl NiktoTool {
    /// Create new Nikto tool instance
    pub fn new() -> Self {
        Self {
            binary_path: "nikto".to_string(),
        }
    }

    /// Create with custom binary path
    pub fn with_binary_path(path: &str) -> Self {
        Self {
            binary_path: path.to_string(),
        }
    }

    /// Build nikto command arguments
    fn build_args(&self, target: &str, options: &ToolOptions) -> Vec<String> {
        let mut args = vec![
            "-h".to_string(),
            target.to_string(),
            "-output".to_string(),
            "-".to_string(), // Output to stdout
            "-Format".to_string(),
            "csv".to_string(), // CSV format
        ];

        // Scan mode based on options
        if options.aggressive {
            args.extend(vec![
                "-Tuning".to_string(),
                "1,2,3,4,5,6,7,8,9,0,a,b,c".to_string(), // All tuning options
                "-T".to_string(),
                "4".to_string(), // Timing template (faster)
            ]);
        } else if options.stealth {
            args.extend(vec![
                "-Tuning".to_string(),
                "1,2,3".to_string(), // Basic tests only
                "-T".to_string(),
                "1".to_string(), // Slow timing
                "-timeout".to_string(),
                "10".to_string(),
            ]);
        } else {
            args.extend(vec![
                "-Tuning".to_string(),
                "1,2,3,4,5,6,7,8,9".to_string(), // Standard tests
                "-T".to_string(),
                "3".to_string(), // Normal timing
            ]);
        }

        // Common options
        args.extend(vec![
            "-ask".to_string(),
            "no".to_string(), // Don't prompt for input
            "-nointeractive".to_string(),
        ]);

        // Add extra arguments
        args.extend(options.extra_args.clone());

        args
    }

    /// Parse Nikto CSV output
    fn parse_csv_output(&self, output: &str, target: &str) -> Result<Vec<Finding>> {
        let mut findings = Vec::new();

        // Skip header comments and find CSV data
        let csv_start = output.lines()
            .position(|line| line.starts_with("\"Host IP\""))
            .unwrap_or(0);

        let csv_content = output.lines().skip(csv_start).collect::<Vec<_>>().join("\n");

        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(csv_content.as_bytes());

        for result in reader.deserialize() {
            match result {
                Ok(record) => {
                    let nikto_record: NiktoRecord = record;

                    let mut metadata = std::collections::HashMap::new();
                    metadata.insert("test_id".to_string(), nikto_record.test_id.clone());
                    metadata.insert("http_method".to_string(), nikto_record.http_method.clone());
                    metadata.insert("test_uri".to_string(), nikto_record.test_uri.clone());
                    metadata.insert("host_ip".to_string(), nikto_record.host_ip.clone());
                    metadata.insert("host_port".to_string(), nikto_record.host_port.clone());

                    if let Some(test_data) = &nikto_record.test_data {
                        if !test_data.is_empty() {
                            metadata.insert("test_data".to_string(), test_data.clone());
                        }
                    }

                    // Determine severity based on test ID and description
                    let severity = self.determine_severity(&nikto_record);

                    let target_url = if nikto_record.test_uri.starts_with('/') {
                        format!("{}{}", target, nikto_record.test_uri)
                    } else {
                        nikto_record.test_uri.clone()
                    };

                    findings.push(Finding {
                        id: Uuid::new_v4().to_string(),
                        title: self.format_finding_title(&nikto_record),
                        description: nikto_record.test_description.clone(),
                        severity,
                        target: target_url,
                        metadata,
                    });
                }
                Err(e) => {
                    warn!(error = %e, "Failed to parse Nikto CSV record");
                }
            }
        }

        Ok(findings)
    }

    /// Determine severity based on Nikto test results
    fn determine_severity(&self, record: &NiktoRecord) -> Severity {
        let test_id = &record.test_id;
        let description = record.test_description.to_lowercase();

        // Critical vulnerabilities
        if description.contains("remote code execution")
            || description.contains("sql injection")
            || description.contains("command injection")
            || test_id.starts_with("999") // Usually critical findings
        {
            return Severity::Critical;
        }

        // High severity
        if description.contains("cross-site scripting")
            || description.contains("xss")
            || description.contains("directory traversal")
            || description.contains("file inclusion")
            || description.contains("authentication bypass")
            || description.contains("arbitrary file")
            || test_id.starts_with("001") // Server issues
        {
            return Severity::High;
        }

        // Medium severity
        if description.contains("information disclosure")
            || description.contains("sensitive")
            || description.contains("backup")
            || description.contains("configuration")
            || description.contains("default")
            || test_id.starts_with("002") // File/directory issues
            || test_id.starts_with("003") // CGI issues
        {
            return Severity::Medium;
        }

        // Low severity
        if description.contains("banner")
            || description.contains("version")
            || description.contains("outdated")
            || description.contains("missing")
            || test_id.starts_with("004") // General issues
        {
            return Severity::Low;
        }

        // Default to info for everything else
        Severity::Info
    }

    /// Format a descriptive title for the finding
    fn format_finding_title(&self, record: &NiktoRecord) -> String {
        let test_id = &record.test_id;
        let description = &record.test_description;

        // Extract the first part of the description for the title
        if let Some(first_sentence) = description.split('.').next() {
            if first_sentence.len() > 80 {
                format!("{}: {}...", test_id, &first_sentence[..77])
            } else {
                format!("{}: {}", test_id, first_sentence)
            }
        } else {
            format!("{}: {}", test_id, description)
        }
    }

    /// Fallback text parsing for Nikto output
    fn parse_text_output(&self, output: &str, target: &str) -> Result<Vec<Finding>> {
        let mut findings = Vec::new();

        for line in output.lines() {
            if line.starts_with('+') && line.contains(':') {
                // Parse lines like "+ /admin/: This might be interesting..."
                let parts: Vec<&str> = line.splitn(2, ':').collect();
                if parts.len() == 2 {
                    let path = parts[0].trim_start_matches('+').trim();
                    let description = parts[1].trim();

                    let mut metadata = std::collections::HashMap::new();
                    metadata.insert("test_uri".to_string(), path.to_string());

                    findings.push(Finding {
                        id: Uuid::new_v4().to_string(),
                        title: format!("Nikto Finding: {}", path),
                        description: description.to_string(),
                        severity: Severity::Info,
                        target: if path.starts_with('/') {
                            format!("{}{}", target, path)
                        } else {
                            path.to_string()
                        },
                        metadata,
                    });
                }
            }
        }

        Ok(findings)
    }
}

impl Default for NiktoTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SecurityTool for NiktoTool {
    fn name(&self) -> &str {
        "nikto"
    }

    #[instrument(skip(self, options), fields(tool = "nikto", target = %target))]
    async fn scan(&self, target: &str, options: &ToolOptions) -> Result<ToolResult> {
        let start_time = chrono::Utc::now();
        let start_instant = std::time::Instant::now();

        info!(target = %target, "Starting Nikto web application scan");

        // Build command arguments
        let args = self.build_args(target, options);
        let command_line = format!("{} {}", self.binary_path, args.join(" "));

        // Execute nikto with timeout
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
                    let findings = self.parse_csv_output(&stdout, target)
                        .unwrap_or_else(|e| {
                            warn!(error = %e, "Failed to parse CSV output, trying text parsing");
                            self.parse_text_output(&stdout, target).unwrap_or_else(|e2| {
                                warn!(error = %e2, "Failed to parse Nikto output, returning empty findings");
                                Vec::new()
                            })
                        });

                    info!(
                        target = %target,
                        duration_ms = %execution_time.as_millis(),
                        findings_count = %findings.len(),
                        "Nikto scan completed successfully"
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
                        format!("Nikto exited with code: {:?}", output.status.code())
                    } else {
                        stderr.to_string()
                    };

                    error!(
                        target = %target,
                        error = %error_msg,
                        "Nikto scan failed"
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
                error!(target = %target, error = %e, "Failed to execute Nikto");
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
                warn!(target = %target, timeout_ms = %options.timeout.as_millis(), "Nikto scan timed out");
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
            .arg("-Version")
            .output()
            .await
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    async fn version(&self) -> Result<String> {
        let output = Command::new(&self.binary_path)
            .arg("-Version")
            .output()
            .await
            .map_err(|e| anyhow!("Failed to get Nikto version: {}", e))?;

        if output.status.success() {
            let version_output = String::from_utf8_lossy(&output.stdout);
            // Parse version from output
            for line in version_output.lines() {
                if line.contains("Nikto") && line.contains("version") {
                    return Ok(line.trim().to_string());
                }
            }
            Ok("Nikto (unknown version)".to_string())
        } else {
            Err(anyhow!("Failed to get Nikto version"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToolOptions, OutputFormat};

    #[test]
    fn test_nikto_tool_creation() {
        let tool = NiktoTool::new();
        assert_eq!(tool.name(), "nikto");
        assert_eq!(tool.binary_path, "nikto");

        let tool_custom = NiktoTool::with_binary_path("/usr/local/bin/nikto");
        assert_eq!(tool_custom.binary_path, "/usr/local/bin/nikto");
    }

    #[test]
    fn test_build_args() {
        let tool = NiktoTool::new();
        let options = ToolOptions {
            aggressive: true,
            extra_args: vec!["-ssl".to_string()],
            ..Default::default()
        };

        let args = tool.build_args("https://example.com", &options);
        assert!(args.contains(&"-h".to_string()));
        assert!(args.contains(&"https://example.com".to_string()));
        assert!(args.contains(&"-Format".to_string()));
        assert!(args.contains(&"csv".to_string()));
        assert!(args.contains(&"-ssl".to_string()));
    }

    #[test]
    fn test_severity_determination() {
        let tool = NiktoTool::new();

        let critical_record = NiktoRecord {
            host_ip: "192.168.1.1".to_string(),
            host_port: "80".to_string(),
            test_id: "999001".to_string(),
            test_description: "SQL injection vulnerability detected".to_string(),
            http_method: "GET".to_string(),
            test_uri: "/admin.php".to_string(),
            test_data: None,
            test_result: "FOUND".to_string(),
        };
        assert_eq!(tool.determine_severity(&critical_record), Severity::Critical);

        let high_record = NiktoRecord {
            host_ip: "192.168.1.1".to_string(),
            host_port: "80".to_string(),
            test_id: "001001".to_string(),
            test_description: "Cross-site scripting (XSS) vulnerability".to_string(),
            http_method: "GET".to_string(),
            test_uri: "/search.php".to_string(),
            test_data: None,
            test_result: "FOUND".to_string(),
        };
        assert_eq!(tool.determine_severity(&high_record), Severity::High);

        let info_record = NiktoRecord {
            host_ip: "192.168.1.1".to_string(),
            host_port: "80".to_string(),
            test_id: "000001".to_string(),
            test_description: "Server banner detected".to_string(),
            http_method: "HEAD".to_string(),
            test_uri: "/".to_string(),
            test_data: None,
            test_result: "FOUND".to_string(),
        };
        assert_eq!(tool.determine_severity(&info_record), Severity::Info);
    }

    #[test]
    fn test_format_finding_title() {
        let tool = NiktoTool::new();

        let record = NiktoRecord {
            host_ip: "192.168.1.1".to_string(),
            host_port: "80".to_string(),
            test_id: "001001".to_string(),
            test_description: "This is a test vulnerability. It could be dangerous.".to_string(),
            http_method: "GET".to_string(),
            test_uri: "/admin.php".to_string(),
            test_data: None,
            test_result: "FOUND".to_string(),
        };

        let title = tool.format_finding_title(&record);
        assert_eq!(title, "001001: This is a test vulnerability");
    }

    #[test]
    fn test_parse_text_output() {
        let tool = NiktoTool::new();
        let text_output = r#"
+ /admin/: This might be interesting
+ /backup/: Backup directory found
+ /config.php: Configuration file found
        "#;

        let findings = tool.parse_text_output(text_output, "https://example.com").unwrap();
        assert_eq!(findings.len(), 3);

        let admin_finding = &findings[0];
        assert!(admin_finding.title.contains("/admin/"));
        assert_eq!(admin_finding.description, "This might be interesting");
        assert_eq!(admin_finding.target, "https://example.com/admin/");
    }

    #[tokio::test]
    async fn test_tool_availability() {
        let tool = NiktoTool::new();
        // This test will pass if nikto is installed, otherwise it will be false
        let _available = tool.is_available().await;
        // We don't assert here since nikto might not be installed in test environment
    }
}
