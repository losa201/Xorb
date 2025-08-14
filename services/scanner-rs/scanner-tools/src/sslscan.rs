//! SSLScan integration for SSL/TLS configuration analysis
//!
//! Provides comprehensive SSL/TLS security assessment including
//! cipher analysis, certificate validation, and protocol support.

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

/// SSLScan tool wrapper
#[derive(Debug, Clone)]
pub struct SslScanTool {
    binary_path: String,
}

/// SSLScan XML output structure
#[derive(Debug, Deserialize)]
struct Document {
    #[serde(rename = "ssltest")]
    ssltest: SslTest,
}

#[derive(Debug, Deserialize)]
struct SslTest {
    #[serde(rename = "host")]
    host: String,
    #[serde(rename = "port")]
    port: u16,
    #[serde(rename = "cipher")]
    ciphers: Vec<Cipher>,
    #[serde(rename = "certificate")]
    certificates: Option<Vec<Certificate>>,
    #[serde(rename = "protocol")]
    protocols: Option<Vec<Protocol>>,
}

#[derive(Debug, Deserialize)]
struct Cipher {
    #[serde(rename = "status")]
    status: String,
    #[serde(rename = "sslversion")]
    sslversion: String,
    #[serde(rename = "bits")]
    bits: Option<u16>,
    #[serde(rename = "cipher")]
    cipher: String,
}

#[derive(Debug, Deserialize)]
struct Certificate {
    #[serde(rename = "type")]
    cert_type: String,
    #[serde(rename = "subject")]
    subject: Option<String>,
    #[serde(rename = "issuer")]
    issuer: Option<String>,
    #[serde(rename = "not-valid-before")]
    not_valid_before: Option<String>,
    #[serde(rename = "not-valid-after")]
    not_valid_after: Option<String>,
    #[serde(rename = "signature-algorithm")]
    signature_algorithm: Option<String>,
    #[serde(rename = "pk-algorithm")]
    pk_algorithm: Option<String>,
    #[serde(rename = "pk-bits")]
    pk_bits: Option<u16>,
}

#[derive(Debug, Deserialize)]
struct Protocol {
    #[serde(rename = "type")]
    protocol_type: String,
    #[serde(rename = "version")]
    version: String,
    #[serde(rename = "enabled")]
    enabled: String,
}

impl SslScanTool {
    /// Create new SSLScan tool instance
    pub fn new() -> Self {
        Self {
            binary_path: "sslscan".to_string(),
        }
    }

    /// Create with custom binary path
    pub fn with_binary_path(path: &str) -> Self {
        Self {
            binary_path: path.to_string(),
        }
    }

    /// Build sslscan command arguments
    fn build_args(&self, target: &str, options: &ToolOptions) -> Vec<String> {
        let mut args = vec![
            "--xml=-".to_string(), // XML output to stdout
        ];

        // Parse host and port from target
        let (host, port) = if target.contains(':') {
            let parts: Vec<&str> = target.split(':').collect();
            (parts[0].to_string(), parts.get(1).unwrap_or(&"443").to_string())
        } else {
            // Default HTTPS port
            (target.to_string(), "443".to_string())
        };

        // Scan mode based on options
        if options.aggressive {
            args.extend(vec![
                "--ssl2".to_string(),
                "--ssl3".to_string(),
                "--tls10".to_string(),
                "--tls11".to_string(),
                "--tls12".to_string(),
                "--tls13".to_string(),
                "--show-certificate".to_string(),
                "--check-certs".to_string(),
            ]);
        } else if options.stealth {
            args.extend(vec![
                "--tls12".to_string(),
                "--tls13".to_string(),
            ]);
        } else {
            args.extend(vec![
                "--ssl3".to_string(),
                "--tls10".to_string(),
                "--tls11".to_string(),
                "--tls12".to_string(),
                "--tls13".to_string(),
                "--show-certificate".to_string(),
            ]);
        }

        // Add extra arguments
        args.extend(options.extra_args.clone());

        // Add target with port
        args.push(format!("{}:{}", host, port));

        args
    }

    /// Parse SSLScan XML output
    fn parse_xml_output(&self, output: &str, target: &str) -> Result<Vec<Finding>> {
        let mut findings = Vec::new();

        match from_str::<Document>(output) {
            Ok(document) => {
                let ssltest = document.ssltest;

                // Analyze ciphers for weak configurations
                for cipher in ssltest.ciphers {
                    if cipher.status == "accepted" {
                        let mut metadata = std::collections::HashMap::new();
                        metadata.insert("ssl_version".to_string(), cipher.sslversion.clone());
                        metadata.insert("cipher_name".to_string(), cipher.cipher.clone());

                        if let Some(bits) = cipher.bits {
                            metadata.insert("key_bits".to_string(), bits.to_string());
                        }

                        // Determine severity based on cipher and protocol
                        let (severity, issue_type) = self.assess_cipher_security(&cipher);

                        if severity != Severity::Info {
                            findings.push(Finding {
                                id: Uuid::new_v4().to_string(),
                                title: format!("{} - {}", issue_type, cipher.cipher),
                                description: format!(
                                    "Weak cipher {} is supported on {} using {}",
                                    cipher.cipher, target, cipher.sslversion
                                ),
                                severity,
                                target: target.to_string(),
                                metadata,
                            });
                        }
                    }
                }

                // Analyze certificates
                if let Some(certificates) = ssltest.certificates {
                    for cert in certificates {
                        let mut metadata = std::collections::HashMap::new();
                        metadata.insert("cert_type".to_string(), cert.cert_type.clone());

                        if let Some(subject) = &cert.subject {
                            metadata.insert("subject".to_string(), subject.clone());
                        }
                        if let Some(issuer) = &cert.issuer {
                            metadata.insert("issuer".to_string(), issuer.clone());
                        }
                        if let Some(sig_alg) = &cert.signature_algorithm {
                            metadata.insert("signature_algorithm".to_string(), sig_alg.clone());
                        }
                        if let Some(pk_bits) = cert.pk_bits {
                            metadata.insert("public_key_bits".to_string(), pk_bits.to_string());
                        }

                        // Check for weak certificates
                        let cert_findings = self.assess_certificate_security(&cert, target);
                        for mut finding in cert_findings {
                            finding.metadata.extend(metadata.clone());
                            findings.push(finding);
                        }
                    }
                }

                // Analyze protocols
                if let Some(protocols) = ssltest.protocols {
                    for protocol in protocols {
                        if protocol.enabled == "1" {
                            let severity = self.assess_protocol_security(&protocol);

                            if severity != Severity::Info {
                                let mut metadata = std::collections::HashMap::new();
                                metadata.insert("protocol_type".to_string(), protocol.protocol_type.clone());
                                metadata.insert("protocol_version".to_string(), protocol.version.clone());

                                findings.push(Finding {
                                    id: Uuid::new_v4().to_string(),
                                    title: format!("Insecure Protocol: {} {}", protocol.protocol_type, protocol.version),
                                    description: format!(
                                        "Insecure protocol {} {} is enabled on {}",
                                        protocol.protocol_type, protocol.version, target
                                    ),
                                    severity,
                                    target: target.to_string(),
                                    metadata,
                                });
                            }
                        }
                    }
                }
            }
            Err(e) => {
                warn!(error = %e, "Failed to parse SSLScan XML output, using text parsing");
                findings.extend(self.parse_text_output(output, target)?);
            }
        }

        Ok(findings)
    }

    /// Assess cipher security level
    fn assess_cipher_security(&self, cipher: &Cipher) -> (Severity, &'static str) {
        // Check for weak protocols
        match cipher.sslversion.as_str() {
            "SSLv2" => return (Severity::Critical, "Critical SSL Protocol"),
            "SSLv3" => return (Severity::High, "Deprecated SSL Protocol"),
            "TLSv1.0" => return (Severity::Medium, "Legacy TLS Protocol"),
            "TLSv1.1" => return (Severity::Low, "Legacy TLS Protocol"),
            _ => {}
        }

        // Check for weak ciphers
        let cipher_name = cipher.cipher.to_lowercase();
        if cipher_name.contains("null") || cipher_name.contains("anon") {
            (Severity::Critical, "Null/Anonymous Cipher")
        } else if cipher_name.contains("des") && !cipher_name.contains("3des") {
            (Severity::High, "Weak DES Cipher")
        } else if cipher_name.contains("rc4") {
            (Severity::High, "Weak RC4 Cipher")
        } else if cipher_name.contains("md5") {
            (Severity::Medium, "Weak MD5 Hash")
        } else if let Some(bits) = cipher.bits {
            if bits < 128 {
                (Severity::High, "Weak Key Length")
            } else {
                (Severity::Info, "Accepted Cipher")
            }
        } else {
            (Severity::Info, "Accepted Cipher")
        }
    }

    /// Assess certificate security
    fn assess_certificate_security(&self, cert: &Certificate, target: &str) -> Vec<Finding> {
        let mut findings = Vec::new();

        // Check signature algorithm
        if let Some(sig_alg) = &cert.signature_algorithm {
            if sig_alg.to_lowercase().contains("md5") {
                findings.push(Finding {
                    id: Uuid::new_v4().to_string(),
                    title: "Weak Certificate Signature Algorithm".to_string(),
                    description: format!("Certificate uses weak MD5 signature algorithm: {}", sig_alg),
                    severity: Severity::Medium,
                    target: target.to_string(),
                    metadata: std::collections::HashMap::new(),
                });
            } else if sig_alg.to_lowercase().contains("sha1") {
                findings.push(Finding {
                    id: Uuid::new_v4().to_string(),
                    title: "Legacy Certificate Signature Algorithm".to_string(),
                    description: format!("Certificate uses legacy SHA1 signature algorithm: {}", sig_alg),
                    severity: Severity::Low,
                    target: target.to_string(),
                    metadata: std::collections::HashMap::new(),
                });
            }
        }

        // Check public key strength
        if let Some(pk_bits) = cert.pk_bits {
            if pk_bits < 2048 {
                findings.push(Finding {
                    id: Uuid::new_v4().to_string(),
                    title: "Weak Certificate Key Length".to_string(),
                    description: format!("Certificate uses weak {}-bit public key", pk_bits),
                    severity: if pk_bits < 1024 { Severity::High } else { Severity::Medium },
                    target: target.to_string(),
                    metadata: std::collections::HashMap::new(),
                });
            }
        }

        findings
    }

    /// Assess protocol security
    fn assess_protocol_security(&self, protocol: &Protocol) -> Severity {
        match protocol.version.as_str() {
            "2.0" if protocol.protocol_type == "SSLv2" => Severity::Critical,
            "3.0" if protocol.protocol_type == "SSLv3" => Severity::High,
            "1.0" if protocol.protocol_type == "TLSv1" => Severity::Medium,
            "1.1" if protocol.protocol_type == "TLSv1" => Severity::Low,
            _ => Severity::Info,
        }
    }

    /// Fallback text parsing for SSLScan output
    fn parse_text_output(&self, output: &str, target: &str) -> Result<Vec<Finding>> {
        let mut findings = Vec::new();

        for line in output.lines() {
            if line.contains("Accepted") && (line.contains("SSLv2") || line.contains("SSLv3")) {
                findings.push(Finding {
                    id: Uuid::new_v4().to_string(),
                    title: "Deprecated SSL Protocol".to_string(),
                    description: format!("Deprecated SSL protocol detected: {}", line.trim()),
                    severity: if line.contains("SSLv2") { Severity::Critical } else { Severity::High },
                    target: target.to_string(),
                    metadata: std::collections::HashMap::new(),
                });
            }
        }

        Ok(findings)
    }
}

impl Default for SslScanTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SecurityTool for SslScanTool {
    fn name(&self) -> &str {
        "sslscan"
    }

    #[instrument(skip(self, options), fields(tool = "sslscan", target = %target))]
    async fn scan(&self, target: &str, options: &ToolOptions) -> Result<ToolResult> {
        let start_time = chrono::Utc::now();
        let start_instant = std::time::Instant::now();

        info!(target = %target, "Starting SSLScan analysis");

        // Build command arguments
        let args = self.build_args(target, options);
        let command_line = format!("{} {}", self.binary_path, args.join(" "));

        // Execute sslscan with timeout
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
                    let findings = self.parse_xml_output(&stdout, target)
                        .unwrap_or_else(|e| {
                            warn!(error = %e, "Failed to parse SSLScan output, returning empty findings");
                            Vec::new()
                        });

                    info!(
                        target = %target,
                        duration_ms = %execution_time.as_millis(),
                        findings_count = %findings.len(),
                        "SSLScan analysis completed successfully"
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
                        format!("SSLScan exited with code: {:?}", output.status.code())
                    } else {
                        stderr.to_string()
                    };

                    error!(
                        target = %target,
                        error = %error_msg,
                        "SSLScan analysis failed"
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
                error!(target = %target, error = %e, "Failed to execute SSLScan");
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
                warn!(target = %target, timeout_ms = %options.timeout.as_millis(), "SSLScan analysis timed out");
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
            .map_err(|e| anyhow!("Failed to get SSLScan version: {}", e))?;

        if output.status.success() {
            let version_output = String::from_utf8_lossy(&output.stdout);
            // Parse version from output
            for line in version_output.lines() {
                if line.contains("sslscan") {
                    return Ok(line.trim().to_string());
                }
            }
            Ok("sslscan (unknown version)".to_string())
        } else {
            Err(anyhow!("Failed to get SSLScan version"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToolOptions, OutputFormat};

    #[test]
    fn test_sslscan_tool_creation() {
        let tool = SslScanTool::new();
        assert_eq!(tool.name(), "sslscan");
        assert_eq!(tool.binary_path, "sslscan");

        let tool_custom = SslScanTool::with_binary_path("/usr/local/bin/sslscan");
        assert_eq!(tool_custom.binary_path, "/usr/local/bin/sslscan");
    }

    #[test]
    fn test_build_args() {
        let tool = SslScanTool::new();
        let options = ToolOptions {
            aggressive: true,
            extra_args: vec!["--bugs".to_string()],
            ..Default::default()
        };

        let args = tool.build_args("example.com:443", &options);
        assert!(args.contains(&"--xml=-".to_string()));
        assert!(args.contains(&"--ssl2".to_string()));
        assert!(args.contains(&"--show-certificate".to_string()));
        assert!(args.contains(&"--bugs".to_string()));
        assert!(args.iter().any(|arg| arg.contains("example.com:443")));
    }

    #[test]
    fn test_cipher_security_assessment() {
        let tool = SslScanTool::new();

        let ssl2_cipher = Cipher {
            status: "accepted".to_string(),
            sslversion: "SSLv2".to_string(),
            bits: Some(128),
            cipher: "RC4-MD5".to_string(),
        };
        let (severity, issue_type) = tool.assess_cipher_security(&ssl2_cipher);
        assert_eq!(severity, Severity::Critical);
        assert_eq!(issue_type, "Critical SSL Protocol");

        let weak_cipher = Cipher {
            status: "accepted".to_string(),
            sslversion: "TLSv1.2".to_string(),
            bits: Some(64),
            cipher: "DES-CBC-SHA".to_string(),
        };
        let (severity, _) = tool.assess_cipher_security(&weak_cipher);
        assert_eq!(severity, Severity::High);
    }

    #[test]
    fn test_protocol_security_assessment() {
        let tool = SslScanTool::new();

        let ssl2_protocol = Protocol {
            protocol_type: "SSLv2".to_string(),
            version: "2.0".to_string(),
            enabled: "1".to_string(),
        };
        assert_eq!(tool.assess_protocol_security(&ssl2_protocol), Severity::Critical);

        let tls12_protocol = Protocol {
            protocol_type: "TLSv1".to_string(),
            version: "1.2".to_string(),
            enabled: "1".to_string(),
        };
        assert_eq!(tool.assess_protocol_security(&tls12_protocol), Severity::Info);
    }

    #[tokio::test]
    async fn test_tool_availability() {
        let tool = SslScanTool::new();
        // This test will pass if sslscan is installed, otherwise it will be false
        let _available = tool.is_available().await;
        // We don't assert here since sslscan might not be installed in test environment
    }
}
