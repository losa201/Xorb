//! NATS JetStream bus integration for exactly-once message semantics
//!
//! Implements the Two-Tier Bus architecture with NATS JetStream
//! for reliable job distribution and result collection.

use anyhow::{Result, anyhow};
use async_nats::jetstream::{self, consumer::PullConsumer, stream::Stream};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{info, warn, error, instrument};
use uuid::Uuid;

/// NATS JetStream bus client
#[derive(Clone)]
pub struct JetStreamBus {
    client: async_nats::Client,
    jetstream: jetstream::Context,
    consumer_group: String,
}

/// Discovery job message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryJob {
    pub job_id: String,
    pub tenant_id: String,
    pub targets: Vec<ScanTarget>,
    pub scan_type: String,
    pub priority: JobPriority,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub deadline: Option<chrono::DateTime<chrono::Utc>>,
}

/// Scan target specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanTarget {
    pub host: String,
    pub ports: Vec<u16>,
    pub scan_profile: String,
}

/// Job priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Discovery result message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryResult {
    pub job_id: String,
    pub tenant_id: String,
    pub asset_id: String,
    pub findings: Vec<ScanFinding>,
    pub fingerprint: Option<AssetFingerprint>,
    pub risk_tags: Vec<RiskTag>,
    pub scan_metadata: ScanMetadata,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Individual scan finding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanFinding {
    pub id: String,
    pub tool: String,
    pub severity: String,
    pub title: String,
    pub description: String,
    pub target: String,
    pub port: Option<u16>,
    pub service: Option<String>,
    pub cvss_score: Option<f64>,
    pub metadata: std::collections::HashMap<String, String>,
}

/// Asset fingerprint data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetFingerprint {
    pub asset_id: String,
    pub os_family: Option<String>,
    pub os_version: Option<String>,
    pub technologies: Vec<String>,
    pub services: Vec<ServiceInfo>,
    pub confidence: f64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Service information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInfo {
    pub port: u16,
    pub protocol: String,
    pub service: String,
    pub version: Option<String>,
    pub banner: Option<String>,
}

/// Risk tag for asset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskTag {
    pub tag: String,
    pub score: f64,
    pub rationale: String,
    pub compliance_frameworks: Vec<String>,
}

/// Scan execution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanMetadata {
    pub scanner_instance: String,
    pub tools_used: Vec<String>,
    pub scan_duration_ms: u64,
    pub scan_started: chrono::DateTime<chrono::Utc>,
    pub scan_completed: chrono::DateTime<chrono::Utc>,
}

impl JetStreamBus {
    /// Create new JetStream bus client
    pub async fn new(nats_url: &str, consumer_group: &str) -> Result<Self> {
        let client = async_nats::connect(nats_url).await
            .map_err(|e| anyhow!("Failed to connect to NATS: {}", e))?;
            
        let jetstream = jetstream::new(client.clone());
        
        let bus = Self {
            client,
            jetstream,
            consumer_group: consumer_group.to_string(),
        };
        
        // Initialize streams
        bus.ensure_streams().await?;
        
        info!(
            nats_url = %nats_url,
            consumer_group = %consumer_group,
            "JetStream bus initialized"
        );
        
        Ok(bus)
    }
    
    /// Ensure required streams exist
    async fn ensure_streams(&self) -> Result<()> {
        let streams = [
            ("discovery-jobs", "tenant.*.discovery.jobs"),
            ("discovery-results", "tenant.*.discovery.results"),
            ("audit-events", "audit.events"),
        ];
        
        for (stream_name, subject) in streams {
            match self.jetstream.get_stream(stream_name).await {
                Ok(_) => {
                    info!(stream = %stream_name, "Stream already exists");
                }
                Err(_) => {
                    info!(stream = %stream_name, subject = %subject, "Creating stream");
                    
                    let stream_config = jetstream::stream::Config {
                        name: stream_name.to_string(),
                        subjects: vec![subject.to_string()],
                        max_age: Duration::from_secs(7 * 24 * 3600), // 7 days
                        storage: jetstream::stream::StorageType::File,
                        ..Default::default()
                    };
                    
                    self.jetstream.create_stream(stream_config).await
                        .map_err(|e| anyhow!("Failed to create stream {}: {}", stream_name, e))?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Subscribe to discovery jobs
    pub async fn subscribe_discovery_jobs(&self, tenant_id: &str) -> Result<PullConsumer> {
        let subject = format!("tenant.{}.discovery.jobs", tenant_id);
        let consumer_name = format!("{}-{}", self.consumer_group, tenant_id);
        
        let consumer_config = jetstream::consumer::pull::Config {
            name: Some(consumer_name.clone()),
            durable_name: Some(consumer_name),
            filter_subject: subject.clone(),
            ack_policy: jetstream::consumer::AckPolicy::Explicit,
            max_deliver: 3,
            ack_wait: Duration::from_secs(300), // 5 minutes
            ..Default::default()
        };
        
        let consumer = self.jetstream
            .create_consumer_on_stream(consumer_config, "discovery-jobs")
            .await
            .map_err(|e| anyhow!("Failed to create consumer for {}: {}", subject, e))?;
            
        info!(
            subject = %subject,
            consumer = %consumer.cached_info().name,
            "Subscribed to discovery jobs"
        );
        
        Ok(consumer)
    }
    
    /// Publish discovery result
    #[instrument(skip(self, result), fields(job_id = %result.job_id, tenant_id = %result.tenant_id))]
    pub async fn publish_discovery_result(&self, result: &DiscoveryResult) -> Result<()> {
        let subject = format!("tenant.{}.discovery.results", result.tenant_id);
        let payload = serde_json::to_vec(result)
            .map_err(|e| anyhow!("Failed to serialize result: {}", e))?;
            
        let publish_ack = self.jetstream
            .publish(subject.clone(), payload.into())
            .await
            .map_err(|e| anyhow!("Failed to publish to {}: {}", subject, e))?;
            
        match publish_ack.await {
            Ok(_) => {
                info!(
                    subject = %subject,
                    job_id = %result.job_id,
                    findings_count = %result.findings.len(),
                    "Discovery result published"
                );
                Ok(())
            }
            Err(e) => {
                error!(
                    subject = %subject,
                    job_id = %result.job_id,
                    error = %e,
                    "Failed to get publish acknowledgment"
                );
                Err(anyhow!("Publish failed: {}", e))
            }
        }
    }
    
    /// Publish audit event
    #[instrument(skip(self, event))]
    pub async fn publish_audit_event(&self, event: &AuditEvent) -> Result<()> {
        let subject = "audit.events";
        let payload = serde_json::to_vec(event)
            .map_err(|e| anyhow!("Failed to serialize audit event: {}", e))?;
            
        let publish_ack = self.jetstream
            .publish(subject, payload.into())
            .await
            .map_err(|e| anyhow!("Failed to publish audit event: {}", e))?;
            
        match publish_ack.await {
            Ok(_) => {
                info!(
                    event_type = %event.event_type,
                    tenant_id = %event.tenant_id,
                    "Audit event published"
                );
                Ok(())
            }
            Err(e) => {
                error!(
                    event_type = %event.event_type,
                    error = %e,
                    "Failed to publish audit event"
                );
                Err(anyhow!("Audit publish failed: {}", e))
            }
        }
    }
}

/// Audit event for security trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub event_id: String,
    pub event_type: String,
    pub tenant_id: String,
    pub user_id: Option<String>,
    pub resource_type: String,
    pub resource_id: String,
    pub action: String,
    pub outcome: String,
    pub details: std::collections::HashMap<String, String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub source_ip: Option<String>,
    pub user_agent: Option<String>,
}

impl AuditEvent {
    /// Create new audit event
    pub fn new(
        event_type: &str,
        tenant_id: &str,
        resource_type: &str,
        resource_id: &str,
        action: &str,
        outcome: &str,
    ) -> Self {
        Self {
            event_id: Uuid::new_v4().to_string(),
            event_type: event_type.to_string(),
            tenant_id: tenant_id.to_string(),
            user_id: None,
            resource_type: resource_type.to_string(),
            resource_id: resource_id.to_string(),
            action: action.to_string(),
            outcome: outcome.to_string(),
            details: std::collections::HashMap::new(),
            timestamp: chrono::Utc::now(),
            source_ip: None,
            user_agent: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_discovery_job_serialization() {
        let job = DiscoveryJob {
            job_id: "job-001".to_string(),
            tenant_id: "tenant-001".to_string(),
            targets: vec![ScanTarget {
                host: "192.168.1.1".to_string(),
                ports: vec![22, 80, 443],
                scan_profile: "comprehensive".to_string(),
            }],
            scan_type: "network".to_string(),
            priority: JobPriority::Medium,
            created_at: chrono::Utc::now(),
            deadline: None,
        };
        
        let json = serde_json::to_string(&job).expect("Serialization failed");
        let deserialized: DiscoveryJob = serde_json::from_str(&json).expect("Deserialization failed");
        
        assert_eq!(job.job_id, deserialized.job_id);
        assert_eq!(job.tenant_id, deserialized.tenant_id);
    }
    
    #[test]
    fn test_audit_event_creation() {
        let event = AuditEvent::new(
            "scan_initiated",
            "tenant-001",
            "discovery_job",
            "job-001",
            "create",
            "success"
        );
        
        assert_eq!(event.event_type, "scan_initiated");
        assert_eq!(event.tenant_id, "tenant-001");
        assert_eq!(event.resource_type, "discovery_job");
        assert_eq!(event.action, "create");
        assert_eq!(event.outcome, "success");
        assert!(!event.event_id.is_empty());
    }
}