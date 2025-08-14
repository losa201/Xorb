# Variables for NATS Tenant Isolation Infrastructure

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "nats_cluster_name" {
  description = "NATS cluster name"
  type        = string
  default     = "xorb-backplane"
}

variable "enable_monitoring" {
  description = "Enable monitoring and metrics collection"
  type        = bool
  default     = true
}

variable "retention_policy" {
  description = "Global retention policy settings"
  type = object({
    default_retention_days = number
    max_retention_days     = number
    min_retention_days     = number
  })
  default = {
    default_retention_days = 30
    max_retention_days     = 90
    min_retention_days     = 7
  }
}

variable "security_settings" {
  description = "Security configuration settings"
  type = object({
    require_tls           = bool
    verify_certificates   = bool
    min_tls_version      = string
    allowed_cipher_suites = list(string)
  })
  default = {
    require_tls           = true
    verify_certificates   = true
    min_tls_version      = "1.3"
    allowed_cipher_suites = [
      "TLS_AES_256_GCM_SHA384",
      "TLS_CHACHA20_POLY1305_SHA256",
      "TLS_AES_128_GCM_SHA256"
    ]
  }
}

variable "rate_limits" {
  description = "Global rate limiting configuration"
  type = object({
    default_rate_limit_bps = number
    max_rate_limit_bps     = number
    burst_multiplier       = number
  })
  default = {
    default_rate_limit_bps = 1048576  # 1MB/s
    max_rate_limit_bps     = 10485760 # 10MB/s
    burst_multiplier       = 2        # 2x burst capacity
  }
}

variable "backup_settings" {
  description = "Backup and disaster recovery settings"
  type = object({
    enable_backups     = bool
    backup_interval    = string
    backup_retention   = string
    backup_compression = bool
  })
  default = {
    enable_backups     = true
    backup_interval    = "4h"
    backup_retention   = "7d"
    backup_compression = true
  }
}

# Local calculated values
locals {
  common_tags = {
    Environment = var.environment
    Platform    = "xorb"
    Component   = "backplane"
    Phase       = "g2"
    ManagedBy   = "terraform"
  }

  # Calculate tenant quotas based on environment
  tenant_quota_multiplier = {
    dev     = 1.0
    staging = 2.0
    prod    = 5.0
  }

  # Subject validation patterns
  subject_patterns = {
    tenant  = "^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$"
    service = "^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$"
  }

  # Valid domain and event enums (immutable v1 schema)
  valid_domains = ["evidence", "scan", "compliance", "control"]
  valid_events  = ["created", "updated", "completed", "failed", "replay"]
}
