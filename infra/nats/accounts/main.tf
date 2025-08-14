# NATS Accounts, Streams, and Consumers IaC

# Provider configuration
terraform {
  required_providers {
    nats = {
      source  = "nats-io/nats"
      version = "0.1.0"
    }
  }
}

provider "nats" {
  # Configuration options
  # server = "nats://localhost:4222"
}

# Example NATS Account resource (commented out)
/*
resource "nats_account" "tenant_account" {
  name = "tenant-${var.tenant_id}"

  # Users within the account
  users = [
    {
      user = "scanner"
      pass = var.scanner_password
    }
  ]

  # Account limits
  limits {
    max_connections = var.max_connections
    max_subscriptions = var.max_subscriptions
    max_payload = var.max_payload
  }

  # JetStream enabled
  jetstream_enabled = true

  # JetStream limits
  jetstream {
    max_memory = var.max_memory
    max_storage = var.max_storage
  }
}
*/

# Example NATS Stream resource (commented out)
/*
resource "nats_stream" "tenant_stream" {
  name = "tenant-${var.tenant_id}-stream"
  subjects = [
    "xorb.${var.tenant_id}.evidence.>",
    "xorb.${var.tenant_id}.scan.>",
    "xorb.${var.tenant_id}.compliance.>",
    "xorb.${var.tenant_id}.control.>"
  ]

  # Stream configuration
  storage = "file"  # or "memory"
  replicas = 3
  max_msgs = 1000000
  max_bytes = 1073741824  # 1GB
  max_age = "720h"  # 30 days
  max_msg_size = 1048576  # 1MB

  # Retention policy
  retention = "limits"  # or "interest" or "workqueue"

  # Discard policy
  discard = "old"

  # Duplicate tracking
  duplicate_window = "2m"
}
*/

# Example NATS Consumer resource (commented out)
/*
resource "nats_consumer" "tenant_consumer" {
  stream_name = nats_stream.tenant_stream.name
  name = "tenant-${var.tenant_id}-consumer"

  # Consumer configuration
  durable_name = "tenant-${var.tenant_id}-durable"
  deliver_subject = "xorb.${var.tenant_id}.deliver.>"
  ack_policy = "explicit"
  ack_wait = "30s"
  max_deliver = 5
  replay_policy = "instant"

  # Rate limiting
  rate_limit = var.rate_limit  # messages per second

  # Sampling for observability
  sample_freq = "100"
}
*/

# Outputs (commented out)
/*
output "account_name" {
  value = nats_account.tenant_account.name
}

output "stream_name" {
  value = nats_stream.tenant_stream.name
}

output "consumer_name" {
  value = nats_consumer.tenant_consumer.name
}
*/
