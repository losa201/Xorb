# NATS JetStream Accounts for Tenant Isolation - Phase G2
# Creates per-tenant NATS accounts with strict quotas and stream isolation

terraform {
  required_providers {
    nats = {
      source  = "nats-io/nats"
      version = "~> 0.1.0"
    }
  }
}

variable "nats_url" {
  description = "NATS server URL"
  type        = string
  default     = "nats://localhost:4222"
}

variable "tenants" {
  description = "List of tenant configurations"
  type = map(object({
    name          = string
    display_name  = string
    max_streams   = number
    max_consumers = number
    max_bytes     = number
    max_messages  = number
    retention_days = number
  }))
  default = {
    "t-qa" = {
      name          = "t-qa"
      display_name  = "QA Testing Tenant"
      max_streams   = 50
      max_consumers = 100
      max_bytes     = 1073741824  # 1GB
      max_messages  = 1000000
      retention_days = 30
    }
    "t-demo" = {
      name          = "t-demo"
      display_name  = "Demo Environment Tenant"
      max_streams   = 25
      max_consumers = 50
      max_bytes     = 536870912   # 512MB
      max_messages  = 500000
      retention_days = 7
    }
  }
}

variable "service_user_creds" {
  description = "Path to NATS service user credentials file"
  type        = string
  default     = "/etc/nats/service-user.creds"
  sensitive   = true
}

# Configure NATS provider with service credentials
provider "nats" {
  servers = [var.nats_url]
  # Use credentials file for authentication
  credentials = var.service_user_creds
}

# Create NATS accounts for each tenant
resource "nats_account" "tenant_accounts" {
  for_each = var.tenants

  name         = "xorb-${each.value.name}"
  display_name = each.value.display_name

  # JetStream limits per tenant
  jetstream {
    enabled = true

    # Storage limits
    max_bytes    = each.value.max_bytes
    max_messages = each.value.max_messages
    max_streams  = each.value.max_streams

    # Consumer limits
    max_consumers = each.value.max_consumers

    # Memory limits (20% of max_bytes)
    max_memory_bytes = each.value.max_bytes * 0.2

    # Operational limits
    max_ack_pending = 1024
    duplicate_window = "2m"
  }

  # Connection limits
  max_connections = 100
  max_subscriptions = 1000

  # Data limits
  max_data = each.value.max_bytes * 2  # Allow 2x for operational overhead
  max_payload = 65536  # 64KB max message payload

  # Export permissions for cross-tenant communication (if needed)
  exports = []

  # Import permissions (initially empty - strict isolation)
  imports = []

  tags = {
    tenant      = each.value.name
    environment = "xorb"
    managed_by  = "terraform"
    phase       = "g2"
  }
}

# Create signing keys for each tenant account
resource "nats_nkey" "tenant_signing_keys" {
  for_each = var.tenants

  type = "account"
  name = "xorb-${each.value.name}-signing"
}

# Create users for each tenant account
resource "nats_user" "tenant_users" {
  for_each = var.tenants

  account = nats_account.tenant_accounts[each.key].name
  name    = "xorb-${each.value.name}-user"

  # User-level permissions (more restrictive than account)
  permissions {
    publish {
      allow = [
        "xorb.${each.value.name}.evidence.>",
        "xorb.${each.value.name}.scan.>",
        "xorb.${each.value.name}.compliance.>",
        "xorb.${each.value.name}.control.>"
      ]
      deny = ["xorb.*.admin.>"]
    }

    subscribe {
      allow = [
        "xorb.${each.value.name}.evidence.>",
        "xorb.${each.value.name}.scan.>",
        "xorb.${each.value.name}.compliance.>",
        "xorb.${each.value.name}.control.>"
      ]
      deny = ["xorb.*.admin.>"]
    }
  }

  # Connection limits per user
  max_connections = 20
  max_subscriptions = 200

  tags = {
    tenant = each.value.name
    type   = "service-user"
  }
}

# Create NATS JWT for each user
resource "nats_jwt" "tenant_user_jwts" {
  for_each = var.tenants

  user_nkey    = nats_user.tenant_users[each.key].nkey
  account_nkey = nats_account.tenant_accounts[each.key].nkey
  signing_nkey = nats_nkey.tenant_signing_keys[each.key].nkey

  # Claims
  claims = {
    name = nats_user.tenant_users[each.key].name
    sub  = nats_user.tenant_users[each.key].nkey
    aud  = "xorb-platform"
    iss  = nats_nkey.tenant_signing_keys[each.key].nkey
  }
}

# Outputs for integration with applications
output "tenant_accounts" {
  description = "Created NATS accounts for tenants"
  value = {
    for k, v in nats_account.tenant_accounts : k => {
      name      = v.name
      nkey      = v.nkey
      tenant_id = var.tenants[k].name
    }
  }
}

output "tenant_users" {
  description = "Created NATS users for tenants"
  value = {
    for k, v in nats_user.tenant_users : k => {
      name      = v.name
      nkey      = v.nkey
      tenant_id = var.tenants[k].name
    }
  }
  sensitive = true
}

output "tenant_credentials" {
  description = "NATS credentials for each tenant"
  value = {
    for k, v in nats_jwt.tenant_user_jwts : k => {
      jwt       = v.jwt
      nkey      = nats_user.tenant_users[k].nkey
      tenant_id = var.tenants[k].name
    }
  }
  sensitive = true
}

# Create configuration files for each tenant
resource "local_file" "tenant_configs" {
  for_each = var.tenants

  filename = "${path.module}/out/tenant-${each.value.name}-config.json"
  content = jsonencode({
    tenant_id = each.value.name
    account = {
      name = nats_account.tenant_accounts[each.key].name
      nkey = nats_account.tenant_accounts[each.key].nkey
    }
    user = {
      name = nats_user.tenant_users[each.key].name
      nkey = nats_user.tenant_users[each.key].nkey
    }
    jwt = nats_jwt.tenant_user_jwts[each.key].jwt
    quotas = {
      max_streams   = each.value.max_streams
      max_consumers = each.value.max_consumers
      max_bytes     = each.value.max_bytes
      max_messages  = each.value.max_messages
      retention_days = each.value.retention_days
    }
    subjects = {
      publish_patterns = [
        "xorb.${each.value.name}.evidence.>",
        "xorb.${each.value.name}.scan.>",
        "xorb.${each.value.name}.compliance.>",
        "xorb.${each.value.name}.control.>"
      ]
      subscribe_patterns = [
        "xorb.${each.value.name}.evidence.>",
        "xorb.${each.value.name}.scan.>",
        "xorb.${each.value.name}.compliance.>",
        "xorb.${each.value.name}.control.>"
      ]
    }
  })

  file_permission = "0600"
}
