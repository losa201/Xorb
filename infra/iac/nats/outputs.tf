# Outputs for NATS Tenant Isolation Infrastructure

output "accounts_summary" {
  description = "Summary of created NATS accounts"
  value = {
    total_accounts = length(var.tenants)
    accounts = {
      for k, account in nats_account.tenant_accounts : k => {
        name         = account.name
        tenant_id    = var.tenants[k].name
        max_streams  = var.tenants[k].max_streams
        max_consumers = var.tenants[k].max_consumers
        max_bytes    = var.tenants[k].max_bytes
        max_messages = var.tenants[k].max_messages
      }
    }
  }
}

output "streams_summary" {
  description = "Summary of created NATS streams"
  value = {
    total_streams = length(nats_jetstream_stream.tenant_streams)
    streams_by_tenant = {
      for tenant_key, tenant in var.tenants : tenant.name => [
        for k, stream in nats_jetstream_stream.tenant_streams : {
          name     = stream.name
          domain   = stream.tags.domain
          class    = stream.tags.stream_class
          subjects = stream.subjects
        } if stream.tags.tenant == tenant.name
      ]
    }
  }
}

output "consumer_config" {
  description = "Consumer configuration with tuned flow control settings"
  value = {
    default_settings = {
      ack_wait         = "30s"
      max_ack_pending  = 1024
      flow_control     = true
      idle_heartbeat   = "5s"
      deliver_policy   = "last"
      rate_limit_bps   = var.rate_limits.default_rate_limit_bps
    }
    consumers = length(nats_jetstream_consumer.tenant_consumers)
  }
}

output "subject_schema" {
  description = "Subject schema information (v1 immutable)"
  value = {
    version = "v1"
    immutable = true
    pattern = "xorb.<tenant>.<domain>.<service>.<event>"
    domains = local.valid_domains
    events  = local.valid_events
    tenant_rules = "alphanumeric, 3-63 chars, no dots/hyphens at start/end"
    service_rules = "alphanumeric with hyphens, 1-32 chars"
  }
}

output "monitoring_endpoints" {
  description = "Monitoring and observability endpoints"
  value = {
    prometheus_metrics = "/metrics"
    health_check = "/healthz"
    stream_info = "/jsz"
    account_info = "/accountz"
    monitoring_config_file = local_file.stream_monitoring_config.filename
  }
}

output "security_configuration" {
  description = "Security configuration summary"
  value = {
    tls_required = var.security_settings.require_tls
    min_tls_version = var.security_settings.min_tls_version
    certificate_verification = var.security_settings.verify_certificates
    tenant_isolation = "strict"
    subject_validation = "enforced"
  }
}

output "quotas_and_limits" {
  description = "Configured quotas and limits"
  value = {
    tenant_quotas = {
      for k, tenant in var.tenants : tenant.name => {
        max_streams   = tenant.max_streams
        max_consumers = tenant.max_consumers
        max_bytes     = tenant.max_bytes
        max_messages  = tenant.max_messages
        retention_days = tenant.retention_days
      }
    }
    global_limits = {
      rate_limit_bps = var.rate_limits.default_rate_limit_bps
      max_rate_limit_bps = var.rate_limits.max_rate_limit_bps
      max_payload = 65536  # 64KB
      duplicate_window = "2m"
    }
  }
}

output "terraform_state_info" {
  description = "Terraform state and management information"
  value = {
    resources_created = {
      accounts = length(nats_account.tenant_accounts)
      users = length(nats_user.tenant_users)
      streams = length(nats_jetstream_stream.tenant_streams)
      consumers = length(nats_jetstream_consumer.tenant_consumers)
      templates = length(nats_jetstream_stream_template.tenant_stream_templates)
    }
    config_files = {
      tenant_configs = [
        for k, tenant in var.tenants :
        "${path.module}/out/tenant-${tenant.name}-config.json"
      ]
      monitoring_config = local_file.stream_monitoring_config.filename
    }
    environment = var.environment
    cluster_name = var.nats_cluster_name
  }
}

# Sensitive outputs (marked as sensitive)
output "connection_details" {
  description = "Connection details for applications (sensitive)"
  sensitive = true
  value = {
    for k, tenant in var.tenants : tenant.name => {
      nats_url = var.nats_url
      account_nkey = nats_account.tenant_accounts[k].nkey
      user_nkey = nats_user.tenant_users[k].nkey
      user_jwt = nats_jwt.tenant_user_jwts[k].jwt
      config_file = "${path.module}/out/tenant-${tenant.name}-config.json"
    }
  }
}
