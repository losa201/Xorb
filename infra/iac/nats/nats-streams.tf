# NATS JetStream Streams for Tenant Isolation - Phase G2
# Creates stream classes with per-tenant isolation and quotas

# Stream class variables
variable "stream_classes" {
  description = "Stream class configurations"
  type = map(object({
    name           = string
    description    = string
    max_age        = string
    max_bytes      = number
    max_messages   = number
    storage        = string
    replicas       = number
    retention      = string
    discard        = string
  }))
  default = {
    "live" = {
      name         = "live"
      description  = "Live operational streams"
      max_age      = "30d"     # 30 days retention
      max_bytes    = 536870912 # 512MB per stream
      max_messages = 1000000
      storage      = "file"    # Durable file storage
      replicas     = 3         # High availability
      retention    = "limits"  # Retention by limits
      discard      = "old"     # Discard old messages when full
    }
    "replay" = {
      name         = "replay"
      description  = "Audit replay streams with WORM characteristics"
      max_age      = "30d"     # 30 days mandatory retention
      max_bytes    = 1073741824 # 1GB per stream
      max_messages = 2000000
      storage      = "file"    # Durable file storage
      replicas     = 3         # High availability
      retention    = "limits"  # Retention by limits (WORM-like)
      discard      = "old"     # Never discard until retention expires
    }
  }
}

# Create streams for each tenant and domain combination
resource "nats_jetstream_stream" "tenant_streams" {
  for_each = {
    for combo in flatten([
      for tenant_key, tenant in var.tenants : [
        for domain in ["evidence", "scan", "compliance", "control"] : [
          for class_key, class in var.stream_classes : {
            key         = "${tenant_key}-${domain}-${class_key}"
            tenant_key  = tenant_key
            tenant      = tenant
            domain      = domain
            class_key   = class_key
            class       = class
          }
        ]
      ]
    ]) : combo.key => combo
  }

  # Stream configuration
  name        = "xorb-${each.value.tenant.name}-${each.value.domain}-${each.value.class.name}"
  description = "${each.value.class.description} for tenant ${each.value.tenant.name} domain ${each.value.domain}"

  # Subject pattern for this tenant/domain combination
  subjects = [
    "xorb.${each.value.tenant.name}.${each.value.domain}.>"
  ]

  # Stream limits (scaled by tenant quotas)
  max_age      = each.value.class.max_age
  max_bytes    = min(each.value.class.max_bytes, each.value.tenant.max_bytes / 4) # Divide tenant quota by 4 domains
  max_messages = min(each.value.class.max_messages, each.value.tenant.max_messages / 4)

  # Storage configuration
  storage   = each.value.class.storage
  replicas  = each.value.class.replicas
  retention = each.value.class.retention
  discard   = each.value.class.discard

  # Duplicate detection window
  duplicate_window = "2m"

  # Compression (for efficiency)
  compression = "s2"

  # Ensure stream is tied to the tenant account
  account = nats_account.tenant_accounts[each.value.tenant_key].name

  tags = {
    tenant       = each.value.tenant.name
    domain       = each.value.domain
    stream_class = each.value.class.name
    managed_by   = "terraform"
    phase        = "g2"
  }
}

# Create consumers for each stream with tuned flow control
resource "nats_jetstream_consumer" "tenant_consumers" {
  for_each = nats_jetstream_stream.tenant_streams

  stream_name = each.value.name
  name        = "${each.value.name}-consumer"
  description = "Default consumer for ${each.value.name} with tuned flow control"

  # Consumer configuration with flow control tuning
  durable             = true
  deliver_policy      = "last"        # Start from last message
  ack_policy          = "explicit"    # Require explicit acks
  ack_wait           = "30s"          # 30 second ack timeout
  max_deliver        = 3              # Retry up to 3 times
  max_ack_pending    = 1024           # Max 1024 unacked messages

  # Flow control settings
  flow_control     = true             # Enable flow control
  idle_heartbeat   = "5s"             # 5 second heartbeat

  # Rate limiting
  rate_limit_bps = 1048576            # 1MB/s rate limit per consumer

  # Consumer-level filtering (inherit from stream subjects)
  filter_subjects = each.value.subjects

  # Replay policy for audit streams
  replay_policy = lookup(each.value.tags, "stream_class", "") == "replay" ? "instant" : "original"

  # Account association
  account = each.value.account

  tags = merge(each.value.tags, {
    consumer_type = "default"
    tuned        = "true"
  })
}

# Create stream templates for dynamic stream creation
resource "nats_jetstream_stream_template" "tenant_stream_templates" {
  for_each = {
    for combo in flatten([
      for tenant_key, tenant in var.tenants : [
        for class_key, class in var.stream_classes : {
          key        = "${tenant_key}-${class_key}"
          tenant_key = tenant_key
          tenant     = tenant
          class_key  = class_key
          class      = class
        }
      ]
    ]) : combo.key => combo
  }

  name        = "xorb-${each.value.tenant.name}-${each.value.class.name}-template"
  description = "Stream template for ${each.value.class.description} in tenant ${each.value.tenant.name}"

  # Template configuration
  max_streams = 10  # Max 10 streams per template

  # Stream configuration template
  config {
    subjects = ["xorb.${each.value.tenant.name}.{{.domain}}.>"]

    max_age      = each.value.class.max_age
    max_bytes    = each.value.class.max_bytes
    max_messages = each.value.class.max_messages
    storage      = each.value.class.storage
    replicas     = each.value.class.replicas
    retention    = each.value.class.retention
    discard      = each.value.class.discard

    duplicate_window = "2m"
    compression      = "s2"
  }

  # Account association
  account = nats_account.tenant_accounts[each.value.tenant_key].name

  tags = {
    tenant       = each.value.tenant.name
    stream_class = each.value.class.name
    type         = "template"
    managed_by   = "terraform"
  }
}

# Outputs for stream information
output "tenant_streams" {
  description = "Created NATS JetStream streams"
  value = {
    for k, v in nats_jetstream_stream.tenant_streams : k => {
      name      = v.name
      subjects  = v.subjects
      tenant_id = v.tags.tenant
      domain    = v.tags.domain
      class     = v.tags.stream_class
    }
  }
}

output "tenant_consumers" {
  description = "Created NATS JetStream consumers"
  value = {
    for k, v in nats_jetstream_consumer.tenant_consumers : k => {
      name           = v.name
      stream_name    = v.stream_name
      deliver_policy = v.deliver_policy
      flow_control   = v.flow_control
    }
  }
}

output "stream_templates" {
  description = "Created NATS JetStream stream templates"
  value = {
    for k, v in nats_jetstream_stream_template.tenant_stream_templates : k => {
      name      = v.name
      tenant_id = v.tags.tenant
      class     = v.tags.stream_class
    }
  }
}

# Create monitoring configuration for streams
resource "local_file" "stream_monitoring_config" {
  filename = "${path.module}/out/stream-monitoring.json"
  content = jsonencode({
    streams = {
      for k, v in nats_jetstream_stream.tenant_streams : k => {
        name           = v.name
        tenant_id      = v.tags.tenant
        domain         = v.tags.domain
        stream_class   = v.tags.stream_class
        subjects       = v.subjects
        max_bytes      = v.max_bytes
        max_messages   = v.max_messages
        consumer_name  = nats_jetstream_consumer.tenant_consumers[k].name
      }
    }
    quotas = var.tenants
    classes = var.stream_classes
  })

  file_permission = "0644"
}
