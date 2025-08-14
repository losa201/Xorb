# NATS JetStream Streams for Tenant Isolation - Phase G4
# Creates stream classes with per-tenant isolation, quotas, and replay-safe streaming

# Stream class variables with Phase G4 replay-safe streaming
variable "stream_classes" {
  description = "Stream class configurations with dedicated replay lanes"
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
    priority       = number  # I/O priority (1=high, 10=low)
    concurrency_cap = number # Max concurrent consumers
    rate_limit_bps  = number # Rate limit for replay streams
  }))
  default = {
    "live" = {
      name            = "live"
      description     = "Live operational streams - high priority"
      max_age         = "30d"     # 30 days retention
      max_bytes       = 536870912 # 512MB per stream
      max_messages    = 1000000
      storage         = "file"    # Durable file storage
      replicas        = 3         # High availability
      retention       = "limits"  # Retention by limits
      discard         = "old"     # Discard old messages when full
      priority        = 1         # High I/O priority
      concurrency_cap = 50        # Max 50 concurrent consumers
      rate_limit_bps  = 10485760  # 10MB/s for live streams
    }
    "replay" = {
      name            = "replay"
      description     = "Dedicated replay lanes with bounded windows and lower I/O priority"
      max_age         = "90d"     # Extended retention for replay
      max_bytes       = 2147483648 # 2GB per replay stream
      max_messages    = 5000000
      storage         = "file"    # Durable file storage
      replicas        = 3         # High availability
      retention       = "limits"  # Retention by limits (WORM-like)
      discard         = "old"     # Discard old when full
      priority        = 5         # Lower I/O priority
      concurrency_cap = 10        # Limited concurrent consumers for replay
      rate_limit_bps  = 2097152   # 2MB/s rate limit for replay
    }
  }
}

# Replay policy configuration
variable "replay_policy" {
  description = "Replay streaming policy configuration"
  type = object({
    time_window_hours     = number  # Bounded replay window
    global_rate_limit_bps = number  # Global rate limit for all replay workers
    max_replay_workers    = number  # Maximum concurrent replay workers
    storage_isolation     = bool    # Enable storage I/O isolation
    start_time_policy     = string  # DeliverPolicy for replay (ByStartTime)
  })
  default = {
    time_window_hours     = 168     # 7 days bounded window
    global_rate_limit_bps = 5242880 # 5MB/s global replay rate limit
    max_replay_workers    = 5       # Max 5 replay workers globally
    storage_isolation     = true    # Enable storage isolation
    start_time_policy     = "ByStartTime"  # Time-bounded replay
  }
}

# Create streams for each tenant and domain combination with replay lanes
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

  # Subject pattern for this tenant/domain combination with replay suffix
  subjects = each.value.class.name == "replay" ? [
    "xorb.${each.value.tenant.name}.${each.value.domain}.*.replay"
  ] : [
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
    tenant          = each.value.tenant.name
    domain          = each.value.domain
    stream_class    = each.value.class.name
    managed_by      = "terraform"
    phase           = "g4"
    priority        = each.value.class.priority
    concurrency_cap = each.value.class.concurrency_cap
    rate_limit_bps  = each.value.class.rate_limit_bps
  }
}

# Create consumers for each stream with tuned flow control and replay-safe settings
resource "nats_jetstream_consumer" "tenant_consumers" {
  for_each = nats_jetstream_stream.tenant_streams

  stream_name = each.value.name
  name        = "${each.value.name}-consumer"
  description = lookup(each.value.tags, "stream_class", "") == "replay" ? 
    "Replay consumer with bounded window and rate limiting" :
    "Live consumer with high-priority flow control"

  # Consumer configuration with stream-class specific tuning
  durable         = true
  deliver_policy  = lookup(each.value.tags, "stream_class", "") == "replay" ? 
    var.replay_policy.start_time_policy : "last"
  ack_policy      = "explicit"    # Require explicit acks
  ack_wait       = lookup(each.value.tags, "stream_class", "") == "replay" ? "60s" : "30s"
  max_deliver    = lookup(each.value.tags, "stream_class", "") == "replay" ? 2 : 3
  max_ack_pending = lookup(each.value.tags, "stream_class", "") == "replay" ? 256 : 1024

  # Flow control settings - more conservative for replay
  flow_control   = true
  idle_heartbeat = lookup(each.value.tags, "stream_class", "") == "replay" ? "10s" : "5s"

  # Rate limiting based on stream class
  rate_limit_bps = lookup(each.value.tags, "rate_limit_bps", 1048576)

  # Consumer-level filtering
  filter_subjects = each.value.subjects

  # Replay policy for different stream types
  replay_policy = lookup(each.value.tags, "stream_class", "") == "replay" ? "instant" : "original"

  # Account association
  account = each.value.account

  tags = merge(each.value.tags, {
    consumer_type   = lookup(each.value.tags, "stream_class", "") == "replay" ? "replay" : "live"
    tuned          = "true"
    concurrency_cap = lookup(each.value.tags, "concurrency_cap", 10)
  })
}

# Create dedicated replay consumer pool with bounded windows
resource "nats_jetstream_consumer" "replay_bounded_consumers" {
  for_each = {
    for k, v in nats_jetstream_stream.tenant_streams : k => v
    if lookup(v.tags, "stream_class", "") == "replay"
  }

  stream_name = each.value.name
  name        = "${each.value.name}-bounded-replay"
  description = "Time-bounded replay consumer with ${var.replay_policy.time_window_hours}h window"

  # Bounded replay configuration
  durable          = true
  deliver_policy   = "ByStartTime"  # Start from specific time
  ack_policy       = "explicit"
  ack_wait        = "120s"         # Longer timeout for replay processing
  max_deliver     = 1              # Single delivery for replay integrity
  max_ack_pending = 128            # Lower pending for bounded replay

  # Conservative flow control for replay
  flow_control   = true
  idle_heartbeat = "30s"           # Less frequent heartbeat

  # Strict rate limiting for replay workers
  rate_limit_bps = var.replay_policy.global_rate_limit_bps / var.replay_policy.max_replay_workers

  # Replay-specific filtering
  filter_subjects = each.value.subjects
  replay_policy   = "instant"      # Instant replay for bounded windows

  # Account association
  account = each.value.account

  tags = merge(each.value.tags, {
    consumer_type    = "bounded_replay"
    tuned           = "true"
    time_bounded    = "true"
    window_hours    = var.replay_policy.time_window_hours
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
    phase        = "g4"
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
      consumer_type  = lookup(v.tags, "consumer_type", "default")
      rate_limit_bps = v.rate_limit_bps
    }
  }
}

output "replay_consumers" {
  description = "Created bounded replay consumers"
  value = {
    for k, v in nats_jetstream_consumer.replay_bounded_consumers : k => {
      name           = v.name
      stream_name    = v.stream_name
      deliver_policy = v.deliver_policy
      window_hours   = lookup(v.tags, "window_hours", 0)
      rate_limit_bps = v.rate_limit_bps
      time_bounded   = lookup(v.tags, "time_bounded", false)
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

# Create monitoring configuration for streams and replay lanes
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
        priority       = v.tags.priority
        concurrency_cap = v.tags.concurrency_cap
        rate_limit_bps = v.tags.rate_limit_bps
        consumer_name  = nats_jetstream_consumer.tenant_consumers[k].name
      }
    }
    replay_consumers = {
      for k, v in nats_jetstream_consumer.replay_bounded_consumers : k => {
        name           = v.name
        stream_name    = v.stream_name
        window_hours   = v.tags.window_hours
        rate_limit_bps = v.rate_limit_bps
        time_bounded   = v.tags.time_bounded
      }
    }
    quotas = var.tenants
    classes = var.stream_classes
    replay_policy = var.replay_policy
    slo_targets = {
      live_p95_ms = 100        # 100ms p95 for live publish→deliver
      replay_success_rate = 0.95  # 95% replay success rate
    }
  })

  file_permission = "0644"
}

# Create replay policy configuration file
resource "local_file" "replay_policy_config" {
  filename = "${path.module}/out/replay-policy.json"
  content = jsonencode({
    time_window_hours     = var.replay_policy.time_window_hours
    global_rate_limit_bps = var.replay_policy.global_rate_limit_bps
    max_replay_workers    = var.replay_policy.max_replay_workers
    storage_isolation     = var.replay_policy.storage_isolation
    start_time_policy     = var.replay_policy.start_time_policy
    concurrency_caps = {
      for class_key, class in var.stream_classes :
      class_key => class.concurrency_cap
    }
  })

  file_permission = "0644"
}
