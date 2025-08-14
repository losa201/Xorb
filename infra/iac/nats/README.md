# NATS Tenant Isolation Infrastructure - Phase G2

This Terraform module creates NATS JetStream accounts, streams, and consumers with strict tenant isolation for the XORB backplane.

## Overview

Implements the Phase G2 tenant isolation architecture with:

- **Per-tenant NATS accounts** with strict quotas and isolation
- **Stream classes** (live, replay) with different retention policies
- **Tuned consumers** with flow control settings
- **Subject schema validation** (v1 immutable)
- **Comprehensive monitoring** and observability

## Architecture

```
NATS JetStream Cluster
├── Account: xorb-t-qa
│   ├── Streams: evidence-live, evidence-replay, scan-live, scan-replay, ...
│   ├── Consumers: tuned with flow control (ack_wait=30s, max_ack_pending=1024)
│   └── Users: xorb-t-qa-user (subject permissions)
├── Account: xorb-t-demo
│   ├── Streams: evidence-live, evidence-replay, scan-live, scan-replay, ...
│   ├── Consumers: tuned with flow control
│   └── Users: xorb-t-demo-user (subject permissions)
└── ...
```

## Subject Schema (v1 IMMUTABLE)

```
Pattern: xorb.<tenant>.<domain>.<service>.<event>

Where:
- tenant: alphanumeric, 3-63 chars, no dots/hyphens at start/end
- domain ∈ {evidence, scan, compliance, control}
- service: alphanumeric with hyphens, 1-32 chars
- event ∈ {created, updated, completed, failed, replay}

Examples:
✅ xorb.t-qa.scan.nmap.created
✅ xorb.tenant-1.evidence.discovery.completed
❌ xorb.t.scan.nmap.started (tenant too short, invalid event)
```

## Usage

### Prerequisites

1. **NATS Server** with JetStream enabled
2. **NATS CLI** for server management
3. **Service credentials** for Terraform provider

```bash
# Install NATS CLI
go install github.com/nats-io/natscli/nats@latest

# Start NATS server with JetStream
nats-server -js -p 4222

# Create service user credentials (one-time setup)
nats account add XORB_SERVICE
nats user add terraform --account XORB_SERVICE
nats user creds terraform > /etc/nats/service-user.creds
```

### Deploy Infrastructure

```bash
# Initialize Terraform
terraform init

# Plan deployment (review resources)
terraform plan -var-file="environments/dev.tfvars"

# Apply infrastructure
terraform apply -var-file="environments/dev.tfvars"

# View outputs
terraform output -json > nats-config.json
```

### Environment Configuration

Create environment-specific variable files:

```hcl
# environments/dev.tfvars
environment = "dev"
nats_url = "nats://localhost:4222"

tenants = {
  "t-qa" = {
    name          = "t-qa"
    display_name  = "QA Testing Tenant"
    max_streams   = 25
    max_consumers = 50
    max_bytes     = 536870912  # 512MB
    max_messages  = 500000
    retention_days = 7
  }
  "t-demo" = {
    name          = "t-demo"
    display_name  = "Demo Environment"
    max_streams   = 10
    max_consumers = 20
    max_bytes     = 268435456  # 256MB
    max_messages  = 250000
    retention_days = 3
  }
}
```

```hcl
# environments/prod.tfvars
environment = "prod"
nats_url = "nats://nats-cluster.internal:4222"

tenants = {
  "customer-1" = {
    name          = "customer-1"
    display_name  = "Customer 1 Production"
    max_streams   = 100
    max_consumers = 200
    max_bytes     = 5368709120  # 5GB
    max_messages  = 5000000
    retention_days = 30
  }
  "customer-2" = {
    name          = "customer-2"
    display_name  = "Customer 2 Production"
    max_streams   = 50
    max_consumers = 100
    max_bytes     = 2684354560  # 2.5GB
    max_messages  = 2500000
    retention_days = 30
  }
}

security_settings = {
  require_tls         = true
  verify_certificates = true
  min_tls_version    = "1.3"
  allowed_cipher_suites = [
    "TLS_AES_256_GCM_SHA384",
    "TLS_CHACHA20_POLY1305_SHA256"
  ]
}
```

## Consumer Configuration

All consumers are created with tuned flow control settings:

```json
{
  "ack_wait": "30s",
  "max_ack_pending": 1024,
  "flow_control": true,
  "idle_heartbeat": "5s",
  "deliver_policy": "last",
  "rate_limit_bps": 1048576
}
```

## Quotas and Limits

### Account-level Quotas
- **Streams**: Per-tenant stream limit
- **Consumers**: Per-tenant consumer limit
- **Storage**: Max bytes per tenant
- **Messages**: Max message count per tenant
- **Connections**: Max concurrent connections

### Stream-level Quotas
- **Live streams**: 512MB, 1M messages, 30d retention
- **Replay streams**: 1GB, 2M messages, 30d retention (WORM-like)
- **Compression**: S2 compression enabled
- **Replication**: 3 replicas for HA

## Tenant Isolation Testing

```bash
# Test 1: Verify tenant A cannot access tenant B subjects
nats pub --creds=tenant-a.creds "xorb.tenant-b.scan.nmap.created" "test"
# Expected: Permission denied

# Test 2: Verify subject schema validation
nats pub --creds=tenant-a.creds "xorb.tenant-a.invalid-domain.service.created" "test"
# Expected: No route (stream doesn't exist)

# Test 3: Verify quota enforcement
# Publish messages until quota exceeded
for i in {1..1000000}; do
  nats pub --creds=tenant-a.creds "xorb.tenant-a.scan.test.created" "data-$i"
done
# Expected: Eventually quota exceeded error
```

## Monitoring and Observability

### Prometheus Metrics

```bash
# Stream metrics
nats_jetstream_stream_messages{stream="xorb-t-qa-scan-live"}
nats_jetstream_stream_bytes{stream="xorb-t-qa-scan-live"}

# Consumer metrics
nats_jetstream_consumer_ack_pending{consumer="xorb-t-qa-scan-live-consumer"}
nats_jetstream_consumer_delivered{consumer="xorb-t-qa-scan-live-consumer"}

# Account metrics
nats_account_connections{account="xorb-t-qa"}
nats_account_subscriptions{account="xorb-t-qa"}
```

### Health Checks

```bash
# Check NATS server health
curl http://nats-server:8222/healthz

# Check JetStream status
nats server info

# Check account status
nats account info xorb-t-qa

# Check stream status
nats stream info xorb-t-qa-scan-live
```

## Disaster Recovery

### Backup Streams

```bash
# Backup all streams for a tenant
nats stream backup xorb-t-qa-scan-live /backups/t-qa/scan-live-$(date +%Y%m%d).tar.gz

# Restore stream from backup
nats stream restore xorb-t-qa-scan-live /backups/t-qa/scan-live-20250814.tar.gz
```

### Account Recovery

```bash
# Export account configuration
nats account info xorb-t-qa --json > t-qa-account-backup.json

# Recreate account from backup (after terraform destroy)
# Use terraform import to restore state
terraform import 'nats_account.tenant_accounts["t-qa"]' xorb-t-qa
```

## Troubleshooting

### Common Issues

1. **Permission Denied**
   ```bash
   # Check user permissions
   nats account info xorb-t-qa
   nats user info xorb-t-qa-user --account xorb-t-qa
   ```

2. **Quota Exceeded**
   ```bash
   # Check current usage
   nats stream info xorb-t-qa-scan-live
   nats account info xorb-t-qa
   ```

3. **Consumer Lag**
   ```bash
   # Check consumer status
   nats consumer info xorb-t-qa-scan-live xorb-t-qa-scan-live-consumer
   ```

4. **Subject Schema Violations**
   ```bash
   # Use the subject linter
   python3 /root/Xorb/tools/backplane/subject_lint.py --paths /path/to/code
   ```

### Debug Mode

```bash
# Enable NATS server debug logging
nats-server -js -D -T

# Enable client debug logging
export NATS_CLIENT_DEBUG=1
nats pub --creds=tenant-a.creds "xorb.tenant-a.scan.test.created" "debug"
```

## Makefile Integration

The following Make targets are available:

```bash
# Lint subject schema compliance
make backplane-lint

# Plan NATS infrastructure changes
make nats-iac-plan

# Apply NATS infrastructure
make nats-iac-apply

# Destroy NATS infrastructure
make nats-iac-destroy

# Run tenant isolation tests
make nats-test-isolation

# Generate tenant credentials
make nats-generate-creds
```

## Security Considerations

1. **Credential Management**: Store NATS credentials securely (e.g., HashiCorp Vault)
2. **TLS Encryption**: Always use TLS in production environments
3. **Certificate Validation**: Enable certificate verification
4. **Audit Logging**: Monitor all account and stream operations
5. **Regular Rotation**: Rotate signing keys and user credentials periodically

## Migration from Redis Streams

If migrating from Redis Streams to NATS:

1. **Parallel Operation**: Run both systems during transition
2. **Message Replay**: Use NATS replay streams for historical data
3. **Consumer Offset**: Map Redis consumer groups to NATS consumer positions
4. **Schema Validation**: Ensure all subjects follow v1 schema before migration

## Related Documentation

- [ADR-002: Two-Tier Bus Architecture](../../docs/architecture/ADR-002-Two-Tier-Bus.md)
- [Subject Linter Tool](../../tools/backplane/subject_lint.py)
- [Backplane CI Workflow](../../.github/workflows/backplane_sanitize.yml)
- [NATS JetStream Documentation](https://docs.nats.io/jetstream)

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review NATS server logs: `nats-server -D -T`
3. Run subject linter: `python3 tools/backplane/subject_lint.py --paths .`
4. Check Terraform state: `terraform show`
