# NATS Accounts, Streams, and Consumers IaC

This directory contains the Terraform infrastructure as code for managing NATS accounts, streams, and consumers with tenant isolation.

## Overview

The resources defined here enforce tenant isolation by:
- Creating separate NATS accounts for each tenant
- Configuring JetStream limits per tenant
- Setting up subject-based permissions
- Managing user credentials securely

## Resources (Commented Examples)

The following resources are defined as commented examples in `main.tf`:

1. **NATS Accounts** - Isolated tenant accounts with user management
2. **NATS Streams** - Tenant-specific streams with retention policies
3. **NATS Consumers** - Durable consumers for processing messages

## Variables

See `variables.tf` for configurable parameters including:
- Tenant quotas (connections, memory, storage)
- Tier-based limits (Starter, Pro, Enterprise)
- Security settings (passwords, rate limits)

## Usage

```bash
# Initialize Terraform
terraform init

# Plan the changes
terraform plan -var-file=examples/tenant_demo.tfvars

# Apply the changes
terraform apply -var-file=examples/tenant_demo.tfvars
```

## Tier-Based Quotas

Different service tiers have different resource quotas:
- **Starter**: Limited connections and storage
- **Pro**: Moderate resource allocation
- **Enterprise**: High-capacity configuration

See `examples/tenant_demo.tfvars` for sample configurations.
