# xorbctl Developer CLI

The `xorbctl` CLI provides a unified interface for local development with XORB, enabling discovery job submission, event tailing, and evidence verification.

## Installation

```bash
# Install dependencies
pip install nats-py python-dotenv

# Add to PATH
export PATH=$PATH:/root/Xorb/tools/xorbctl
```

## Usage

### 1. Submit Discovery Job
```bash
# Submit discovery targets for tenant 't-qa'
export NATS_URL=nats://localhost:4222
xorbctl submit --tenant t-qa --targets targets.json
```

### 2. Tail Events
```bash
# Follow evidence events for tenant 't-qa'
xorbctl tail --tenant t-qa --domain evidence --since 5m
```

### 3. Verify Evidence
```bash
# Validate object signature and timestamp
xorbctl verify 
  --object s3://bucket/object 
  --sig $(cat signature.b64) 
  --tsr $(cat timestamp.tsr)
```

## Configuration

Supported environment variables:
- `NATS_URL` - NATS server connection URL
- `NATS_CREDENTIALS` - Path to NATS credentials file
- `XORB_SUBJECT_SCHEMA` - Subject schema version (default: v1)

## Development

```bash
# Run with debug logging
XORB_DEBUG=1 xorbctl tail --tenant t-qa --domain scan --since 1m
```

## Autocomplete

```bash
# Enable shell autocomplete
eval "$(register-python-argcomplete xorbctl)"
```

For more details on subject schema and event formats, see [NATS Subject Schema](../docs/nats-subjects.md).