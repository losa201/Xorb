# XORB Chaos Toolkit

A minimal, deterministic chaos testing framework for the XORB platform.

## Overview

This toolkit provides controlled chaos experiments to validate the resilience and reliability of the XORB platform under various failure conditions.

## Scenarios

1. **NATS Node Kill** (`nats_node_kill.py`) - Simulates NATS node failure and verifies system recovery
2. **Replay Storm** (`replay_storm.py`) - Generates high replay load to test system limits
3. **Corrupted Evidence Injection** (`corrupted_evidence_inject.py`) - Tests evidence verification resilience

## Running Locally

### Prerequisites

- Docker Compose environment
- Python 3.8+

### Commands

```bash
# Run all scenarios in dry-run mode (no actual chaos)
make chaos

# Run specific scenarios
make chaos-nats-kill
make chaos-replay-storm
make chaos-corrupt-evidence

# Run with docker-compose
python3 tools/chaos/run.py --scenario all --compose

# Run with kubernetes (if kubectl available)
python3 tools/chaos/run.py --scenario all --kube
```

## Running in CI

The chaos toolkit is integrated into the CI pipeline via the `post_release_guardrails.yml` workflow. It runs in dry-run mode to validate scenario definitions without causing actual disruptions.

## Scenario Details

### NATS Node Kill

Kills a NATS container/pod and observes:
- `consumer_redelivery_rate` stays within SLO
- `bus_publish_to_deliver_p95_ms` stays under 50ms

### Replay Storm

Starts replay consumers on `replay.*` and asserts:
- Live P95 latency < 50ms
- Backlog depth < threshold

### Corrupted Evidence Injection

Uploads an object with wrong signature and expects:
- `evidence_verify_fail_total` increase
- HTTP 4xx on read

## Development

To add new scenarios:
1. Create a new Python file in `tools/chaos/scenarios/`
2. Implement the scenario logic
3. Add the scenario to the runner in `tools/chaos/run.py`
4. Add a Make target in the root Makefile
