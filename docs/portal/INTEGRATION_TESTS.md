# Integration Testing

This document describes how to run the integration tests for the XORB platform, which validate the Two-Tier Bus (ADR-002), Authentication (ADR-003), and Evidence Schema (ADR-004).

## Prerequisites

- A local development environment set up for the XORB project.
- `nats-server` installed and available in your `$PATH`. You can download it from [nats.io](https://nats.io/download/).
- Python dependencies installed (including `pytest`, `pytest-asyncio`, `nats-py`, `cryptography`). These should be included in the main `requirements.txt` or a test-specific one.
- Protobuf files compiled. Run `make proto-gen` (or the equivalent command for your project) to generate Python classes from `.proto` files.

## Running the Tests

The integration tests are located in `/tests/integration/` and can be executed using the provided Makefile target:

```bash
make integration-test
```

This command will:
1.  Start a local `nats-server` in JetStream mode.
2.  Execute the `pytest` suite located in `/tests/integration/`.
3.  Automatically manage the lifecycle of the NATS server (start before tests, stop after).
4.  Run tests that:
    *   Use mTLS for client authentication.
    *   Use JWT tokens for authorization and tenant context.
    *   Publish messages to the simulated Tier-1 bus (UDS) which are then routed to Tier-2 (NATS JetStream).
    *   Consume messages from Tier-2.
    *   Produce and validate `Evidence` protobuf messages, ensuring they conform to the schema and are correctly signed.
    *   Assert WORM-like retention policies are configured on NATS streams.
    *   Ensure that Redis is not used in the core bus communication flow.

## Test Structure

- `/tests/integration/conftest.py`: Central configuration and session-scoped fixtures.
- `/tests/integration/test_bus_end_to_end.py`: The main end-to-end test.
- `/tests/integration/fixtures/`:
  - `nats_server.py`: Manages the lifecycle of the local NATS JetStream server.
  - `uds_ring.py`: Provides a placeholder path for the Tier-1 UDS ring.
  - `certs.py`: Generates test mTLS certificates.
  - `jwt.py`: Provides utilities to create test JWT tokens.
- `/tests/integration/proto_helpers.py`: Utilities for creating, serializing, and loading Protobuf messages.

## Output

- Test results are printed to the console by `pytest`.
- A summary report in JSON format is generated at `tools/reports/integration_summary.json`.
