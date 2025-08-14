# API & Schema Compatibility Gates

This document explains the contract compatibility checking system that prevents breaking changes to public APIs and schemas.

## What Counts as Breaking Changes

### Protobuf Breaking Changes

The following changes to `.proto` files are considered breaking:

1. **Field Removal/Renaming**
   - Removing a field from a message
   - Renaming a field (changes field number semantics)
   - Changing a field's number (tag)

2. **Type Changes**
   - Changing a field's type (e.g., `string` to `int32`)
   - Changing repeated fields to singular or vice versa
   - Changing `optional` to `required`

3. **Required/Optional Flips**
   - Adding `required` fields to existing messages
   - Changing `required` fields to `optional` (can break senders)

4. **Service/RPC Changes**
   - Removing or renaming services
   - Removing or renaming RPC methods
   - Changing RPC method signatures (request/response types)

### OpenAPI Breaking Changes

The following changes to `xorb.openapi.yaml` are considered breaking:

1. **Endpoint Removal**
   - Removing paths/endpoints entirely
   - Removing HTTP methods from existing paths

2. **Response Schema Narrowing**
   - Removing response codes (e.g., removing 404 response)
   - Making response fields required when they were optional

3. **Request Schema Changes**
   - Adding required fields to request bodies
   - Narrowing enum values (removing allowed values)

4. **Authentication/Security Changes**
   - Removing or changing security schemes
   - Making previously optional authentication required

## How to Fix Breaking Changes

### Field Deprecation (Preferred Approach)

Instead of removing fields:
1. Mark fields as deprecated in comments
2. Add new fields for replacement functionality
3. Maintain backward compatibility in service implementations
4. Remove deprecated fields only in major version releases

```protobuf
message ExampleMessage {
  string old_field = 1 [deprecated = true];  // Deprecated as of v2.0
  string new_field = 2;                      // Use this instead
}
```

### Versioning Strategy

1. **Additive-Only Changes**
   - Add new fields as optional
   - Add new endpoints rather than modifying existing ones
   - Add new enum values rather than removing existing ones

2. **Major Version Bumps**
   - For truly breaking changes, increment major version
   - Provide migration guides for clients
   - Maintain both versions during transition period

### Safe Refactoring Patterns

1. **Gradual Migration**
   - Introduce new API alongside old one
   - Redirect old API to new implementation
   - Deprecate and eventually remove old API

2. **Feature Flags**
   - Use feature flags to control new behavior
   - Enable for testing before full rollout
   - Remove flag and old code path together

## How to Run Compatibility Checks Locally

### Prerequisites

Install required tools:
```bash
make install-contract-deps
```

This installs:
- `buf` - Protobuf linter and breaking change detector
- `protoc` - Protocol Buffer compiler
- `pyyaml` - Python YAML parser

### Running Checks

Run all contract compatibility checks:
```bash
make contract-check
```

View the latest compatibility reports:
```bash
make contract-report
```

### Interpreting Results

Reports are generated in:
- `tools/contracts/reports/proto_compat.json` (JSON format)
- `tools/contracts/reports/proto_compat.md` (Human-readable Markdown)
- `tools/contracts/reports/openapi_compat.json` (JSON format)
- `tools/contracts/reports/openapi_compat.md` (Human-readable Markdown)

Each report includes:
- What changed
- Where the change occurred
- Why it's breaking
- Suggested fix

## CI/CD Integration

The compatibility checks run automatically on pull requests that modify:
- `proto/**` files
- `docs/api/xorb-openapi-spec.yaml`
- `tools/contracts/**` files

If breaking changes are detected, the PR will be blocked with a comment summarizing the issues.

## Testing Breaking Changes

To test the compatibility checker, you can simulate a breaking change:

1. Modify a proto file or the OpenAPI spec to introduce a breaking change
2. Run `make contract-check`
3. Observe the breaking change being detected in the report

Example breaking change in OpenAPI:
```yaml
# Before (in main branch)
components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
      required:
        - id

# After (breaking change - adding required field)
components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
        email:  # New required field - BREAKING
          type: string
      required:
        - id
        - email  # BREAKING: New required field
```

This would be flagged as:
| What | Where | Why | Suggested Fix |
|------|-------|-----|---------------|
| Added required field: email | PATH: /users POST | Newly required field breaks existing clients | Make field optional or provide default value |
