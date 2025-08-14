# XORB Repository Index

This document provides a mapping from the current repository structure to the canonical structure defined in ADR-001.

## ADR-001 Canonical Structure Rules

According to ADR-001, the canonical monorepo structure is:

```
/root/Xorb/
├── docs/                    # Documentation, ADRs, compliance
│   ├── architecture/       # Architecture Decision Records
│   └── compliance/         # Compliance reports and procedures
├── proto/                  # Protobuf definitions (Source of Truth)
│   ├── audit/             # Audit event schemas
│   ├── compliance/        # Compliance evidence schemas
│   ├── discovery/         # Discovery job and fingerprint schemas
│   ├── threat/            # Threat intelligence schemas
│   └── vuln/              # Vulnerability schemas
├── platform/              # Core platform infrastructure
│   ├── bus/               # Two-Tier Bus implementation (Go)
│   ├── auth/              # Authentication services (Go/Python)
│   └── evidence/          # Evidence collection and storage (Go)
├── services/              # Business logic microservices
│   ├── discovery/         # Discovery orchestration and scanning (Python)
│   ├── intelligence/      # Threat correlation and analysis (Python/Rust)
│   ├── ptaas/             # Penetration Testing as a Service (Python)
│   └── ui/                # User interface (TypeScript/React)
├── src/                   # Legacy Python code (migration target)
├── tests/                 # Comprehensive test suite
├── tools/                 # Operational and development tooling
│   ├── adr/               # ADR compliance tooling
│   └── secrets/           # Secret management and remediation
├── infra/                 # Infrastructure as Code
│   ├── kubernetes/        # K8s manifests
│   ├── monitoring/        # Observability stack
│   └── vault/             # HashiCorp Vault policies and configs
├── .github/               # GitHub workflows and configurations
├── Makefile               # Build and operational targets
└── README.md              # Project overview and entry point
```

## Current to Canonical Path Mapping

| Original Path | Canonical Path | Status |
|---------------|----------------|--------|
| README.md | README.md | ✅ Correct |
| docs/ | docs/ | ✅ Correct |
| .github/ | .github/ | ✅ Correct |
| compose/ | infra/ | ⚠️ Needs migration |
| config/ | infra/ | ⚠️ Needs migration |
| deploy/ | infra/ | ⚠️ Needs migration |
| infra/ | infra/ | ✅ Correct |
| agents/ | services/ | ⚠️ Needs migration |
| services/ | services/ | ✅ Partially correct |
| ptaas/ | services/ptaas/ | ⚠️ Needs migration |
| security/ | platform/ | ⚠️ Needs migration |
| src/ | src/ | ✅ Correct (deprecated) |
| tests/ | tests/ | ✅ Correct |
| tools/ | tools/ | ✅ Correct |
| scripts/ | tools/ | ⚠️ Needs migration |
| sdks/ | tools/ or separate | ⚠️ Needs review |
| demos/ | examples/ | ⚠️ Needs migration |
| crypto/ | platform/ | ⚠️ Needs migration |
| control_plane_storage/ | platform/ | ⚠️ Needs migration |
| xorb_platform_bus/ | platform/bus/ | ⚠️ Needs migration |

## Notes

1. **LOCKED STATUS**: The canonical structure is locked as per ADR-001. All new development must follow this structure.
2. **Migration Path**: Legacy directories (marked with ⚠️) need to be migrated to their canonical locations.
3. **Source of Truth**: The `proto/` directory should contain all API and event schemas as Protobuf files.
4. **Deprecation**: The `src/` directory is deprecated for new development and should only be used for migration targets.

## Compliance Status

- Language Stack: ✅ Locked (Go, Python, Rust, TypeScript)
- Repository Structure: ⚠️ Partially compliant (migration in progress)
- Protobuf Source of Truth: ❌ Not implemented yet
- Governance: ✅ CODEOWNERS and branch protection in place