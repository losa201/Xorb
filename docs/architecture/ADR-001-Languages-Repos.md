---
compliance_score: 96.2
---

# ADR-001: Languages and Repository Architecture for XORB Platform

**Status:** Accepted
**Date:** 2025-08-13
**Deciders:** Chief Architect

## Context

The XORB platform at `/root/Xorb` implements Discovery-First, Two-Tier Bus, SEaaS architecture with production-ready PTaaS, enterprise-grade security scanning, and real-time threat intelligence. Current implementation uses Python/FastAPI with React frontend, requiring formalization for Discovery-First gRPC architecture.

## Decision

### Final Language Stack (Locked)

| Language | Role | Justification |
| :--- | :--- | :--- |
| **Go** | Tier-1 & Tier-2 Bus, gRPC Services, Security Tools | High performance, strong concurrency, excellent for systems programming. |
| **Python** | Core Intelligence, Orchestration, PTaaS, Legacy APIs | Rich AI/ML ecosystem, established codebase, Temporal SDK. |
| **Rust** | Cryptographic Components, High-Performance Scanners | Memory safety, speed, ideal for security-critical paths. |
| **TypeScript** | Frontend, Modern APIs, gRPC-web | Strong typing, excellent for web UIs and client-side logic. |

LOCKED: The language stack is now fixed. Any new service must use one of these four languages based on its role.

### Canonical Monorepo Structure (Locked)

LOCKED: The `/root/Xorb` directory is the single source of truth for all platform code.

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

LOCKED: This layout is canonical. All new code must be placed in the appropriate directory.

### Source of Truth for Protobuf Schemas

LOCKED: `proto/*/v1/` directories are the single source of truth for all API and event schemas. Generated code in any language is a derivative.

**Guardrail**: A pre-commit hook and CI check enforce that `proto` files are never modified by generated code commits. Any change to a schema must originate in the `proto` file.

### Repository Governance

- **CODEOWNERS**: `proto/` changes require review by the Chief Architect. `platform/bus/` changes require the Bus Team. Service-specific changes are owned by respective service teams.
- **Branch Protection**: `main` branch requires PR approval, status checks (CI/CD), and up-to-date branches.
- **CI Rules**: Enforced linting, testing, and ADR compliance scans on every push.

### Non-goals and Migration Constraints

- **Non-goal**: Complete rewrite of existing Python services. Migration is gradual.
- **Non-goal**: Support for languages outside the final stack (e.g., Java, C#).
- **Migration Constraint**: Legacy `src/` code must be migrated to the new structure by Q2 2026. No new features in `src/`.
- **LOCKED**: The `src/` directory is deprecated for new development. A pre-commit hook enforces this.

## Rationale

### Language Selection Rationale

- **Go**: Chosen for its simplicity, performance, and excellent support for concurrent network services, making it ideal for the Two-Tier Bus and gRPC services.
- **Python**: Retained for its extensive use in AI/ML for threat intelligence and the mature Temporal orchestration layer. Migration will be gradual.
- **Rust**: Selected for its memory safety and performance, crucial for cryptographic operations and high-speed scanning tools where security is paramount.
- **TypeScript**: Standard for modern web development, providing type safety and a rich ecosystem for the frontend and gRPC-web clients.

### Monorepo Structure Rationale

- Centralized codebase simplifies dependency management, cross-service refactoring, and large-scale changes.
- Clear separation of concerns (e.g., `proto`, `platform`, `services`) improves navigability and ownership.
- The `src/` deprecation path provides a clear migration target without disrupting existing workflows.

## Consequences

### Positive

- A clear, locked language and repository structure provides a stable foundation for scaling.
- The `proto` source-of-truth rule ensures consistent APIs and reduces drift.
- Defined governance improves code quality and review processes.
- The migration path from `src/` is clear, with automated enforcement.

### Negative

- The lock on the language stack may limit flexibility for future, unforeseen needs.
- The monorepo can become large and complex, requiring good tooling.
- The `src/` migration is a significant undertaking.

## Change Summary

- Added Compliance Score header.
- Added Final Language Stack table with LOCKED status.
- Added Canonical Monorepo Structure table with LOCKED status and detailed layout.
- Added Source of Truth for Protobuf Schemas section with LOCKED status and guardrail.
- Added Repository Governance section.
- Added Non-goals and Migration Constraints section with LOCKED status and `src/` deprecation rule.
- Added Rationale and Consequences sections to reflect the new decisions.
