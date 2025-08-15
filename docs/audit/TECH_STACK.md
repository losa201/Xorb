# XORB Technology Stack Inventory

**Audit Date**: 2025-08-15
**Total Technologies**: 89 distinct technologies identified
**Evidence Sources**: requirements.txt, Cargo.toml, package.json, Dockerfiles

## Languages & Runtimes

| Language | Version | Files | Purpose | Evidence Path |
|----------|---------|-------|---------|---------------|
| **Python** | 3.11-3.12 | 1,247 | Core platform, APIs, orchestration | requirements.txt:18 |
| **TypeScript** | 5.5.3 | 312 | Frontend, PTaaS backend | ptaas/package.json:15 |
| **JavaScript** | ES2022 | 89 | Legacy frontend, Node services | ptaas-backend/package.json |
| **Rust** | 1.75+ | 89 | High-performance scanners | services/scanner-rs/Cargo.toml |
| **Bash/Shell** | 5.0+ | 156 | Automation scripts, deployment | tools/scripts/** |
| **SQL** | PostgreSQL 15 | 23 | Database schemas, migrations | src/api/migrations/** |

## Core Frameworks

### Python Stack
| Framework | Version | Purpose | Evidence |
|-----------|---------|---------|----------|
| **FastAPI** | 0.116.1 | REST API framework | requirements.txt:18 |
| **Uvicorn** | 0.35.0 | ASGI server | requirements.txt:19 |
| **Pydantic** | 2.11.7 | Data validation | requirements.txt:20 |
| **SQLAlchemy** | 2.0.42 | Database ORM | requirements.txt:85 |
| **Alembic** | 1.16.4 | Database migrations | requirements.txt:86 |
| **AsyncPG** | 0.30.0 | PostgreSQL async driver | requirements.txt:87 |
| **Temporal** | 1.15.0 | Workflow orchestration | requirements.txt:105 |
| **Click** | 8.1.7 | CLI framework | requirements.txt:75 |

### Frontend Stack
| Framework | Version | Purpose | Evidence |
|-----------|---------|---------|----------|
| **React** | 18.3.1 | UI framework | ptaas/package.json |
| **Vite** | 5.4.1 | Build tool | ptaas/package.json |
| **TypeScript** | 5.5.3 | Type system | ptaas/package.json |
| **Tailwind CSS** | 3.4.11 | Styling | ptaas/package.json |
| **React Router** | 6.26.2 | Client routing | ptaas/package.json |
| **React Query** | @tanstack/react-query | State management | ptaas/package.json |
| **Radix UI** | Various | UI components | ptaas/package.json |

### Rust Stack
| Crate | Version | Purpose | Evidence |
|-------|---------|---------|----------|
| **tokio** | 1.0 | Async runtime | services/scanner-rs/Cargo.toml |
| **serde** | 1.0 | Serialization | services/scanner-rs/Cargo.toml |
| **clap** | 4.0 | CLI parsing | services/scanner-rs/Cargo.toml |
| **tracing** | 0.1 | Observability | services/scanner-rs/Cargo.toml |
| **sqlx** | 0.7 | Database access | services/scanner-rs/Cargo.toml |

## Messaging & Queueing

| Technology | Version | Purpose | Evidence | ADR Compliance |
|------------|---------|---------|----------|-----------------|
| **NATS JetStream** | 2.10+ | Primary message bus | docker-compose.yml:25 | ✅ ADR-002 |
| **Redis** | 7.0+ | Cache only (no pub/sub) | docker-compose.yml:30 | ⚠️ Some violations |

### NATS Subject Taxonomy
```
xorb.<tenant>.ptaas.job.{queued|running|completed|failed|audit}
xorb.<tenant>.discovery.{started|completed|failed}
xorb.<tenant>.evidence.{collected|verified|stored}
xorb.<tenant>.alerts.{security|performance|compliance}
```

## Data Stores

| Database | Version | Purpose | Evidence | Schema Tables |
|----------|---------|---------|----------|---------------|
| **PostgreSQL** | 15+ | Primary database | docker-compose.yml:35 | 15 tables |
| **SQLite** | 3.40+ | PTaaS job storage | src/orchestrator/config.py:45 | ptaas_jobs.db |
| **Redis** | 7.0+ | Caching, sessions | docker-compose.yml:30 | Cache only |

### Key Database Tables
- `users`, `tenants`, `roles` (Authentication)
- `ptaas_jobs`, `ptaas_sessions`, `ptaas_findings` (PTaaS)
- `vulnerabilities`, `threat_indicators` (Intelligence)
- `audit_logs`, `evidence_chains` (Compliance)

## Container & Runtime

| Technology | Version | Purpose | Evidence |
|------------|---------|---------|----------|
| **Docker** | 24.0+ | Containerization | Dockerfile.unified:5 |
| **Docker Compose** | 2.20+ | Local orchestration | docker-compose.yml |
| **Python Base** | 3.11-slim | Container base | Dockerfile.unified:5 |
| **Alpine Linux** | 3.18+ | Lightweight base | Various Dockerfiles |
| **Kubernetes** | 1.28+ | Production orchestration | infra/k8s/** |

### Container Images
| Service | Base Image | Size | Security |
|---------|------------|------|----------|
| xorb-api | python:3.11-slim | ~200MB | Non-root user |
| xorb-orchestrator | python:3.11-slim | ~180MB | Non-root user |
| scanner-core | rust:1.75-alpine | ~150MB | Non-root user |
| ptaas-frontend | node:18-alpine | ~120MB | Non-root user |

## Observability & Monitoring

| Tool | Version | Purpose | Evidence |
|------|---------|---------|----------|
| **Prometheus** | 2.45+ | Metrics collection | infra/monitoring/prometheus.yml |
| **Grafana** | 10.0+ | Visualization | infra/monitoring/grafana/ |
| **OpenTelemetry** | 1.20+ | Distributed tracing | requirements.txt:95 |
| **Jaeger** | 1.45+ | Trace storage | docker-compose.monitoring.yml |
| **Loki** | 2.9+ | Log aggregation | infra/monitoring/loki/ |

### Metrics Endpoints
- `/metrics` - Prometheus metrics
- `/health` - Health checks
- `/readiness` - Kubernetes readiness
- `/info` - Service information

## Security & Compliance

| Tool | Version | Purpose | Evidence |
|------|---------|---------|----------|
| **Bandit** | 1.7.5 | Python security | .pre-commit-config.yaml:16 |
| **Ruff** | 0.6.5 | Python linting | .pre-commit-config.yaml:10 |
| **Gitleaks** | 8.18.4 | Secret detection | .pre-commit-config.yaml:22 |
| **Semgrep** | 1.45+ | SAST scanning | .github/workflows/security.yml |
| **Trivy** | 0.45+ | Container scanning | tools/security/container-scan.sh |

### Security Configurations
| Component | Configuration | Evidence |
|-----------|---------------|----------|
| TLS/mTLS | cert-manager, Vault | infra/vault/vault-config.hcl |
| RBAC | Kubernetes RBAC | k8s/rbac/** |
| Network Policies | Calico/Cilium | k8s/network-policies/** |
| Secret Management | HashiCorp Vault | infra/vault/** |

## CI/CD & Automation

| Tool | Version | Purpose | Evidence |
|------|---------|---------|----------|
| **GitHub Actions** | Latest | CI/CD pipeline | .github/workflows/** |
| **Pre-commit** | 3.5+ | Code quality hooks | .pre-commit-config.yaml |
| **k6** | 0.45+ | Performance testing | tests/perf/k6/ |
| **Pytest** | 7.4+ | Python testing | requirements.txt:110 |
| **Jest** | 30.0.5 | TypeScript testing | ptaas/package.json |

### CI Workflow Summary
| Workflow | Triggers | Jobs | Duration |
|----------|----------|------|----------|
| ci.yml | Push, PR | 8 jobs | ~15 min |
| security.yml | Push to main | 5 jobs | ~10 min |
| performance.yml | Label trigger | 3 jobs | ~20 min |

## Build & Package Management

| Tool | Version | Purpose | Evidence |
|------|---------|---------|----------|
| **pip** | 24.0+ | Python packages | requirements.txt |
| **npm** | 9.0+ | Node packages | ptaas/package.json |
| **pnpm** | 8.0+ | Fast Node packages | ptaas-backend/package.json |
| **Cargo** | 1.75+ | Rust packages | services/scanner-rs/Cargo.toml |
| **Make** | 4.3+ | Build automation | Makefile, Makefile.perf |

### Package Pinning Status
| Ecosystem | Pinned | Unpinned | Lock File |
|-----------|--------|----------|-----------|
| Python | 111 | 0 | requirements.lock |
| Node.js | 89 | 0 | package-lock.json |
| Rust | 45 | 0 | Cargo.lock |

## Authentication & Authorization

| Technology | Version | Purpose | Evidence |
|------------|---------|---------|----------|
| **JWT** | RS256 | API authentication | src/api/app/auth/ |
| **OAuth 2.0** | Latest | External auth | src/api/app/integrations/ |
| **RBAC** | Custom | Role-based access | src/api/app/rbac/ |
| **mTLS** | TLS 1.3 | Service-to-service | infra/tls/** |

## Networking & Load Balancing

| Technology | Version | Purpose | Evidence |
|------------|---------|---------|----------|
| **Envoy Proxy** | 1.27+ | Load balancing | k8s/gateway/ |
| **Istio** | 1.18+ | Service mesh | k8s/istio/ |
| **Calico** | 3.26+ | Network policies | k8s/network/ |
| **CoreDNS** | 1.10+ | Service discovery | k8s/dns/ |

## External Integrations

| Service | Version | Purpose | Evidence |
|---------|---------|---------|----------|
| **NVIDIA API** | v1 | AI/ML services | src/api/app/integrations/nvidia.py |
| **OpenRouter** | v1 | LLM services | src/api/app/integrations/openrouter.py |
| **GitHub API** | v4 | Source control | src/api/app/integrations/github.py |
| **Slack API** | v1 | Notifications | ptaas-backend/src/services/notifications/ |

## Version Compliance Matrix

| Category | Total | Pinned | Latest | Outdated | Security Risk |
|----------|-------|--------|--------|----------|---------------|
| Python | 111 | 111 (100%) | 95 (86%) | 16 (14%) | 2 (2%) |
| Node.js | 89 | 89 (100%) | 78 (88%) | 11 (12%) | 1 (1%) |
| Rust | 45 | 45 (100%) | 42 (93%) | 3 (7%) | 0 (0%) |
| Container | 12 | 8 (67%) | 10 (83%) | 2 (17%) | 0 (0%) |

## Security Assessment

### Supply Chain Security
- ✅ All dependencies pinned with lock files
- ✅ Container base images using official sources
- ⚠️ Missing SBOM generation in CI
- ⚠️ No signature verification for container images
- ❌ No vulnerability scanning in CI pipeline

### Compliance Status
- ✅ NIST Cybersecurity Framework alignment
- ✅ SOC 2 Type II controls implemented
- ⚠️ PCI DSS compliance gaps in payment handling
- ⚠️ GDPR compliance needs validation

## Recommendations

### Immediate (P01)
1. **Add SBOM generation** with Syft in CI pipeline
2. **Implement container signing** with Cosign
3. **Fix ADR-002 violations** in Redis usage

### Short-term (P02)
1. **Enable vulnerability scanning** with Grype/Trivy
2. **Complete OpenTelemetry** instrumentation
3. **Standardize health endpoints** across services

### Long-term (P03)
1. **Migrate to distroless** container images
2. **Implement chaos engineering** with Litmus
3. **Add performance profiling** with pprof/py-spy

---

*This technology inventory provides a comprehensive view of all technologies, frameworks, and tools used across the XORB platform, with version tracking and security assessment.*
