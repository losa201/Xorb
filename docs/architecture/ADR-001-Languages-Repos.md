# ADR-001: Languages and Repository Architecture for XORB Platform

**Status:** Accepted  
**Date:** 2025-08-13  
**Deciders:** Chief Architect  

## Context

The XORB platform at `/root/Xorb` implements Discovery-First, Two-Tier Bus, SEaaS architecture with production-ready PTaaS, enterprise-grade security scanning, and real-time threat intelligence. Current implementation uses Python/FastAPI with React frontend, requiring formalization for Discovery-First gRPC architecture.

## Decision

### Current Language Stack (Maintained)
- **Python 3.11+**: Primary backend (FastAPI 0.117.1, Temporal 1.6.0)
  - `src/api/` - FastAPI REST APIs, PTaaS services
  - `src/orchestrator/` - Temporal workflow orchestration
  - `src/xorb/` - Core platform intelligence engine
  - `ptaas/` - Security scanning and behavioral analytics

- **TypeScript/JavaScript**: Frontend and modern APIs
  - `services/ptaas/web/` - React 18.3.1 + Vite 5.4.1 dashboard
  - Node.js microservices for real-time components

- **Rust**: Security-critical components (future enhancement)
  - Cryptographic operations and high-performance scanning
  - Memory-safe security tools integration

### Repository Structure (Current /root/Xorb)
```
/root/Xorb/
├── src/                    # Core backend services
│   ├── api/               # FastAPI REST APIs → gRPC gateway target
│   ├── orchestrator/      # Temporal workflows → Discovery orchestration
│   ├── xorb/             # Intelligence engine → Discovery correlation
│   └── common/           # Shared utilities, Vault integration
├── services/             # Microservices architecture
│   ├── ptaas/           # PTaaS service → Discovery-driven scanning
│   ├── xorb-core/       # Backend platform → gRPC services
│   └── infrastructure/ # Monitoring, databases → Service mesh
├── infra/               # Infrastructure as code
│   ├── kubernetes/      # K8s manifests for Two-Tier Bus
│   ├── monitoring/      # Prometheus/Grafana observability
│   └── vault/          # HashiCorp Vault secret management
├── tests/              # Comprehensive test suite
└── tools/              # Operational tooling
```

### Discovery-First Enhancement Strategy
1. **Phase 1**: Add gRPC proto definitions to existing Python services
2. **Phase 2**: Implement Two-Tier Bus with gRPC streaming
3. **Phase 3**: Enhanced discovery with asset fingerprinting
4. **Phase 4**: Rust security components for performance-critical paths

## Rationale

### Python Advantages
- Existing production-ready codebase (150+ dependencies)
- Rich AI/ML ecosystem (sklearn, threat intelligence)
- Temporal workflow SDK for complex orchestration
- FastAPI performance with async/await patterns

### Discovery-First Integration
- Current PTaaS scanner integration (Nmap, Nuclei, Nikto, SSLScan)
- Behavioral analytics and threat hunting capabilities
- Existing Redis/PostgreSQL data layer
- Vault-backed secret management

### Two-Tier Bus Compatibility
- Current middleware stack supports gRPC addition
- Existing tenant isolation and rate limiting
- Enterprise authentication with JWT/OIDC
- Monitoring stack (Prometheus, Grafana) ready for gRPC metrics

## Consequences

### Positive
- Leverage existing production-ready infrastructure
- Gradual migration to Discovery-First without disruption
- Proven scalability patterns already implemented
- Rich testing framework (pytest, Jest) in place

### Negative
- Python GIL limitations for CPU-intensive discovery
- Need additional protobuf tooling for existing services
- gRPC streaming requires async refactoring of some components

## Implementation Plan

### Immediate (Week 1-2)
1. Add `.proto` definitions for existing FastAPI endpoints
2. Implement gRPC gateway alongside REST APIs
3. Enable discovery streaming APIs

### Short-term (Month 1)
1. Migrate orchestrator to gRPC-based discovery workflows
2. Implement Two-Tier Bus message routing
3. Add discovery fingerprinting to PTaaS scanners

### Long-term (Month 2-3)
1. Performance-critical components in Rust
2. Advanced discovery correlation engine
3. Full Discovery-First SEaaS capabilities

## Compatibility Matrix

| Component | Current | Discovery-First | Two-Tier Bus |
|-----------|---------|----------------|--------------|
| API Layer | FastAPI REST | + gRPC Gateway | Message Bus |
| Frontend | React/REST | + gRPC-Web | Event Streaming |
| Security | JWT/RBAC | + mTLS | Tenant Isolation |
| Storage | PostgreSQL | + Discovery Schema | Event Sourcing |
| Monitoring | Prometheus | + gRPC Metrics | Bus Observability |