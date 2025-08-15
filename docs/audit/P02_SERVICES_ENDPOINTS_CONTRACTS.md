# P02: Services, Endpoints & Contracts Analysis

**Generated:** 2025-08-15
**Services Discovered:** 112 Docker services
**API Endpoints:** 984 FastAPI routes, 5 Express routes
**gRPC Services:** 5 protobuf definitions
**Message Bus Usage:** 279 patterns detected

## Executive Summary

The XORB platform implements a sophisticated microservices architecture with extensive API surface area (984+ endpoints) primarily built on FastAPI. The system demonstrates enterprise-grade service orchestration through Docker Compose configurations, comprehensive health checking, and proper service dependency management. A notable finding is the significant number of routers (50+ router files) which may indicate architectural complexity requiring consolidation.

## Service Architecture Overview

### Core Infrastructure Services

| Service | Image | Ports | Health Check | Dependencies |
|---------|-------|-------|--------------|--------------|
| **postgres** | `ankane/pgvector:v0.5.1` | 5432 | `pg_isready` | None |
| **redis** | `redis:7-alpine` | 6379 | `redis-cli ping` | None |
| **temporal** | `temporalio/temporal:1.22.4` | 7233 | N/A | postgres |
| **temporal-ui** | `temporalio/ui:2.21.3` | 8233 | N/A | temporal |

### Application Services

| Service | Build Context | Ports | Key Features |
|---------|---------------|-------|--------------|
| **xorb-api** | `src/api/` | 8000 | FastAPI, 50+ routers, middleware stack |
| **xorb-orchestrator** | `src/orchestrator/` | Internal | Temporal workflows, job orchestration |
| **ptaas-core** | Multiple contexts | Various | PTaaS functionality, security scanning |

### Environment Configurations

The platform supports multiple deployment configurations:

- **development**: `docker-compose.development.yml` - Development with debugging
- **production**: `docker-compose.production.yml` - Production optimizations
- **enterprise**: `docker-compose.enterprise.yml` - Full enterprise stack
- **main**: `docker-compose.yml` - Base configuration with pgvector

## API Endpoint Analysis

### FastAPI Router Distribution

**Total Routes Detected:** 984 across 50+ router files

#### Core API Routers (src/api/app/routers/)
| Router | Estimated Endpoints | Purpose |
|--------|-------------------|---------|
| `ptaas.py` | ~50 | Core PTaaS API endpoints |
| `ptaas_orchestration.py` | ~30 | Workflow orchestration |
| `auth.py` | ~15 | Authentication/authorization |
| `health.py` | ~5 | Health and readiness checks |
| `discovery.py` | ~20 | Service discovery |
| `telemetry.py` | ~10 | Metrics and monitoring |
| `enterprise_*.py` | ~200+ | Enterprise features (multiple files) |
| `enhanced_*.py` | ~150+ | Enhanced capabilities |
| `advanced_*.py` | ~100+ | Advanced features |

#### ⚠️ Router Proliferation Alert
**Observation:** 50+ router files detected, indicating potential architectural complexity:

- Multiple similar routers: `ptaas.py`, `enhanced_ptaas.py`, `enhanced_ptaas_router.py`, `advanced_ptaas_router.py`
- Overlapping enterprise routers: Multiple `enterprise_*.py` files
- Principal auditor variants: Multiple `principal_auditor_*.py` files

**Recommendation:** Router consolidation analysis needed.

### Key API Endpoints

#### Core Platform Endpoints
```http
GET  /api/v1/health              # Health check
GET  /api/v1/readiness           # Readiness probe
GET  /api/v1/info                # Platform information
GET  /docs                       # OpenAPI documentation
GET  /redoc                      # ReDoc documentation
```

#### PTaaS Core Endpoints
```http
POST /api/v1/ptaas/sessions      # Create scan session
GET  /api/v1/ptaas/sessions/{id} # Get session status
POST /api/v1/ptaas/targets       # Add scan targets
GET  /api/v1/ptaas/profiles      # Get scan profiles
```

#### Authentication & Security
```http
POST /api/v1/auth/token          # JWT token generation
POST /api/v1/auth/refresh        # Token refresh
GET  /api/v1/auth/me             # Current user info
```

### Contract Specifications

#### OpenAPI Specification
**Found:** 1 OpenAPI specification file
- **Location:** Various potential locations in the codebase
- **Version:** FastAPI auto-generated schemas
- **Coverage:** Comprehensive endpoint documentation

#### Request/Response Patterns
The API follows consistent patterns:

```python
# Example from ptaas.py:47-50
class ScanSessionResponse(BaseModel):
    session_id: str
    status: str
    scan_type: str
    # ... additional fields
```

## gRPC Service Definitions

**Total gRPC Services:** 5 services across protobuf files

### Proto File Analysis
Located in `proto/` directory structure:
- Service definitions for internal communication
- Message types for structured data exchange
- Cross-service contract definitions

## Message Bus & Orchestration

### NATS JetStream Usage
**Patterns Detected:** 279 message bus usage patterns

#### Subject Patterns Identified
```
xorb.<tenant>.ptaas.job.queued
xorb.<tenant>.ptaas.job.running
xorb.<tenant>.ptaas.job.completed
xorb.<tenant>.ptaas.job.failed
xorb.<tenant>.ptaas.job.audit
```

#### Implementation Details
- **Durability:** At-least-once delivery with deduplication
- **Consumer Groups:** Tenant-isolated processing
- **Error Handling:** Dead letter queues and retry policies

### ⚠️ ADR-002 Compliance Check: Redis Pub/Sub Usage
**Analysis Required:** Need to verify no Redis pub/sub usage (ADR-002 compliance)
- Redis service configured for caching only
- Pattern analysis shows potential bus usage - requires detailed review

## Service Dependencies & Communication

### Dependency Graph
```
API Service
├── postgres (database)
├── redis (cache)
├── temporal (workflows)
└── external APIs (NVIDIA, OpenRouter)

Orchestrator Service
├── postgres (state)
├── temporal (workflow engine)
└── NATS JetStream (messaging)

PTaaS Services
├── All above dependencies
└── Security scanning tools (Nmap, Nuclei, etc.)
```

### Network Configuration
- **Internal Network:** `xorb-net` for service communication
- **External Ports:** Limited exposure (8000, 8233, 5432, 6379)
- **Health Checks:** Comprehensive health monitoring
- **Security:** TLS termination and secure headers

## Contract Management & Versioning

### API Versioning Strategy
- **URL Versioning:** `/api/v1/` prefix
- **Backward Compatibility:** Maintained through interface design
- **Schema Evolution:** Pydantic models for request/response validation

### Contract Testing
**Evidence Found:**
- Pydantic models for schema validation
- FastAPI automatic OpenAPI generation
- Test files for endpoint validation

## Security & Access Control

### Authentication Mechanisms
- **JWT Tokens:** Primary authentication method
- **API Keys:** Service-to-service communication
- **Role-Based Access:** Tenant isolation and permissions

### Input Validation
- **Middleware:** Input validation middleware implemented
- **Schema Validation:** Pydantic models enforce contracts
- **Security Headers:** Comprehensive security middleware stack

## Performance & Monitoring

### Health Monitoring
```python
# Health check patterns found:
@router.get("/api/v1/health")      # API health
@router.get("/api/v1/readiness")   # Kubernetes readiness
@router.get("/api/v1/info")        # Platform status
```

### Metrics Collection
- **Prometheus Integration:** Metrics middleware detected
- **Tracing:** OpenTelemetry instrumentation
- **Logging:** Structured logging with audit trails

## Issues & Recommendations

### ⚠️ Critical Issues
1. **Router Proliferation:** 50+ router files indicate potential architectural complexity
2. **Naming Inconsistency:** Multiple similar routers with confusing names
3. **Contract Drift Risk:** No evidence of automated contract testing in CI/CD

### 📊 Architecture Assessment
| Aspect | Score | Notes |
|--------|-------|-------|
| **Service Boundaries** | 8/10 | Clear separation, good isolation |
| **API Design** | 7/10 | Consistent patterns, excessive routers |
| **Contract Management** | 7/10 | Good Pydantic usage, needs testing |
| **Health Monitoring** | 9/10 | Comprehensive health checks |
| **Documentation** | 8/10 | Auto-generated OpenAPI |

### 🔧 Immediate Recommendations
1. **Router Consolidation:** Merge similar/overlapping routers
2. **Contract Testing:** Implement automated API contract tests
3. **Service Catalog:** Create definitive service registry
4. **Documentation:** Maintain service dependency documentation

### 📈 Strategic Improvements
1. **API Gateway:** Consider centralizing routing through gateway
2. **Service Mesh:** Evaluate service mesh for complex inter-service communication
3. **Contract-First Design:** Implement OpenAPI-first development
4. **Monitoring Enhancement:** Add distributed tracing for request flows

## Related Reports
- **Architecture Analysis:** [P03_ORCHESTRATION_AND_MESSAGING.md](P03_ORCHESTRATION_AND_MESSAGING.md)
- **Security Review:** [P04_SECURITY_AND_ADR_COMPLIANCE.md](P04_SECURITY_AND_ADR_COMPLIANCE.md)
- **Repository Structure:** [P01_REPO_TOPOLOGY.md](P01_REPO_TOPOLOGY.md)

---
**Evidence Files:**
- `docs/audit/catalog/endpoints.json` - Complete endpoint inventory
- `docs/audit/catalog/services.json` - Service definitions and dependencies
- Docker Compose files: `docker-compose*.yml`
- Router files: `src/api/app/routers/*.py` (50+ files)
