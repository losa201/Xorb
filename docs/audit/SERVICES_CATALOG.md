# XORB Services Catalog

**Audit Date**: 2025-08-15
**Total Services**: 12 active services
**Container Images**: 8 production images
**API Endpoints**: 47 documented endpoints

## Core Services

### 1. XORB API Service
**Name**: xorb-ptaas-api
**Type**: REST API
**Entrypoints**:
- `src/api/app/main.py` (Primary FastAPI app)
- `src/api/Dockerfile` (Container)

**Ports**:
- 8000/tcp (HTTP/HTTPS)
- 9090/tcp (Metrics)

**Health Endpoints**:
- `/health` - Liveness probe
- `/readiness` - Readiness probe
- `/api/v1/health` - Application health
- `/api/v1/info` - Service information

**Metrics**: `/metrics` (Prometheus format)

**OpenTelemetry**:
```yaml
service.name: xorb-ptaas-api
service.version: 2025.08-rc2
```

**NATS Subjects**:
- **Publish**: `xorb.*.ptaas.job.queued`, `xorb.*.audit.api`
- **Subscribe**: `xorb.*.ptaas.job.completed`, `xorb.*.ptaas.job.failed`

**Configuration**:
```env
DATABASE_URL=postgresql://user:pass@localhost:5432/xorb
REDIS_URL=redis://localhost:6379
NATS_URL=nats://localhost:4222
JWT_SECRET=<vault:secret/xorb/config#jwt_secret>
LOG_LEVEL=INFO
PTAAS_WORKERS=28
```

**Storage**:
- PostgreSQL tables: `users`, `tenants`, `ptaas_sessions`
- Redis cache: sessions, rate limits

**Start Commands**:
```bash
# Development
cd src/api && uvicorn app.main:app --reload --port 8000

# Production
docker run -p 8000:8000 xorb/api:latest

# Make target
make ptaas-quickstart
```

### 2. PTaaS Orchestrator
**Name**: xorb-ptaas-orchestrator
**Type**: Workflow Engine
**Entrypoints**:
- `src/orchestrator/main.py` (Temporal orchestrator)
- `src/orchestrator/service/orchestrator_loop.py` (NATS consumer)

**Ports**:
- 7233/tcp (Temporal client)
- 9091/tcp (Metrics)

**Health Endpoints**:
- `/health` - Circuit breaker status
- `/metrics` - Performance metrics

**OpenTelemetry**:
```yaml
service.name: xorb-ptaas-orchestrator
service.version: 2025.08-rc2
```

**NATS Subjects**:
- **Subscribe**: `xorb.*.ptaas.job.queued`, `xorb.*.ptaas.job.pause`
- **Publish**: `xorb.*.ptaas.job.running`, `xorb.*.ptaas.job.completed`

**Configuration**:
```env
TEMPORAL_HOST=temporal:7233
NATS_URL=nats://localhost:4222
PTAAS_WORKERS=28
PTAAS_CPU_POOL=16
PTAAS_IO_CONCURRENCY=128
MAX_CONCURRENT_JOBS=80
```

**Storage**:
- SQLite: `ptaas_jobs.db`, `ptaas_jobs.db-wal`
- NATS streams: job state, evidence

**Start Commands**:
```bash
# Development
cd src/orchestrator && python main.py

# Production
docker run xorb/orchestrator:latest
```

### 3. PTaaS Frontend
**Name**: xorb-ptaas-frontend
**Type**: React SPA
**Entrypoints**:
- `ptaas/src/main.tsx` (React entry point)
- `ptaas/vite.config.ts` (Vite configuration)

**Ports**:
- 3000/tcp (Development server)
- 80/tcp (Production Nginx)

**Health Endpoints**:
- `/health.json` - Application health
- `/version.json` - Build information

**Configuration**:
```env
VITE_API_BASE_URL=http://localhost:8000
VITE_ENVIRONMENT=production
VITE_SENTRY_DSN=<sentry-dsn>
```

**Start Commands**:
```bash
# Development
cd ptaas && npm run dev

# Production
cd ptaas && npm run build && npm run preview
```

### 4. Scanner Core (Rust)
**Name**: xorb-scanner-core
**Type**: Security Scanner
**Entrypoints**:
- `services/scanner-rs/scanner-core/src/main.rs`
- `services/scanner-rs/scanner-core/Cargo.toml`

**Ports**:
- 8080/tcp (gRPC API)
- 9092/tcp (Metrics)

**Health Endpoints**:
- `/health` - Scanner status
- `/metrics` - Performance metrics

**NATS Subjects**:
- **Subscribe**: `xorb.*.scanner.requests`
- **Publish**: `xorb.*.scanner.results`, `xorb.*.findings`

**Configuration**:
```env
RUST_LOG=info
NATS_URL=nats://localhost:4222
SCANNER_THREADS=16
MAX_CONCURRENT_SCANS=50
```

**Start Commands**:
```bash
# Development
cd services/scanner-rs && cargo run

# Production
docker run xorb/scanner-core:latest
```

### 5. Intelligence Engine
**Name**: xorb-intelligence-engine
**Type**: AI/ML Service
**Entrypoints**:
- `src/xorb/intelligence_engine/main.py`

**Ports**:
- 8001/tcp (HTTP API)
- 9093/tcp (Metrics)

**Health Endpoints**:
- `/health` - Model status
- `/metrics` - ML metrics

**NATS Subjects**:
- **Subscribe**: `xorb.*.intelligence.requests`
- **Publish**: `xorb.*.intelligence.analysis`, `xorb.*.threats`

**Configuration**:
```env
NVIDIA_API_KEY=<vault:secret/xorb/external#nvidia_api_key>
OPENROUTER_API_KEY=<vault:secret/xorb/external#openrouter_api_key>
MODEL_CACHE_SIZE=2048
INFERENCE_TIMEOUT=30s
```

### 6. Evidence Service (G7)
**Name**: xorb-evidence-service
**Type**: Compliance Engine
**Entrypoints**:
- `src/xorb/audit/audit_logger.py`
- `compliance/soc2/evidence_collector.py`

**Ports**:
- 8002/tcp (Evidence API)
- 9094/tcp (Metrics)

**Health Endpoints**:
- `/health` - Chain integrity
- `/metrics` - Evidence metrics

**NATS Subjects**:
- **Subscribe**: `xorb.*.evidence.collect`
- **Publish**: `xorb.*.evidence.stored`, `xorb.*.audit.trails`

**Configuration**:
```env
EVIDENCE_STORE_PATH=/app/evidence
CHAIN_VALIDATION=strict
RETENTION_DAYS=2555  # 7 years
SIGNING_KEY_PATH=/vault/evidence-signing.key
```

### 7. Control Plane (G8)
**Name**: xorb-control-plane
**Type**: Resource Controller
**Entrypoints**:
- `src/api/app/routers/g8_control_plane.py`

**Ports**:
- 8003/tcp (Control API)
- 9095/tcp (Metrics)

**Health Endpoints**:
- `/health` - Quota status
- `/metrics` - Resource metrics

**NATS Subjects**:
- **Subscribe**: `xorb.*.quota.requests`
- **Publish**: `xorb.*.quota.limits`, `xorb.*.fairness.metrics`

**Configuration**:
```env
WFQ_ENABLED=true
TENANT_MAX_CONCURRENT=8
FAIRNESS_WINDOW_SEC=60
QUOTA_ENFORCEMENT=strict
```

## Supporting Services

### 8. Temporal Server
**Name**: temporal
**Type**: Workflow Engine
**Entrypoints**: External service (Docker image)

**Ports**:
- 7233/tcp (gRPC)
- 8233/tcp (Web UI)

**Health Endpoints**:
- `/health` - Temporal health

**Configuration**: `infra/temporal/temporal-config/development-sql.yaml`

### 9. NATS JetStream
**Name**: nats
**Type**: Message Broker
**Entrypoints**: External service (Docker image)

**Ports**:
- 4222/tcp (Client)
- 8222/tcp (HTTP Monitor)
- 6222/tcp (Cluster)

**Health Endpoints**:
- `/healthz` - NATS health
- `/varz` - Variable stats

### 10. PostgreSQL
**Name**: postgres
**Type**: Database
**Entrypoints**: External service (Docker image)

**Ports**:
- 5432/tcp (PostgreSQL)

**Health Endpoints**:
- Custom health check via psql

**Configuration**:
```env
POSTGRES_DB=xorb
POSTGRES_USER=xorb
POSTGRES_PASSWORD=<vault:secret/xorb/db#password>
```

### 11. Redis
**Name**: redis
**Type**: Cache Store
**Entrypoints**: External service (Docker image)

**Ports**:
- 6379/tcp (Redis)

**Health Endpoints**:
- `PING` command response

**Configuration**:
```env
REDIS_PASSWORD=<vault:secret/xorb/db#redis_password>
REDIS_MAXMEMORY=2gb
REDIS_MAXMEMORY_POLICY=allkeys-lru
```

### 12. Prometheus
**Name**: prometheus
**Type**: Metrics Collection
**Entrypoints**: External service (Docker image)

**Ports**:
- 9090/tcp (Prometheus UI)

**Health Endpoints**:
- `/-/healthy` - Prometheus health

**Configuration**: `infra/monitoring/prometheus.yml`

## Service Interaction Graph

```mermaid
graph LR
    Client[Client Apps] --> LB[Load Balancer]
    LB --> API[XORB API]
    API --> NATS[NATS JetStream]
    API --> DB[(PostgreSQL)]
    API --> Cache[(Redis)]

    NATS --> Orchestrator[PTaaS Orchestrator]
    NATS --> Scanner[Scanner Core]
    NATS --> Intelligence[Intelligence Engine]

    Orchestrator --> Temporal[Temporal Server]
    Orchestrator --> Evidence[Evidence Service]

    Intelligence --> NVIDIA[NVIDIA API]
    Intelligence --> OpenRouter[OpenRouter API]

    Evidence --> AuditDB[(Audit Database)]
    API --> ControlPlane[Control Plane G8]

    Prometheus[Prometheus] --> API
    Prometheus --> Orchestrator
    Prometheus --> Scanner
    Prometheus --> Intelligence

    Grafana[Grafana] --> Prometheus
```

## Service Dependencies Matrix

| Service | PostgreSQL | Redis | NATS | Temporal | External APIs |
|---------|------------|-------|------|----------|---------------|
| API | ✅ | ✅ | ✅ | ❌ | ✅ GitHub, Slack |
| Orchestrator | ❌ | ❌ | ✅ | ✅ | ❌ |
| Scanner | ❌ | ❌ | ✅ | ❌ | ❌ |
| Intelligence | ❌ | ✅ | ✅ | ❌ | ✅ NVIDIA, OpenRouter |
| Evidence | ✅ | ❌ | ✅ | ❌ | ❌ |
| Control Plane | ✅ | ✅ | ✅ | ❌ | ❌ |

## Deployment Configuration

### Docker Compose Services
```yaml
version: '3.8'
services:
  api:
    image: xorb/api:latest
    ports: ["8000:8000"]
    depends_on: [postgres, redis, nats]

  orchestrator:
    image: xorb/orchestrator:latest
    depends_on: [nats, temporal]

  scanner:
    image: xorb/scanner-core:latest
    depends_on: [nats]

  intelligence:
    image: xorb/intelligence:latest
    depends_on: [nats, redis]
```

### Kubernetes Resources
- Deployments: 8 services
- Services: 12 endpoints
- ConfigMaps: 6 configurations
- Secrets: 4 secret stores
- PersistentVolumes: 3 storage claims

## Performance Characteristics

| Service | CPU Request | Memory Request | Max Replicas | Response Time |
|---------|-------------|----------------|--------------|---------------|
| API | 500m | 1Gi | 10 | 95th: 200ms |
| Orchestrator | 1000m | 2Gi | 3 | 95th: 1.8s |
| Scanner | 2000m | 1Gi | 5 | 95th: 30s |
| Intelligence | 4000m | 8Gi | 2 | 95th: 5s |
| Evidence | 200m | 512Mi | 3 | 95th: 100ms |

## Monitoring & Alerting

### Critical Alerts
- API response time > 2s
- Orchestrator queue depth > 100
- Scanner failure rate > 5%
- Database connection pool exhaustion
- NATS consumer lag > 500 messages

### SLA Targets
- API availability: 99.9%
- PTaaS job completion: 95% within 2 minutes
- Evidence integrity: 100%
- Alert response time: < 5 minutes

---

*This services catalog provides comprehensive information about all running services in the XORB platform, their configurations, dependencies, and operational characteristics.*
