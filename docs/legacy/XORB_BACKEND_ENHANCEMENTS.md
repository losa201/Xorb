# Xorb Backend Enhancement Modules

This document outlines the principal engineering enhancements delivered for the Xorb backend platform, implementing production-ready security, performance, and operational capabilities.

##  Overview

Eight core modules have been implemented with a code-first, security-focused approach:

1. **AuthN/AuthZ** - OIDC login with role/tenant claims mapping
2. **Multi-tenancy** - Postgres RLS with safe tenant isolation
3. **Evidence/Uploads** - Secure file storage with validation
4. **Job Orchestration** - Reliable scheduler with idempotency
5. **Performance** - uvloop, DB pooling, pgvector optimization
6. **Observability** - OpenTelemetry, metrics, structured logging
7. **Security/DX** - Rate limiting, validation, error handling
8. **Build/CI** - Secure containerization and comprehensive testing

##  Module 1: AuthN/AuthZ - OIDC Integration & RBAC

- **Intent**: Replace basic auth with production OIDC integration and fine-grained RBAC.

- **Risk**: Medium - Authentication changes require careful migration.

###  Changes

```diff
+ src/api/app/auth/
  + models.py          # User claims, roles, permissions model
  + oidc.py           # OIDC provider with caching
  + dependencies.py   # FastAPI auth dependencies & RBAC decorators
  + routes.py         # Login/logout/callback endpoints
+ src/api/app/infrastructure/cache.py  # Redis caching backend
+ src/api/tests/test_auth.py           # Comprehensive auth tests
```text

- **Key Features**:
- OIDC discovery document caching
- JWT validation with role extraction
- Per-route RBAC decorators: `@rbac(permissions=[Permission.EVIDENCE_READ])`
- Tenant claims mapping from OIDC tokens
- Backwards compatibility shims

- **Dependencies Added**:
```toml
authlib>=1.3.0
httpx>=0.28.0
redis>=5.1.0
```text

###  Usage

```python
from app.auth.dependencies import rbac, require_permissions
from app.auth.models import Permission

@app.get("/evidence")
@rbac(permissions=[Permission.EVIDENCE_READ])
async def get_evidence(request: Request):
    user = request.state.user  # UserClaims object
    return {"tenant_id": user.tenant_id}
```text

##  Module 2: Multi-tenancy - Postgres RLS & Safe Migrations

- **Intent**: Implement secure tenant isolation using Postgres Row Level Security.

- **Risk**: High - Database schema changes affect data isolation.

###  Changes

```diff
+ src/api/app/domain/tenant_entities.py    # Tenant domain models
+ src/api/app/services/tenant_service.py   # Tenant management service
+ src/api/app/middleware/tenant_context.py # Request tenant context
+ src/api/migrations/versions/001_add_tenant_isolation.py
+ src/api/migrations/versions/002_create_tenant_tables.py
+ src/api/tests/test_multitenancy.py       # RLS and isolation tests
```text

- **Key Features**:
- Automatic tenant context via middleware: `SET app.tenant_id`
- RLS policies on all tenant-scoped tables
- Super admin bypass capability
- Safe backfill functions for existing data
- Tenant user management with role inheritance

- **Migration Safety**:
```sql
- - Enable RLS gradually
ALTER TABLE evidence ENABLE ROW LEVEL SECURITY;

- - Policy with super admin bypass
CREATE POLICY evidence_tenant_isolation ON evidence
USING (tenant_id::text = current_setting('app.tenant_id', true) OR
       bypass_rls_for_user(current_setting('app.user_role', true)));
```text

###  Rollback Plan
1. Disable RLS: `ALTER TABLE evidence DISABLE ROW LEVEL SECURITY`
2. Drop policies: `DROP POLICY evidence_tenant_isolation ON evidence`
3. Remove tenant columns (after data migration)

##  Module 3: Evidence/Uploads - Secure File Storage

- **Intent**: Build pluggable storage with filesystem/S3 backends and comprehensive validation.

- **Risk**: Medium - File handling requires security controls.

###  Changes

```diff
+ src/api/app/storage/
  + interface.py       # Storage driver interface & models
  + filesystem.py      # Local filesystem implementation
  + s3.py             # S3/MinIO implementation
  + validation.py     # File validation & malware scanning
+ src/api/app/services/storage_service.py  # Storage service layer
+ src/api/tests/test_storage.py           # Storage & validation tests
```text

- **Key Features**:
- Presigned URL generation for direct uploads
- MIME type validation with python-magic
- ClamAV integration (optional)
- Size limits by file category
- SHA256 integrity checking
- Tenant isolation in storage paths

- **Dependencies Added**:
```toml
boto3>=1.35.0
aiofiles>=24.1.0
python-magic>=0.4.27
```text

###  Usage

```python
from app.storage.interface import StorageDriverFactory, FilesystemConfig

# Configure storage backend
config = FilesystemConfig(
    backend=StorageBackend.FILESYSTEM,
    storage_root="/var/xorb/evidence",
    max_file_size=100*1024*1024
)

driver = StorageDriverFactory.create_driver(config)
service = StorageService(driver)

# Create upload URL
upload_info = await service.create_upload_url(
    filename="evidence.pdf",
    content_type="application/pdf",
    size_bytes=1024,
    tenant_id=tenant_id,
    uploaded_by=user_id
)
```text

##  Module 4: Job Orchestration - Reliable Scheduler & Workers

- **Intent**: Implement production job system with Redis queues, retries, and DLQ.

- **Risk**: Medium - Job reliability critical for system operations.

###  Changes

```diff
+ src/api/app/jobs/
  + models.py          # Job definitions, execution, retry policies
  + queue.py           # Redis-backed job queue with priorities
  + worker.py          # Async worker with graceful shutdown
  + service.py         # Job scheduling service
+ src/api/tests/test_jobs.py              # Job system tests
```text

- **Key Features**:
- Priority queues with Redis sorted sets
- Exponential backoff with jitter
- Idempotency key support
- Dead letter queue for failed jobs
- Worker health monitoring
- Graceful shutdown handling

###  Usage

```python
from app.jobs.service import JobService
from app.jobs.models import JobScheduleRequest, JobType

service = JobService(redis_client)

# Schedule job
job_info = await service.schedule_job(JobScheduleRequest(
    job_type=JobType.EVIDENCE_PROCESSING,
    payload={"evidence_id": str(evidence_id)},
    priority=JobPriority.HIGH,
    idempotency_key=f"process-{evidence_id}",
    tenant_id=tenant_id
))

# Start worker
worker = JobWorker(redis_client, queues=["default", "priority"])
worker.register_handler(JobType.EVIDENCE_PROCESSING, process_evidence)
await worker.start()
```text

##  Module 5: Performance - uvloop, DB Pooling, pgvector

- **Intent**: Optimize async performance with uvloop, database pooling, and vector search.

- **Risk**: Low - Performance improvements with compatibility fallbacks.

###  Changes

```diff
~ src/api/app/infrastructure/database.py   # Enhanced with pooling & optimization
+ src/api/app/infrastructure/vector_store.py  # pgvector similarity search
+ src/api/app/infrastructure/performance.py   # Performance monitoring
+ src/api/migrations/versions/003_add_pgvector_support.py
+ src/api/tests/test_performance.py           # Performance tests
```text

- **Key Features**:
- uvloop event loop for 30-40% async performance boost
- Optimized asyncpg connection pooling (5-20 connections)
- pgvector HNSW indexes for similarity search
- orjson for faster JSON serialization
- Connection pool monitoring and metrics
- Prepared statement caching

- **Dependencies Added**:
```toml
uvloop>=0.20.0
orjson>=3.10.0
psutil>=6.1.0
numpy>=1.26.0
```text

###  Configuration

```bash
# Environment variables for optimization
DB_MIN_POOL_SIZE=5
DB_MAX_POOL_SIZE=20
DB_STATEMENT_CACHE_SIZE=100
ENABLE_UVLOOP=true
ENABLE_ORJSON=true
```text

###  Vector Search Usage

```python
from app.infrastructure.vector_store import get_vector_store

vector_store = get_vector_store(dimension=1536)

# Add vector
await vector_store.add_vector(
    vector=embedding,
    tenant_id=tenant_id,
    source_type="evidence",
    source_id=evidence_id,
    content_hash=sha256_hash,
    embedding_model="text-embedding-ada-002"
)

# Search similar
results = await vector_store.search_similar(
    query_vector=query_embedding,
    tenant_id=tenant_id,
    limit=10,
    similarity_threshold=0.8
)
```text

##  Run Instructions

###  Prerequisites

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y postgresql-14 postgresql-14-pgvector redis-server libmagic1

# Optional: ClamAV for malware scanning
sudo apt-get install -y clamav clamav-daemon
sudo systemctl enable clamav-freshclam
```text

###  Database Setup

```bash
# Create database and user
sudo -u postgres psql
CREATE DATABASE xorb;
CREATE USER xorb WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE xorb TO xorb;
\q

# Enable pgvector extension
sudo -u postgres psql -d xorb -c "CREATE EXTENSION vector;"
```text

###  Application Setup

```bash
# Install dependencies
cd src/api
pip install -e .

# Environment configuration
cp .env.template .env
# Edit .env with database and Redis connection details

# Run migrations
alembic upgrade head

# Start services
uvicorn app.main:app --factory --workers 1 --port 8000 &

# Start job worker (separate process)
python -m app.jobs.worker &
```text

###  Development Commands

```bash
# Run tests
pytest -q

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Type checking
mypy src/

# Linting
ruff check src/
black src/

# Load testing
bombardier -c 64 -n 20000 http://127.0.0.1:8000/health
```text

###  Production Deployment

```bash
# Build optimized container
docker build -f Dockerfile.secure -t xorb-api:latest .

# Deploy with docker-compose
docker-compose -f infra/docker-compose.production.yml up -d

# Check health
curl http://localhost:8000/health
curl http://localhost:8000/readiness
```text

##  Safety Notes & Migration Order

###  Critical Migration Sequence

1. **Pre-migration backup**: Full database backup before RLS changes
2. **Deploy auth module**: New endpoints without breaking existing auth
3. **Gradual RLS rollout**: Enable per table with super admin bypass
4. **Tenant backfill**: Populate tenant_id for existing data
5. **Storage migration**: Migrate existing files to new storage structure
6. **Performance optimizations**: Enable uvloop and connection pooling
7. **Job system**: Deploy workers before scheduling jobs

###  Rollback Procedures

- **Auth Rollback**:
```bash
# Revert to previous auth dependencies
git checkout HEAD~1 -- app/dependencies.py
# Remove OIDC routes from main.py
```text

- **RLS Rollback**:
```sql
- - Emergency disable RLS
ALTER TABLE evidence DISABLE ROW LEVEL SECURITY;
ALTER TABLE findings DISABLE ROW LEVEL SECURITY;
ALTER TABLE embedding_vectors DISABLE ROW LEVEL SECURITY;
```text

- **Storage Rollback**:
- Keep old storage interface available during transition
- Dual-write during migration window
- Feature flag for new storage backend

###  Feature Flags

```python
# app/config.py
ENABLE_OIDC_AUTH = os.getenv("ENABLE_OIDC_AUTH", "false") == "true"
ENABLE_NEW_STORAGE = os.getenv("ENABLE_NEW_STORAGE", "false") == "true"
ENABLE_JOB_SYSTEM = os.getenv("ENABLE_JOB_SYSTEM", "false") == "true"
```text

###  Monitoring & Alerts

- **Key Metrics to Monitor**:
- Database connection pool utilization
- RLS policy execution time
- Job queue depth and processing time
- File upload success/failure rates
- Authentication token validation latency

- **Critical Alerts**:
- Database connection pool exhaustion
- Job dead letter queue growth
- Storage backend failures
- Authentication service outages

##  Performance Benchmarks

- **Target Performance (Definition of Done)**:
- p95 < 300ms @ 200 RPS on /health + hot GET endpoints
- Success rate ≥ 99.9% under normal load
- Database queries < 100ms p95 with proper indexing
- File upload/download throughput > 10MB/s
- Vector similarity search < 50ms for 1M vectors

- **Load Testing**:
```bash
# Health endpoint
bombardier -c 64 -n 20000 http://localhost:8000/health

# Evidence listing (authenticated)
bombardier -c 32 -n 5000 -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/evidence

# Vector search performance
bombardier -c 16 -n 1000 -H "Authorization: Bearer $TOKEN" \
  -m POST -f vector_search_payload.json \
  http://localhost:8000/api/vectors/search
```text

##  Security Considerations

- **Authentication Security**:
- OIDC token validation with proper issuer verification
- JWT signature verification with cached JWKS
- Role-based access control with least privilege
- Tenant isolation at token validation level

- **Storage Security**:
- File type validation with MIME detection
- Size limits and upload quotas per tenant
- Malware scanning integration (ClamAV)
- Signed URLs with time-based expiration
- Path traversal protection

- **Database Security**:
- Row Level Security (RLS) for tenant isolation
- Prepared statements to prevent SQL injection
- Connection encryption (SSL/TLS)
- Database user with minimal privileges
- Audit logging for sensitive operations

- **Infrastructure Security**:
- Non-root container execution
- Read-only root filesystem
- Resource limits and quotas
- Network segmentation
- Secret management via environment variables

##  Conclusion

This enhancement delivers a production-ready, secure, and performant backend platform with:

- **Security-first design** with OIDC authentication and tenant isolation
- **High performance** with uvloop, connection pooling, and vector search
- **Operational excellence** with comprehensive monitoring and job orchestration
- **Developer experience** with strong typing, testing, and tooling
- **Scalability** through async patterns and efficient resource usage

All modules follow clean architecture principles, maintain backwards compatibility where possible, and include comprehensive test coverage for reliability and maintainability.