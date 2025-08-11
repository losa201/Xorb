#  XORB Docker Containerization Guide

##  Overview

The XORB platform has been completely containerized with multi-stage Docker builds optimized for security, performance, and operational efficiency. This guide covers the containerization strategy, build process, and deployment patterns.

##  Container Architecture

###  Multi-stage Build Strategy

Each service uses a multi-stage Dockerfile with the following stages:

1. **Builder Stage** - Compiles dependencies and build artifacts
2. **Runtime Base** - Minimal runtime environment with security hardening
3. **Development Stage** - Full development environment with debugging tools
4. **Production Stage** - Optimized production runtime with minimal attack surface
5. **Testing Stage** - Specialized environment for running tests

###  Security Hardening

All containers implement security best practices:

- **Non-root user**: All services run as `xorb` user (UID 1000)
- **Minimal base images**: Using `python:3.11-slim-bookworm` for smaller attack surface
- **No new privileges**: `security_opt: no-new-privileges:true`
- **Dependency isolation**: Virtual environments for Python dependencies
- **File permissions**: Strict file permissions and ownership
- **Process management**: `dumb-init` for proper signal handling

##  Service Containers

###  API Service (`src/api/Dockerfile`)

**Features:**
- FastAPI application with Gunicorn in production
- Multi-stage build with development and production targets
- Health checks and graceful shutdown handling
- Configuration management integration
- Security middleware and rate limiting

**Build Targets:**
- `development`: Hot-reload, debugging tools, verbose logging
- `production`: Optimized runtime, multiple workers, security hardening
- `testing`: Test execution environment with coverage tools

**Key Environment Variables:**
```bash
XORB_ENV=production          # Environment configuration
WORKERS=8                    # Gunicorn worker processes
TIMEOUT=60                   # Request timeout
DEBUG=false                  # Debug mode
LOG_LEVEL=INFO              # Logging level
```

###  Orchestrator Service (`src/orchestrator/Dockerfile`)

**Features:**
- Temporal workflow orchestration
- Circuit breaker pattern for resilience
- Async workflow execution
- Configuration-driven service discovery

**Environment Variables:**
```bash
TEMPORAL_HOST=temporal:7233  # Temporal server connection
XORB_ENV=production         # Environment configuration
```

###  Worker Service (`src/services/worker/Dockerfile`)

**Features:**
- Background job processing
- Scalable worker processes
- Resource-aware concurrency
- Redis queue integration

**Environment Variables:**
```bash
WORKER_CONCURRENCY=8        # Worker process count
WORKER_MAX_MEMORY=512m      # Memory limit per worker
```

##  Docker Compose Configurations

###  Development Environment (`docker-compose.development.yml`)

**Features:**
- Hot-reload for all services
- Development databases with logging
- Monitoring stack (Prometheus, Grafana)
- Volume mounts for code changes
- Debug-friendly configuration

**Services:**
- PostgreSQL with pgvector extension
- Redis with persistence
- API, Orchestrator, Worker services
- Temporal server
- Prometheus metrics
- Grafana dashboards

**Usage:**
```bash
#  Start development environment
docker-compose -f docker-compose.development.yml up -d

#  View logs
docker-compose -f docker-compose.development.yml logs -f api-dev

#  Scale workers
docker-compose -f docker-compose.development.yml up -d --scale worker-dev=4
```

###  Production Environment (`docker-compose.production.yml`)

**Features:**
- Production-optimized containers
- Secrets management
- Resource limits and reservations
- Health checks and restart policies
- Load balancing and SSL termination
- Comprehensive monitoring stack

**Production Services:**
- PostgreSQL with performance tuning
- Redis with clustering support
- Multiple API replicas (3x)
- Multiple worker replicas (4x)
- Nginx reverse proxy
- Temporal server
- Full monitoring stack

**Resource Allocation:**
```yaml
api-prod:
  deploy:
    replicas: 3
    resources:
      limits:
        memory: 1G
        cpus: '1.0'
      reservations:
        memory: 512M
        cpus: '0.5'
```

##  Container Management Scripts

###  Build Script (`tools/scripts/docker-build.sh`)

Comprehensive container management tool with the following capabilities:

**Commands:**
```bash
#  Build all services for development
./tools/scripts/docker-build.sh build --environment development --target development

#  Build and push production images
./tools/scripts/docker-build.sh build --environment production --target production --push --version v1.2.3

#  Run security scans
./tools/scripts/docker-build.sh security-scan api orchestrator worker

#  Generate size report
./tools/scripts/docker-build.sh size-report

#  Deploy services
./tools/scripts/docker-build.sh deploy --environment production

#  Clean up resources
./tools/scripts/docker-build.sh clean --environment development
```

**Features:**
- Parallel and sequential builds
- Build caching optimization
- Security vulnerability scanning
- Image size optimization
- Registry push/pull operations
- Resource cleanup

###  Container Testing (`test_containers.py`)

Automated container testing suite covering:

**Test Categories:**
1. **Docker Environment** - Verify Docker daemon and resources
2. **Image Building** - Test multi-stage build process
3. **Security Validation** - Verify non-root user, labels, permissions
4. **Startup Testing** - Validate container startup and health
5. **Configuration Integration** - Test centralized config management
6. **Size Analysis** - Monitor and report image sizes
7. **Health Endpoints** - Verify service health checks
8. **Volume Mounts** - Test file system integration

**Usage:**
```bash
#  Run all container tests
python test_containers.py

#  Expected output:
🔧 XORB Container Test Suite
============================================================
🧪 Running: Docker Environment
🐳 Testing Docker Environment...
  ✅ Docker daemon connected (version: 24.0.7)
  ✅ Found 5 containers
  ✅ Found 23 images

🧪 Running: Build Test Images
🔨 Building Test Images...
  Building api service...
  ✅ Built api image: sha256:abc123
  # ... more tests

📊 Test Results:
  ✅ PASS - Docker Environment
  ✅ PASS - Build Test Images
  ✅ PASS - Container Security
  # ... all results

🎯 Summary: 8/8 tests passed
🎉 All container tests passed!
```

##  Configuration Management Integration

###  Centralized Configuration

All containers use the centralized configuration management system:

```python
from common.config_manager import get_config

#  Configuration automatically loaded based on XORB_ENV
config = get_config()
db_url = config.database.get_url()
api_port = config.api_service.port
```

###  Environment-specific Configs

Containers automatically load configuration based on the `XORB_ENV` environment variable:

- `development` - Development settings with debug logging
- `staging` - Staging environment for testing
- `production` - Production settings with security hardening
- `test` - Testing environment for automated tests

###  Secret Management

Production containers integrate with HashiCorp Vault and Docker secrets:

```yaml
#  Docker Compose secrets
secrets:
  postgres_password:
    file: ./secrets/postgres_password
  jwt_secret:
    file: ./secrets/jwt_secret

#  Service configuration
api-prod:
  secrets:
    - postgres_password
    - jwt_secret
```

##  Performance Optimization

###  Image Size Optimization

**Before Optimization:**
- API Service: ~1.2GB
- Orchestrator: ~1.1GB
- Worker: ~1.0GB
- **Total: ~3.3GB**

**After Multi-stage Optimization:**
- API Service: ~400MB
- Orchestrator: ~350MB
- Worker: ~300MB
- **Total: ~1.05GB** (68% reduction)

###  Build Optimization

**Caching Strategy:**
```dockerfile
#  Dependencies cached separately from source code
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

#  Source code copied after dependencies
COPY --chown=xorb:xorb . .
```

**Build Cache Tags:**
```bash
#  Build with cache reference
docker build --cache-from xorb/api:dev-cache --target development -t xorb/api:dev .

#  Push cache for CI/CD
docker push xorb/api:dev-cache
```

###  Runtime Optimization

**Resource Limits:**
```yaml
deploy:
  resources:
    limits:
      memory: 1G
      cpus: '1.0'
    reservations:
      memory: 512M
      cpus: '0.5'
```

**Health Checks:**
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

##  Monitoring and Observability

###  Container Metrics

Prometheus collects metrics from all containers:
- CPU and memory usage
- Request rates and response times
- Error rates and status codes
- Custom application metrics

###  Log Aggregation

Structured logging from all containers:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "service": "api",
  "container_id": "abc123",
  "message": "Request processed",
  "duration": 0.245,
  "status_code": 200
}
```

###  Dashboards

Grafana dashboards for container monitoring:
- **Container Overview**: Resource usage across all services
- **Service Health**: Application-specific metrics
- **Performance**: Response times and throughput
- **Errors**: Error rates and failure analysis

##  Security Scanning

###  Vulnerability Scanning

Automated security scanning with Trivy:
```bash
#  Scan for vulnerabilities
./tools/scripts/docker-build.sh security-scan

#  Expected output:
🔒 Running security scans on images...
Scanning api for vulnerabilities...
✅ No HIGH or CRITICAL vulnerabilities found
Scanning orchestrator for vulnerabilities...
✅ No HIGH or CRITICAL vulnerabilities found
```

###  Security Compliance

All containers meet security compliance requirements:
- ✅ Non-root user execution
- ✅ Minimal base images
- ✅ No secrets in images
- ✅ Read-only root filesystem where possible
- ✅ Network security policies
- ✅ Resource limitations

##  Deployment Strategies

###  Blue-Green Deployment

Production deployment with zero downtime:

```bash
#  Build new version
./tools/scripts/docker-build.sh build --version v1.2.3 --target production --push

#  Deploy to staging
./tools/scripts/docker-build.sh deploy --environment staging

#  Run tests against staging
./tools/scripts/docker-build.sh test --environment staging

#  Deploy to production
./tools/scripts/docker-build.sh deploy --environment production --version v1.2.3
```

###  Rolling Updates

Docker Compose rolling updates:
```yaml
deploy:
  update_config:
    parallelism: 1
    delay: 30s
    failure_action: rollback
    order: stop-first
  rollback_config:
    parallelism: 1
    delay: 30s
```

###  Scaling Operations

Dynamic service scaling:
```bash
#  Scale API services
docker-compose -f docker-compose.production.yml up -d --scale api-prod=5

#  Scale workers based on load
docker-compose -f docker-compose.production.yml up -d --scale worker-prod=8
```

##  Troubleshooting

###  Common Issues

**1. Container Won't Start**
```bash
#  Check logs
docker-compose logs -f api-prod

#  Check configuration
./tools/scripts/config-manager.sh validate production
```

**2. Out of Memory Errors**
```bash
#  Check resource usage
docker stats

#  Increase memory limits
#  Edit docker-compose.yml resources section
```

**3. Health Check Failures**
```bash
#  Test health endpoint manually
curl http://localhost:8000/health

#  Check container status
docker-compose ps
```

###  Debugging Tools

**Container Shell Access:**
```bash
#  Development containers
docker-compose exec api-dev /bin/bash

#  Production containers (limited shell)
docker-compose exec api-prod /bin/sh
```

**Log Analysis:**
```bash
#  Follow logs for specific service
docker-compose logs -f --tail=100 api-prod

#  Search logs for errors
docker-compose logs api-prod 2>&1 | grep ERROR
```

**Performance Analysis:**
```bash
#  Monitor resource usage
docker stats api-prod orchestrator-prod worker-prod

#  Generate performance report
./tools/scripts/docker-build.sh size-report
```

##  Best Practices

###  Development Workflow

1. **Use development containers** for local development with hot-reload
2. **Test configuration changes** in development environment first
3. **Run container tests** before pushing changes
4. **Use build caching** to speed up development builds

###  Production Deployment

1. **Always use production targets** for production deployments
2. **Implement health checks** for all services
3. **Set resource limits** to prevent resource exhaustion
4. **Use secrets management** for sensitive configuration
5. **Monitor container metrics** and set up alerting
6. **Test disaster recovery** procedures regularly

###  Security Considerations

1. **Never run as root** in production containers
2. **Scan images** for vulnerabilities before deployment
3. **Use minimal base images** to reduce attack surface
4. **Implement proper logging** without exposing secrets
5. **Keep base images updated** with security patches

##  Migration Guide

###  From Legacy Containers

If migrating from previous container setup:

1. **Backup existing data** and configurations
2. **Update environment variables** to use new configuration system
3. **Test new containers** in development environment
4. **Gradually migrate services** using blue-green deployment
5. **Monitor performance** and adjust resource limits as needed

###  Configuration Updates

Update existing configurations to use centralized config management:

```python
#  Old approach
database_url = os.getenv('DATABASE_URL')

#  New approach
from common.config_manager import get_config
config = get_config()
database_url = config.database.get_url()
```

---

This containerization implementation provides a robust, secure, and scalable foundation for the XORB platform with comprehensive tooling for development, testing, and production operations.