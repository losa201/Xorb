# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Development Setup
```bash
# Create and activate virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies (from project root)
pip install -r requirements.lock  # or pip install -e .

# Install frontend dependencies (React + Vite - PTaaS directory)
cd services/ptaas/web && npm install

# Start main API service (FastAPI) - PRODUCTION-READY COMMAND
cd src/api && source ../../venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Alternative: Direct Python import (from API directory)
cd src/api && source ../../venv/bin/activate && python -c "from app.main import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000)"

# Start orchestrator service (Temporal-based)
cd src/orchestrator && source ../../venv/bin/activate && python main.py

# Start worker service (if available)
cd src/services/worker && source ../../venv/bin/activate && python worker.py
```

### Docker Development
```bash
# Enterprise deployment with all services
docker-compose -f docker-compose.enterprise.yml up -d

# Development environment
docker-compose -f docker-compose.development.yml up -d

# Production deployment
docker-compose -f docker-compose.production.yml up -d

# View logs
docker-compose logs -f [service_name]
```

### Frontend Development
```bash
# Start React + Vite development server
cd services/ptaas/web && npm run dev

# Build production version
cd services/ptaas/web && npm run build

# Preview production build
cd services/ptaas/web && npm run preview

# Linting
cd services/ptaas/web && npm run lint

# Run tests
cd services/ptaas/web && npm test

# Run tests with coverage
cd services/ptaas/web && npm run test:coverage
```

### Testing
```bash
# Activate virtual environment first
source venv/bin/activate

# Run all tests with pytest (from project root)
pytest

# Run specific test directories
pytest tests/unit/                    # Unit tests
pytest tests/integration/             # Integration tests
pytest tests/e2e/                     # End-to-end tests
pytest tests/security/                # Security tests

# Run specific test file
pytest tests/unit/test_auth.py

# Run tests by category using markers
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m e2e           # End-to-end tests only
pytest -m security      # Security tests only

# Environment validation script (checks dependencies, config)
python tools/scripts/validate_environment.py

# Note: Coverage currently disabled due to configuration issues
# To re-enable: uncomment coverage lines in pytest.ini
```

### Infrastructure & Deployment
```bash
# Deploy using infrastructure automation
cd services/infrastructure && python infrastructure_automation.py

# Deploy microservices
cd services/infrastructure && python microservices_deployment.py

# Check deployment status
curl http://localhost:8000/api/v1/health
```

## Architecture Overview

### Repository Structure
The repository follows enterprise microservices architecture with clear service boundaries:

- **src/** - Main application source code
  - **api/** - FastAPI REST API with clean architecture (main entry at `app/main.py`)
  - **orchestrator/** - Temporal workflow orchestration service
  - **xorb/** - Core platform modules and services
  - **common/** - Shared utilities, encryption, and configurations
  - **services/worker/** - Background worker service for job execution
- **services/** - Microservices architecture
  - **ptaas/** - PTaaS Frontend Service (React + TypeScript web interface)
  - **xorb-core/** - XORB Backend Platform (FastAPI, orchestration, AI services)
  - **infrastructure/** - Shared infrastructure services (monitoring, vault, databases)
- **packages/** - Shared libraries and configurations
  - **common/** - Shared utilities, encryption, and configurations
  - **types/** - TypeScript/Python type definitions
  - **configs/** - Configuration templates and environment settings
- **tools/** - Development and operations tools
  - **scripts/** - Deployment and automation scripts
  - **utilities/** - Core utilities and operational scripts
- **tests/** - Comprehensive test suite (unit, integration, e2e, security)
- **docs/** - Consolidated documentation organized by purpose
- **legacy/** - Deprecated code and legacy structures (safely preserved)

### Core Services Architecture
The platform follows clean architecture principles with **PRODUCTION-READY PTaaS implementation**:

1. **API Service** (`src/api/`)
   - **Main Application** - FastAPI app in `app/main.py` with comprehensive middleware stack
   - **PTaaS Router** - Production-ready penetration testing endpoints (`app/routers/ptaas.py`)
   - **PTaaS Orchestration** - Advanced workflow automation (`app/routers/ptaas_orchestration.py`)
   - **Scanner Service** - Real-world security tool integration (`app/services/ptaas_scanner_service.py`)
   - **Services** - Business logic in `app/services/` (auth, embedding, discovery, tenant)
   - **Repositories** - Data access abstraction in `app/infrastructure/repositories.py`
   - **Domain** - Entities and business rules in `app/domain/`
   - **Dependency Injection** - Managed via `app/container.py`
   - **Middleware** - Advanced rate limiting, audit logging, security, tenant context
   - **Routers** - API endpoints for auth, discovery, agents, orchestration, telemetry

2. **PTaaS Service** (`ptaas/`)
   - **Real-World Scanner Integration** - Nmap, Nuclei, Nikto, SSLScan production integration
   - **Behavioral Analytics** - ML-powered user behavior analysis with sklearn support
   - **Threat Hunting Engine** - Custom query language for threat investigations
   - **Forensics Engine** - Legal-grade evidence collection with chain of custody
   - **Network Microsegmentation** - Zero-trust network policy engine

3. **Orchestrator Service** (`src/orchestrator/`)
   - **Main Loop** - Autonomous orchestrator in `main.py` with circuit breaker pattern
   - **Temporal Integration** - Workflow client and execution engine
   - **Error Handling** - Exponential backoff, circuit breaker, retry policies
   - **Workflow Management** - Dynamic scan workflows with priority handling

4. **Worker Service** (`src/services/worker/`)
   - **Background Jobs** - Asynchronous task execution
   - **Temporal Activities** - Workflow activity implementations
   - **Job Queue Integration** - Redis-backed job processing

5. **Core Platform** (`src/xorb/`)
   - **Intelligence Engine** - Threat intelligence, ML models, vulnerability correlation
   - **Execution Engine** - Security scanning, stealth web engine, pristine core
   - **Core Platform** - Service mesh, authentication, rate limiting
   - **Security** - Zero trust, monitoring, chaos testing, CI integration

### Frontend Architecture (React + Vite)
- **React 18.3.1** with TypeScript 5.5.3
- **Vite 5.4.1** for fast development and build tooling
- **Tailwind CSS 3.4.11** for styling with custom components
- **Radix UI** components for accessible UI primitives
- **React Query (@tanstack/react-query)** for server state management
- **React Router DOM 6.26.2** for client-side routing
- **React Hook Form 7.53.0** with Zod validation
- **Recharts 2.12.7** for data visualization
- **Framer Motion 12.23.12** for animations
- **Jest 30.0.5** with Testing Library for unit testing

### Key Technologies
- **Backend**: FastAPI 0.117.1, Temporal 1.6.0, AsyncPG 0.30.0, Redis 5.1.0, Prometheus Client
- **Security Tools**: Nmap, Nuclei, Nikto, SSLScan, Dirb, Gobuster (production integrated)
- **Frontend**: React 18.3.1, Vite 5.4.1, TypeScript 5.5.3, Tailwind CSS 3.4.11
- **Database**: PostgreSQL with pgvector extension (ankane/pgvector:v0.5.1)
- **Infrastructure**: Docker, Docker Compose, Terraform
- **Monitoring**: Prometheus, Grafana
- **Security**: Advanced rate limiting, audit logging, MFA, API security middleware
- **Testing**: pytest with 80% coverage requirement, Jest for frontend

### Service Communication
- REST APIs for external communication
- Temporal workflows for complex orchestration
- Redis for caching and session management
- PostgreSQL for persistent data storage
- Prometheus for metrics collection

## PTaaS Production Implementation

### Security Scanner Integration
The platform now includes **PRODUCTION-READY** real-world security scanner integration:

#### Available Security Tools
- **Nmap**: Network discovery, port scanning, service detection, OS fingerprinting
- **Nuclei**: Modern vulnerability scanner with 3000+ templates
- **Nikto**: Web application security scanner
- **SSLScan**: SSL/TLS configuration analysis
- **Dirb/Gobuster**: Directory and file discovery
- **Custom Security Checks**: Advanced vulnerability analysis

#### PTaaS API Endpoints
```bash
# Create comprehensive security scan
curl -X POST "http://localhost:8000/api/v1/ptaas/sessions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer TOKEN" \
  -d '{
    "targets": [{
      "host": "scanme.nmap.org",
      "ports": [22, 80, 443],
      "scan_profile": "comprehensive"
    }],
    "scan_type": "comprehensive"
  }'

# Check scan status
curl "http://localhost:8000/api/v1/ptaas/sessions/{session_id}" \
  -H "Authorization: Bearer TOKEN"

# Get available scan profiles
curl "http://localhost:8000/api/v1/ptaas/profiles" \
  -H "Authorization: Bearer TOKEN"
```

#### Advanced Orchestration
```bash
# Create automated workflow
curl -X POST "http://localhost:8000/api/v1/ptaas/orchestration/workflows" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Weekly Security Scan",
    "targets": ["*.company.com"],
    "triggers": [{"trigger_type": "scheduled", "schedule": "0 2 * * 1"}]
  }'

# Compliance scanning
curl -X POST "http://localhost:8000/api/v1/ptaas/orchestration/compliance-scan" \
  -H "Content-Type: application/json" \
  -d '{
    "compliance_framework": "PCI-DSS",
    "targets": ["web.company.com"]
  }'

# Threat simulation
curl -X POST "http://localhost:8000/api/v1/ptaas/orchestration/threat-simulation" \
  -H "Content-Type: application/json" \
  -d '{
    "simulation_type": "apt_simulation",
    "attack_vectors": ["spear_phishing", "lateral_movement"]
  }'
```

### Scan Profiles Available
- **Quick** (5 min): Fast network scan with basic service detection
- **Comprehensive** (30 min): Full security assessment with vulnerability scanning
- **Stealth** (60 min): Low-profile scanning to avoid detection
- **Web-Focused** (20 min): Specialized web application security testing

### Compliance Frameworks
- **PCI-DSS**: Payment Card Industry compliance
- **HIPAA**: Healthcare data protection
- **SOX**: Sarbanes-Oxley compliance
- **ISO-27001**: Information security management
- **GDPR**: General Data Protection Regulation
- **NIST**: National Institute of Standards

## Development Patterns

### Clean Architecture Implementation
- Controllers handle HTTP requests and delegate to services
- Services contain business logic and coordinate between repositories
- Repositories abstract data access
- Domain entities define business rules
- Dependency injection via `container.py`

### Database Access
- Use async patterns with AsyncPG 0.30.0
- Repository pattern for data access abstraction
- Database migrations via Alembic (configured in `alembic.ini`)
- PostgreSQL with pgvector extension for vector operations
- Redis for caching and session management

### API Development Workflow
1. Add routes in `app/routers/` (e.g., `ptaas.py`, `ptaas_orchestration.py`)
2. Implement controllers in `app/controllers/` that handle HTTP requests
3. Create business logic in `app/services/` with interface-based design
4. Add repository methods in `app/infrastructure/repositories.py` if data access needed
5. Register dependencies in `app/container.py` using dependency injection
6. Add middleware in `app/middleware/` for cross-cutting concerns
7. Update `app/main.py` to include new routers with proper error handling

### Dependency Injection Pattern
- All services are registered in `app/container.py`
- Use interfaces defined in `app/services/interfaces.py`
- Singleton pattern for stateful services (repositories, caches)
- Transient pattern for stateless services
- Override capabilities for testing

### Frontend Development Workflow
- **Component Structure** - Components in `services/ptaas/web/src/components/` with UI components in `ui/` subdirectory
- **Page Routing** - Pages in `services/ptaas/web/src/pages/` using React Router DOM
- **State Management** - React Query for server state, custom hooks in `hooks/` directory
- **Styling** - Tailwind CSS with custom utility classes in `lib/utils.ts`
- **Form Handling** - React Hook Form with Zod schema validation
- **TypeScript** - Strict type checking with comprehensive type definitions
- **Testing** - Jest + Testing Library with coverage reports
- **Build Process** - Vite for fast development and optimized production builds

### Environment Configuration
Key environment variables:
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string with password
- `TEMPORAL_HOST` - Temporal server address (default: temporal:7233)
- `LOG_LEVEL` - Application logging level
- `ENVIRONMENT` - Deployment environment (dev/staging/prod)
- `JWT_SECRET` - Secret key for JWT token generation
- `NVIDIA_API_KEY` - NVIDIA API key for AI services
- `OPENROUTER_API_KEY` - OpenRouter API key for LLM services
- `RATE_LIMIT_PER_MINUTE` - API rate limiting (default: 60)
- `RATE_LIMIT_PER_HOUR` - API rate limiting (default: 1000)
- `CORS_ALLOW_ORIGINS` - Allowed CORS origins (comma-separated)
- `ENABLE_METRICS` - Enable Prometheus metrics (true/false)

### Vault Secret Management
XORB includes comprehensive HashiCorp Vault integration for secure secret management:

**Vault Infrastructure** (Located in `infra/vault/`):
- `vault-config.hcl` - Production Vault configuration with KV, database, transit engines
- `vault-dev-config.hcl` - Development Vault configuration
- `init-vault.sh` - Production initialization script with policies and roles
- `setup-vault-dev.sh` - Development setup with auto-generated secrets

**Vault Client** (`src/common/vault_client.py`):
- Production-ready client with AppRole authentication and token fallback
- Automatic environment variable fallback when Vault unavailable
- Dynamic database credential management with PostgreSQL integration
- JWT signing and encryption via Vault transit engine
- Secret caching with configurable TTL and version management
- Health monitoring and connection status checking

**Vault Management CLI** (`src/common/vault_manager.py`):
```bash
# Health check and connection test
python3 src/common/vault_manager.py health

# List and retrieve secrets (with masking)
python3 src/common/vault_manager.py list-secrets
python3 src/common/vault_manager.py get-secret xorb/config

# Rotate JWT signing keys in transit engine
python3 src/common/vault_manager.py rotate-jwt-key

# Development secret backup (dev environment only)
python3 src/common/vault_manager.py backup secrets-backup.json

# Comprehensive integration test
python3 src/common/vault_manager.py test
```

**Vault Secret Structure**:
- `secret/xorb/config` - JWT secrets, database config, XORB API keys
- `secret/xorb/external` - Third-party API keys (NVIDIA, OpenRouter, Azure, Google, GitHub)
- `database/creds/xorb-app` - Dynamic database credentials with TTL
- `transit/jwt-signing` - JWT signing and encryption key with rotation support

**Development Setup**:
```bash
# Initialize development Vault with secrets
cd infra/vault && ./setup-vault-dev.sh

# Access Vault UI (development)
open http://127.0.0.1:8200/ui
```

### CI/CD Pipeline & DevSecOps
XORB includes a comprehensive DevSecOps pipeline with multiple security scanning stages:

**Pipeline Components**:
- **Pre-commit Security Checks** - Fast secret scanning, license compliance, commit message validation
- **Static Application Security Testing (SAST)** - Bandit, Semgrep for code analysis
- **Dependency Vulnerability Scanning** - Safety, FOSSA for supply chain security
- **Container Security Scanning** - Trivy, Grype, Dockle for container vulnerabilities
- **Dynamic Application Security Testing (DAST)** - OWASP ZAP for runtime security testing
- **Infrastructure as Code Security** - Checkov, Hadolint for infrastructure scanning
- **Compliance and Policy Enforcement** - Security policy gates and compliance reporting

**GitHub Actions Workflows**:
```bash
# Main CI workflow
.github/workflows/ci.yml                    # Basic CI/CD pipeline

# Comprehensive security pipeline
.github/workflows/security-scan.yml         # Multi-stage security scanning
.github/workflows/devsecops-pipeline.yml    # Full DevSecOps workflow
.github/workflows/infrastructure-security.yml # Infrastructure security scanning
```

**Security Scanning Tools**:
```bash
# Run comprehensive security scan locally
./tools/scripts/security-scan.sh

# Run specific security scans
./tools/scripts/security-scan.sh secrets      # Secret detection
./tools/scripts/security-scan.sh sast         # Static analysis
./tools/scripts/security-scan.sh dependencies # Dependency vulnerabilities
./tools/scripts/security-scan.sh container    # Container security
./tools/scripts/security-scan.sh infrastructure # Infrastructure scanning

# Install security tools
./tools/scripts/security-scan.sh install

# Clean old reports
./tools/scripts/security-scan.sh clean
```

**Pre-commit Security Hooks**:
```bash
# Install pre-commit hooks (includes security scanning)
cd src/api && pre-commit install

# Run pre-commit hooks manually
pre-commit run --all-files

# Security tools included in pre-commit:
# - Bandit (Python security linting)
# - detect-secrets (Secret detection)
# - Checkov (Infrastructure security)
# - Hadolint (Dockerfile security)
# - Safety (Dependency vulnerabilities)
```

**Security Configuration Files**:
- `src/api/.gitleaks.toml` - Secret detection configuration
- `src/api/.pre-commit-config.yaml` - Pre-commit security hooks
- `.zap/rules.tsv` - OWASP ZAP scanning rules
- `.yamllint.yaml` - YAML linting and validation
- `src/api/Dockerfile.secure` - Hardened production container

### Production Monitoring Stack
XORB includes a comprehensive monitoring and observability stack:

**Monitoring Components**:
- **Prometheus**: Metrics collection and time-series database
- **Grafana**: Visualization dashboards and alerting
- **AlertManager**: Alert routing and notification management
- **Node Exporter**: System metrics (CPU, memory, disk, network)
- **cAdvisor**: Container metrics and resource usage
- **Blackbox Exporter**: Endpoint health monitoring
- **Database Exporters**: PostgreSQL and Redis metrics

**Monitoring Setup**:
```bash
# Setup complete monitoring stack
./tools/scripts/setup-monitoring.sh

# Individual management commands
./tools/scripts/setup-monitoring.sh start     # Start monitoring
./tools/scripts/setup-monitoring.sh stop      # Stop monitoring
./tools/scripts/setup-monitoring.sh restart   # Restart services
./tools/scripts/setup-monitoring.sh logs      # View logs
./tools/scripts/setup-monitoring.sh status    # Check status

# Using Docker Compose directly
docker-compose -f docker-compose.monitoring.yml up -d
```

**Configuration Files**:
- `infra/monitoring/prometheus.yml` - Prometheus scrape configuration
- `infra/monitoring/prometheus-rules.yml` - Alert rules for all services
- `infra/monitoring/alertmanager.yml` - Alert routing and notification rules
- `infra/monitoring/grafana/` - Dashboard provisioning and data sources
- `docker-compose.monitoring.yml` - Complete monitoring stack deployment

### Access Points
- **API Documentation**: http://localhost:8000/docs (FastAPI auto-generated with custom styling)
- **API Health Check**: http://localhost:8000/api/v1/health
- **API Readiness Check**: http://localhost:8000/api/v1/readiness
- **Platform Info**: http://localhost:8000/api/v1/info
- **PTaaS API**: http://localhost:8000/api/v1/ptaas
- **Frontend Application**: http://localhost:3000 (PTaaS React app)
- **Temporal Web UI**: http://localhost:8233

**Monitoring Access Points** (when monitoring stack is running):
- **Prometheus**: http://localhost:9092 - Metrics and queries
- **Grafana**: http://localhost:3010 - Dashboards and visualization (admin / SecureAdminPass123!)
- **AlertManager**: http://localhost:9093 - Alert management
- **Node Exporter**: http://localhost:9100 - System metrics
- **cAdvisor**: http://localhost:8083 - Container metrics
- **Blackbox Exporter**: http://localhost:9115 - Endpoint monitoring

## Current Status & Known Issues

### ✅ Working Components
- **FastAPI Application**: Successfully imports and starts with all routers including PTaaS
- **PTaaS Production Implementation**: Real-world security scanner integration complete
- **Security Scanner Service**: Nmap, Nuclei, Nikto, SSLScan integration working
- **PTaaS Orchestration**: Advanced workflow automation and compliance scanning
- **Behavioral Analytics**: ML-powered analysis with graceful sklearn fallbacks
- **Threat Hunting Engine**: Custom query language and real-time analysis
- **Forensics Engine**: Legal-grade evidence collection with chain of custody
- **Virtual Environment**: Properly configured with required dependencies
- **Docker Compose**: Both development and production configs validated
- **Test Framework**: pytest runs with test discovery working
- **Environment Validation**: Comprehensive validation script available

### ⚠️ Known Issues & Limitations
- **Security Tool Dependencies**: Some scanners may not be installed on all systems
- **OpenTelemetry**: Partially available - some instrumentation packages missing (optional)
- **Test Coverage**: HTML reports disabled due to coverage package file issues
- **Async Test Fixtures**: Need to use `@pytest_asyncio.fixture` for async fixtures
- **Missing Dependencies**: `authlib` installed but some router imports still fail (non-critical)

### 🚀 Quick Start (Validated Commands)
```bash
# 1. Setup environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.lock  # or pip install -e .

# 2. Validate setup
python tools/scripts/validate_environment.py

# 3. Start API server (PRODUCTION-READY)
cd src/api && uvicorn app.main:app --reload --port 8000

# 4. Test PTaaS API
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/ptaas/profiles
```

## Development Guidelines

### Middleware Stack (FastAPI)
The API service uses a carefully ordered middleware stack in `app/main.py`:
1. **GlobalErrorHandler** (outermost) - Comprehensive error handling and logging
2. **APISecurityMiddleware** - Security headers, request validation
3. **AdvancedRateLimitingMiddleware** - Redis-backed rate limiting with tenant support
4. **TenantContextMiddleware** - Multi-tenant request context
5. **RequestLoggingMiddleware** - Structured request/response logging
6. **PerformanceMiddleware** - Performance monitoring and metrics
7. **AuditLoggingMiddleware** - Security audit trail
8. **GZipMiddleware** - Response compression
9. **RequestIdMiddleware** (innermost) - Unique request tracking

### Error Handling Patterns

#### Orchestrator Service (Circuit Breaker)
- **Circuit Breaker Pattern** in `src/orchestrator/main.py`
- **Exponential Backoff** with configurable delays
- **Error Threshold** monitoring (5 errors in 60 seconds trips breaker)
- **Automatic Recovery** after error window expires
- **Workflow Retry Policies** with different priorities (high/medium/low)

#### API Service (Graceful Degradation)
- **Health Checks** with dependency validation (Redis, PostgreSQL, Temporal)
- **Readiness Probes** for Kubernetes deployment
- **Graceful Shutdown** handling in lifespan management
- **Service Degradation** when dependencies are unavailable

### Code Quality Standards
- **Clean Architecture** with clear separation of concerns
- **Dependency Injection** for testability and flexibility
- **Async/Await** patterns for all I/O operations
- **Type Hints** required for all Python code
- **Interface-Based Design** for service abstractions
- **Repository Pattern** for data access abstraction

### Security Considerations
- **Never commit secrets** - Use environment variables and secret management
- **Rate Limiting** with Redis backing and tenant isolation
- **Security Middleware** with comprehensive header protection
- **Input Validation** using Pydantic models and Zod schemas
- **Authentication/Authorization** with JWT tokens and role-based access
- **Audit Logging** for all security-sensitive operations

### Testing Strategy
- **Unit Tests** - pytest with 80% minimum coverage requirement
- **Integration Tests** - API endpoint and service interaction testing
- **E2E Tests** - Complete workflow testing
- **Security Tests** - Vulnerability and penetration testing
- **Performance Tests** - Load testing and scalability validation
- **Frontend Tests** - Jest with React Testing Library
- **Test Markers** - Categorized tests (unit, integration, e2e, security, performance)
