# XORB 2.0 - Refactored Architecture

> **AI-Powered Cybersecurity Intelligence Platform** - Production-Ready, Scalable, Secure

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)

## 🏗️ Refactored Architecture & Domain-Driven Design

XORB 2.0 has been completely refactored for production readiness with clean domain separation, enterprise security, and scalable microservices architecture. The new domain-driven design ensures maintainability, modularity, and long-term scalability.

### 📁 New Directory Structure

```
Xorb/
├── domains/                    # Clean domain separation
│   ├── agents/                # Agent framework and implementations
│   ├── orchestration/         # Campaign orchestration and scheduling
│   ├── core/                  # Core business logic and configuration
│   ├── security/              # Security and compliance modules
│   ├── llm/                   # Language model integrations
│   ├── utils/                 # Shared utilities and helpers
│   └── infra/                 # Infrastructure and database management
├── services/                  # Microservices
│   ├── api/                   # FastAPI REST interface
│   ├── worker/                # Temporal workflow workers
│   └── orchestrator/          # Campaign management service
├── legacy/                    # Legacy demos and phase files
│   ├── demos/                 # Demo scripts and prototypes
│   ├── phase_files/           # Historical phase documentation
│   └── standalone_scripts/    # Standalone utility scripts
├── gitops/                    # Kubernetes deployment manifests
├── monitoring/                # Observability stack
└── tests/                     # Comprehensive test suite
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+**
- **Docker & Docker Compose**
- **Poetry** (recommended) or pip
- **16+ CPU cores** (64+ recommended for EPYC optimization)
- **32GB+ RAM** (128GB+ recommended)

### Development Setup

```bash
# Clone and setup
git clone https://github.com/losa201/Xorb.git
cd Xorb

# Install dependencies and setup environment
make setup

# Start development environment
make dev

# Run quality checks
make quality
```

### Production Deployment

```bash
# Production deployment with full security scanning
make deploy-prod

# Or using Docker Compose
docker-compose -f docker-compose.production.yml up -d

# Verify deployment
make k8s-status
```

## 🎯 Key Features

### ✨ Refactoring Improvements

- **🏗️ Domain-Driven Design**: Clean separation of concerns across domains
- **🔒 Security Hardening**: All secrets externalized, no hardcoded credentials
- **⚡ Async Optimization**: High-performance async operations throughout
- **📊 Comprehensive Testing**: 70%+ test coverage with integration tests
- **🛡️ Security Scanning**: Automated vulnerability detection and mitigation
- **📈 Observability**: Full monitoring with Prometheus/Grafana
- **🔧 DevOps Ready**: Complete CI/CD pipeline with GitOps

### 🤖 Core Capabilities

- **32-Agent Swarm Intelligence**: Concurrent AI-powered security testing
- **Multi-LLM Integration**: OpenRouter, OpenAI, Anthropic support
- **Knowledge Fabric**: Vector-based intelligence with graph relationships
- **Autonomous Orchestration**: Self-managing campaign execution
- **Real-time Monitoring**: Live dashboards and alerting
- **EPYC Optimization**: AMD EPYC processor optimizations

## 📖 Development Workflow

### Available Commands

```bash
# Development
make setup              # Initial environment setup
make dev               # Start development environment
make clean             # Clean build artifacts

# Code Quality
make format            # Format code with black/isort
make lint              # Run linting (ruff, mypy, bandit)
make test              # Run tests with coverage
make test-fast         # Run tests without coverage
make security-scan     # Comprehensive security scanning
make quality           # Run all quality checks

# Services
make up                # Start services
make down              # Stop services
make logs              # View service logs
make shell             # Open shell in API container

# Deployment
make k8s-apply         # Apply Kubernetes manifests
make k8s-status        # Show deployment status
make benchmark         # Run performance benchmarks

# Utilities
make agent-discovery   # Test agent discovery
make db-migrate        # Run database migrations
make monitor           # Start monitoring stack
```

### Code Quality Standards

- **Formatting**: Black with 88-character line length
- **Linting**: Ruff for fast Python linting
- **Type Checking**: MyPy with strict configuration
- **Security**: Bandit for security vulnerability scanning
- **Testing**: Pytest with 70%+ coverage requirement
- **Pre-commit**: Automated checks on every commit

## 🏛️ Architecture Domains

### 🤖 Agents Domain (`domains/agents/`)

Autonomous AI agents for cybersecurity operations:

- **Agent Registry**: Centralized discovery and management
- **Base Agent Framework**: Common interface and capabilities
- **Specialized Agents**: Threat hunting, vulnerability assessment, etc.
- **Stealth Operations**: Advanced evasion techniques

### 🎭 Orchestration Domain (`domains/orchestration/`)

Campaign management and agent coordination:

- **Enhanced Orchestrator**: Multi-armed bandit scheduling
- **EPYC Optimization**: AMD processor-specific optimizations
- **Dynamic Resource Management**: Intelligent resource allocation
- **Event-Driven Architecture**: CloudEvents integration

### 🧠 Core Domain (`domains/core/`)

Central business logic and configuration:

- **Configuration Management**: Environment-based config
- **Domain Models**: Core entities and data structures
- **Exception Handling**: Comprehensive error management
- **Shared Enums**: Common enumerations

### 🔒 Security Domain (`domains/security/`)

Security and compliance framework:

- **JWT Authentication**: Secure token management
- **Rules of Engagement**: Automated compliance checking
- **Secrets Management**: Secure credential handling
- **Security Hardening**: Production security measures

### 🧠 LLM Domain (`domains/llm/`)

Language model integrations:

- **Multi-Provider Client**: OpenRouter, OpenAI, Anthropic
- **Creative Payload Generation**: AI-powered attack simulation
- **Intelligent Caching**: Optimized response caching
- **Rate Limiting**: Provider-aware rate management

## 🐳 Deployment Options

### Docker Compose (Single Node)

```bash
# Development
docker-compose up -d

# Production
docker-compose -f docker-compose.production.yml up -d
```

### Kubernetes (Multi-Node)

```bash
# Apply manifests
kubectl apply -f gitops/helm/xorb-core/templates/

# Using Helm
helm install xorb ./gitops/helm/xorb-core
```

### GitOps with ArgoCD

```bash
# Deploy ApplicationSet
kubectl apply -f gitops/argocd/applicationset.yaml
```

## 📊 Monitoring & Observability

### Metrics (Prometheus)

- `xorb_agent_executions_total`: Agent execution counters
- `xorb_campaign_operations_total`: Campaign lifecycle metrics
- `xorb_discovered_agents_total`: Agent registry metrics
- `xorb_threats_detected_total`: Threat detection metrics

### Dashboards (Grafana)

- **XORB Overview**: Platform status and key metrics
- **Agent Performance**: Individual agent statistics
- **Campaign Analytics**: Campaign success rates and timing
- **Infrastructure Health**: System resource utilization

### Access Points

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **API Documentation**: http://localhost:8000/docs

## 🧪 Testing

### Test Structure

```
tests/
├── unit/                      # Unit tests for individual components
├── integration/               # Integration tests across services
├── e2e/                      # End-to-end workflow tests
└── performance/              # Performance and load tests
```

### Running Tests

```bash
# All tests with coverage
make test

# Fast tests without coverage
make test-fast

# Specific test file
pytest tests/unit/test_agents.py -v

# Performance benchmarks
make benchmark
```

## 🔧 Configuration

### Environment Variables

```bash
# Core Configuration
ENVIRONMENT=development|staging|production
DEBUG=true|false
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=xorb
POSTGRES_USER=xorb
POSTGRES_PASSWORD=secure_password

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=secure_password

# AI Configuration
OPENROUTER_API_KEY=your_api_key
NVIDIA_API_KEY=your_api_key
DEFAULT_LLM_MODEL=qwen/qwen-2.5-72b-instruct

# Security Configuration
JWT_SECRET_KEY=your_secret_key
ENCRYPTION_KEY=your_encryption_key

# Orchestration Configuration
MAX_CONCURRENT_AGENTS=32
AGENT_TIMEOUT=300
CAMPAIGN_TIMEOUT=3600
```

### Configuration Files

- `.env`: Local environment variables
- `domains/core/config.py`: Centralized configuration management
- `config/config.json`: JSON-based configuration (secrets externalized)

## 🛡️ Security Best Practices

### Implemented Security Measures

- ✅ **No Hardcoded Secrets**: All credentials externalized
- ✅ **Environment Variable Management**: Secure configuration
- ✅ **Security Scanning**: Automated vulnerability detection
- ✅ **Container Hardening**: Security-optimized Docker images
- ✅ **Network Security**: mTLS and network policies
- ✅ **Access Control**: RBAC and JWT authentication
- ✅ **Audit Logging**: Comprehensive security logging

### Security Scanning

```bash
# Run all security checks
make security-scan

# Individual scans
bandit -r domains/ services/          # Python security issues
safety check                         # Dependency vulnerabilities
semgrep --config=auto domains/       # Static analysis
trivy fs .                           # Container vulnerabilities
```

## 🎯 Performance Optimization

### EPYC Processor Optimization

- **High Concurrency**: 32+ concurrent agents
- **NUMA Awareness**: Memory locality optimization
- **CPU Affinity**: Core allocation strategies
- **Resource Quotas**: EPYC-specific resource limits

### Async Performance

- **Connection Pooling**: Optimized database connections
- **Batch Operations**: Efficient bulk processing
- **Circuit Breakers**: Fault tolerance patterns
- **Caching Strategies**: Multi-level caching

## 📚 Documentation

### Available Documentation

- [Deployment Guide](docs/deployment.md): Production deployment instructions
- [API Documentation](http://localhost:8000/docs): Interactive API docs
- [Monitoring Setup](docs/MONITORING_SETUP_GUIDE.md): Observability configuration
- [Disaster Recovery](docs/disaster-recovery-drillbook.md): DR procedures

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make changes following code quality standards
4. Run full quality checks: `make quality`
5. Submit a pull request

### Code Standards

- Follow the established domain architecture
- Maintain 70%+ test coverage
- Use type hints throughout
- Document all public APIs
- Follow security best practices

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Emergency Contacts

- **Platform Team**: platform-team@xorb.security
- **DevOps Team**: devops@xorb.security
- **Security Team**: security@xorb.security
- **On-Call**: +1-555-XORB-OPS

### Getting Help

1. Check the [documentation](docs/)
2. Search [existing issues](https://github.com/losa201/Xorb/issues)
3. Create a [new issue](https://github.com/losa201/Xorb/issues/new)
4. Contact the team via email

---

**XORB 2.0** - Built for enterprise security teams who demand reliability, scalability, and security.