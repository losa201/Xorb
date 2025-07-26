# XORB 2.0 - Refactored Repository Structure

## 🎯 Overview

This repository has been comprehensively refactored and organized for improved modularity, maintainability, and production readiness. The XORB ecosystem is now structured as a clean, scalable autonomous security intelligence platform.

## 📁 Repository Structure

```
Xorb/
├── xorb_core/                     # Core business logic (modular)
│   ├── agents/                    # All agent implementations
│   ├── orchestration/             # Campaign orchestration
│   ├── knowledge_fabric/          # Knowledge management
│   ├── security/                  # Security components
│   ├── llm/                      # LLM integrations
│   ├── utils/                    # Shared utilities
│   ├── infrastructure/           # Infrastructure components
│   └── common/                   # Common interfaces
├── services/                     # Microservices
│   ├── api/                     # FastAPI REST interface
│   ├── worker/                  # Temporal workflow workers
│   ├── orchestrator/            # Campaign management
│   └── ...                      # Additional services
├── docker/                      # Containerization
│   ├── api/Dockerfile          # API service container
│   ├── worker/Dockerfile       # Worker service container
│   └── ...                     # Service-specific containers
├── config/                      # Configuration files
│   ├── .xorb.env               # Environment template
│   ├── targets.json            # Target configurations
│   └── prometheus.yml          # Monitoring config
├── scripts/                     # Deployment & utility scripts
│   ├── launch/                 # Environment bootstrap scripts
│   └── ...                     # Additional scripts
├── monitoring/                  # Observability stack
│   ├── prometheus/             # Prometheus configuration
│   ├── grafana/               # Grafana dashboards
│   └── ...                    # Monitoring components
├── tests/                      # Testing framework
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── e2e/                   # End-to-end tests
├── docs/                       # Documentation
│   ├── phases/                # Phase documentation
│   └── ...                    # Additional docs
├── docker-compose.yml          # Unified stack deployment
├── Makefile.organized          # Build and deployment automation
└── bootstrap.sh               # Quick start script
```

## 🚀 Quick Start

### Environment Auto-Detection Bootstrap

```bash
# Auto-detect environment and start XORB
make bootstrap

# Or use the direct script
./scripts/launch/bootstrap.sh
```

### Manual Environment Setup

```bash
# Development environment
make dev

# Production deployment
make production-deploy

# Staging deployment
make staging-deploy
```

## 🔧 Available Commands

### Development
```bash
make setup          # Initial setup
make deps           # Install dependencies
make dev            # Start development environment
```

### Testing
```bash
make test           # Run all tests
make test-unit      # Unit tests only
make test-integration # Integration tests only
make test-e2e       # End-to-end tests
```

### Code Quality
```bash
make lint           # Code linting
make format         # Code formatting
make security-scan  # Security scanning
make quality        # All quality checks
```

### Docker Operations
```bash
make build          # Build images
make up             # Start services
make down           # Stop services
make restart        # Restart services
make logs           # View logs
```

### Monitoring
```bash
make monitor        # Start monitoring stack
make prometheus     # Open Prometheus
make grafana        # Open Grafana
```

## 🏗️ Architecture Changes

### Core Improvements

1. **Modular Structure**: All agents consolidated into `xorb_core/agents/`
2. **Clean Separation**: Infrastructure, utilities, and business logic separated
3. **Standardized Containers**: Docker files organized by service
4. **Unified Configuration**: Centralized config management
5. **Comprehensive Testing**: Structured test framework
6. **Environment Detection**: Auto-configuration for dev/staging/production

### Service Organization

- **API Service**: REST interface with hexagonal architecture
- **Worker Service**: Temporal workflow execution
- **Orchestrator**: Campaign management and agent coordination
- **Monitoring Stack**: Prometheus, Grafana, and alerting

### EPYC Optimization

The platform maintains EPYC processor optimization:
- High concurrency settings (32+ agents)
- NUMA awareness
- Memory-efficient caching
- Optimized resource allocation

## 🔒 Security Features

- mTLS between services
- RoE (Rules of Engagement) validation
- Audit logging
- Secret management
- Network policies

## 📊 Monitoring & Observability

- **Metrics**: Prometheus with custom XORB metrics
- **Dashboards**: Pre-configured Grafana dashboards
- **Logging**: Structured logging with centralized collection
- **Alerting**: Production-ready alert rules

## 🐳 Container Strategy

- **Layer Caching**: Optimized Dockerfile structure
- **Health Checks**: Built-in container health monitoring
- **Environment Variables**: Flexible configuration
- **Multi-stage Builds**: Efficient image construction

## 🌍 Environment Support

### Development
- Local Docker Compose
- Hot reloading
- Debug configurations

### Staging
- K3s cluster ready
- Reduced resource allocation
- Testing integrations

### Production
- Full Kubernetes support
- EPYC optimization
- High availability
- GitOps deployment

### Raspberry Pi
- ARM64 support
- Resource-constrained optimization
- Edge deployment ready

## 📝 Migration Notes

### Import Changes
- `xorb_common.*` → `xorb_core.*`
- Agents now in `xorb_core.agents.*`
- Infrastructure in `xorb_core.infrastructure.*`

### Configuration
- Environment configs moved to `config/`
- Docker configs in `docker/`
- Scripts in `scripts/launch/`

### Testing
- Tests organized by type in `tests/`
- Pytest configuration included
- CI/CD pipeline ready

## 🚦 Getting Started

1. **Clone and Setup**:
   ```bash
   git clone <repository>
   cd Xorb
   make setup
   ```

2. **Start Development**:
   ```bash
   make dev
   ```

3. **Run Tests**:
   ```bash
   make test
   ```

4. **Deploy Production**:
   ```bash
   make production-deploy
   ```

## 📚 Documentation

- **Architecture**: See `docs/` for detailed documentation
- **Phase Documentation**: Available in `docs/phases/`
- **API Documentation**: Auto-generated from FastAPI
- **Agent Registry**: See `xorb_core/agents/`

## 🔄 Continuous Integration

The refactored structure supports:
- GitHub Actions workflows
- Automated testing
- Security scanning
- Code quality checks
- Automated deployments

---

**Note**: This refactoring maintains full backward compatibility while providing a clean, scalable foundation for future development.