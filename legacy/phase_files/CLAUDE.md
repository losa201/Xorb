# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Xorb 2.0 is an AI-powered security intelligence platform built as a modern, resilient microservices architecture. It features reinforcement learning-enhanced orchestration, multi-armed bandit agent scheduling, and comprehensive knowledge fabric with vector storage capabilities. The platform runs as a single-tenant, on-premises solution optimized for AMD EPYC processors (64 cores/128 threads).

## Development Commands

### Environment Setup
```bash
make setup          # Initial development environment setup
make deps           # Install/update dependencies  
make dev            # Complete dev setup and start services
```

### Running Individual Tests
```bash
pytest tests/                              # Run all tests
pytest tests/test_specific_file.py         # Run specific test file
pytest tests/test_specific_file.py::test_function  # Run specific test function
pytest -k "test_pattern"                   # Run tests matching pattern
pytest --tb=short -v                       # Verbose output with short tracebacks
```

### Code Quality
```bash
make format         # Format code with black and isort
make lint           # Run flint8, mypy, and bandit
make test           # Run pytest with coverage
make test-fast      # Run tests without coverage (faster)
make security-scan  # Run security scanning
make quality        # Run all quality checks (format, lint, test, security)
```

### Local Development
```bash
make up             # Start development environment (docker-compose)
make down           # Stop development environment
make restart        # Restart development environment
make logs           # Show service logs
make shell          # Open shell in API container
make python-shell   # Open Python shell with project imports
```

### Kubernetes Operations
```bash
make k8s-apply      # Apply Kubernetes manifests (ENV=development|staging|production)
make k8s-status     # Show deployment status
make k8s-logs       # Show pod logs
make helm-install   # Install Helm chart
make helm-upgrade   # Upgrade Helm chart
make gitops-apply   # Apply ArgoCD ApplicationSet
```

### Testing & Performance
```bash
make load-test      # Run k6 load tests
make benchmark      # Run performance benchmarks
make agent-discovery # Test agent discovery system
```

## Architecture Overview

### Core Components

1. **Enhanced Orchestrator** (`packages/xorb_core/xorb_core/orchestration/enhanced_orchestrator.py`)
   - Dynamic agent discovery via entry points and plugin directories
   - Concurrent execution with asyncio and EPYC optimization (32 concurrent agents default)
   - CloudEvents integration for event-driven architecture
   - Prometheus metrics and structured logging
   - Multi-armed bandit scheduling for agent selection

2. **Knowledge Fabric** (`packages/xorb_core/xorb_core/knowledge_fabric/`)
   - Hot/warm storage architecture (Redis + SQLAlchemy)
   - Knowledge atoms with confidence scoring and ML prediction
   - Graph relationships and semantic search
   - Vector embeddings via Qdrant integration
   - Automated knowledge gap analysis and cleanup

3. **Agent Framework** (`packages/xorb_core/xorb_core/agents/`)
   - Base agent with capability-based discovery
   - Stealth agents with advanced evasion techniques
   - Multi-engine agents (playwright, selenium, requests)
   - Plugin-based extensibility

4. **LLM Integration** (`packages/xorb_core/xorb_core/llm/`)
   - Multi-provider client with OpenRouter.ai gateway
   - Creative payload generation with Qwen security specialist
   - Hybrid client with fallback mechanisms
   - Intelligent caching and rate limiting

### Services Architecture

- **API Service** (`services/api/`): FastAPI-based REST interface
- **Worker Service** (`services/worker/`): Temporal workflow execution
- **Orchestrator Service** (`services/orchestrator/`): Campaign management and agent coordination

### Data Persistence

- **PostgreSQL**: Primary structured data store with PGvector for embedings
- **Neo4j**: Graph database for relationship intelligence
- **Qdrant**: Vector database for semantic search and embeddings
- **Redis**: Hot cache and session storage
- **ClickHouse**: Experience store for RL feedback loops

### Message/Event Streaming

- **NATS JetStream**: CloudEvents messaging for event-driven architecture
- **Temporal**: Workflow orchestration and durable execution

## EPYC Optimization Features

The platform is specifically tuned for AMD EPYC 7702 (64 cores/128 threads):

- High concurrency settings (32+ concurrent agents)
- CPU affinity and NUMA awareness
- Memory-efficient caching strategies
- Optimized database configurations
- Resource quotas configured for EPYC capacity

## GitOps & Deployment

### Structure
- `gitops/helm/`: Helm charts with EPYC-optimized values
- `gitops/overlays/`: Kustomize overlays for different environments
- `gitops/argocd/`: ArgoCD ApplicationSets for GitOps workflow
- `gitops/monitoring/`: Prometheus rules and Grafana dashboards

### Environments
- **Development**: Local docker-compose with hot reloading
- **Staging**: K3s cluster with reduced resources
- **Production**: Full K8s cluster with EPYC optimization and HA

## Monitoring & Observability

### Metrics (Prometheus)
- `xorb_agent_executions_total`: Agent execution counters
- `xorb_campaign_operations_total`: Campaign lifecycle metrics  
- `xorb_agent_execution_duration_seconds`: Execution time histograms
- `xorb_discovered_agents_total`: Agent registry metrics

### Service Mesh (Linkerd)
- mTLS between all services
- Traffic policies and service profiles
- Circuit breaking and retry policies

### Logging
- Structured logging with structured
- Centralized via Linkerd and Prometheus

## Development Patterns

### Agent Development
1. Inherit from `BaseAgent` class
2. Define capabilities and resource requirements
3. Implement `execute()` method with proper error handling
4. Register via entry points or place in plugin directory
5. Test with `make agent-discovery`

### Knowledge Integration
1. Create `KnowledgeAtom` instances with confidence scores
2. Use appropriate `AtomType` and tags for categorization
3. Leverage ML predictor for confidence adjustment
4. Establish relationships between related atoms

### Campaign Creation
1. Define targets with proper RoE validation
2. Specify agent requirements by capability
3. Configure timeouts and retry policies
4. Monitor via orchestrator APIs and metrics

## Security Considerations

- All secrets managed via Kubernetes secrets
- mTLS encryption for inter-service communication
- RoE (Rules of Engagement) validation for all targets
- Audit logging for compliance requirements
- Network policies for pod-to-pod communication
- RBAC for service accounts and user access

## Common Debugging

### Service Issues
- Check `make k8s-status` for pod states
- View logs with `make k8s-logs` or `make logs`
- Verify service mesh with Linkerd dashboard
- Check metrics at Prometheus endpoints

### Agent Problems
- Test discovery with `make agent-discovery`
- Verify agent registration in orchestrator logs
- Check capability matching in campaign creation
- Monitor execution metrics and error rates

### Performance Issues
- Review EPYC-specific resource allocations
- Check HPA scaling thresholds and metrics
- Monitor database query performance
- Analyze hot/warm cache hit rates in knowledge fabric

## Important Notes

- Never commit secrets or credentials to the repository
- All database migrations handled via SQLAlchemy/Alembic
- Use structured logging for all new components
- Follow capability-based agent selection patterns
- Implement proper graceful shutdown for all services
- Test EPYC optimization settings in staging before production

## Project Structure

```
Xorb/
├── packages/xorb_core/              # Core Python package with business logic
│   └── xorb_core/
│       ├── agents/                  # Agent framework and implementations
│       ├── ai/                      # AI integrations (Cerebras, etc.)
│       ├── knowledge_fabric/        # Knowledge management and ML
│       ├── llm/                     # Language model integrations
│       ├── orchestration/           # Campaign orchestration and scheduling
│       └── security/                # Security and compliance modules
├── services/                        # Microservices
│   ├── api/                        # FastAPI REST interface
│   ├── worker/                     # Temporal workflow workers
│   └── orchestrator/               # Campaign management service
├── gitops/                         # Kubernetes deployment manifests
│   ├── helm/                       # Helm charts for services
│   ├── overlays/                   # Kustomize overlays per environment
│   ├── monitoring/                 # Prometheus/Grafana configs
│   └── security/                   # Security policies and constraints
└── docker-compose*.yml            # Local development environments
```

## Development Tools Integration

The project uses:
- **Poetry/pip**: Dependency management (see requirements.txt)
- **Ruff**: Fast Python linter and formatter (config in pyproject.toml)
- **Pytest**: Testing framework with asyncio support
- **Docker Compose**: Local development environment
- **Temporal**: Workflow orchestration for long-running tasks
- **Kubernetes**: Production deployment via GitOps