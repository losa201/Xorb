# XORB Core Platform

## Overview
XORB Core provides the backend platform and services that power the entire cybersecurity ecosystem. It includes API gateways, orchestration engines, threat intelligence, and security services.

## Service Architecture
```
xorb-core/
├── api/                    # Main API Gateway (Port 8000)
│   ├── app/                # FastAPI application
│   ├── routers/            # API endpoints
│   ├── services/           # Business logic
│   ├── middleware/         # Security & monitoring
│   └── infrastructure/     # Data access layer
├── orchestrator/           # Temporal Workflow Engine
│   ├── core/               # Workflow definitions
│   ├── executors/          # Task executors
│   └── api/                # Workflow API
├── intelligence/           # Threat Intelligence Engine
│   ├── llm_integration.py  # AI/ML services
│   ├── threat_intelligence_engine.py
│   └── vulnerability_correlation_engine.py
└── security/               # Security Services
    ├── monitoring.py       # Security monitoring
    ├── zero_trust.py       # Zero trust architecture
    └── audit.py            # Security auditing
```

## Key Features
- **FastAPI Gateway**: Production-ready API with comprehensive middleware
- **Temporal Orchestration**: Workflow engine for complex security operations
- **Threat Intelligence**: Real-time threat detection and correlation
- **Zero Trust Security**: Complete security framework implementation
- **Multi-tenant Support**: Enterprise-grade tenant isolation

## Development
```bash
# API Service
cd services/xorb-core/api
source ../../../venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Orchestrator Service
cd services/xorb-core/orchestrator
python main.py
```

## Database Dependencies
- **PostgreSQL**: Primary database with pgvector extensions
- **Redis**: Session management and caching
- **Temporal**: Workflow persistence and execution