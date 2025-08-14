# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start API service (main development server)
cd src/api && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start orchestrator service (workflow engine)
cd src/orchestrator && python main.py

# Alternative: Use simple versions
cd src/api && python simple_main.py
cd src/orchestrator && python simple_main.py
```

### Docker Development
```bash
# Full platform deployment
docker-compose -f deploy/configs/docker-compose.yml up -d

# Development environment
docker-compose -f deploy/configs/docker-compose.dev.yml up -d

# Build specific services
cd src/api && docker build -t xorb-api .
cd src/orchestrator && docker build -t xorb-orchestrator .
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m security

# Run tests in legacy directory
cd legacy && pytest
```

### Quick Deployment
```bash
# Quick start platform (recommended for development)
./tools/scripts/quick_start.sh development

# Production deployment
./tools/scripts/quick_start.sh production

# Platform status check
curl http://localhost:8000/health
```

## Architecture Overview

### Core Structure
- **src/api/** - FastAPI web service with clean architecture (controllers, services, repositories, domain)
- **src/orchestrator/** - Temporal-based workflow orchestration engine
- **src/xorb/** - Core platform modules (intelligence, execution, monitoring)
- **src/common/** - Shared utilities and models
- **legacy/** - Legacy services and artifacts (contains older implementations)

### Service Architecture
The platform follows Domain-Driven Design with:
- **Controllers** - HTTP request handlers
- **Services** - Business logic layer with dependency injection
- **Repositories** - Data access abstractions
- **Domain** - Entities, value objects, and business rules
- **Infrastructure** - External service implementations

### Key Services
- **API Gateway** - Main REST API (port 8000)
- **Orchestrator** - Workflow engine using Temporal
- **PostgreSQL** - Primary database (port 5432)
- **Redis** - Caching and session storage (port 6379)
- **Prometheus** - Metrics collection (port 9090)
- **Grafana** - Monitoring dashboards (port 3000)

## Key Dependencies
- **FastAPI** - Web framework
- **Temporal** - Workflow orchestration
- **PostgreSQL** - Database with AsyncPG driver
- **Redis** - Caching
- **Prometheus** - Monitoring
- **Docker** - Containerization

## Development Patterns

### Service Registration
Services use dependency injection through `container.py` - register new services there and inject via constructors.

### Database Access
Use async repositories pattern - extend base repository classes in `infrastructure/repositories.py`.

### API Development
Follow clean architecture:
1. Add routes in `routers/`
2. Implement controllers in `controllers/`
3. Create service logic in `services/`
4. Add repository methods if needed

### Workflow Development
Temporal workflows go in `src/orchestrator/` - use the existing patterns for activity registration.

## Environment Configuration
- `DATABASE_URL` - PostgreSQL connection
- `REDIS_URL` - Redis connection
- `TEMPORAL_HOST` - Temporal server (default: localhost:7233)
- `LOG_LEVEL` - Logging verbosity
- `ENVIRONMENT` - deployment environment

## Platform Access Points
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Monitoring: http://localhost:3000
- Metrics: http://localhost:9090
