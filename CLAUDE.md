# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Development Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies (Next.js)
cd ptaas-frontend && npm install

# Start main API service
cd src/api && uvicorn main.py:app --reload --host 0.0.0.0 --port 8000

# Alternative: Start with enhanced main
cd src/api && python enhanced_main.py

# Start orchestrator service
cd src/orchestrator && python main.py

# Simple versions for testing
cd src/api && python simple_main.py
cd src/orchestrator && python simple_main.py
```

### Docker Development
```bash
# Full platform with all services
docker-compose -f infra/docker-compose.yml up -d

# Development environment
docker-compose -f infra/docker-compose.production.yml up -d

# Quick start with core services
docker-compose -f deploy/configs/docker-compose.yml up -d

# View logs
docker-compose logs -f [service_name]
```

### Frontend Development
```bash
# Start Next.js development server
cd ptaas-frontend && npm run dev

# Build production version
cd ptaas-frontend && npm run build

# Start production server
cd ptaas-frontend && npm start

# Linting
cd ptaas-frontend && npm run lint
```

### Testing
```bash
# Run all tests (excludes legacy directory)
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_file.py

# Run legacy tests separately
cd legacy && pytest
```

### Infrastructure & Deployment
```bash
# Deploy using infrastructure automation
cd infra && python infrastructure_automation.py

# Deploy microservices
cd infra && python microservices_deployment.py

# Check deployment status
curl http://localhost:8000/health
```

## Architecture Overview

### Repository Structure
- **src/** - Main application source code
  - **api/** - FastAPI REST API with clean architecture
  - **orchestrator/** - Temporal workflow orchestration
  - **xorb/** - Core platform modules and services
  - **common/** - Shared utilities and configurations
- **ptaas-frontend/** - Next.js React frontend application
- **infra/** - Infrastructure, Docker, and deployment configurations
- **legacy/** - Archived legacy services and implementations
- **documentation/** - Project documentation and guides

### Core Services Architecture
The platform follows clean architecture principles:

1. **API Service** (`src/api/`)
   - **Controllers** - HTTP request handlers in `app/controllers/`
   - **Services** - Business logic in `app/services/`
   - **Repositories** - Data access in `app/infrastructure/repositories.py`
   - **Domain** - Entities and business rules in `app/domain/`
   - **Middleware** - Rate limiting, audit logging, feature flags

2. **Orchestrator Service** (`src/orchestrator/`)
   - Temporal-based workflow engine
   - Workflow orchestration and activity management
   - Supports both full and simple execution modes

3. **Core Platform** (`src/xorb/`)
   - **Intelligence Engine** - Threat intelligence and ML models
   - **Execution Engine** - Security scanning and assessment
   - **Architecture Components** - Service mesh, observability, fault tolerance
   - **SIEM** - Security Information and Event Management

### Frontend Architecture (`ptaas-frontend/`)
- Next.js 14 with TypeScript
- React 18 with modern hooks
- Tailwind CSS for styling
- i18next for internationalization
- Recharts for data visualization
- Framer Motion for animations

### Key Technologies
- **Backend**: FastAPI, Temporal, AsyncPG, Redis, Prometheus
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS
- **Database**: PostgreSQL with async drivers
- **Infrastructure**: Docker, Docker Compose, Terraform
- **Monitoring**: Prometheus, Grafana
- **Security**: Advanced rate limiting, audit logging, MFA

### Service Communication
- REST APIs for external communication
- Temporal workflows for complex orchestration
- Redis for caching and session management
- PostgreSQL for persistent data storage
- Prometheus for metrics collection

### Development Patterns

#### Clean Architecture Implementation
- Controllers handle HTTP requests and delegate to services
- Services contain business logic and coordinate between repositories
- Repositories abstract data access
- Domain entities define business rules
- Dependency injection via `container.py`

#### Database Access
- Use async patterns with AsyncPG
- Repository pattern for data access abstraction
- Database migrations via Alembic

#### API Development
1. Add routes in `routers/`
2. Implement controllers in `controllers/`
3. Create business logic in `services/`
4. Add repository methods if data access needed
5. Register dependencies in `container.py`

#### Frontend Development
- Component-based React architecture
- Custom hooks for state management
- TypeScript for type safety
- Responsive design with Tailwind CSS

### Environment Configuration
Key environment variables:
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `TEMPORAL_HOST` - Temporal server address
- `LOG_LEVEL` - Application logging level
- `ENVIRONMENT` - Deployment environment (dev/staging/prod)

### Access Points
- API Documentation: http://localhost:8000/docs
- API Health Check: http://localhost:8000/health
- Frontend Application: http://localhost:3000
- Grafana Monitoring: http://localhost:3000 (when deployed)
- Prometheus Metrics: http://localhost:9090 (when deployed)

## Development Guidelines

### Code Quality
- Follow clean architecture principles
- Use async/await patterns for I/O operations  
- Implement comprehensive error handling
- Add type hints for Python code
- Use TypeScript for frontend development

### Security Considerations
- Never commit secrets or API keys
- Use environment variables for configuration
- Implement proper authentication and authorization
- Follow secure coding practices
- Regular security scanning and updates

### Testing Strategy
- Unit tests for individual components
- Integration tests for service interactions
- End-to-end tests for critical workflows
- Security tests for vulnerability scanning
- Performance tests for scalability validation