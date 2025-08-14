# XORB Enterprise Service Architecture

## Service Overview

The XORB platform follows a microservices architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                        XORB Ecosystem                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────────────┐ │
│  │   PTaaS Web     │    │          XORB Core Platform         │ │
│  │   Frontend      │◄───┤                                     │ │
│  │   (React)       │    │  ┌─────────────┐ ┌─────────────────┐ │ │
│  │   Port 8080     │    │  │ API Gateway │ │   Orchestrator  │ │ │
│  └─────────────────┘    │  │ Port 8000   │ │   Port 8080     │ │ │
│                         │  └─────────────┘ └─────────────────┘ │ │
│                         │  ┌─────────────┐ ┌─────────────────┐ │ │
│                         │  │Intelligence │ │   Security      │ │ │
│                         │  │   Engine    │ │   Services      │ │ │
│                         │  └─────────────┘ └─────────────────┘ │ │
│                         └─────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Infrastructure Services                      │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐ │
│  │   Monitoring    │ │      Vault      │ │     Databases       │ │
│  │  (Prometheus)   │ │  (Secrets Mgmt) │ │ (PostgreSQL/Redis)  │ │
│  │   Port 9092     │ │   Port 8200     │ │   Ports 5432-5434   │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Service Responsibilities

### PTaaS Frontend Service
- **Primary Role**: User interface and experience layer
- **Technology**: React 18.3.1 + TypeScript 5.5.3
- **Features**:
  - Enterprise dashboard and reporting
  - Real-time security monitoring
  - Vulnerability workflow management
  - Threat intelligence visualization
  - Multi-tenant organization management

### XORB Core Platform
- **Primary Role**: Backend business logic and API services
- **Components**:
  - **API Gateway**: Unified REST API with comprehensive middleware
  - **Orchestrator**: Temporal-based workflow engine for complex operations
  - **Intelligence Engine**: AI-powered threat detection and correlation
  - **Security Services**: Zero trust, monitoring, and compliance

### Infrastructure Services
- **Primary Role**: Shared platform services and data management
- **Components**:
  - **Monitoring**: Prometheus, Grafana, AlertManager stack
  - **Vault**: HashiCorp Vault for secret management and encryption
  - **Databases**: PostgreSQL clusters with Redis caching

## Communication Patterns

### HTTP REST APIs
- **PTaaS ↔ XORB Core**: Standard REST endpoints for CRUD operations
- **Authentication**: JWT-based with automatic refresh
- **Rate Limiting**: Redis-backed with tenant isolation

### WebSocket Connections
- **Real-time Updates**: Live scan progress and security alerts
- **Event Streaming**: Redis pub/sub for service notifications
- **Dashboard Updates**: Live metrics and status information

### Workflow Orchestration
- **Temporal Workflows**: Long-running security operations
- **Activity Functions**: Distributed task execution
- **Event Sourcing**: Complete audit trail of operations

## Data Flow Architecture

```
User Action (PTaaS) → API Gateway (XORB) → Service Layer →
Orchestrator (Workflows) → Executors → Database →
Cache (Redis) → Response → Real-time Updates (WebSocket)
```

## Security Architecture

### Authentication & Authorization
- **Multi-tenant JWT**: Secure session management
- **Role-based Access**: Fine-grained permissions
- **API Security**: Rate limiting, input validation, security headers

### Data Protection
- **Encryption at Rest**: Database and file system encryption
- **Encryption in Transit**: TLS 1.3 for all communications
- **Secret Management**: Vault-based credential rotation

### Monitoring & Auditing
- **Security Events**: Comprehensive audit logging
- **Threat Detection**: Real-time security monitoring
- **Compliance**: Automated compliance reporting and validation

## Deployment Architecture

### Development Environment
```bash
# PTaaS Frontend
cd services/ptaas/web && npm run dev

# XORB API Gateway
cd services/xorb-core/api && uvicorn app.main:app --reload

# Infrastructure
docker-compose -f docker-compose.infrastructure.yml up -d
```

### Production Environment
```bash
# Container orchestration with Docker Swarm/Kubernetes
# Load balancing with NGINX
# CDN deployment for frontend assets
# Multi-region database replication
# Automated scaling and health monitoring
```

## Performance Characteristics

- **API Response Time**: < 100ms for standard operations
- **WebSocket Latency**: < 50ms for real-time updates
- **Concurrent Users**: 10,000+ with horizontal scaling
- **Data Processing**: Real-time threat intelligence correlation
- **Storage**: Multi-TB security data with efficient querying
