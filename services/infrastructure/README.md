# XORB Infrastructure Services

## Overview
Shared infrastructure services that support the entire XORB ecosystem, including monitoring, secret management, and database configurations.

## Components

### Monitoring Stack
- **Prometheus**: Metrics collection and time-series database
- **Grafana**: Visualization dashboards and alerting
- **AlertManager**: Alert routing and notification management
- **Node Exporter**: System metrics collection
- **cAdvisor**: Container metrics and resource monitoring

### Vault Services
- **HashiCorp Vault**: Centralized secret management
- **Dynamic Credentials**: Database credential rotation
- **Transit Engine**: JWT signing and encryption
- **Policy Management**: Fine-grained access control

### Database Services
- **PostgreSQL Cluster**: Multi-tenant data storage with pgvector
- **Redis Cluster**: Session management and real-time caching
- **Backup Systems**: Automated backup and disaster recovery

## Service Structure
```
infrastructure/
├── monitoring/
│   ├── prometheus.yml      # Metrics configuration
│   ├── grafana/           # Dashboard definitions
│   └── alertmanager.yml   # Alert routing
├── vault/
│   ├── vault-config.hcl   # Vault configuration
│   ├── policies/          # Access policies
│   └── init-scripts/      # Initialization scripts
└── databases/
    ├── postgresql/        # PostgreSQL configurations
    └── redis/             # Redis cluster setup
```

## Access Points
- **Prometheus**: http://localhost:9092
- **Grafana**: http://localhost:3010 (admin / SecureAdminPass123!)
- **Vault UI**: http://localhost:8200
- **PostgreSQL**: localhost:5432-5434
- **Redis**: localhost:6380-6381

## Deployment
```bash
# Start monitoring stack
./tools/scripts/setup-monitoring.sh start

# Initialize Vault
cd services/infrastructure/vault
./setup-vault-dev.sh

# Start database services
docker-compose -f docker-compose.infrastructure.yml up -d
```
