# 🐋 XORB Platform Docker Compose Configurations

[![Compose Status](https://img.shields.io/badge/Compose-Standardized-green)](#standardized-configurations)
[![Environments](https://img.shields.io/badge/Environments-Multi--Tier-blue)](#environment-configurations)
[![Security](https://img.shields.io/badge/Security-Enterprise%20Grade-orange)](#security-configurations)

> **Standardized Docker Compose Configurations**: Organized collection of Docker Compose files for different deployment scenarios and specialized services.

## 📁 Configuration Structure

```
infra/compose-configs/
├── README.md                         # This configuration guide
├── docker-compose.red-blue-agents.yml   # Red/Blue team agent framework
├── docker-compose.runtime-security.yml  # Runtime security monitoring
├── docker-compose.siem.yml               # SIEM and log aggregation
└── docker-compose.tls.yml               # TLS/mTLS security stack
```

## 🎯 Available Configurations

### 🔴🔵 **Red/Blue Team Agents** (`docker-compose.red-blue-agents.yml`)
Advanced red team and blue team agent framework for autonomous security testing and defense.

**Key Services:**
- **Red Team Agents**: Autonomous penetration testing agents
- **Blue Team Agents**: Defensive monitoring and response agents
- **Agent Orchestrator**: Centralized agent coordination
- **Sandbox Environment**: Isolated testing environment

**Usage:**
```bash
# Deploy red/blue team framework
docker-compose -f infra/compose-configs/docker-compose.red-blue-agents.yml up -d

# Monitor agent activities
docker-compose -f infra/compose-configs/docker-compose.red-blue-agents.yml logs -f
```

### 🛡️ **Runtime Security** (`docker-compose.runtime-security.yml`)
Comprehensive runtime security monitoring with Falco, container scanning, and threat detection.

**Key Services:**
- **Falco**: Runtime security monitoring and anomaly detection
- **Container Scanner**: Continuous container vulnerability scanning
- **Security Dashboard**: Real-time security metrics and alerts
- **Threat Detection**: ML-powered threat detection engine

**Usage:**
```bash
# Deploy runtime security stack
docker-compose -f infra/compose-configs/docker-compose.runtime-security.yml up -d

# Access security dashboard
open http://localhost:3001/security-dashboard
```

### 📊 **SIEM Stack** (`docker-compose.siem.yml`)
Security Information and Event Management (SIEM) with log aggregation, analysis, and alerting.

**Key Services:**
- **Elasticsearch**: Log storage and search engine
- **Logstash**: Log processing and transformation
- **Kibana**: Security dashboard and visualization
- **Security Analytics**: Automated threat analysis

**Usage:**
```bash
# Deploy SIEM stack
docker-compose -f infra/compose-configs/docker-compose.siem.yml up -d

# Access Kibana dashboard
open http://localhost:5601
```

### 🔐 **TLS/mTLS Security** (`docker-compose.tls.yml`)
Enterprise-grade TLS/mTLS implementation with certificate management and secure communication.

**Key Services:**
- **Certificate Authority**: Internal CA for certificate management
- **TLS Proxy**: Envoy proxy with mTLS termination
- **Certificate Manager**: Automated certificate rotation
- **Security Validation**: TLS configuration validation

**Usage:**
```bash
# Deploy TLS security stack
docker-compose -f infra/compose-configs/docker-compose.tls.yml up -d

# Validate TLS configuration
./scripts/validate/test_tls.sh
```

## 🏗️ Integration with Main Platform

### Environment Integration
These specialized configurations are designed to integrate seamlessly with the main XORB platform:

```bash
# Start main platform
docker-compose -f docker-compose.production.yml up -d

# Add specialized services
docker-compose -f infra/compose-configs/docker-compose.red-blue-agents.yml up -d
docker-compose -f infra/compose-configs/docker-compose.runtime-security.yml up -d
```

### Network Configuration
All configurations use shared networks for seamless integration:

```yaml
networks:
  xorb-network:
    external: true
  security-network:
    external: true
```

### Service Discovery
Services are configured for automatic discovery and integration:

- **DNS Resolution**: Automatic service discovery via Docker DNS
- **Health Checks**: Comprehensive health monitoring
- **Load Balancing**: Automatic load balancing for scaled services
- **Security Policies**: Consistent security policies across all services

## 🔧 Configuration Management

### Environment Variables
Each configuration supports environment-specific customization:

```bash
# Development environment
export ENVIRONMENT=development
export LOG_LEVEL=debug

# Production environment
export ENVIRONMENT=production
export LOG_LEVEL=info
export SECURITY_LEVEL=strict
```

### Volume Management
Consistent volume management across configurations:

```yaml
volumes:
  xorb-data:
    external: true
  security-logs:
    external: true
  certificates:
    external: true
```

### Secret Management
Integrated secret management with HashiCorp Vault:

```yaml
secrets:
  database_password:
    external: true
  api_keys:
    external: true
  certificates:
    external: true
```

## 📊 Monitoring and Observability

### Prometheus Integration
All configurations include Prometheus metrics:

```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

### Grafana Dashboards
Pre-configured Grafana dashboards for each service stack:

- **Red/Blue Agents**: Agent performance and security metrics
- **Runtime Security**: Security alerts and threat detection
- **SIEM**: Log analysis and security intelligence
- **TLS Security**: Certificate status and TLS metrics

### Alerting
Comprehensive alerting for all critical events:

```yaml
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

## 🛡️ Security Considerations

### Network Security
- **Network Isolation**: Services isolated in dedicated networks
- **Firewall Rules**: Strict ingress/egress policies
- **TLS Encryption**: All inter-service communication encrypted
- **Access Control**: Role-based access control (RBAC)

### Container Security
- **Security Contexts**: Non-root containers with security contexts
- **Image Scanning**: Regular vulnerability scanning
- **Runtime Protection**: Falco runtime security monitoring
- **Resource Limits**: CPU and memory limits for all containers

### Data Protection
- **Encryption at Rest**: All persistent data encrypted
- **Encryption in Transit**: TLS/mTLS for all communications
- **Secret Management**: Centralized secret management with Vault
- **Audit Logging**: Comprehensive audit trails

## 🚀 Deployment Strategies

### Development Deployment
```bash
# Quick development setup
docker-compose -f docker-compose.development.yml up -d
docker-compose -f infra/compose-configs/docker-compose.runtime-security.yml up -d
```

### Staging Deployment
```bash
# Staging environment with security monitoring
docker-compose -f docker-compose.production.yml up -d
docker-compose -f infra/compose-configs/docker-compose.siem.yml up -d
docker-compose -f infra/compose-configs/docker-compose.tls.yml up -d
```

### Production Deployment
```bash
# Full production stack with all security services
docker-compose -f docker-compose.production.yml up -d
docker-compose -f infra/compose-configs/docker-compose.red-blue-agents.yml up -d
docker-compose -f infra/compose-configs/docker-compose.runtime-security.yml up -d
docker-compose -f infra/compose-configs/docker-compose.siem.yml up -d
docker-compose -f infra/compose-configs/docker-compose.tls.yml up -d
```

## 🔄 Maintenance and Updates

### Regular Maintenance
```bash
# Update all configurations
docker-compose -f infra/compose-configs/docker-compose.*.yml pull

# Restart services with updated images
docker-compose -f infra/compose-configs/docker-compose.*.yml up -d --force-recreate
```

### Health Monitoring
```bash
# Check all service health
docker-compose -f infra/compose-configs/docker-compose.*.yml ps

# View service logs
docker-compose -f infra/compose-configs/docker-compose.*.yml logs -f [service_name]
```

### Backup and Recovery
```bash
# Backup configuration data
docker run --rm -v xorb-data:/data -v $(pwd):/backup busybox tar czf /backup/xorb-data-backup.tar.gz /data

# Restore configuration data
docker run --rm -v xorb-data:/data -v $(pwd):/backup busybox tar xzf /backup/xorb-data-backup.tar.gz -C /
```

## 📋 Best Practices

### Configuration Best Practices
1. **Environment Separation**: Use separate configurations for dev/staging/prod
2. **Secret Management**: Never store secrets in compose files
3. **Resource Limits**: Always define CPU and memory limits
4. **Health Checks**: Include health checks for all services
5. **Logging**: Centralized logging with structured formats

### Security Best Practices
1. **Least Privilege**: Run containers with minimal privileges
2. **Network Segmentation**: Isolate services in dedicated networks
3. **Regular Updates**: Keep images and configurations updated
4. **Monitoring**: Continuous security monitoring and alerting
5. **Compliance**: Ensure configurations meet compliance requirements

### Operational Best Practices
1. **Documentation**: Maintain comprehensive documentation
2. **Version Control**: Version control all configuration changes
3. **Testing**: Test configurations in non-production environments
4. **Rollback**: Maintain rollback capabilities for all changes
5. **Monitoring**: Monitor configuration drift and compliance

---

*These standardized Docker Compose configurations provide enterprise-grade deployment flexibility while maintaining the security and operational excellence of the XORB platform.*