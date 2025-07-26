# 🚀 XORB 2.0 Complete Deployment Guide

## Overview

XORB 2.0 is a comprehensive AI-powered security intelligence platform featuring:
- **Real-world threat intelligence integration** with VirusTotal, OTX, and MISP
- **Automated vulnerability lifecycle management** with remediation workflows
- **Advanced AI-powered threat hunting** with behavioral analysis and anomaly detection
- **Distributed campaign coordination** with consensus algorithms and fault tolerance
- **Enterprise-grade monitoring, logging, and compliance** features

## 📋 Prerequisites

### System Requirements

#### Development Environment
- **CPU**: 4 cores minimum
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 50GB available space
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2

#### Production Environment (EPYC Optimized)
- **CPU**: AMD EPYC 7702 (64 cores/128 threads) or equivalent
- **RAM**: 128GB minimum (256GB recommended)
- **Storage**: 1TB NVMe SSD minimum
- **Network**: 10Gbps network interface
- **OS**: Ubuntu 22.04 LTS or RHEL 8+

### Software Dependencies

```bash
# Core dependencies
- Python 3.9+
- Docker 20.10+
- Docker Compose 2.0+
- Git 2.30+

# Database systems
- PostgreSQL 14+ (with PGVector extension)
- Redis 7.0+
- Neo4j 5.0+
- ClickHouse 22.8+
- Qdrant 1.3+

# Message systems
- NATS JetStream 2.9+
- Temporal 1.20+

# Monitoring
- Prometheus 2.40+
- Grafana 9.0+
- Jaeger 1.35+

# Orchestration (Production)
- Kubernetes 1.25+
- Helm 3.10+
- ArgoCD 2.5+
```

## 🔧 Quick Start Deployment

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url> xorb-ecosystem
cd xorb-ecosystem

# Setup Python environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-autoconfigure.txt

# Install additional advanced feature dependencies
pip install prometheus-client structlog cryptography psutil aiohttp
```

### 2. Environment Configuration

```bash
# Auto-configure based on your environment
python autoconfigure.py

# Or manually configure
cp .xorb.env.example .xorb.env
# Edit .xorb.env with your specific settings
```

### 3. Initialize Database and Services

```bash
# Setup databases
docker-compose -f docker-compose.unified.yml up -d postgres redis neo4j clickhouse qdrant

# Wait for services to be ready
sleep 30

# Initialize database schemas
python scripts/init_databases.py
```

### 4. Start XORB Ecosystem

```bash
# Development deployment
make -f Makefile.advanced dev

# Or production deployment
make -f Makefile.advanced production-deploy
```

### 5. Verify Deployment

```bash
# Run comprehensive tests
make -f Makefile.advanced advanced-tests

# Run feature demonstrations
make -f Makefile.advanced advanced-demo
make -f Makefile.advanced vulnerability-demo  
make -f Makefile.advanced ai-hunting-demo
```

## 🏗️ Production Deployment Options

### Option 1: Docker Compose (Recommended for Single Node)

```bash
# Production-optimized compose
docker-compose -f docker-compose.production.yml up -d

# With monitoring stack
docker-compose -f docker-compose.production.yml -f docker-compose.monitoring.yml up -d
```

### Option 2: Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f gitops/overlays/production/

# Or using Helm
helm install xorb gitops/helm/xorb \
  --namespace xorb-system \
  --create-namespace \
  --values gitops/helm/xorb/values-production.yaml
```

### Option 3: GitOps with ArgoCD

```bash
# Apply ArgoCD ApplicationSet
kubectl apply -f gitops/argocd/applicationset.yaml

# Monitor deployment
argocd app sync xorb-production
argocd app wait xorb-production
```

## ⚙️ Configuration

### Environment Variables

Create `.xorb.env` file with:

```bash
# Environment
XORB_ENV=production
XORB_LOG_LEVEL=INFO
XORB_DEBUG=false

# Advanced Features
XORB_ADVANCED_FEATURES=true
XORB_PREDICTIVE_SCALING=true
XORB_COMPLIANCE_MODE=soc2
XORB_AUDIT_ENCRYPTION=true

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=xorb
POSTGRES_USER=xorb
POSTGRES_PASSWORD=secure_password

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=redis_password

NEO4J_HOST=localhost
NEO4J_PORT=7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j_password

# Threat Intelligence APIs
VIRUSTOTAL_API_KEY=your_vt_api_key
OTX_API_KEY=your_otx_api_key
MISP_URL=https://your-misp-instance.com
MISP_API_KEY=your_misp_api_key

# Monitoring
PROMETHEUS_HOST=localhost
PROMETHEUS_PORT=9090
GRAFANA_HOST=localhost
GRAFANA_PORT=3000

# Resource Optimization (EPYC)
XORB_MAX_CONCURRENT_AGENTS=32
XORB_CPU_AFFINITY=true
XORB_NUMA_AWARENESS=true
```

### Service Configuration

#### API Service
```yaml
# config/api.yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 16  # EPYC optimized

security:
  jwt_secret: "your-jwt-secret"
  cors_origins: ["*"]
  
rate_limiting:
  requests_per_minute: 1000
  burst_size: 100
```

#### Orchestrator Service
```yaml
# config/orchestrator.yaml
orchestrator:
  max_concurrent_campaigns: 32
  agent_discovery_interval: 300
  resource_check_interval: 60
  
coordination:
  consensus_algorithm: "raft"
  heartbeat_interval: 30
  failure_detection_timeout: 90
```

## 🔐 Security Configuration

### 1. TLS/SSL Setup

```bash
# Generate certificates
openssl req -x509 -newkey rsa:4096 -keyout private/key.pem -out certs/cert.pem -days 365

# Update configuration
echo "XORB_TLS_ENABLED=true" >> .xorb.env
echo "XORB_TLS_CERT_PATH=certs/cert.pem" >> .xorb.env
echo "XORB_TLS_KEY_PATH=private/key.pem" >> .xorb.env
```

### 2. Authentication & Authorization

```bash
# Enable authentication
echo "XORB_AUTH_ENABLED=true" >> .xorb.env
echo "XORB_AUTH_PROVIDER=jwt" >> .xorb.env

# Configure RBAC
python scripts/setup_rbac.py
```

### 3. Network Security

```bash
# Configure firewall (example for UFW)
sudo ufw allow 8000/tcp  # API
sudo ufw allow 8080/tcp  # Orchestrator
sudo ufw allow 3000/tcp  # Grafana
sudo ufw allow 9090/tcp  # Prometheus
sudo ufw deny 5432/tcp   # PostgreSQL (internal only)
sudo ufw deny 6379/tcp   # Redis (internal only)
```

## 📊 Monitoring Setup

### 1. Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'xorb-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    
  - job_name: 'xorb-orchestrator'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    
  - job_name: 'xorb-workers'
    static_configs:
      - targets: ['localhost:8081', 'localhost:8082']
```

### 2. Grafana Dashboards

Pre-configured dashboards are available in `grafana/dashboards/`:
- **XORB Ecosystem Overview**: System health and performance
- **Threat Intelligence Dashboard**: IoC tracking and correlation
- **Vulnerability Management**: Lifecycle and remediation tracking  
- **AI Threat Hunting**: Anomaly detection and hypothesis generation
- **Campaign Coordination**: Distributed execution monitoring

### 3. Alerting Rules

```yaml
# monitoring/alert-rules.yml
groups:
  - name: xorb-alerts
    rules:
      - alert: HighThreatActivity
        expr: rate(xorb_threats_detected_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High threat detection rate"
          
      - alert: VulnerabilityBacklog
        expr: xorb_active_vulnerabilities{severity="critical"} > 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Critical vulnerability backlog"
```

## 🧪 Testing & Validation

### 1. System Health Checks

```bash
# Check all services
make -f Makefile.advanced status-report

# Test individual components
make -f Makefile.advanced agent-discovery
make -f Makefile.advanced vulnerability-test
make -f Makefile.advanced threat-intel-test
make -f Makefile.advanced ai-hunting-test
```

### 2. Load Testing

```bash
# Run performance benchmarks
make -f Makefile.advanced benchmark

# Load testing
make -f Makefile.advanced load-test

# Stress testing (production)
python scripts/stress_test.py --duration 3600 --concurrent-users 100
```

### 3. Security Testing

```bash
# Security audit
make -f Makefile.advanced security-audit

# Compliance validation
make -f Makefile.advanced compliance-check

# Penetration testing
python scripts/security_validation.py
```

## 🔄 Maintenance & Operations

### 1. Backup Procedures

```bash
# Database backups
python scripts/backup_databases.py --output /backup/$(date +%Y%m%d)

# Configuration backup
tar -czf /backup/xorb-config-$(date +%Y%m%d).tar.gz config/ .xorb.env

# Application data backup
python scripts/backup_application_data.py
```

### 2. Updates & Upgrades

```bash
# Update system
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Apply database migrations
python scripts/migrate_databases.py

# Rolling restart
python scripts/rolling_restart.py
```

### 3. Log Management

```bash
# Log rotation
make -f Makefile.advanced clean-logs

# Log analysis
python scripts/analyze_logs.py --since 24h

# Export logs for compliance
python scripts/export_audit_logs.py --format json --period 30d
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Service Startup Failures

```bash
# Check service logs
docker-compose logs xorb-api
docker-compose logs xorb-orchestrator

# Check resource availability
docker stats
df -h
free -m
```

#### 2. Database Connection Issues

```bash
# Test database connectivity
python scripts/test_connections.py

# Reset database connections
docker-compose restart postgres redis neo4j
```

#### 3. Performance Issues

```bash
# Check resource usage
htop
iostat -x 1

# Analyze slow queries
python scripts/analyze_performance.py

# Check EPYC optimization
python scripts/check_epyc_optimization.py
```

#### 4. Memory Issues

```bash
# Check memory usage
python scripts/memory_analysis.py

# Clear caches
python scripts/clear_caches.py

# Restart with memory profiling
XORB_MEMORY_PROFILING=true docker-compose restart
```

### Advanced Debugging

#### 1. Enable Debug Mode

```bash
echo "XORB_DEBUG=true" >> .xorb.env
echo "XORB_LOG_LEVEL=DEBUG" >> .xorb.env
docker-compose restart
```

#### 2. Distributed Tracing

```bash
# Enable Jaeger tracing
echo "XORB_TRACING_ENABLED=true" >> .xorb.env
echo "JAEGER_ENDPOINT=http://localhost:14268/api/traces" >> .xorb.env
```

#### 3. Performance Profiling

```bash
# Enable profiling
python scripts/enable_profiling.py

# Generate performance report
python scripts/generate_performance_report.py
```

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] System requirements verified
- [ ] Dependencies installed
- [ ] Configuration files reviewed
- [ ] Security settings configured
- [ ] Backup procedures tested

### Deployment
- [ ] Environment variables set
- [ ] Databases initialized
- [ ] Services started successfully
- [ ] Health checks passing
- [ ] Monitoring configured

### Post-Deployment
- [ ] Load testing completed
- [ ] Security testing passed
- [ ] Performance baselines established
- [ ] Alerting rules activated
- [ ] Documentation updated

### Production Readiness
- [ ] Backup procedures automated
- [ ] Log rotation configured
- [ ] Update procedures documented
- [ ] Incident response plan ready
- [ ] Team training completed

## 🔗 Additional Resources

- **API Documentation**: `http://localhost:8000/docs`
- **Metrics Endpoint**: `http://localhost:8000/metrics`
- **Grafana Dashboards**: `http://localhost:3000`
- **Prometheus Metrics**: `http://localhost:9090`
- **System Status**: `make -f Makefile.advanced status-report`

## 📞 Support

For deployment assistance:

1. **Check logs**: `docker-compose logs -f`
2. **Run diagnostics**: `python scripts/system_diagnostics.py`
3. **Review documentation**: Check module-specific README files
4. **Test components**: `make -f Makefile.advanced advanced-tests`

---

**🎉 Congratulations!** You now have a fully deployed XORB 2.0 ecosystem with enterprise-grade security intelligence capabilities.