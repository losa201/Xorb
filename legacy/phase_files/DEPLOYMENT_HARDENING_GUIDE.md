# Xorb Security Intelligence Platform - Production Deployment Guide

## Overview

This guide covers the hardened production deployment of Xorb 2.0 on a single 16 vCPU / 32 GiB EPYC VPS with optional Pi 5 edge worker support.

## Architecture

### Runtime Topology

```
docker-compose.vps.yml
├─ api          (xorb-api:2.0.x, uid 10101, read-only)
├─ worker       (xorb-worker:2.0.x, uid 10102, read-only)
├─ embedding    (xorb-embed:2.0.x, uid 10103, read-only)
├─ postgres     (bitnami/postgresql:16, uid 1001, tmpfs /tmp)
├─ neo4j        (neo4j:5-enterprise)
├─ redis        (redis:7-alpine, tmpfs /data)
├─ nats         (nats:2.10-alpine, JetStream enabled)
├─ prometheus   (prom/prometheus:v2.52)
├─ grafana      (grafana/grafana:11)
├─ tempo        (grafana/tempo:2.4)
└─ alertmanager (prom/alertmanager:v0.27)
```

### Security Hardening Features

- **Non-root containers**: All app containers run as dedicated UIDs (10101-10103)
- **Read-only filesystem**: Application containers use read-only root filesystem
- **Capability dropping**: All unnecessary capabilities removed
- **Seccomp profiles**: Default seccomp security profiles applied
- **No new privileges**: Prevents privilege escalation
- **Tmpfs mounts**: Temporary filesystems for sensitive data
- **Resource limits**: CPU and memory constraints applied
- **Network isolation**: Internal bridge network with minimal exposed ports

## Prerequisites

### VPS Requirements

- **CPU**: 16 vCPU (AMD EPYC preferred)
- **Memory**: 32 GiB RAM
- **Storage**: 500 GB SSD minimum
- **OS**: Ubuntu 22.04 LTS or similar
- **Network**: Public IP with firewall capability

### Required Accounts

- NVIDIA Developer Account (for embedding API)
- Container registry access (GitHub Container Registry)
- SMTP account for alerting (optional)

## Deployment Methods

### Method 1: Cloud-Init Automated Deployment

Use the provided cloud-init configuration for fully automated deployment:

```bash
# Upload cloud-config.yml to your VPS provider
terraform apply -var="cloud_init_file=terraform/modules/vps_cloud_init/cloud-config.yml"
```

The cloud-init script will:
- Install Docker and dependencies
- Create system users
- Configure firewall
- Clone repository
- Setup secrets
- Deploy services
- Enable monitoring

### Method 2: Manual Deployment

#### 1. VPS Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y docker.io docker-compose-v2 git curl wget

# Create xorb user
sudo useradd -m -u 10101 xorb
sudo usermod -aG docker xorb

# Setup firewall
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 8000/tcp  # API
sudo ufw allow 3000/tcp  # Grafana
sudo ufw allow 9090/tcp  # Prometheus
```

#### 2. Repository Setup

```bash
# Clone repository
sudo -u xorb git clone https://github.com/xorb-platform/xorb.git /opt/xorb
cd /opt/xorb

# Setup secrets
sudo -u xorb mkdir -p .secrets
sudo -u xorb chmod 700 .secrets

# Generate PostgreSQL password
sudo -u xorb openssl rand -base64 32 > .secrets/postgres_password

# Add NVIDIA API key
sudo -u xorb echo "YOUR_NVIDIA_API_KEY" > .secrets/nvidia_api_key

# Set permissions
sudo -u xorb chmod 600 .secrets/*
```

#### 3. Environment Configuration

```bash
# Create environment file
sudo -u xorb cat > .env << 'EOF'
XORB_VERSION=2.0.0
POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password
NEO4J_PASSWORD=xorb_neo4j_2024
GRAFANA_PASSWORD=xorb_grafana_2024
COMPOSE_PROJECT_NAME=xorb
COMPOSE_FILE=docker-compose.vps.yml
EOF
```

#### 4. Service Deployment

```bash
# Pull images
sudo -u xorb docker compose -f docker-compose.vps.yml pull

# Start services
sudo -u xorb docker compose -f docker-compose.vps.yml up -d

# Verify deployment
python3 scripts/verify_hardened_deploy.py
```

## Edge Worker Setup (Optional)

Deploy lightweight workers on Raspberry Pi 5 for distributed processing:

```bash
# Deploy to Pi
./scripts/deploy_pi.sh pi5.local YOUR_VPS_IP

# Monitor edge worker
ssh pi@pi5.local '~/xorb/scripts/monitor_edge.sh'
```

## Security Configuration

### Container Security Rules

| Check | Implementation |
|-------|----------------|
| Non-root | `USER 10101` in all Dockerfiles |
| Read-only FS | `read_only: true` in compose |
| Seccomp | `seccomp:unconfined` security option |
| SBOM | `docker buildx bake --sbom` in CI |
| Vuln Gate | `trivy image --severity CRITICAL --exit-code 1` |

### Network Security

- Internal bridge network (172.20.0.0/16)
- Minimal port exposure (8000, 3000, 9090)
- TLS termination at load balancer
- Firewall rules for ingress traffic

### Data Protection

- Encrypted secrets via Docker secrets
- Tmpfs for temporary data
- Regular backup automation
- Log rotation and retention

## Monitoring and Observability

### Prometheus Metrics

- Application performance metrics
- Container resource usage
- Security event monitoring
- Cost tracking (NVIDIA API usage)

### Grafana Dashboards

- **ID 17476**: Tempo distributed tracing overview
- **ID 17966**: Docker container resource monitoring
- **Custom**: Embedding cost monitoring (tokens × $0.004)

### Alerting Rules

Key alerts configured:
- Container restart loops
- High memory/CPU usage
- Authentication failures
- Database connectivity
- API response times
- Cost thresholds

### Log Management

- Structured JSON logging
- Centralized log aggregation
- 7-day retention policy
- Security event correlation

## Operational Procedures

### Health Monitoring

```bash
# Check overall system health
curl http://localhost:8000/health

# Verify all services
docker compose ps

# Run full verification
python3 scripts/verify_hardened_deploy.py
```

### Backup Operations

```bash
# Run manual backup
./scripts/backup_data.sh

# Automated backups via cron
0 2 * * * cd /opt/xorb && ./scripts/backup_data.sh
```

### Updates and Maintenance

```bash
# Update platform
./scripts/update_xorb.sh

# View logs
docker compose logs -f api

# Restart specific service
docker compose restart api
```

## Cost Management

### NVIDIA API Monitoring

- Real-time cost tracking in Grafana
- Alert thresholds for budget control
- Token usage optimization
- Batch processing for efficiency

### Resource Optimization

- Horizontal pod autoscaling
- Resource quota enforcement
- Performance budget validation
- Capacity planning alerts

## Troubleshooting

### Common Issues

1. **Container won't start**
   ```bash
   docker compose logs [service_name]
   ```

2. **Database connection failed**
   ```bash
   docker compose exec postgres pg_isready
   ```

3. **High memory usage**
   ```bash
   docker stats
   ```

4. **API not responding**
   ```bash
   curl -f http://localhost:8000/health
   ```

### Performance Tuning

- Adjust container resource limits
- Optimize database queries
- Configure connection pooling
- Enable caching strategies

## Security Hardening Checklist

- [ ] Non-root containers configured
- [ ] Read-only filesystems enabled
- [ ] Secrets properly configured
- [ ] Firewall rules applied
- [ ] TLS certificates installed
- [ ] Monitoring alerts configured
- [ ] Backup automation enabled
- [ ] Log rotation configured
- [ ] Security scanning enabled
- [ ] Vulnerability management in place

## Compliance and Auditing

### Security Scanning

- Container vulnerability scanning
- Dependency vulnerability checks
- SBOM generation and tracking
- Security policy enforcement

### Audit Logging

- All API requests logged
- Authentication events tracked
- Administrative actions recorded
- Security events correlated

## Support and Maintenance

### Regular Tasks

- **Daily**: Monitor dashboards and alerts
- **Weekly**: Review security logs and metrics
- **Monthly**: Update dependencies and patches
- **Quarterly**: Disaster recovery testing

### Emergency Procedures

- Incident response playbooks
- Escalation procedures
- Recovery point objectives
- Business continuity planning

## Documentation Links

- [API Documentation](./docs/API_DOCUMENTATION.md)
- [Monitoring Setup](./docs/MONITORING_SETUP_GUIDE.md)
- [Disaster Recovery](./docs/disaster-recovery-drillbook.md)
- [Security Policies](./compose/security/security-policies.yaml)

---

**Security Notice**: This deployment includes production-grade security hardening. Ensure all secrets are properly configured and regularly rotated. Monitor the security dashboards and respond to alerts promptly.