# 🚀 XORB Production Deployment Guide

## 📋 **Overview**

This guide provides comprehensive instructions for deploying the XORB Autonomous Cybersecurity Platform in production environments using secure, automated, and observable infrastructure practices.

## 🔧 **Prerequisites**

### **Infrastructure Requirements**
- **Docker Engine**: 20.10+ with BuildKit support
- **Docker Compose**: 2.0+ for multi-service orchestration
- **Kubernetes**: 1.24+ (for production scaling)
- **CPU**: 16+ cores (64+ cores recommended for EPYC optimization)
- **Memory**: 32GB+ RAM (128GB+ recommended for full deployment)
- **Storage**: 500GB+ SSD for persistent data
- **Network**: 1Gbps+ bandwidth for optimal performance

### **Security Requirements**
- **TLS Certificates**: Valid SSL/TLS certificates for HTTPS
- **Secrets Management**: Secure key management system
- **Firewall**: Properly configured network security
- **Monitoring**: Centralized logging and monitoring system

### **Access Requirements**
- **Container Registry**: Access to GitHub Container Registry (ghcr.io)
- **API Keys**: Valid NVIDIA, OpenRouter, and other service credentials
- **Database**: Production-grade PostgreSQL, Redis instances

## 🔐 **Security Configuration**

### **1. Secrets Management**

Create and configure secure environment variables:

```bash
# Copy environment template
cp .env.example .env

# Configure required secrets (use secure random generation)
# NEVER commit .env to version control
```

**Required Environment Variables:**
```bash
# AI Service API Keys
NVIDIA_API_KEY=nvapi-your-secure-key-here
OPENROUTER_API_KEY=sk-or-your-secure-key-here

# Database Credentials (use strong passwords)
POSTGRES_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
NEO4J_PASSWORD=$(openssl rand -base64 32)

# Application Security
JWT_SECRET_KEY=$(openssl rand -base64 64)
ENCRYPTION_KEY=$(openssl rand -base64 32)

# Environment Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
```

### **2. Container Security**

Verify security scanning results:
```bash
# Run security scan before deployment
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image ghcr.io/losa201/xorb-api:latest

# Check for vulnerabilities
docker run --rm -v $PWD:/app \
  returntocorp/semgrep --config=auto /app
```

### **3. Network Security**

Configure firewall rules:
```bash
# Allow only necessary ports
ufw allow 22/tcp      # SSH
ufw allow 80/tcp      # HTTP (redirects to HTTPS)
ufw allow 443/tcp     # HTTPS
ufw allow 8000/tcp    # XORB API (internal)
ufw enable
```

## 🚀 **Deployment Methods**

### **Method 1: Docker Compose (Recommended for Single-Node)**

#### **Quick Production Deployment**
```bash
# 1. Clone repository
git clone https://github.com/losa201/Xorb.git
cd Xorb

# 2. Configure environment
cp .env.example .env
# Edit .env with your production values

# 3. Deploy with production configuration
docker-compose -f docker-compose.production.yml up -d

# 4. Verify deployment
docker-compose -f docker-compose.production.yml ps
```

#### **Advanced Production Deployment**
```bash
# 1. Build custom images with security patches
docker-compose -f docker-compose.production.yml build --no-cache

# 2. Run security validation
python3 scripts/deployment_readiness_check.py

# 3. Deploy with monitoring
docker-compose -f docker-compose.production.yml \
  -f docker-compose.monitoring.yml up -d

# 4. Configure log rotation and backups
./scripts/setup-production-logging.sh
./scripts/setup-automated-backups.sh
```

### **Method 2: Kubernetes (Recommended for Multi-Node)**

#### **Production Kubernetes Deployment**
```bash
# 1. Configure Kubernetes context
kubectl config use-context production

# 2. Create namespace and secrets
kubectl create namespace xorb-production
kubectl create secret generic xorb-secrets \
  --from-env-file=.env \
  --namespace=xorb-production

# 3. Deploy with Helm
helm upgrade --install xorb ./gitops/helm/xorb-core \
  --namespace xorb-production \
  --values ./gitops/helm/xorb-core/values-production.yaml

# 4. Verify deployment
kubectl get pods -n xorb-production
kubectl get services -n xorb-production
```

#### **GitOps Deployment with ArgoCD**
```bash
# 1. Apply ArgoCD ApplicationSet
kubectl apply -f gitops/argocd/applicationset.yaml

# 2. Sync applications
argocd app sync xorb-production

# 3. Monitor deployment
argocd app get xorb-production
kubectl get applications -n argocd
```

## 📊 **Monitoring and Observability**

### **1. Prometheus Monitoring**

Access monitoring dashboard:
```bash
# Prometheus metrics
curl http://localhost:9090/metrics

# Grafana dashboard
open http://localhost:3000
# Login: admin / ${GRAFANA_PASSWORD}
```

### **2. Health Checks**

Verify system health:
```bash
# API health check
curl -f http://localhost:8000/health

# Worker health check  
curl -f http://localhost:9000/health

# Orchestrator health check
curl -f http://localhost:8080/health

# Database connectivity
docker exec xorb-postgres pg_isready

# Redis connectivity
docker exec xorb-redis redis-cli ping
```

### **3. Log Management**

Configure centralized logging:
```bash
# View application logs
docker-compose logs -f api worker orchestrator

# Export logs for analysis
docker-compose logs --since 1h > xorb-logs-$(date +%Y%m%d).log

# Monitor real-time metrics
watch -n 5 'docker stats --no-stream'
```

## 🔧 **Performance Optimization**

### **1. EPYC CPU Optimization**

For AMD EPYC processors:
```bash
# Configure NUMA awareness
echo 'vm.zone_reclaim_mode = 1' >> /etc/sysctl.conf
echo 'kernel.numa_balancing = 1' >> /etc/sysctl.conf

# Set CPU governor for performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Configure container CPU affinity
docker-compose -f docker-compose.production.yml \
  -f docker-compose.epyc.yml up -d
```

### **2. Memory Optimization**

```bash
# Configure swappiness for production
echo 'vm.swappiness = 10' >> /etc/sysctl.conf

# Set memory overcommit for containers
echo 'vm.overcommit_memory = 1' >> /etc/sysctl.conf

# Apply changes
sysctl -p
```

### **3. Network Optimization**

```bash
# Increase network buffer sizes
echo 'net.core.rmem_max = 268435456' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 268435456' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_rmem = 4096 65536 268435456' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 268435456' >> /etc/sysctl.conf
```

## 🛠️ **Maintenance Operations**

### **1. Updates and Upgrades**

```bash
# Pull latest images
docker-compose -f docker-compose.production.yml pull

# Perform rolling update (zero-downtime)
docker-compose -f docker-compose.production.yml up -d --no-deps api
docker-compose -f docker-compose.production.yml up -d --no-deps worker
docker-compose -f docker-compose.production.yml up -d --no-deps orchestrator

# Verify health after updates
python3 scripts/deployment_readiness_check.py
```

### **2. Backup and Recovery**

```bash
# Database backup
docker exec xorb-postgres pg_dump -U xorb xorb_db > backup-$(date +%Y%m%d).sql

# Redis backup
docker exec xorb-redis redis-cli BGSAVE
docker cp xorb-redis:/data/dump.rdb ./redis-backup-$(date +%Y%m%d).rdb

# Configuration backup
tar -czf config-backup-$(date +%Y%m%d).tar.gz .env monitoring/ gitops/
```

### **3. Scaling Operations**

```bash
# Scale workers horizontally
docker-compose -f docker-compose.production.yml up -d --scale worker=5

# Scale with Kubernetes
kubectl scale deployment xorb-worker --replicas=10 -n xorb-production

# Monitor scaling impact
kubectl top pods -n xorb-production
```

## 🚨 **Troubleshooting**

### **Common Issues and Solutions**

#### **1. Container Won't Start**
```bash
# Check container logs
docker-compose logs <service-name>

# Verify environment variables
docker-compose config

# Check resource constraints
docker system df
docker system prune -f
```

#### **2. Database Connection Issues**
```bash
# Test database connectivity
docker exec xorb-postgres pg_isready -U xorb

# Check database logs
docker logs xorb-postgres

# Verify credentials
echo $DATABASE_URL
```

#### **3. Performance Issues**
```bash
# Monitor resource usage
docker stats

# Check system metrics
top -p $(pgrep -d, -f xorb)

# Analyze slow queries
docker exec xorb-postgres psql -U xorb -c "SELECT * FROM pg_stat_activity;"
```

#### **4. Security Scan Failures**
```bash
# Re-run security scan
trivy image ghcr.io/losa201/xorb-api:latest

# Check for hardcoded secrets
grep -r "api.*key" . --exclude-dir=.git

# Verify certificate validity
openssl x509 -in cert.pem -text -noout
```

## 📈 **Production Checklist**

### **Pre-Deployment**
- [ ] Environment variables configured and secured
- [ ] SSL/TLS certificates installed and valid
- [ ] Firewall rules configured
- [ ] Database backups scheduled
- [ ] Monitoring and alerting configured
- [ ] Security scanning completed successfully
- [ ] Load testing performed
- [ ] Disaster recovery plan documented

### **Post-Deployment**
- [ ] Health checks passing
- [ ] Monitoring dashboards accessible
- [ ] Log aggregation working
- [ ] Backup procedures verified
- [ ] Performance baselines established
- [ ] Alert thresholds configured
- [ ] Documentation updated
- [ ] Team trained on operations

## 🎯 **Best Practices**

### **Security**
- Use strong, unique passwords for all services
- Regularly rotate API keys and certificates
- Keep container images updated with latest security patches
- Implement network segmentation
- Enable audit logging for all components

### **Reliability**
- Implement health checks for all services
- Use rolling deployments for zero-downtime updates
- Configure automatic restarts for failed containers
- Set up automated backups with tested recovery procedures
- Monitor key performance indicators and set alerts

### **Performance**
- Size containers appropriately for workload
- Use SSD storage for database persistence
- Configure memory limits to prevent OOM kills
- Implement connection pooling for databases
- Enable compression for network communication

### **Observability**
- Collect metrics from all components
- Centralize logs with structured logging
- Set up distributed tracing for request flows
- Create dashboards for business and technical metrics
- Implement alerting for critical conditions

## 📞 **Support and Escalation**

### **Emergency Contacts**
- **Platform Team**: platform-team@xorb.security
- **DevOps Team**: devops@xorb.security  
- **Security Team**: security@xorb.security
- **On-Call**: +1-555-XORB-OPS

### **Escalation Procedures**
1. **Level 1**: Check logs and restart affected services
2. **Level 2**: Contact platform team via Slack #xorb-ops
3. **Level 3**: Page on-call engineer for critical issues
4. **Level 4**: Escalate to engineering leadership

---

**Last Updated**: 2025-07-27  
**Version**: 1.0  
**Maintained By**: XORB DevOps Team