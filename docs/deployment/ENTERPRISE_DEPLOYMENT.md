# XORB Enterprise Deployment Guide

##  Deployment Overview

XORB supports multiple deployment strategies for enterprise environments, from single-node development to multi-region production clusters.

##  Deployment Architecture

###  Single Node Development
```bash
# Quick start for development and testing
cd /root/Xorb
docker-compose -f docker-compose.development.yml up -d

# Services will be available at:
# - PTaaS Frontend: http://localhost:8080
# - XORB API: http://localhost:8000
# - Grafana: http://localhost:3010
# - Prometheus: http://localhost:9092
```text

###  Production Cluster
```bash
# Multi-node production deployment
docker-compose -f docker-compose.production.yml up -d

# With additional services:
# - Load balancers
# - SSL termination
# - Database replication
# - Backup systems
```text

##  Service Deployment

###  PTaaS Frontend Deployment
```bash
# Build production assets
cd services/ptaas/web
npm run build

# Deploy to CDN (Vercel, Netlify, Cloudflare)
vercel deploy --prod

# Or deploy to static hosting
aws s3 sync dist/ s3://your-bucket/
aws cloudfront create-invalidation --distribution-id YOUR_ID --paths "/*"
```text

###  XORB Core Platform Deployment
```bash
# Container deployment with health checks
docker run -d \
  --name xorb-api \
  --restart unless-stopped \
  --health-cmd "curl -f http://localhost:8000/health || exit 1" \
  --health-interval 30s \
  --health-retries 3 \
  -p 8000:8000 \
  xorb/api:latest

# Orchestrator service
docker run -d \
  --name xorb-orchestrator \
  --restart unless-stopped \
  -e TEMPORAL_HOST=temporal:7233 \
  xorb/orchestrator:latest
```text

###  Infrastructure Services
```bash
# Monitoring stack
./tools/scripts/setup-monitoring.sh production

# Vault cluster
cd services/infrastructure/vault
./setup-vault-production.sh

# Database cluster with replication
docker run -d \
  --name postgres-primary \
  -e POSTGRES_REPLICATION_MODE=master \
  -e POSTGRES_REPLICATION_USER=replicator \
  -p 5432:5432 \
  postgres:15-alpine
```text

##  Environment Configuration

###  Production Environment Variables
```bash
# Core Platform
export ENVIRONMENT=production
export DATABASE_URL=postgresql://user:pass@postgres:5432/xorb
export REDIS_URL=redis://redis:6379/0
export TEMPORAL_HOST=temporal:7233
export VAULT_ADDR=https://vault.company.com:8200

# Security Configuration
export JWT_SECRET_KEY=vault:secret/xorb/jwt#secret_key
export ENCRYPTION_KEY=vault:secret/xorb/encryption#key
export TLS_CERT_PATH=/etc/ssl/certs/xorb.crt
export TLS_KEY_PATH=/etc/ssl/private/xorb.key

# API Configuration
export API_RATE_LIMIT_PER_MINUTE=1000
export API_RATE_LIMIT_PER_HOUR=50000
export CORS_ALLOW_ORIGINS=https://ptaas.company.com
export ENABLE_METRICS=true
export LOG_LEVEL=INFO

# External Services
export NVIDIA_API_KEY=vault:secret/xorb/external#nvidia_key
export OPENROUTER_API_KEY=vault:secret/xorb/external#openrouter_key
```text

###  Multi-Tenant Configuration
```yaml
# tenant-config.yaml
tenants:
  default:
    database_url: "postgresql://app:pass@postgres:5432/tenant_default"
    redis_prefix: "tenant:default"
    rate_limits:
      api_calls_per_minute: 1000
      concurrent_scans: 10
  enterprise:
    database_url: "postgresql://app:pass@postgres:5432/tenant_enterprise"
    redis_prefix: "tenant:enterprise"
    rate_limits:
      api_calls_per_minute: 5000
      concurrent_scans: 50
```text

##  Load Balancing & High Availability

###  NGINX Load Balancer Configuration
```nginx
upstream ptaas_frontend {
    server ptaas-1:8080;
    server ptaas-2:8080;
    server ptaas-3:8080;
}

upstream xorb_api {
    server api-1:8000;
    server api-2:8000;
    server api-3:8000;
}

server {
    listen 443 ssl http2;
    server_name ptaas.company.com;

    ssl_certificate /etc/ssl/certs/company.crt;
    ssl_certificate_key /etc/ssl/private/company.key;

    location / {
        proxy_pass http://ptaas_frontend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api/ {
        proxy_pass http://xorb_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```text

###  Kubernetes Deployment
```yaml
# ptaas-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ptaas-frontend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ptaas-frontend
  template:
    metadata:
      labels:
        app: ptaas-frontend
    spec:
      containers:
      - name: ptaas
        image: xorb/ptaas:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```text

##  Monitoring & Observability

###  Health Check Endpoints
```bash
# Service health checks
curl http://localhost:8000/health
curl http://localhost:8000/readiness

# Detailed service status
curl http://localhost:8000/api/v1/status
```text

###  Monitoring Configuration
```yaml
# prometheus-config.yml
scrape_configs:
  - job_name: 'xorb-api'
    static_configs:
      - targets: ['api-1:8000', 'api-2:8000', 'api-3:8000']
    metrics_path: '/metrics'

  - job_name: 'xorb-orchestrator'
    static_configs:
      - targets: ['orchestrator-1:8080', 'orchestrator-2:8080']
    metrics_path: '/metrics'
```text

##  Security Deployment

###  SSL/TLS Configuration
```bash
# Generate production certificates
openssl req -x509 -newkey rsa:4096 -keyout xorb.key -out xorb.crt \
  -days 365 -nodes -subj "/CN=ptaas.company.com"

# Deploy certificates
kubectl create secret tls xorb-tls-secret \
  --cert=xorb.crt --key=xorb.key
```text

###  Network Security
```bash
# Firewall configuration
ufw allow 22/tcp    # SSH
ufw allow 443/tcp   # HTTPS
ufw allow 8000/tcp  # API (internal only)
ufw deny 5432/tcp   # PostgreSQL (internal only)
ufw deny 6379/tcp   # Redis (internal only)
```text

##  Backup & Disaster Recovery

###  Database Backup
```bash
# Automated PostgreSQL backup
pg_dump -h postgres -U xorb -d xorb_production | \
  gzip > /backups/xorb_$(date +%Y%m%d_%H%M%S).sql.gz

# Redis backup
redis-cli --rdb /backups/redis_$(date +%Y%m%d_%H%M%S).rdb
```text

###  Service Recovery
```bash
# Restore from backup
gunzip -c /backups/xorb_latest.sql.gz | \
  psql -h postgres -U xorb -d xorb_production

# Restart services with health checks
docker-compose restart
./tools/scripts/validate_environment.py
```text

##  Performance Tuning

###  Database Optimization
```sql
- - PostgreSQL performance tuning
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
SELECT pg_reload_conf();
```text

###  Application Scaling
```bash
# Horizontal scaling
docker-compose scale api=3 orchestrator=2

# Resource limits
docker update --memory=2g --cpus="1.5" xorb-api
```text