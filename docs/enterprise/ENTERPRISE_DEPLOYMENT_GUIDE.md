# XORB Enterprise Deployment Guide

- **Version**: 2.0
- **Date**: January 2025
- **Audience**: Enterprise IT Teams, DevOps Engineers, Security Teams

##  🎯 **Executive Summary**

XORB is an enterprise-grade cybersecurity platform providing Penetration Testing as a Service (PTaaS), AI-powered threat intelligence, and comprehensive security orchestration. This guide provides complete deployment instructions for Fortune 500 environments.

##  📋 **Prerequisites**

###  **Infrastructure Requirements**

| Component | Minimum | Recommended | Enterprise |
|-----------|---------|-------------|------------|
| **CPU** | 8 cores | 16 cores | 32+ cores |
| **Memory** | 16 GB | 32 GB | 64+ GB |
| **Storage** | 100 GB SSD | 500 GB SSD | 1 TB+ NVMe |
| **Network** | 1 Gbps | 10 Gbps | 25+ Gbps |

###  **Software Dependencies**

```yaml
Required Components:
  - Kubernetes: v1.28+
  - Docker: v24.0+
  - PostgreSQL: v15+ with pgvector
  - Redis: v7.0+
  - Temporal: v1.6+
  - HashiCorp Vault: v1.15+

Optional Components:
  - Prometheus: v2.45+
  - Grafana: v10.0+
  - Jaeger: v1.49+
  - ElasticSearch: v8.0+ (for log aggregation)
```text

###  **Network Requirements**

```yaml
Firewall Rules:
  Inbound:
    - 80/tcp (HTTP - redirect to HTTPS)
    - 443/tcp (HTTPS - main application)
    - 22/tcp (SSH - admin access)
    - 8080/tcp (Orchestrator UI)
    - 9090/tcp (Prometheus - monitoring)
    - 3000/tcp (Grafana - dashboards)

  Outbound:
    - 443/tcp (External API calls)
    - 53/tcp,udp (DNS resolution)
    - 25/tcp (SMTP - notifications)
    - 123/udp (NTP - time sync)

Internal Communication:
  - PostgreSQL: 5432/tcp
  - Redis: 6379/tcp
  - Temporal: 7233/tcp
  - Vault: 8200/tcp
```text

##  🚀 **Deployment Options**

###  **Option 1: Kubernetes Deployment (Recommended)**

####  **1.1 Prepare Kubernetes Cluster**

```bash
# Create namespace
kubectl create namespace xorb-production

# Create service accounts
kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: xorb-api
  namespace: xorb-production
- --
apiVersion: v1
kind: ServiceAccount
metadata:
  name: xorb-orchestrator
  namespace: xorb-production
EOF
```text

####  **1.2 Deploy Infrastructure Components**

```bash
# Deploy PostgreSQL with pgvector
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install xorb-postgres bitnami/postgresql \
  --namespace xorb-production \
  --set auth.postgresPassword=SecurePassword123! \
  --set auth.database=xorb_production \
  --set primary.persistence.size=100Gi \
  --set primary.resources.requests.memory=2Gi \
  --set primary.resources.requests.cpu=1000m

# Deploy Redis Cluster
helm install xorb-redis bitnami/redis \
  --namespace xorb-production \
  --set auth.password=SecureRedisPassword123! \
  --set replica.replicaCount=3 \
  --set sentinel.enabled=true

# Deploy Temporal
helm repo add temporalio https://go.temporal.io/helm-charts
helm install xorb-temporal temporalio/temporal \
  --namespace xorb-production \
  --set server.replicaCount=3 \
  --set cassandra.config.cluster_size=3 \
  --set web.enabled=true
```text

####  **1.3 Deploy XORB Platform**

```bash
# Clone repository
git clone https://github.com/your-org/xorb.git
cd xorb

# Create configuration secrets
kubectl create secret generic xorb-config \
  --namespace xorb-production \
  --from-literal=database-url="postgresql://postgres:SecurePassword123!@xorb-postgres:5432/xorb_production" \
  --from-literal=redis-url="redis://:SecureRedisPassword123!@xorb-redis-master:6379" \
  --from-literal=temporal-host="xorb-temporal-frontend:7233" \
  --from-literal=jwt-secret="your-jwt-secret-here" \
  --from-literal=nvidia-api-key="your-nvidia-key" \
  --from-literal=openrouter-api-key="your-openrouter-key"

# Deploy XORB services
kubectl apply -f deploy/kubernetes/production/

# Verify deployment
kubectl get pods -n xorb-production
kubectl get services -n xorb-production
```text

###  **Option 2: Docker Compose Deployment**

####  **2.1 Production Docker Compose**

```bash
# Clone repository
git clone https://github.com/your-org/xorb.git
cd xorb

# Copy environment template
cp .env.template .env.production

# Edit configuration
vim .env.production
```text

```env
# .env.production
ENVIRONMENT=production
DATABASE_URL=postgresql://postgres:SecurePassword123!@postgres:5432/xorb_production
REDIS_URL=redis://:SecureRedisPassword123!@redis:6379
TEMPORAL_HOST=temporal:7233
LOG_LEVEL=INFO
JWT_SECRET=your-super-secure-jwt-secret-here
NVIDIA_API_KEY=your-nvidia-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
RATE_LIMIT_PER_MINUTE=1000
RATE_LIMIT_PER_HOUR=10000
CORS_ALLOW_ORIGINS=https://your-domain.com
ENABLE_METRICS=true
```text

```bash
# Deploy with production configuration
docker-compose -f docker-compose.enterprise.yml up -d

# Verify deployment
docker-compose -f docker-compose.enterprise.yml ps
docker-compose -f docker-compose.enterprise.yml logs
```text

##  🔐 **Security Configuration**

###  **SSL/TLS Configuration**

```yaml
# ingress-tls.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: xorb-ingress
  namespace: xorb-production
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.your-domain.com
    - app.your-domain.com
    secretName: xorb-tls-secret
  rules:
  - host: api.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: xorb-api
            port:
              number: 80
```text

###  **Vault Secret Management**

```bash
# Initialize Vault
kubectl exec -it vault-0 -n xorb-production -- vault operator init

# Configure authentication
kubectl exec -it vault-0 -n xorb-production -- vault auth enable kubernetes

# Create policies
kubectl exec -it vault-0 -n xorb-production -- vault policy write xorb-policy - <<EOF
path "secret/xorb/*" {
  capabilities = ["read", "list"]
}
path "database/creds/xorb-app" {
  capabilities = ["read"]
}
EOF

# Store secrets
kubectl exec -it vault-0 -n xorb-production -- vault kv put secret/xorb/config \
  jwt_secret="your-jwt-secret" \
  database_password="SecurePassword123!" \
  redis_password="SecureRedisPassword123!"
```text

###  **Network Policies**

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: xorb-network-policy
  namespace: xorb-production
spec:
  podSelector:
    matchLabels:
      app: xorb
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: xorb-production
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
```text

##  📊 **Monitoring & Observability**

###  **Prometheus Configuration**

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: xorb-production
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s

    scrape_configs:
    - job_name: 'xorb-api'
      static_configs:
      - targets: ['xorb-api:8000']
      metrics_path: /metrics
      scrape_interval: 10s

    - job_name: 'xorb-orchestrator'
      static_configs:
      - targets: ['xorb-orchestrator:8080']
      metrics_path: /metrics
      scrape_interval: 10s

    - job_name: 'postgres-exporter'
      static_configs:
      - targets: ['postgres-exporter:9187']

    - job_name: 'redis-exporter'
      static_configs:
      - targets: ['redis-exporter:9121']
```text

###  **Grafana Dashboards**

```bash
# Import XORB dashboards
kubectl create configmap xorb-dashboards \
  --namespace xorb-production \
  --from-file=services/infrastructure/monitoring/grafana/dashboards/

# Configure Grafana data sources
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-datasources
  namespace: xorb-production
data:
  datasources.yaml: |
    apiVersion: 1
    datasources:
    - name: Prometheus
      type: prometheus
      url: http://prometheus:9090
      access: proxy
      isDefault: true
EOF
```text

##  🔄 **Database Setup**

###  **PostgreSQL Initialization**

```sql
- - Connect to PostgreSQL
psql -h localhost -U postgres -d xorb_production

- - Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

- - Create application user
CREATE USER xorb_app WITH PASSWORD 'SecureAppPassword123!';

- - Grant permissions
GRANT CONNECT ON DATABASE xorb_production TO xorb_app;
GRANT USAGE ON SCHEMA public TO xorb_app;
GRANT CREATE ON SCHEMA public TO xorb_app;

- - Create initial tables (run migrations)
- - This would typically be done via Alembic migrations
```text

###  **Database Migrations**

```bash
# Run database migrations
kubectl exec -it xorb-api-0 -n xorb-production -- \
  python -m alembic upgrade head

# Verify migrations
kubectl exec -it xorb-api-0 -n xorb-production -- \
  python -m alembic current
```text

##  🧪 **Testing Deployment**

###  **Health Checks**

```bash
# API health check
curl -f https://api.your-domain.com/health

# Readiness check
curl -f https://api.your-domain.com/readiness

# Orchestrator health
curl -f https://api.your-domain.com:8080/health

# Database connectivity
kubectl exec -it xorb-api-0 -n xorb-production -- \
  python -c "import asyncpg; import asyncio; asyncio.run(asyncpg.connect('${DATABASE_URL}').close())"
```text

###  **Functional Testing**

```bash
# Test API authentication
curl -X POST https://api.your-domain.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "test123"}'

# Test PTaaS functionality
curl -X POST https://api.your-domain.com/ptaas/scan \
  -H "Authorization: Bearer ${JWT_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"targets": [{"host": "test.example.com", "ports": [80, 443]}], "scan_type": "quick"}'

# Test threat intelligence
curl -X GET https://api.your-domain.com/intelligence/threats \
  -H "Authorization: Bearer ${JWT_TOKEN}"
```text

###  **Performance Testing**

```bash
# Install testing tools
kubectl run load-test --image=loadimpact/k6:latest --rm -it --restart=Never -- \
  run - <<EOF
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 },
    { duration: '5m', target: 100 },
    { duration: '2m', target: 200 },
    { duration: '5m', target: 200 },
    { duration: '2m', target: 0 },
  ],
};

export default function () {
  let response = http.get('https://api.your-domain.com/health');
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
  sleep(1);
}
EOF
```text

##  🔧 **Configuration Management**

###  **Environment Variables**

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: xorb-config
  namespace: xorb-production
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  RATE_LIMIT_PER_MINUTE: "1000"
  RATE_LIMIT_PER_HOUR: "10000"
  ENABLE_METRICS: "true"
  CORS_ALLOW_ORIGINS: "https://your-domain.com"
  MAX_WORKERS: "100"
  CACHE_TTL: "3600"
  DATABASE_POOL_SIZE: "20"
  REDIS_POOL_SIZE: "10"
```text

###  **Multi-Environment Support**

```bash
# Staging environment
kubectl create namespace xorb-staging
kubectl apply -f deploy/kubernetes/staging/ -n xorb-staging

# Development environment
kubectl create namespace xorb-development
kubectl apply -f deploy/kubernetes/development/ -n xorb-development
```text

##  📈 **Scaling Configuration**

###  **Horizontal Pod Autoscaling**

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: xorb-api-hpa
  namespace: xorb-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: xorb-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```text

###  **Vertical Pod Autoscaling**

```yaml
# vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: xorb-api-vpa
  namespace: xorb-production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: xorb-api
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: xorb-api
      maxAllowed:
        cpu: 2
        memory: 4Gi
      minAllowed:
        cpu: 100m
        memory: 256Mi
```text

##  🔄 **Backup & Disaster Recovery**

###  **Database Backup**

```bash
# Create backup job
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: xorb-production
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: postgres-backup
            image: postgres:15
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: xorb-config
                  key: database-password
            command:
            - /bin/bash
            - -c
            - |
              DATE=$(date +%Y%m%d_%H%M%S)
              pg_dump -h xorb-postgres -U postgres xorb_production > /backup/xorb_backup_\$DATE.sql
              # Upload to S3 or other storage
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
EOF
```text

###  **Application State Backup**

```bash
# Backup secrets and configurations
kubectl get secrets -n xorb-production -o yaml > xorb-secrets-backup.yaml
kubectl get configmaps -n xorb-production -o yaml > xorb-config-backup.yaml
kubectl get persistentvolumeclaims -n xorb-production -o yaml > xorb-pvc-backup.yaml
```text

##  🚨 **Troubleshooting**

###  **Common Issues**

####  **1. Pod Startup Issues**

```bash
# Check pod status
kubectl describe pod xorb-api-0 -n xorb-production

# Check logs
kubectl logs xorb-api-0 -n xorb-production --follow

# Check events
kubectl get events -n xorb-production --sort-by='.lastTimestamp'
```text

####  **2. Database Connection Issues**

```bash
# Test database connectivity
kubectl exec -it xorb-postgres-0 -n xorb-production -- \
  psql -U postgres -d xorb_production -c "SELECT version();"

# Check database logs
kubectl logs xorb-postgres-0 -n xorb-production
```text

####  **3. Performance Issues**

```bash
# Check resource usage
kubectl top pods -n xorb-production
kubectl top nodes

# Check HPA status
kubectl describe hpa xorb-api-hpa -n xorb-production

# Check metrics
curl -s https://api.your-domain.com/metrics | grep -E "(http_requests|response_time)"
```text

###  **Log Analysis**

```bash
# Centralized logging with ELK
kubectl logs -l app=xorb -n xorb-production --since=1h | \
  jq -r 'select(.level == "ERROR") | .message'

# Performance metrics
kubectl logs -l app=xorb-api -n xorb-production --since=10m | \
  grep "response_time" | awk '{print $NF}' | sort -n
```text

##  📞 **Support & Maintenance**

###  **Maintenance Windows**

```bash
# Planned maintenance procedure
1. Scale down to minimum replicas
2. Run maintenance tasks
3. Scale back up
4. Verify functionality

# Example maintenance scaling
kubectl scale deployment xorb-api --replicas=1 -n xorb-production
# Perform maintenance
kubectl scale deployment xorb-api --replicas=3 -n xorb-production
```text

###  **Update Procedures**

```bash
# Rolling update
kubectl set image deployment/xorb-api \
  xorb-api=ghcr.io/your-org/xorb:v2.1.0 \
  -n xorb-production

# Monitor rollout
kubectl rollout status deployment/xorb-api -n xorb-production

# Rollback if needed
kubectl rollout undo deployment/xorb-api -n xorb-production
```text

##  🏆 **Best Practices**

###  **Security Best Practices**
- Use least privilege access principles
- Regularly rotate secrets and certificates
- Enable audit logging for all components
- Implement network segmentation
- Use Pod Security Standards

###  **Performance Best Practices**
- Monitor resource usage and scale proactively
- Use connection pooling for databases
- Implement proper caching strategies
- Optimize database queries and indexes
- Use CDN for static assets

###  **Operational Best Practices**
- Implement comprehensive monitoring
- Set up automated alerting
- Maintain disaster recovery procedures
- Regular backup testing
- Document all procedures and configurations

- --

- *For additional support or questions, contact:**
- **Email**: enterprise-support@xorb-security.com
- **Documentation**: https://docs.xorb-security.com
- **Support Portal**: https://support.xorb-security.com