#  Deployment Guide

This guide provides comprehensive instructions for deploying the XORB Red/Blue Agent Framework in different environments, from development to production.

##  🎯 Deployment Overview

The XORB Red/Blue Agent Framework supports multiple deployment scenarios:

- **Development**: Single-machine setup for development and testing
- **Staging**: Multi-container setup for integration testing
- **Production**: Highly available, scalable deployment
- **Kubernetes**: Container orchestration for enterprise deployment
- **Hybrid Cloud**: Multi-cloud deployment with edge computing

##  🚀 Quick Start Deployment

###  Prerequisites Validation

Before deploying, ensure your system meets the requirements:

```bash
#  Check Docker version (20.10+ required)
docker --version

#  Check Docker Compose version (2.0+ required)
docker-compose --version

#  Check available resources
docker system df
free -h
df -h

#  Validate network connectivity
ping -c 3 google.com
```

###  Environment Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/xorb-security/red-blue-agents.git
   cd red-blue-agents
   ```

2. **Create Environment Configuration**
   ```bash
   # Copy example environment file
   cp .env.example .env

   # Generate secure secrets
   python3 scripts/generate-secrets.py

   # Edit configuration
   vim .env
   ```

3. **Initialize Infrastructure**
   ```bash
   # Start core infrastructure services
   docker-compose -f docker-compose.red-blue-agents.yml up -d postgres redis temporal

   # Wait for services to be ready
   scripts/wait-for-services.sh

   # Initialize database schema
   docker-compose exec agent-scheduler python -m alembic upgrade head
   ```

###  Framework Deployment

####  Development Deployment

For development and testing:

```bash
#  Start all services with development profile
docker-compose -f docker-compose.red-blue-agents.yml --profile development up -d

#  Verify deployment
curl http://localhost:8000/red-blue-agents/health

#  View logs
docker-compose -f docker-compose.red-blue-agents.yml logs -f agent-scheduler
```

####  Production Deployment

For production environments:

```bash
#  Set production environment
export XORB_ENV=production
export DEBUG=false

#  Start core services with production profile
docker-compose -f docker-compose.red-blue-agents.yml --profile production up -d

#  Enable monitoring
docker-compose -f docker-compose.red-blue-agents.yml --profile monitoring up -d

#  Verify deployment
scripts/health-check.sh
```

##  🏗️ Architecture Deployment

###  Component Overview

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[Traefik/Nginx]
    end

    subgraph "Application Layer"
        AS[Agent Scheduler]
        SO[Sandbox Orchestrator]
        TC[Telemetry Collector]
    end

    subgraph "Agent Sandboxes"
        RA[Red Agents]
        BA[Blue Agents]
    end

    subgraph "Data Layer"
        PG[(PostgreSQL)]
        RD[(Redis)]
        TM[Temporal]
    end

    subgraph "Monitoring"
        PR[Prometheus]
        GR[Grafana]
        JG[Jaeger]
    end

    LB --> AS
    LB --> SO
    LB --> TC

    AS --> RA
    AS --> BA
    SO --> RA
    SO --> BA

    AS --> PG
    AS --> RD
    AS --> TM
    TC --> PG
    TC --> RD

    AS --> PR
    SO --> PR
    TC --> PR
    PR --> GR
```

###  Service Dependencies

```yaml
#  Service startup order
1. Infrastructure Layer:
   - PostgreSQL (database)
   - Redis (caching)
   - Temporal (workflows)

2. Core Services:
   - Agent Scheduler
   - Sandbox Orchestrator
   - Telemetry Collector

3. Agent Containers:
   - Red Team Agents (on-demand)
   - Blue Team Agents (on-demand)

4. Monitoring Stack:
   - Prometheus
   - Grafana
   - Jaeger (optional)
```

##  🔧 Configuration Management

###  Environment-Specific Configurations

####  Development Environment

```yaml
#  .env.development
XORB_ENV=development
DEBUG=true
LOG_LEVEL=DEBUG

#  Relaxed security for development
JWT_SECRET=dev_jwt_secret_key
ENCRYPTION_KEY=dev_encryption_key_32_chars_long

#  Local services
DATABASE_URL=postgresql://xorb:dev_password@localhost:5432/xorb_agents_dev
REDIS_URL=redis://localhost:6379/0

#  Resource limits
MAX_SANDBOXES_GLOBAL=20
DEFAULT_SANDBOX_TTL=7200

#  Learning settings
ML_ENABLED=true
ANALYTICS_ENABLED=true
```

####  Staging Environment

```yaml
#  .env.staging
XORB_ENV=staging
DEBUG=false
LOG_LEVEL=INFO

#  Secure secrets (use secret management)
JWT_SECRET=${VAULT_JWT_SECRET}
ENCRYPTION_KEY=${VAULT_ENCRYPTION_KEY}

#  Staging infrastructure
DATABASE_URL=postgresql://xorb:${DB_PASSWORD}@postgres-staging:5432/xorb_agents_staging
REDIS_URL=redis://:${REDIS_PASSWORD}@redis-staging:6379/0

#  Moderate resource limits
MAX_SANDBOXES_GLOBAL=50
DEFAULT_SANDBOX_TTL=3600

#  Full feature set
ML_ENABLED=true
ANALYTICS_ENABLED=true
PROMETHEUS_ENABLED=true
```

####  Production Environment

```yaml
#  .env.production
XORB_ENV=production
DEBUG=false
LOG_LEVEL=WARNING

#  Production secrets (from HashiCorp Vault)
JWT_SECRET=${VAULT_JWT_SECRET}
ENCRYPTION_KEY=${VAULT_ENCRYPTION_KEY}

#  Production infrastructure (HA setup)
DATABASE_URL=postgresql://xorb:${DB_PASSWORD}@postgres-cluster:5432/xorb_agents
REDIS_URL=redis://:${REDIS_PASSWORD}@redis-cluster:6379/0

#  Production resource limits
MAX_SANDBOXES_GLOBAL=200
DEFAULT_SANDBOX_TTL=1800

#  Production features
ML_ENABLED=true
ANALYTICS_ENABLED=true
PROMETHEUS_ENABLED=true
AUDIT_LOG_ENABLED=true
```

###  Secret Management

####  HashiCorp Vault Integration

```bash
#  Initialize Vault for production
vault auth -method=userpass username=xorb-admin

#  Store database credentials
vault kv put secret/xorb-agents/db \
  username=xorb \
  password="$(openssl rand -base64 32)"

#  Store application secrets
vault kv put secret/xorb-agents/app \
  jwt_secret="$(openssl rand -base64 64)" \
  encryption_key="$(openssl rand -base64 32)"

#  Store external API keys
vault kv put secret/xorb-agents/external \
  virustotal_api_key="${VIRUSTOTAL_API_KEY}" \
  shodan_api_key="${SHODAN_API_KEY}"
```

####  Kubernetes Secrets

```yaml
#  k8s-secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: xorb-agents-secrets
  namespace: xorb-agents
type: Opaque
data:
  jwt-secret: <base64-encoded-secret>
  encryption-key: <base64-encoded-key>
  database-password: <base64-encoded-password>
  redis-password: <base64-encoded-password>
```

##  ☸️ Kubernetes Deployment

###  Namespace Setup

```bash
#  Create namespace
kubectl create namespace xorb-agents

#  Apply RBAC
kubectl apply -f k8s/rbac.yaml

#  Apply network policies
kubectl apply -f k8s/network-policies.yaml
```

###  Helm Chart Deployment

```bash
#  Add XORB Helm repository
helm repo add xorb https://charts.xorb-security.com
helm repo update

#  Deploy with custom values
helm install xorb-agents xorb/red-blue-agents \
  --namespace xorb-agents \
  --values values.production.yaml \
  --wait --timeout=600s

#  Verify deployment
kubectl get pods -n xorb-agents
kubectl get services -n xorb-agents
```

###  Custom Kubernetes Manifests

```yaml
#  k8s/agent-scheduler-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-scheduler
  namespace: xorb-agents
  labels:
    app: agent-scheduler
    component: core
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: agent-scheduler
  template:
    metadata:
      labels:
        app: agent-scheduler
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: xorb-agents
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        fsGroup: 1001
      containers:
      - name: agent-scheduler
        image: xorb/agent-scheduler:v2.1.0
        imagePullPolicy: IfNotPresent
        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        env:
        - name: XORB_ENV
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: xorb-agents-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: xorb-agents-secrets
              key: redis-url
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: xorb-agents-secrets
              key: jwt-secret
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
        volumeMounts:
        - name: config
          mountPath: /app/configs
          readOnly: true
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: config
        configMap:
          name: xorb-agents-config
      - name: logs
        emptyDir: {}
      nodeSelector:
        kubernetes.io/arch: amd64
      tolerations:
      - key: "xorb-agents"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
```

###  Service Mesh Integration (Istio)

```yaml
#  istio/virtual-service.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: xorb-agents
  namespace: xorb-agents
spec:
  hosts:
  - agents.xorb-security.com
  gateways:
  - xorb-agents-gateway
  http:
  - match:
    - uri:
        prefix: /red-blue-agents/
    route:
    - destination:
        host: agent-scheduler
        port:
          number: 8000
    fault:
      delay:
        percentage:
          value: 0.1
        fixedDelay: 5s
    retries:
      attempts: 3
      perTryTimeout: 30s
```

##  🌐 Cloud Platform Deployment

###  AWS Deployment

####  EKS Cluster Setup

```bash
#  Create EKS cluster
eksctl create cluster \
  --name xorb-agents-cluster \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type m5.xlarge \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed

#  Install AWS Load Balancer Controller
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  --set clusterName=xorb-agents-cluster \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller \
  -n kube-system

#  Deploy XORB Agents
helm install xorb-agents xorb/red-blue-agents \
  --namespace xorb-agents \
  --values values.aws.yaml
```

####  RDS Database Setup

```bash
#  Create RDS PostgreSQL instance
aws rds create-db-instance \
  --db-instance-identifier xorb-agents-db \
  --db-instance-class db.r5.large \
  --engine postgres \
  --engine-version 14.9 \
  --allocated-storage 100 \
  --storage-type gp2 \
  --storage-encrypted \
  --master-username xorb \
  --master-user-password "$(openssl rand -base64 32)" \
  --vpc-security-group-ids sg-xxxxxxxx \
  --db-subnet-group-name xorb-agents-subnet-group \
  --backup-retention-period 7 \
  --multi-az \
  --deletion-protection
```

####  ElastiCache Redis Setup

```bash
#  Create ElastiCache Redis cluster
aws elasticache create-replication-group \
  --replication-group-id xorb-agents-redis \
  --description "XORB Agents Redis Cluster" \
  --node-type cache.r6g.large \
  --engine redis \
  --engine-version 7.0 \
  --num-cache-clusters 3 \
  --cache-parameter-group default.redis7 \
  --cache-subnet-group-name xorb-agents-cache-subnet \
  --security-group-ids sg-yyyyyyyy \
  --at-rest-encryption-enabled \
  --transit-encryption-enabled
```

###  Azure Deployment

####  AKS Cluster Setup

```bash
#  Create resource group
az group create \
  --name xorb-agents-rg \
  --location westus2

#  Create AKS cluster
az aks create \
  --resource-group xorb-agents-rg \
  --name xorb-agents-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-addons monitoring \
  --enable-managed-identity \
  --generate-ssh-keys

#  Get credentials
az aks get-credentials \
  --resource-group xorb-agents-rg \
  --name xorb-agents-cluster
```

####  Azure Database for PostgreSQL

```bash
#  Create PostgreSQL server
az postgres flexible-server create \
  --resource-group xorb-agents-rg \
  --name xorb-agents-db \
  --location westus2 \
  --admin-user xorb \
  --admin-password "$(openssl rand -base64 32)" \
  --sku-name Standard_D4s_v3 \
  --tier GeneralPurpose \
  --storage-size 128 \
  --version 14
```

###  Google Cloud Platform Deployment

####  GKE Cluster Setup

```bash
#  Create GKE cluster
gcloud container clusters create xorb-agents-cluster \
  --zone us-central1-a \
  --machine-type e2-standard-4 \
  --num-nodes 3 \
  --enable-autorepair \
  --enable-autoupgrade \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10

#  Get credentials
gcloud container clusters get-credentials xorb-agents-cluster \
  --zone us-central1-a
```

####  Cloud SQL PostgreSQL

```bash
#  Create Cloud SQL instance
gcloud sql instances create xorb-agents-db \
  --database-version POSTGRES_14 \
  --tier db-standard-4 \
  --region us-central1 \
  --storage-size 100GB \
  --storage-type SSD \
  --storage-auto-increase \
  --backup-start-time 02:00 \
  --maintenance-window-day SAT \
  --maintenance-window-hour 06
```

##  🔍 Health Checks and Monitoring

###  Application Health Checks

```bash
# !/bin/bash
#  scripts/health-check.sh

echo "🏥 XORB Red/Blue Agent Framework Health Check"
echo "============================================="

#  Check core services
services=(
  "agent-scheduler:8000"
  "sandbox-orchestrator:8001"
  "telemetry-collector:8002"
)

for service in "${services[@]}"; do
  name="${service%:*}"
  port="${service#*:}"

  echo -n "Checking $name... "

  if curl -sf "http://localhost:$port/health" > /dev/null; then
    echo "✅ Healthy"
  else
    echo "❌ Unhealthy"
    exit 1
  fi
done

#  Check infrastructure
echo -n "Checking PostgreSQL... "
if docker-compose exec -T postgres pg_isready -U xorb -d xorb_agents > /dev/null; then
  echo "✅ Healthy"
else
  echo "❌ Unhealthy"
  exit 1
fi

echo -n "Checking Redis... "
if docker-compose exec -T redis redis-cli ping > /dev/null; then
  echo "✅ Healthy"
else
  echo "❌ Unhealthy"
  exit 1
fi

echo -n "Checking Temporal... "
if curl -sf "http://localhost:8233" > /dev/null; then
  echo "✅ Healthy"
else
  echo "❌ Unhealthy"
  exit 1
fi

echo ""
echo "🎉 All services are healthy!"
```

###  Prometheus Monitoring

```yaml
#  configs/prometheus/prometheus-agents.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules.yml"

scrape_configs:
  - job_name: 'xorb-agents'
    static_configs:
      - targets:
        - 'agent-scheduler:8000'
        - 'sandbox-orchestrator:8001'
        - 'telemetry-collector:8002'
    metrics_path: /metrics
    scrape_interval: 15s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

###  Alert Rules

```yaml
#  configs/prometheus/rules-agents.yml
groups:
  - name: xorb-agents
    rules:
      - alert: ServiceDown
        expr: up == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "XORB Agent service is down"
          description: "{{ $labels.job }} has been down for more than 5 minutes"

      - alert: HighMemoryUsage
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 90
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 90% for more than 10 minutes"

      - alert: MissionFailureRate
        expr: rate(missions_failed_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High mission failure rate"
          description: "Mission failure rate is above 10% for the last 5 minutes"

      - alert: SandboxResourceExhaustion
        expr: sandbox_utilization_percent > 90
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Sandbox resources exhausted"
          description: "Sandbox resource utilization is above 90%"
```

##  🔒 Security Hardening

###  Container Security

```dockerfile
#  Security-hardened Dockerfile example
FROM python:3.11-slim

#  Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

#  Install security updates only
RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

#  Copy application with minimal permissions
COPY --chown=appuser:appuser . /app
WORKDIR /app

#  Switch to non-root user
USER appuser

#  Run with minimal capabilities
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

###  Network Security

```yaml
#  Network policies for Kubernetes
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: xorb-agents-network-policy
  namespace: xorb-agents
spec:
  podSelector:
    matchLabels:
      app: agent-scheduler
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
          name: xorb-agents
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
```

###  Secret Rotation

```bash
# !/bin/bash
#  scripts/rotate-secrets.sh

echo "🔄 Rotating XORB Agent secrets..."

#  Generate new secrets
NEW_JWT_SECRET=$(openssl rand -base64 64)
NEW_ENCRYPTION_KEY=$(openssl rand -base64 32)

#  Update Vault
vault kv put secret/xorb-agents/app \
  jwt_secret="$NEW_JWT_SECRET" \
  encryption_key="$NEW_ENCRYPTION_KEY"

#  Update Kubernetes secrets
kubectl create secret generic xorb-agents-secrets-new \
  --from-literal=jwt-secret="$NEW_JWT_SECRET" \
  --from-literal=encryption-key="$NEW_ENCRYPTION_KEY" \
  --namespace=xorb-agents

#  Rolling update deployments
kubectl patch deployment agent-scheduler \
  --namespace=xorb-agents \
  --patch='{"spec":{"template":{"spec":{"volumes":[{"name":"secrets","secret":{"secretName":"xorb-agents-secrets-new"}}]}}}}'

#  Wait for rollout
kubectl rollout status deployment/agent-scheduler --namespace=xorb-agents

echo "✅ Secret rotation complete"
```

##  📈 Scaling and Performance

###  Horizontal Pod Autoscaling

```yaml
#  k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-scheduler-hpa
  namespace: xorb-agents
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-scheduler
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
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

###  Database Performance Tuning

```sql
-- PostgreSQL performance optimization
-- /configs/postgres/postgresql.conf

#  Memory settings
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
work_mem = 4MB

#  Checkpoint settings
checkpoint_completion_target = 0.9
wal_buffers = 16MB
checkpoint_timeout = 10min

#  Query planner
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200

#  Logging
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on
log_temp_files = 0

#  Autovacuum
autovacuum = on
autovacuum_max_workers = 3
autovacuum_naptime = 1min
```

###  Redis Performance Tuning

```bash
#  Redis configuration optimizations
#  /configs/redis/redis.conf

#  Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

#  Persistence
save 900 1
save 300 10
save 60 10000
appendonly yes
appendfsync everysec

#  Network
tcp-keepalive 300
timeout 0

#  Performance
lazyfree-lazy-eviction yes
lazyfree-lazy-expire yes
lazyfree-lazy-server-del yes
replica-lazy-flush yes
```

##  🔄 Backup and Recovery

###  Automated Backup Strategy

```bash
# !/bin/bash
#  scripts/backup.sh

BACKUP_DIR="/backups/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

echo "📦 Starting XORB Agents backup..."

#  Database backup
echo "Backing up PostgreSQL..."
docker-compose exec -T postgres pg_dump -U xorb xorb_agents | gzip > "$BACKUP_DIR/postgres.sql.gz"

#  Redis backup
echo "Backing up Redis..."
docker-compose exec redis redis-cli BGSAVE
docker cp "$(docker-compose ps -q redis)":/data/dump.rdb "$BACKUP_DIR/redis.rdb"

#  Configuration backup
echo "Backing up configurations..."
tar -czf "$BACKUP_DIR/configs.tar.gz" configs/

#  Models backup
echo "Backing up ML models..."
tar -czf "$BACKUP_DIR/models.tar.gz" models/

#  Upload to cloud storage
echo "Uploading to S3..."
aws s3 sync "$BACKUP_DIR" "s3://xorb-agents-backups/$(date +%Y%m%d)/" --storage-class STANDARD_IA

echo "✅ Backup complete"

#  Cleanup old backups (keep 30 days)
find /backups -type d -mtime +30 -exec rm -rf {} \;
```

###  Disaster Recovery

```bash
# !/bin/bash
#  scripts/restore.sh

BACKUP_DATE="${1:-$(date +%Y%m%d)}"
BACKUP_DIR="/backups/$BACKUP_DATE"

echo "🔄 Starting XORB Agents restore from $BACKUP_DATE..."

#  Download from cloud storage
aws s3 sync "s3://xorb-agents-backups/$BACKUP_DATE/" "$BACKUP_DIR/"

#  Stop services
docker-compose down

#  Restore database
echo "Restoring PostgreSQL..."
docker-compose up -d postgres
sleep 30
gunzip -c "$BACKUP_DIR/postgres.sql.gz" | docker-compose exec -T postgres psql -U xorb xorb_agents

#  Restore Redis
echo "Restoring Redis..."
docker cp "$BACKUP_DIR/redis.rdb" "$(docker-compose ps -q redis)":/data/dump.rdb
docker-compose restart redis

#  Restore configurations
echo "Restoring configurations..."
tar -xzf "$BACKUP_DIR/configs.tar.gz"

#  Restore models
echo "Restoring ML models..."
tar -xzf "$BACKUP_DIR/models.tar.gz"

#  Start services
docker-compose up -d

echo "✅ Restore complete"
```

##  🧪 Testing Deployment

###  Smoke Tests

```bash
# !/bin/bash
#  scripts/smoke-test.sh

echo "🧪 Running smoke tests..."

#  Test API endpoints
curl -f http://localhost:8000/red-blue-agents/health || exit 1
curl -f http://localhost:8000/red-blue-agents/capabilities/agents || exit 1

#  Test mission creation
MISSION_ID=$(curl -s -X POST http://localhost:8000/red-blue-agents/missions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TEST_TOKEN" \
  -d '{
    "name": "Smoke Test Mission",
    "description": "Automated smoke test",
    "environment": "development",
    "objectives": ["Test basic functionality"],
    "targets": [{"type": "web_app", "host": "httpbin.org", "ports": [80, 443]}]
  }' | jq -r '.mission_id')

echo "Created test mission: $MISSION_ID"

#  Wait for mission to start
sleep 10

#  Check mission status
STATUS=$(curl -s http://localhost:8000/red-blue-agents/missions/$MISSION_ID \
  -H "Authorization: Bearer $TEST_TOKEN" | jq -r '.status')

echo "Mission status: $STATUS"

#  Cleanup
curl -s -X POST http://localhost:8000/red-blue-agents/missions/$MISSION_ID/stop \
  -H "Authorization: Bearer $TEST_TOKEN"

echo "✅ Smoke tests passed"
```

###  Load Testing

```bash
# !/bin/bash
#  scripts/load-test.sh

echo "🚀 Running load tests..."

#  Install k6 if not available
if ! command -v k6 &> /dev/null; then
  echo "Installing k6..."
  sudo apt-get install k6
fi

#  Run load test
k6 run --vus 50 --duration 5m scripts/load-test.js

echo "✅ Load tests complete"
```

```javascript
// scripts/load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  vus: 50,
  duration: '5m',
  thresholds: {
    http_req_duration: ['p(95)<2000'],
    http_req_failed: ['rate<0.1'],
  },
};

export default function() {
  let response = http.get('http://localhost:8000/red-blue-agents/health');

  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });

  sleep(1);
}
```

##  📚 Deployment Troubleshooting

###  Common Issues

####  Service Startup Failures

```bash
#  Check service logs
docker-compose logs agent-scheduler

#  Check resource usage
docker stats

#  Verify environment variables
docker-compose exec agent-scheduler env | grep XORB

#  Test database connectivity
docker-compose exec agent-scheduler python -c "
import asyncpg
import asyncio
async def test():
    conn = await asyncpg.connect('$DATABASE_URL')
    print('Database connection successful')
    await conn.close()
asyncio.run(test())
"
```

####  Network Connectivity Issues

```bash
#  Test internal networking
docker-compose exec agent-scheduler ping postgres
docker-compose exec agent-scheduler ping redis

#  Check port availability
netstat -tulpn | grep :8000
netstat -tulpn | grep :5432
netstat -tulpn | grep :6379

#  Test external connectivity
docker-compose exec agent-scheduler curl -I https://google.com
```

####  Performance Issues

```bash
#  Monitor resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

#  Check database performance
docker-compose exec postgres psql -U xorb -d xorb_agents -c "
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;"

#  Monitor Redis performance
docker-compose exec redis redis-cli info stats
```

###  Debug Mode Deployment

```bash
#  Enable debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG

#  Start with verbose logging
docker-compose -f docker-compose.red-blue-agents.yml up --build

#  Attach to running container for debugging
docker-compose exec agent-scheduler bash

#  Check application logs in detail
tail -f logs/agent-scheduler.log
```

##  🎯 Production Checklist

Before deploying to production, ensure:

###  Security Checklist
- [ ] All default passwords changed
- [ ] TLS/SSL certificates configured
- [ ] Network policies implemented
- [ ] Secret management configured
- [ ] Container security scanning passed
- [ ] Vulnerability assessment completed

###  Performance Checklist
- [ ] Resource limits configured
- [ ] Auto-scaling policies set
- [ ] Database performance tuned
- [ ] Cache configuration optimized
- [ ] Load testing completed
- [ ] Monitoring dashboards configured

###  Reliability Checklist
- [ ] Health checks implemented
- [ ] Backup strategy configured
- [ ] Disaster recovery tested
- [ ] Circuit breakers configured
- [ ] Retry policies implemented
- [ ] Graceful shutdown handling

###  Compliance Checklist
- [ ] Audit logging enabled
- [ ] Data retention policies configured
- [ ] Privacy controls implemented
- [ ] Compliance scanning passed
- [ ] Security policies enforced
- [ ] Documentation updated

---

For environment-specific deployment guides, see:
- [AWS Deployment Guide](./deployment/aws.md)
- [Azure Deployment Guide](./deployment/azure.md)
- [GCP Deployment Guide](./deployment/gcp.md)
- [On-Premises Deployment Guide](./deployment/on-premises.md)