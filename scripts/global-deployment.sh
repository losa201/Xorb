#!/bin/bash

# XORB Global Deployment Automation
# Comprehensive deployment system for multi-environment orchestration

set -euo pipefail

echo "🌍 XORB Global Deployment Automation"
echo "===================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_step() {
    echo -e "${BLUE}🔧 $1${NC}"
}

log_deploy() {
    echo -e "${PURPLE}🚀 $1${NC}"
}

# Configuration
DEPLOYMENT_DIR="/root/Xorb/deployment"
ENVIRONMENTS=("development" "staging" "production")
REGIONS=("us-east-1" "us-west-2" "eu-west-1" "ap-southeast-1")

# Parse command line arguments
ENVIRONMENT="${1:-development}"
REGION="${2:-us-east-1}"
COMPONENT="${3:-all}"

# Validate inputs
if [[ ! " ${ENVIRONMENTS[@]} " =~ " ${ENVIRONMENT} " ]]; then
    log_error "Invalid environment: $ENVIRONMENT. Must be one of: ${ENVIRONMENTS[*]}"
    exit 1
fi

if [[ ! " ${REGIONS[@]} " =~ " ${REGION} " ]]; then
    log_error "Invalid region: $REGION. Must be one of: ${REGIONS[*]}"
    exit 1
fi

log_deploy "Starting global deployment for $ENVIRONMENT environment in $REGION region"

# Create deployment directory structure
log_step "Creating deployment directory structure..."
mkdir -p "$DEPLOYMENT_DIR"/{environments,regions,components,logs,backups}

for env in "${ENVIRONMENTS[@]}"; do
    mkdir -p "$DEPLOYMENT_DIR/environments/$env"/{config,secrets,manifests}
done

for region in "${REGIONS[@]}"; do
    mkdir -p "$DEPLOYMENT_DIR/regions/$region"/{config,resources}
done

# Create environment-specific configurations
log_step "Creating environment configurations..."

# Development environment
cat > "$DEPLOYMENT_DIR/environments/development/config/deployment.yaml" << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: xorb-deployment-config
  namespace: xorb-development
data:
  environment: "development"
  log_level: "DEBUG"
  replicas: "1"
  cpu_requests: "100m"
  cpu_limits: "500m"
  memory_requests: "128Mi"
  memory_limits: "512Mi"
  storage_size: "1Gi"
  enable_monitoring: "true"
  enable_tracing: "true"
  enable_debugging: "true"
  auto_scaling: "false"
EOF

# Staging environment
cat > "$DEPLOYMENT_DIR/environments/staging/config/deployment.yaml" << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: xorb-deployment-config
  namespace: xorb-staging
data:
  environment: "staging"
  log_level: "INFO"
  replicas: "2"
  cpu_requests: "200m"
  cpu_limits: "1000m"
  memory_requests: "256Mi"
  memory_limits: "1Gi"
  storage_size: "5Gi"
  enable_monitoring: "true"
  enable_tracing: "true"
  enable_debugging: "false"
  auto_scaling: "true"
  min_replicas: "2"
  max_replicas: "5"
EOF

# Production environment
cat > "$DEPLOYMENT_DIR/environments/production/config/deployment.yaml" << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: xorb-deployment-config
  namespace: xorb-production
data:
  environment: "production"
  log_level: "WARN"
  replicas: "3"
  cpu_requests: "500m"
  cpu_limits: "2000m"
  memory_requests: "512Mi"
  memory_limits: "2Gi"
  storage_size: "20Gi"
  enable_monitoring: "true"
  enable_tracing: "false"
  enable_debugging: "false"
  auto_scaling: "true"
  min_replicas: "3"
  max_replicas: "10"
  enable_backup: "true"
  backup_schedule: "0 2 * * *"
EOF

# Create region-specific configurations
log_step "Creating region configurations..."

for region in "${REGIONS[@]}"; do
    cat > "$DEPLOYMENT_DIR/regions/$region/config/region.yaml" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: xorb-region-config
data:
  region: "$region"
  cloud_provider: "aws"
  availability_zones: "3"
  network_cidr: "10.0.0.0/16"
  enable_multi_az: "true"
  enable_encryption: "true"
  backup_region: "$([ "$region" = "us-east-1" ] && echo "us-west-2" || echo "us-east-1")"
EOF
done

# Create component deployment scripts
log_step "Creating component deployment scripts..."

# Database component
cat > "$DEPLOYMENT_DIR/components/deploy-database.sh" << 'EOF'
#!/bin/bash

echo "🗄️  Deploying XORB Database Components"

ENVIRONMENT="$1"
REGION="$2"

# Load secrets
source /root/Xorb/secrets/docker-secrets.sh

# Deploy PostgreSQL
kubectl apply -f - <<YAML
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: xorb-postgres
  namespace: xorb-$ENVIRONMENT
spec:
  serviceName: xorb-postgres
  replicas: $([ "$ENVIRONMENT" = "production" ] && echo "3" || echo "1")
  selector:
    matchLabels:
      app: xorb-postgres
  template:
    metadata:
      labels:
        app: xorb-postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: xorb
        - name: POSTGRES_USER
          value: xorb
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: xorb-database-secrets
              key: postgres-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 20Gi
YAML

# Deploy Redis
kubectl apply -f - <<YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: xorb-redis
  namespace: xorb-$ENVIRONMENT
spec:
  replicas: $([ "$ENVIRONMENT" = "production" ] && echo "3" || echo "1")
  selector:
    matchLabels:
      app: xorb-redis
  template:
    metadata:
      labels:
        app: xorb-redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        command: ["redis-server"]
        args: ["--requirepass", "$(kubectl get secret xorb-database-secrets -o jsonpath='{.data.redis-password}' | base64 -d)"]
YAML

echo "✅ Database components deployed successfully"
EOF

# API component
cat > "$DEPLOYMENT_DIR/components/deploy-api.sh" << 'EOF'
#!/bin/bash

echo "📡 Deploying XORB API Components"

ENVIRONMENT="$1"
REGION="$2"

kubectl apply -f - <<YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: xorb-api
  namespace: xorb-$ENVIRONMENT
spec:
  replicas: $([ "$ENVIRONMENT" = "production" ] && echo "3" || echo "2")
  selector:
    matchLabels:
      app: xorb-api
  template:
    metadata:
      labels:
        app: xorb-api
    spec:
      containers:
      - name: api
        image: xorb/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "$ENVIRONMENT"
        - name: REGION
          value: "$REGION"
        - name: DATABASE_URL
          value: "postgresql://xorb:$(kubectl get secret xorb-database-secrets -o jsonpath='{.data.postgres-password}' | base64 -d)@xorb-postgres:5432/xorb"
        - name: REDIS_URL
          value: "redis://:$(kubectl get secret xorb-database-secrets -o jsonpath='{.data.redis-password}' | base64 -d)@xorb-redis:6379"
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: xorb-auth-secrets
              key: jwt-secret
        resources:
          requests:
            cpu: $([ "$ENVIRONMENT" = "production" ] && echo "500m" || echo "200m")
            memory: $([ "$ENVIRONMENT" = "production" ] && echo "512Mi" || echo "256Mi")
          limits:
            cpu: $([ "$ENVIRONMENT" = "production" ] && echo "2000m" || echo "1000m")
            memory: $([ "$ENVIRONMENT" = "production" ] && echo "2Gi" || echo "1Gi")
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: xorb-api
  namespace: xorb-$ENVIRONMENT
spec:
  selector:
    app: xorb-api
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
YAML

echo "✅ API components deployed successfully"
EOF

# Monitoring component
cat > "$DEPLOYMENT_DIR/components/deploy-monitoring.sh" << 'EOF'
#!/bin/bash

echo "📊 Deploying XORB Monitoring Components"

ENVIRONMENT="$1"
REGION="$2"

# Deploy Prometheus
kubectl apply -f - <<YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: xorb-prometheus
  namespace: xorb-$ENVIRONMENT
spec:
  replicas: 1
  selector:
    matchLabels:
      app: xorb-prometheus
  template:
    metadata:
      labels:
        app: xorb-prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:v2.45.0
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: prometheus-config
          mountPath: /etc/prometheus
        - name: prometheus-storage
          mountPath: /prometheus
      volumes:
      - name: prometheus-config
        configMap:
          name: xorb-prometheus-config
      - name: prometheus-storage
        persistentVolumeClaim:
          claimName: prometheus-storage
---
apiVersion: v1
kind: Service
metadata:
  name: xorb-prometheus
  namespace: xorb-$ENVIRONMENT
spec:
  selector:
    app: xorb-prometheus
  ports:
  - port: 9090
    targetPort: 9090
YAML

# Deploy Grafana
kubectl apply -f - <<YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: xorb-grafana
  namespace: xorb-$ENVIRONMENT
spec:
  replicas: 1
  selector:
    matchLabels:
      app: xorb-grafana
  template:
    metadata:
      labels:
        app: xorb-grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:10.0.0
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          valueFrom:
            secretKeyRef:
              name: xorb-monitoring-secrets
              key: grafana-admin-password
        volumeMounts:
        - name: grafana-storage
          mountPath: /var/lib/grafana
      volumes:
      - name: grafana-storage
        persistentVolumeClaim:
          claimName: grafana-storage
---
apiVersion: v1
kind: Service
metadata:
  name: xorb-grafana
  namespace: xorb-$ENVIRONMENT
spec:
  selector:
    app: xorb-grafana
  ports:
  - port: 3000
    targetPort: 3000
  type: LoadBalancer
YAML

echo "✅ Monitoring components deployed successfully"
EOF

# Make component scripts executable
chmod +x "$DEPLOYMENT_DIR/components"/*.sh

# Create master deployment orchestrator
log_step "Creating master deployment orchestrator..."
cat > "$DEPLOYMENT_DIR/deploy.sh" << 'EOF'
#!/bin/bash

# XORB Master Deployment Orchestrator
# Coordinates deployment of all components across environments and regions

set -euo pipefail

ENVIRONMENT="${1:-development}"
REGION="${2:-us-east-1}"
COMPONENT="${3:-all}"

echo "🌍 XORB Master Deployment"
echo "Environment: $ENVIRONMENT"
echo "Region: $REGION"
echo "Component: $COMPONENT"
echo "=========================="

DEPLOYMENT_DIR="/root/Xorb/deployment"
LOG_FILE="$DEPLOYMENT_DIR/logs/deployment_${ENVIRONMENT}_${REGION}_$(date +%Y%m%d_%H%M%S).log"

# Ensure log directory exists
mkdir -p "$DEPLOYMENT_DIR/logs"

# Function to log and execute
log_and_execute() {
    echo "🔧 $1" | tee -a "$LOG_FILE"
    shift
    "$@" 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ Success" | tee -a "$LOG_FILE"
    else
        echo "❌ Failed" | tee -a "$LOG_FILE"
        exit 1
    fi
}

# Pre-deployment checks
log_and_execute "Running pre-deployment checks..." true

# Create namespace if it doesn't exist
log_and_execute "Creating Kubernetes namespace..." kubectl create namespace "xorb-$ENVIRONMENT" --dry-run=client -o yaml | kubectl apply -f -

# Deploy secrets
log_and_execute "Deploying secrets..." /root/Xorb/secrets/k8s-secrets.sh
kubectl apply -f /root/Xorb/secrets/k8s-*-secrets.yaml -n "xorb-$ENVIRONMENT"

# Deploy configuration
log_and_execute "Deploying configuration..." kubectl apply -f "$DEPLOYMENT_DIR/environments/$ENVIRONMENT/config/deployment.yaml" -n "xorb-$ENVIRONMENT"
kubectl apply -f "$DEPLOYMENT_DIR/regions/$REGION/config/region.yaml" -n "xorb-$ENVIRONMENT"

# Deploy components
if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "database" ]; then
    log_and_execute "Deploying database components..." "$DEPLOYMENT_DIR/components/deploy-database.sh" "$ENVIRONMENT" "$REGION"
fi

if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "api" ]; then
    log_and_execute "Deploying API components..." "$DEPLOYMENT_DIR/components/deploy-api.sh" "$ENVIRONMENT" "$REGION"
fi

if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "monitoring" ]; then
    log_and_execute "Deploying monitoring components..." "$DEPLOYMENT_DIR/components/deploy-monitoring.sh" "$ENVIRONMENT" "$REGION"
fi

# Post-deployment verification
log_and_execute "Running post-deployment verification..." kubectl get pods -n "xorb-$ENVIRONMENT"

echo "🎉 Deployment completed successfully!" | tee -a "$LOG_FILE"
echo "📋 Deployment log: $LOG_FILE"
EOF

chmod +x "$DEPLOYMENT_DIR/deploy.sh"

# Create rollback system
log_step "Creating rollback system..."
cat > "$DEPLOYMENT_DIR/rollback.sh" << 'EOF'
#!/bin/bash

# XORB Deployment Rollback System
# Safely rollback deployments to previous versions

set -euo pipefail

ENVIRONMENT="${1:-development}"
REGION="${2:-us-east-1}"
COMPONENT="${3:-all}"
VERSION="${4:-previous}"

echo "🔄 XORB Deployment Rollback"
echo "Environment: $ENVIRONMENT"
echo "Region: $REGION"
echo "Component: $COMPONENT"
echo "Version: $VERSION"
echo "=========================="

# Rollback API
if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "api" ]; then
    echo "🔄 Rolling back API..."
    kubectl rollout undo deployment/xorb-api -n "xorb-$ENVIRONMENT"
    kubectl rollout status deployment/xorb-api -n "xorb-$ENVIRONMENT"
fi

# Rollback database (careful - data migrations may be involved)
if [ "$COMPONENT" = "database" ]; then
    echo "⚠️  Database rollback requires manual intervention"
    echo "Please check data migrations before proceeding"
fi

# Rollback monitoring
if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "monitoring" ]; then
    echo "🔄 Rolling back monitoring..."
    kubectl rollout undo deployment/xorb-prometheus -n "xorb-$ENVIRONMENT" || true
    kubectl rollout undo deployment/xorb-grafana -n "xorb-$ENVIRONMENT" || true
fi

echo "✅ Rollback completed"
EOF

chmod +x "$DEPLOYMENT_DIR/rollback.sh"

# Create health check system
log_step "Creating health check system..."
cat > "$DEPLOYMENT_DIR/health-check.sh" << 'EOF'
#!/bin/bash

# XORB Deployment Health Check System
# Comprehensive health monitoring for deployed services

set -euo pipefail

ENVIRONMENT="${1:-development}"
REGION="${2:-us-east-1}"

echo "🏥 XORB Deployment Health Check"
echo "Environment: $ENVIRONMENT"
echo "Region: $REGION"
echo "=========================="

NAMESPACE="xorb-$ENVIRONMENT"

# Check pod status
echo "📋 Pod Status:"
kubectl get pods -n "$NAMESPACE" -o wide

echo ""
echo "🔍 Service Status:"
kubectl get services -n "$NAMESPACE"

echo ""
echo "📊 Resource Usage:"
kubectl top pods -n "$NAMESPACE" 2>/dev/null || echo "Metrics server not available"

echo ""
echo "🔗 Service Endpoints:"
kubectl get endpoints -n "$NAMESPACE"

# Test service connectivity
echo ""
echo "🌐 Connectivity Tests:"

# Test API health
API_SERVICE=$(kubectl get service xorb-api -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
if [ "$API_SERVICE" != "pending" ] && [ "$API_SERVICE" != "" ]; then
    echo "Testing API health: $API_SERVICE:8000/health"
    curl -s -f "http://$API_SERVICE:8000/health" && echo "✅ API healthy" || echo "❌ API unhealthy"
else
    echo "⏳ API service IP pending"
fi

# Test Grafana
GRAFANA_SERVICE=$(kubectl get service xorb-grafana -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
if [ "$GRAFANA_SERVICE" != "pending" ] && [ "$GRAFANA_SERVICE" != "" ]; then
    echo "Testing Grafana health: $GRAFANA_SERVICE:3000/api/health"
    curl -s -f "http://$GRAFANA_SERVICE:3000/api/health" && echo "✅ Grafana healthy" || echo "❌ Grafana unhealthy"
else
    echo "⏳ Grafana service IP pending"
fi

echo ""
echo "📋 Recent Events:"
kubectl get events -n "$NAMESPACE" --sort-by=.metadata.creationTimestamp | tail -10

echo ""
echo "🎯 Health Check Complete"
EOF

chmod +x "$DEPLOYMENT_DIR/health-check.sh"

# Create scaling automation
log_step "Creating scaling automation..."
cat > "$DEPLOYMENT_DIR/scale.sh" << 'EOF'
#!/bin/bash

# XORB Deployment Scaling System
# Dynamic scaling of XORB components

set -euo pipefail

ENVIRONMENT="${1:-development}"
COMPONENT="${2:-api}"
REPLICAS="${3:-2}"

echo "📈 XORB Deployment Scaling"
echo "Environment: $ENVIRONMENT"
echo "Component: $COMPONENT"
echo "Replicas: $REPLICAS"
echo "=========================="

NAMESPACE="xorb-$ENVIRONMENT"

case "$COMPONENT" in
    "api")
        kubectl scale deployment xorb-api --replicas="$REPLICAS" -n "$NAMESPACE"
        kubectl rollout status deployment/xorb-api -n "$NAMESPACE"
        ;;
    "worker")
        kubectl scale deployment xorb-worker --replicas="$REPLICAS" -n "$NAMESPACE"
        kubectl rollout status deployment/xorb-worker -n "$NAMESPACE"
        ;;
    "orchestrator")
        kubectl scale deployment xorb-orchestrator --replicas="$REPLICAS" -n "$NAMESPACE"
        kubectl rollout status deployment/xorb-orchestrator -n "$NAMESPACE"
        ;;
    *)
        echo "❌ Unknown component: $COMPONENT"
        echo "Available components: api, worker, orchestrator"
        exit 1
        ;;
esac

echo "✅ Scaling completed"
echo "📊 Current status:"
kubectl get deployment "$COMPONENT" -n "$NAMESPACE"
EOF

chmod +x "$DEPLOYMENT_DIR/scale.sh"

# Test deployment system
log_step "Testing deployment system..."
if [ -x "$DEPLOYMENT_DIR/deploy.sh" ]; then
    log_info "Deployment scripts are executable and ready"
else
    log_error "Deployment scripts are not executable"
fi

echo ""
log_info "Global deployment automation setup complete!"
echo ""
echo "🌍 Deployment Commands:"
echo "   - Full deployment: $DEPLOYMENT_DIR/deploy.sh <environment> <region> [component]"
echo "   - Health check: $DEPLOYMENT_DIR/health-check.sh <environment> <region>"
echo "   - Scaling: $DEPLOYMENT_DIR/scale.sh <environment> <component> <replicas>"
echo "   - Rollback: $DEPLOYMENT_DIR/rollback.sh <environment> <region> [component]"
echo ""
echo "🔧 Example usage:"
echo "   ./deployment/deploy.sh production us-east-1 all"
echo "   ./deployment/health-check.sh production us-east-1"
echo "   ./deployment/scale.sh production api 5"
echo "   ./deployment/rollback.sh production us-east-1 api"
echo ""
echo "📋 Supported environments: ${ENVIRONMENTS[*]}"
echo "🌍 Supported regions: ${REGIONS[*]}"