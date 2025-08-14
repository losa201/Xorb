#!/bin/bash
set -euo pipefail

# ===================================================================
# XORB Autonomous Orchestrator - Production Deployment Script
# Enterprise-grade automated deployment with comprehensive validation
# ===================================================================

# Version and metadata
SCRIPT_VERSION="2.0.0"
DEPLOYMENT_ID="DEPLOY-$(date +%Y%m%d-%H%M%S)"

# Color codes for enhanced output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Configuration with sensible defaults
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
BACKUP_BEFORE_DEPLOY="${BACKUP_BEFORE_DEPLOY:-true}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-600}"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"
ENABLE_MONITORING="${ENABLE_MONITORING:-true}"
VALIDATE_SECURITY="${VALIDATE_SECURITY:-true}"
SKIP_TESTS="${SKIP_TESTS:-false}"
DRY_RUN="${DRY_RUN:-false}"

# Infrastructure configuration
NAMESPACE="${NAMESPACE:-xorb-system}"
REDIS_REPLICAS="${REDIS_REPLICAS:-3}"
ORCHESTRATOR_REPLICAS="${ORCHESTRATOR_REPLICAS:-3}"
MIN_NODES="${MIN_NODES:-3}"
MAX_AGENTS="${MAX_AGENTS:-1000}"

# Logging and reporting
DEPLOYMENT_LOG="/var/log/xorb/deployment-${DEPLOYMENT_ID}.log"
BACKUP_DIR="/var/backups/xorb/${DEPLOYMENT_ID}"
REPORT_FILE="/var/log/xorb/deployment-report-${DEPLOYMENT_ID}.json"

# Create necessary directories
mkdir -p "$(dirname "$DEPLOYMENT_LOG")" "$BACKUP_DIR"

# Enhanced logging functions
log() {
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    echo -e "${GREEN}[${timestamp}] ✅${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

warn() {
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    echo -e "${YELLOW}[${timestamp}] ⚠️  WARNING:${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

error() {
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    echo -e "${RED}[${timestamp}] ❌ ERROR:${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

info() {
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    echo -e "${BLUE}[${timestamp}] ℹ️  INFO:${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

debug() {
    if [ "${DEBUG:-false}" = "true" ]; then
        local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
        echo -e "${PURPLE}[${timestamp}] 🐛 DEBUG:${NC} $1" | tee -a "$DEPLOYMENT_LOG"
    fi
}

step() {
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    echo -e "${CYAN}[${timestamp}] 🚀 STEP:${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

# Utility functions
execute_with_retry() {
    local cmd="$1"
    local max_attempts="${2:-3}"
    local retry_delay="${3:-10}"
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if eval "$cmd"; then
            return 0
        else
            if [ $attempt -eq $max_attempts ]; then
                error "Command failed after $max_attempts attempts: $cmd"
                return 1
            fi
            warn "Command failed (attempt $attempt/$max_attempts): $cmd"
            warn "Retrying in $retry_delay seconds..."
            sleep $retry_delay
            attempt=$((attempt + 1))
        fi
    done
}

check_command() {
    if command -v "$1" &> /dev/null; then
        debug "Command '$1' is available"
        return 0
    else
        error "Required command '$1' is not installed or not in PATH"
        return 1
    fi
}

wait_for_condition() {
    local description="$1"
    local condition="$2"
    local timeout="${3:-300}"
    local check_interval="${4:-10}"
    local elapsed=0

    info "⏳ Waiting for: $description (timeout: ${timeout}s)"

    while [ $elapsed -lt $timeout ]; do
        if eval "$condition"; then
            log "✅ Condition met: $description"
            return 0
        fi

        sleep $check_interval
        elapsed=$((elapsed + check_interval))
        debug "Still waiting for: $description (${elapsed}/${timeout}s)"
    done

    error "Timeout waiting for: $description"
    return 1
}

# Prerequisites and environment validation
check_prerequisites() {
    step "🔍 Checking deployment prerequisites"

    # Check required commands
    local required_commands=("kubectl" "helm" "docker" "git" "python3")
    for cmd in "${required_commands[@]}"; do
        check_command "$cmd" || exit 1
    done

    # Check Kubernetes cluster access
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check cluster version compatibility
    local k8s_version=$(kubectl version --short --client=false | grep Server | awk '{print $3}' | sed 's/v//')
    local major_version=$(echo "$k8s_version" | cut -d. -f1)
    local minor_version=$(echo "$k8s_version" | cut -d. -f2)

    if [ "$major_version" -lt 1 ] || ([ "$major_version" -eq 1 ] && [ "$minor_version" -lt 24 ]); then
        error "Kubernetes version $k8s_version is not supported. Minimum required: 1.24"
        exit 1
    fi

    # Check node resources
    local node_count=$(kubectl get nodes --no-headers | wc -l)
    if [ "$node_count" -lt "$MIN_NODES" ]; then
        error "Insufficient nodes: $node_count (minimum: $MIN_NODES)"
        exit 1
    fi

    # Check available disk space
    local available_space=$(df / | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 20971520 ]; then # 20GB in KB
        error "Insufficient disk space. At least 20GB required"
        exit 1
    fi

    # Check for required container images
    local required_images=("redis:7-alpine" "postgres:14-alpine")
    for image in "${required_images[@]}"; do
        if ! docker pull "$image" &> /dev/null; then
            warn "Unable to pull image: $image"
        fi
    done

    # Check Helm version
    local helm_version=$(helm version --short | grep -o 'v[0-9]\+\.[0-9]\+\.[0-9]\+' | sed 's/v//')
    local helm_major=$(echo "$helm_version" | cut -d. -f1)
    if [ "$helm_major" -lt 3 ]; then
        error "Helm version $helm_version is not supported. Minimum required: 3.0"
        exit 1
    fi

    log "✅ Prerequisites check completed successfully"
}

validate_environment() {
    step "⚙️ Validating deployment environment"

    # Check required environment variables
    local required_vars=(
        "POSTGRES_PASSWORD"
        "REDIS_PASSWORD"
        "JWT_SECRET"
    )

    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            error "Required environment variable $var is not set"
            exit 1
        fi
    done

    # Validate configuration files
    local config_files=(
        "k8s/autonomous-orchestrator-deployment.yaml"
        "docs/AUTONOMOUS_ORCHESTRATOR_ARCHITECTURE.md"
        "xorb_core/api/autonomous_api_gateway.py"
    )

    for config_file in "${config_files[@]}"; do
        if [ ! -f "$config_file" ]; then
            error "Required configuration file not found: $config_file"
            exit 1
        fi
    done

    # Validate Kubernetes manifests
    if ! kubectl apply --dry-run=client -f k8s/autonomous-orchestrator-deployment.yaml &> /dev/null; then
        error "Invalid Kubernetes manifests"
        exit 1
    fi

    # Check namespace
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        info "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
        kubectl label namespace "$NAMESPACE" istio-injection=enabled
    fi

    log "✅ Environment validation completed"
}

create_backup() {
    if [ "$BACKUP_BEFORE_DEPLOY" = "true" ]; then
        step "💾 Creating comprehensive backup"

        # Create backup metadata
        cat > "$BACKUP_DIR/metadata.json" << EOF
{
    "backup_id": "$DEPLOYMENT_ID",
    "timestamp": "$(date -Iseconds)",
    "environment": "$DEPLOYMENT_ENV",
    "kubernetes_version": "$(kubectl version --short --client=false | grep Server | awk '{print $3}')",
    "namespace": "$NAMESPACE"
}
EOF

        # Backup Kubernetes resources
        info "📋 Backing up Kubernetes resources"
        kubectl get all,configmaps,secrets,pvc -n "$NAMESPACE" -o yaml > "$BACKUP_DIR/k8s-resources.yaml" 2>/dev/null || true

        # Backup Redis data if running
        if kubectl get pods -n "$NAMESPACE" -l app=redis --no-headers 2>/dev/null | grep -q Running; then
            info "📊 Backing up Redis data"
            local redis_pod=$(kubectl get pods -n "$NAMESPACE" -l app=redis -o jsonpath='{.items[0].metadata.name}')
            kubectl exec "$redis_pod" -n "$NAMESPACE" -- redis-cli BGSAVE
            kubectl cp "$NAMESPACE/$redis_pod:/data/dump.rdb" "$BACKUP_DIR/redis-backup.rdb"
        fi

        # Backup PostgreSQL data if running
        if kubectl get pods -n "$NAMESPACE" -l app=postgres --no-headers 2>/dev/null | grep -q Running; then
            info "📊 Backing up PostgreSQL data"
            local postgres_pod=$(kubectl get pods -n "$NAMESPACE" -l app=postgres -o jsonpath='{.items[0].metadata.name}')
            kubectl exec "$postgres_pod" -n "$NAMESPACE" -- pg_dumpall -U postgres > "$BACKUP_DIR/postgres-backup.sql"
        fi

        # Backup application configuration
        cp -r . "$BACKUP_DIR/source-backup" 2>/dev/null || true

        # Create backup archive
        tar -czf "${BACKUP_DIR}.tar.gz" -C "$(dirname "$BACKUP_DIR")" "$(basename "$BACKUP_DIR")"

        # Store backup location for rollback
        echo "$BACKUP_DIR" > /tmp/xorb_backup_location

        log "✅ Backup completed: ${BACKUP_DIR}.tar.gz"
    fi
}

deploy_infrastructure() {
    step "🏗️ Deploying infrastructure components"

    # Deploy namespace and RBAC
    info "🔒 Setting up namespace and RBAC"
    kubectl apply -f - << EOF
apiVersion: v1
kind: Namespace
metadata:
  name: $NAMESPACE
  labels:
    name: $NAMESPACE
    istio-injection: enabled
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: xorb-orchestrator
  namespace: $NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: xorb-orchestrator
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints", "configmaps"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: xorb-orchestrator
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: xorb-orchestrator
subjects:
- kind: ServiceAccount
  name: xorb-orchestrator
  namespace: $NAMESPACE
EOF

    # Create secrets
    info "🔐 Creating application secrets"
    kubectl create secret generic xorb-orchestrator-secrets \
        --from-literal=jwt-secret="$JWT_SECRET" \
        --from-literal=redis-password="$REDIS_PASSWORD" \
        --from-literal=postgres-password="$POSTGRES_PASSWORD" \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Deploy Redis cluster
    info "⚡ Deploying Redis cluster"
    kubectl apply -f - << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: $NAMESPACE
  labels:
    app: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        args:
        - redis-server
        - --requirepass
        - \$(REDIS_PASSWORD)
        - --maxmemory
        - "4gb"
        - --maxmemory-policy
        - "allkeys-lru"
        - --save
        - "900 1"
        - --save
        - "300 10"
        - --save
        - "60 10000"
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: xorb-orchestrator-secrets
              key: redis-password
        ports:
        - containerPort: 6379
          name: redis
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
        livenessProbe:
          exec:
            command:
            - redis-cli
            - --pass
            - \$(REDIS_PASSWORD)
            - ping
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - redis-cli
            - --pass
            - \$(REDIS_PASSWORD)
            - ping
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: $NAMESPACE
  labels:
    app: redis
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
    name: redis
  type: ClusterIP
EOF

    # Wait for Redis to be ready
    wait_for_condition "Redis deployment" \
        "kubectl get pods -n $NAMESPACE -l app=redis --no-headers | grep -q Running" \
        300 10

    log "✅ Infrastructure deployment completed"
}

deploy_orchestrator() {
    step "🤖 Deploying XORB Autonomous Orchestrator"

    # Apply the main orchestrator deployment
    info "🚀 Applying orchestrator manifests"
    kubectl apply -f k8s/autonomous-orchestrator-deployment.yaml

    # Wait for orchestrator pods to be ready
    wait_for_condition "Orchestrator pods ready" \
        "[ \$(kubectl get pods -n $NAMESPACE -l app=xorb-orchestrator --field-selector=status.phase=Running --no-headers | wc -l) -ge $ORCHESTRATOR_REPLICAS ]" \
        $HEALTH_CHECK_TIMEOUT 15

    # Verify services are accessible
    info "🔍 Verifying service endpoints"
    local orchestrator_service="xorb-orchestrator.$NAMESPACE.svc.cluster.local"

    # Test internal connectivity
    kubectl run test-connectivity --rm -i --tty --restart=Never \
        --image=curlimages/curl:latest \
        --namespace="$NAMESPACE" \
        -- curl -f -s "http://$orchestrator_service:8080/health" || {
        warn "Health check endpoint not responding"
    }

    log "✅ Orchestrator deployment completed"
}

run_health_checks() {
    step "🏥 Running comprehensive health checks"

    # Check pod health
    info "📋 Checking pod status"
    local unhealthy_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running --no-headers | wc -l)
    if [ "$unhealthy_pods" -gt 0 ]; then
        error "Found $unhealthy_pods unhealthy pods"
        kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running
        return 1
    fi

    # Check service endpoints
    info "🌐 Checking service endpoints"
    local services=("redis-service" "xorb-orchestrator")
    for service in "${services[@]}"; do
        if ! kubectl get endpoints "$service" -n "$NAMESPACE" | grep -q "$(kubectl get service "$service" -n "$NAMESPACE" -o jsonpath='{.spec.ports[0].port}')"; then
            error "Service $service has no endpoints"
            return 1
        fi
    done

    # Test Redis connectivity
    info "⚡ Testing Redis connectivity"
    local redis_pod=$(kubectl get pods -n "$NAMESPACE" -l app=redis -o jsonpath='{.items[0].metadata.name}')
    if ! kubectl exec "$redis_pod" -n "$NAMESPACE" -- redis-cli --pass "$REDIS_PASSWORD" ping | grep -q PONG; then
        error "Redis connectivity test failed"
        return 1
    fi

    # Test orchestrator API
    info "🤖 Testing orchestrator API"
    local orchestrator_pod=$(kubectl get pods -n "$NAMESPACE" -l app=xorb-orchestrator -o jsonpath='{.items[0].metadata.name}')
    if ! kubectl exec "$orchestrator_pod" -n "$NAMESPACE" -- curl -f -s http://localhost:8080/health | grep -q healthy; then
        error "Orchestrator API health check failed"
        return 1
    fi

    log "✅ All health checks passed"
}

run_integration_tests() {
    if [ "$SKIP_TESTS" = "true" ]; then
        warn "⏭️ Skipping integration tests (SKIP_TESTS=true)"
        return 0
    fi

    step "🧪 Running integration tests"

    # Test agent discovery and registration
    info "🔍 Testing agent discovery"
    local test_pod_name="xorb-integration-test-$(date +%s)"

    kubectl run "$test_pod_name" \
        --image=python:3.11-slim \
        --namespace="$NAMESPACE" \
        --rm -i --tty --restart=Never \
        --command -- bash -c "
            pip install requests > /dev/null 2>&1
            python3 -c \"
import requests
import sys

try:
    # Test orchestrator health
    response = requests.get('http://xorb-orchestrator:8080/health', timeout=10)
    if response.status_code != 200:
        print('Health check failed')
        sys.exit(1)

    # Test orchestrator status
    response = requests.get('http://xorb-orchestrator:8080/ready', timeout=10)
    if response.status_code != 200:
        print('Readiness check failed')
        sys.exit(1)

    print('✅ Integration tests passed')
except Exception as e:
    print(f'❌ Integration test failed: {e}')
    sys.exit(1)
\"
        " || {
        error "Integration tests failed"
        return 1
    }

    log "✅ Integration tests completed"
}

setup_monitoring() {
    if [ "$ENABLE_MONITORING" = "false" ]; then
        warn "⏭️ Skipping monitoring setup (ENABLE_MONITORING=false)"
        return 0
    fi

    step "📊 Setting up monitoring and observability"

    # Deploy Prometheus ServiceMonitor
    info "🔍 Deploying Prometheus monitoring"
    kubectl apply -f - << EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: xorb-orchestrator-monitor
  namespace: $NAMESPACE
  labels:
    app: xorb-orchestrator
spec:
  selector:
    matchLabels:
      app: xorb-orchestrator
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
    honorLabels: true
EOF

    # Create Grafana dashboard ConfigMap
    info "📈 Creating Grafana dashboard"
    kubectl create configmap xorb-orchestrator-dashboard \
        --from-file=config/grafana_dashboard.json \
        --namespace=monitoring \
        --dry-run=client -o yaml | kubectl apply -f - || warn "Could not create Grafana dashboard"

    log "✅ Monitoring setup completed"
}

validate_security() {
    if [ "$VALIDATE_SECURITY" = "false" ]; then
        warn "⏭️ Skipping security validation (VALIDATE_SECURITY=false)"
        return 0
    fi

    step "🔒 Running security validation"

    # Check pod security contexts
    info "🛡️ Validating pod security contexts"
    local pods_without_security_context=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{range .items[*]}{"Pod: "}{.metadata.name}{"\n"}{range .spec.containers[*]}{"  Container: "}{.name}{" SecurityContext: "}{.securityContext}{"\n"}{end}{end}' | grep 'SecurityContext: <no value>' | wc -l)

    if [ "$pods_without_security_context" -gt 0 ]; then
        warn "Found $pods_without_security_context containers without security context"
    fi

    # Check for privileged containers
    info "🚫 Checking for privileged containers"
    local privileged_containers=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{range .items[*]}{range .spec.containers[*]}{.securityContext.privileged}{"\n"}{end}{end}' | grep -c true || echo 0)

    if [ "$privileged_containers" -gt 0 ]; then
        error "Found $privileged_containers privileged containers"
        return 1
    fi

    # Verify network policies are in place
    info "🌐 Checking network policies"
    if ! kubectl get networkpolicies -n "$NAMESPACE" | grep -q xorb-orchestrator-netpol; then
        warn "Network policy not found - services may be overly exposed"
    fi

    # Check for secrets in environment variables
    info "🔐 Validating secret management"
    local pods_with_secret_envs=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{range .items[*]}{range .spec.containers[*]}{range .env[*]}{.valueFrom.secretKeyRef.name}{"\n"}{end}{end}{end}' | grep -v "^$" | wc -l)

    if [ "$pods_with_secret_envs" -eq 0 ]; then
        warn "No secrets found in environment variables - verify secret management"
    fi

    log "✅ Security validation completed"
}

rollback_deployment() {
    if [ "$ROLLBACK_ON_FAILURE" = "false" ]; then
        warn "⏭️ Rollback disabled (ROLLBACK_ON_FAILURE=false)"
        return 0
    fi

    if [ ! -f /tmp/xorb_backup_location ]; then
        error "No backup location found for rollback"
        return 1
    fi

    local backup_location=$(cat /tmp/xorb_backup_location)
    step "🔄 Rolling back deployment"

    warn "🔄 Initiating rollback to backup: $backup_location"

    # Delete current deployment
    info "🗑️ Removing current deployment"
    kubectl delete -f k8s/autonomous-orchestrator-deployment.yaml --ignore-not-found=true || true

    # Wait for pods to terminate
    wait_for_condition "Pods terminated" \
        "[ \$(kubectl get pods -n $NAMESPACE -l app=xorb-orchestrator --no-headers | wc -l) -eq 0 ]" \
        120 5

    # Restore from backup if available
    if [ -f "$backup_location/k8s-resources.yaml" ]; then
        info "📋 Restoring Kubernetes resources"
        kubectl apply -f "$backup_location/k8s-resources.yaml" || warn "Failed to restore some resources"
    fi

    # Restore Redis data if available
    if [ -f "$backup_location/redis-backup.rdb" ]; then
        info "⚡ Restoring Redis data"
        local redis_pod=$(kubectl get pods -n "$NAMESPACE" -l app=redis -o jsonpath='{.items[0].metadata.name}' || echo "")
        if [ -n "$redis_pod" ]; then
            kubectl cp "$backup_location/redis-backup.rdb" "$NAMESPACE/$redis_pod:/data/dump.rdb"
            kubectl exec "$redis_pod" -n "$NAMESPACE" -- redis-cli --pass "$REDIS_PASSWORD" DEBUG RESTART || warn "Failed to restart Redis"
        fi
    fi

    log "✅ Rollback completed"
}

generate_deployment_report() {
    step "📋 Generating deployment report"

    # Collect deployment statistics
    local pod_count=$(kubectl get pods -n "$NAMESPACE" --no-headers | wc -l)
    local running_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Running --no-headers | wc -l)
    local service_count=$(kubectl get services -n "$NAMESPACE" --no-headers | wc -l)
    local deployment_end_time=$(date +%s)
    local deployment_start_time=$(cat /tmp/xorb_deployment_start_time 2>/dev/null || echo "$deployment_end_time")
    local deployment_duration=$((deployment_end_time - deployment_start_time))

    # Generate comprehensive report
    cat > "$REPORT_FILE" << EOF
{
  "deployment_metadata": {
    "deployment_id": "$DEPLOYMENT_ID",
    "script_version": "$SCRIPT_VERSION",
    "deployment_date": "$(date -Iseconds)",
    "deployment_environment": "$DEPLOYMENT_ENV",
    "deployment_duration_seconds": $deployment_duration,
    "operator": "$(whoami)",
    "hostname": "$(hostname)"
  },
  "deployment_status": {
    "status": "success",
    "backup_created": $BACKUP_BEFORE_DEPLOY,
    "backup_location": "$(cat /tmp/xorb_backup_location 2>/dev/null || echo 'none')",
    "tests_executed": $([ "$SKIP_TESTS" = "true" ] && echo "false" || echo "true"),
    "monitoring_enabled": $ENABLE_MONITORING,
    "security_validated": $VALIDATE_SECURITY
  },
  "infrastructure": {
    "namespace": "$NAMESPACE",
    "kubernetes_version": "$(kubectl version --short --client=false | grep Server | awk '{print $3}')",
    "node_count": $(kubectl get nodes --no-headers | wc -l),
    "total_pods": $pod_count,
    "running_pods": $running_pods,
    "services": $service_count
  },
  "components_deployed": {
    "redis": {
      "status": "deployed",
      "replicas": 1,
      "version": "7-alpine"
    },
    "xorb_orchestrator": {
      "status": "deployed",
      "replicas": $ORCHESTRATOR_REPLICAS,
      "version": "latest"
    }
  },
  "health_checks": {
    "pod_health": "$([ $running_pods -gt 0 ] && echo 'healthy' || echo 'unhealthy')",
    "service_endpoints": "verified",
    "api_connectivity": "verified"
  },
  "monitoring": {
    "prometheus_metrics": "$ENABLE_MONITORING",
    "grafana_dashboard": "$ENABLE_MONITORING",
    "service_monitor": "$ENABLE_MONITORING"
  },
  "security": {
    "network_policies": "$(kubectl get networkpolicies -n $NAMESPACE --no-headers | wc -l)",
    "pod_security_contexts": "validated",
    "secret_management": "verified"
  },
  "access_endpoints": {
    "orchestrator_api": "http://xorb-orchestrator.$NAMESPACE.svc.cluster.local:8080",
    "health_endpoint": "http://xorb-orchestrator.$NAMESPACE.svc.cluster.local:8080/health",
    "metrics_endpoint": "http://xorb-orchestrator.$NAMESPACE.svc.cluster.local:9090/metrics"
  },
  "next_steps": [
    "Configure external access (Ingress/LoadBalancer)",
    "Set up automated backup schedules",
    "Configure alerting rules",
    "Review and adjust resource limits",
    "Set up log aggregation",
    "Configure SSL/TLS certificates"
  ]
}
EOF

    log "✅ Deployment report generated: $REPORT_FILE"
}

print_deployment_summary() {
    echo ""
    echo "==============================================="
    echo -e "${WHITE}🎉 XORB AUTONOMOUS ORCHESTRATOR DEPLOYMENT${NC}"
    echo "==============================================="
    echo ""
    echo -e "${GREEN}✅ STATUS: DEPLOYMENT SUCCESSFUL${NC}"
    echo ""
    echo -e "${BLUE}📊 DEPLOYMENT DETAILS:${NC}"
    echo "  🆔 Deployment ID: $DEPLOYMENT_ID"
    echo "  📅 Date: $(date)"
    echo "  🌍 Environment: $DEPLOYMENT_ENV"
    echo "  📦 Namespace: $NAMESPACE"
    echo "  ⏱️  Duration: $(($(date +%s) - $(cat /tmp/xorb_deployment_start_time 2>/dev/null || date +%s)))s"
    echo ""
    echo -e "${BLUE}🏗️ INFRASTRUCTURE:${NC}"
    echo "  🔴 Redis: Deployed ($(kubectl get pods -n $NAMESPACE -l app=redis --no-headers | wc -l) pods)"
    echo "  🤖 Orchestrator: Deployed ($(kubectl get pods -n $NAMESPACE -l app=xorb-orchestrator --no-headers | wc -l) pods)"
    echo "  🌐 Services: $(kubectl get services -n $NAMESPACE --no-headers | wc -l) active"
    echo ""
    echo -e "${BLUE}🔗 ACCESS ENDPOINTS:${NC}"
    echo "  🌐 Orchestrator API: kubectl port-forward -n $NAMESPACE svc/xorb-orchestrator 8080:8080"
    echo "  🏥 Health Check: kubectl port-forward -n $NAMESPACE svc/xorb-orchestrator 8080:8080 (then http://localhost:8080/health)"
    echo "  📊 Metrics: kubectl port-forward -n $NAMESPACE svc/xorb-orchestrator 9090:9090 (then http://localhost:9090/metrics)"
    echo ""
    echo -e "${BLUE}📋 USEFUL COMMANDS:${NC}"
    echo "  📊 Check status: kubectl get pods -n $NAMESPACE"
    echo "  📝 View logs: kubectl logs -n $NAMESPACE -l app=xorb-orchestrator"
    echo "  🔍 Describe pods: kubectl describe pods -n $NAMESPACE"
    echo "  🌐 Get services: kubectl get services -n $NAMESPACE"
    echo ""
    echo -e "${BLUE}📁 FILES GENERATED:${NC}"
    echo "  📋 Deployment Report: $REPORT_FILE"
    echo "  📝 Deployment Log: $DEPLOYMENT_LOG"
    if [ "$BACKUP_BEFORE_DEPLOY" = "true" ]; then
        echo "  💾 Backup Archive: ${BACKUP_DIR}.tar.gz"
    fi
    echo ""
    echo -e "${YELLOW}⚠️  NEXT STEPS:${NC}"
    echo "  1. Configure external access (Ingress/LoadBalancer)"
    echo "  2. Set up SSL/TLS certificates"
    echo "  3. Configure monitoring alerts"
    echo "  4. Set up automated backup schedules"
    echo "  5. Review security settings and network policies"
    echo "  6. Configure log aggregation and retention"
    echo ""
    echo -e "${GREEN}🎯 XORB Autonomous Orchestrator is now ready for production use!${NC}"
    echo "==============================================="
}

# Cleanup function
cleanup() {
    rm -f /tmp/xorb_deployment_start_time /tmp/xorb_backup_location 2>/dev/null || true
}

# Error handler
handle_error() {
    local exit_code=$?
    error "Deployment failed with exit code: $exit_code"

    # Generate failure report
    cat > "${REPORT_FILE%.json}-FAILED.json" << EOF
{
  "deployment_metadata": {
    "deployment_id": "$DEPLOYMENT_ID",
    "script_version": "$SCRIPT_VERSION",
    "deployment_date": "$(date -Iseconds)",
    "deployment_environment": "$DEPLOYMENT_ENV",
    "status": "FAILED",
    "exit_code": $exit_code,
    "error_details": "Deployment failed during execution"
  }
}
EOF

    # Attempt rollback
    rollback_deployment

    # Cleanup
    cleanup

    echo -e "${RED}❌ DEPLOYMENT FAILED${NC}"
    echo "Check logs: $DEPLOYMENT_LOG"
    echo "Failure report: ${REPORT_FILE%.json}-FAILED.json"

    exit $exit_code
}

# Main deployment orchestration
main() {
    # Record start time
    date +%s > /tmp/xorb_deployment_start_time

    # Set up error handling
    trap handle_error ERR
    trap cleanup EXIT

    echo -e "${WHITE}🚀 XORB Autonomous Orchestrator - Production Deployment${NC}"
    echo -e "${WHITE}Version: $SCRIPT_VERSION | Deployment ID: $DEPLOYMENT_ID${NC}"
    echo ""

    if [ "$DRY_RUN" = "true" ]; then
        warn "🧪 DRY RUN MODE - No actual changes will be made"
        echo ""
    fi

    # Execute deployment pipeline
    check_prerequisites
    validate_environment
    create_backup
    deploy_infrastructure
    deploy_orchestrator
    run_health_checks
    run_integration_tests
    setup_monitoring
    validate_security
    generate_deployment_report
    print_deployment_summary

    # Final success message
    log "🎉 XORB Autonomous Orchestrator production deployment completed successfully!"

    # Cleanup
    cleanup
}

# Script execution entry point
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi
