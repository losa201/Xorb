#!/bin/bash
#
# Blue-Green Deployment Strategy for Xorb Security Intelligence Platform
# Implements zero-downtime updates with intelligent traffic routing
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NAMESPACE="${NAMESPACE:-xorb-system}"
APP_NAME="${APP_NAME:-xorb}"
REGISTRY="${REGISTRY:-ghcr.io/xorb}"
VERSION="${VERSION:-$(git rev-parse --short HEAD)}"
PREVIOUS_VERSION="${PREVIOUS_VERSION:-latest}"
TIMEOUT="${TIMEOUT:-600}"  # 10 minutes
HEALTH_CHECK_URL="${HEALTH_CHECK_URL:-/health}"
READINESS_CHECK_URL="${READINESS_CHECK_URL:-/ready}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] INFO: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARN: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] SUCCESS: $1${NC}"
}

header() {
    echo -e "\n${PURPLE}=====================================${NC}"
    echo -e "${PURPLE} $1${NC}"
    echo -e "${PURPLE}=====================================${NC}\n"
}

# Utility functions
check_prerequisites() {
    log "Checking prerequisites..."
    
    command -v kubectl >/dev/null 2>&1 || error "kubectl is required"
    command -v helm >/dev/null 2>&1 || error "helm is required"
    command -v jq >/dev/null 2>&1 || error "jq is required"
    command -v curl >/dev/null 2>&1 || error "curl is required"
    
    # Check cluster connectivity
    kubectl cluster-info >/dev/null 2>&1 || error "Cannot connect to Kubernetes cluster"
    
    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        warn "Namespace $NAMESPACE does not exist, creating..."
        kubectl create namespace "$NAMESPACE"
    fi
    
    success "Prerequisites check passed"
}

get_current_deployment_color() {
    local current_selector
    current_selector=$(kubectl get service "$APP_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.selector.color}' 2>/dev/null || echo "")
    
    if [[ -z "$current_selector" ]]; then
        echo "blue"  # Default to blue if no deployment exists
    else
        echo "$current_selector"
    fi
}

get_target_deployment_color() {
    local current_color="$1"
    if [[ "$current_color" == "blue" ]]; then
        echo "green"
    else
        echo "blue"
    fi
}

build_and_push_images() {
    local target_color="$1"
    
    header "Building and Pushing Docker Images"
    
    log "Building images for version $VERSION..."
    
    # Build API image
    info "Building API image..."
    docker build -f Dockerfile.api.hardened \
        -t "$REGISTRY/$APP_NAME-api:$VERSION" \
        -t "$REGISTRY/$APP_NAME-api:$target_color" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg BUILD_COMMIT="$(git rev-parse HEAD)" \
        --build-arg BUILD_BRANCH="$(git branch --show-current)" \
        .
    
    # Build Worker image
    info "Building Worker image..."
    docker build -f Dockerfile.worker.hardened \
        -t "$REGISTRY/$APP_NAME-worker:$VERSION" \
        -t "$REGISTRY/$APP_NAME-worker:$target_color" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg BUILD_COMMIT="$(git rev-parse HEAD)" \
        --build-arg BUILD_BRANCH="$(git branch --show-current)" \
        .
    
    # Build Orchestrator image
    info "Building Orchestrator image..."
    docker build -f Dockerfile.orchestrator \
        -t "$REGISTRY/$APP_NAME-orchestrator:$VERSION" \
        -t "$REGISTRY/$APP_NAME-orchestrator:$target_color" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg BUILD_COMMIT="$(git rev-parse HEAD)" \
        --build-arg BUILD_BRANCH="$(git branch --show-current)" \
        .
    
    log "Pushing images to registry..."
    docker push "$REGISTRY/$APP_NAME-api:$VERSION"
    docker push "$REGISTRY/$APP_NAME-api:$target_color"
    docker push "$REGISTRY/$APP_NAME-worker:$VERSION"
    docker push "$REGISTRY/$APP_NAME-worker:$target_color"
    docker push "$REGISTRY/$APP_NAME-orchestrator:$VERSION"
    docker push "$REGISTRY/$APP_NAME-orchestrator:$target_color"
    
    success "Images built and pushed successfully"
}

deploy_target_environment() {
    local target_color="$1"
    local current_color="$2"
    
    header "Deploying $target_color Environment"
    
    log "Deploying $target_color deployment..."
    
    # Create temporary values file for target deployment
    local values_file="/tmp/values-$target_color.yaml"
    cat > "$values_file" << EOF
# Blue-Green Deployment Values for $target_color
global:
  environment: production
  namespace: $NAMESPACE
  color: $target_color

image:
  registry: $REGISTRY
  tag: $VERSION
  pullPolicy: Always

deployment:
  color: $target_color
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0
      maxSurge: 1

service:
  name: $APP_NAME-$target_color
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: false  # Traffic routing handled separately

resources:
  api:
    requests:
      cpu: 200m
      memory: 256Mi
    limits:
      cpu: 1000m
      memory: 1Gi
  worker:
    requests:
      cpu: 300m
      memory: 512Mi
    limits:
      cpu: 2000m
      memory: 2Gi
  orchestrator:
    requests:
      cpu: 400m
      memory: 512Mi
    limits:
      cpu: 2000m
      memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s

health:
  livenessProbe:
    httpGet:
      path: $HEALTH_CHECK_URL
      port: 8000
    initialDelaySeconds: 30
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 3
  readinessProbe:
    httpGet:
      path: $READINESS_CHECK_URL
      port: 8000
    initialDelaySeconds: 15
    periodSeconds: 5
    timeoutSeconds: 3
    failureThreshold: 2

security:
  podSecurityPolicy:
    enabled: true
  networkPolicy:
    enabled: true
  rbac:
    create: true

configMap:
  data:
    DEPLOYMENT_COLOR: $target_color
    DEPLOYMENT_VERSION: $VERSION
    DEPLOYMENT_TIMESTAMP: "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    BLUE_GREEN_MODE: "true"
    CANARY_ENABLED: "false"
EOF

    # Deploy using Helm
    helm upgrade --install "$APP_NAME-$target_color" \
        "$PROJECT_ROOT/gitops/helm/xorb-core" \
        --namespace "$NAMESPACE" \
        --values "$values_file" \
        --set global.xorb.color="$target_color" \
        --set global.xorb.version="$VERSION" \
        --timeout "${TIMEOUT}s" \
        --wait \
        --wait-for-jobs
    
    # Clean up temporary file
    rm -f "$values_file"
    
    success "$target_color environment deployed successfully"
}

wait_for_deployment_ready() {
    local target_color="$1"
    local timeout="$2"
    
    header "Waiting for $target_color Deployment to be Ready"
    
    log "Waiting for deployment rollout to complete..."
    kubectl rollout status deployment/"$APP_NAME-api-$target_color" -n "$NAMESPACE" --timeout="${timeout}s"
    kubectl rollout status deployment/"$APP_NAME-worker-$target_color" -n "$NAMESPACE" --timeout="${timeout}s"
    kubectl rollout status deployment/"$APP_NAME-orchestrator-$target_color" -n "$NAMESPACE" --timeout="${timeout}s"
    
    log "Waiting for pods to be ready..."
    kubectl wait --for=condition=ready pod \
        -l "app.kubernetes.io/name=$APP_NAME,color=$target_color" \
        -n "$NAMESPACE" \
        --timeout="${timeout}s"
    
    success "$target_color deployment is ready"
}

run_health_checks() {
    local target_color="$1"
    
    header "Running Health Checks on $target_color Environment"
    
    local api_pod
    api_pod=$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=${APP_NAME}-api,color=$target_color" -o jsonpath='{.items[0].metadata.name}')
    
    if [[ -z "$api_pod" ]]; then
        error "No API pod found for $target_color deployment"
    fi
    
    log "Running health checks on pod: $api_pod"
    
    # Health check
    info "Checking health endpoint..."
    if kubectl exec -n "$NAMESPACE" "$api_pod" -- curl -f "http://localhost:8000$HEALTH_CHECK_URL" >/dev/null 2>&1; then
        success "Health check passed"
    else
        error "Health check failed"
    fi
    
    # Readiness check
    info "Checking readiness endpoint..."
    if kubectl exec -n "$NAMESPACE" "$api_pod" -- curl -f "http://localhost:8000$READINESS_CHECK_URL" >/dev/null 2>&1; then
        success "Readiness check passed"
    else
        error "Readiness check failed"
    fi
    
    # API functionality check
    info "Checking API functionality..."
    local api_response
    api_response=$(kubectl exec -n "$NAMESPACE" "$api_pod" -- curl -s "http://localhost:8000/api/v1/status" | jq -r '.status' 2>/dev/null || echo "error")
    
    if [[ "$api_response" == "healthy" ]]; then
        success "API functionality check passed"
    else
        warn "API functionality check returned: $api_response"
    fi
    
    # Database connectivity check
    info "Checking database connectivity..."
    if kubectl exec -n "$NAMESPACE" "$api_pod" -- curl -s "http://localhost:8000/api/v1/health/db" | grep -q "healthy"; then
        success "Database connectivity check passed"
    else
        warn "Database connectivity check failed"
    fi
    
    success "All health checks completed for $target_color environment"
}

run_smoke_tests() {
    local target_color="$1"
    
    header "Running Smoke Tests on $target_color Environment"
    
    log "Executing smoke test suite..."
    
    # Create a test job
    cat << EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: smoke-test-$target_color-$(date +%s)
  namespace: $NAMESPACE
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: smoke-test
        image: $REGISTRY/$APP_NAME-api:$VERSION
        command: ["python", "-m", "pytest", "tests/smoke/", "-v", "--tb=short"]
        env:
        - name: TARGET_COLOR
          value: $target_color
        - name: API_BASE_URL
          value: "http://$APP_NAME-$target_color:80"
        - name: DEPLOYMENT_VERSION
          value: $VERSION
      ttlSecondsAfterFinished: 300
EOF
    
    # Wait for job completion
    local job_name
    job_name=$(kubectl get job -n "$NAMESPACE" -l "app.kubernetes.io/name=smoke-test" --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}')
    
    log "Waiting for smoke test job: $job_name"
    kubectl wait --for=condition=complete job/"$job_name" -n "$NAMESPACE" --timeout=300s
    
    # Check job result
    local job_status
    job_status=$(kubectl get job "$job_name" -n "$NAMESPACE" -o jsonpath='{.status.conditions[0].type}')
    
    if [[ "$job_status" == "Complete" ]]; then
        success "Smoke tests passed"
    else
        error "Smoke tests failed"
    fi
    
    # Clean up job
    kubectl delete job "$job_name" -n "$NAMESPACE" --ignore-not-found=true
}

switch_traffic() {
    local target_color="$1"
    local current_color="$2"
    
    header "Switching Traffic to $target_color Environment"
    
    log "Updating main service to point to $target_color..."
    
    # Update the main service selector
    kubectl patch service "$APP_NAME" -n "$NAMESPACE" \
        -p '{"spec":{"selector":{"color":"'$target_color'"}}}'
    
    # Update ingress if it exists
    if kubectl get ingress "$APP_NAME" -n "$NAMESPACE" >/dev/null 2>&1; then
        log "Updating ingress to point to $target_color service..."
        kubectl patch ingress "$APP_NAME" -n "$NAMESPACE" \
            -p '{"spec":{"rules":[{"http":{"paths":[{"backend":{"service":{"name":"'$APP_NAME-$target_color'"}}}]}}]}}'
    fi
    
    # Wait for service endpoint update
    log "Waiting for service endpoints to update..."
    sleep 10
    
    # Verify traffic is going to the right pods
    local endpoint_ips
    endpoint_ips=$(kubectl get endpoints "$APP_NAME" -n "$NAMESPACE" -o jsonpath='{.subsets[0].addresses[*].ip}')
    
    local target_pod_ips
    target_pod_ips=$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=${APP_NAME}-api,color=$target_color" -o jsonpath='{.items[*].status.podIP}')
    
    log "Service endpoints: $endpoint_ips"
    log "Target pod IPs: $target_pod_ips"
    
    success "Traffic switched to $target_color environment"
}

run_post_deployment_validation() {
    local target_color="$1"
    
    header "Running Post-Deployment Validation"
    
    log "Validating traffic routing..."
    
    # Test through the main service
    local api_pod
    api_pod=$(kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=${APP_NAME}-api,color=$target_color" -o jsonpath='{.items[0].metadata.name}')
    
    # Port forward temporarily for testing
    kubectl port-forward -n "$NAMESPACE" service/"$APP_NAME" 8080:80 &
    local port_forward_pid=$!
    sleep 5
    
    # Test API through port forward
    local api_response
    api_response=$(curl -s "http://localhost:8080/api/v1/status" | jq -r '.version' 2>/dev/null || echo "error")
    
    # Clean up port forward
    kill $port_forward_pid >/dev/null 2>&1 || true
    
    if [[ "$api_response" == "$VERSION" ]]; then
        success "Post-deployment validation passed - version $VERSION is live"
    else
        error "Post-deployment validation failed - expected version $VERSION, got $api_response"
    fi
    
    log "Checking deployment metrics..."
    
    # Check if monitoring is collecting metrics
    local metrics_available
    metrics_available=$(kubectl exec -n "$NAMESPACE" "$api_pod" -- curl -s "http://localhost:8000/metrics" | head -5 | wc -l)
    
    if [[ "$metrics_available" -gt 0 ]]; then
        success "Metrics are being collected"
    else
        warn "Metrics collection may not be working properly"
    fi
    
    success "Post-deployment validation completed"
}

cleanup_old_deployment() {
    local old_color="$1"
    local keep_old="${2:-false}"
    
    if [[ "$keep_old" == "true" ]]; then
        log "Keeping old $old_color deployment as requested"
        return 0
    fi
    
    header "Cleaning Up Old $old_color Deployment"
    
    log "Scaling down $old_color deployment..."
    kubectl scale deployment "$APP_NAME-api-$old_color" -n "$NAMESPACE" --replicas=0 || true
    kubectl scale deployment "$APP_NAME-worker-$old_color" -n "$NAMESPACE" --replicas=0 || true
    kubectl scale deployment "$APP_NAME-orchestrator-$old_color" -n "$NAMESPACE" --replicas=0 || true
    
    # Wait a bit for graceful shutdown
    sleep 30
    
    log "Removing $old_color Helm release..."
    helm uninstall "$APP_NAME-$old_color" -n "$NAMESPACE" || true
    
    # Clean up any remaining resources
    kubectl delete all -l "color=$old_color" -n "$NAMESPACE" --ignore-not-found=true
    
    success "Old $old_color deployment cleaned up"
}

create_deployment_record() {
    local target_color="$1"
    local current_color="$2"
    
    log "Creating deployment record..."
    
    cat << EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: deployment-history-$(date +%s)
  namespace: $NAMESPACE
  labels:
    app.kubernetes.io/name: $APP_NAME
    deployment-record: "true"
data:
  deployment_time: "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  from_color: "$current_color"
  to_color: "$target_color"
  version: "$VERSION"
  previous_version: "$PREVIOUS_VERSION"
  deployed_by: "$(whoami)"
  git_commit: "$(git rev-parse HEAD)"
  git_branch: "$(git branch --show-current)"
  deployment_type: "blue-green"
  status: "completed"
EOF
}

rollback_deployment() {
    local failed_color="$1"
    local stable_color="$2"
    
    header "Rolling Back Deployment"
    
    error "Deployment to $failed_color failed, rolling back to $stable_color"
    
    log "Switching traffic back to $stable_color..."
    kubectl patch service "$APP_NAME" -n "$NAMESPACE" \
        -p '{"spec":{"selector":{"color":"'$stable_color'"}}}'
    
    log "Scaling down failed $failed_color deployment..."
    kubectl scale deployment "$APP_NAME-api-$failed_color" -n "$NAMESPACE" --replicas=0 || true
    kubectl scale deployment "$APP_NAME-worker-$failed_color" -n "$NAMESPACE" --replicas=0 || true
    kubectl scale deployment "$APP_NAME-orchestrator-$failed_color" -n "$NAMESPACE" --replicas=0 || true
    
    # Create rollback record
    cat << EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: rollback-record-$(date +%s)
  namespace: $NAMESPACE
  labels:
    app.kubernetes.io/name: $APP_NAME
    rollback-record: "true"
data:
  rollback_time: "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  failed_color: "$failed_color"
  stable_color: "$stable_color"
  target_version: "$VERSION"
  rolled_back_by: "$(whoami)"
  reason: "deployment_validation_failed"
EOF
    
    error "Rollback completed. System is running on $stable_color environment."
    exit 1
}

main() {
    local action="${1:-deploy}"
    local force_color="${2:-}"
    local keep_old="${3:-false}"
    
    header "Xorb Blue-Green Deployment Manager"
    
    case "$action" in
        "deploy"|"upgrade")
            log "Starting blue-green deployment..."
            
            check_prerequisites
            
            local current_color
            current_color=$(get_current_deployment_color)
            
            local target_color
            if [[ -n "$force_color" ]]; then
                target_color="$force_color"
            else
                target_color=$(get_target_deployment_color "$current_color")
            fi
            
            log "Current deployment: $current_color"
            log "Target deployment: $target_color"
            log "Version: $VERSION"
            
            # Build and deploy
            build_and_push_images "$target_color"
            deploy_target_environment "$target_color" "$current_color"
            wait_for_deployment_ready "$target_color" "$TIMEOUT"
            
            # Validation
            if ! run_health_checks "$target_color"; then
                rollback_deployment "$target_color" "$current_color"
            fi
            
            if ! run_smoke_tests "$target_color"; then
                rollback_deployment "$target_color" "$current_color"
            fi
            
            # Switch traffic
            switch_traffic "$target_color" "$current_color"
            
            # Final validation
            if ! run_post_deployment_validation "$target_color"; then
                rollback_deployment "$target_color" "$current_color"
            fi
            
            # Cleanup
            cleanup_old_deployment "$current_color" "$keep_old"
            
            # Record deployment
            create_deployment_record "$target_color" "$current_color"
            
            success "Blue-green deployment completed successfully!"
            success "Active environment: $target_color (version $VERSION)"
            ;;
            
        "status")
            log "Checking deployment status..."
            
            local current_color
            current_color=$(get_current_deployment_color)
            
            echo -e "\n${CYAN}=== Deployment Status ===${NC}"
            echo -e "Active Color: ${GREEN}$current_color${NC}"
            echo -e "Namespace: ${BLUE}$NAMESPACE${NC}"
            
            echo -e "\n${CYAN}=== Deployments ===${NC}"
            kubectl get deployments -n "$NAMESPACE" -l "app.kubernetes.io/name=$APP_NAME"
            
            echo -e "\n${CYAN}=== Services ===${NC}"
            kubectl get services -n "$NAMESPACE" -l "app.kubernetes.io/name=$APP_NAME"
            
            echo -e "\n${CYAN}=== Pods ===${NC}"
            kubectl get pods -n "$NAMESPACE" -l "app.kubernetes.io/name=$APP_NAME"
            ;;
            
        "rollback")
            log "Rolling back deployment..."
            
            local current_color
            current_color=$(get_current_deployment_color)
            
            local previous_color
            previous_color=$(get_target_deployment_color "$current_color")
            
            # Check if previous deployment exists
            if ! kubectl get deployment "$APP_NAME-api-$previous_color" -n "$NAMESPACE" >/dev/null 2>&1; then
                error "Previous deployment ($previous_color) not found. Cannot rollback."
            fi
            
            log "Rolling back from $current_color to $previous_color"
            
            switch_traffic "$previous_color" "$current_color"
            
            success "Rollback completed. Active environment: $previous_color"
            ;;
            
        "cleanup")
            local color_to_cleanup="${2:-}"
            
            if [[ -z "$color_to_cleanup" ]]; then
                error "Please specify color to cleanup: blue or green"
            fi
            
            log "Cleaning up $color_to_cleanup deployment..."
            cleanup_old_deployment "$color_to_cleanup" "false"
            ;;
            
        "help"|"--help"|"-h")
            echo "Xorb Blue-Green Deployment Manager"
            echo ""
            echo "Usage: $0 <command> [options]"
            echo ""
            echo "Commands:"
            echo "  deploy [color] [keep-old]  - Deploy new version using blue-green strategy"
            echo "  status                     - Show current deployment status"
            echo "  rollback                   - Rollback to previous deployment"
            echo "  cleanup <color>            - Cleanup specified color deployment"
            echo "  help                       - Show this help message"
            echo ""
            echo "Options:"
            echo "  color                      - Force deployment to specific color (blue/green)"
            echo "  keep-old                   - Keep old deployment after successful switch"
            echo ""
            echo "Environment Variables:"
            echo "  NAMESPACE                  - Kubernetes namespace (default: xorb-system)"
            echo "  VERSION                    - Version to deploy (default: git commit hash)"
            echo "  REGISTRY                   - Docker registry (default: ghcr.io/xorb)"
            echo "  TIMEOUT                    - Deployment timeout in seconds (default: 600)"
            echo ""
            echo "Examples:"
            echo "  $0 deploy                  - Deploy using automatic color selection"
            echo "  $0 deploy green            - Force deploy to green environment"
            echo "  $0 deploy blue true        - Deploy to blue and keep old deployment"
            echo "  $0 status                  - Show deployment status"
            echo "  $0 rollback                - Rollback to previous version"
            echo "  $0 cleanup green           - Remove green deployment"
            ;;
            
        *)
            error "Unknown command: $action. Use '$0 help' for usage information."
            ;;
    esac
}

# Handle script termination
trap 'echo -e "\n${RED}Deployment interrupted${NC}"; exit 1' INT TERM

# Run main function
main "$@"