#!/bin/bash
set -euo pipefail

# XORB Monitoring Stack Deployment Script
# Deploys comprehensive monitoring, logging, and tracing infrastructure

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
K8S_DIR="${PROJECT_ROOT}/k8s/monitoring"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed or not in PATH"
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        error "helm is not installed or not in PATH"
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
    fi
    
    log "Prerequisites check passed"
}

# Create namespaces
create_namespaces() {
    log "Creating monitoring namespaces..."
    kubectl apply -f "${K8S_DIR}/namespace.yaml"
    
    # Wait for namespaces to be ready
    kubectl wait --for=condition=Active namespace/xorb-monitoring --timeout=30s
    kubectl wait --for=condition=Active namespace/xorb-logging --timeout=30s
    kubectl wait --for=condition=Active namespace/xorb-tracing --timeout=30s
    
    log "Namespaces created successfully"
}

# Deploy Prometheus
deploy_prometheus() {
    log "Deploying Prometheus..."
    kubectl apply -f "${K8S_DIR}/prometheus/configmap.yaml"
    kubectl apply -f "${K8S_DIR}/prometheus/deployment.yaml"
    
    # Wait for Prometheus to be ready
    kubectl wait --for=condition=Available deployment/prometheus -n xorb-monitoring --timeout=300s
    
    log "Prometheus deployed successfully"
}

# Deploy Grafana
deploy_grafana() {
    log "Deploying Grafana..."
    kubectl apply -f "${K8S_DIR}/grafana/configmap.yaml"
    kubectl apply -f "${K8S_DIR}/grafana/deployment.yaml"
    
    # Wait for Grafana to be ready
    kubectl wait --for=condition=Available deployment/grafana -n xorb-monitoring --timeout=300s
    
    log "Grafana deployed successfully"
}

# Deploy Alertmanager
deploy_alertmanager() {
    log "Deploying Alertmanager..."
    kubectl apply -f "${K8S_DIR}/alertmanager/configmap.yaml"
    kubectl apply -f "${K8S_DIR}/alertmanager/deployment.yaml"
    
    # Wait for Alertmanager to be ready
    kubectl wait --for=condition=Available deployment/alertmanager -n xorb-monitoring --timeout=300s
    
    log "Alertmanager deployed successfully"
}

# Deploy Loki
deploy_loki() {
    log "Deploying Loki..."
    kubectl apply -f "${K8S_DIR}/loki/deployment.yaml"
    
    # Wait for Loki to be ready
    kubectl wait --for=condition=Available deployment/loki -n xorb-logging --timeout=300s
    
    log "Loki deployed successfully"
}

# Deploy Promtail
deploy_promtail() {
    log "Deploying Promtail..."
    kubectl apply -f "${K8S_DIR}/promtail/daemonset.yaml"
    
    # Wait for Promtail DaemonSet to be ready
    kubectl rollout status daemonset/promtail -n xorb-logging --timeout=300s
    
    log "Promtail deployed successfully"
}

# Deploy Jaeger
deploy_jaeger() {
    log "Deploying Jaeger..."
    kubectl apply -f "${K8S_DIR}/jaeger/deployment.yaml"
    
    # Wait for Jaeger to be ready
    kubectl wait --for=condition=Available deployment/jaeger-all-in-one -n xorb-tracing --timeout=300s
    
    log "Jaeger deployed successfully"
}

# Deploy Node Exporter
deploy_node_exporter() {
    log "Deploying Node Exporter..."
    kubectl apply -f "${K8S_DIR}/node-exporter/daemonset.yaml"
    
    # Wait for Node Exporter DaemonSet to be ready
    kubectl rollout status daemonset/node-exporter -n xorb-monitoring --timeout=300s
    
    log "Node Exporter deployed successfully"
}

# Deploy ingress controllers
deploy_ingress() {
    log "Deploying monitoring ingress resources..."
    
    # Check if SSL certificate ARN is provided
    if [[ -z "${SSL_CERTIFICATE_ARN:-}" ]]; then
        warn "SSL_CERTIFICATE_ARN not set, using placeholder in ingress"
        export SSL_CERTIFICATE_ARN="arn:aws:acm:us-west-2:123456789012:certificate/12345678-1234-1234-1234-123456789012"
    fi
    
    # Apply ingress with environment variable substitution
    envsubst < "${K8S_DIR}/ingress.yaml" | kubectl apply -f -
    
    log "Ingress resources deployed successfully"
}

# Verify deployment
verify_deployment() {
    log "Verifying monitoring stack deployment..."
    
    info "Checking Prometheus..."
    kubectl get pods -n xorb-monitoring -l app=prometheus
    
    info "Checking Grafana..."
    kubectl get pods -n xorb-monitoring -l app=grafana
    
    info "Checking Alertmanager..."
    kubectl get pods -n xorb-monitoring -l app=alertmanager
    
    info "Checking Loki..."
    kubectl get pods -n xorb-logging -l app=loki
    
    info "Checking Promtail..."
    kubectl get pods -n xorb-logging -l app=promtail
    
    info "Checking Jaeger..."
    kubectl get pods -n xorb-tracing -l app=jaeger
    
    info "Checking Node Exporter..."
    kubectl get pods -n xorb-monitoring -l app=node-exporter
    
    log "Deployment verification completed"
}

# Display access information
display_access_info() {
    log "Monitoring stack deployed successfully!"
    
    echo ""
    echo "Access Information:"
    echo "=================="
    
    # Get LoadBalancer IPs/hostnames
    local prometheus_lb=$(kubectl get ingress xorb-monitoring-ingress -n xorb-monitoring -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "Pending...")
    local loki_lb=$(kubectl get ingress xorb-logging-ingress -n xorb-logging -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "Pending...")
    local jaeger_lb=$(kubectl get ingress xorb-tracing-ingress -n xorb-tracing -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "Pending...")
    
    echo "Prometheus: http://prometheus.xorb.local (${prometheus_lb})"
    echo "Grafana: http://grafana.xorb.local (${prometheus_lb})"
    echo "  - Username: admin"
    echo "  - Password: xorb-admin-2024"
    echo "Alertmanager: http://alertmanager.xorb.local (${prometheus_lb})"
    echo "Loki: http://loki.xorb.local (${loki_lb})"
    echo "Jaeger: http://jaeger.xorb.local (${jaeger_lb})"
    
    echo ""
    echo "Port Forwarding (for local access):"
    echo "==================================="
    echo "kubectl port-forward -n xorb-monitoring svc/prometheus 9090:9090"
    echo "kubectl port-forward -n xorb-monitoring svc/grafana 3000:3000"
    echo "kubectl port-forward -n xorb-monitoring svc/alertmanager 9093:9093"
    echo "kubectl port-forward -n xorb-logging svc/loki 3100:3100"
    echo "kubectl port-forward -n xorb-tracing svc/jaeger-query 16686:16686"
    
    echo ""
    echo "Next Steps:"
    echo "==========="
    echo "1. Configure DNS or update /etc/hosts to point domains to LoadBalancer IPs"
    echo "2. Deploy XORB applications with monitoring annotations"
    echo "3. Configure alert notification channels in Alertmanager"
    echo "4. Set up long-term storage for metrics (optional)"
}

# Main deployment function
main() {
    log "Starting XORB monitoring stack deployment..."
    
    check_prerequisites
    create_namespaces
    deploy_prometheus
    deploy_grafana
    deploy_alertmanager
    deploy_loki
    deploy_promtail
    deploy_jaeger
    deploy_node_exporter
    deploy_ingress
    verify_deployment
    display_access_info
    
    log "XORB monitoring stack deployment completed successfully!"
}

# Handle script arguments
case "${1:-deploy}" in
    deploy)
        main
        ;;
    verify)
        verify_deployment
        ;;
    info)
        display_access_info
        ;;
    clean)
        log "Cleaning up monitoring stack..."
        kubectl delete namespace xorb-monitoring xorb-logging xorb-tracing --ignore-not-found=true
        log "Cleanup completed"
        ;;
    *)
        echo "Usage: $0 {deploy|verify|info|clean}"
        echo "  deploy  - Deploy the complete monitoring stack (default)"
        echo "  verify  - Verify deployment status"
        echo "  info    - Display access information"
        echo "  clean   - Remove all monitoring components"
        exit 1
        ;;
esac