#!/bin/bash

# XORB Enterprise Platform - Production Deployment Script
# Strategic deployment automation for enterprise environments
# Author: Principal Auditor & Engineering Expert

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
NAMESPACE="${NAMESPACE:-xorb-production}"
HELM_RELEASE_NAME="${HELM_RELEASE_NAME:-xorb}"
KUBECONFIG="${KUBECONFIG:-~/.kube/config}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
print_banner() {
    echo "=================================================================="
    echo "  XORB Enterprise Platform - Production Deployment"
    echo "  Strategic deployment automation for enterprise environments"
    echo "=================================================================="
}

# Prerequisites check
check_prerequisites() {
    log_info "Checking deployment prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "Helm is not installed. Please install Helm first."
        exit 1
    fi
    
    # Check docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    # Check cluster version
    K8S_VERSION=$(kubectl version --output=json | jq -r '.serverVersion.gitVersion')
    log_info "Kubernetes cluster version: ${K8S_VERSION}"
    
    # Check if namespace exists
    if kubectl get namespace "${NAMESPACE}" &> /dev/null; then
        log_warning "Namespace ${NAMESPACE} already exists."
    fi
    
    log_success "Prerequisites check completed successfully"
}

# Generate secrets
generate_secrets() {
    log_info "Generating production secrets..."
    
    # Database credentials
    DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    DB_URL="postgresql://xorb:${DB_PASSWORD}@postgres-primary:5432/xorb_production"
    
    # Redis credentials
    REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    REDIS_URL="redis://:${REDIS_PASSWORD}@redis:6379/0"
    
    # JWT secret
    JWT_SECRET=$(openssl rand -base64 64 | tr -d "=+/" | cut -c1-64)
    
    # Create secrets directory
    mkdir -p "${PROJECT_ROOT}/deploy/secrets"
    
    # Write secrets to file (encrypted)
    cat > "${PROJECT_ROOT}/deploy/secrets/production-secrets.yaml" << EOF
apiVersion: v1
kind: Secret
metadata:
  name: database-credentials
  namespace: ${NAMESPACE}
type: Opaque
stringData:
  username: "xorb"
  password: "${DB_PASSWORD}"
  url: "${DB_URL}"
---
apiVersion: v1
kind: Secret
metadata:
  name: redis-credentials
  namespace: ${NAMESPACE}
type: Opaque
stringData:
  password: "${REDIS_PASSWORD}"
  url: "${REDIS_URL}"
---
apiVersion: v1
kind: Secret
metadata:
  name: jwt-secrets
  namespace: ${NAMESPACE}
type: Opaque
stringData:
  secret: "${JWT_SECRET}"
---
apiVersion: v1
kind: Secret
metadata:
  name: external-api-keys
  namespace: ${NAMESPACE}
type: Opaque
stringData:
  nvidia: "${NVIDIA_API_KEY:-}"
  openrouter: "${OPENROUTER_API_KEY:-}"
EOF

    log_success "Production secrets generated successfully"
    log_warning "Secrets file created at: deploy/secrets/production-secrets.yaml"
    log_warning "Please secure this file and consider using external secret management in production"
}

# Build and push images
build_and_push_images() {
    log_info "Building and pushing Docker images..."
    
    # Read image registry from environment
    IMAGE_REGISTRY="${IMAGE_REGISTRY:-docker.io/xorb}"
    IMAGE_TAG="${IMAGE_TAG:-1.0.0}"
    
    # Build API image
    log_info "Building XORB API image..."
    docker build -t "${IMAGE_REGISTRY}/api:${IMAGE_TAG}" \
        -f "${PROJECT_ROOT}/src/api/Dockerfile" \
        "${PROJECT_ROOT}"
    
    # Build Orchestrator image
    log_info "Building XORB Orchestrator image..."
    docker build -t "${IMAGE_REGISTRY}/orchestrator:${IMAGE_TAG}" \
        -f "${PROJECT_ROOT}/src/orchestrator/Dockerfile" \
        "${PROJECT_ROOT}"
    
    # Push images if registry is not local
    if [[ "${IMAGE_REGISTRY}" != "localhost"* ]]; then
        log_info "Pushing images to registry..."
        docker push "${IMAGE_REGISTRY}/api:${IMAGE_TAG}"
        docker push "${IMAGE_REGISTRY}/orchestrator:${IMAGE_TAG}"
        log_success "Images pushed successfully"
    else
        log_info "Using local registry, skipping push"
    fi
}

# Deploy infrastructure
deploy_infrastructure() {
    log_info "Deploying infrastructure components..."
    
    # Create namespace
    kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply namespace configuration
    kubectl apply -f "${PROJECT_ROOT}/deploy/kubernetes/production/namespace.yaml"
    
    # Deploy monitoring stack
    log_info "Deploying monitoring stack..."
    kubectl apply -f "${PROJECT_ROOT}/infra/monitoring/production-monitoring-stack.yaml"
    
    # Wait for monitoring stack to be ready
    log_info "Waiting for monitoring stack to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/prometheus -n monitoring
    
    log_success "Infrastructure components deployed successfully"
}

# Deploy XORB platform using Helm
deploy_xorb_platform() {
    log_info "Deploying XORB Enterprise Platform using Helm..."
    
    # Add required Helm repositories
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
    helm repo update
    
    # Apply secrets first
    kubectl apply -f "${PROJECT_ROOT}/deploy/secrets/production-secrets.yaml"
    
    # Create custom values file for production
    cat > "${PROJECT_ROOT}/deploy/helm/production-values.yaml" << EOF
global:
  storageClass: "${STORAGE_CLASS:-fast-ssd}"

api:
  replicaCount: 3
  image:
    repository: ${IMAGE_REGISTRY:-docker.io/xorb}/api
    tag: ${IMAGE_TAG:-1.0.0}
  
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 20

orchestrator:
  replicaCount: 2
  image:
    repository: ${IMAGE_REGISTRY:-docker.io/xorb}/orchestrator
    tag: ${IMAGE_TAG:-1.0.0}

postgresql:
  enabled: true
  auth:
    existingSecret: database-credentials
    secretKeys:
      adminPasswordKey: password
      userPasswordKey: password
  primary:
    persistence:
      size: 100Gi
      storageClass: "${STORAGE_CLASS:-fast-ssd}"

redis:
  enabled: true
  auth:
    existingSecret: redis-credentials
    existingSecretPasswordKey: password
  master:
    persistence:
      size: 20Gi
      storageClass: "${STORAGE_CLASS:-fast-ssd}"

ingress:
  enabled: ${INGRESS_ENABLED:-true}
  hosts:
    - host: ${DOMAIN_NAME:-api.xorb-security.com}
      paths:
        - path: /
          pathType: Prefix

monitoring:
  enabled: true

security:
  networkPolicy:
    enabled: true
  podSecurityPolicy:
    enabled: true
EOF
    
    # Deploy using Helm
    helm upgrade --install "${HELM_RELEASE_NAME}" \
        "${PROJECT_ROOT}/deploy/helm/xorb" \
        --namespace "${NAMESPACE}" \
        --values "${PROJECT_ROOT}/deploy/helm/production-values.yaml" \
        --timeout 20m \
        --wait
    
    log_success "XORB Enterprise Platform deployed successfully"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pod status
    log_info "Checking pod status..."
    kubectl get pods -n "${NAMESPACE}"
    
    # Wait for all deployments to be ready
    log_info "Waiting for deployments to be ready..."
    kubectl wait --for=condition=available --timeout=600s deployment/xorb-api -n "${NAMESPACE}"
    kubectl wait --for=condition=available --timeout=600s deployment/xorb-orchestrator -n "${NAMESPACE}"
    
    # Check services
    log_info "Checking services..."
    kubectl get services -n "${NAMESPACE}"
    
    # Check ingress
    if [[ "${INGRESS_ENABLED:-true}" == "true" ]]; then
        log_info "Checking ingress..."
        kubectl get ingress -n "${NAMESPACE}"
    fi
    
    # Health check
    log_info "Performing health checks..."
    API_POD=$(kubectl get pods -n "${NAMESPACE}" -l app=xorb-api -o jsonpath='{.items[0].metadata.name}')
    
    if kubectl exec -n "${NAMESPACE}" "${API_POD}" -- curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        return 1
    fi
    
    log_success "Deployment verification completed successfully"
}

# Display deployment information
display_deployment_info() {
    log_info "Deployment Information:"
    echo "=================================="
    echo "Namespace: ${NAMESPACE}"
    echo "Helm Release: ${HELM_RELEASE_NAME}"
    echo "Environment: ${DEPLOYMENT_ENV}"
    echo ""
    
    # Get LoadBalancer IP
    if [[ "${INGRESS_ENABLED:-true}" == "true" ]]; then
        INGRESS_IP=$(kubectl get ingress -n "${NAMESPACE}" -o jsonpath='{.items[0].status.loadBalancer.ingress[0].ip}')
        if [[ -n "${INGRESS_IP}" ]]; then
            echo "External IP: ${INGRESS_IP}"
            echo "API Endpoint: https://${DOMAIN_NAME:-api.xorb-security.com}"
        else
            echo "Ingress IP: Pending..."
        fi
    fi
    
    echo ""
    echo "Useful Commands:"
    echo "  View pods: kubectl get pods -n ${NAMESPACE}"
    echo "  View logs: kubectl logs -f deployment/xorb-api -n ${NAMESPACE}"
    echo "  Port forward: kubectl port-forward svc/xorb-api 8000:80 -n ${NAMESPACE}"
    echo "  Delete deployment: helm uninstall ${HELM_RELEASE_NAME} -n ${NAMESPACE}"
    echo "=================================="
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f "${PROJECT_ROOT}/deploy/helm/production-values.yaml"
}

# Main deployment function
main() {
    print_banner
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-infra)
                SKIP_INFRA=true
                shift
                ;;
            --domain)
                DOMAIN_NAME="$2"
                shift 2
                ;;
            --registry)
                IMAGE_REGISTRY="$2"
                shift 2
                ;;
            --storage-class)
                STORAGE_CLASS="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --skip-build        Skip building and pushing images"
                echo "  --skip-infra        Skip infrastructure deployment"
                echo "  --domain DOMAIN     Set custom domain name"
                echo "  --registry REGISTRY Set custom image registry"
                echo "  --storage-class SC  Set custom storage class"
                echo "  --help              Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Trap cleanup on exit
    trap cleanup EXIT
    
    # Execute deployment steps
    check_prerequisites
    generate_secrets
    
    if [[ "${SKIP_BUILD:-false}" != "true" ]]; then
        build_and_push_images
    fi
    
    if [[ "${SKIP_INFRA:-false}" != "true" ]]; then
        deploy_infrastructure
    fi
    
    deploy_xorb_platform
    verify_deployment
    display_deployment_info
    
    log_success "XORB Enterprise Platform deployment completed successfully!"
    log_info "The platform is now ready for production use."
}

# Execute main function
main "$@"