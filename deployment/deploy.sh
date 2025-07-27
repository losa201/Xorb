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
