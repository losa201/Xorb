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
