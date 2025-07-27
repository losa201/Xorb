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
