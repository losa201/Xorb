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
