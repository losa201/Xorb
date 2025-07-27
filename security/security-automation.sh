#!/bin/bash

# XORB Security Automation System
# Automated security policy deployment and management

set -euo pipefail

echo "🤖 XORB Security Automation"
echo "=========================="

ACTION="${1:-deploy}"
ENVIRONMENT="${2:-development}"

case "$ACTION" in
    "deploy")
        echo "🚀 Deploying security policies..."
        
        # Deploy Kubernetes network policies
        if command -v kubectl &> /dev/null; then
            echo "📋 Applying Kubernetes network policies..."
            kubectl apply -f /root/Xorb/security/policies/kubernetes/ -n "xorb-$ENVIRONMENT" || echo "⚠️  Kubernetes not available"
        fi
        
        # Configure Docker network security
        echo "🐳 Configuring Docker network security..."
        /root/Xorb/security/policies/docker/docker-network-security.sh
        
        # Apply iptables rules (only in production)
        if [ "$ENVIRONMENT" = "production" ]; then
            echo "🔥 Applying firewall rules..."
            /root/Xorb/security/firewall/iptables-rules.sh
        fi
        
        # Start network monitoring
        echo "🔍 Starting network monitoring..."
        nohup /root/Xorb/security/monitoring/network-monitoring.sh > /var/log/xorb/network-monitor.out 2>&1 &
        
        echo "✅ Security automation deployed"
        ;;
        
    "status")
        echo "📊 Security Status Report"
        echo "========================"
        
        # Check Kubernetes network policies
        if command -v kubectl &> /dev/null; then
            echo "📋 Kubernetes Network Policies:"
            kubectl get networkpolicy -n "xorb-$ENVIRONMENT" 2>/dev/null || echo "No network policies found"
        fi
        
        # Check Docker networks
        echo ""
        echo "🐳 Docker Networks:"
        docker network ls | grep xorb
        
        # Check iptables rules
        echo ""
        echo "🔥 Active Firewall Rules:"
        iptables -L INPUT -n | head -10
        
        # Check monitoring processes
        echo ""
        echo "🔍 Security Monitoring:"
        ps aux | grep network-monitoring || echo "Network monitoring not running"
        ;;
        
    "stop")
        echo "🛑 Stopping security monitoring..."
        pkill -f network-monitoring.sh || echo "No monitoring processes found"
        echo "✅ Security monitoring stopped"
        ;;
        
    *)
        echo "Usage: $0 {deploy|status|stop} [environment]"
        echo "Available environments: development, staging, production"
        exit 1
        ;;
esac
