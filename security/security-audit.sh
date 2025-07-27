#!/bin/bash

# XORB Network Security Audit
# Comprehensive security assessment and compliance checking

echo "🔒 XORB Network Security Audit"
echo "=============================="
echo "Generated: $(date)"
echo ""

# Check firewall status
echo "🔥 Firewall Status:"
if command -v iptables &> /dev/null; then
    echo "  - iptables rules: $(iptables -L INPUT | wc -l) rules configured"
    echo "  - Default policy: $(iptables -L | grep "Chain INPUT" | awk '{print $4}')"
else
    echo "  - iptables: Not available"
fi

# Check network policies
echo ""
echo "📋 Network Policies:"
if command -v kubectl &> /dev/null; then
    for ns in xorb-development xorb-staging xorb-production; do
        policy_count=$(kubectl get networkpolicy -n "$ns" 2>/dev/null | wc -l)
        echo "  - $ns: $policy_count network policies"
    done
else
    echo "  - Kubernetes: Not available"
fi

# Check open ports
echo ""
echo "🚪 Open Ports Analysis:"
netstat -tuln | grep LISTEN | while read line; do
    port=$(echo "$line" | awk '{print $4}' | cut -d: -f2)
    protocol=$(echo "$line" | awk '{print $1}')
    case "$port" in
        22) echo "  - Port $port ($protocol): SSH - Secure" ;;
        80) echo "  - Port $port ($protocol): HTTP - Should redirect to HTTPS" ;;
        443) echo "  - Port $port ($protocol): HTTPS - Secure" ;;
        8000|8080|9000) echo "  - Port $port ($protocol): XORB Service - Verify access controls" ;;
        5432) echo "  - Port $port ($protocol): PostgreSQL - Should be internal only" ;;
        6379) echo "  - Port $port ($protocol): Redis - Should be internal only" ;;
        9090|3000) echo "  - Port $port ($protocol): Monitoring - Should be restricted" ;;
        *) echo "  - Port $port ($protocol): Unknown service - Review required" ;;
    esac
done

# Check Docker security
echo ""
echo "🐳 Docker Security:"
if command -v docker &> /dev/null; then
    echo "  - Docker networks: $(docker network ls | grep xorb | wc -l) XORB networks"
    echo "  - Running containers: $(docker ps | grep xorb | wc -l) XORB containers"
else
    echo "  - Docker: Not available"
fi

# Check SSL/TLS configuration
echo ""
echo "🔒 SSL/TLS Configuration:"
if [ -f "/root/Xorb/ssl/certs/server-cert.pem" ]; then
    expiry=$(openssl x509 -in /root/Xorb/ssl/certs/server-cert.pem -enddate -noout | cut -d= -f2)
    echo "  - SSL certificates: Present"
    echo "  - Certificate expiry: $expiry"
else
    echo "  - SSL certificates: Not found"
fi

# Security recommendations
echo ""
echo "🎯 Security Recommendations:"
echo "  1. Regularly rotate SSL certificates"
echo "  2. Monitor network traffic for anomalies"
echo "  3. Keep fail2ban rules updated"
echo "  4. Review firewall rules monthly"
echo "  5. Implement network segmentation"
echo "  6. Enable audit logging for all services"
echo "  7. Use VPN for administrative access"
echo "  8. Implement intrusion detection system"

echo ""
echo "📊 Audit completed. Review findings and implement recommendations."
