#!/bin/bash

# XORB Advanced Network Security Policies
# Comprehensive network security automation and zero-trust implementation

set -euo pipefail

echo "🛡️  XORB Advanced Network Security Policies"
echo "==========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_step() {
    echo -e "${BLUE}🔧 $1${NC}"
}

log_security() {
    echo -e "${PURPLE}🔒 $1${NC}"
}

# Configuration
SECURITY_DIR="/root/Xorb/security"
POLICIES_DIR="$SECURITY_DIR/policies"
FIREWALL_DIR="$SECURITY_DIR/firewall"
MONITORING_DIR="$SECURITY_DIR/monitoring"

# Create security directory structure
log_step "Creating network security directory structure..."
mkdir -p "$POLICIES_DIR"/{kubernetes,docker,iptables} "$FIREWALL_DIR" "$MONITORING_DIR"

# Create Kubernetes Network Policies
log_step "Creating Kubernetes network policies..."

# Default deny all policy
cat > "$POLICIES_DIR/kubernetes/00-default-deny.yaml" << 'EOF'
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: xorb-production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: xorb-staging
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: xorb-development
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
EOF

# API service network policy
cat > "$POLICIES_DIR/kubernetes/api-network-policy.yaml" << 'EOF'
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: xorb-api-network-policy
spec:
  podSelector:
    matchLabels:
      app: xorb-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - podSelector:
        matchLabels:
          app: xorb-orchestrator
    - podSelector:
        matchLabels:
          app: xorb-worker
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: xorb-postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: xorb-redis
    ports:
    - protocol: TCP
      port: 6379
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
  - to: []
    ports:
    - protocol: TCP
      port: 443
EOF

# Database network policy
cat > "$POLICIES_DIR/kubernetes/database-network-policy.yaml" << 'EOF'
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: xorb-database-network-policy
spec:
  podSelector:
    matchLabels:
      app: xorb-postgres
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: xorb-api
    - podSelector:
        matchLabels:
          app: xorb-orchestrator
    - podSelector:
        matchLabels:
          app: xorb-worker
    ports:
    - protocol: TCP
      port: 5432
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: xorb-redis-network-policy
spec:
  podSelector:
    matchLabels:
      app: xorb-redis
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: xorb-api
    - podSelector:
        matchLabels:
          app: xorb-orchestrator
    - podSelector:
        matchLabels:
          app: xorb-worker
    ports:
    - protocol: TCP
      port: 6379
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
EOF

# Monitoring network policy
cat > "$POLICIES_DIR/kubernetes/monitoring-network-policy.yaml" << 'EOF'
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: xorb-monitoring-network-policy
spec:
  podSelector:
    matchLabels:
      app: xorb-prometheus
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: xorb-grafana
    ports:
    - protocol: TCP
      port: 9090
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: xorb-api
    - podSelector:
        matchLabels:
          app: xorb-orchestrator
    - podSelector:
        matchLabels:
          app: xorb-worker
    ports:
    - protocol: TCP
      port: 8000
    - protocol: TCP
      port: 8080
    - protocol: TCP
      port: 9000
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: xorb-grafana-network-policy
spec:
  podSelector:
    matchLabels:
      app: xorb-grafana
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 3000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: xorb-prometheus
    ports:
    - protocol: TCP
      port: 9090
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
  - to: []
    ports:
    - protocol: TCP
      port: 443
EOF

# Create Docker network security policies
log_step "Creating Docker network security policies..."

cat > "$POLICIES_DIR/docker/docker-network-security.sh" << 'EOF'
#!/bin/bash

# Docker Network Security Configuration
echo "🐳 Configuring Docker network security..."

# Create isolated networks for different environments
docker network create --driver bridge \
  --subnet=172.20.0.0/16 \
  --ip-range=172.20.1.0/24 \
  --gateway=172.20.0.1 \
  xorb-production-network 2>/dev/null || echo "Production network exists"

docker network create --driver bridge \
  --subnet=172.21.0.0/16 \
  --ip-range=172.21.1.0/24 \
  --gateway=172.21.0.1 \
  xorb-staging-network 2>/dev/null || echo "Staging network exists"

docker network create --driver bridge \
  --subnet=172.22.0.0/16 \
  --ip-range=172.22.1.0/24 \
  --gateway=172.22.0.1 \
  xorb-development-network 2>/dev/null || echo "Development network exists"

# Create DMZ network for external-facing services
docker network create --driver bridge \
  --subnet=172.30.0.0/16 \
  --ip-range=172.30.1.0/24 \
  --gateway=172.30.0.1 \
  xorb-dmz-network 2>/dev/null || echo "DMZ network exists"

echo "✅ Docker networks configured"
EOF

chmod +x "$POLICIES_DIR/docker/docker-network-security.sh"

# Create iptables firewall rules
log_step "Creating iptables firewall rules..."

cat > "$FIREWALL_DIR/iptables-rules.sh" << 'EOF'
#!/bin/bash

# XORB Advanced Firewall Rules
# Comprehensive iptables configuration for XORB security

echo "🔥 Configuring advanced firewall rules..."

# Backup existing rules
iptables-save > /root/Xorb/security/firewall/iptables-backup-$(date +%Y%m%d_%H%M%S).rules

# Flush existing rules
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X

# Set default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow established and related connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (rate limited)
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m limit --limit 5/min --limit-burst 10 -j ACCEPT

# Allow HTTP/HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow XORB services (restricted to specific source IPs in production)
iptables -A INPUT -p tcp --dport 8000 -s 172.0.0.0/8 -j ACCEPT  # API
iptables -A INPUT -p tcp --dport 8080 -s 172.0.0.0/8 -j ACCEPT  # Orchestrator
iptables -A INPUT -p tcp --dport 9000 -s 172.0.0.0/8 -j ACCEPT  # Worker

# Allow monitoring services (restricted)
iptables -A INPUT -p tcp --dport 9090 -s 172.0.0.0/8 -j ACCEPT  # Prometheus
iptables -A INPUT -p tcp --dport 3000 -s 172.0.0.0/8 -j ACCEPT  # Grafana

# Allow database connections (very restricted)
iptables -A INPUT -p tcp --dport 5432 -s 172.20.0.0/16 -j ACCEPT  # PostgreSQL
iptables -A INPUT -p tcp --dport 6379 -s 172.20.0.0/16 -j ACCEPT  # Redis

# DDoS protection
iptables -A INPUT -p tcp --dport 80 -m limit --limit 100/minute --limit-burst 200 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -m limit --limit 100/minute --limit-burst 200 -j ACCEPT

# Log dropped packets
iptables -A INPUT -m limit --limit 5/min -j LOG --log-prefix "iptables-dropped: " --log-level 4

# Drop everything else
iptables -A INPUT -j DROP

# Save rules
iptables-save > /etc/iptables/rules.v4 2>/dev/null || iptables-save > /root/Xorb/security/firewall/iptables-active.rules

echo "✅ Firewall rules configured"
EOF

chmod +x "$FIREWALL_DIR/iptables-rules.sh"

# Create fail2ban configuration
log_step "Creating fail2ban configuration..."

cat > "$FIREWALL_DIR/fail2ban-xorb.conf" << 'EOF'
# XORB Fail2ban Configuration
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5
backend = auto

[xorb-api]
enabled = true
port = 8000
filter = xorb-api
logpath = /var/log/xorb/api.log
maxretry = 10
bantime = 1800

[xorb-orchestrator]
enabled = true
port = 8080
filter = xorb-orchestrator
logpath = /var/log/xorb/orchestrator.log
maxretry = 5
bantime = 3600

[xorb-worker]
enabled = true
port = 9000
filter = xorb-worker
logpath = /var/log/xorb/worker.log
maxretry = 5
bantime = 3600

[ssh-aggressive]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 86400
EOF

# Create fail2ban filters
mkdir -p "$FIREWALL_DIR/fail2ban-filters"

cat > "$FIREWALL_DIR/fail2ban-filters/xorb-api.conf" << 'EOF'
[Definition]
failregex = ^.*\[ERROR\].*Client <HOST>.*authentication failed.*$
            ^.*\[WARNING\].*Client <HOST>.*rate limit exceeded.*$
            ^.*\[ERROR\].*Client <HOST>.*malicious request detected.*$
ignoreregex =
EOF

# Create network intrusion detection
log_step "Creating network intrusion detection..."

cat > "$MONITORING_DIR/network-monitoring.sh" << 'EOF'
#!/bin/bash

# XORB Network Intrusion Detection System
# Real-time network monitoring and threat detection

echo "🔍 Starting XORB network monitoring..."

# Function to monitor suspicious activity
monitor_connections() {
    echo "📡 Monitoring network connections..."

    # Monitor for port scanning
    netstat -tuln | grep LISTEN | while read line; do
        port=$(echo "$line" | awk '{print $4}' | cut -d: -f2)
        if [ "$port" -gt 1024 ] && [ "$port" -lt 65535 ]; then
            echo "$(date): Listening on port $port" >> /var/log/xorb/network-monitor.log
        fi
    done

    # Monitor for unusual traffic patterns
    ss -tuln | grep -E "(8000|8080|9000|5432|6379)" | while read line; do
        echo "$(date): XORB service connection: $line" >> /var/log/xorb/network-monitor.log
    done
}

# Function to detect anomalies
detect_anomalies() {
    echo "🚨 Detecting network anomalies..."

    # Check for excessive connections
    CONN_COUNT=$(netstat -an | grep ESTABLISHED | wc -l)
    if [ "$CONN_COUNT" -gt 1000 ]; then
        echo "$(date): WARNING: High connection count: $CONN_COUNT" >> /var/log/xorb/security-alerts.log
    fi

    # Check for suspicious IPs
    netstat -an | grep ESTABLISHED | awk '{print $5}' | cut -d: -f1 | sort | uniq -c | sort -nr | head -10 | while read count ip; do
        if [ "$count" -gt 50 ]; then
            echo "$(date): WARNING: Suspicious IP activity: $ip ($count connections)" >> /var/log/xorb/security-alerts.log
        fi
    done
}

# Function to analyze traffic
analyze_traffic() {
    echo "📊 Analyzing network traffic..."

    # Monitor XORB service ports
    for port in 8000 8080 9000; do
        connections=$(netstat -an | grep ":$port " | grep ESTABLISHED | wc -l)
        echo "$(date): Port $port active connections: $connections" >> /var/log/xorb/traffic-analysis.log
    done
}

# Create log directories
mkdir -p /var/log/xorb

# Main monitoring loop
while true; do
    monitor_connections
    detect_anomalies
    analyze_traffic
    sleep 60
done
EOF

chmod +x "$MONITORING_DIR/network-monitoring.sh"

# Create security automation
log_step "Creating security automation..."

cat > "$SECURITY_DIR/security-automation.sh" << 'EOF'
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
EOF

chmod +x "$SECURITY_DIR/security-automation.sh"

# Create network security audit
log_step "Creating network security audit..."

cat > "$SECURITY_DIR/security-audit.sh" << 'EOF'
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
EOF

chmod +x "$SECURITY_DIR/security-audit.sh"

# Set secure permissions
log_step "Setting secure permissions..."
chmod 700 "$SECURITY_DIR"
chmod 600 "$SECURITY_DIR"/*.sh
find "$POLICIES_DIR" -type f -exec chmod 644 {} \;

# Test security automation
log_step "Testing security automation..."
if [ -x "$SECURITY_DIR/security-automation.sh" ]; then
    log_info "Security automation scripts are ready"
else
    log_error "Security automation scripts are not executable"
fi

echo ""
log_info "Advanced network security policies setup complete!"
echo ""
echo "🔒 Security Management Commands:"
echo "   - Deploy policies: $SECURITY_DIR/security-automation.sh deploy <environment>"
echo "   - Check status: $SECURITY_DIR/security-automation.sh status <environment>"
echo "   - Security audit: $SECURITY_DIR/security-audit.sh"
echo "   - Stop monitoring: $SECURITY_DIR/security-automation.sh stop"
echo ""
echo "🛡️  Security Components:"
echo "   - Kubernetes network policies: $POLICIES_DIR/kubernetes/"
echo "   - Docker network security: $POLICIES_DIR/docker/"
echo "   - Firewall rules: $FIREWALL_DIR/"
echo "   - Network monitoring: $MONITORING_DIR/"
echo ""
echo "🚀 Deploy security for production:"
echo "   $SECURITY_DIR/security-automation.sh deploy production"
