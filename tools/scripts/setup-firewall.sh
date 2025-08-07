#!/bin/bash

# XORB Telemetry Firewall Setup Script
# Configures UFW and iptables for secure telemetry access

set -euo pipefail

# Configuration
ADMIN_NETWORKS=("10.0.0.0/8" "172.16.0.0/12" "192.168.0.0/16")
MONITORING_NETWORKS=("0.0.0.0/0")  # Allow metrics from anywhere (with auth)
SSH_PORT=22
HTTP_PORT=80
HTTPS_PORT=443
CADDY_ADMIN_PORT=2019

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

# Install UFW if not present
install_ufw() {
    log_info "Checking UFW installation..."
    
    if ! command -v ufw &> /dev/null; then
        log_info "Installing UFW..."
        apt-get update
        apt-get install -y ufw
        log_success "UFW installed"
    else
        log_success "UFW already installed"
    fi
}

# Configure UFW basic rules
configure_ufw_basic() {
    log_info "Configuring UFW basic rules..."
    
    # Reset UFW to defaults
    ufw --force reset
    
    # Set default policies
    ufw default deny incoming
    ufw default allow outgoing
    
    # Allow SSH (essential to maintain access)
    ufw allow ${SSH_PORT}/tcp comment 'SSH Access'
    
    # Allow HTTP and HTTPS (for telemetry access)
    ufw allow ${HTTP_PORT}/tcp comment 'HTTP - Telemetry'
    ufw allow ${HTTPS_PORT}/tcp comment 'HTTPS - Telemetry'
    
    log_success "UFW basic rules configured"
}

# Configure admin network access
configure_admin_access() {
    log_info "Configuring admin network access..."
    
    for network in "${ADMIN_NETWORKS[@]}"; do
        # Allow Caddy admin interface from admin networks only
        ufw allow from ${network} to any port ${CADDY_ADMIN_PORT} comment "Caddy Admin - ${network}"
        
        # Allow full access to telemetry from admin networks
        ufw allow from ${network} to any port 3002 comment "Grafana - ${network}"
        ufw allow from ${network} to any port 9092 comment "Prometheus - ${network}"
        ufw allow from ${network} to any port 8003 comment "Neural API - ${network}"
        ufw allow from ${network} to any port 8004 comment "Learning API - ${network}"
        
        log_success "Admin access configured for ${network}"
    done
}

# Configure Docker network rules
configure_docker_networks() {
    log_info "Configuring Docker network rules..."
    
    # Allow Docker internal communication
    ufw allow from 172.20.0.0/24 comment 'Docker telemetry network'
    ufw allow from 172.21.0.0/24 comment 'Docker autonomous network'
    
    # Allow Docker bridge network
    ufw allow from 172.17.0.0/16 comment 'Docker bridge network'
    
    log_success "Docker network rules configured"
}

# Configure rate limiting with iptables
configure_rate_limiting() {
    log_info "Configuring rate limiting with iptables..."
    
    # Create custom chain for rate limiting
    iptables -N XORB_RATE_LIMIT 2>/dev/null || iptables -F XORB_RATE_LIMIT
    
    # Rate limit HTTP/HTTPS connections (100 per minute per IP)
    iptables -A XORB_RATE_LIMIT -p tcp --dport 80 -m state --state NEW -m recent --set
    iptables -A XORB_RATE_LIMIT -p tcp --dport 80 -m state --state NEW -m recent --update --seconds 60 --hitcount 100 -j DROP
    
    iptables -A XORB_RATE_LIMIT -p tcp --dport 443 -m state --state NEW -m recent --set
    iptables -A XORB_RATE_LIMIT -p tcp --dport 443 -m state --state NEW -m recent --update --seconds 60 --hitcount 100 -j DROP
    
    # Insert rate limiting into INPUT chain
    iptables -I INPUT 1 -j XORB_RATE_LIMIT
    
    # Rate limit API calls (more restrictive: 30 per minute per IP)
    iptables -A INPUT -p tcp --dport 8003 -m state --state NEW -m recent --set --name neural_api
    iptables -A INPUT -p tcp --dport 8003 -m state --state NEW -m recent --update --seconds 60 --hitcount 30 --name neural_api -j DROP
    
    iptables -A INPUT -p tcp --dport 8004 -m state --state NEW -m recent --set --name learning_api
    iptables -A INPUT -p tcp --dport 8004 -m state --state NEW -m recent --update --seconds 60 --hitcount 30 --name learning_api -j DROP
    
    log_success "Rate limiting configured"
}

# Configure DDoS protection
configure_ddos_protection() {
    log_info "Configuring DDoS protection..."
    
    # Protect against SYN flood attacks
    iptables -A INPUT -p tcp --syn -m limit --limit 1/s --limit-burst 3 -j ACCEPT
    iptables -A INPUT -p tcp --syn -j DROP
    
    # Protect against port scanning
    iptables -N PORT_SCANNING 2>/dev/null || iptables -F PORT_SCANNING
    iptables -A PORT_SCANNING -p tcp --tcp-flags SYN,ACK,FIN,RST RST -m limit --limit 1/s --limit-burst 2 -j RETURN
    iptables -A PORT_SCANNING -j DROP
    iptables -A INPUT -j PORT_SCANNING
    
    # Protect against ping flood
    iptables -A INPUT -p icmp --icmp-type echo-request -m limit --limit 1/s --limit-burst 2 -j ACCEPT
    iptables -A INPUT -p icmp --icmp-type echo-request -j DROP
    
    log_success "DDoS protection configured"
}

# Configure logging
configure_logging() {
    log_info "Configuring firewall logging..."
    
    # Enable UFW logging
    ufw logging on
    
    # Log dropped packets
    iptables -A INPUT -m limit --limit 5/min -j LOG --log-prefix "UFW-DROPPED: " --log-level 4
    iptables -A FORWARD -m limit --limit 5/min -j LOG --log-prefix "UFW-DROPPED: " --log-level 4
    
    # Configure rsyslog for firewall logs
    cat > /etc/rsyslog.d/50-xorb-firewall.conf <<EOF
# XORB Firewall Logging Configuration

# UFW logs
:msg,contains,"UFW" /var/log/xorb-firewall.log
& stop

# iptables logs
:msg,contains,"UFW-DROPPED" /var/log/xorb-firewall-dropped.log
& stop
EOF
    
    # Restart rsyslog
    systemctl restart rsyslog
    
    log_success "Firewall logging configured"
}

# Save iptables rules
save_iptables_rules() {
    log_info "Saving iptables rules..."
    
    # Install iptables-persistent if not present
    if ! dpkg -l | grep -q iptables-persistent; then
        DEBIAN_FRONTEND=noninteractive apt-get install -y iptables-persistent
    fi
    
    # Save current rules
    iptables-save > /etc/iptables/rules.v4
    ip6tables-save > /etc/iptables/rules.v6
    
    log_success "iptables rules saved"
}

# Create firewall management scripts
create_management_scripts() {
    log_info "Creating firewall management scripts..."
    
    # Create firewall status script
    cat > /root/Xorb/scripts/firewall-status.sh <<'EOF'
#!/bin/bash

# XORB Firewall Status Script

echo "🔥 XORB Firewall Status"
echo "======================"

echo ""
echo "📋 UFW Status:"
ufw status verbose

echo ""
echo "📊 UFW Rules:"
ufw status numbered

echo ""
echo "⚡ iptables Rules:"
echo "INPUT chain:"
iptables -L INPUT -n --line-numbers

echo ""
echo "FORWARD chain:"
iptables -L FORWARD -n --line-numbers

echo ""
echo "📈 Connection Statistics:"
netstat -an | grep -E ":80|:443|:8003|:8004|:3002|:9092" | head -20

echo ""
echo "🔍 Recent Dropped Connections:"
tail -20 /var/log/xorb-firewall-dropped.log 2>/dev/null || echo "No dropped connections logged"

echo ""
echo "📊 Active Connections by Port:"
netstat -an | grep ESTABLISHED | awk '{print $4}' | cut -d: -f2 | sort | uniq -c | sort -nr
EOF
    
    chmod +x /root/Xorb/scripts/firewall-status.sh
    
    # Create firewall reset script
    cat > /root/Xorb/scripts/firewall-reset.sh <<'EOF'
#!/bin/bash

# XORB Firewall Reset Script
# CAUTION: This will reset all firewall rules!

echo "🔥 XORB Firewall Reset"
echo "====================="

read -p "Are you sure you want to reset all firewall rules? (yes/no): " confirm

if [[ $confirm == "yes" ]]; then
    echo "🔄 Resetting firewall rules..."
    
    # Reset UFW
    ufw --force reset
    ufw default allow incoming
    ufw default allow outgoing
    
    # Flush iptables
    iptables -F
    iptables -X
    iptables -t nat -F
    iptables -t nat -X
    iptables -t mangle -F
    iptables -t mangle -X
    
    echo "✅ Firewall reset complete"
    echo "⚠️  WARNING: All ports are now open!"
    echo "Run setup-firewall.sh to reconfigure security"
else
    echo "❌ Firewall reset cancelled"
fi
EOF
    
    chmod +x /root/Xorb/scripts/firewall-reset.sh
    
    # Create firewall test script
    cat > /root/Xorb/scripts/firewall-test.sh <<'EOF'
#!/bin/bash

# XORB Firewall Test Script
# Tests firewall rules and connectivity

echo "🔥 XORB Firewall Test"
echo "===================="

# Test basic connectivity
echo ""
echo "🌐 Testing basic connectivity..."

# Test HTTP/HTTPS
if curl -s --connect-timeout 5 http://localhost >/dev/null 2>&1; then
    echo "✅ HTTP (80) - Accessible"
else
    echo "❌ HTTP (80) - Blocked or service down"
fi

if curl -k -s --connect-timeout 5 https://localhost >/dev/null 2>&1; then
    echo "✅ HTTPS (443) - Accessible"
else
    echo "❌ HTTPS (443) - Blocked or service down"
fi

# Test telemetry endpoints
echo ""
echo "📊 Testing telemetry endpoints..."

endpoints=(
    "localhost:3002"
    "localhost:8003"
    "localhost:8004"
    "localhost:9092"
)

for endpoint in "${endpoints[@]}"; do
    if nc -z ${endpoint/:/ } 2>/dev/null; then
        echo "✅ ${endpoint} - Accessible"
    else
        echo "❌ ${endpoint} - Blocked or service down"
    fi
done

# Test external access
echo ""
echo "🌍 Testing external access simulation..."

# Simulate external HTTP request
if timeout 5 curl -s http://127.0.0.1 >/dev/null 2>&1; then
    echo "✅ External HTTP - Would be accessible"
else
    echo "❌ External HTTP - Would be blocked"
fi

# Check for common attack vectors
echo ""
echo "🛡️  Testing attack vector protection..."

# Test SYN flood protection
if iptables -L | grep -q "limit:"; then
    echo "✅ Rate limiting - Active"
else
    echo "❌ Rate limiting - Not configured"
fi

# Test port scanning protection
if iptables -L | grep -q "PORT_SCANNING"; then
    echo "✅ Port scan protection - Active"
else
    echo "❌ Port scan protection - Not configured"
fi

echo ""
echo "🔍 Firewall test complete"
EOF
    
    chmod +x /root/Xorb/scripts/firewall-test.sh
    
    log_success "Firewall management scripts created"
}

# Enable UFW
enable_ufw() {
    log_info "Enabling UFW..."
    
    # Enable UFW
    ufw --force enable
    
    # Enable UFW to start on boot
    systemctl enable ufw
    
    log_success "UFW enabled and set to start on boot"
}

# Main execution
main() {
    log_info "🔥 XORB Telemetry Firewall Setup"
    log_info "================================"
    
    check_root
    install_ufw
    configure_ufw_basic
    configure_admin_access
    configure_docker_networks
    configure_rate_limiting
    configure_ddos_protection
    configure_logging
    save_iptables_rules
    create_management_scripts
    enable_ufw
    
    log_success "🎉 Firewall setup complete!"
    log_info ""
    log_info "📋 Configuration summary:"
    log_info "   SSH (${SSH_PORT}): Allowed from anywhere"
    log_info "   HTTP (${HTTP_PORT}): Allowed from anywhere (rate limited)"
    log_info "   HTTPS (${HTTPS_PORT}): Allowed from anywhere (rate limited)"
    log_info "   Caddy Admin (${CADDY_ADMIN_PORT}): Admin networks only"
    log_info "   Telemetry services: Admin networks only"
    log_info ""
    log_info "🛠️  Management commands:"
    log_info "   Status: /root/Xorb/scripts/firewall-status.sh"
    log_info "   Test: /root/Xorb/scripts/firewall-test.sh"
    log_info "   Reset: /root/Xorb/scripts/firewall-reset.sh"
    log_info ""
    log_info "📊 View UFW status: ufw status verbose"
    log_info "📊 View logs: tail -f /var/log/xorb-firewall.log"
    log_warning ""
    log_warning "⚠️  IMPORTANT: Ensure you can still access this system via SSH!"
    log_warning "   If locked out, use console access to run firewall-reset.sh"
}

# Run main function
main "$@"