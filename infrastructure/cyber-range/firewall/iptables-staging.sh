#!/bin/bash
# XORB PTaaS Cyber Range - iptables Configuration for STAGING Mode
# This script configures firewall rules for safe training exercises
# Red team attacks are blocked, blue team monitoring is allowed

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
LOGFILE="/var/log/cyber-range/firewall-staging.log"
mkdir -p "$(dirname "$LOGFILE")"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [STAGING] $*" | tee -a "$LOGFILE"
}

error() {
    echo -e "${RED}ERROR: $*${NC}" | tee -a "$LOGFILE"
    exit 1
}

info() {
    echo -e "${BLUE}INFO: $*${NC}" | tee -a "$LOGFILE"
}

warning() {
    echo -e "${YELLOW}WARNING: $*${NC}" | tee -a "$LOGFILE"
}

success() {
    echo -e "${GREEN}SUCCESS: $*${NC}" | tee -a "$LOGFILE"
}

# Check if running as root
[[ $EUID -eq 0 ]] || error "This script must be run as root"

# Network definitions
CONTROL_PLANE_NETWORK="10.10.10.0/24"
RED_TEAM_NETWORK="10.20.0.0/16"
BLUE_TEAM_NETWORK="10.30.0.0/24"
TARGET_WEB_NETWORK="10.100.0.0/24"
TARGET_INTERNAL_NETWORK="10.110.0.0/24"
TARGET_OT_NETWORK="10.120.0.0/24"
SIMULATION_NETWORK="10.200.0.0/24"
VPC_CIDR="10.0.0.0/16"

# Backup existing rules
backup_iptables() {
    info "Backing up existing iptables rules..."
    iptables-save > "/var/log/cyber-range/iptables-backup-staging-$(date +%Y%m%d-%H%M%S).rules"
    success "iptables rules backed up"
}

# Clear existing rules
clear_iptables() {
    info "Clearing existing iptables rules..."
    
    # Set default policies to ACCEPT temporarily
    iptables -P INPUT ACCEPT
    iptables -P FORWARD ACCEPT
    iptables -P OUTPUT ACCEPT
    
    # Flush all chains
    iptables -F
    iptables -X
    iptables -t nat -F
    iptables -t nat -X
    iptables -t mangle -F
    iptables -t mangle -X
    
    success "iptables rules cleared"
}

# Set default policies
set_default_policies() {
    info "Setting default policies for STAGING mode..."
    
    # Default policies - restrictive
    iptables -P INPUT DROP
    iptables -P FORWARD DROP
    iptables -P OUTPUT ACCEPT
    
    success "Default policies set"
}

# Basic system rules
setup_basic_rules() {
    info "Setting up basic system rules..."
    
    # Allow loopback traffic
    iptables -A INPUT -i lo -j ACCEPT
    iptables -A OUTPUT -o lo -j ACCEPT
    
    # Allow established and related connections
    iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
    iptables -A FORWARD -m state --state ESTABLISHED,RELATED -j ACCEPT
    
    # Allow SSH for management (restrict to management networks in production)
    iptables -A INPUT -p tcp --dport 22 -s $CONTROL_PLANE_NETWORK -j ACCEPT
    iptables -A INPUT -p tcp --dport 22 -s $BLUE_TEAM_NETWORK -j ACCEPT
    
    # Allow ICMP (ping) for network diagnostics
    iptables -A INPUT -p icmp --icmp-type echo-request -j ACCEPT
    iptables -A FORWARD -p icmp --icmp-type echo-request -j ACCEPT
    
    success "Basic system rules configured"
}

# Control plane rules
setup_control_plane_rules() {
    info "Setting up control plane rules..."
    
    # Allow control plane to manage all networks
    iptables -A FORWARD -s $CONTROL_PLANE_NETWORK -j ACCEPT
    iptables -A FORWARD -d $CONTROL_PLANE_NETWORK -j ACCEPT
    
    # XORB Orchestrator access
    iptables -A INPUT -p tcp --dport 8080 -s $VPC_CIDR -j ACCEPT
    
    # Kill switch emergency access (priority rule)
    iptables -I INPUT 1 -p tcp --dport 8081 -s $CONTROL_PLANE_NETWORK -j ACCEPT
    
    # Monitoring and metrics
    iptables -A INPUT -p tcp --dport 9090 -s $CONTROL_PLANE_NETWORK -j ACCEPT  # Prometheus
    iptables -A INPUT -p tcp --dport 3000 -s $CONTROL_PLANE_NETWORK -j ACCEPT  # Grafana
    
    success "Control plane rules configured"
}

# Red team rules (STAGING MODE - BLOCKED)
setup_red_team_rules_staging() {
    info "Setting up red team rules for STAGING mode (ATTACKS BLOCKED)..."
    
    # LOG and DROP red team attacks to targets
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $TARGET_WEB_NETWORK \
        -m limit --limit 5/min --limit-burst 10 \
        -j LOG --log-prefix "CYBER-RANGE-STAGING: RED->WEB BLOCKED: "
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $TARGET_WEB_NETWORK -j DROP
    
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $TARGET_INTERNAL_NETWORK \
        -m limit --limit 5/min --limit-burst 10 \
        -j LOG --log-prefix "CYBER-RANGE-STAGING: RED->INT BLOCKED: "
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $TARGET_INTERNAL_NETWORK -j DROP
    
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $TARGET_OT_NETWORK \
        -m limit --limit 5/min --limit-burst 10 \
        -j LOG --log-prefix "CYBER-RANGE-STAGING: RED->OT BLOCKED: "
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $TARGET_OT_NETWORK -j DROP
    
    # Block red team access to blue team and control plane
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $BLUE_TEAM_NETWORK -j DROP
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $CONTROL_PLANE_NETWORK -j DROP
    
    # Allow red team internal communication and internet access for updates
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $RED_TEAM_NETWORK -j ACCEPT
    iptables -A FORWARD -s $RED_TEAM_NETWORK -o eth0 -j ACCEPT  # Internet access
    
    warning "Red team attacks to targets are BLOCKED in staging mode"
    success "Red team staging rules configured"
}

# Blue team rules (monitoring allowed)
setup_blue_team_rules() {
    info "Setting up blue team monitoring rules..."
    
    # Allow blue team to monitor all target networks
    iptables -A FORWARD -s $BLUE_TEAM_NETWORK -d $TARGET_WEB_NETWORK -j ACCEPT
    iptables -A FORWARD -s $BLUE_TEAM_NETWORK -d $TARGET_INTERNAL_NETWORK -j ACCEPT
    iptables -A FORWARD -s $BLUE_TEAM_NETWORK -d $TARGET_OT_NETWORK -j ACCEPT
    
    # Allow targets to send logs/data to blue team SIEM
    iptables -A FORWARD -s $TARGET_WEB_NETWORK -d $BLUE_TEAM_NETWORK -p tcp --dport 5044:5046 -j ACCEPT  # Logstash
    iptables -A FORWARD -s $TARGET_INTERNAL_NETWORK -d $BLUE_TEAM_NETWORK -p tcp --dport 5044:5046 -j ACCEPT
    iptables -A FORWARD -s $TARGET_OT_NETWORK -d $BLUE_TEAM_NETWORK -p tcp --dport 5044:5046 -j ACCEPT
    
    # Wazuh agent communication
    iptables -A FORWARD -s $TARGET_WEB_NETWORK -d $BLUE_TEAM_NETWORK -p tcp --dport 1514:1516 -j ACCEPT
    iptables -A FORWARD -s $TARGET_INTERNAL_NETWORK -d $BLUE_TEAM_NETWORK -p tcp --dport 1514:1516 -j ACCEPT
    iptables -A FORWARD -s $TARGET_OT_NETWORK -d $BLUE_TEAM_NETWORK -p tcp --dport 1514:1516 -j ACCEPT
    
    # Allow blue team access to control plane for reporting
    iptables -A FORWARD -s $BLUE_TEAM_NETWORK -d $CONTROL_PLANE_NETWORK -j ACCEPT
    
    # Block blue team from directly accessing red team (prevent cheating)
    iptables -A FORWARD -s $BLUE_TEAM_NETWORK -d $RED_TEAM_NETWORK -j DROP
    
    success "Blue team monitoring rules configured"
}

# Target environment rules
setup_target_rules() {
    info "Setting up target environment rules..."
    
    # Allow inter-target communication (lateral movement simulation)
    iptables -A FORWARD -s $TARGET_WEB_NETWORK -d $TARGET_INTERNAL_NETWORK -j ACCEPT
    iptables -A FORWARD -s $TARGET_INTERNAL_NETWORK -d $TARGET_WEB_NETWORK -j ACCEPT
    iptables -A FORWARD -s $TARGET_INTERNAL_NETWORK -d $TARGET_OT_NETWORK -j ACCEPT
    
    # Allow simulation traffic to targets
    iptables -A FORWARD -s $SIMULATION_NETWORK -d $TARGET_WEB_NETWORK -j ACCEPT
    iptables -A FORWARD -s $SIMULATION_NETWORK -d $TARGET_INTERNAL_NETWORK -j ACCEPT
    iptables -A FORWARD -s $SIMULATION_NETWORK -d $TARGET_OT_NETWORK -j ACCEPT
    
    # Allow targets limited internet access (can be disabled)
    iptables -A FORWARD -s $TARGET_WEB_NETWORK -o eth0 -p tcp --dport 80 -j ACCEPT
    iptables -A FORWARD -s $TARGET_WEB_NETWORK -o eth0 -p tcp --dport 443 -j ACCEPT
    iptables -A FORWARD -s $TARGET_INTERNAL_NETWORK -o eth0 -p tcp --dport 80 -j ACCEPT
    iptables -A FORWARD -s $TARGET_INTERNAL_NETWORK -o eth0 -p tcp --dport 443 -j ACCEPT
    
    success "Target environment rules configured"
}

# Simulation network rules
setup_simulation_rules() {
    info "Setting up simulation network rules..."
    
    # Allow simulation to generate background traffic
    iptables -A FORWARD -s $SIMULATION_NETWORK -j ACCEPT
    
    # Allow control plane to manage simulation
    iptables -A FORWARD -s $CONTROL_PLANE_NETWORK -d $SIMULATION_NETWORK -j ACCEPT
    
    success "Simulation network rules configured"
}

# DNS rules
setup_dns_rules() {
    info "Setting up DNS rules..."
    
    # Allow DNS queries from all networks
    iptables -A FORWARD -p udp --dport 53 -j ACCEPT
    iptables -A FORWARD -p tcp --dport 53 -j ACCEPT
    iptables -A INPUT -p udp --dport 53 -j ACCEPT
    iptables -A INPUT -p tcp --dport 53 -j ACCEPT
    
    success "DNS rules configured"
}

# Kubernetes specific rules
setup_kubernetes_rules() {
    info "Setting up Kubernetes-specific rules..."
    
    # Allow Kubernetes API server access
    iptables -A INPUT -p tcp --dport 6443 -s $VPC_CIDR -j ACCEPT
    
    # Allow kubelet communication
    iptables -A INPUT -p tcp --dport 10250 -s $VPC_CIDR -j ACCEPT
    
    # Allow NodePort services
    iptables -A INPUT -p tcp --dport 30000:32767 -s $VPC_CIDR -j ACCEPT
    
    # Allow pod-to-pod communication within nodes
    iptables -A FORWARD -s 192.168.0.0/16 -d 192.168.0.0/16 -j ACCEPT  # Pod CIDR
    
    success "Kubernetes rules configured"
}

# Emergency kill switch rules
setup_kill_switch_rules() {
    info "Setting up emergency kill switch rules..."
    
    # Create custom chain for kill switch
    iptables -N CYBER_RANGE_KILL_SWITCH 2>/dev/null || true
    
    # Jump to kill switch chain for all red team traffic (highest priority)
    iptables -I FORWARD 1 -s $RED_TEAM_NETWORK -j CYBER_RANGE_KILL_SWITCH
    
    # Kill switch chain initially allows traffic (will be modified by kill switch)
    iptables -A CYBER_RANGE_KILL_SWITCH -j RETURN
    
    success "Emergency kill switch rules configured"
}

# Logging and monitoring rules
setup_logging_rules() {
    info "Setting up logging and monitoring rules..."
    
    # Log dropped packets (rate limited)
    iptables -A INPUT -m limit --limit 5/min --limit-burst 10 \
        -j LOG --log-prefix "CYBER-RANGE-STAGING: INPUT DROP: "
    
    iptables -A FORWARD -m limit --limit 5/min --limit-burst 10 \
        -j LOG --log-prefix "CYBER-RANGE-STAGING: FORWARD DROP: "
    
    success "Logging rules configured"
}

# Apply rate limiting
setup_rate_limiting() {
    info "Setting up rate limiting rules..."
    
    # Rate limit SSH connections
    iptables -I INPUT -p tcp --dport 22 -m state --state NEW -m recent --set
    iptables -I INPUT -p tcp --dport 22 -m state --state NEW -m recent --update --seconds 60 --hitcount 4 -j DROP
    
    # Rate limit HTTP connections to prevent DoS
    iptables -I FORWARD -p tcp --dport 80 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT
    iptables -I FORWARD -p tcp --dport 443 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT
    
    success "Rate limiting configured"
}

# Save rules
save_iptables() {
    info "Saving iptables rules..."
    
    # Create systemd service for persistence
    cat > /etc/systemd/system/cyber-range-iptables.service << 'EOF'
[Unit]
Description=XORB Cyber Range iptables rules
After=network.target

[Service]
Type=oneshot
ExecStart=/sbin/iptables-restore /etc/iptables/cyber-range-staging.rules
ExecReload=/sbin/iptables-restore /etc/iptables/cyber-range-staging.rules
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

    # Save current rules
    mkdir -p /etc/iptables
    iptables-save > /etc/iptables/cyber-range-staging.rules
    
    # Enable service
    systemctl daemon-reload
    systemctl enable cyber-range-iptables
    
    success "iptables rules saved and service enabled"
}

# Verify configuration
verify_config() {
    info "Verifying firewall configuration..."
    
    local errors=0
    
    # Check if rules are applied
    if ! iptables -L | grep -q "CYBER_RANGE_KILL_SWITCH"; then
        error "Kill switch chain not found"
        ((errors++))
    fi
    
    # Check default policies
    if ! iptables -L | grep -q "policy DROP"; then
        warning "Default policies may not be properly set"
        ((errors++))
    fi
    
    # Test key rules
    info "Testing key firewall rules..."
    
    # Check if red team is blocked from targets
    if iptables -L FORWARD | grep -q "DROP.*10\.20\.0\.0/16.*10\.100\.0\.0/24"; then
        success "Red team -> Web targets: BLOCKED ✓"
    else
        warning "Red team -> Web targets: Rule not found"
        ((errors++))
    fi
    
    if iptables -L FORWARD | grep -q "ACCEPT.*10\.30\.0\.0/24.*10\.100\.0\.0/24"; then
        success "Blue team -> Web targets: ALLOWED ✓"
    else
        warning "Blue team -> Web targets: Rule not found"
        ((errors++))
    fi
    
    if [ $errors -eq 0 ]; then
        success "Firewall configuration verification passed"
        return 0
    else
        warning "Firewall configuration has $errors issues"
        return 1
    fi
}

# Display status
show_status() {
    echo -e "\n${BLUE}=== XORB Cyber Range Firewall Status (STAGING MODE) ===${NC}"
    echo -e "${GREEN}Mode: STAGING (Red team attacks BLOCKED)${NC}"
    echo -e "${YELLOW}Red Team Network: $RED_TEAM_NETWORK (RESTRICTED)${NC}"
    echo -e "${BLUE}Blue Team Network: $BLUE_TEAM_NETWORK (MONITORING ALLOWED)${NC}"
    echo -e "${GREEN}Control Plane: $CONTROL_PLANE_NETWORK (FULL ACCESS)${NC}"
    echo
    echo -e "${BLUE}Active Rules Summary:${NC}"
    echo "- Red team attacks to targets: BLOCKED"
    echo "- Blue team monitoring: ALLOWED"
    echo "- Control plane management: ALLOWED"
    echo "- Emergency kill switch: ENABLED"
    echo "- Logging: ENABLED"
    echo
    echo "Rule counts:"
    echo "- INPUT rules: $(iptables -L INPUT --line-numbers | wc -l)"
    echo "- FORWARD rules: $(iptables -L FORWARD --line-numbers | wc -l)"
    echo "- OUTPUT rules: $(iptables -L OUTPUT --line-numbers | wc -l)"
    echo
    echo -e "${GREEN}To switch to LIVE mode, run: ./iptables-live.sh${NC}"
    echo -e "${RED}Emergency kill switch: ./kill-switch.sh activate${NC}"
}

# Main execution
main() {
    info "Starting XORB Cyber Range firewall configuration for STAGING mode..."
    
    backup_iptables
    clear_iptables
    set_default_policies
    setup_basic_rules
    setup_control_plane_rules
    setup_red_team_rules_staging  # Red team blocked in staging
    setup_blue_team_rules
    setup_target_rules
    setup_simulation_rules
    setup_dns_rules
    setup_kubernetes_rules
    setup_kill_switch_rules
    setup_logging_rules
    setup_rate_limiting
    save_iptables
    
    if verify_config; then
        show_status
        success "XORB Cyber Range firewall configured for STAGING mode"
        warning "RED TEAM ATTACKS ARE BLOCKED - This is safe training mode"
    else
        error "Firewall configuration failed verification"
    fi
}

# Handle script arguments
case "${1:-}" in
    "verify")
        verify_config
        ;;
    "status")
        show_status
        ;;
    *)
        main
        ;;
esac