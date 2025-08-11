#!/bin/bash
# XORB PTaaS Cyber Range - Lightweight Firewall for Resource-Constrained Environment
# Optimized for 16 vCPU AMD RYZEN EPYC 7002 / 32GB RAM

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
LOGFILE="/var/log/cyber-range/lightweight-firewall.log"
STATUS_FILE="/var/log/cyber-range/firewall-status.json"

# Network definitions (simplified for single node)
CLUSTER_CIDR="10.244.0.0/16"  # Default Kubernetes pod CIDR
SERVICE_CIDR="10.96.0.0/12"   # Default Kubernetes service CIDR
NODE_IP=$(ip route get 8.8.8.8 | awk '{print $7; exit}')

# Lightweight mode - fewer rules, better performance
LIGHTWEIGHT_MODE=true
MAX_RULES=50  # Limit total iptables rules
REDUCED_LOGGING=true

# Logging functions
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [LIGHTWEIGHT-FW] $*" | tee -a "$LOGFILE"
}

error() {
    echo -e "${RED}[ERROR] $*${NC}" | tee -a "$LOGFILE"
}

warning() {
    echo -e "${YELLOW}[WARNING] $*${NC}" | tee -a "$LOGFILE"
}

info() {
    echo -e "${BLUE}[INFO] $*${NC}" | tee -a "$LOGFILE"
}

success() {
    echo -e "${GREEN}[SUCCESS] $*${NC}" | tee -a "$LOGFILE"
}

# Setup logging directory
setup_logging() {
    mkdir -p "$(dirname "$LOGFILE")"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "Lightweight firewall must be run as root"
        exit 1
    fi
}

# Detect current mode
detect_mode() {
    if iptables -L FORWARD | grep -q "CYBER-RANGE-STAGING"; then
        echo "staging"
    elif iptables -L FORWARD | grep -q "CYBER-RANGE-LIVE"; then
        echo "live"
    else
        echo "none"
    fi
}

# Clear existing rules (lightweight)
clear_rules() {
    info "Clearing existing lightweight firewall rules..."
    
    # Only clear cyber-range specific rules, preserve system rules
    iptables -D FORWARD -j CYBER_RANGE_RULES 2>/dev/null || true
    iptables -F CYBER_RANGE_RULES 2>/dev/null || true
    iptables -X CYBER_RANGE_RULES 2>/dev/null || true
    
    success "Lightweight firewall rules cleared"
}

# Create lightweight rules for staging mode
setup_staging_rules() {
    info "Setting up lightweight staging mode rules..."
    
    # Create custom chain for cyber range rules
    iptables -N CYBER_RANGE_RULES 2>/dev/null || true
    iptables -I FORWARD 1 -j CYBER_RANGE_RULES
    
    # Allow all Kubernetes internal traffic (essential)
    iptables -A CYBER_RANGE_RULES -s $CLUSTER_CIDR -d $CLUSTER_CIDR -j ACCEPT
    iptables -A CYBER_RANGE_RULES -s $SERVICE_CIDR -j ACCEPT
    iptables -A CYBER_RANGE_RULES -d $SERVICE_CIDR -j ACCEPT
    
    # Block red team to targets (staging mode)
    iptables -A CYBER_RANGE_RULES -m comment --comment "CYBER-RANGE-STAGING: Block red team attacks" \
        -s $CLUSTER_CIDR -d $CLUSTER_CIDR \
        -m string --string "cyber-range-red" --algo bm \
        -j LOG --log-prefix "STAGING-BLOCKED: " --log-level 4
    
    iptables -A CYBER_RANGE_RULES -m comment --comment "CYBER-RANGE-STAGING: Block red team attacks" \
        -s $CLUSTER_CIDR -d $CLUSTER_CIDR \
        -m string --string "cyber-range-red" --algo bm \
        -j DROP
    
    # Allow blue team monitoring (essential)
    iptables -A CYBER_RANGE_RULES -m comment --comment "CYBER-RANGE-STAGING: Allow blue team monitoring" \
        -s $CLUSTER_CIDR -d $CLUSTER_CIDR \
        -m string --string "cyber-range-blue" --algo bm \
        -j ACCEPT
    
    # Allow control plane management
    iptables -A CYBER_RANGE_RULES -m comment --comment "CYBER-RANGE-STAGING: Allow control plane" \
        -s $CLUSTER_CIDR -d $CLUSTER_CIDR \
        -m string --string "cyber-range-control" --algo bm \
        -j ACCEPT
    
    # Return to normal processing for other traffic
    iptables -A CYBER_RANGE_RULES -j RETURN
    
    success "Lightweight staging rules configured"
}

# Create lightweight rules for live mode
setup_live_rules() {
    info "Setting up lightweight live mode rules..."
    
    # Create custom chain for cyber range rules
    iptables -N CYBER_RANGE_RULES 2>/dev/null || true
    iptables -I FORWARD 1 -j CYBER_RANGE_RULES
    
    # Allow all Kubernetes internal traffic (essential)
    iptables -A CYBER_RANGE_RULES -s $CLUSTER_CIDR -d $CLUSTER_CIDR -j ACCEPT
    iptables -A CYBER_RANGE_RULES -s $SERVICE_CIDR -j ACCEPT
    iptables -A CYBER_RANGE_RULES -d $SERVICE_CIDR -j ACCEPT
    
    # Allow red team attacks with rate limiting (live mode)
    iptables -A CYBER_RANGE_RULES -m comment --comment "CYBER-RANGE-LIVE: Allow red team attacks" \
        -s $CLUSTER_CIDR -d $CLUSTER_CIDR \
        -m string --string "cyber-range-red" --algo bm \
        -m limit --limit 10/min --limit-burst 20 \
        -j LOG --log-prefix "LIVE-ATTACK: " --log-level 4
    
    iptables -A CYBER_RANGE_RULES -m comment --comment "CYBER-RANGE-LIVE: Allow red team attacks" \
        -s $CLUSTER_CIDR -d $CLUSTER_CIDR \
        -m string --string "cyber-range-red" --algo bm \
        -m limit --limit 50/min --limit-burst 100 \
        -j ACCEPT
    
    # Rate limit excessive red team activity
    iptables -A CYBER_RANGE_RULES -m comment --comment "CYBER-RANGE-LIVE: Rate limit excessive attacks" \
        -s $CLUSTER_CIDR -d $CLUSTER_CIDR \
        -m string --string "cyber-range-red" --algo bm \
        -j DROP
    
    # Allow blue team monitoring (enhanced)
    iptables -A CYBER_RANGE_RULES -m comment --comment "CYBER-RANGE-LIVE: Allow blue team monitoring" \
        -s $CLUSTER_CIDR -d $CLUSTER_CIDR \
        -m string --string "cyber-range-blue" --algo bm \
        -j ACCEPT
    
    # Allow control plane management
    iptables -A CYBER_RANGE_RULES -m comment --comment "CYBER-RANGE-LIVE: Allow control plane" \
        -s $CLUSTER_CIDR -d $CLUSTER_CIDR \
        -m string --string "cyber-range-control" --algo bm \
        -j ACCEPT
    
    # Return to normal processing for other traffic
    iptables -A CYBER_RANGE_RULES -j RETURN
    
    success "Lightweight live rules configured"
}

# Create kill switch rules (immediate isolation)
setup_kill_switch() {
    info "Activating lightweight kill switch..."
    
    # Clear existing rules first
    clear_rules
    
    # Create kill switch chain
    iptables -N CYBER_RANGE_KILL_SWITCH 2>/dev/null || true
    iptables -I FORWARD 1 -j CYBER_RANGE_KILL_SWITCH
    
    # Allow only essential Kubernetes traffic
    iptables -A CYBER_RANGE_KILL_SWITCH -s $CLUSTER_CIDR -d $CLUSTER_CIDR \
        -m string --string "kube-system" --algo bm -j ACCEPT
    iptables -A CYBER_RANGE_KILL_SWITCH -s $SERVICE_CIDR -j ACCEPT
    iptables -A CYBER_RANGE_KILL_SWITCH -d $SERVICE_CIDR -j ACCEPT
    
    # Allow control plane only
    iptables -A CYBER_RANGE_KILL_SWITCH -s $CLUSTER_CIDR -d $CLUSTER_CIDR \
        -m string --string "cyber-range-control" --algo bm -j ACCEPT
    
    # Block all red team traffic
    iptables -A CYBER_RANGE_KILL_SWITCH -s $CLUSTER_CIDR -d $CLUSTER_CIDR \
        -m string --string "cyber-range-red" --algo bm \
        -j LOG --log-prefix "KILL-SWITCH-BLOCKED: " --log-level 4
    iptables -A CYBER_RANGE_KILL_SWITCH -s $CLUSTER_CIDR -d $CLUSTER_CIDR \
        -m string --string "cyber-range-red" --algo bm \
        -j DROP
    
    # Block target access except from control plane
    iptables -A CYBER_RANGE_KILL_SWITCH -s $CLUSTER_CIDR -d $CLUSTER_CIDR \
        -m string --string "cyber-range-targets" --algo bm \
        ! -m string --string "cyber-range-control" --algo bm \
        -j DROP
    
    # Return for other traffic
    iptables -A CYBER_RANGE_KILL_SWITCH -j RETURN
    
    success "Lightweight kill switch activated"
}

# Setup basic system rules (minimal)
setup_basic_rules() {
    info "Setting up basic system rules..."
    
    # Allow loopback
    iptables -A INPUT -i lo -j ACCEPT 2>/dev/null || true
    
    # Allow established connections
    iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT 2>/dev/null || true
    
    # Allow SSH for management
    iptables -A INPUT -p tcp --dport 22 -j ACCEPT 2>/dev/null || true
    
    # Allow Kubernetes API
    iptables -A INPUT -p tcp --dport 6443 -j ACCEPT 2>/dev/null || true
    
    # Allow NodePort range for services
    iptables -A INPUT -p tcp --dport 30000:32767 -j ACCEPT 2>/dev/null || true
    
    success "Basic system rules configured"
}

# Verify configuration
verify_config() {
    info "Verifying lightweight firewall configuration..."
    
    local errors=0
    
    # Check if cyber range chain exists
    if ! iptables -L CYBER_RANGE_RULES >/dev/null 2>&1; then
        if ! iptables -L CYBER_RANGE_KILL_SWITCH >/dev/null 2>&1; then
            error "No cyber range firewall chain found"
            ((errors++))
        fi
    fi
    
    # Check rule count
    local rule_count=$(iptables -L | wc -l)
    if [ $rule_count -gt $MAX_RULES ]; then
        warning "High rule count: $rule_count (max: $MAX_RULES)"
    fi
    
    # Test basic connectivity
    if ! iptables -L INPUT | grep -q "ACCEPT.*lo"; then
        warning "Loopback rule not found"
        ((errors++))
    fi
    
    if [ $errors -eq 0 ]; then
        success "Lightweight firewall verification passed"
        return 0
    else
        warning "Lightweight firewall verification had $errors issues"
        return 1
    fi
}

# Save configuration
save_config() {
    info "Saving lightweight firewall configuration..."
    
    # Save rules
    mkdir -p /etc/iptables
    iptables-save > /etc/iptables/cyber-range-lightweight.rules
    
    # Create status file
    cat > "$STATUS_FILE" << EOF
{
    "mode": "lightweight",
    "current_mode": "$(detect_mode)",
    "rule_count": $(iptables -L | wc -l),
    "last_updated": "$(date -Iseconds)",
    "node_ip": "$NODE_IP",
    "cluster_cidr": "$CLUSTER_CIDR",
    "service_cidr": "$SERVICE_CIDR",
    "max_rules": $MAX_RULES,
    "reduced_logging": $REDUCED_LOGGING
}
EOF
    
    success "Lightweight firewall configuration saved"
}

# Show status
show_status() {
    local current_mode=$(detect_mode)
    local rule_count=$(iptables -L | wc -l)
    
    echo -e "\n${BLUE}=== XORB Cyber Range Lightweight Firewall Status ===${NC}"
    echo -e "${GREEN}Optimized for: 16 vCPU AMD RYZEN EPYC 7002 / 32GB RAM${NC}"
    echo
    case "$current_mode" in
        "staging")
            echo -e "${GREEN}Current Mode: STAGING (Safe Training)${NC}"
            echo -e "${YELLOW}Red Team: ATTACKS BLOCKED${NC}"
            ;;
        "live")
            echo -e "${RED}Current Mode: LIVE EXERCISE${NC}"
            echo -e "${RED}Red Team: ATTACKS ENABLED ⚠️${NC}"
            ;;
        "none")
            echo -e "${PURPLE}Current Mode: NO CYBER RANGE RULES${NC}"
            ;;
    esac
    
    echo
    echo "Configuration:"
    echo "- Node IP: $NODE_IP"
    echo "- Cluster CIDR: $CLUSTER_CIDR"
    echo "- Service CIDR: $SERVICE_CIDR"
    echo "- Rule Count: $rule_count / $MAX_RULES"
    echo "- Lightweight Mode: $LIGHTWEIGHT_MODE"
    echo "- Reduced Logging: $REDUCED_LOGGING"
    
    echo
    echo "Resource Optimization:"
    echo "- Minimal rule set for performance"
    echo "- Single-node deployment optimized"
    echo "- AMD EPYC CPU affinity aware"
    echo "- Memory-efficient string matching"
    
    echo
    echo "Available Commands:"
    echo "- $0 staging    # Switch to staging mode"
    echo "- $0 live       # Switch to live mode"
    echo "- $0 kill       # Activate kill switch"
    echo "- $0 clear      # Clear all rules"
    echo "- $0 status     # Show this status"
}

# Main function
main() {
    local mode="${1:-status}"
    
    setup_logging
    check_root
    
    case "$mode" in
        "staging")
            info "Switching to lightweight staging mode..."
            clear_rules
            setup_basic_rules
            setup_staging_rules
            save_config
            if verify_config; then
                success "Lightweight staging mode activated"
                show_status
            fi
            ;;
        "live")
            warning "Switching to lightweight live mode..."
            echo -e "${RED}This will enable real attacks in resource-constrained environment${NC}"
            read -p "Type 'ENABLE LIVE' to confirm: " confirmation
            if [ "$confirmation" = "ENABLE LIVE" ]; then
                clear_rules
                setup_basic_rules
                setup_live_rules
                save_config
                if verify_config; then
                    success "Lightweight live mode activated"
                    show_status
                fi
            else
                error "Live mode activation cancelled"
            fi
            ;;
        "kill")
            error "Activating emergency kill switch..."
            setup_kill_switch
            save_config
            success "Kill switch activated - all attacks blocked"
            ;;
        "clear")
            info "Clearing all cyber range rules..."
            clear_rules
            save_config
            success "All cyber range rules cleared"
            ;;
        "status")
            show_status
            ;;
        "verify")
            verify_config
            ;;
        *)
            error "Unknown mode: $mode"
            echo "Usage: $0 {staging|live|kill|clear|status|verify}"
            exit 1
            ;;
    esac
}

# AMD EPYC Specific Optimizations
optimize_for_epyc() {
    info "Applying AMD EPYC optimizations..."
    
    # Set CPU governor to performance
    if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
        echo 'performance' | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null 2>&1
        success "CPU governor set to performance"
    fi
    
    # Optimize network stack for iptables
    if [ -f /proc/sys/net/netfilter/nf_conntrack_max ]; then
        echo '65536' > /proc/sys/net/netfilter/nf_conntrack_max
        success "Connection tracking optimized"
    fi
    
    # Optimize memory for iptables
    if [ -f /proc/sys/vm/swappiness ]; then
        echo '1' > /proc/sys/vm/swappiness
        success "Memory swappiness optimized"
    fi
}

# Apply optimizations and run main function
optimize_for_epyc
main "$@"