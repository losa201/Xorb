#!/bin/bash
# XORB PTaaS Cyber Range - iptables Configuration for LIVE Mode
# This script configures firewall rules for active red vs blue exercises
# Red team attacks are ALLOWED to targets, comprehensive monitoring enabled

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging
LOGFILE="/var/log/cyber-range/firewall-live.log"
mkdir -p "$(dirname "$LOGFILE")"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [LIVE] $*" | tee -a "$LOGFILE"
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

critical() {
    echo -e "${PURPLE}CRITICAL: $*${NC}" | tee -a "$LOGFILE"
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

# Exercise parameters
MAX_EXERCISE_DURATION=${MAX_EXERCISE_DURATION:-28800}  # 8 hours in seconds
EXERCISE_START_TIME=$(date +%s)
EXERCISE_END_TIME=$((EXERCISE_START_TIME + MAX_EXERCISE_DURATION))

# Confirmation for live mode
confirm_live_mode() {
    echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                    ⚠️  DANGER ZONE ⚠️                        ║${NC}"
    echo -e "${RED}║                                                              ║${NC}"
    echo -e "${RED}║  You are about to enable LIVE CYBER RANGE MODE              ║${NC}"
    echo -e "${RED}║                                                              ║${NC}"
    echo -e "${RED}║  This will allow REAL ATTACKS between red and blue teams    ║${NC}"
    echo -e "${RED}║  Network traffic will be monitored and logged               ║${NC}"
    echo -e "${RED}║  Exercise will auto-terminate in $(($MAX_EXERCISE_DURATION/3600)) hours                ║${NC}"
    echo -e "${RED}║                                                              ║${NC}"
    echo -e "${RED}║  Emergency kill switch: ./kill-switch.sh activate          ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo
    
    if [[ "${FORCE_LIVE_MODE:-}" != "true" ]]; then
        read -p "Type 'ENABLE LIVE MODE' to confirm: " confirmation
        if [[ "$confirmation" != "ENABLE LIVE MODE" ]]; then
            error "Live mode activation cancelled"
        fi
    fi
    
    critical "LIVE MODE CONFIRMED - Starting in 10 seconds..."
    critical "Press Ctrl+C now to abort"
    sleep 10
}

# Backup existing rules
backup_iptables() {
    info "Backing up existing iptables rules..."
    iptables-save > "/var/log/cyber-range/iptables-backup-live-$(date +%Y%m%d-%H%M%S).rules"
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
    info "Setting default policies for LIVE mode..."
    
    # Default policies - restrictive but allowing cyber range traffic
    iptables -P INPUT DROP
    iptables -P FORWARD DROP
    iptables -P OUTPUT ACCEPT
    
    success "Default policies set for live exercise"
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
    
    # Allow SSH for emergency management
    iptables -A INPUT -p tcp --dport 22 -s $CONTROL_PLANE_NETWORK -j ACCEPT
    iptables -A INPUT -p tcp --dport 22 -s $BLUE_TEAM_NETWORK -j ACCEPT
    
    # Allow ICMP for network diagnostics (limited)
    iptables -A INPUT -p icmp --icmp-type echo-request -m limit --limit 5/sec -j ACCEPT
    iptables -A FORWARD -p icmp --icmp-type echo-request -m limit --limit 10/sec -j ACCEPT
    
    success "Basic system rules configured for live exercise"
}

# Control plane rules
setup_control_plane_rules() {
    info "Setting up control plane rules for live exercise..."
    
    # Allow control plane to manage all networks
    iptables -A FORWARD -s $CONTROL_PLANE_NETWORK -j ACCEPT
    iptables -A FORWARD -d $CONTROL_PLANE_NETWORK -j ACCEPT
    
    # XORB Orchestrator access
    iptables -A INPUT -p tcp --dport 8080 -s $VPC_CIDR -j ACCEPT
    
    # Kill switch emergency access (highest priority)
    iptables -I INPUT 1 -p tcp --dport 8081 -j ACCEPT
    
    # Monitoring and metrics
    iptables -A INPUT -p tcp --dport 9090 -s $VPC_CIDR -j ACCEPT  # Prometheus
    iptables -A INPUT -p tcp --dport 3000 -s $VPC_CIDR -j ACCEPT  # Grafana
    
    # Exercise management
    iptables -A INPUT -p tcp --dport 8082 -s $CONTROL_PLANE_NETWORK -j ACCEPT  # Exercise controller
    
    success "Control plane rules configured for live exercise"
}

# Red team rules (LIVE MODE - ATTACKS ALLOWED)
setup_red_team_rules_live() {
    critical "Setting up red team rules for LIVE mode (ATTACKS ENABLED)..."
    
    # LOG and ALLOW red team attacks to targets (with rate limiting)
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $TARGET_WEB_NETWORK \
        -m limit --limit 100/min --limit-burst 200 \
        -j LOG --log-prefix "CYBER-RANGE-LIVE: RED->WEB ATTACK: " --log-level 4
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $TARGET_WEB_NETWORK \
        -m limit --limit 500/min --limit-burst 1000 -j ACCEPT
    
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $TARGET_INTERNAL_NETWORK \
        -m limit --limit 100/min --limit-burst 200 \
        -j LOG --log-prefix "CYBER-RANGE-LIVE: RED->INT ATTACK: " --log-level 4
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $TARGET_INTERNAL_NETWORK \
        -m limit --limit 500/min --limit-burst 1000 -j ACCEPT
    
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $TARGET_OT_NETWORK \
        -m limit --limit 50/min --limit-burst 100 \
        -j LOG --log-prefix "CYBER-RANGE-LIVE: RED->OT ATTACK: " --log-level 4
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $TARGET_OT_NETWORK \
        -m limit --limit 200/min --limit-burst 500 -j ACCEPT
    
    # Still block red team access to blue team and control plane
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $BLUE_TEAM_NETWORK \
        -j LOG --log-prefix "CYBER-RANGE-LIVE: RED->BLUE BLOCKED: "
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $BLUE_TEAM_NETWORK -j DROP
    
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $CONTROL_PLANE_NETWORK \
        -j LOG --log-prefix "CYBER-RANGE-LIVE: RED->CONTROL BLOCKED: "
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $CONTROL_PLANE_NETWORK -j DROP
    
    # Allow red team internal communication
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $RED_TEAM_NETWORK -j ACCEPT
    
    # Limited internet access for red team (for C2, updates, etc.)
    iptables -A FORWARD -s $RED_TEAM_NETWORK -o eth0 -p tcp --dport 80 -j ACCEPT
    iptables -A FORWARD -s $RED_TEAM_NETWORK -o eth0 -p tcp --dport 443 -j ACCEPT
    iptables -A FORWARD -s $RED_TEAM_NETWORK -o eth0 -p tcp --dport 53 -j ACCEPT
    iptables -A FORWARD -s $RED_TEAM_NETWORK -o eth0 -p udp --dport 53 -j ACCEPT
    
    # Special rules for common attack vectors
    setup_attack_vector_rules
    
    critical "Red team attacks to targets are NOW ENABLED"
    success "Red team live rules configured"
}

# Attack vector specific rules
setup_attack_vector_rules() {
    info "Setting up attack vector specific rules..."
    
    # Web application attacks
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $TARGET_WEB_NETWORK -p tcp --dport 80 \
        -m string --string "' OR '1'='1" --algo bm -j LOG --log-prefix "SQL_INJECTION_ATTEMPT: "
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $TARGET_WEB_NETWORK -p tcp --dport 80 \
        -m string --string "<script>" --algo bm -j LOG --log-prefix "XSS_ATTEMPT: "
    
    # Common exploit ports
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $TARGET_INTERNAL_NETWORK -p tcp --dport 445 \
        -j LOG --log-prefix "SMB_ATTACK: "  # SMB attacks
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $TARGET_INTERNAL_NETWORK -p tcp --dport 3389 \
        -j LOG --log-prefix "RDP_ATTACK: "  # RDP attacks
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $TARGET_INTERNAL_NETWORK -p tcp --dport 22 \
        -j LOG --log-prefix "SSH_ATTACK: "  # SSH attacks
    
    # Database attacks
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $TARGET_INTERNAL_NETWORK -p tcp --dport 1433 \
        -j LOG --log-prefix "MSSQL_ATTACK: "
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $TARGET_INTERNAL_NETWORK -p tcp --dport 3306 \
        -j LOG --log-prefix "MYSQL_ATTACK: "
    iptables -A FORWARD -s $RED_TEAM_NETWORK -d $TARGET_INTERNAL_NETWORK -p tcp --dport 5432 \
        -j LOG --log-prefix "POSTGRESQL_ATTACK: "
    
    # Reverse shell detection (common ports)
    iptables -A FORWARD -s $TARGET_WEB_NETWORK -d $RED_TEAM_NETWORK -p tcp --dport 4444 \
        -j LOG --log-prefix "REVERSE_SHELL_4444: "
    iptables -A FORWARD -s $TARGET_INTERNAL_NETWORK -d $RED_TEAM_NETWORK -p tcp --dport 4444 \
        -j LOG --log-prefix "REVERSE_SHELL_4444: "
    iptables -A FORWARD -s $TARGET_WEB_NETWORK -d $RED_TEAM_NETWORK -p tcp --dport 1234 \
        -j LOG --log-prefix "REVERSE_SHELL_1234: "
    
    success "Attack vector specific rules configured"
}

# Blue team rules (enhanced monitoring for live exercise)
setup_blue_team_rules() {
    info "Setting up blue team monitoring rules for live exercise..."
    
    # Allow blue team to monitor all target networks (enhanced)
    iptables -A FORWARD -s $BLUE_TEAM_NETWORK -d $TARGET_WEB_NETWORK -j ACCEPT
    iptables -A FORWARD -s $BLUE_TEAM_NETWORK -d $TARGET_INTERNAL_NETWORK -j ACCEPT
    iptables -A FORWARD -s $BLUE_TEAM_NETWORK -d $TARGET_OT_NETWORK -j ACCEPT
    iptables -A FORWARD -s $BLUE_TEAM_NETWORK -d $SIMULATION_NETWORK -j ACCEPT
    
    # Allow targets to send all logs/data to blue team SIEM
    iptables -A FORWARD -s $TARGET_WEB_NETWORK -d $BLUE_TEAM_NETWORK -j ACCEPT
    iptables -A FORWARD -s $TARGET_INTERNAL_NETWORK -d $BLUE_TEAM_NETWORK -j ACCEPT
    iptables -A FORWARD -s $TARGET_OT_NETWORK -d $BLUE_TEAM_NETWORK -j ACCEPT
    
    # Allow blue team access to control plane for real-time reporting
    iptables -A FORWARD -s $BLUE_TEAM_NETWORK -d $CONTROL_PLANE_NETWORK -j ACCEPT
    
    # Block blue team from directly accessing red team (prevent cheating)
    iptables -A FORWARD -s $BLUE_TEAM_NETWORK -d $RED_TEAM_NETWORK \
        -j LOG --log-prefix "BLUE_TEAM_CHEATING_ATTEMPT: "
    iptables -A FORWARD -s $BLUE_TEAM_NETWORK -d $RED_TEAM_NETWORK -j DROP
    
    # Allow blue team internet access for threat intel feeds
    iptables -A FORWARD -s $BLUE_TEAM_NETWORK -o eth0 -j ACCEPT
    
    success "Blue team monitoring rules configured for live exercise"
}

# Target environment rules (live exercise)
setup_target_rules() {
    info "Setting up target environment rules for live exercise..."
    
    # Allow inter-target communication (lateral movement)
    iptables -A FORWARD -s $TARGET_WEB_NETWORK -d $TARGET_INTERNAL_NETWORK -j ACCEPT
    iptables -A FORWARD -s $TARGET_INTERNAL_NETWORK -d $TARGET_WEB_NETWORK -j ACCEPT
    iptables -A FORWARD -s $TARGET_INTERNAL_NETWORK -d $TARGET_OT_NETWORK -j ACCEPT
    
    # Log lateral movement attempts
    iptables -A FORWARD -s $TARGET_WEB_NETWORK -d $TARGET_INTERNAL_NETWORK \
        -m limit --limit 10/min -j LOG --log-prefix "LATERAL_MOVEMENT_WEB->INT: "
    iptables -A FORWARD -s $TARGET_INTERNAL_NETWORK -d $TARGET_OT_NETWORK \
        -m limit --limit 5/min -j LOG --log-prefix "LATERAL_MOVEMENT_INT->OT: "
    
    # Allow simulation traffic to targets
    iptables -A FORWARD -s $SIMULATION_NETWORK -d $TARGET_WEB_NETWORK -j ACCEPT
    iptables -A FORWARD -s $SIMULATION_NETWORK -d $TARGET_INTERNAL_NETWORK -j ACCEPT
    iptables -A FORWARD -s $SIMULATION_NETWORK -d $TARGET_OT_NETWORK -j ACCEPT
    
    # Controlled internet access for targets (can be exploited)
    iptables -A FORWARD -s $TARGET_WEB_NETWORK -o eth0 -p tcp --dport 80 -j ACCEPT
    iptables -A FORWARD -s $TARGET_WEB_NETWORK -o eth0 -p tcp --dport 443 -j ACCEPT
    iptables -A FORWARD -s $TARGET_INTERNAL_NETWORK -o eth0 -p tcp --dport 80 -m limit --limit 20/min -j ACCEPT
    iptables -A FORWARD -s $TARGET_INTERNAL_NETWORK -o eth0 -p tcp --dport 443 -m limit --limit 20/min -j ACCEPT
    
    # Log potential data exfiltration
    iptables -A FORWARD -s $TARGET_WEB_NETWORK -o eth0 -p tcp \
        -m length --length 1000: -m limit --limit 5/min \
        -j LOG --log-prefix "POTENTIAL_EXFILTRATION_WEB: "
    iptables -A FORWARD -s $TARGET_INTERNAL_NETWORK -o eth0 -p tcp \
        -m length --length 1000: -m limit --limit 5/min \
        -j LOG --log-prefix "POTENTIAL_EXFILTRATION_INT: "
    
    success "Target environment rules configured for live exercise"
}

# Enhanced simulation network rules
setup_simulation_rules() {
    info "Setting up enhanced simulation network rules..."
    
    # Allow simulation to generate realistic background traffic
    iptables -A FORWARD -s $SIMULATION_NETWORK -j ACCEPT
    
    # Allow control plane to manage simulation
    iptables -A FORWARD -s $CONTROL_PLANE_NETWORK -d $SIMULATION_NETWORK -j ACCEPT
    
    # Log simulation traffic for analysis
    iptables -A FORWARD -s $SIMULATION_NETWORK -m limit --limit 20/min \
        -j LOG --log-prefix "SIMULATION_TRAFFIC: "
    
    success "Enhanced simulation network rules configured"
}

# DNS rules with monitoring
setup_dns_rules() {
    info "Setting up DNS rules with monitoring..."
    
    # Allow DNS queries from all networks
    iptables -A FORWARD -p udp --dport 53 -j ACCEPT
    iptables -A FORWARD -p tcp --dport 53 -j ACCEPT
    iptables -A INPUT -p udp --dport 53 -j ACCEPT
    iptables -A INPUT -p tcp --dport 53 -j ACCEPT
    
    # Log suspicious DNS queries (potential C2 communication)
    iptables -A FORWARD -p udp --dport 53 -m string --string "bit.ly" --algo bm \
        -j LOG --log-prefix "SUSPICIOUS_DNS_BITLY: "
    iptables -A FORWARD -p udp --dport 53 -m string --string "pastebin" --algo bm \
        -j LOG --log-prefix "SUSPICIOUS_DNS_PASTEBIN: "
    iptables -A FORWARD -p udp --dport 53 -m length --length 100: \
        -j LOG --log-prefix "LONG_DNS_QUERY: "
    
    success "DNS rules with monitoring configured"
}

# Kubernetes specific rules for live exercise
setup_kubernetes_rules() {
    info "Setting up Kubernetes-specific rules for live exercise..."
    
    # Allow Kubernetes API server access (monitored)
    iptables -A INPUT -p tcp --dport 6443 -s $VPC_CIDR -j ACCEPT
    
    # Allow kubelet communication
    iptables -A INPUT -p tcp --dport 10250 -s $VPC_CIDR -j ACCEPT
    
    # Allow NodePort services
    iptables -A INPUT -p tcp --dport 30000:32767 -s $VPC_CIDR -j ACCEPT
    
    # Allow pod-to-pod communication within nodes
    iptables -A FORWARD -s 192.168.0.0/16 -d 192.168.0.0/16 -j ACCEPT  # Pod CIDR
    
    # Log Kubernetes API access from red team (potential container escape)
    iptables -A INPUT -p tcp --dport 6443 -s $RED_TEAM_NETWORK \
        -j LOG --log-prefix "RED_TEAM_K8S_API_ACCESS: "
    
    success "Kubernetes rules for live exercise configured"
}

# Emergency kill switch rules (enhanced for live mode)
setup_kill_switch_rules() {
    critical "Setting up enhanced emergency kill switch rules..."
    
    # Create custom chain for kill switch
    iptables -N CYBER_RANGE_KILL_SWITCH 2>/dev/null || true
    iptables -N CYBER_RANGE_LIVE_MONITOR 2>/dev/null || true
    
    # Jump to kill switch chain for all red team traffic (highest priority)
    iptables -I FORWARD 1 -s $RED_TEAM_NETWORK -j CYBER_RANGE_KILL_SWITCH
    
    # Jump to live monitor chain for all traffic
    iptables -I FORWARD 2 -j CYBER_RANGE_LIVE_MONITOR
    
    # Kill switch chain initially allows traffic (will be modified by kill switch)
    iptables -A CYBER_RANGE_KILL_SWITCH -j RETURN
    
    # Live monitor chain for real-time analysis
    iptables -A CYBER_RANGE_LIVE_MONITOR -j RETURN
    
    # Emergency termination rule (can be activated remotely)
    iptables -N EMERGENCY_TERMINATE 2>/dev/null || true
    
    success "Enhanced emergency kill switch rules configured"
}

# Exercise time limits
setup_exercise_time_limits() {
    info "Setting up exercise time limits..."
    
    # Create cron job to auto-terminate exercise after max duration
    cat > /etc/cron.d/cyber-range-auto-terminate << EOF
# Auto-terminate cyber range exercise after max duration
$(date -d "@$EXERCISE_END_TIME" '+%M %H %d %m *') root /opt/cyber-range/scripts/kill-switch.sh activate-time-limit 2>&1 | logger -t cyber-range-auto-terminate
EOF
    
    # Set up warning notifications
    WARN_TIME_1H=$((EXERCISE_END_TIME - 3600))  # 1 hour before
    WARN_TIME_30M=$((EXERCISE_END_TIME - 1800))  # 30 minutes before
    WARN_TIME_10M=$((EXERCISE_END_TIME - 600))   # 10 minutes before
    
    echo "$(date -d "@$WARN_TIME_1H" '+%M %H %d %m *') root echo 'WARNING: Cyber range exercise will auto-terminate in 1 hour' | wall" >> /etc/cron.d/cyber-range-auto-terminate
    echo "$(date -d "@$WARN_TIME_30M" '+%M %H %d %m *') root echo 'WARNING: Cyber range exercise will auto-terminate in 30 minutes' | wall" >> /etc/cron.d/cyber-range-auto-terminate
    echo "$(date -d "@$WARN_TIME_10M" '+%M %H %d %m *') root echo 'CRITICAL: Cyber range exercise will auto-terminate in 10 minutes' | wall" >> /etc/cron.d/cyber-range-auto-terminate
    
    success "Exercise time limits configured"
    info "Exercise will auto-terminate at: $(date -d "@$EXERCISE_END_TIME")"
}

# Enhanced logging and monitoring
setup_enhanced_logging() {
    critical "Setting up enhanced logging for live exercise..."
    
    # High-frequency logging for live exercise
    iptables -A INPUT -m limit --limit 20/min --limit-burst 50 \
        -j LOG --log-prefix "CYBER-RANGE-LIVE: INPUT DROP: " --log-level 4
    
    iptables -A FORWARD -m limit --limit 50/min --limit-burst 100 \
        -j LOG --log-prefix "CYBER-RANGE-LIVE: FORWARD DROP: " --log-level 4
    
    # Create real-time log analysis pipe
    mkfifo /var/log/cyber-range/live-analysis-pipe 2>/dev/null || true
    
    # Set up rsyslog for real-time forwarding to blue team SIEM
    cat > /etc/rsyslog.d/99-cyber-range-live.conf << 'EOF'
# Forward cyber range logs to blue team SIEM
:msg, contains, "CYBER-RANGE-LIVE" @@10.30.0.100:5045
:msg, contains, "RED->WEB ATTACK" @@10.30.0.100:5045
:msg, contains, "RED->INT ATTACK" @@10.30.0.100:5045
:msg, contains, "REVERSE_SHELL" @@10.30.0.100:5045
:msg, contains, "LATERAL_MOVEMENT" @@10.30.0.100:5045
:msg, contains, "POTENTIAL_EXFILTRATION" @@10.30.0.100:5045
EOF
    
    systemctl restart rsyslog
    
    success "Enhanced logging configured for live exercise"
}

# Advanced rate limiting for live exercise
setup_advanced_rate_limiting() {
    info "Setting up advanced rate limiting for live exercise..."
    
    # Protect against DoS while allowing realistic attack traffic
    iptables -I INPUT -p tcp --dport 22 -m state --state NEW -m recent --set
    iptables -I INPUT -p tcp --dport 22 -m state --state NEW -m recent --update --seconds 60 --hitcount 10 -j DROP
    
    # Rate limit HTTP connections but allow burst attacks
    iptables -I FORWARD -p tcp --dport 80 -m limit --limit 200/minute --limit-burst 500 -j ACCEPT
    iptables -I FORWARD -p tcp --dport 443 -m limit --limit 200/minute --limit-burst 500 -j ACCEPT
    
    # Protect blue team infrastructure from overwhelming
    iptables -I FORWARD -d $BLUE_TEAM_NETWORK -m limit --limit 1000/minute --limit-burst 2000 -j ACCEPT
    iptables -I FORWARD -d $BLUE_TEAM_NETWORK -j DROP
    
    # Allow burst attacks but prevent sustained DoS
    iptables -I FORWARD -s $RED_TEAM_NETWORK -m hashlimit \
        --hashlimit-mode srcip --hashlimit 10/sec \
        --hashlimit-burst 100 --hashlimit-name red_team_attack \
        --hashlimit-htable-expire 300000 -j ACCEPT
    
    success "Advanced rate limiting configured"
}

# Malware detection rules
setup_malware_detection() {
    info "Setting up malware detection rules..."
    
    # Detect common malware signatures
    iptables -A FORWARD -m string --string "powershell -enc" --algo bm \
        -j LOG --log-prefix "POWERSHELL_ENCODED: "
    iptables -A FORWARD -m string --string "cmd.exe /c" --algo bm \
        -j LOG --log-prefix "CMD_EXECUTION: "
    iptables -A FORWARD -m string --string "bash -i" --algo bm \
        -j LOG --log-prefix "BASH_INTERACTIVE: "
    
    # Detect potential ransomware
    iptables -A FORWARD -m string --string "YOUR FILES HAVE BEEN ENCRYPTED" --algo bm \
        -j LOG --log-prefix "RANSOMWARE_DETECTED: "
    
    # Detect potential cryptocurrency mining
    iptables -A FORWARD -p tcp --dport 8333 \
        -j LOG --log-prefix "BITCOIN_MINING: "
    iptables -A FORWARD -p tcp --dport 4444 -m string --string "stratum" --algo bm \
        -j LOG --log-prefix "MINING_STRATUM: "
    
    success "Malware detection rules configured"
}

# Save rules for live exercise
save_iptables() {
    critical "Saving iptables rules for live exercise..."
    
    # Create systemd service for persistence
    cat > /etc/systemd/system/cyber-range-iptables-live.service << 'EOF'
[Unit]
Description=XORB Cyber Range iptables rules (LIVE MODE)
After=network.target

[Service]
Type=oneshot
ExecStart=/sbin/iptables-restore /etc/iptables/cyber-range-live.rules
ExecReload=/sbin/iptables-restore /etc/iptables/cyber-range-live.rules
ExecStop=/opt/cyber-range/scripts/kill-switch.sh activate
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

    # Save current rules
    mkdir -p /etc/iptables
    iptables-save > /etc/iptables/cyber-range-live.rules
    
    # Enable service
    systemctl daemon-reload
    systemctl enable cyber-range-iptables-live
    systemctl disable cyber-range-iptables 2>/dev/null || true  # Disable staging service
    
    # Create exercise metadata
    cat > /var/log/cyber-range/live-exercise-metadata.json << EOF
{
    "exercise_id": "$(uuidgen)",
    "start_time": "$EXERCISE_START_TIME",
    "end_time": "$EXERCISE_END_TIME",
    "max_duration_seconds": $MAX_EXERCISE_DURATION,
    "mode": "LIVE",
    "red_team_network": "$RED_TEAM_NETWORK",
    "blue_team_network": "$BLUE_TEAM_NETWORK",
    "target_networks": [
        "$TARGET_WEB_NETWORK",
        "$TARGET_INTERNAL_NETWORK", 
        "$TARGET_OT_NETWORK"
    ],
    "kill_switch_enabled": true,
    "auto_terminate": true,
    "logging_level": "enhanced"
}
EOF
    
    success "iptables rules saved and live exercise service enabled"
}

# Comprehensive verification for live mode
verify_live_config() {
    info "Verifying live exercise firewall configuration..."
    
    local errors=0
    local warnings=0
    
    # Check if rules are applied
    if ! iptables -L | grep -q "CYBER_RANGE_KILL_SWITCH"; then
        error "Kill switch chain not found"
        ((errors++))
    fi
    
    if ! iptables -L | grep -q "CYBER_RANGE_LIVE_MONITOR"; then
        error "Live monitor chain not found"
        ((errors++))
    fi
    
    # Check default policies
    if ! iptables -L | grep -q "policy DROP"; then
        warning "Default policies may not be properly set"
        ((warnings++))
    fi
    
    # Test key rules for live mode
    info "Testing key firewall rules for live exercise..."
    
    # Check if red team can attack targets
    if iptables -L FORWARD | grep -q "ACCEPT.*10\.20\.0\.0/16.*10\.100\.0\.0/24"; then
        critical "Red team -> Web targets: ATTACKS ENABLED ⚠️"
    else
        error "Red team -> Web targets: Attack rules not found"
        ((errors++))
    fi
    
    if iptables -L FORWARD | grep -q "ACCEPT.*10\.30\.0\.0/24.*10\.100\.0\.0/24"; then
        success "Blue team -> Web targets: MONITORING ENABLED ✓"
    else
        warning "Blue team -> Web targets: Monitoring rule not found"
        ((warnings++))
    fi
    
    # Check if red team is still blocked from blue team
    if iptables -L FORWARD | grep -q "DROP.*10\.20\.0\.0/16.*10\.30\.0\.0/24"; then
        success "Red team -> Blue team: BLOCKED ✓"
    else
        error "Red team -> Blue team: Should be blocked"
        ((errors++))
    fi
    
    # Check exercise time limits
    if [ -f /etc/cron.d/cyber-range-auto-terminate ]; then
        success "Exercise auto-termination: CONFIGURED ✓"
    else
        warning "Exercise auto-termination: Not configured"
        ((warnings++))
    fi
    
    # Check logging configuration
    if [ -f /etc/rsyslog.d/99-cyber-range-live.conf ]; then
        success "Enhanced logging: CONFIGURED ✓"
    else
        warning "Enhanced logging: Not configured"
        ((warnings++))
    fi
    
    if [ $errors -eq 0 ]; then
        if [ $warnings -eq 0 ]; then
            success "Live exercise firewall verification passed completely"
        else
            warning "Live exercise firewall verification passed with $warnings warnings"
        fi
        return 0
    else
        error "Live exercise firewall configuration has $errors critical issues"
        return 1
    fi
}

# Display live exercise status
show_live_status() {
    echo -e "\n${RED}=== XORB Cyber Range Firewall Status (LIVE EXERCISE MODE) ===${NC}"
    echo -e "${RED}⚠️  LIVE EXERCISE IN PROGRESS ⚠️${NC}"
    echo
    echo -e "${RED}Mode: LIVE EXERCISE (Red team attacks ENABLED)${NC}"
    echo -e "${YELLOW}Red Team Network: $RED_TEAM_NETWORK (ATTACKS ACTIVE)${NC}"
    echo -e "${BLUE}Blue Team Network: $BLUE_TEAM_NETWORK (MONITORING ACTIVE)${NC}"
    echo -e "${GREEN}Control Plane: $CONTROL_PLANE_NETWORK (MANAGEMENT ACTIVE)${NC}"
    echo
    echo -e "${PURPLE}Exercise Information:${NC}"
    echo "- Exercise Start: $(date -d "@$EXERCISE_START_TIME")"
    echo "- Exercise End: $(date -d "@$EXERCISE_END_TIME")"
    echo "- Remaining Time: $(( (EXERCISE_END_TIME - $(date +%s)) / 60 )) minutes"
    echo "- Max Duration: $(($MAX_EXERCISE_DURATION / 3600)) hours"
    echo
    echo -e "${RED}Active Security Rules:${NC}"
    echo "- Red team attacks to targets: ENABLED ⚠️"
    echo "- Blue team monitoring: ENHANCED ✓"
    echo "- Control plane management: ACTIVE ✓"
    echo "- Emergency kill switch: ARMED ✓"
    echo "- Enhanced logging: ACTIVE ✓"
    echo "- Auto-termination: SCHEDULED ✓"
    echo "- Malware detection: ACTIVE ✓"
    echo "- Rate limiting: ADVANCED ✓"
    echo
    echo "Rule counts:"
    echo "- INPUT rules: $(iptables -L INPUT --line-numbers | wc -l)"
    echo "- FORWARD rules: $(iptables -L FORWARD --line-numbers | wc -l)"
    echo "- OUTPUT rules: $(iptables -L OUTPUT --line-numbers | wc -l)"
    echo
    echo -e "${RED}Emergency Controls:${NC}"
    echo -e "${RED}- Immediate kill switch: ./kill-switch.sh activate${NC}"
    echo -e "${YELLOW}- Return to staging: ./iptables-staging.sh${NC}"
    echo -e "${BLUE}- View live logs: tail -f /var/log/cyber-range/firewall-live.log${NC}"
}

# Main execution for live mode
main() {
    critical "Starting XORB Cyber Range firewall configuration for LIVE EXERCISE mode..."
    
    confirm_live_mode
    backup_iptables
    clear_iptables
    set_default_policies
    setup_basic_rules
    setup_control_plane_rules
    setup_red_team_rules_live  # Red team attacks ENABLED
    setup_blue_team_rules
    setup_target_rules
    setup_simulation_rules
    setup_dns_rules
    setup_kubernetes_rules
    setup_kill_switch_rules
    setup_exercise_time_limits
    setup_enhanced_logging
    setup_advanced_rate_limiting
    setup_malware_detection
    save_iptables
    
    if verify_live_config; then
        show_live_status
        critical "XORB Cyber Range firewall configured for LIVE EXERCISE mode"
        critical "RED TEAM ATTACKS ARE NOW ENABLED - Live exercise in progress"
        critical "Exercise will auto-terminate at: $(date -d "@$EXERCISE_END_TIME")"
    else
        error "Live exercise firewall configuration failed verification"
    fi
}

# Handle script arguments
case "${1:-}" in
    "verify")
        verify_live_config
        ;;
    "status")
        show_live_status
        ;;
    "force")
        FORCE_LIVE_MODE=true
        main
        ;;
    *)
        main
        ;;
esac