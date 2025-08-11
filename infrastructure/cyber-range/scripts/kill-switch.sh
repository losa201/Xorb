#!/bin/bash
# XORB PTaaS Cyber Range - Emergency Kill Switch
# Provides immediate shutdown and isolation capabilities for cyber range exercises

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
LOGFILE="/var/log/cyber-range/kill-switch.log"
LOCKFILE="/var/run/cyber-range-kill-switch.lock"
STATUS_FILE="/var/log/cyber-range/kill-switch-status.json"
BACKUP_DIR="/var/log/cyber-range/backups"

# Kill switch activation reasons
KILL_REASONS=(
    "manual_activation"
    "time_limit_exceeded"
    "security_breach"
    "malware_detected"
    "unauthorized_access"
    "system_compromise"
    "emergency_stop"
    "exercise_complete"
)

# Logging functions
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [KILL-SWITCH] $*" | tee -a "$LOGFILE"
}

error() {
    echo -e "${RED}[CRITICAL ERROR] $*${NC}" | tee -a "$LOGFILE"
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

critical() {
    echo -e "${PURPLE}[CRITICAL] $*${NC}" | tee -a "$LOGFILE"
}

emergency() {
    echo -e "${CYAN}[EMERGENCY] $*${NC}" | tee -a "$LOGFILE"
}

# Setup logging directory
setup_logging() {
    mkdir -p "$(dirname "$LOGFILE")"
    mkdir -p "$BACKUP_DIR"
}

# Check if script is running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "Kill switch must be run as root"
        exit 1
    fi
}

# Create lock file to prevent concurrent executions
acquire_lock() {
    if [ -f "$LOCKFILE" ]; then
        local lock_pid=$(cat "$LOCKFILE" 2>/dev/null || echo "unknown")
        if kill -0 "$lock_pid" 2>/dev/null; then
            error "Kill switch is already running (PID: $lock_pid)"
            exit 1
        else
            warning "Removing stale lock file"
            rm -f "$LOCKFILE"
        fi
    fi
    echo $$ > "$LOCKFILE"
}

# Release lock file
release_lock() {
    rm -f "$LOCKFILE"
}

# Trap to ensure cleanup
cleanup() {
    release_lock
}
trap cleanup EXIT

# Display kill switch banner
show_banner() {
    echo -e "${RED}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║                                                               ║"
    echo "║    🚨 XORB PTaaS CYBER RANGE EMERGENCY KILL SWITCH 🚨        ║"
    echo "║                                                               ║"
    echo "║               ⚠️  IMMEDIATE SHUTDOWN SYSTEM ⚠️                ║"
    echo "║                                                               ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Backup current state before kill switch activation
backup_current_state() {
    info "Backing up current cyber range state..."
    
    local backup_timestamp=$(date +%Y%m%d-%H%M%S)
    local backup_subdir="$BACKUP_DIR/kill-switch-$backup_timestamp"
    mkdir -p "$backup_subdir"
    
    # Backup iptables rules
    iptables-save > "$backup_subdir/iptables-rules.txt" 2>/dev/null || true
    
    # Backup Kubernetes network policies
    kubectl get networkpolicies --all-namespaces -o yaml > "$backup_subdir/network-policies.yaml" 2>/dev/null || true
    
    # Backup running pods state
    kubectl get pods --all-namespaces -o wide > "$backup_subdir/pods-state.txt" 2>/dev/null || true
    
    # Backup services state
    kubectl get services --all-namespaces > "$backup_subdir/services-state.txt" 2>/dev/null || true
    
    # Backup exercise metadata
    if [ -f "/var/log/cyber-range/live-exercise-metadata.json" ]; then
        cp "/var/log/cyber-range/live-exercise-metadata.json" "$backup_subdir/"
    fi
    
    # Create kill switch metadata
    cat > "$backup_subdir/kill-switch-metadata.json" << EOF
{
    "kill_switch_activated": "$(date -Iseconds)",
    "reason": "${1:-manual_activation}",
    "operator": "$(whoami)",
    "hostname": "$(hostname)",
    "backup_location": "$backup_subdir",
    "exercise_state": "terminated"
}
EOF
    
    success "State backed up to: $backup_subdir"
    echo "$backup_subdir" > /tmp/last-backup-location
}

# Immediate network isolation using iptables
emergency_network_isolation() {
    critical "Activating emergency network isolation..."
    
    # Set all default policies to DROP immediately
    iptables -P INPUT DROP
    iptables -P FORWARD DROP
    iptables -P OUTPUT DROP
    
    # Flush all existing rules
    iptables -F
    iptables -X
    iptables -t nat -F
    iptables -t nat -X
    iptables -t mangle -F
    iptables -t mangle -X
    
    # Allow only loopback and SSH from control plane for emergency management
    iptables -A INPUT -i lo -j ACCEPT
    iptables -A OUTPUT -o lo -j ACCEPT
    
    # Allow established connections to prevent breaking current SSH sessions
    iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
    iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
    
    # Allow SSH from control plane only (emergency access)
    iptables -A INPUT -p tcp --dport 22 -s 10.10.10.0/24 -j ACCEPT
    
    # Allow kill switch control traffic
    iptables -A INPUT -p tcp --dport 8081 -j ACCEPT
    iptables -A OUTPUT -p tcp --sport 8081 -j ACCEPT
    
    # Block ALL other traffic
    iptables -A INPUT -j LOG --log-prefix "KILL-SWITCH: INPUT BLOCKED: "
    iptables -A FORWARD -j LOG --log-prefix "KILL-SWITCH: FORWARD BLOCKED: "
    iptables -A OUTPUT -j LOG --log-prefix "KILL-SWITCH: OUTPUT BLOCKED: "
    
    iptables -A INPUT -j DROP
    iptables -A FORWARD -j DROP
    iptables -A OUTPUT -j DROP
    
    success "Emergency network isolation activated"
}

# Apply Kubernetes kill switch network policies
activate_k8s_kill_switch() {
    critical "Activating Kubernetes kill switch policies..."
    
    # Apply kill switch network policy to red team namespace
    kubectl patch networkpolicy red-team-staging-mode -n cyber-range-red --type='merge' -p='{"spec":{"podSelector":{"matchLabels":{"cyber-range.xorb.io/kill-switch":"active"}}}}' 2>/dev/null || true
    
    kubectl patch networkpolicy red-team-live-mode -n cyber-range-red --type='merge' -p='{"spec":{"podSelector":{"matchLabels":{"cyber-range.xorb.io/kill-switch":"active"}}}}' 2>/dev/null || true
    
    # Label all red team pods to activate kill switch policy
    kubectl label pods --all cyber-range.xorb.io/kill-switch=active -n cyber-range-red 2>/dev/null || true
    
    # Apply emergency isolation policy
    cat << 'EOF' | kubectl apply -f - 2>/dev/null || true
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: emergency-kill-switch
  namespace: cyber-range-red
  labels:
    cyber-range.xorb.io/emergency: active
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress: []
  egress: []
EOF
    
    # Apply kill switch to targets as well (stop all attack traffic)
    cat << 'EOF' | kubectl apply -f - 2>/dev/null || true
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: emergency-kill-switch
  namespace: cyber-range-targets
  labels:
    cyber-range.xorb.io/emergency: active
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          app.kubernetes.io/name: cyber-range-control
    ports:
    - protocol: TCP
      port: 22
  egress:
  - to: []
    ports:
    - protocol: UDP
      port: 53
EOF
    
    success "Kubernetes kill switch policies activated"
}

# Stop all red team containers
stop_red_team_containers() {
    critical "Stopping all red team containers..."
    
    # Scale down all deployments in red team namespace
    kubectl scale deployment --all --replicas=0 -n cyber-range-red 2>/dev/null || true
    
    # Delete all jobs in red team namespace
    kubectl delete jobs --all -n cyber-range-red 2>/dev/null || true
    
    # Delete all pods with force (immediate termination)
    kubectl delete pods --all --force --grace-period=0 -n cyber-range-red 2>/dev/null || true
    
    success "Red team containers stopped"
}

# Stop target environment containers (preserve blue team for analysis)
stop_target_containers() {
    info "Stopping target environment containers..."
    
    # Scale down target deployments
    kubectl scale deployment --all --replicas=0 -n cyber-range-targets 2>/dev/null || true
    
    # Delete target pods with force
    kubectl delete pods --all --force --grace-period=0 -n cyber-range-targets 2>/dev/null || true
    
    success "Target environment containers stopped"
}

# Preserve blue team for post-exercise analysis
preserve_blue_team() {
    info "Preserving blue team infrastructure for post-exercise analysis..."
    
    # Keep blue team running but isolate it
    cat << 'EOF' | kubectl apply -f - 2>/dev/null || true
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: blue-team-analysis-mode
  namespace: cyber-range-blue
  labels:
    cyber-range.xorb.io/mode: post-exercise-analysis
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          app.kubernetes.io/name: cyber-range-control
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          app.kubernetes.io/name: cyber-range-control
  - to: []
    ports:
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 80
    - protocol: TCP
      port: 443
EOF
    
    success "Blue team preserved for analysis"
}

# Send notifications
send_notifications() {
    local reason="${1:-manual_activation}"
    
    critical "Sending kill switch notifications..."
    
    # Log to syslog
    logger -t cyber-range-kill-switch "KILL SWITCH ACTIVATED: Reason=$reason, Time=$(date -Iseconds)"
    
    # Send wall message to all logged in users
    echo "🚨 CYBER RANGE KILL SWITCH ACTIVATED 🚨" | wall
    echo "Reason: $reason" | wall
    echo "Time: $(date)" | wall
    echo "All attack activities have been terminated" | wall
    
    # Write to status file
    cat > "$STATUS_FILE" << EOF
{
    "status": "kill_switch_active",
    "activated_at": "$(date -Iseconds)",
    "reason": "$reason",
    "operator": "$(whoami)",
    "hostname": "$(hostname)",
    "red_team_status": "terminated",
    "blue_team_status": "preserved",
    "targets_status": "terminated",
    "network_status": "isolated",
    "restoration_required": true
}
EOF
    
    # Try to send webhook notification (if configured)
    if [ -n "${WEBHOOK_URL:-}" ]; then
        curl -X POST "$WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"event\":\"kill_switch_activated\",\"reason\":\"$reason\",\"timestamp\":\"$(date -Iseconds)\"}" \
            2>/dev/null || warning "Failed to send webhook notification"
    fi
    
    success "Notifications sent"
}

# Generate incident report
generate_incident_report() {
    local reason="${1:-manual_activation}"
    
    info "Generating incident report..."
    
    local report_file="/var/log/cyber-range/incident-report-$(date +%Y%m%d-%H%M%S).md"
    
    cat > "$report_file" << EOF
# XORB PTaaS Cyber Range Incident Report

## Kill Switch Activation

**Date/Time:** $(date -Iseconds)
**Reason:** $reason
**Operator:** $(whoami)
**Hostname:** $(hostname)

## Actions Taken

1. **Network Isolation:** Emergency iptables rules applied
2. **Container Termination:** Red team and target containers stopped
3. **Policy Enforcement:** Kubernetes network policies activated
4. **Data Preservation:** Blue team infrastructure preserved for analysis
5. **State Backup:** System state backed up to $BACKUP_DIR

## System Status

- **Red Team:** TERMINATED
- **Blue Team:** PRESERVED (analysis mode)
- **Targets:** TERMINATED
- **Network:** ISOLATED
- **Control Plane:** EMERGENCY MANAGEMENT MODE

## Recovery Actions Required

1. Review incident logs and determine root cause
2. Verify system integrity
3. Clean up terminated containers and resources
4. Reset network configurations
5. Restore exercise environment if needed

## Log Locations

- Kill Switch Log: $LOGFILE
- System Backup: $(cat /tmp/last-backup-location 2>/dev/null || echo "Unknown")
- Status File: $STATUS_FILE

## Generated At

$(date)

---
**This report was automatically generated by the XORB PTaaS Kill Switch system**
EOF
    
    success "Incident report generated: $report_file"
    echo "$report_file"
}

# Check current kill switch status
check_status() {
    if [ -f "$STATUS_FILE" ]; then
        local status=$(jq -r '.status' "$STATUS_FILE" 2>/dev/null || echo "unknown")
        if [ "$status" = "kill_switch_active" ]; then
            echo -e "${RED}Kill Switch Status: ACTIVE${NC}"
            echo "Details:"
            cat "$STATUS_FILE" | jq . 2>/dev/null || cat "$STATUS_FILE"
            return 0
        fi
    fi
    
    echo -e "${GREEN}Kill Switch Status: INACTIVE${NC}"
    return 1
}

# Restore system from backup
restore_system() {
    local backup_dir="${1:-}"
    
    if [ -z "$backup_dir" ]; then
        if [ -f "/tmp/last-backup-location" ]; then
            backup_dir=$(cat /tmp/last-backup-location)
        else
            error "No backup directory specified and no last backup found"
            exit 1
        fi
    fi
    
    if [ ! -d "$backup_dir" ]; then
        error "Backup directory not found: $backup_dir"
        exit 1
    fi
    
    warning "Restoring system from backup: $backup_dir"
    read -p "Are you sure? Type 'RESTORE' to confirm: " confirmation
    if [ "$confirmation" != "RESTORE" ]; then
        error "Restoration cancelled"
        exit 1
    fi
    
    info "Restoring iptables rules..."
    if [ -f "$backup_dir/iptables-rules.txt" ]; then
        iptables-restore < "$backup_dir/iptables-rules.txt"
        success "iptables rules restored"
    fi
    
    info "Restoring Kubernetes network policies..."
    if [ -f "$backup_dir/network-policies.yaml" ]; then
        kubectl apply -f "$backup_dir/network-policies.yaml" 2>/dev/null || warning "Some network policies failed to restore"
        success "Network policies restored"
    fi
    
    info "Removing kill switch status..."
    rm -f "$STATUS_FILE"
    
    success "System restoration completed"
}

# Clean up kill switch state
deactivate_kill_switch() {
    info "Deactivating kill switch..."
    
    # Remove kill switch network policies
    kubectl delete networkpolicy emergency-kill-switch -n cyber-range-red 2>/dev/null || true
    kubectl delete networkpolicy emergency-kill-switch -n cyber-range-targets 2>/dev/null || true
    kubectl delete networkpolicy blue-team-analysis-mode -n cyber-range-blue 2>/dev/null || true
    
    # Remove kill switch labels
    kubectl label pods --all cyber-range.xorb.io/kill-switch- -n cyber-range-red 2>/dev/null || true
    
    # Remove status file
    rm -f "$STATUS_FILE"
    
    success "Kill switch deactivated"
}

# Test kill switch functionality
test_kill_switch() {
    info "Testing kill switch functionality..."
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        warning "Test should be run as root for full functionality check"
    fi
    
    # Test iptables commands
    if command -v iptables >/dev/null 2>&1; then
        success "iptables command available"
    else
        error "iptables command not found"
    fi
    
    # Test kubectl commands
    if command -v kubectl >/dev/null 2>&1; then
        if kubectl cluster-info >/dev/null 2>&1; then
            success "kubectl configured and cluster accessible"
        else
            warning "kubectl found but cluster not accessible"
        fi
    else
        error "kubectl command not found"
    fi
    
    # Test namespace existence
    for ns in cyber-range-control cyber-range-red cyber-range-blue cyber-range-targets; do
        if kubectl get namespace "$ns" >/dev/null 2>&1; then
            success "Namespace $ns exists"
        else
            warning "Namespace $ns not found"
        fi
    done
    
    # Test log directory
    if [ -w "$(dirname "$LOGFILE")" ]; then
        success "Log directory writable"
    else
        error "Log directory not writable: $(dirname "$LOGFILE")"
    fi
    
    info "Kill switch test completed"
}

# Main kill switch activation function
activate_kill_switch() {
    local reason="${1:-manual_activation}"
    
    show_banner
    
    critical "🚨 ACTIVATING EMERGENCY KILL SWITCH 🚨"
    critical "Reason: $reason"
    critical "Time: $(date)"
    
    # Confirmation for manual activation
    if [ "$reason" = "manual_activation" ]; then
        echo -e "${RED}This will immediately terminate all cyber range activities!${NC}"
        read -p "Type 'KILL SWITCH' to confirm: " confirmation
        if [ "$confirmation" != "KILL SWITCH" ]; then
            error "Kill switch activation cancelled"
            exit 1
        fi
    fi
    
    critical "Kill switch activation confirmed - proceeding in 5 seconds..."
    sleep 5
    
    # Execute kill switch sequence
    backup_current_state "$reason"
    emergency_network_isolation
    activate_k8s_kill_switch
    stop_red_team_containers
    stop_target_containers
    preserve_blue_team
    send_notifications "$reason"
    local report_file=$(generate_incident_report "$reason")
    
    emergency "🚨 KILL SWITCH ACTIVATED SUCCESSFULLY 🚨"
    emergency "All attack activities terminated"
    emergency "System isolated and secured"
    emergency "Incident report: $report_file"
    emergency "Blue team preserved for analysis"
    emergency "To restore: $0 restore"
}

# Usage information
show_usage() {
    echo "XORB PTaaS Cyber Range Kill Switch"
    echo
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  activate [REASON]     Activate kill switch (default: manual_activation)"
    echo "  activate-time-limit   Activate due to time limit exceeded"
    echo "  activate-security     Activate due to security breach"
    echo "  activate-malware      Activate due to malware detection"
    echo "  status               Check kill switch status"
    echo "  deactivate           Deactivate kill switch"
    echo "  restore [BACKUP_DIR] Restore system from backup"
    echo "  test                 Test kill switch functionality"
    echo "  help                 Show this help message"
    echo
    echo "Valid reasons:"
    for reason in "${KILL_REASONS[@]}"; do
        echo "  - $reason"
    done
    echo
    echo "Examples:"
    echo "  $0 activate                    # Manual activation"
    echo "  $0 activate security_breach    # Activate due to security breach"
    echo "  $0 status                      # Check current status"
    echo "  $0 restore /path/to/backup     # Restore from specific backup"
    echo
    echo "Emergency: This script provides immediate shutdown capabilities for cyber range exercises."
}

# Main script execution
main() {
    setup_logging
    check_root
    acquire_lock
    
    local command="${1:-help}"
    
    case "$command" in
        "activate")
            activate_kill_switch "${2:-manual_activation}"
            ;;
        "activate-time-limit")
            activate_kill_switch "time_limit_exceeded"
            ;;
        "activate-security")
            activate_kill_switch "security_breach"
            ;;
        "activate-malware")
            activate_kill_switch "malware_detected"
            ;;
        "status")
            check_status
            ;;
        "deactivate")
            deactivate_kill_switch
            ;;
        "restore")
            restore_system "$2"
            ;;
        "test")
            test_kill_switch
            ;;
        "help"|"--help"|"-h")
            show_usage
            ;;
        *)
            error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"