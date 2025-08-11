#!/bin/bash
# XORB PTaaS Cyber Range - Mode Switching Script
# Switches between staging and live modes with comprehensive validation

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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIREWALL_DIR="$(dirname "$SCRIPT_DIR")/firewall"
K8S_DIR="$(dirname "$SCRIPT_DIR")/k8s"
LOGFILE="/var/log/cyber-range/mode-switch.log"
STATUS_FILE="/var/log/cyber-range/current-mode.json"
LOCK_FILE="/var/run/cyber-range-mode-switch.lock"

# Current mode detection
CURRENT_MODE="unknown"

# Logging functions
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [MODE-SWITCH] $*" | tee -a "$LOGFILE"
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

critical() {
    echo -e "${PURPLE}[CRITICAL] $*${NC}" | tee -a "$LOGFILE"
}

highlight() {
    echo -e "${CYAN}[HIGHLIGHT] $*${NC}" | tee -a "$LOGFILE"
}

# Setup logging directory
setup_logging() {
    mkdir -p "$(dirname "$LOGFILE")"
}

# Check if script is running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "Mode switching requires root privileges"
        exit 1
    fi
}

# Create lock file to prevent concurrent executions
acquire_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local lock_pid=$(cat "$LOCK_FILE" 2>/dev/null || echo "unknown")
        if kill -0 "$lock_pid" 2>/dev/null; then
            error "Mode switch is already running (PID: $lock_pid)"
            exit 1
        else
            warning "Removing stale lock file"
            rm -f "$LOCK_FILE"
        fi
    fi
    echo $$ > "$LOCK_FILE"
}

# Release lock file
release_lock() {
    rm -f "$LOCK_FILE"
}

# Trap to ensure cleanup
cleanup() {
    release_lock
}
trap cleanup EXIT

# Detect current mode
detect_current_mode() {
    info "Detecting current cyber range mode..."
    
    # Check status file first
    if [ -f "$STATUS_FILE" ]; then
        CURRENT_MODE=$(jq -r '.mode' "$STATUS_FILE" 2>/dev/null || echo "unknown")
        if [ "$CURRENT_MODE" != "unknown" ] && [ "$CURRENT_MODE" != "null" ]; then
            success "Current mode detected from status file: $CURRENT_MODE"
            return
        fi
    fi
    
    # Check iptables rules for mode detection
    if iptables -L FORWARD | grep -q "CYBER-RANGE-STAGING.*BLOCKED"; then
        CURRENT_MODE="staging"
        success "Current mode detected from iptables: staging"
        return
    elif iptables -L FORWARD | grep -q "CYBER-RANGE-LIVE.*ATTACK"; then
        CURRENT_MODE="live"
        success "Current mode detected from iptables: live"
        return
    fi
    
    # Check Kubernetes network policies
    if kubectl get networkpolicy red-team-staging-mode -n cyber-range-red >/dev/null 2>&1; then
        if kubectl get networkpolicy red-team-live-mode -n cyber-range-red >/dev/null 2>&1; then
            warning "Both staging and live policies found - checking active state"
            CURRENT_MODE="staging"  # Default to staging for safety
        else
            CURRENT_MODE="staging"
        fi
    elif kubectl get networkpolicy red-team-live-mode -n cyber-range-red >/dev/null 2>&1; then
        CURRENT_MODE="live"
    else
        CURRENT_MODE="unknown"
    fi
    
    info "Current mode detected: $CURRENT_MODE"
}

# Validate prerequisites
validate_prerequisites() {
    info "Validating prerequisites for mode switching..."
    
    local errors=0
    
    # Check required commands
    for cmd in iptables kubectl jq; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            error "Required command not found: $cmd"
            ((errors++))
        fi
    done
    
    # Check firewall scripts
    for script in "$FIREWALL_DIR/iptables-staging.sh" "$FIREWALL_DIR/iptables-live.sh"; do
        if [ ! -x "$script" ]; then
            error "Firewall script not found or not executable: $script"
            ((errors++))
        fi
    done
    
    # Check Kubernetes connectivity
    if ! kubectl cluster-info >/dev/null 2>&1; then
        error "Cannot connect to Kubernetes cluster"
        ((errors++))
    fi
    
    # Check required namespaces
    for ns in cyber-range-control cyber-range-red cyber-range-blue cyber-range-targets; do
        if ! kubectl get namespace "$ns" >/dev/null 2>&1; then
            error "Required namespace not found: $ns"
            ((errors++))
        fi
    done
    
    # Check if kill switch is active
    if [ -f "/var/log/cyber-range/kill-switch-status.json" ]; then
        local kill_status=$(jq -r '.status' "/var/log/cyber-range/kill-switch-status.json" 2>/dev/null || echo "unknown")
        if [ "$kill_status" = "kill_switch_active" ]; then
            error "Cannot switch modes while kill switch is active"
            error "Run: $SCRIPT_DIR/kill-switch.sh deactivate"
            ((errors++))
        fi
    fi
    
    if [ $errors -gt 0 ]; then
        error "Prerequisites validation failed with $errors errors"
        return 1
    fi
    
    success "Prerequisites validation passed"
    return 0
}

# Backup current configuration
backup_current_config() {
    local target_mode="$1"
    
    info "Backing up current configuration..."
    
    local backup_timestamp=$(date +%Y%m%d-%H%M%S)
    local backup_dir="/var/log/cyber-range/backups/mode-switch-$backup_timestamp"
    mkdir -p "$backup_dir"
    
    # Backup iptables rules
    iptables-save > "$backup_dir/iptables-current.rules" 2>/dev/null || true
    
    # Backup Kubernetes resources
    kubectl get networkpolicies --all-namespaces -o yaml > "$backup_dir/network-policies-current.yaml" 2>/dev/null || true
    kubectl get pods --all-namespaces -o wide > "$backup_dir/pods-current.txt" 2>/dev/null || true
    kubectl get services --all-namespaces > "$backup_dir/services-current.txt" 2>/dev/null || true
    
    # Backup current status
    if [ -f "$STATUS_FILE" ]; then
        cp "$STATUS_FILE" "$backup_dir/mode-status-current.json"
    fi
    
    # Create backup metadata
    cat > "$backup_dir/backup-metadata.json" << EOF
{
    "backup_timestamp": "$backup_timestamp",
    "backup_reason": "mode_switch_to_$target_mode",
    "source_mode": "$CURRENT_MODE",
    "target_mode": "$target_mode",
    "operator": "$(whoami)",
    "hostname": "$(hostname)"
}
EOF
    
    success "Configuration backed up to: $backup_dir"
    echo "$backup_dir" > /tmp/last-mode-switch-backup
}

# Apply iptables configuration for target mode
apply_iptables_config() {
    local target_mode="$1"
    
    info "Applying iptables configuration for $target_mode mode..."
    
    local firewall_script=""
    case "$target_mode" in
        "staging")
            firewall_script="$FIREWALL_DIR/iptables-staging.sh"
            ;;
        "live")
            firewall_script="$FIREWALL_DIR/iptables-live.sh"
            ;;
        *)
            error "Unknown target mode: $target_mode"
            return 1
            ;;
    esac
    
    if [ ! -x "$firewall_script" ]; then
        error "Firewall script not executable: $firewall_script"
        return 1
    fi
    
    # Execute firewall script
    if [ "$target_mode" = "live" ]; then
        # For live mode, set environment variable to skip confirmation
        FORCE_LIVE_MODE=true "$firewall_script" force
    else
        "$firewall_script"
    fi
    
    success "iptables configuration applied for $target_mode mode"
}

# Apply Kubernetes network policies for target mode
apply_k8s_policies() {
    local target_mode="$1"
    
    info "Applying Kubernetes network policies for $target_mode mode..."
    
    # Remove existing mode-specific policies
    kubectl delete networkpolicy red-team-staging-mode -n cyber-range-red 2>/dev/null || true
    kubectl delete networkpolicy red-team-live-mode -n cyber-range-red 2>/dev/null || true
    
    # Apply base network policies if not already applied
    if ! kubectl get networkpolicy default-deny-all-ingress-egress -n cyber-range-red >/dev/null 2>&1; then
        kubectl apply -f "$K8S_DIR/network-policies.yaml"
    fi
    
    # Apply mode-specific policies
    case "$target_mode" in
        "staging")
            # Apply staging mode policy (blocks red team attacks)
            kubectl apply -f - << 'EOF'
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: red-team-staging-mode
  namespace: cyber-range-red
  labels:
    cyber-range.xorb.io/policy-type: attack-staging
    cyber-range.xorb.io/mode: staging
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
      port: 8080
    - protocol: TCP
      port: 22
  - from:
    - podSelector: {}
  egress:
  - to: []
    ports:
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 53
  - to:
    - podSelector: {}
  - to:
    - namespaceSelector:
        matchLabels:
          app.kubernetes.io/name: cyber-range-control
    ports:
    - protocol: TCP
      port: 8080
EOF
            ;;
        "live")
            # Apply live mode policy (allows red team attacks)
            kubectl apply -f - << 'EOF'
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: red-team-live-mode
  namespace: cyber-range-red
  labels:
    cyber-range.xorb.io/policy-type: attack-live
    cyber-range.xorb.io/mode: live
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
      port: 8080
    - protocol: TCP
      port: 22
  - from:
    - podSelector: {}
  - from:
    - namespaceSelector:
        matchLabels:
          app.kubernetes.io/name: cyber-range-targets
    ports:
    - protocol: TCP
      port: 4444
    - protocol: TCP
      port: 1234
    - protocol: TCP
      port: 8080
  egress:
  - to: []
    ports:
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 53
  - to:
    - podSelector: {}
  - to:
    - namespaceSelector:
        matchLabels:
          app.kubernetes.io/name: cyber-range-control
    ports:
    - protocol: TCP
      port: 8080
  - to:
    - namespaceSelector:
        matchLabels:
          app.kubernetes.io/name: cyber-range-targets
  - to: []
    ports:
    - protocol: TCP
      port: 80
    - protocol: TCP
      port: 443
EOF
            ;;
        *)
            error "Unknown target mode: $target_mode"
            return 1
            ;;
    esac
    
    success "Kubernetes network policies applied for $target_mode mode"
}

# Update XORB orchestrator configuration
update_orchestrator_config() {
    local target_mode="$1"
    
    info "Updating XORB orchestrator configuration for $target_mode mode..."
    
    # Update configmap with new mode
    kubectl patch configmap xorb-control-config -n cyber-range-control --patch "
data:
  config.yaml: |
    cyber_range:
      mode: \"$target_mode\"
      exercise_duration: \"4h\"
      auto_reset: true
      kill_switch_enabled: true
      
    security:
      network_isolation: true
      traffic_monitoring: true
      malware_detection: true
      geographic_restrictions: true
      
    logging:
      level: \"INFO\"
      audit_enabled: true
      retention_days: 30
      
    teams:
      red_team:
        namespace: \"cyber-range-red\"
        max_concurrent_attacks: $([ "$target_mode" = "live" ] && echo "10" || echo "0")
        rate_limiting: true
        attacks_enabled: $([ "$target_mode" = "live" ] && echo "true" || echo "false")
      blue_team:
        namespace: \"cyber-range-blue\"
        monitoring_enabled: true
        alert_thresholds:
          critical: 5
          warning: 20
          
    targets:
      auto_restore: true
      snapshot_interval: \"15m\"
      backup_retention: \"7d\"
      
    scenarios:
      available:
        - \"web_app_pentest\"
        - \"network_lateral_movement\"
        - \"apt_simulation\"
        - \"insider_threat\"
        - \"ransomware_defense\"
      default_scenario: \"web_app_pentest\"
"
    
    # Restart orchestrator to pick up new config
    kubectl rollout restart deployment xorb-orchestrator -n cyber-range-control
    
    success "XORB orchestrator configuration updated for $target_mode mode"
}

# Restart affected services
restart_services() {
    local target_mode="$1"
    
    info "Restarting affected services for $target_mode mode..."
    
    # Restart red team services to pick up new network policies
    kubectl rollout restart deployment --all -n cyber-range-red 2>/dev/null || true
    
    # Wait for rollouts to complete
    kubectl rollout status deployment --all -n cyber-range-red --timeout=300s 2>/dev/null || warning "Some red team deployments did not restart cleanly"
    
    # Restart blue team monitoring to pick up new mode
    kubectl rollout restart deployment --all -n cyber-range-blue 2>/dev/null || true
    
    success "Services restarted for $target_mode mode"
}

# Update mode status file
update_status_file() {
    local target_mode="$1"
    
    info "Updating mode status file..."
    
    cat > "$STATUS_FILE" << EOF
{
    "mode": "$target_mode",
    "switched_at": "$(date -Iseconds)",
    "switched_from": "$CURRENT_MODE",
    "operator": "$(whoami)",
    "hostname": "$(hostname)",
    "iptables_config": "applied",
    "k8s_policies": "applied",
    "orchestrator_config": "updated",
    "services_restarted": true,
    "backup_location": "$(cat /tmp/last-mode-switch-backup 2>/dev/null || echo "unknown")"
}
EOF
    
    success "Mode status file updated"
}

# Verify mode switch was successful
verify_mode_switch() {
    local target_mode="$1"
    
    info "Verifying mode switch to $target_mode..."
    
    local errors=0
    local warnings=0
    
    # Check iptables rules
    case "$target_mode" in
        "staging")
            if iptables -L FORWARD | grep -q "CYBER-RANGE-STAGING.*BLOCKED"; then
                success "iptables staging rules verified ✓"
            else
                error "iptables staging rules not found"
                ((errors++))
            fi
            ;;
        "live")
            if iptables -L FORWARD | grep -q "CYBER-RANGE-LIVE.*ATTACK"; then
                success "iptables live rules verified ✓"
            else
                error "iptables live rules not found"
                ((errors++))
            fi
            ;;
    esac
    
    # Check Kubernetes network policies
    if kubectl get networkpolicy "red-team-$target_mode-mode" -n cyber-range-red >/dev/null 2>&1; then
        success "Kubernetes $target_mode mode policy verified ✓"
    else
        error "Kubernetes $target_mode mode policy not found"
        ((errors++))
    fi
    
    # Check XORB orchestrator config
    local orchestrator_mode=$(kubectl get configmap xorb-control-config -n cyber-range-control -o jsonpath='{.data.config\.yaml}' | grep -o 'mode: "[^"]*"' | cut -d'"' -f2)
    if [ "$orchestrator_mode" = "$target_mode" ]; then
        success "XORB orchestrator mode verified: $target_mode ✓"
    else
        warning "XORB orchestrator mode mismatch: expected $target_mode, got $orchestrator_mode"
        ((warnings++))
    fi
    
    # Check pod status
    local red_pods=$(kubectl get pods -n cyber-range-red --no-headers | wc -l)
    local blue_pods=$(kubectl get pods -n cyber-range-blue --no-headers | wc -l)
    
    if [ "$red_pods" -gt 0 ]; then
        success "Red team pods running: $red_pods ✓"
    else
        warning "No red team pods running"
        ((warnings++))
    fi
    
    if [ "$blue_pods" -gt 0 ]; then
        success "Blue team pods running: $blue_pods ✓"
    else
        warning "No blue team pods running"
        ((warnings++))
    fi
    
    # Overall verification result
    if [ $errors -eq 0 ]; then
        if [ $warnings -eq 0 ]; then
            success "Mode switch verification PASSED completely ✓"
            return 0
        else
            warning "Mode switch verification PASSED with $warnings warnings ⚠️"
            return 0
        fi
    else
        error "Mode switch verification FAILED with $errors errors ❌"
        return 1
    fi
}

# Display current status
show_status() {
    detect_current_mode
    
    echo -e "\n${CYAN}=== XORB Cyber Range Mode Status ===${NC}"
    echo
    
    case "$CURRENT_MODE" in
        "staging")
            echo -e "${GREEN}Current Mode: STAGING${NC}"
            echo -e "${YELLOW}Red Team Status: ATTACKS BLOCKED (Safe Training)${NC}"
            echo -e "${BLUE}Blue Team Status: MONITORING ACTIVE${NC}"
            echo "Description: Safe training mode with red team attacks blocked"
            ;;
        "live")
            echo -e "${RED}Current Mode: LIVE EXERCISE${NC}"
            echo -e "${RED}Red Team Status: ATTACKS ENABLED ⚠️${NC}"
            echo -e "${BLUE}Blue Team Status: MONITORING ACTIVE${NC}"
            echo "Description: Active exercise with real red team attacks"
            ;;
        "unknown")
            echo -e "${YELLOW}Current Mode: UNKNOWN${NC}"
            echo "Unable to determine current mode from system state"
            ;;
    esac
    
    echo
    echo "System Information:"
    
    # Check iptables status
    if iptables -L FORWARD | grep -q "CYBER-RANGE"; then
        echo "- iptables: Cyber range rules active"
    else
        echo "- iptables: No cyber range rules found"
    fi
    
    # Check Kubernetes status
    local red_policies=$(kubectl get networkpolicy -n cyber-range-red --no-headers | wc -l)
    echo "- Kubernetes: $red_policies network policies in red team namespace"
    
    # Check kill switch status
    if [ -f "/var/log/cyber-range/kill-switch-status.json" ]; then
        local kill_status=$(jq -r '.status' "/var/log/cyber-range/kill-switch-status.json" 2>/dev/null || echo "unknown")
        if [ "$kill_status" = "kill_switch_active" ]; then
            echo -e "- Kill Switch: ${RED}ACTIVE${NC} ⚠️"
        else
            echo -e "- Kill Switch: ${GREEN}INACTIVE${NC}"
        fi
    else
        echo -e "- Kill Switch: ${GREEN}INACTIVE${NC}"
    fi
    
    # Show last mode switch info
    if [ -f "$STATUS_FILE" ]; then
        echo
        echo "Last Mode Switch:"
        local last_switch=$(jq -r '.switched_at' "$STATUS_FILE" 2>/dev/null || echo "unknown")
        local switched_from=$(jq -r '.switched_from' "$STATUS_FILE" 2>/dev/null || echo "unknown")
        echo "- Time: $last_switch"
        echo "- From: $switched_from → $CURRENT_MODE"
    fi
    
    echo
    echo "Available Commands:"
    echo "- Switch to staging: $0 staging"
    echo "- Switch to live: $0 live"
    echo "- Emergency stop: $SCRIPT_DIR/kill-switch.sh activate"
}

# Main mode switching function
switch_mode() {
    local target_mode="$1"
    
    detect_current_mode
    
    # Check if already in target mode
    if [ "$CURRENT_MODE" = "$target_mode" ]; then
        warning "Already in $target_mode mode"
        show_status
        return 0
    fi
    
    critical "Switching cyber range mode: $CURRENT_MODE → $target_mode"
    
    # Special confirmation for live mode
    if [ "$target_mode" = "live" ]; then
        echo -e "${RED}⚠️  WARNING: Switching to LIVE mode will enable real attacks ⚠️${NC}"
        echo "This allows red team to actually attack target systems"
        echo "Make sure all participants are ready and safety measures are in place"
        echo
        read -p "Type 'ENABLE LIVE MODE' to confirm: " confirmation
        if [ "$confirmation" != "ENABLE LIVE MODE" ]; then
            error "Live mode switch cancelled"
            return 1
        fi
    fi
    
    # Execute mode switch sequence
    info "Starting mode switch sequence..."
    
    if ! validate_prerequisites; then
        error "Prerequisites validation failed"
        return 1
    fi
    
    backup_current_config "$target_mode"
    apply_iptables_config "$target_mode"
    apply_k8s_policies "$target_mode"
    update_orchestrator_config "$target_mode"
    restart_services "$target_mode"
    update_status_file "$target_mode"
    
    if verify_mode_switch "$target_mode"; then
        critical "Mode switch completed successfully: $CURRENT_MODE → $target_mode"
        show_status
        
        if [ "$target_mode" = "live" ]; then
            echo
            critical "🚨 RED TEAM ATTACKS ARE NOW ENABLED 🚨"
            critical "Emergency stop: $SCRIPT_DIR/kill-switch.sh activate"
        fi
        
        return 0
    else
        error "Mode switch verification failed"
        return 1
    fi
}

# Dry run mode switch (test without applying changes)
dry_run_mode_switch() {
    local target_mode="$1"
    
    detect_current_mode
    
    info "Performing dry run mode switch: $CURRENT_MODE → $target_mode"
    
    if ! validate_prerequisites; then
        error "Dry run failed: Prerequisites validation failed"
        return 1
    fi
    
    info "Dry run checks:"
    
    # Check firewall script
    local firewall_script=""
    case "$target_mode" in
        "staging")
            firewall_script="$FIREWALL_DIR/iptables-staging.sh"
            ;;
        "live")
            firewall_script="$FIREWALL_DIR/iptables-live.sh"
            ;;
    esac
    
    if [ -x "$firewall_script" ]; then
        success "Firewall script available: $firewall_script"
    else
        error "Firewall script not available: $firewall_script"
    fi
    
    # Check network policies
    if [ -f "$K8S_DIR/network-policies.yaml" ]; then
        success "Network policies file available"
    else
        error "Network policies file not found"
    fi
    
    # Check orchestrator config
    if kubectl get configmap xorb-control-config -n cyber-range-control >/dev/null 2>&1; then
        success "XORB orchestrator config accessible"
    else
        error "XORB orchestrator config not accessible"
    fi
    
    success "Dry run completed - mode switch to $target_mode appears feasible"
}

# Usage information
show_usage() {
    echo "XORB PTaaS Cyber Range Mode Switching"
    echo
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  staging              Switch to staging mode (safe training)"
    echo "  live                 Switch to live exercise mode (attacks enabled)"
    echo "  status               Show current mode and system status"
    echo "  dry-run MODE         Test mode switch without applying changes"
    echo "  help                 Show this help message"
    echo
    echo "Modes:"
    echo "  staging              Safe training mode - red team attacks blocked"
    echo "  live                 Live exercise mode - red team attacks enabled"
    echo
    echo "Examples:"
    echo "  $0 staging           # Switch to safe training mode"
    echo "  $0 live              # Switch to live exercise mode (with confirmation)"
    echo "  $0 status            # Check current mode"
    echo "  $0 dry-run live      # Test live mode switch"
    echo
    echo "Safety:"
    echo "  - All mode switches are logged and backed up"
    echo "  - Live mode requires explicit confirmation"
    echo "  - Emergency kill switch available: $SCRIPT_DIR/kill-switch.sh"
}

# Main script execution
main() {
    setup_logging
    check_root
    acquire_lock
    
    local command="${1:-status}"
    
    case "$command" in
        "staging")
            switch_mode "staging"
            ;;
        "live")
            switch_mode "live"
            ;;
        "status")
            show_status
            ;;
        "dry-run")
            if [ -n "${2:-}" ]; then
                dry_run_mode_switch "$2"
            else
                error "Dry run requires target mode (staging/live)"
                show_usage
                exit 1
            fi
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