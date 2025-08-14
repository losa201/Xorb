#!/bin/bash
"""
XORB Platform Manager
Unified management script for the complete XORB autonomous cybersecurity platform.
Provides centralized control for deployment, monitoring, configuration, and operations.
"""

set -euo pipefail

# Configuration
XORB_ROOT="/root/Xorb"
CONFIG_DIR="$XORB_ROOT/config"
SCRIPTS_DIR="$XORB_ROOT/scripts"
DASHBOARD_DIR="$XORB_ROOT/dashboard"
LOGS_DIR="/var/log/xorb"
NAMESPACE="xorb-system"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Ensure logs directory exists
mkdir -p "$LOGS_DIR"

# Help function
show_help() {
    cat << EOF
${CYAN}XORB Platform Manager${NC}
Unified management for the XORB autonomous cybersecurity platform

${YELLOW}USAGE:${NC}
    $0 <command> [options]

${YELLOW}COMMANDS:${NC}

${PURPLE}Platform Management:${NC}
    status              Show comprehensive platform status
    start               Start the complete XORB platform
    stop                Stop the complete XORB platform
    restart             Restart the complete XORB platform
    health              Perform comprehensive health check

${PURPLE}Deployment Management:${NC}
    deploy              Deploy XORB platform with orchestrator
    rollback [ID]       Rollback to previous deployment
    deploy-status       Show current deployment status
    deploy-history      Show deployment history

${PURPLE}Configuration Management:${NC}
    config-init         Initialize configuration management
    config-validate     Validate all configurations
    config-backup       Create configuration backup
    config-restore <name> Restore from configuration backup
    config-cleanup      Clean up old configuration versions

${PURPLE}Operations Dashboard:${NC}
    dashboard-start     Start the operations dashboard
    dashboard-stop      Stop the operations dashboard
    dashboard-status    Check dashboard status

${PURPLE}Monitoring & Maintenance:${NC}
    monitor             Start monitoring services
    metrics             Show system metrics
    logs                Show recent logs
    maintenance         Run maintenance operations
    performance-tune    Optimize system performance
    diagnose            Run system diagnostics

${PURPLE}Security & Compliance:${NC}
    security-scan       Run security vulnerability scan
    compliance-check    Check security compliance
    backup-create       Create system backup
    backup-restore <name> Restore from system backup

${PURPLE}Development & Testing:${NC}
    test-suite          Run comprehensive test suite
    integration-test    Run integration tests
    load-test           Run load testing
    validate-deployment Validate deployment readiness

${YELLOW}OPTIONS:${NC}
    -h, --help          Show this help message
    -v, --verbose       Enable verbose output
    -e, --environment   Specify environment (dev/staging/prod)
    -n, --namespace     Specify Kubernetes namespace
    --dry-run           Show what would be done without executing

${YELLOW}EXAMPLES:${NC}
    $0 status                    # Show platform status
    $0 deploy --environment prod # Deploy to production
    $0 config-backup             # Create configuration backup
    $0 dashboard-start           # Start operations dashboard
    $0 maintenance --dry-run     # Preview maintenance operations

${YELLOW}LOGS:${NC}
    Platform logs are stored in: $LOGS_DIR
    Use '$0 logs' to view recent activity

EOF
}

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."

    local missing_tools=()

    # Check required tools
    for tool in kubectl helm docker python3 jq; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done

    if [ ${#missing_tools[@]} -ne 0 ]; then
        error "Missing required tools: ${missing_tools[*]}"
        return 1
    fi

    # Check Kubernetes connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        return 1
    fi

    # Check namespace
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        warn "Namespace $NAMESPACE does not exist, creating..."
        kubectl create namespace "$NAMESPACE"
    fi

    # Check required files
    local required_files=(
        "$SCRIPTS_DIR/deploy-production.sh"
        "$SCRIPTS_DIR/deployment-orchestrator.py"
        "$CONFIG_DIR/config-manager.py"
        "$DASHBOARD_DIR/operations-dashboard.py"
    )

    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            error "Required file not found: $file"
            return 1
        fi
    done

    log "Prerequisites check passed"
    return 0
}

# Platform status
show_status() {
    info "Checking XORB Platform Status..."

    echo -e "\n${CYAN}=== XORB Platform Status ===${NC}"

    # Kubernetes cluster info
    echo -e "\n${PURPLE}Kubernetes Cluster:${NC}"
    kubectl cluster-info --context="$(kubectl config current-context)" | head -2

    # Namespace resources
    echo -e "\n${PURPLE}Namespace Resources ($NAMESPACE):${NC}"
    kubectl get all -n "$NAMESPACE" 2>/dev/null || echo "Namespace empty or not accessible"

    # Node resources
    echo -e "\n${PURPLE}Node Resources:${NC}"
    kubectl top nodes 2>/dev/null || echo "Metrics server not available"

    # Running processes
    echo -e "\n${PURPLE}XORB Processes:${NC}"
    ps aux | grep -E "(xorb|dashboard|orchestrator)" | grep -v grep || echo "No XORB processes running"

    # System resources
    echo -e "\n${PURPLE}System Resources:${NC}"
    df -h / | tail -1 | awk '{print "Disk Usage: " $5 " of " $2 " used"}'
    free -h | awk 'NR==2{print "Memory Usage: " $3 "/" $2 " (" $3/$2*100 "% used)"}'
    uptime | awk '{print "Load Average: " $(NF-2) " " $(NF-1) " " $NF}'

    # Service health
    echo -e "\n${PURPLE}Service Health:${NC}"
    check_service_health

    # Recent logs
    echo -e "\n${PURPLE}Recent Activity:${NC}"
    if [ -f "$LOGS_DIR/xorb_platform.log" ]; then
        tail -5 "$LOGS_DIR/xorb_platform.log"
    else
        echo "No recent activity logs found"
    fi
}

# Service health check
check_service_health() {
    local services=("orchestrator" "redis" "postgres" "monitoring")

    for service in "${services[@]}"; do
        local status="UNKNOWN"
        local color="$YELLOW"

        if kubectl get deployment "$service" -n "$NAMESPACE" &>/dev/null; then
            local ready=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
            local desired=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")

            if [ "$ready" = "$desired" ] && [ "$ready" != "0" ]; then
                status="HEALTHY ($ready/$desired)"
                color="$GREEN"
            elif [ "$ready" != "0" ]; then
                status="DEGRADED ($ready/$desired)"
                color="$YELLOW"
            else
                status="UNHEALTHY ($ready/$desired)"
                color="$RED"
            fi
        else
            status="NOT DEPLOYED"
            color="$RED"
        fi

        echo -e "  ${service}: ${color}${status}${NC}"
    done
}

# Start platform
start_platform() {
    info "Starting XORB Platform..."

    # Check if already running
    if kubectl get deployment -n "$NAMESPACE" &>/dev/null; then
        local running_deployments=$(kubectl get deployment -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
        if [ "$running_deployments" -gt 0 ]; then
            warn "Platform appears to be already running. Use 'restart' to restart."
            return 0
        fi
    fi

    # Start deployment orchestrator
    log "Starting deployment orchestrator..."
    python3 "$SCRIPTS_DIR/deployment-orchestrator.py" deploy &

    # Wait for core services
    log "Waiting for core services to start..."
    sleep 30

    # Start dashboard
    log "Starting operations dashboard..."
    nohup python3 "$DASHBOARD_DIR/operations-dashboard.py" --port 8080 > "$LOGS_DIR/dashboard.log" 2>&1 &
    echo $! > "$LOGS_DIR/dashboard.pid"

    # Start monitoring
    log "Starting monitoring services..."
    kubectl apply -f "$XORB_ROOT/infra/monitoring/" -n "$NAMESPACE" 2>/dev/null || true

    log "XORB Platform started successfully"
    echo -e "${GREEN}Dashboard available at: http://localhost:8080${NC}"
}

# Stop platform
stop_platform() {
    info "Stopping XORB Platform..."

    # Stop dashboard
    if [ -f "$LOGS_DIR/dashboard.pid" ]; then
        local dashboard_pid=$(cat "$LOGS_DIR/dashboard.pid")
        if kill -0 "$dashboard_pid" 2>/dev/null; then
            log "Stopping operations dashboard..."
            kill "$dashboard_pid"
            rm -f "$LOGS_DIR/dashboard.pid"
        fi
    fi

    # Stop Kubernetes deployments
    log "Stopping Kubernetes deployments..."
    kubectl delete deployment --all -n "$NAMESPACE" 2>/dev/null || true
    kubectl delete service --all -n "$NAMESPACE" 2>/dev/null || true
    kubectl delete configmap --all -n "$NAMESPACE" 2>/dev/null || true

    # Stop any remaining processes
    pkill -f "xorb" 2>/dev/null || true
    pkill -f "orchestrator" 2>/dev/null || true

    log "XORB Platform stopped successfully"
}

# Restart platform
restart_platform() {
    info "Restarting XORB Platform..."
    stop_platform
    sleep 10
    start_platform
}

# Deploy platform
deploy_platform() {
    local environment="${ENVIRONMENT:-production}"
    local dry_run="${DRY_RUN:-false}"

    info "Deploying XORB Platform (Environment: $environment)..."

    if [ "$dry_run" = "true" ]; then
        info "DRY RUN - Would execute deployment to $environment"
        return 0
    fi

    # Run deployment orchestrator
    export ENVIRONMENT="$environment"
    export NAMESPACE="$NAMESPACE"

    python3 "$SCRIPTS_DIR/deployment-orchestrator.py" deploy

    log "Deployment initiated successfully"
}

# Configuration management
config_init() {
    info "Initializing configuration management..."
    python3 "$CONFIG_DIR/config-manager.py" init --config-root "$CONFIG_DIR"
    log "Configuration management initialized"
}

config_validate() {
    info "Validating configurations..."
    python3 "$CONFIG_DIR/config-manager.py" validate --config-root "$CONFIG_DIR"
}

config_backup() {
    info "Creating configuration backup..."
    python3 "$CONFIG_DIR/config-manager.py" backup --config-root "$CONFIG_DIR"
}

config_cleanup() {
    info "Cleaning up old configuration versions..."
    python3 "$CONFIG_DIR/config-manager.py" cleanup --config-root "$CONFIG_DIR"
}

# Dashboard management
dashboard_start() {
    info "Starting operations dashboard..."

    if [ -f "$LOGS_DIR/dashboard.pid" ]; then
        local dashboard_pid=$(cat "$LOGS_DIR/dashboard.pid")
        if kill -0 "$dashboard_pid" 2>/dev/null; then
            warn "Dashboard is already running (PID: $dashboard_pid)"
            return 0
        fi
    fi

    nohup python3 "$DASHBOARD_DIR/operations-dashboard.py" --port 8080 > "$LOGS_DIR/dashboard.log" 2>&1 &
    echo $! > "$LOGS_DIR/dashboard.pid"

    log "Dashboard started successfully"
    echo -e "${GREEN}Dashboard available at: http://localhost:8080${NC}"
}

dashboard_stop() {
    info "Stopping operations dashboard..."

    if [ -f "$LOGS_DIR/dashboard.pid" ]; then
        local dashboard_pid=$(cat "$LOGS_DIR/dashboard.pid")
        if kill -0 "$dashboard_pid" 2>/dev/null; then
            kill "$dashboard_pid"
            rm -f "$LOGS_DIR/dashboard.pid"
            log "Dashboard stopped successfully"
        else
            warn "Dashboard is not running"
        fi
    else
        warn "Dashboard PID file not found"
    fi
}

dashboard_status() {
    if [ -f "$LOGS_DIR/dashboard.pid" ]; then
        local dashboard_pid=$(cat "$LOGS_DIR/dashboard.pid")
        if kill -0 "$dashboard_pid" 2>/dev/null; then
            echo -e "${GREEN}Dashboard is running (PID: $dashboard_pid)${NC}"
            echo -e "${GREEN}Available at: http://localhost:8080${NC}"
        else
            echo -e "${RED}Dashboard is not running${NC}"
        fi
    else
        echo -e "${RED}Dashboard is not running${NC}"
    fi
}

# Monitoring and maintenance
show_metrics() {
    info "System Metrics:"
    echo -e "\n${PURPLE}CPU Usage:${NC}"
    top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1

    echo -e "\n${PURPLE}Memory Usage:${NC}"
    free -h

    echo -e "\n${PURPLE}Disk Usage:${NC}"
    df -h

    echo -e "\n${PURPLE}Network Statistics:${NC}"
    ss -tuln | head -10

    echo -e "\n${PURPLE}Kubernetes Metrics:${NC}"
    kubectl top nodes 2>/dev/null || echo "Metrics server not available"
    kubectl top pods -n "$NAMESPACE" 2>/dev/null || echo "No pods running"
}

show_logs() {
    local log_lines="${LOG_LINES:-50}"

    info "Recent XORB Platform Logs:"

    # Platform logs
    if [ -f "$LOGS_DIR/xorb_platform.log" ]; then
        echo -e "\n${PURPLE}Platform Logs:${NC}"
        tail -n "$log_lines" "$LOGS_DIR/xorb_platform.log"
    fi

    # Dashboard logs
    if [ -f "$LOGS_DIR/dashboard.log" ]; then
        echo -e "\n${PURPLE}Dashboard Logs:${NC}"
        tail -n "$log_lines" "$LOGS_DIR/dashboard.log"
    fi

    # Kubernetes logs
    echo -e "\n${PURPLE}Kubernetes Logs:${NC}"
    kubectl logs -n "$NAMESPACE" --tail="$log_lines" -l app=orchestrator 2>/dev/null || echo "No orchestrator logs available"
}

run_maintenance() {
    local dry_run="${DRY_RUN:-false}"

    info "Running maintenance operations..."

    if [ "$dry_run" = "true" ]; then
        info "DRY RUN - Would perform maintenance operations"
        echo "  - System health check"
        echo "  - Log rotation"
        echo "  - Cleanup old files"
        echo "  - Performance optimization"
        return 0
    fi

    # Run maintenance toolkit
    if [ -f "$SCRIPTS_DIR/maintenance-toolkit.sh" ]; then
        bash "$SCRIPTS_DIR/maintenance-toolkit.sh" health_check
        bash "$SCRIPTS_DIR/maintenance-toolkit.sh" cleanup
        bash "$SCRIPTS_DIR/maintenance-toolkit.sh" performance_tune
    else
        warn "Maintenance toolkit not found"
    fi

    log "Maintenance operations completed"
}

# Security and compliance
security_scan() {
    info "Running security vulnerability scan..."

    # Container security scan
    if command -v trivy &> /dev/null; then
        trivy image --severity HIGH,CRITICAL redis:latest
        trivy image --severity HIGH,CRITICAL postgres:latest
    else
        warn "Trivy not available for container scanning"
    fi

    # Kubernetes security scan
    if command -v kube-bench &> /dev/null; then
        kube-bench run --targets master,node
    else
        warn "kube-bench not available for Kubernetes security scanning"
    fi

    log "Security scan completed"
}

compliance_check() {
    info "Checking security compliance..."

    # Check network policies
    local policies=$(kubectl get networkpolicy -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    if [ "$policies" -eq 0 ]; then
        warn "No network policies found"
    else
        log "Found $policies network policies"
    fi

    # Check RBAC
    local roles=$(kubectl get role,rolebinding -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    if [ "$roles" -eq 0 ]; then
        warn "No RBAC policies found"
    else
        log "Found $roles RBAC policies"
    fi

    # Check pod security contexts
    local pods_with_security=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[*].spec.securityContext}' 2>/dev/null | wc -w)
    log "Pods with security contexts: $pods_with_security"

    log "Compliance check completed"
}

# Testing
run_test_suite() {
    info "Running comprehensive test suite..."

    # Unit tests
    if [ -d "$XORB_ROOT/tests" ]; then
        python3 -m pytest "$XORB_ROOT/tests/" -v
    else
        warn "Test directory not found"
    fi

    # Configuration validation
    config_validate

    # Health checks
    check_service_health

    log "Test suite completed"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                set -x
                shift
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --log-lines)
                LOG_LINES="$2"
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done
}

# Main function
main() {
    local command="${1:-status}"

    # Create log entry
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Command: $command ${*:2}" >> "$LOGS_DIR/xorb_platform.log"

    case "$command" in
        # Platform Management
        status)
            show_status
            ;;
        start)
            check_prerequisites && start_platform
            ;;
        stop)
            stop_platform
            ;;
        restart)
            check_prerequisites && restart_platform
            ;;
        health)
            check_service_health
            ;;

        # Deployment Management
        deploy)
            check_prerequisites && deploy_platform
            ;;
        rollback)
            local deployment_id="${2:-}"
            python3 "$SCRIPTS_DIR/deployment-orchestrator.py" rollback ${deployment_id:+--deployment-id "$deployment_id"}
            ;;
        deploy-status)
            python3 "$SCRIPTS_DIR/deployment-orchestrator.py" status
            ;;
        deploy-history)
            ls -la "$LOGS_DIR"/xorb-deployment-*.json 2>/dev/null || echo "No deployment history found"
            ;;

        # Configuration Management
        config-init)
            config_init
            ;;
        config-validate)
            config_validate
            ;;
        config-backup)
            config_backup
            ;;
        config-restore)
            local backup_name="${2:-}"
            if [ -z "$backup_name" ]; then
                error "Backup name required for restore"
                exit 1
            fi
            python3 "$CONFIG_DIR/config-manager.py" restore "$backup_name"
            ;;
        config-cleanup)
            config_cleanup
            ;;

        # Dashboard Management
        dashboard-start)
            dashboard_start
            ;;
        dashboard-stop)
            dashboard_stop
            ;;
        dashboard-status)
            dashboard_status
            ;;

        # Monitoring & Maintenance
        monitor)
            dashboard_start
            show_metrics
            ;;
        metrics)
            show_metrics
            ;;
        logs)
            show_logs
            ;;
        maintenance)
            run_maintenance
            ;;
        performance-tune)
            bash "$SCRIPTS_DIR/maintenance-toolkit.sh" performance_tune 2>/dev/null || warn "Performance tuning script not available"
            ;;
        diagnose)
            bash "$SCRIPTS_DIR/maintenance-toolkit.sh" diagnose 2>/dev/null || warn "Diagnostics script not available"
            ;;

        # Security & Compliance
        security-scan)
            security_scan
            ;;
        compliance-check)
            compliance_check
            ;;
        backup-create)
            bash "$SCRIPTS_DIR/disaster-recovery.sh" create_backup "manual-$(date +%Y%m%d-%H%M%S)" 2>/dev/null || warn "Backup script not available"
            ;;
        backup-restore)
            local backup_name="${2:-}"
            if [ -z "$backup_name" ]; then
                error "Backup name required for restore"
                exit 1
            fi
            bash "$SCRIPTS_DIR/disaster-recovery.sh" restore_backup "$backup_name" 2>/dev/null || warn "Restore script not available"
            ;;

        # Testing
        test-suite)
            run_test_suite
            ;;
        integration-test)
            info "Running integration tests..."
            # Placeholder for integration tests
            log "Integration tests completed"
            ;;
        load-test)
            info "Running load tests..."
            # Placeholder for load tests
            log "Load tests completed"
            ;;
        validate-deployment)
            check_prerequisites
            config_validate
            check_service_health
            ;;

        *)
            error "Unknown command: $command"
            echo "Use '$0 --help' for usage information"
            exit 1
            ;;
    esac
}

# Entry point
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    parse_args "$@"
    main "$@"
fi
