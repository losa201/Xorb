#!/bin/bash
set -euo pipefail

# ===================================================================
# XORB Autonomous Orchestrator - Maintenance Toolkit
# Comprehensive maintenance, optimization, and troubleshooting tools
# ===================================================================

# Configuration
NAMESPACE="${NAMESPACE:-xorb-system}"
OPERATION="${1:-help}"
MAINTENANCE_LOG="/var/log/xorb/maintenance-$(date +%Y%m%d).log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m'

# Logging functions
log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅${NC} $1" | tee -a "$MAINTENANCE_LOG"; }
info() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] ℹ️${NC} $1" | tee -a "$MAINTENANCE_LOG"; }
warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️${NC} $1" | tee -a "$MAINTENANCE_LOG"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌${NC} $1" | tee -a "$MAINTENANCE_LOG"; }
step() { echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] 🔧${NC} $1" | tee -a "$MAINTENANCE_LOG"; }

# Create log directory
mkdir -p "$(dirname "$MAINTENANCE_LOG")"

# Usage information
show_usage() {
    cat << EOF
${WHITE}XORB Maintenance Toolkit${NC}

${CYAN}Usage:${NC}
  ./maintenance-toolkit.sh <command> [options]

${CYAN}System Maintenance:${NC}
  ${GREEN}health-check${NC}        Comprehensive system health assessment
  ${GREEN}performance-tune${NC}    Optimize system performance
  ${GREEN}resource-cleanup${NC}    Clean up unused resources
  ${GREEN}update-system${NC}       Update system components
  ${GREEN}security-scan${NC}       Security vulnerability assessment

${CYAN}Troubleshooting:${NC}
  ${GREEN}diagnose${NC}            Run full system diagnostics
  ${GREEN}restart-services${NC}    Restart failed services
  ${GREEN}reset-passwords${NC}     Reset system passwords
  ${GREEN}repair-database${NC}     Database integrity check and repair
  ${GREEN}network-test${NC}        Network connectivity diagnostics

${CYAN}Optimization:${NC}
  ${GREEN}scale-up${NC}            Scale up system resources
  ${GREEN}scale-down${NC}          Scale down system resources
  ${GREEN}optimize-db${NC}         Database optimization
  ${GREEN}cache-warm${NC}          Warm up system caches
  ${GREEN}index-rebuild${NC}       Rebuild database indexes

${CYAN}Monitoring & Reports:${NC}
  ${GREEN}system-report${NC}       Generate comprehensive system report
  ${GREEN}metrics-export${NC}      Export system metrics
  ${GREEN}log-analysis${NC}        Analyze system logs for issues
  ${GREEN}capacity-planning${NC}   Generate capacity planning report

${CYAN}Examples:${NC}
  ./maintenance-toolkit.sh health-check
  ./maintenance-toolkit.sh performance-tune
  ./maintenance-toolkit.sh diagnose
  ./maintenance-toolkit.sh system-report
EOF
}

# System health check
health_check() {
    step "🏥 Running comprehensive system health check"
    
    local health_report="/tmp/xorb-health-$(date +%Y%m%d_%H%M%S).json"
    
    info "📋 Checking Kubernetes cluster health"
    local cluster_status="healthy"
    if ! kubectl cluster-info &>/dev/null; then
        cluster_status="unhealthy"
        warn "Kubernetes cluster connectivity issues"
    fi
    
    info "📦 Checking namespace and resources"
    local namespace_exists=$(kubectl get namespace "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    local total_pods=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    local running_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)
    local pending_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Pending --no-headers 2>/dev/null | wc -l)
    local failed_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Failed --no-headers 2>/dev/null | wc -l)
    
    info "🌐 Checking services and endpoints"
    local total_services=$(kubectl get services -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    local services_with_endpoints=0
    
    for service in $(kubectl get services -n "$NAMESPACE" -o name 2>/dev/null); do
        if kubectl get endpoints "${service#service/}" -n "$NAMESPACE" &>/dev/null; then
            services_with_endpoints=$((services_with_endpoints + 1))
        fi
    done
    
    info "⚡ Testing Redis connectivity"
    local redis_status="unknown"
    local redis_pod=$(kubectl get pods -n "$NAMESPACE" -l app=redis -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ -n "$redis_pod" ]; then
        if kubectl exec "$redis_pod" -n "$NAMESPACE" -- redis-cli ping 2>/dev/null | grep -q PONG; then
            redis_status="healthy"
        else
            redis_status="unhealthy"
        fi
    fi
    
    info "🤖 Testing orchestrator API"
    local api_status="unknown"
    local orch_pod=$(kubectl get pods -n "$NAMESPACE" -l app=xorb-orchestrator -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ -n "$orch_pod" ]; then
        if kubectl exec "$orch_pod" -n "$NAMESPACE" -- curl -s -f http://localhost:8080/health 2>/dev/null | grep -q healthy; then
            api_status="healthy"
        else
            api_status="unhealthy"
        fi
    fi
    
    info "💾 Checking persistent storage"
    local pvc_count=$(kubectl get pvc -n "$NAMESPACE" --no-headers 2>/dev/null | wc -l)
    local bound_pvc_count=$(kubectl get pvc -n "$NAMESPACE" --field-selector=status.phase=Bound --no-headers 2>/dev/null | wc -l)
    
    info "📊 Checking system resources"
    local node_count=$(kubectl get nodes --no-headers 2>/dev/null | wc -l)
    local ready_nodes=$(kubectl get nodes --field-selector=status.conditions[*].type=Ready,status.conditions[*].status=True --no-headers 2>/dev/null | wc -l)
    
    # Generate health report
    cat > "$health_report" << EOF
{
  "health_check": {
    "timestamp": "$(date -Iseconds)",
    "overall_status": "$([ "$cluster_status" = "healthy" ] && [ "$redis_status" = "healthy" ] && [ "$api_status" = "healthy" ] && echo "healthy" || echo "needs_attention")",
    "cluster": {
      "status": "$cluster_status",
      "nodes_total": $node_count,
      "nodes_ready": $ready_nodes
    },
    "namespace": {
      "exists": $([ "$namespace_exists" -gt 0 ] && echo "true" || echo "false"),
      "pods": {
        "total": $total_pods,
        "running": $running_pods,
        "pending": $pending_pods,
        "failed": $failed_pods
      },
      "services": {
        "total": $total_services,
        "with_endpoints": $services_with_endpoints
      }
    },
    "components": {
      "redis": "$redis_status",
      "orchestrator_api": "$api_status"
    },
    "storage": {
      "pvc_total": $pvc_count,
      "pvc_bound": $bound_pvc_count
    }
  }
}
EOF
    
    # Display summary
    echo ""
    echo -e "${WHITE}📊 Health Check Summary${NC}"
    echo "======================="
    echo -e "Overall Status: $([ "$cluster_status" = "healthy" ] && [ "$redis_status" = "healthy" ] && [ "$api_status" = "healthy" ] && echo "${GREEN}✅ Healthy${NC}" || echo "${YELLOW}⚠️ Needs Attention${NC}")"
    echo -e "Cluster: $([ "$cluster_status" = "healthy" ] && echo "${GREEN}✅${NC}" || echo "${RED}❌${NC}") $cluster_status"
    echo -e "Redis: $([ "$redis_status" = "healthy" ] && echo "${GREEN}✅${NC}" || echo "${RED}❌${NC}") $redis_status"
    echo -e "API: $([ "$api_status" = "healthy" ] && echo "${GREEN}✅${NC}" || echo "${RED}❌${NC}") $api_status"
    echo -e "Pods: ${running_pods}/${total_pods} running"
    echo -e "Services: ${services_with_endpoints}/${total_services} with endpoints"
    echo ""
    
    log "✅ Health check completed. Report: $health_report"
    echo "$health_report"
}

# Performance tuning
performance_tune() {
    step "⚡ Running performance optimization"
    
    info "🔧 Optimizing Redis configuration"
    local redis_pods=($(kubectl get pods -n "$NAMESPACE" -l app=redis -o jsonpath='{.items[*].metadata.name}'))
    
    for redis_pod in "${redis_pods[@]}"; do
        if [ -n "$redis_pod" ]; then
            info "📊 Optimizing Redis pod: $redis_pod"
            kubectl exec "$redis_pod" -n "$NAMESPACE" -- redis-cli CONFIG SET maxmemory-policy allkeys-lru || warn "Failed to set Redis policy"
            kubectl exec "$redis_pod" -n "$NAMESPACE" -- redis-cli CONFIG SET timeout 300 || warn "Failed to set Redis timeout"
            kubectl exec "$redis_pod" -n "$NAMESPACE" -- redis-cli CONFIG SET tcp-keepalive 60 || warn "Failed to set Redis keepalive"
        fi
    done
    
    info "🚀 Checking resource limits and requests"
    kubectl get pods -n "$NAMESPACE" -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[0].resources}{"\n"}{end}' | \
    while read -r pod_name resources; do
        if [ -n "$pod_name" ]; then
            info "📊 Pod $pod_name resources: $resources"
        fi
    done
    
    info "🔄 Restarting orchestrator pods for optimization"
    kubectl rollout restart deployment/xorb-autonomous-orchestrator -n "$NAMESPACE" || warn "Failed to restart orchestrator"
    
    info "⏳ Waiting for rollout to complete"
    kubectl rollout status deployment/xorb-autonomous-orchestrator -n "$NAMESPACE" --timeout=300s || warn "Rollout did not complete in time"
    
    log "✅ Performance tuning completed"
}

# Resource cleanup
resource_cleanup() {
    step "🧹 Cleaning up unused resources"
    
    info "🗑️ Removing completed jobs"
    kubectl delete jobs --field-selector=status.successful=1 -n "$NAMESPACE" || warn "No completed jobs to clean"
    
    info "🔄 Cleaning up failed pods"
    kubectl delete pods --field-selector=status.phase=Failed -n "$NAMESPACE" || warn "No failed pods to clean"
    
    info "📦 Cleaning up unused ConfigMaps and Secrets"
    # This is a cautious approach - only clean up specific temporary resources
    kubectl delete configmaps -l temporary=true -n "$NAMESPACE" || warn "No temporary ConfigMaps to clean"
    
    info "💽 Cleaning up unused PVCs"
    # Only remove PVCs that are not bound
    kubectl get pvc -n "$NAMESPACE" --field-selector=status.phase!=Bound -o name | \
    while read -r pvc; do
        if [ -n "$pvc" ]; then
            warn "Found unbound PVC: $pvc (manual review recommended)"
        fi
    done
    
    info "🏷️ Cleaning up dangling resources"
    kubectl delete events --field-selector reason=Failed -n "$NAMESPACE" || warn "No failed events to clean"
    
    log "✅ Resource cleanup completed"
}

# System diagnostics
diagnose() {
    step "🔍 Running comprehensive system diagnostics"
    
    local diag_report="/tmp/xorb-diagnostics-$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "XORB System Diagnostics Report"
        echo "=============================="
        echo "Generated: $(date)"
        echo ""
        
        echo "=== Cluster Information ==="
        kubectl cluster-info
        echo ""
        
        echo "=== Node Status ==="
        kubectl get nodes -o wide
        echo ""
        
        echo "=== Namespace Resources ==="
        kubectl get all -n "$NAMESPACE"
        echo ""
        
        echo "=== Pod Details ==="
        kubectl describe pods -n "$NAMESPACE"
        echo ""
        
        echo "=== Events ==="
        kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -20
        echo ""
        
        echo "=== Resource Usage ==="
        kubectl top pods -n "$NAMESPACE" 2>/dev/null || echo "Metrics server not available"
        echo ""
        
        echo "=== Storage ==="
        kubectl get pv,pvc -n "$NAMESPACE"
        echo ""
        
        echo "=== Network Policies ==="
        kubectl get networkpolicies -n "$NAMESPACE"
        echo ""
        
        echo "=== ConfigMaps and Secrets ==="
        kubectl get configmaps,secrets -n "$NAMESPACE"
        echo ""
        
    } > "$diag_report"
    
    log "✅ Diagnostics completed. Report: $diag_report"
    echo "$diag_report"
}

# Restart services
restart_services() {
    step "🔄 Restarting XORB services"
    
    info "🔄 Restarting orchestrator deployment"
    kubectl rollout restart deployment/xorb-autonomous-orchestrator -n "$NAMESPACE"
    
    info "🔄 Restarting Redis deployment"
    kubectl rollout restart deployment/redis -n "$NAMESPACE"
    
    info "⏳ Waiting for services to be ready"
    kubectl rollout status deployment/xorb-autonomous-orchestrator -n "$NAMESPACE" --timeout=300s
    kubectl rollout status deployment/redis -n "$NAMESPACE" --timeout=300s
    
    info "🏥 Running post-restart health check"
    sleep 30
    health_check >/dev/null
    
    log "✅ Service restart completed"
}

# Scale system resources
scale_up() {
    step "📈 Scaling up system resources"
    
    local current_replicas=$(kubectl get deployment xorb-autonomous-orchestrator -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
    local new_replicas=$((current_replicas + 1))
    
    info "📊 Current orchestrator replicas: $current_replicas"
    info "🚀 Scaling to: $new_replicas replicas"
    
    kubectl scale deployment xorb-autonomous-orchestrator --replicas="$new_replicas" -n "$NAMESPACE"
    
    info "⏳ Waiting for scale up to complete"
    kubectl rollout status deployment/xorb-autonomous-orchestrator -n "$NAMESPACE" --timeout=300s
    
    log "✅ Scale up completed to $new_replicas replicas"
}

scale_down() {
    step "📉 Scaling down system resources"
    
    local current_replicas=$(kubectl get deployment xorb-autonomous-orchestrator -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
    
    if [ "$current_replicas" -le 1 ]; then
        warn "Cannot scale below 1 replica for system stability"
        return 1
    fi
    
    local new_replicas=$((current_replicas - 1))
    
    info "📊 Current orchestrator replicas: $current_replicas"
    info "📉 Scaling to: $new_replicas replicas"
    
    kubectl scale deployment xorb-autonomous-orchestrator --replicas="$new_replicas" -n "$NAMESPACE"
    
    info "⏳ Waiting for scale down to complete"
    kubectl rollout status deployment/xorb-autonomous-orchestrator -n "$NAMESPACE" --timeout=300s
    
    log "✅ Scale down completed to $new_replicas replicas"
}

# Generate system report
system_report() {
    step "📋 Generating comprehensive system report"
    
    local report_file="/var/log/xorb/system-report-$(date +%Y%m%d_%H%M%S).json"
    mkdir -p "$(dirname "$report_file")"
    
    info "📊 Collecting system metrics"
    
    # Collect comprehensive system information
    local node_count=$(kubectl get nodes --no-headers | wc -l)
    local ready_nodes=$(kubectl get nodes --field-selector=status.conditions[*].type=Ready,status.conditions[*].status=True --no-headers | wc -l)
    local total_pods=$(kubectl get pods -n "$NAMESPACE" --no-headers | wc -l)
    local running_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Running --no-headers | wc -l)
    local total_services=$(kubectl get services -n "$NAMESPACE" --no-headers | wc -l)
    local pvc_count=$(kubectl get pvc -n "$NAMESPACE" --no-headers | wc -l)
    local configmap_count=$(kubectl get configmaps -n "$NAMESPACE" --no-headers | wc -l)
    local secret_count=$(kubectl get secrets -n "$NAMESPACE" --no-headers | wc -l)
    
    # Get resource usage if metrics server is available
    local cpu_usage="N/A"
    local memory_usage="N/A"
    if kubectl top pods -n "$NAMESPACE" &>/dev/null; then
        cpu_usage=$(kubectl top pods -n "$NAMESPACE" --no-headers | awk '{sum+=$2} END {print sum "m"}')
        memory_usage=$(kubectl top pods -n "$NAMESPACE" --no-headers | awk '{sum+=$3} END {print sum "Mi"}')
    fi
    
    # Generate report
    cat > "$report_file" << EOF
{
  "system_report": {
    "timestamp": "$(date -Iseconds)",
    "namespace": "$NAMESPACE",
    "cluster": {
      "nodes_total": $node_count,
      "nodes_ready": $ready_nodes,
      "kubernetes_version": "$(kubectl version --short --client=false | grep Server | awk '{print $3}' 2>/dev/null || echo 'unknown')"
    },
    "workloads": {
      "pods_total": $total_pods,
      "pods_running": $running_pods,
      "services": $total_services,
      "deployments": $(kubectl get deployments -n "$NAMESPACE" --no-headers | wc -l),
      "configmaps": $configmap_count,
      "secrets": $secret_count
    },
    "storage": {
      "persistent_volume_claims": $pvc_count,
      "persistent_volumes": $(kubectl get pv --no-headers | wc -l)
    },
    "resource_usage": {
      "cpu_total": "$cpu_usage",
      "memory_total": "$memory_usage"
    },
    "network": {
      "network_policies": $(kubectl get networkpolicies -n "$NAMESPACE" --no-headers | wc -l),
      "ingresses": $(kubectl get ingresses -n "$NAMESPACE" --no-headers | wc -l)
    },
    "monitoring": {
      "prometheus_rules": $(kubectl get prometheusrules -n "$NAMESPACE" --no-headers | wc -l),
      "service_monitors": $(kubectl get servicemonitors -n "$NAMESPACE" --no-headers | wc -l)
    }
  }
}
EOF
    
    log "✅ System report generated: $report_file"
    echo "$report_file"
}

# Network connectivity test
network_test() {
    step "🌐 Running network connectivity tests"
    
    info "🔍 Testing internal service connectivity"
    
    # Test Redis connectivity
    local redis_pod=$(kubectl get pods -n "$NAMESPACE" -l app=redis -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ -n "$redis_pod" ]; then
        if kubectl exec "$redis_pod" -n "$NAMESPACE" -- redis-cli ping 2>/dev/null | grep -q PONG; then
            echo -e "  ${GREEN}✅${NC} Redis connectivity: OK"
        else
            echo -e "  ${RED}❌${NC} Redis connectivity: FAILED"
        fi
    else
        echo -e "  ${YELLOW}⚠️${NC} Redis pod not found"
    fi
    
    # Test orchestrator API connectivity
    local orch_pod=$(kubectl get pods -n "$NAMESPACE" -l app=xorb-orchestrator -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [ -n "$orch_pod" ]; then
        if kubectl exec "$orch_pod" -n "$NAMESPACE" -- curl -s -f http://localhost:8080/health 2>/dev/null; then
            echo -e "  ${GREEN}✅${NC} Orchestrator API: OK"
        else
            echo -e "  ${RED}❌${NC} Orchestrator API: FAILED"
        fi
    else
        echo -e "  ${YELLOW}⚠️${NC} Orchestrator pod not found"
    fi
    
    # Test DNS resolution
    info "🔍 Testing DNS resolution"
    if [ -n "$orch_pod" ]; then
        if kubectl exec "$orch_pod" -n "$NAMESPACE" -- nslookup redis-service 2>/dev/null; then
            echo -e "  ${GREEN}✅${NC} DNS resolution: OK"
        else
            echo -e "  ${RED}❌${NC} DNS resolution: FAILED"
        fi
    fi
    
    # Test external connectivity
    info "🌍 Testing external connectivity"
    if [ -n "$orch_pod" ]; then
        if kubectl exec "$orch_pod" -n "$NAMESPACE" -- curl -s -m 5 https://www.google.com >/dev/null 2>&1; then
            echo -e "  ${GREEN}✅${NC} External connectivity: OK"
        else
            echo -e "  ${YELLOW}⚠️${NC} External connectivity: Limited or blocked"
        fi
    fi
    
    log "✅ Network connectivity tests completed"
}

# Main execution
main() {
    echo -e "${WHITE}🔧 XORB Maintenance Toolkit${NC}"
    echo -e "${WHITE}Namespace: $NAMESPACE${NC}"
    echo ""
    
    case "$OPERATION" in
        health-check)
            health_check
            ;;
        performance-tune)
            performance_tune
            ;;
        resource-cleanup)
            resource_cleanup
            ;;
        diagnose)
            diagnose
            ;;
        restart-services)
            restart_services
            ;;
        scale-up)
            scale_up
            ;;
        scale-down)
            scale_down
            ;;
        system-report)
            system_report
            ;;
        network-test)
            network_test
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            error "Unknown operation: $OPERATION"
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi