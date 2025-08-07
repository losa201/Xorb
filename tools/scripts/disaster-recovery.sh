#!/bin/bash
set -euo pipefail

# ===================================================================
# XORB Autonomous Orchestrator - Disaster Recovery Script
# Complete backup, restore, and disaster recovery automation
# ===================================================================

# Configuration
NAMESPACE="${NAMESPACE:-xorb-system}"
BACKUP_STORAGE="${BACKUP_STORAGE:-/var/backups/xorb}"
S3_BUCKET="${S3_BUCKET:-}"
RESTORE_POINT="${RESTORE_POINT:-}"
OPERATION="${1:-help}"

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
log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅${NC} $1"; }
info() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] ℹ️${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️${NC} $1"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌${NC} $1"; }
step() { echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] 🚀${NC} $1"; }

# Usage information
show_usage() {
    cat << EOF
${WHITE}XORB Disaster Recovery Tool${NC}

${CYAN}Usage:${NC}
  ./disaster-recovery.sh <command> [options]

${CYAN}Commands:${NC}
  ${GREEN}backup${NC}          Create full system backup
  ${GREEN}restore${NC}         Restore from backup
  ${GREEN}list-backups${NC}    List available backups
  ${GREEN}verify-backup${NC}   Verify backup integrity
  ${GREEN}disaster-init${NC}   Initialize disaster recovery
  ${GREEN}health-check${NC}    Post-recovery health check

${CYAN}Environment Variables:${NC}
  NAMESPACE         Kubernetes namespace (default: xorb-system)
  BACKUP_STORAGE    Local backup storage path
  S3_BUCKET         S3 bucket for remote backups
  RESTORE_POINT     Specific backup to restore from

${CYAN}Examples:${NC}
  ./disaster-recovery.sh backup
  ./disaster-recovery.sh restore
  RESTORE_POINT=backup-20250129-120000 ./disaster-recovery.sh restore
  ./disaster-recovery.sh list-backups
EOF
}

# Validate environment
validate_environment() {
    step "🔍 Validating disaster recovery environment"
    
    # Check kubectl access
    if ! kubectl cluster-info &>/dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
        error "Namespace $NAMESPACE does not exist"
        exit 1
    fi
    
    # Create backup storage directory
    mkdir -p "$BACKUP_STORAGE"
    
    # Check required tools
    local required_tools=("kubectl" "tar" "gzip")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &>/dev/null; then
            error "Required tool '$tool' not found"
            exit 1
        fi
    done
    
    log "✅ Environment validation completed"
}

# Create comprehensive backup
create_backup() {
    step "💾 Creating comprehensive XORB backup"
    
    local backup_id="backup-$(date +%Y%m%d-%H%M%S)"
    local backup_dir="$BACKUP_STORAGE/$backup_id"
    mkdir -p "$backup_dir"
    
    info "📦 Backup ID: $backup_id"
    info "📁 Backup Directory: $backup_dir"
    
    # Create backup metadata
    info "📋 Creating backup metadata"
    cat > "$backup_dir/metadata.json" << EOF
{
  "backup_id": "$backup_id",
  "timestamp": "$(date -Iseconds)",
  "namespace": "$NAMESPACE",
  "kubernetes_version": "$(kubectl version --short --client=false | grep Server | awk '{print $3}')",
  "node_count": $(kubectl get nodes --no-headers | wc -l),
  "backup_type": "disaster_recovery",
  "components": ["kubernetes", "redis", "configurations", "secrets"]
}
EOF
    
    # Backup Kubernetes resources
    info "☸️ Backing up Kubernetes resources"
    kubectl get all,configmaps,secrets,pvc,networkpolicies,prometheusrules -n "$NAMESPACE" -o yaml > "$backup_dir/kubernetes-resources.yaml"
    
    # Backup cluster-wide resources
    info "🌐 Backing up cluster resources"
    kubectl get clusterroles,clusterrolebindings -o yaml | grep -A 1000 "xorb" > "$backup_dir/cluster-resources.yaml" || true
    
    # Backup Redis data
    info "⚡ Backing up Redis data"
    local redis_pods=($(kubectl get pods -n "$NAMESPACE" -l app=redis -o jsonpath='{.items[*].metadata.name}'))
    
    for pod in "${redis_pods[@]}"; do
        if [ -n "$pod" ]; then
            info "📊 Backing up Redis pod: $pod"
            kubectl exec "$pod" -n "$NAMESPACE" -- redis-cli BGSAVE
            sleep 5
            kubectl exec "$pod" -n "$NAMESPACE" -- tar -czf - /data > "$backup_dir/redis-$pod.tar.gz"
        fi
    done
    
    # Backup persistent volumes
    info "💽 Backing up persistent volume data"
    local pvcs=($(kubectl get pvc -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}'))
    
    for pvc in "${pvcs[@]}"; do
        if [ -n "$pvc" ]; then
            info "📦 Creating PVC snapshot: $pvc"
            kubectl run "pvc-backup-$pvc" --rm -i --tty=false --restart=Never \
                --image=alpine:latest --namespace="$NAMESPACE" \
                --overrides="{
                  \"spec\": {
                    \"containers\": [{
                      \"name\": \"backup\",
                      \"image\": \"alpine:latest\",
                      \"command\": [\"tar\", \"-czf\", \"-\", \"/data\"],
                      \"volumeMounts\": [{
                        \"name\": \"data\",
                        \"mountPath\": \"/data\"
                      }]
                    }],
                    \"volumes\": [{
                      \"name\": \"data\",
                      \"persistentVolumeClaim\": {
                        \"claimName\": \"$pvc\"
                      }
                    }],
                    \"restartPolicy\": \"Never\"
                  }
                }" > "$backup_dir/pvc-$pvc.tar.gz" 2>/dev/null || warn "Failed to backup PVC: $pvc"
        fi
    done
    
    # Backup application configuration
    info "⚙️ Backing up application configurations"
    if [ -d "/opt/xorb" ]; then
        tar -czf "$backup_dir/application-config.tar.gz" -C /opt/xorb . 2>/dev/null || warn "No application config found"
    fi
    
    # Backup TLS certificates
    info "🔐 Backing up TLS certificates"
    kubectl get certificates,issuers,clusterissuers -n "$NAMESPACE" -o yaml > "$backup_dir/tls-certificates.yaml" 2>/dev/null || warn "No certificates found"
    
    # Create backup verification file
    info "🔍 Creating backup verification"
    cat > "$backup_dir/verification.json" << EOF
{
  "files": {
    "metadata": "$(sha256sum "$backup_dir/metadata.json" | awk '{print $1}' 2>/dev/null || echo 'missing')",
    "kubernetes_resources": "$(sha256sum "$backup_dir/kubernetes-resources.yaml" | awk '{print $1}' 2>/dev/null || echo 'missing')",
    "cluster_resources": "$(sha256sum "$backup_dir/cluster-resources.yaml" | awk '{print $1}' 2>/dev/null || echo 'missing')",
    "tls_certificates": "$(sha256sum "$backup_dir/tls-certificates.yaml" | awk '{print $1}' 2>/dev/null || echo 'missing')"
  },
  "redis_backups": $(ls "$backup_dir"/redis-*.tar.gz 2>/dev/null | wc -l),
  "pvc_backups": $(ls "$backup_dir"/pvc-*.tar.gz 2>/dev/null | wc -l),
  "total_size_bytes": $(du -sb "$backup_dir" | awk '{print $1}'),
  "backup_completed": "$(date -Iseconds)"
}
EOF
    
    # Create compressed archive
    info "📦 Creating backup archive"
    tar -czf "$BACKUP_STORAGE/${backup_id}.tar.gz" -C "$BACKUP_STORAGE" "$backup_id"
    
    # Upload to S3 if configured
    if [ -n "$S3_BUCKET" ] && command -v aws &>/dev/null; then
        info "☁️ Uploading to S3: $S3_BUCKET"
        aws s3 cp "$BACKUP_STORAGE/${backup_id}.tar.gz" "s3://$S3_BUCKET/xorb-backups/" || warn "S3 upload failed"
    fi
    
    # Cleanup temporary directory
    rm -rf "$backup_dir"
    
    log "✅ Backup completed: ${backup_id}.tar.gz ($(du -sh "$BACKUP_STORAGE/${backup_id}.tar.gz" | awk '{print $1}'))"
    echo "$backup_id"
}

# List available backups
list_backups() {
    step "📋 Listing available backups"
    
    echo ""
    echo -e "${WHITE}Local Backups:${NC}"
    echo "=============="
    
    if ls "$BACKUP_STORAGE"/*.tar.gz &>/dev/null; then
        for backup in "$BACKUP_STORAGE"/*.tar.gz; do
            local backup_name=$(basename "$backup" .tar.gz)
            local backup_size=$(du -sh "$backup" | awk '{print $1}')
            local backup_date=$(echo "$backup_name" | grep -o '[0-9]\{8\}-[0-9]\{6\}' | sed 's/-/ /')
            echo -e "${GREEN}📦${NC} $backup_name (${backup_size}) - $backup_date"
        done
    else
        echo "No local backups found"
    fi
    
    # List S3 backups if configured
    if [ -n "$S3_BUCKET" ] && command -v aws &>/dev/null; then
        echo ""
        echo -e "${WHITE}S3 Backups:${NC}"
        echo "==========="
        aws s3 ls "s3://$S3_BUCKET/xorb-backups/" --human-readable --summarize 2>/dev/null || echo "Cannot access S3 or no backups found"
    fi
    
    echo ""
}

# Verify backup integrity
verify_backup() {
    local backup_name="${1:-}"
    
    if [ -z "$backup_name" ]; then
        error "Please specify backup name to verify"
        exit 1
    fi
    
    step "🔍 Verifying backup: $backup_name"
    
    local backup_file="$BACKUP_STORAGE/${backup_name}.tar.gz"
    
    if [ ! -f "$backup_file" ]; then
        error "Backup file not found: $backup_file"
        exit 1
    fi
    
    # Extract to temporary directory
    local temp_dir=$(mktemp -d)
    info "📦 Extracting backup for verification"
    tar -xzf "$backup_file" -C "$temp_dir"
    
    local backup_dir="$temp_dir/$backup_name"
    
    # Verify essential files
    info "📋 Checking essential files"
    local essential_files=("metadata.json" "kubernetes-resources.yaml" "verification.json")
    
    for file in "${essential_files[@]}"; do
        if [ -f "$backup_dir/$file" ]; then
            echo -e "  ${GREEN}✅${NC} $file"
        else
            echo -e "${RED}❌${NC} $file - MISSING"
            error "Essential file missing: $file"
            rm -rf "$temp_dir"
            exit 1
        fi
    done
    
    # Verify checksums if verification file exists
    if [ -f "$backup_dir/verification.json" ]; then
        info "🔐 Verifying checksums"
        
        # Check metadata checksum
        local expected_hash=$(jq -r '.files.metadata' "$backup_dir/verification.json")
        local actual_hash=$(sha256sum "$backup_dir/metadata.json" | awk '{print $1}')
        
        if [ "$expected_hash" = "$actual_hash" ]; then
            echo -e "  ${GREEN}✅${NC} Metadata checksum verified"
        else
            echo -e "  ${RED}❌${NC} Metadata checksum mismatch"
            warn "Expected: $expected_hash"
            warn "Actual: $actual_hash"
        fi
    fi
    
    # Verify Kubernetes resources
    info "☸️ Validating Kubernetes resources"
    if kubectl apply --dry-run=client -f "$backup_dir/kubernetes-resources.yaml" &>/dev/null; then
        echo -e "  ${GREEN}✅${NC} Kubernetes resources are valid"
    else
        echo -e "  ${YELLOW}⚠️${NC} Some Kubernetes resources may be invalid"
    fi
    
    # Check backup completeness
    info "📊 Backup completeness check"
    local redis_backups=$(ls "$backup_dir"/redis-*.tar.gz 2>/dev/null | wc -l)
    local pvc_backups=$(ls "$backup_dir"/pvc-*.tar.gz 2>/dev/null | wc -l)
    
    echo -e "  ${BLUE}📊${NC} Redis backups: $redis_backups"
    echo -e "  ${BLUE}💽${NC} PVC backups: $pvc_backups"
    
    # Cleanup
    rm -rf "$temp_dir"
    
    log "✅ Backup verification completed: $backup_name"
}

# Restore from backup
restore_backup() {
    local backup_name="${RESTORE_POINT:-}"
    
    if [ -z "$backup_name" ]; then
        # List available backups and prompt user
        echo -e "${YELLOW}Available backups:${NC}"
        list_backups
        echo ""
        read -p "Enter backup name to restore (without .tar.gz): " backup_name
    fi
    
    if [ -z "$backup_name" ]; then
        error "No backup specified"
        exit 1
    fi
    
    step "🔄 Restoring from backup: $backup_name"
    
    local backup_file="$BACKUP_STORAGE/${backup_name}.tar.gz"
    
    # Download from S3 if not local
    if [ ! -f "$backup_file" ] && [ -n "$S3_BUCKET" ] && command -v aws &>/dev/null; then
        info "☁️ Downloading backup from S3"
        aws s3 cp "s3://$S3_BUCKET/xorb-backups/${backup_name}.tar.gz" "$backup_file" || {
            error "Failed to download backup from S3"
            exit 1
        }
    fi
    
    if [ ! -f "$backup_file" ]; then
        error "Backup file not found: $backup_file"
        exit 1
    fi
    
    # Verify backup before restore
    verify_backup "$backup_name"
    
    # Extract backup
    local temp_dir=$(mktemp -d)
    info "📦 Extracting backup"
    tar -xzf "$backup_file" -C "$temp_dir"
    local backup_dir="$temp_dir/$backup_name"
    
    # Confirmation prompt
    echo ""
    warn "⚠️  WARNING: This will completely replace the current XORB deployment!"
    warn "Current namespace '$NAMESPACE' will be deleted and recreated."
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        error "Restore aborted by user"
        rm -rf "$temp_dir"
        exit 1
    fi
    
    # Create pre-restore backup
    info "💾 Creating pre-restore backup"
    local pre_restore_backup=$(create_backup)
    info "🔄 Pre-restore backup created: $pre_restore_backup"
    
    # Delete current deployment
    step "🗑️ Removing current deployment"
    kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
    
    # Wait for namespace deletion
    info "⏳ Waiting for namespace deletion"
    while kubectl get namespace "$NAMESPACE" &>/dev/null; do
        sleep 5
        echo -n "."
    done
    echo ""
    
    # Recreate namespace
    info "🏗️ Recreating namespace"
    kubectl create namespace "$NAMESPACE"
    kubectl label namespace "$NAMESPACE" name="$NAMESPACE"
    
    # Restore cluster resources first
    if [ -f "$backup_dir/cluster-resources.yaml" ]; then
        info "🌐 Restoring cluster resources"
        kubectl apply -f "$backup_dir/cluster-resources.yaml" || warn "Some cluster resources failed to restore"
    fi
    
    # Restore Kubernetes resources
    info "☸️ Restoring Kubernetes resources"
    kubectl apply -f "$backup_dir/kubernetes-resources.yaml"
    
    # Restore TLS certificates
    if [ -f "$backup_dir/tls-certificates.yaml" ]; then
        info "🔐 Restoring TLS certificates"
        kubectl apply -f "$backup_dir/tls-certificates.yaml" || warn "Some certificates failed to restore"
    fi
    
    # Wait for pods to be ready
    info "⏳ Waiting for pods to be ready"
    kubectl wait --for=condition=ready pod -l app=redis -n "$NAMESPACE" --timeout=300s || warn "Redis pods not ready"
    kubectl wait --for=condition=ready pod -l app=xorb-orchestrator -n "$NAMESPACE" --timeout=300s || warn "Orchestrator pods not ready"
    
    # Restore Redis data
    info "⚡ Restoring Redis data"
    local redis_backups=($(ls "$backup_dir"/redis-*.tar.gz 2>/dev/null))
    
    for redis_backup in "${redis_backups[@]}"; do
        if [ -f "$redis_backup" ]; then
            local pod_name=$(basename "$redis_backup" .tar.gz | sed 's/redis-//')
            info "📊 Restoring Redis data for pod: $pod_name"
            
            # Find current Redis pod
            local current_redis_pod=$(kubectl get pods -n "$NAMESPACE" -l app=redis -o jsonpath='{.items[0].metadata.name}')
            
            if [ -n "$current_redis_pod" ]; then
                kubectl exec "$current_redis_pod" -n "$NAMESPACE" -- tar -xzf - -C / < "$redis_backup" || warn "Failed to restore Redis data"
                kubectl exec "$current_redis_pod" -n "$NAMESPACE" -- redis-cli DEBUG RESTART || warn "Failed to restart Redis"
            fi
        fi
    done
    
    # Cleanup
    rm -rf "$temp_dir"
    
    log "✅ Restore completed from backup: $backup_name"
    warn "📋 Pre-restore backup available: $pre_restore_backup"
    
    # Run health check
    health_check
}

# Initialize disaster recovery
initialize_disaster_recovery() {
    step "🚨 Initializing disaster recovery"
    
    info "📋 Creating disaster recovery namespace"
    kubectl create namespace xorb-disaster-recovery --dry-run=client -o yaml | kubectl apply -f -
    
    # Create disaster recovery configmap
    info "⚙️ Creating disaster recovery configuration"
    kubectl create configmap disaster-recovery-config \
        --namespace=xorb-disaster-recovery \
        --from-literal=backup-schedule="0 */6 * * *" \
        --from-literal=retention-days="30" \
        --from-literal=s3-bucket="$S3_BUCKET" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log "✅ Disaster recovery initialized"
}

# Post-recovery health check
health_check() {
    step "🏥 Running post-recovery health check"
    
    # Check namespace
    info "📦 Checking namespace"
    if kubectl get namespace "$NAMESPACE" &>/dev/null; then
        echo -e "  ${GREEN}✅${NC} Namespace exists"
    else
        echo -e "  ${RED}❌${NC} Namespace missing"
        return 1
    fi
    
    # Check pods
    info "🏃 Checking pod status"
    local total_pods=$(kubectl get pods -n "$NAMESPACE" --no-headers | wc -l)
    local running_pods=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=Running --no-headers | wc -l)
    
    echo -e "  ${BLUE}📊${NC} Pods: $running_pods/$total_pods running"
    
    if [ "$running_pods" -eq "$total_pods" ] && [ "$total_pods" -gt 0 ]; then
        echo -e "  ${GREEN}✅${NC} All pods running"
    else
        echo -e "  ${YELLOW}⚠️${NC} Some pods not running"
        kubectl get pods -n "$NAMESPACE" | grep -v Running || true
    fi
    
    # Check services
    info "🌐 Checking services"
    local services=$(kubectl get services -n "$NAMESPACE" --no-headers | wc -l)
    echo -e "${BLUE}📊${NC} Services: $services active"
    
    if [ "$services" -gt 0 ]; then
        echo -e "  ${GREEN}✅${NC} Services available"
    else
        echo -e "  ${RED}❌${NC} No services found"
    fi
    
    # Test Redis connectivity
    info "⚡ Testing Redis connectivity"
    local redis_pod=$(kubectl get pods -n "$NAMESPACE" -l app=redis -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    
    if [ -n "$redis_pod" ]; then
        if kubectl exec "$redis_pod" -n "$NAMESPACE" -- redis-cli ping 2>/dev/null | grep -q PONG; then
            echo -e "  ${GREEN}✅${NC} Redis responding"
        else
            echo -e "  ${RED}❌${NC} Redis not responding"
        fi
    else
        echo -e "  ${YELLOW}⚠️${NC} No Redis pods found"
    fi
    
    # Test orchestrator API
    info "🤖 Testing orchestrator API"
    local orch_pod=$(kubectl get pods -n "$NAMESPACE" -l app=xorb-orchestrator -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    
    if [ -n "$orch_pod" ]; then
        if kubectl exec "$orch_pod" -n "$NAMESPACE" -- curl -s -f http://localhost:8080/health 2>/dev/null | grep -q healthy; then
            echo -e "  ${GREEN}✅${NC} Orchestrator API healthy"
        else
            echo -e "  ${RED}❌${NC} Orchestrator API not responding"
        fi
    else
        echo -e "  ${YELLOW}⚠️${NC} No orchestrator pods found"
    fi
    
    log "✅ Health check completed"
}

# Main execution
main() {
    case "$OPERATION" in
        backup)
            validate_environment
            create_backup
            ;;
        restore)
            validate_environment
            restore_backup
            ;;
        list-backups|list)
            list_backups
            ;;
        verify-backup|verify)
            validate_environment
            verify_backup "${2:-}"
            ;;
        disaster-init)
            validate_environment
            initialize_disaster_recovery
            ;;
        health-check)
            health_check
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