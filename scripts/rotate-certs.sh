#!/bin/bash
# Certificate Rotation Script for XORB Platform
# Automates certificate renewal and hot reload for zero-downtime operation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CA_SCRIPT_DIR="${SCRIPT_DIR}/ca"
SECRETS_DIR="${SCRIPT_DIR}/../secrets/tls"
LOG_DIR="${SCRIPT_DIR}/../logs/cert-rotation"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RENEWAL_THRESHOLD_DAYS=7  # Renew certificates expiring within 7 days
BACKUP_RETENTION_DAYS=30  # Keep certificate backups for 30 days
SERVICES_TO_RESTART=(
    "redis"
    "postgres"
    "temporal"
)
SERVICES_TO_RELOAD=(
    "envoy-api"
    "envoy-agent"
    "prometheus"
    "grafana"
)

log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message"
            ;;
        "DEBUG")
            if [[ "${VERBOSE:-false}" == "true" ]]; then
                echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message"
            fi
            ;;
    esac
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -s, --service NAME      Rotate certificate for specific service only
    -f, --force             Force rotation even if not near expiry
    -t, --threshold DAYS    Set renewal threshold in days (default: 7)
    -d, --dry-run          Show what would be rotated without making changes
    -v, --verbose          Verbose output
    -h, --help             Show this help message

EXAMPLES:
    $0                      # Rotate certificates nearing expiry
    $0 -s api              # Rotate API service certificate only
    $0 -f                  # Force rotation of all certificates
    $0 -d                  # Dry run to see what would be rotated
EOF
}

setup_logging() {
    mkdir -p "$LOG_DIR"
    local log_file="${LOG_DIR}/rotation-$(date +%Y%m%d-%H%M%S).log"
    exec > >(tee -a "$log_file")
    exec 2>&1
    log "INFO" "Certificate rotation started - Log: $log_file"
}

check_certificate_expiry() {
    local cert_file="$1"
    local service_name="$2"
    
    if [[ ! -f "$cert_file" ]]; then
        log "WARN" "Certificate not found: $cert_file"
        return 2
    fi
    
    local expire_date=$(openssl x509 -in "$cert_file" -noout -enddate | cut -d= -f2)
    local expire_timestamp=$(date -d "$expire_date" +%s)
    local current_timestamp=$(date +%s)
    local days_remaining=$(( (expire_timestamp - current_timestamp) / 86400 ))
    
    log "DEBUG" "$service_name certificate expires in $days_remaining days"
    
    if [[ $days_remaining -le $RENEWAL_THRESHOLD_DAYS ]]; then
        log "WARN" "$service_name certificate expires in $days_remaining days - renewal needed"
        return 0  # Needs renewal
    else
        log "INFO" "$service_name certificate valid for $days_remaining days"
        return 1  # No renewal needed
    fi
}

backup_certificate() {
    local service_name="$1"
    local service_dir="${SECRETS_DIR}/${service_name}"
    local backup_dir="${SECRETS_DIR}/backups/${service_name}/$(date +%Y%m%d-%H%M%S)"
    
    if [[ ! -d "$service_dir" ]]; then
        log "WARN" "Service directory not found: $service_dir"
        return 1
    fi
    
    log "INFO" "Backing up $service_name certificates"
    mkdir -p "$backup_dir"
    
    # Copy all certificate files
    for cert_file in "$service_dir"/*.pem "$service_dir"/*.p12; do
        if [[ -f "$cert_file" ]]; then
            cp "$cert_file" "$backup_dir/"
            log "DEBUG" "Backed up: $(basename "$cert_file")"
        fi
    done
    
    # Set proper permissions
    chmod -R 400 "$backup_dir"/*.pem "$backup_dir"/*.p12 2>/dev/null || true
    
    log "INFO" "Backup completed: $backup_dir"
    return 0
}

rotate_service_certificate() {
    local service_name="$1"
    local force_rotation="${2:-false}"
    local cert_file="${SECRETS_DIR}/${service_name}/cert.pem"
    
    log "INFO" "Checking certificate rotation for $service_name"
    
    # Check if rotation is needed
    if [[ "$force_rotation" != "true" ]]; then
        if ! check_certificate_expiry "$cert_file" "$service_name"; then
            local exit_code=$?
            if [[ $exit_code -eq 1 ]]; then
                log "INFO" "$service_name certificate rotation not needed"
                return 0
            elif [[ $exit_code -eq 2 ]]; then
                log "ERROR" "$service_name certificate file not found"
                return 1
            fi
        fi
    fi
    
    # Backup existing certificate
    if ! backup_certificate "$service_name"; then
        log "ERROR" "Failed to backup $service_name certificate"
        return 1
    fi
    
    # Determine certificate type based on service
    local cert_type="server"
    case "$service_name" in
        *-client|orchestrator|scanner)
            cert_type="client"
            ;;
        api|agent)
            cert_type="both"
            ;;
    esac
    
    # Generate new certificate
    log "INFO" "Generating new certificate for $service_name (type: $cert_type)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would generate new certificate for $service_name"
        return 0
    fi
    
    if "${CA_SCRIPT_DIR}/issue-cert.sh" "$service_name" "$cert_type"; then
        log "INFO" "✅ New certificate generated for $service_name"
        
        # Verify new certificate
        if verify_new_certificate "$service_name"; then
            log "INFO" "✅ New certificate verified for $service_name"
            return 0
        else
            log "ERROR" "❌ New certificate verification failed for $service_name"
            restore_certificate_backup "$service_name"
            return 1
        fi
    else
        log "ERROR" "❌ Failed to generate new certificate for $service_name"
        restore_certificate_backup "$service_name"
        return 1
    fi
}

verify_new_certificate() {
    local service_name="$1"
    local cert_file="${SECRETS_DIR}/${service_name}/cert.pem"
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    
    log "DEBUG" "Verifying new certificate for $service_name"
    
    # Basic certificate validation
    if ! openssl x509 -in "$cert_file" -noout -text >/dev/null 2>&1; then
        log "ERROR" "Invalid certificate format: $cert_file"
        return 1
    fi
    
    # Verify certificate chain
    if ! openssl verify -CAfile "$ca_file" "$cert_file" >/dev/null 2>&1; then
        log "ERROR" "Certificate chain verification failed: $cert_file"
        return 1
    fi
    
    # Check expiry
    local expire_date=$(openssl x509 -in "$cert_file" -noout -enddate | cut -d= -f2)
    local expire_timestamp=$(date -d "$expire_date" +%s)
    local current_timestamp=$(date +%s)
    local days_valid=$(( (expire_timestamp - current_timestamp) / 86400 ))
    
    if [[ $days_valid -lt 1 ]]; then
        log "ERROR" "New certificate is already expired or expires too soon"
        return 1
    fi
    
    log "DEBUG" "New certificate valid for $days_valid days"
    return 0
}

restore_certificate_backup() {
    local service_name="$1"
    local service_dir="${SECRETS_DIR}/${service_name}"
    local backup_base="${SECRETS_DIR}/backups/${service_name}"
    
    log "WARN" "Restoring certificate backup for $service_name"
    
    # Find latest backup
    local latest_backup=$(find "$backup_base" -maxdepth 1 -type d -name "????????-??????" | sort -r | head -1)
    
    if [[ -z "$latest_backup" ]]; then
        log "ERROR" "No backup found for $service_name"
        return 1
    fi
    
    log "INFO" "Restoring from backup: $latest_backup"
    
    # Restore files
    for backup_file in "$latest_backup"/*; do
        if [[ -f "$backup_file" ]]; then
            cp "$backup_file" "$service_dir/"
            log "DEBUG" "Restored: $(basename "$backup_file")"
        fi
    done
    
    log "INFO" "Certificate backup restored for $service_name"
    return 0
}

reload_service() {
    local service_name="$1"
    local reload_method="${2:-restart}"
    
    log "INFO" "Reloading $service_name ($reload_method)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would reload $service_name"
        return 0
    fi
    
    case "$reload_method" in
        "restart")
            if docker-compose restart "$service_name" >/dev/null 2>&1; then
                log "INFO" "✅ $service_name restarted successfully"
                
                # Wait for service to be healthy
                local retries=0
                while [[ $retries -lt 30 ]]; do
                    if docker-compose ps "$service_name" | grep -q "healthy\|Up"; then
                        log "INFO" "✅ $service_name is healthy after restart"
                        return 0
                    fi
                    sleep 2
                    retries=$((retries + 1))
                done
                
                log "WARN" "⚠️  $service_name restart completed but health check timed out"
                return 1
            else
                log "ERROR" "❌ Failed to restart $service_name"
                return 1
            fi
            ;;
        "reload")
            case "$service_name" in
                "envoy-"*)
                    # Envoy graceful reload via admin API
                    local admin_port
                    case "$service_name" in
                        "envoy-api") admin_port="9901" ;;
                        "envoy-agent") admin_port="9902" ;;
                    esac
                    
                    if curl -s -X POST "http://localhost:${admin_port}/reload" >/dev/null 2>&1; then
                        log "INFO" "✅ $service_name reloaded via admin API"
                        return 0
                    else
                        log "WARN" "⚠️  Failed to reload $service_name via admin API, restarting"
                        reload_service "$service_name" "restart"
                        return $?
                    fi
                    ;;
                *)
                    # Default to restart for other services
                    reload_service "$service_name" "restart"
                    return $?
                    ;;
            esac
            ;;
    esac
}

cleanup_old_backups() {
    local backup_dir="${SECRETS_DIR}/backups"
    
    log "INFO" "Cleaning up old certificate backups (older than $BACKUP_RETENTION_DAYS days)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "INFO" "[DRY RUN] Would clean up old backups"
        return 0
    fi
    
    local cleaned=0
    for service_backup_dir in "$backup_dir"/*; do
        if [[ -d "$service_backup_dir" ]]; then
            local service_name=$(basename "$service_backup_dir")
            
            # Find and remove old backups
            while IFS= read -r -d '' old_backup; do
                local backup_date=$(basename "$old_backup")
                local backup_timestamp=$(date -d "${backup_date:0:8} ${backup_date:9:2}:${backup_date:11:2}:${backup_date:13:2}" +%s 2>/dev/null || continue)
                local cutoff_timestamp=$(date -d "$BACKUP_RETENTION_DAYS days ago" +%s)
                
                if [[ $backup_timestamp -lt $cutoff_timestamp ]]; then
                    rm -rf "$old_backup"
                    log "DEBUG" "Removed old backup: $old_backup"
                    cleaned=$((cleaned + 1))
                fi
            done < <(find "$service_backup_dir" -maxdepth 1 -type d -name "????????-??????" -print0)
        fi
    done
    
    if [[ $cleaned -gt 0 ]]; then
        log "INFO" "Cleaned up $cleaned old certificate backups"
    else
        log "INFO" "No old backups to clean up"
    fi
}

generate_rotation_report() {
    local rotation_summary="$1"
    local report_file="${LOG_DIR}/rotation-summary-$(date +%Y%m%d-%H%M%S).html"
    
    log "INFO" "Generating certificate rotation report"
    
    cat > "$report_file" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Certificate Rotation Report - XORB Platform</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #8e44ad; color: white; padding: 20px; border-radius: 5px; }
        .section { background: #ecf0f1; margin: 20px 0; padding: 15px; border-radius: 5px; }
        .success { color: #27ae60; font-weight: bold; }
        .warning { color: #f39c12; font-weight: bold; }
        .error { color: #e74c3c; font-weight: bold; }
        .details { margin-top: 10px; font-family: monospace; background: #2c3e50; color: #ecf0f1; padding: 10px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔄 Certificate Rotation Report</h1>
        <p>Generated: $(date)</p>
        <p>XORB Platform TLS Certificate Management</p>
    </div>
    
    <div class="section">
        <h2>📊 Rotation Summary</h2>
        <div class="details">
EOF

    echo "$rotation_summary" | sed 's/</\&lt;/g; s/>/\&gt;/g' >> "$report_file"
    
    cat >> "$report_file" << 'EOF'
        </div>
    </div>
    
    <div class="section">
        <h2>🔄 Next Steps</h2>
        <ul>
            <li><strong>Validation:</strong> Run TLS validation scripts to verify new certificates</li>
            <li><strong>Monitoring:</strong> Monitor service health after certificate rotation</li>
            <li><strong>Backup:</strong> Verify certificate backups are stored securely</li>
            <li><strong>Schedule:</strong> Plan next rotation based on certificate expiry dates</li>
        </ul>
    </div>
</body>
</html>
EOF

    log "INFO" "Rotation report generated: $report_file"
}

# Parse command line arguments
SPECIFIC_SERVICE=""
FORCE_ROTATION=false
DRY_RUN=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--service)
            SPECIFIC_SERVICE="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_ROTATION=true
            shift
            ;;
        -t|--threshold)
            RENEWAL_THRESHOLD_DAYS="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1" >&2
            usage
            exit 1
            ;;
    esac
done

# Initialize
setup_logging

log "INFO" "🔄 Starting XORB Platform certificate rotation"
log "INFO" "Renewal threshold: $RENEWAL_THRESHOLD_DAYS days"
log "INFO" "Force rotation: $FORCE_ROTATION"
log "INFO" "Dry run: $DRY_RUN"

# Discover services
services_to_rotate=()
if [[ -n "$SPECIFIC_SERVICE" ]]; then
    services_to_rotate=("$SPECIFIC_SERVICE")
    log "INFO" "Rotating certificate for specific service: $SPECIFIC_SERVICE"
else
    # Auto-discover services from secrets directory
    for service_dir in "$SECRETS_DIR"/*; do
        if [[ -d "$service_dir" && -f "$service_dir/cert.pem" ]]; then
            local service_name=$(basename "$service_dir")
            if [[ "$service_name" != "ca" && "$service_name" != "backups" ]]; then
                services_to_rotate+=("$service_name")
            fi
        fi
    done
    log "INFO" "Auto-discovered ${#services_to_rotate[@]} services for rotation"
fi

# Perform certificate rotation
rotation_results=()
failed_rotations=0
successful_rotations=0

for service_name in "${services_to_rotate[@]}"; do
    log "INFO" "Processing $service_name..."
    
    if rotate_service_certificate "$service_name" "$FORCE_ROTATION"; then
        rotation_results+=("$service_name:SUCCESS")
        successful_rotations=$((successful_rotations + 1))
        
        # Reload service if certificate was rotated
        if [[ "$DRY_RUN" != "true" ]]; then
            # Determine reload method
            local reload_method="restart"
            for reload_service in "${SERVICES_TO_RELOAD[@]}"; do
                if [[ "$service_name" == "$reload_service" ]]; then
                    reload_method="reload"
                    break
                fi
            done
            
            if reload_service "$service_name" "$reload_method"; then
                log "INFO" "✅ $service_name certificate rotation and reload completed"
            else
                log "WARN" "⚠️  $service_name certificate rotated but reload failed"
            fi
        fi
    else
        rotation_results+=("$service_name:FAILED")
        failed_rotations=$((failed_rotations + 1))
        log "ERROR" "❌ $service_name certificate rotation failed"
    fi
    
    echo "---"
done

# Cleanup old backups
cleanup_old_backups

# Generate summary
rotation_summary=$(cat << EOF
Certificate Rotation Summary
===========================
Total Services: ${#services_to_rotate[@]}
Successful: $successful_rotations
Failed: $failed_rotations
Success Rate: $(( successful_rotations * 100 / ${#services_to_rotate[@]} ))%

Service Results:
$(printf '%s\n' "${rotation_results[@]}")

Configuration:
- Renewal Threshold: $RENEWAL_THRESHOLD_DAYS days
- Force Rotation: $FORCE_ROTATION
- Dry Run: $DRY_RUN
- Backup Retention: $BACKUP_RETENTION_DAYS days
EOF
)

echo "$rotation_summary"
generate_rotation_report "$rotation_summary"

# Final status
if [[ $failed_rotations -eq 0 ]]; then
    log "INFO" "🎉 Certificate rotation completed successfully!"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        log "INFO" "💡 Recommendation: Run TLS validation scripts to verify rotated certificates"
    fi
    
    exit 0
else
    log "ERROR" "💥 Certificate rotation completed with $failed_rotations failure(s)"
    exit 1
fi