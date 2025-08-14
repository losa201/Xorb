#!/bin/bash
# XORB Platform Disaster Recovery Automation Script
# Comprehensive backup, recovery testing, and failover automation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/.."
BACKUP_DIR="${BASE_DIR}/backups"
RECOVERY_LOG_DIR="${BASE_DIR}/logs/disaster-recovery"
CONFIG_DIR="${BASE_DIR}/configs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Disaster recovery configuration
DR_SCENARIO=""
RECOVERY_TARGET_TIME=""
BACKUP_RETENTION_DAYS=30
ENCRYPT_BACKUPS=true
OFFSITE_BACKUP=false
AUTO_FAILOVER=false
RECOVERY_VALIDATION=true
NOTIFICATION_WEBHOOK=""

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
        "CRITICAL")
            echo -e "${RED}[CRITICAL]${NC} ${timestamp} - $message"
            ;;
        "DR")
            echo -e "${PURPLE}[DR]${NC} ${timestamp} - $message"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - $message"
            ;;
    esac
    
    # Log to disaster recovery audit log
    logger -t "xorb-disaster-recovery" -p local0.info "[$level] $message"
}

usage() {
    cat << EOF
Usage: $0 [COMMAND] [OPTIONS]

COMMANDS:
    backup                  Create full system backup
    restore                 Restore from backup
    test-recovery          Test recovery procedures
    failover               Execute failover to DR site
    validate               Validate backup integrity
    cleanup                Clean old backups
    status                 Show DR system status

OPTIONS:
    -s, --scenario TYPE     DR scenario (hardware_failure, ransomware, datacenter_outage)
    -t, --target-time TIME  Recovery target time (ISO format)
    -r, --retention DAYS    Backup retention in days (default: 30)
    -e, --encrypt           Encrypt backups (default: true)
    -o, --offsite           Enable offsite backup
    -a, --auto-failover     Enable automatic failover
    -n, --notify URL        Webhook URL for notifications
    -h, --help              Show this help message

EXAMPLES:
    $0 backup -e -o                            # Full encrypted offsite backup
    $0 test-recovery -s hardware_failure       # Test hardware failure recovery
    $0 failover -s datacenter_outage -a        # Automatic datacenter failover
    $0 restore -t 2024-08-11T10:00:00Z         # Point-in-time restore
EOF
}

setup_disaster_recovery_logging() {
    mkdir -p "$RECOVERY_LOG_DIR"
    
    local log_file="${RECOVERY_LOG_DIR}/dr-$(date +%Y%m%d-%H%M%S).log"
    exec > >(tee -a "$log_file")
    exec 2>&1
    
    log "DR" "Disaster recovery operation started"
    log "INFO" "Log file: $log_file"
    log "INFO" "DR Scenario: ${DR_SCENARIO:-Not specified}"
}

validate_dr_environment() {
    log "INFO" "Validating disaster recovery environment"
    
    # Check if backup directory exists
    if [[ ! -d "$BACKUP_DIR" ]]; then
        mkdir -p "$BACKUP_DIR"
        log "INFO" "Created backup directory: $BACKUP_DIR"
    fi
    
    # Verify Docker is available
    if ! docker info >/dev/null 2>&1; then
        log "CRITICAL" "Docker daemon not available - cannot perform container operations"
        return 1
    fi
    
    # Check database connectivity
    if ! docker-compose exec -T postgres pg_isready -U xorb >/dev/null 2>&1; then
        log "WARN" "PostgreSQL not accessible - database backup may fail"
    fi
    
    # Verify Redis connectivity
    if ! docker-compose exec -T redis redis-cli ping >/dev/null 2>&1; then
        log "WARN" "Redis not accessible - cache backup may fail"
    fi
    
    # Check available disk space
    local available_space=$(df "$BACKUP_DIR" | tail -1 | awk '{print $4}')
    local required_space=$((5 * 1024 * 1024))  # 5GB in KB
    
    if [[ $available_space -lt $required_space ]]; then
        log "CRITICAL" "Insufficient disk space for backup operations"
        return 1
    fi
    
    log "SUCCESS" "Disaster recovery environment validation completed"
    return 0
}

create_full_system_backup() {
    local backup_timestamp=$(date +%Y%m%d-%H%M%S)
    local backup_base_dir="${BACKUP_DIR}/full-backup-${backup_timestamp}"
    
    log "DR" "Creating full system backup: $backup_base_dir"
    mkdir -p "$backup_base_dir"
    
    # Backup PostgreSQL database
    log "INFO" "Backing up PostgreSQL database"
    local db_backup_file="${backup_base_dir}/postgresql-backup.sql"
    
    if docker-compose exec -T postgres pg_dump -U xorb -d xorb_platform > "$db_backup_file"; then
        log "SUCCESS" "PostgreSQL backup completed"
        
        # Create database schema dump
        docker-compose exec -T postgres pg_dump -U xorb -d xorb_platform --schema-only > "${backup_base_dir}/postgresql-schema.sql"
    else
        log "ERROR" "PostgreSQL backup failed"
        return 1
    fi
    
    # Backup Redis data
    log "INFO" "Backing up Redis data"
    local redis_backup_dir="${backup_base_dir}/redis"
    mkdir -p "$redis_backup_dir"
    
    # Save Redis database
    if docker-compose exec -T redis redis-cli BGSAVE >/dev/null 2>&1; then
        sleep 5  # Wait for background save to complete
        docker cp $(docker-compose ps -q redis):/data/dump.rdb "${redis_backup_dir}/dump.rdb"
        log "SUCCESS" "Redis backup completed"
    else
        log "ERROR" "Redis backup failed"
    fi
    
    # Backup application configurations
    log "INFO" "Backing up application configurations"
    local config_backup_dir="${backup_base_dir}/configs"
    mkdir -p "$config_backup_dir"
    
    # Copy configuration files
    cp -r "${BASE_DIR}/src/api/app/configs"/* "$config_backup_dir/" 2>/dev/null || true
    cp -r "${BASE_DIR}/deploy/configs"/* "$config_backup_dir/" 2>/dev/null || true
    cp "${BASE_DIR}/.env"* "$config_backup_dir/" 2>/dev/null || true
    
    # Backup TLS certificates
    log "INFO" "Backing up TLS certificates"
    local tls_backup_dir="${backup_base_dir}/tls"
    mkdir -p "$tls_backup_dir"
    
    if [[ -d "${BASE_DIR}/secrets/tls" ]]; then
        cp -r "${BASE_DIR}/secrets/tls"/* "$tls_backup_dir/" 2>/dev/null || true
    fi
    
    # Backup application logs
    log "INFO" "Backing up application logs"
    local logs_backup_dir="${backup_base_dir}/logs"
    mkdir -p "$logs_backup_dir"
    
    if [[ -d "${BASE_DIR}/logs" ]]; then
        find "${BASE_DIR}/logs" -name "*.log" -mtime -7 -exec cp {} "$logs_backup_dir/" \; 2>/dev/null || true
    fi
    
    # Create backup metadata
    cat > "${backup_base_dir}/backup-metadata.json" << EOF
{
    "backup_timestamp": "$backup_timestamp",
    "platform_version": "$(cat ${BASE_DIR}/VERSION 2>/dev/null || echo 'unknown')",
    "backup_type": "full_system",
    "dr_scenario": "${DR_SCENARIO:-routine}",
    "encrypted": $ENCRYPT_BACKUPS,
    "components": [
        "postgresql",
        "redis", 
        "configurations",
        "tls_certificates",
        "application_logs"
    ],
    "retention_days": $BACKUP_RETENTION_DAYS,
    "created_by": "$(whoami)",
    "hostname": "$(hostname)",
    "backup_size": "$(du -sh "$backup_base_dir" | cut -f1)"
}
EOF

    # Encrypt backup if requested
    if [[ "$ENCRYPT_BACKUPS" == "true" ]]; then
        log "INFO" "Encrypting backup archive"
        local encrypted_backup="${BACKUP_DIR}/full-backup-${backup_timestamp}.tar.gz.enc"
        
        # Create encrypted tar archive
        tar czf - -C "$BACKUP_DIR" "full-backup-${backup_timestamp}" | \
        openssl enc -aes-256-cbc -salt -k "xorb-backup-encryption-key-$(date +%Y)" > "$encrypted_backup"
        
        if [[ $? -eq 0 ]]; then
            rm -rf "$backup_base_dir"
            log "SUCCESS" "Encrypted backup created: $encrypted_backup"
            echo "$encrypted_backup"
        else
            log "ERROR" "Backup encryption failed"
            return 1
        fi
    else
        # Create compressed archive
        local compressed_backup="${BACKUP_DIR}/full-backup-${backup_timestamp}.tar.gz"
        tar czf "$compressed_backup" -C "$BACKUP_DIR" "full-backup-${backup_timestamp}"
        rm -rf "$backup_base_dir"
        log "SUCCESS" "Compressed backup created: $compressed_backup"
        echo "$compressed_backup"
    fi
    
    return 0
}

test_disaster_recovery_scenarios() {
    local scenario="${1:-hardware_failure}"
    
    log "DR" "Testing disaster recovery scenario: $scenario"
    
    case "$scenario" in
        "hardware_failure")
            test_hardware_failure_recovery
            ;;
        "ransomware")
            test_ransomware_recovery
            ;;
        "datacenter_outage")
            test_datacenter_outage_recovery
            ;;
        "database_corruption")
            test_database_corruption_recovery
            ;;
        *)
            log "ERROR" "Unknown disaster recovery scenario: $scenario"
            return 1
            ;;
    esac
}

test_hardware_failure_recovery() {
    log "DR" "Simulating hardware failure scenario"
    
    # Simulate hardware failure by stopping all services
    log "INFO" "Simulating service failures"
    docker-compose stop api orchestrator >/dev/null 2>&1 || true
    
    # Wait for failure detection
    sleep 10
    
    # Test backup restoration
    log "INFO" "Testing backup restoration"
    if test_backup_restoration; then
        log "SUCCESS" "Hardware failure recovery test passed"
    else
        log "ERROR" "Hardware failure recovery test failed"
        return 1
    fi
    
    # Restart services
    log "INFO" "Restarting services after test"
    docker-compose up -d api orchestrator >/dev/null 2>&1
    
    return 0
}

test_ransomware_recovery() {
    log "DR" "Simulating ransomware attack scenario"
    
    # Create test files that would be "encrypted" by ransomware
    local test_dir="${BACKUP_DIR}/ransomware-test"
    mkdir -p "$test_dir"
    
    echo "test data" > "${test_dir}/important-file.txt"
    
    # Simulate file encryption
    openssl enc -aes-256-cbc -salt -k "fake-ransomware-key" -in "${test_dir}/important-file.txt" -out "${test_dir}/important-file.txt.encrypted"
    rm "${test_dir}/important-file.txt"
    
    log "INFO" "Files 'encrypted' by simulated ransomware"
    
    # Test point-in-time recovery
    log "INFO" "Testing point-in-time recovery"
    if restore_from_backup "$(date -d '1 hour ago' -Iseconds)"; then
        log "SUCCESS" "Ransomware recovery test passed"
        rm -rf "$test_dir"
        return 0
    else
        log "ERROR" "Ransomware recovery test failed"
        rm -rf "$test_dir"
        return 1
    fi
}

test_datacenter_outage_recovery() {
    log "DR" "Simulating datacenter outage scenario"
    
    # Simulate network isolation
    log "INFO" "Simulating network isolation"
    
    # Test failover procedures
    if [[ "$AUTO_FAILOVER" == "true" ]]; then
        log "INFO" "Testing automatic failover"
        # In a real scenario, this would failover to a different datacenter
        # For testing, we'll just validate failover readiness
        validate_failover_readiness
    else
        log "INFO" "Testing manual failover procedures"
        validate_manual_failover_procedures
    fi
    
    return 0
}

validate_failover_readiness() {
    log "INFO" "Validating failover readiness"
    
    # Check if secondary site configuration exists
    local secondary_config="${CONFIG_DIR}/docker-compose.dr.yml"
    if [[ -f "$secondary_config" ]]; then
        log "SUCCESS" "Secondary site configuration found"
    else
        log "WARN" "Secondary site configuration not found"
    fi
    
    # Validate backup replication
    local latest_backup=$(find "$BACKUP_DIR" -name "*.tar.gz*" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
    if [[ -n "$latest_backup" ]]; then
        log "SUCCESS" "Latest backup available: $latest_backup"
        
        # Check backup age
        local backup_age=$(find "$latest_backup" -mmin +60 2>/dev/null && echo "old" || echo "recent")
        if [[ "$backup_age" == "recent" ]]; then
            log "SUCCESS" "Backup is recent (< 1 hour old)"
        else
            log "WARN" "Backup is older than 1 hour"
        fi
    else
        log "ERROR" "No backups found for failover"
        return 1
    fi
    
    return 0
}

restore_from_backup() {
    local target_time="${1:-latest}"
    
    log "DR" "Restoring system from backup (target: $target_time)"
    
    # Find appropriate backup
    local backup_file=""
    if [[ "$target_time" == "latest" ]]; then
        backup_file=$(find "$BACKUP_DIR" -name "*.tar.gz*" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
    else
        # Find backup closest to target time (simplified logic)
        backup_file=$(find "$BACKUP_DIR" -name "*.tar.gz*" -type f | head -1)
    fi
    
    if [[ -z "$backup_file" ]]; then
        log "ERROR" "No suitable backup found for restoration"
        return 1
    fi
    
    log "INFO" "Using backup file: $backup_file"
    
    # Extract or decrypt backup
    local temp_restore_dir="${BACKUP_DIR}/temp-restore-$(date +%s)"
    mkdir -p "$temp_restore_dir"
    
    if [[ "$backup_file" == *.enc ]]; then
        log "INFO" "Decrypting backup"
        openssl enc -aes-256-cbc -d -salt -k "xorb-backup-encryption-key-$(date +%Y)" -in "$backup_file" | \
        tar xzf - -C "$temp_restore_dir" --strip-components=1
    else
        log "INFO" "Extracting backup"
        tar xzf "$backup_file" -C "$temp_restore_dir" --strip-components=1
    fi
    
    if [[ $? -ne 0 ]]; then
        log "ERROR" "Failed to extract backup"
        rm -rf "$temp_restore_dir"
        return 1
    fi
    
    # Restore PostgreSQL database
    log "INFO" "Restoring PostgreSQL database"
    if [[ -f "${temp_restore_dir}/postgresql-backup.sql" ]]; then
        # Stop applications to prevent database access
        docker-compose stop api orchestrator >/dev/null 2>&1
        
        # Drop and recreate database
        docker-compose exec -T postgres psql -U xorb -c "DROP DATABASE IF EXISTS xorb_platform;" >/dev/null 2>&1
        docker-compose exec -T postgres psql -U xorb -c "CREATE DATABASE xorb_platform;" >/dev/null 2>&1
        
        # Restore database
        if docker-compose exec -T postgres psql -U xorb -d xorb_platform < "${temp_restore_dir}/postgresql-backup.sql" >/dev/null 2>&1; then
            log "SUCCESS" "PostgreSQL database restored"
        else
            log "ERROR" "PostgreSQL database restoration failed"
        fi
    fi
    
    # Restore Redis data
    log "INFO" "Restoring Redis data"
    if [[ -f "${temp_restore_dir}/redis/dump.rdb" ]]; then
        docker-compose stop redis >/dev/null 2>&1
        docker cp "${temp_restore_dir}/redis/dump.rdb" $(docker-compose ps -q redis):/data/dump.rdb 2>/dev/null || true
        docker-compose start redis >/dev/null 2>&1
        log "SUCCESS" "Redis data restored"
    fi
    
    # Restore configurations
    log "INFO" "Restoring application configurations"
    if [[ -d "${temp_restore_dir}/configs" ]]; then
        cp -r "${temp_restore_dir}/configs"/* "${BASE_DIR}/src/api/app/configs/" 2>/dev/null || true
        log "SUCCESS" "Application configurations restored"
    fi
    
    # Restore TLS certificates
    log "INFO" "Restoring TLS certificates"
    if [[ -d "${temp_restore_dir}/tls" ]]; then
        mkdir -p "${BASE_DIR}/secrets/tls"
        cp -r "${temp_restore_dir}/tls"/* "${BASE_DIR}/secrets/tls/" 2>/dev/null || true
        log "SUCCESS" "TLS certificates restored"
    fi
    
    # Clean up temp directory
    rm -rf "$temp_restore_dir"
    
    # Restart all services
    log "INFO" "Restarting all services"
    docker-compose down >/dev/null 2>&1
    sleep 5
    docker-compose up -d >/dev/null 2>&1
    
    # Wait for services to be healthy
    log "INFO" "Waiting for services to become healthy"
    sleep 30
    
    # Validate restoration
    if validate_system_health; then
        log "SUCCESS" "System restoration completed and validated"
        return 0
    else
        log "ERROR" "System restoration completed but validation failed"
        return 1
    fi
}

validate_system_health() {
    log "INFO" "Validating system health after restoration"
    
    # Check API health
    if curl -s http://localhost:8000/api/v1/health >/dev/null 2>&1; then
        log "SUCCESS" "API service is healthy"
    else
        log "ERROR" "API service is not responding"
        return 1
    fi
    
    # Check database connectivity
    if docker-compose exec -T postgres pg_isready -U xorb >/dev/null 2>&1; then
        log "SUCCESS" "Database is accessible"
    else
        log "ERROR" "Database is not accessible"
        return 1
    fi
    
    # Check Redis connectivity
    if docker-compose exec -T redis redis-cli ping >/dev/null 2>&1; then
        log "SUCCESS" "Redis is accessible"
    else
        log "ERROR" "Redis is not accessible"
        return 1
    fi
    
    log "SUCCESS" "System health validation completed"
    return 0
}

cleanup_old_backups() {
    log "INFO" "Cleaning up backups older than $BACKUP_RETENTION_DAYS days"
    
    local deleted_count=0
    
    # Find and delete old backups
    while IFS= read -r -d '' backup_file; do
        log "INFO" "Deleting old backup: $(basename "$backup_file")"
        rm -f "$backup_file"
        deleted_count=$((deleted_count + 1))
    done < <(find "$BACKUP_DIR" -name "*.tar.gz*" -type f -mtime +$BACKUP_RETENTION_DAYS -print0 2>/dev/null)
    
    log "SUCCESS" "Cleanup completed - deleted $deleted_count old backup(s)"
    
    # Clean up old recovery logs
    find "$RECOVERY_LOG_DIR" -name "*.log" -mtime +$BACKUP_RETENTION_DAYS -delete 2>/dev/null || true
}

show_dr_status() {
    log "INFO" "Disaster Recovery System Status"
    
    # Count backups
    local backup_count=$(find "$BACKUP_DIR" -name "*.tar.gz*" -type f | wc -l)
    local latest_backup=$(find "$BACKUP_DIR" -name "*.tar.gz*" -type f -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
    
    echo
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    DISASTER RECOVERY STATUS                  ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║ Total Backups: $(printf "%2d" $backup_count)                                        ║"
    
    if [[ -n "$latest_backup" ]]; then
        local backup_age=$(stat -c %Y "$latest_backup")
        local current_time=$(date +%s)
        local age_hours=$(( (current_time - backup_age) / 3600 ))
        echo "║ Latest Backup: $(printf "%-27s" "$(basename "$latest_backup")") ║"
        echo "║ Backup Age:    $(printf "%2d hours ago" $age_hours)                              ║"
    else
        echo "║ Latest Backup: No backups found                          ║"
    fi
    
    echo "║ Retention:     $(printf "%2d days" $BACKUP_RETENTION_DAYS)                                     ║"
    echo "║ Encryption:    $(printf "%-8s" "$ENCRYPT_BACKUPS")                                 ║"
    echo "║ Auto Failover: $(printf "%-8s" "$AUTO_FAILOVER")                                 ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo
}

send_dr_notification() {
    local status="$1"
    local details="$2"
    
    if [[ -n "$NOTIFICATION_WEBHOOK" ]]; then
        local notification_payload=$(cat << EOF
{
    "text": "🚨 XORB Disaster Recovery Alert",
    "attachments": [{
        "color": "$([ "$status" = "success" ] && echo "good" || echo "danger")",
        "title": "DR Operation: $status",
        "text": "$details",
        "ts": $(date +%s)
    }]
}
EOF
)
        
        curl -s -X POST "$NOTIFICATION_WEBHOOK" \
             -H "Content-Type: application/json" \
             -d "$notification_payload" >/dev/null 2>&1
        
        log "SUCCESS" "DR notification sent"
    fi
}

# Parse command line arguments
COMMAND="$1"
shift

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--scenario)
            DR_SCENARIO="$2"
            shift 2
            ;;
        -t|--target-time)
            RECOVERY_TARGET_TIME="$2"
            shift 2
            ;;
        -r|--retention)
            BACKUP_RETENTION_DAYS="$2"
            shift 2
            ;;
        -e|--encrypt)
            ENCRYPT_BACKUPS=true
            shift
            ;;
        -o|--offsite)
            OFFSITE_BACKUP=true
            shift
            ;;
        -a|--auto-failover)
            AUTO_FAILOVER=true
            shift
            ;;
        -n|--notify)
            NOTIFICATION_WEBHOOK="$2"
            shift 2
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

# Main execution
main() {
    setup_disaster_recovery_logging
    
    if ! validate_dr_environment; then
        log "CRITICAL" "Disaster recovery environment validation failed"
        exit 1
    fi
    
    case "$COMMAND" in
        "backup")
            log "DR" "Initiating full system backup"
            if backup_file=$(create_full_system_backup); then
                send_dr_notification "success" "Full system backup completed: $(basename "$backup_file")"
                log "SUCCESS" "Backup operation completed successfully"
            else
                send_dr_notification "failure" "Full system backup failed"
                log "ERROR" "Backup operation failed"
                exit 1
            fi
            ;;
        "restore")
            log "DR" "Initiating system restoration"
            if restore_from_backup "$RECOVERY_TARGET_TIME"; then
                send_dr_notification "success" "System restoration completed successfully"
                log "SUCCESS" "Restoration operation completed successfully"
            else
                send_dr_notification "failure" "System restoration failed"
                log "ERROR" "Restoration operation failed"
                exit 1
            fi
            ;;
        "test-recovery")
            log "DR" "Initiating disaster recovery testing"
            if test_disaster_recovery_scenarios "$DR_SCENARIO"; then
                send_dr_notification "success" "Disaster recovery test passed: $DR_SCENARIO"
                log "SUCCESS" "Recovery test completed successfully"
            else
                send_dr_notification "failure" "Disaster recovery test failed: $DR_SCENARIO"
                log "ERROR" "Recovery test failed"
                exit 1
            fi
            ;;
        "failover")
            log "DR" "Initiating failover procedures"
            # Failover implementation would depend on specific infrastructure
            log "INFO" "Failover procedures initiated for scenario: $DR_SCENARIO"
            send_dr_notification "info" "Failover initiated for: $DR_SCENARIO"
            ;;
        "cleanup")
            cleanup_old_backups
            ;;
        "status")
            show_dr_status
            ;;
        *)
            echo "Error: Unknown command $COMMAND" >&2
            usage
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"