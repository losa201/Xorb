#!/bin/bash
# PostgreSQL backup script using wal-g for Xorb PTaaS
# Optimized for EPYC single-node deployment

set -euo pipefail

# Configuration
BACKUP_LOG="/var/log/backups/postgres-backup.log"
RETENTION_DAYS=7
MAX_RETRIES=3

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BACKUP_LOG"
}

# Error handling
handle_error() {
    log "ERROR: $1"
    # Send metrics to Prometheus
    echo "xorb_backup_postgres_failed_total 1" | curl -X POST --data-binary @- http://prometheus:9090/api/v1/import/prometheus
    exit 1
}

# Wait for PostgreSQL to be ready
wait_for_postgres() {
    local retries=0
    while [ $retries -lt $MAX_RETRIES ]; do
        if pg_isready -h "$PGHOST" -U "$PGUSER" -d "$PGDATABASE"; then
            log "PostgreSQL is ready"
            return 0
        fi
        retries=$((retries + 1))
        log "Waiting for PostgreSQL... attempt $retries/$MAX_RETRIES"
        sleep 10
    done
    handle_error "PostgreSQL is not ready after $MAX_RETRIES attempts"
}

# Perform base backup
perform_base_backup() {
    log "Starting PostgreSQL base backup"
    
    # Create base backup
    if wal-g backup-push /var/lib/postgresql/data; then
        log "Base backup completed successfully"
        
        # Send success metrics
        echo "xorb_backup_postgres_success_total 1" | curl -X POST --data-binary @- http://prometheus:9090/api/v1/import/prometheus
        echo "xorb_backup_postgres_last_success_timestamp $(date +%s)" | curl -X POST --data-binary @- http://prometheus:9090/api/v1/import/prometheus
    else
        handle_error "Base backup failed"
    fi
}

# Clean old backups
cleanup_old_backups() {
    log "Cleaning up old backups (keeping last $RETENTION_DAYS days)"
    
    if wal-g delete retain FULL $RETENTION_DAYS; then
        log "Old backups cleaned up successfully"
    else
        log "WARNING: Failed to clean up old backups"
    fi
}

# Health check
health_check() {
    # Check if we can connect to PostgreSQL
    if ! pg_isready -h "$PGHOST" -U "$PGUSER" -d "$PGDATABASE"; then
        handle_error "PostgreSQL health check failed"
    fi
    
    # Check if S3 is accessible
    if ! aws s3 ls "$WALG_S3_PREFIX" >/dev/null 2>&1; then
        handle_error "S3 backup destination is not accessible"
    fi
    
    log "Health check passed"
}

# Continuous WAL archiving mode
wal_archiving_mode() {
    log "Starting continuous WAL archiving mode"
    
    while true; do
        # Health check every hour
        health_check
        
        # Perform base backup every 24 hours
        local current_hour=$(date +%H)
        if [ "$current_hour" = "02" ]; then  # 2 AM
            perform_base_backup
            cleanup_old_backups
        fi
        
        # Sleep for 1 hour
        sleep 3600
    done
}

# One-time backup mode
one_time_backup() {
    log "Performing one-time PostgreSQL backup"
    
    health_check
    perform_base_backup
    cleanup_old_backups
    
    log "One-time backup completed"
}

# Main execution
main() {
    log "Starting PostgreSQL backup service"
    
    # Wait for PostgreSQL to be ready
    wait_for_postgres
    
    # Check if running in one-time mode
    if [ "${BACKUP_MODE:-continuous}" = "once" ]; then
        one_time_backup
    else
        wal_archiving_mode
    fi
}

# Run main function
main "$@"