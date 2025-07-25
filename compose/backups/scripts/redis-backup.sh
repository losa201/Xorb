#!/bin/bash
# Redis backup script for Xorb PTaaS
# Automated snapshots with S3 upload

set -euo pipefail

# Configuration
BACKUP_LOG="/var/log/backups/redis-backup.log"
BACKUP_INTERVAL=${BACKUP_INTERVAL:-900}  # 15 minutes
RETENTION_DAYS=7
MAX_RETRIES=3

# Install AWS CLI
apk add --no-cache aws-cli

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BACKUP_LOG"
}

# Error handling
handle_error() {
    log "ERROR: $1"
    # Send metrics to Prometheus
    echo "xorb_backup_redis_failed_total 1" | curl -X POST --data-binary @- http://prometheus:9090/api/v1/import/prometheus
    exit 1
}

# Wait for Redis to be ready
wait_for_redis() {
    local retries=0
    while [ $retries -lt $MAX_RETRIES ]; do
        if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping >/dev/null 2>&1; then
            log "Redis is ready"
            return 0
        fi
        retries=$((retries + 1))
        log "Waiting for Redis... attempt $retries/$MAX_RETRIES"
        sleep 5
    done
    handle_error "Redis is not ready after $MAX_RETRIES attempts"
}

# Create Redis snapshot
create_snapshot() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local snapshot_file="/tmp/redis_backup_${timestamp}.rdb"
    
    log "Creating Redis snapshot: $snapshot_file"
    
    # Trigger BGSAVE on Redis
    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" BGSAVE >/dev/null; then
        log "BGSAVE command sent successfully"
        
        # Wait for BGSAVE to complete
        while [ "$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" LASTSAVE)" = "$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" LASTSAVE)" ]; do
            sleep 1
        done
        
        # Copy the RDB file
        if cp /data/dump.rdb "$snapshot_file"; then
            log "Snapshot created successfully: $snapshot_file"
            echo "$snapshot_file"
        else
            handle_error "Failed to copy RDB file"
        fi
    else
        handle_error "Failed to trigger BGSAVE"
    fi
}

# Upload to S3
upload_to_s3() {
    local snapshot_file="$1"
    local s3_key="$BACKUP_S3_PREFIX/$(basename $snapshot_file)"
    
    log "Uploading $snapshot_file to s3://$BACKUP_S3_BUCKET/$s3_key"
    
    if aws s3 cp "$snapshot_file" "s3://$BACKUP_S3_BUCKET/$s3_key"; then
        log "Upload completed successfully"
        
        # Send success metrics
        echo "xorb_backup_redis_success_total 1" | curl -X POST --data-binary @- http://prometheus:9090/api/v1/import/prometheus
        echo "xorb_backup_redis_last_success_timestamp $(date +%s)" | curl -X POST --data-binary @- http://prometheus:9090/api/v1/import/prometheus
        
        # Calculate and report backup size
        local backup_size=$(stat -c%s "$snapshot_file")
        echo "xorb_backup_redis_size_bytes $backup_size" | curl -X POST --data-binary @- http://prometheus:9090/api/v1/import/prometheus
        
        # Clean up local file
        rm -f "$snapshot_file"
    else
        handle_error "Failed to upload to S3"
    fi
}

# Clean old backups from S3
cleanup_old_backups() {
    log "Cleaning up old Redis backups (keeping last $RETENTION_DAYS days)"
    
    local cutoff_date=$(date -d "$RETENTION_DAYS days ago" +%Y%m%d)
    
    # List and delete old backups
    aws s3 ls "s3://$BACKUP_S3_BUCKET/$BACKUP_S3_PREFIX/" | while read -r line; do
        local filename=$(echo "$line" | awk '{print $4}')
        if [[ "$filename" =~ redis_backup_([0-9]{8})_ ]]; then
            local backup_date="${BASH_REMATCH[1]}"
            if [ "$backup_date" -lt "$cutoff_date" ]; then
                log "Deleting old backup: $filename"
                aws s3 rm "s3://$BACKUP_S3_BUCKET/$BACKUP_S3_PREFIX/$filename"
            fi
        fi
    done
}

# Health check
health_check() {
    # Check Redis connectivity
    if ! redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping >/dev/null 2>&1; then
        handle_error "Redis health check failed"
    fi
    
    # Check S3 connectivity
    if ! aws s3 ls "s3://$BACKUP_S3_BUCKET/$BACKUP_S3_PREFIX/" >/dev/null 2>&1; then
        handle_error "S3 backup destination is not accessible"
    fi
    
    log "Health check passed"
}

# Get Redis info for monitoring
report_redis_metrics() {
    local info=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" INFO memory | grep used_memory_human)
    local memory_usage=$(echo "$info" | cut -d: -f2 | tr -d '\r\n')
    log "Redis memory usage: $memory_usage"
    
    # Report to Prometheus
    local memory_bytes=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" INFO memory | grep '^used_memory:' | cut -d: -f2 | tr -d '\r\n')
    echo "xorb_redis_memory_used_bytes $memory_bytes" | curl -X POST --data-binary @- http://prometheus:9090/api/v1/import/prometheus
}

# Continuous backup mode
continuous_backup() {
    log "Starting continuous Redis backup mode (interval: ${BACKUP_INTERVAL}s)"
    
    while true; do
        # Health check
        health_check
        
        # Report metrics
        report_redis_metrics
        
        # Create and upload snapshot
        local snapshot_file=$(create_snapshot)
        upload_to_s3 "$snapshot_file"
        
        # Cleanup old backups every 24 hours
        local current_hour=$(date +%H)
        if [ "$current_hour" = "03" ]; then  # 3 AM
            cleanup_old_backups
        fi
        
        # Sleep until next backup
        sleep "$BACKUP_INTERVAL"
    done
}

# One-time backup mode
one_time_backup() {
    log "Performing one-time Redis backup"
    
    health_check
    report_redis_metrics
    
    local snapshot_file=$(create_snapshot)
    upload_to_s3 "$snapshot_file"
    cleanup_old_backups
    
    log "One-time backup completed"
}

# Main execution
main() {
    log "Starting Redis backup service"
    
    # Wait for Redis to be ready
    wait_for_redis
    
    # Check if running in one-time mode
    if [ "${BACKUP_MODE:-continuous}" = "once" ]; then
        one_time_backup
    else
        continuous_backup
    fi
}

# Run main function
main "$@"