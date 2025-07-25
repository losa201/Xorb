#!/bin/bash
# File and configuration backup script for Xorb PTaaS
# Backs up source code, configurations, and logs

set -euo pipefail

# Install required tools
apk add --no-cache aws-cli tar gzip

# Configuration
BACKUP_LOG="/var/log/backups/file-backup.log"
BACKUP_INTERVAL=${BACKUP_INTERVAL:-86400}  # 24 hours
RETENTION_DAYS=30
MAX_RETRIES=3

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BACKUP_LOG"
}

# Error handling
handle_error() {
    log "ERROR: $1"
    # Send metrics to Prometheus
    echo "xorb_backup_files_failed_total 1" | curl -X POST --data-binary @- http://prometheus:9090/api/v1/import/prometheus
    exit 1
}

# Create compressed archive
create_archive() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local archive_name="xorb_files_backup_${timestamp}.tar.gz"
    local archive_path="/tmp/$archive_name"
    
    log "Creating file archive: $archive_name"
    
    # Create archive with configurations and source code
    if tar -czf "$archive_path" \
        -C /backup \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        --exclude='*.log.*' \
        --exclude='*.tmp' \
        --exclude='.git' \
        compose/ \
        xorb_common/ \
        services/ \
        system_logs/; then
        
        log "Archive created successfully: $archive_path"
        
        # Calculate and log archive size
        local archive_size=$(stat -c%s "$archive_path")
        local archive_size_mb=$((archive_size / 1024 / 1024))
        log "Archive size: ${archive_size_mb}MB"
        
        # Send metrics
        echo "xorb_backup_files_size_bytes $archive_size" | curl -X POST --data-binary @- http://prometheus:9090/api/v1/import/prometheus
        
        echo "$archive_path"
    else
        handle_error "Failed to create archive"
    fi
}

# Upload to S3
upload_to_s3() {
    local archive_path="$1"
    local archive_name=$(basename "$archive_path")
    local s3_key="$BACKUP_S3_PREFIX/$archive_name"
    
    log "Uploading $archive_name to s3://$BACKUP_S3_BUCKET/$s3_key"
    
    if aws s3 cp "$archive_path" "s3://$BACKUP_S3_BUCKET/$s3_key" \
        --storage-class STANDARD_IA \
        --metadata "backup-type=files,created=$(date -Iseconds)"; then
        
        log "Upload completed successfully"
        
        # Send success metrics
        echo "xorb_backup_files_success_total 1" | curl -X POST --data-binary @- http://prometheus:9090/api/v1/import/prometheus
        echo "xorb_backup_files_last_success_timestamp $(date +%s)" | curl -X POST --data-binary @- http://prometheus:9090/api/v1/import/prometheus
        
        # Clean up local file
        rm -f "$archive_path"
    else
        handle_error "Failed to upload to S3"
    fi
}

# Create incremental backup (only changed files)
create_incremental_archive() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local archive_name="xorb_files_incremental_${timestamp}.tar.gz"
    local archive_path="/tmp/$archive_name"
    local last_backup_file="/tmp/last_backup_timestamp"
    
    log "Creating incremental file archive: $archive_name"
    
    # Find files modified since last backup
    local find_args=()
    if [ -f "$last_backup_file" ]; then
        local last_backup=$(cat "$last_backup_file")
        find_args=("-newer" "$last_backup_file")
        log "Looking for files modified since $(date -d @$last_backup)"
    else
        log "No previous backup found, creating full backup"
    fi
    
    # Create list of changed files
    local changed_files="/tmp/changed_files.txt"
    find /backup \
        "${find_args[@]}" \
        -type f \
        -not -path "*/.*" \
        -not -name "*.pyc" \
        -not -name "*.log.*" \
        -not -name "*.tmp" \
        > "$changed_files" || true
    
    local file_count=$(wc -l < "$changed_files")
    log "Found $file_count changed files"
    
    if [ "$file_count" -gt 0 ]; then
        # Create incremental archive
        if tar -czf "$archive_path" \
            -C /backup \
            -T "$changed_files"; then
            
            log "Incremental archive created successfully: $archive_path"
            
            # Update last backup timestamp
            echo "$(date +%s)" > "$last_backup_file"
            
            echo "$archive_path"
        else
            handle_error "Failed to create incremental archive"
        fi
    else
        log "No changes detected, skipping incremental backup"
        return 1
    fi
}

# Clean old backups from S3
cleanup_old_backups() {
    log "Cleaning up old file backups (keeping last $RETENTION_DAYS days)"
    
    local cutoff_date=$(date -d "$RETENTION_DAYS days ago" +%Y%m%d)
    
    # List and delete old backups
    aws s3 ls "s3://$BACKUP_S3_BUCKET/$BACKUP_S3_PREFIX/" | while read -r line; do
        local filename=$(echo "$line" | awk '{print $4}')
        if [[ "$filename" =~ xorb_files.*_([0-9]{8})_ ]]; then
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
    # Check if backup directories exist
    for dir in /backup/compose /backup/xorb_common /backup/services; do
        if [ ! -d "$dir" ]; then
            handle_error "Backup directory $dir does not exist"
        fi
    done
    
    # Check S3 connectivity
    if ! aws s3 ls "s3://$BACKUP_S3_BUCKET/$BACKUP_S3_PREFIX/" >/dev/null 2>&1; then
        handle_error "S3 backup destination is not accessible"
    fi
    
    log "Health check passed"
}

# Backup system information
backup_system_info() {
    local info_file="/tmp/xorb_system_info_$(date +%Y%m%d_%H%M%S).json"
    
    log "Collecting system information"
    
    cat > "$info_file" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "kernel": "$(uname -r)",
    "os": "$(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"')",
    "uptime": "$(uptime -p)",
    "disk_usage": $(df -h / | tail -1 | awk '{print "{\"filesystem\":\""$1"\", \"size\":\""$2"\", \"used\":\""$3"\", \"available\":\""$4"\", \"use_percent\":\""$5"\"}'}),
    "docker_info": {
        "version": "$(docker --version 2>/dev/null || echo 'N/A')",
        "containers": $(docker ps --format "table {{.Names}}" 2>/dev/null | tail -n +2 | wc -l || echo 0)
    },
    "backup_type": "files",
    "epyc_optimization": true
}
EOF
    
    # Upload system info to S3
    aws s3 cp "$info_file" "s3://$BACKUP_S3_BUCKET/$BACKUP_S3_PREFIX/system_info/$(basename $info_file)"
    rm -f "$info_file"
}

# Continuous backup mode
continuous_backup() {
    log "Starting continuous file backup mode (interval: ${BACKUP_INTERVAL}s)"
    
    while true; do
        # Health check
        health_check
        
        # Backup system information
        backup_system_info
        
        # Determine backup type based on time
        local current_hour=$(date +%H)
        
        if [ "$current_hour" = "01" ]; then
            # Full backup at 1 AM
            log "Performing full backup"
            local archive_path=$(create_archive)
            upload_to_s3 "$archive_path"
        else
            # Incremental backup other times
            log "Attempting incremental backup"
            if create_incremental_archive >/dev/null 2>&1; then
                local archive_path=$(create_incremental_archive)
                upload_to_s3 "$archive_path"
            else
                log "No incremental changes, skipping backup"
            fi
        fi
        
        # Cleanup old backups once a day at 4 AM
        if [ "$current_hour" = "04" ]; then
            cleanup_old_backups
        fi
        
        # Sleep until next backup
        sleep "$BACKUP_INTERVAL"
    done
}

# One-time backup mode
one_time_backup() {
    log "Performing one-time file backup"
    
    health_check
    backup_system_info
    
    local archive_path=$(create_archive)
    upload_to_s3 "$archive_path"
    cleanup_old_backups
    
    log "One-time backup completed"
}

# Main execution
main() {
    log "Starting file backup service"
    
    # Check if running in one-time mode
    if [ "${BACKUP_MODE:-continuous}" = "once" ]; then
        one_time_backup
    else
        continuous_backup
    fi
}

# Run main function
main "$@"