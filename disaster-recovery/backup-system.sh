#!/bin/bash

# XORB Comprehensive Backup System
# Automated backup of all critical components

set -euo pipefail

BACKUP_DIR="/root/Xorb/disaster-recovery/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/root/Xorb/disaster-recovery/logs/backup_$TIMESTAMP.log"

echo "💾 XORB Backup System - $TIMESTAMP" | tee "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"

# Function to log and execute
log_backup() {
    echo "🔧 $1" | tee -a "$LOG_FILE"
    shift
    if "$@" 2>&1 | tee -a "$LOG_FILE"; then
        echo "✅ Success" | tee -a "$LOG_FILE"
        return 0
    else
        echo "❌ Failed" | tee -a "$LOG_FILE"
        return 1
    fi
}

# Database backup
backup_database() {
    echo "🗄️  Database Backup" | tee -a "$LOG_FILE"
    
    # PostgreSQL backup
    if docker ps | grep -q postgres; then
        log_backup "Backing up PostgreSQL..." \
            docker exec -t $(docker ps | grep postgres | awk '{print $1}') \
            pg_dumpall -U xorb > "$BACKUP_DIR/database/postgres_backup_$TIMESTAMP.sql"
    else
        echo "⚠️  PostgreSQL container not running" | tee -a "$LOG_FILE"
    fi
    
    # Redis backup
    if docker ps | grep -q redis; then
        log_backup "Backing up Redis..." \
            docker exec -t $(docker ps | grep redis | awk '{print $1}') \
            redis-cli BGSAVE
        
        # Copy Redis dump
        docker cp $(docker ps | grep redis | awk '{print $1}'):/data/dump.rdb \
            "$BACKUP_DIR/database/redis_backup_$TIMESTAMP.rdb" 2>&1 | tee -a "$LOG_FILE"
    else
        echo "⚠️  Redis container not running" | tee -a "$LOG_FILE"
    fi
}

# Configuration backup
backup_configuration() {
    echo "⚙️  Configuration Backup" | tee -a "$LOG_FILE"
    
    # Docker compose files
    log_backup "Backing up Docker configurations..." \
        tar -czf "$BACKUP_DIR/config/docker-configs_$TIMESTAMP.tar.gz" \
        docker-compose*.yml monitoring/ ssl/ 2>/dev/null || true
    
    # Kubernetes manifests
    if [ -d "/root/Xorb/deployment" ]; then
        log_backup "Backing up Kubernetes manifests..." \
            tar -czf "$BACKUP_DIR/config/k8s-manifests_$TIMESTAMP.tar.gz" \
            deployment/ 2>/dev/null || true
    fi
    
    # Environment configurations
    if [ -d "/root/Xorb/domains" ]; then
        log_backup "Backing up domain configurations..." \
            tar -czf "$BACKUP_DIR/config/domain-configs_$TIMESTAMP.tar.gz" \
            domains/ 2>/dev/null || true
    fi
}

# Secrets backup
backup_secrets() {
    echo "🔐 Secrets Backup" | tee -a "$LOG_FILE"
    
    if [ -d "/root/Xorb/secrets" ]; then
        log_backup "Backing up encrypted secrets..." \
            /root/Xorb/secrets/backup-secrets.sh
        
        # Copy backup to DR location
        cp /root/Xorb/backups/secrets/secrets_vault_*.tar.gz* \
            "$BACKUP_DIR/secrets/" 2>/dev/null || true
    else
        echo "⚠️  Secrets directory not found" | tee -a "$LOG_FILE"
    fi
}

# Code backup
backup_code() {
    echo "💻 Code Backup" | tee -a "$LOG_FILE"
    
    log_backup "Backing up XORB codebase..." \
        tar --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' \
        --exclude='logs' --exclude='backups' \
        -czf "$BACKUP_DIR/code/xorb-codebase_$TIMESTAMP.tar.gz" \
        /root/Xorb/
}

# SSL certificates backup
backup_ssl() {
    echo "🔒 SSL Certificates Backup" | tee -a "$LOG_FILE"
    
    if [ -d "/root/Xorb/ssl" ]; then
        log_backup "Backing up SSL certificates..." \
            tar -czf "$BACKUP_DIR/config/ssl-certificates_$TIMESTAMP.tar.gz" \
            ssl/
    else
        echo "⚠️  SSL directory not found" | tee -a "$LOG_FILE"
    fi
}

# Execute all backup functions
backup_database
backup_configuration
backup_secrets
backup_code
backup_ssl

# Cleanup old backups (keep last 30 days)
echo "🧹 Cleaning up old backups..." | tee -a "$LOG_FILE"
find "$BACKUP_DIR" -type f -name "*_*.tar.gz" -mtime +30 -delete 2>/dev/null || true
find "$BACKUP_DIR" -type f -name "*_*.sql" -mtime +30 -delete 2>/dev/null || true
find "$BACKUP_DIR" -type f -name "*_*.rdb" -mtime +30 -delete 2>/dev/null || true

# Generate backup report
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
BACKUP_COUNT=$(find "$BACKUP_DIR" -type f -name "*_$TIMESTAMP*" | wc -l)

echo "" | tee -a "$LOG_FILE"
echo "📊 Backup Summary:" | tee -a "$LOG_FILE"
echo "  - Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "  - Files backed up: $BACKUP_COUNT" | tee -a "$LOG_FILE"
echo "  - Total backup size: $BACKUP_SIZE" | tee -a "$LOG_FILE"
echo "  - Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "✅ Backup completed successfully!" | tee -a "$LOG_FILE"
