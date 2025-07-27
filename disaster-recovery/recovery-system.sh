#!/bin/bash

# XORB Disaster Recovery System
# Automated recovery from backups

set -euo pipefail

BACKUP_DIR="/root/Xorb/disaster-recovery/backups"
RECOVERY_DIR="/root/Xorb/disaster-recovery/recovery"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/root/Xorb/disaster-recovery/logs/recovery_$TIMESTAMP.log"

RECOVERY_TYPE="${1:-full}"
BACKUP_DATE="${2:-latest}"

echo "🚨 XORB Disaster Recovery - $TIMESTAMP" | tee "$LOG_FILE"
echo "Recovery Type: $RECOVERY_TYPE" | tee -a "$LOG_FILE"
echo "Backup Date: $BACKUP_DATE" | tee -a "$LOG_FILE"
echo "====================================" | tee -a "$LOG_FILE"

# Function to log and execute
log_recovery() {
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

# Get latest backup files
get_latest_backup() {
    local backup_type="$1"
    if [ "$BACKUP_DATE" = "latest" ]; then
        find "$BACKUP_DIR/$backup_type" -name "*_*.tar.gz" -o -name "*_*.sql" -o -name "*_*.rdb" | \
        sort | tail -1
    else
        find "$BACKUP_DIR/$backup_type" -name "*_${BACKUP_DATE}_*" | head -1
    fi
}

# Database recovery
recover_database() {
    echo "🗄️  Database Recovery" | tee -a "$LOG_FILE"
    
    # Stop existing database containers
    docker stop $(docker ps -q --filter "name=postgres") 2>/dev/null || true
    docker stop $(docker ps -q --filter "name=redis") 2>/dev/null || true
    
    # PostgreSQL recovery
    POSTGRES_BACKUP=$(get_latest_backup "database" | grep postgres || echo "")
    if [ -n "$POSTGRES_BACKUP" ]; then
        log_recovery "Recovering PostgreSQL from $POSTGRES_BACKUP..." \
            docker-compose -f docker-compose.production.yml up -d postgres
        
        sleep 30  # Wait for PostgreSQL to start
        
        log_recovery "Restoring PostgreSQL data..." \
            docker exec -i $(docker ps | grep postgres | awk '{print $1}') \
            psql -U xorb < "$POSTGRES_BACKUP"
    else
        echo "⚠️  No PostgreSQL backup found" | tee -a "$LOG_FILE"
    fi
    
    # Redis recovery
    REDIS_BACKUP=$(get_latest_backup "database" | grep redis || echo "")
    if [ -n "$REDIS_BACKUP" ]; then
        log_recovery "Recovering Redis from $REDIS_BACKUP..." \
            docker-compose -f docker-compose.production.yml up -d redis
        
        sleep 10  # Wait for Redis to start
        
        log_recovery "Restoring Redis data..." \
            docker cp "$REDIS_BACKUP" $(docker ps | grep redis | awk '{print $1}'):/data/dump.rdb
        
        docker restart $(docker ps | grep redis | awk '{print $1}')
    else
        echo "⚠️  No Redis backup found" | tee -a "$LOG_FILE"
    fi
}

# Configuration recovery
recover_configuration() {
    echo "⚙️  Configuration Recovery" | tee -a "$LOG_FILE"
    
    # Docker configurations
    DOCKER_BACKUP=$(get_latest_backup "config" | grep docker-configs || echo "")
    if [ -n "$DOCKER_BACKUP" ]; then
        log_recovery "Recovering Docker configurations..." \
            tar -xzf "$DOCKER_BACKUP" -C /root/Xorb/
    fi
    
    # Kubernetes manifests
    K8S_BACKUP=$(get_latest_backup "config" | grep k8s-manifests || echo "")
    if [ -n "$K8S_BACKUP" ]; then
        log_recovery "Recovering Kubernetes manifests..." \
            tar -xzf "$K8S_BACKUP" -C /root/Xorb/
    fi
    
    # Domain configurations
    DOMAIN_BACKUP=$(get_latest_backup "config" | grep domain-configs || echo "")
    if [ -n "$DOMAIN_BACKUP" ]; then
        log_recovery "Recovering domain configurations..." \
            tar -xzf "$DOMAIN_BACKUP" -C /root/Xorb/
    fi
    
    # SSL certificates
    SSL_BACKUP=$(get_latest_backup "config" | grep ssl-certificates || echo "")
    if [ -n "$SSL_BACKUP" ]; then
        log_recovery "Recovering SSL certificates..." \
            tar -xzf "$SSL_BACKUP" -C /root/Xorb/
    fi
}

# Secrets recovery
recover_secrets() {
    echo "🔐 Secrets Recovery" | tee -a "$LOG_FILE"
    
    SECRETS_BACKUP=$(get_latest_backup "secrets" | head -1)
    if [ -n "$SECRETS_BACKUP" ]; then
        log_recovery "Recovering secrets vault..." \
            tar -xzf "$SECRETS_BACKUP" -C /root/Xorb/secrets/
    else
        echo "⚠️  No secrets backup found" | tee -a "$LOG_FILE"
    fi
}

# Code recovery
recover_code() {
    echo "💻 Code Recovery" | tee -a "$LOG_FILE"
    
    CODE_BACKUP=$(get_latest_backup "code" | head -1)
    if [ -n "$CODE_BACKUP" ]; then
        # Create recovery environment
        mkdir -p "$RECOVERY_DIR/xorb-recovered"
        
        log_recovery "Recovering XORB codebase..." \
            tar -xzf "$CODE_BACKUP" -C "$RECOVERY_DIR/"
        
        echo "📁 Code recovered to: $RECOVERY_DIR/" | tee -a "$LOG_FILE"
    else
        echo "⚠️  No code backup found" | tee -a "$LOG_FILE"
    fi
}

# Service recovery
recover_services() {
    echo "🚀 Service Recovery" | tee -a "$LOG_FILE"
    
    # Start core services
    log_recovery "Starting core services..." \
        docker-compose -f docker-compose.production.yml up -d
    
    # Wait for services to be ready
    echo "⏳ Waiting for services to be ready..." | tee -a "$LOG_FILE"
    sleep 60
    
    # Verify services
    log_recovery "Verifying service health..." \
        python3 enterprise_deployment_verification.py
}

# Execute recovery based on type
case "$RECOVERY_TYPE" in
    "full")
        echo "🚨 Performing full disaster recovery..." | tee -a "$LOG_FILE"
        recover_database
        recover_configuration
        recover_secrets
        recover_services
        ;;
    "database")
        recover_database
        ;;
    "config")
        recover_configuration
        ;;
    "secrets")
        recover_secrets
        ;;
    "code")
        recover_code
        ;;
    "services")
        recover_services
        ;;
    *)
        echo "❌ Unknown recovery type: $RECOVERY_TYPE" | tee -a "$LOG_FILE"
        echo "Available types: full, database, config, secrets, code, services" | tee -a "$LOG_FILE"
        exit 1
        ;;
esac

echo "" | tee -a "$LOG_FILE"
echo "✅ Recovery completed successfully!" | tee -a "$LOG_FILE"
echo "📋 Recovery log: $LOG_FILE" | tee -a "$LOG_FILE"
