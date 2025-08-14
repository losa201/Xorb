#!/bin/bash
# XORB Production Deployment Script
# Addresses all pre-flight security and performance requirements

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_LOG="/root/Xorb/deployment-$(date +%Y%m%d_%H%M%S).log"
REQUIRED_VARS=("JWT_SECRET" "XORB_API_KEY" "DATABASE_URL" "POSTGRES_PASSWORD")
BACKUP_DIR="/root/Xorb/backups"
HEALTH_CHECK_TIMEOUT=60
ROLLBACK_ENABLED=true

# Logging functions
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$DEPLOYMENT_LOG"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

# Error handling
handle_error() {
    local exit_code=$?
    local line_number=$1
    log_error "Deployment failed at line $line_number with exit code $exit_code"

    if [[ "$ROLLBACK_ENABLED" == "true" ]]; then
        log_warning "Initiating rollback procedure..."
        rollback_deployment
    fi

    exit $exit_code
}

trap 'handle_error $LINENO' ERR

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Starting pre-deployment checks..."

    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi

    # Check system resources
    local free_memory=$(free -m | awk 'NR==2{print $7}')
    local disk_space=$(df -h / | awk 'NR==2{print $4}' | sed 's/[^0-9]*//g')

    if [[ $free_memory -lt 1024 ]]; then
        log_warning "Low memory detected: ${free_memory}MB available"
    fi

    if [[ $disk_space -lt 5 ]]; then
        log_error "Insufficient disk space: ${disk_space}GB available"
        exit 1
    fi

    # Check environment variables
    log_info "Validating environment variables..."
    local missing_vars=()

    for var in "${REQUIRED_VARS[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_vars+=("$var")
        fi
    done

    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required environment variables: ${missing_vars[*]}"
        log_error "Please set these variables in /root/Xorb/.env or export them"
        exit 1
    fi

    # Validate JWT secret strength
    local jwt_length=${#JWT_SECRET}
    if [[ $jwt_length -lt 32 ]]; then
        log_error "JWT_SECRET must be at least 32 characters long (current: $jwt_length)"
        exit 1
    fi

    # Check for default/weak secrets
    if [[ "$JWT_SECRET" == *"change"* ]] || [[ "$JWT_SECRET" == *"secret"* ]]; then
        log_error "JWT_SECRET appears to contain default/weak values"
        exit 1
    fi

    # Test database connection
    log_info "Testing database connection..."
    if ! docker exec xorb-postgres pg_isready -U postgres >/dev/null 2>&1; then
        log_warning "Database not ready, will attempt to start..."
    fi

    log_success "Pre-deployment checks completed"
}

# Security hardening
security_hardening() {
    log_info "Applying security hardening..."

    # Set secure file permissions
    find /root/Xorb -type f -name "*.py" -exec chmod 644 {} \;
    find /root/Xorb -type f -name "*.sh" -exec chmod 755 {} \;
    find /root/Xorb -type d -exec chmod 755 {} \;

    # Secure sensitive files
    if [[ -f "/root/Xorb/.env" ]]; then
        chmod 600 /root/Xorb/.env
        chown root:root /root/Xorb/.env
    fi

    # SSL certificates
    find /root/Xorb/ssl -name "*.key" -exec chmod 600 {} \;
    find /root/Xorb/ssl -name "*.crt" -exec chmod 644 {} \;

    # Remove any temporary or sensitive files
    find /root/Xorb -name "*.tmp" -delete
    find /root/Xorb -name "*.log.old" -delete
    find /root/Xorb -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

    # Set up log rotation
    cat > /etc/logrotate.d/xorb << 'EOF'
/root/Xorb/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
    create 644 root root
}
EOF

    log_success "Security hardening applied"
}

# Create backup
create_backup() {
    log_info "Creating deployment backup..."

    mkdir -p "$BACKUP_DIR"
    local backup_file="$BACKUP_DIR/xorb-backup-$(date +%Y%m%d_%H%M%S).tar.gz"

    # Backup current deployment
    tar -czf "$backup_file" \
        --exclude="node_modules" \
        --exclude="__pycache__" \
        --exclude="*.pyc" \
        --exclude="logs" \
        -C /root Xorb

    # Keep only last 10 backups
    ls -t "$BACKUP_DIR"/xorb-backup-*.tar.gz | tail -n +11 | xargs rm -f 2>/dev/null || true

    log_success "Backup created: $backup_file"
    echo "$backup_file" > /tmp/last_backup_file
}

# Database setup and migration
setup_database() {
    log_info "Setting up database..."

    # Start PostgreSQL if not running
    if ! docker ps | grep -q xorb-postgres; then
        log_info "Starting PostgreSQL container..."
        cd /root/Xorb
        docker-compose -f infra/docker-compose.yml up -d postgres

        # Wait for database to be ready
        local retries=30
        while ! docker exec xorb-postgres pg_isready -U postgres >/dev/null 2>&1; do
            if [[ $retries -eq 0 ]]; then
                log_error "Database failed to start"
                exit 1
            fi
            log_info "Waiting for database... ($retries retries left)"
            sleep 2
            ((retries--))
        done
    fi

    # Run database migrations
    log_info "Running database migrations..."
    cd /root/Xorb/src/api
    if [[ -f "alembic.ini" ]]; then
        python -m alembic upgrade head
    else
        log_warning "No alembic.ini found, skipping migrations"
    fi

    log_success "Database setup completed"
}

# Build and deploy services
deploy_services() {
    log_info "Deploying XORB services..."

    cd /root/Xorb

    # Build updated images
    log_info "Building Docker images..."
    docker-compose -f infra/docker-compose.production.yml build --no-cache

    # Start services in production mode
    log_info "Starting services..."
    ENVIRONMENT=production docker-compose -f infra/docker-compose.production.yml up -d

    # Wait for services to start
    local services=("xorb-api" "xorb-orchestrator" "xorb-redis")
    for service in "${services[@]}"; do
        local retries=30
        while ! docker ps | grep -q "$service"; do
            if [[ $retries -eq 0 ]]; then
                log_error "Service $service failed to start"
                exit 1
            fi
            log_info "Waiting for $service to start... ($retries retries left)"
            sleep 2
            ((retries--))
        done
        log_success "$service started successfully"
    done

    log_success "All services deployed"
}

# Health checks
perform_health_checks() {
    log_info "Performing health checks..."

    local endpoints=(
        "http://localhost:8080/api/health"
        "http://localhost:8081/health"  # Orchestrator
    )

    for endpoint in "${endpoints[@]}"; do
        log_info "Checking $endpoint..."

        local retries=30
        local success=false

        while [[ $retries -gt 0 ]]; do
            if curl -f -s "$endpoint" >/dev/null 2>&1; then
                log_success "$endpoint is healthy"
                success=true
                break
            fi

            log_info "Health check failed for $endpoint, retrying... ($retries retries left)"
            sleep 2
            ((retries--))
        done

        if [[ "$success" != "true" ]]; then
            log_error "Health check failed for $endpoint"
            exit 1
        fi
    done

    # Test API functionality
    log_info "Testing API functionality..."
    local response=$(curl -s -w "%{http_code}" http://localhost:8080/api/health)
    local status_code="${response: -3}"

    if [[ "$status_code" != "200" ]]; then
        log_error "API health check returned status $status_code"
        exit 1
    fi

    log_success "All health checks passed"
}

# Performance optimization
optimize_performance() {
    log_info "Applying performance optimizations..."

    # Set container resource limits
    docker update --cpus="2.0" --memory="4g" xorb-api 2>/dev/null || true
    docker update --cpus="1.0" --memory="2g" xorb-orchestrator 2>/dev/null || true
    docker update --cpus="0.5" --memory="1g" xorb-redis 2>/dev/null || true

    # Configure PostgreSQL for performance
    docker exec xorb-postgres psql -U postgres -c "
        ALTER SYSTEM SET shared_buffers = '256MB';
        ALTER SYSTEM SET effective_cache_size = '1GB';
        ALTER SYSTEM SET maintenance_work_mem = '64MB';
        ALTER SYSTEM SET checkpoint_completion_target = 0.9;
        ALTER SYSTEM SET wal_buffers = '16MB';
        SELECT pg_reload_conf();
    " 2>/dev/null || log_warning "Failed to optimize PostgreSQL settings"

    # Set kernel parameters for high performance
    echo 'net.core.somaxconn = 65535' >> /etc/sysctl.conf
    echo 'net.core.netdev_max_backlog = 5000' >> /etc/sysctl.conf
    sysctl -p >/dev/null 2>&1 || true

    log_success "Performance optimizations applied"
}

# Setup monitoring and alerts
setup_monitoring() {
    log_info "Setting up monitoring and alerting..."

    # Start Prometheus and Grafana if configured
    if docker-compose -f infra/docker-compose.yml config --services | grep -q prometheus; then
        docker-compose -f infra/docker-compose.yml up -d prometheus grafana
        log_success "Monitoring services started"
    else
        log_warning "Monitoring services not configured"
    fi

    # Set up log monitoring
    mkdir -p /root/Xorb/logs

    # Create systemd service for log monitoring
    cat > /etc/systemd/system/xorb-monitor.service << 'EOF'
[Unit]
Description=XORB Log Monitor
After=docker.service

[Service]
Type=simple
ExecStart=/bin/bash -c 'tail -f /root/Xorb/logs/*.log | grep -E "(ERROR|CRITICAL|FATAL)" | logger -t xorb-alert'
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable xorb-monitor.service
    systemctl start xorb-monitor.service

    log_success "Monitoring and alerting configured"
}

# Rollback function
rollback_deployment() {
    log_warning "Rolling back deployment..."

    if [[ -f "/tmp/last_backup_file" ]]; then
        local backup_file=$(cat /tmp/last_backup_file)
        if [[ -f "$backup_file" ]]; then
            # Stop current services
            docker-compose -f infra/docker-compose.production.yml down || true

            # Restore backup
            cd /root
            tar -xzf "$backup_file"

            # Restart services
            cd /root/Xorb
            docker-compose -f infra/docker-compose.production.yml up -d

            log_success "Rollback completed"
            return 0
        fi
    fi

    log_error "No backup found for rollback"
    return 1
}

# Final validation
final_validation() {
    log_info "Performing final validation..."

    # Check all services are running
    local expected_services=("xorb-api" "xorb-orchestrator" "xorb-postgres" "xorb-redis")
    local running_services=$(docker ps --format "table {{.Names}}" | grep xorb | wc -l)

    if [[ $running_services -lt ${#expected_services[@]} ]]; then
        log_error "Not all services are running (expected: ${#expected_services[@]}, running: $running_services)"
        return 1
    fi

    # Test critical endpoints
    local critical_tests=(
        "curl -f -s http://localhost:8080/api/health"
        "curl -f -s http://localhost:8081/health"
    )

    for test in "${critical_tests[@]}"; do
        if ! eval "$test" >/dev/null 2>&1; then
            log_error "Critical test failed: $test"
            return 1
        fi
    done

    # Check logs for errors
    if docker logs xorb-api 2>&1 | grep -i "error" | tail -5 | grep -v "404" >/dev/null; then
        log_warning "Found recent errors in API logs:"
        docker logs xorb-api 2>&1 | grep -i "error" | tail -5 | grep -v "404"
    fi

    log_success "Final validation completed"
}

# Generate deployment report
generate_report() {
    log_info "Generating deployment report..."

    local report_file="/root/Xorb/deployment-report-$(date +%Y%m%d_%H%M%S).json"

    cat > "$report_file" << EOF
{
    "deployment": {
        "timestamp": "$(date -Iseconds)",
        "version": "$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')",
        "environment": "production",
        "status": "success"
    },
    "services": {
        "api": {
            "status": "$(docker inspect --format='{{.State.Status}}' xorb-api 2>/dev/null || echo 'unknown')",
            "health": "$(curl -s http://localhost:8080/api/health | jq -r '.status' 2>/dev/null || echo 'unknown')"
        },
        "orchestrator": {
            "status": "$(docker inspect --format='{{.State.Status}}' xorb-orchestrator 2>/dev/null || echo 'unknown')",
            "health": "$(curl -s http://localhost:8081/health | jq -r '.status' 2>/dev/null || echo 'unknown')"
        },
        "database": {
            "status": "$(docker inspect --format='{{.State.Status}}' xorb-postgres 2>/dev/null || echo 'unknown')"
        },
        "redis": {
            "status": "$(docker inspect --format='{{.State.Status}}' xorb-redis 2>/dev/null || echo 'unknown')"
        }
    },
    "security": {
        "jwt_secret_configured": "$([[ -n "$JWT_SECRET" ]] && echo 'true' || echo 'false')",
        "api_key_configured": "$([[ -n "$XORB_API_KEY" ]] && echo 'true' || echo 'false')",
        "rate_limiting_enabled": "$(echo ${ENABLE_RATE_LIMITING:-true})",
        "ssl_certificates": "$(ls -1 /root/Xorb/ssl/*.crt 2>/dev/null | wc -l)"
    },
    "performance": {
        "memory_usage": "$(free -m | awk 'NR==2{printf "%.1f", ($3/$2)*100}')",
        "disk_usage": "$(df -h / | awk 'NR==2{print $5}')",
        "docker_containers": "$(docker ps -q | wc -l)"
    }
}
EOF

    log_success "Deployment report generated: $report_file"
}

# Main deployment function
main() {
    log_info "Starting XORB production deployment..."
    log_info "Deployment log: $DEPLOYMENT_LOG"

    # Load environment variables if .env exists
    if [[ -f "/root/Xorb/.env" ]]; then
        source /root/Xorb/.env
        log_info "Environment variables loaded from .env file"
    fi

    # Execute deployment steps
    pre_deployment_checks
    create_backup
    security_hardening
    setup_database
    deploy_services
    optimize_performance
    setup_monitoring
    perform_health_checks
    final_validation
    generate_report

    log_success "XORB production deployment completed successfully!"
    log_info "Services available at:"
    log_info "  - API: https://localhost:8080"
    log_info "  - Orchestrator: https://localhost:8081"
    log_info "  - Grafana: http://localhost:3000 (if enabled)"
    log_info "  - Health: https://localhost:8080/api/health"

    log_info "Next steps:"
    log_info "  1. Monitor service logs: docker-compose logs -f"
    log_info "  2. Check deployment report: $(ls -t /root/Xorb/deployment-report-*.json | head -1)"
    log_info "  3. Verify SSL certificates are properly configured"
    log_info "  4. Run security scan: nmap -sC -sV localhost"
}

# Execute main function
main "$@"
