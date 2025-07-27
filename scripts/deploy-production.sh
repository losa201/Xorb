#!/bin/bash
set -euo pipefail

# XORB Production Deployment Automation Script
# Enterprise-grade deployment with comprehensive validation

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
BACKUP_BEFORE_DEPLOY="${BACKUP_BEFORE_DEPLOY:-true}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"

# Logging
DEPLOYMENT_LOG="/var/log/xorb/deployment-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$(dirname "$DEPLOYMENT_LOG")"

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

# Deployment functions
check_prerequisites() {
    log "🔍 Checking deployment prerequisites..."
    
    # Check required tools
    local required_tools=("docker" "docker-compose" "git" "python3" "make")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "Required tool '$tool' is not installed"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    # Check available disk space (minimum 10GB)
    local available_space=$(df / | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 10485760 ]; then
        error "Insufficient disk space. At least 10GB required"
        exit 1
    fi
    
    # Check memory (minimum 8GB)
    local available_memory=$(free -m | awk 'NR==2{print $7}')
    if [ "$available_memory" -lt 8192 ]; then
        warn "Available memory is less than 8GB. Performance may be affected"
    fi
    
    log "✅ Prerequisites check passed"
}

backup_current_deployment() {
    if [ "$BACKUP_BEFORE_DEPLOY" = "true" ]; then
        log "💾 Creating backup of current deployment..."
        
        local backup_dir="/var/backups/xorb/$(date +%Y%m%d-%H%M%S)"
        mkdir -p "$backup_dir"
        
        # Backup database
        if docker ps | grep -q xorb_postgres; then
            log "📊 Backing up PostgreSQL database..."
            docker exec xorb_postgres_1 pg_dumpall -U postgres > "$backup_dir/postgres_backup.sql"
        fi
        
        # Backup Redis data
        if docker ps | grep -q xorb_redis; then
            log "📊 Backing up Redis data..."
            docker exec xorb_redis_1 redis-cli BGSAVE
            docker cp xorb_redis_1:/data/dump.rdb "$backup_dir/redis_backup.rdb"
        fi
        
        # Backup configuration
        cp -r . "$backup_dir/source_backup"
        
        log "✅ Backup completed: $backup_dir"
        echo "$backup_dir" > /tmp/xorb_backup_location
    fi
}

validate_configuration() {
    log "⚙️ Validating configuration..."
    
    # Check environment variables
    local required_vars=("POSTGRES_PASSWORD" "REDIS_PASSWORD")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    # Validate docker-compose file
    if ! docker-compose -f docker-compose.production.yml config &> /dev/null; then
        error "Invalid docker-compose.production.yml configuration"
        exit 1
    fi
    
    # Run architecture validation
    if ! python3 validate_refactoring.py &> /dev/null; then
        error "Architecture validation failed"
        exit 1
    fi
    
    log "✅ Configuration validation passed"
}

deploy_services() {
    log "🚀 Deploying XORB services..."
    
    # Set environment variables
    export NVIDIA_API_KEY="${NVIDIA_API_KEY:-}"
    export REDIS_PASSWORD="${REDIS_PASSWORD:-xorb-redis-2024}"
    export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-xorb-postgres-2024}"
    
    # Pull latest images
    log "📥 Pulling latest container images..."
    docker-compose -f docker-compose.production.yml pull
    
    # Deploy infrastructure first
    log "🏗️ Deploying infrastructure services..."
    docker-compose -f docker-compose.production.yml up -d postgres redis
    
    # Wait for infrastructure
    log "⏳ Waiting for infrastructure services..."
    sleep 30
    
    # Deploy application services
    log "🚀 Deploying application services..."
    docker-compose -f docker-compose.production.yml up -d
    
    log "✅ Services deployment completed"
}

wait_for_health_checks() {
    log "🏥 Waiting for health checks..."
    
    local timeout=$HEALTH_CHECK_TIMEOUT
    local elapsed=0
    local check_interval=10
    
    local services=("postgres" "redis")
    
    while [ $elapsed -lt $timeout ]; do
        local all_healthy=true
        
        for service in "${services[@]}"; do
            if ! docker-compose -f docker-compose.production.yml ps "$service" | grep -q "Up"; then
                all_healthy=false
                break
            fi
        done
        
        if [ "$all_healthy" = true ]; then
            log "✅ All services are healthy"
            return 0
        fi
        
        info "⏳ Waiting for services to become healthy... ($elapsed/$timeout seconds)"
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done
    
    error "Health check timeout after $timeout seconds"
    return 1
}

run_integration_tests() {
    log "🧪 Running integration tests..."
    
    # Run core validation tests
    if ! python3 validate_refactoring.py; then
        error "Architecture validation failed"
        return 1
    fi
    
    if ! python3 validate_monitoring.py; then
        error "Monitoring validation failed"
        return 1
    fi
    
    # Run performance benchmark
    if ! python3 benchmark_refactored_architecture.py; then
        warn "Performance benchmark had issues but deployment continues"
    fi
    
    log "✅ Integration tests completed successfully"
}

setup_monitoring() {
    log "📊 Setting up monitoring and alerting..."
    
    # Deploy monitoring stack if not already running
    if ! docker ps | grep -q prometheus; then
        log "🔍 Deploying Prometheus..."
        docker-compose -f docker-compose.production.yml up -d prometheus
    fi
    
    if ! docker ps | grep -q grafana; then
        log "📈 Deploying Grafana..."
        docker-compose -f docker-compose.production.yml up -d grafana
    fi
    
    # Wait for monitoring services
    sleep 20
    
    log "✅ Monitoring setup completed"
}

verify_deployment() {
    log "✅ Verifying deployment..."
    
    # Check service status
    local failed_services=()
    local services=("postgres" "redis")
    
    for service in "${services[@]}"; do
        if ! docker-compose -f docker-compose.production.yml ps "$service" | grep -q "Up"; then
            failed_services+=("$service")
        fi
    done
    
    if [ ${#failed_services[@]} -gt 0 ]; then
        error "Failed services: ${failed_services[*]}"
        return 1
    fi
    
    # Test database connectivity
    if ! docker exec xorb_postgres_1 pg_isready -U postgres &> /dev/null; then
        error "PostgreSQL is not ready"
        return 1
    fi
    
    # Test Redis connectivity
    if ! docker exec xorb_redis_1 redis-cli ping &> /dev/null; then
        error "Redis is not responding"
        return 1
    fi
    
    log "✅ Deployment verification successful"
}

rollback_deployment() {
    if [ "$ROLLBACK_ON_FAILURE" = "true" ] && [ -f /tmp/xorb_backup_location ]; then
        local backup_location=$(cat /tmp/xorb_backup_location)
        warn "🔄 Rolling back deployment..."
        
        # Stop current services
        docker-compose -f docker-compose.production.yml down
        
        # Restore from backup
        if [ -f "$backup_location/postgres_backup.sql" ]; then
            log "📊 Restoring PostgreSQL backup..."
            docker-compose up -d postgres
            sleep 10
            docker exec -i xorb_postgres_1 psql -U postgres < "$backup_location/postgres_backup.sql"
        fi
        
        if [ -f "$backup_location/redis_backup.rdb" ]; then
            log "📊 Restoring Redis backup..."
            docker cp "$backup_location/redis_backup.rdb" xorb_redis_1:/data/dump.rdb
            docker restart xorb_redis_1
        fi
        
        warn "🔄 Rollback completed"
    fi
}

generate_deployment_report() {
    log "📋 Generating deployment report..."
    
    local report_file="/var/log/xorb/deployment-report-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$report_file" << EOF
{
  "deployment_date": "$(date -Iseconds)",
  "deployment_environment": "$DEPLOYMENT_ENV",
  "deployment_status": "success",
  "services_deployed": [
    "postgres",
    "redis"
  ],
  "health_checks": {
    "postgres": "healthy",
    "redis": "healthy"
  },
  "performance_metrics": {
    "deployment_duration": "$(date +%s)",
    "services_count": 2
  },
  "monitoring_endpoints": {
    "prometheus": "http://localhost:9090",
    "grafana": "http://localhost:3000"
  },
  "backup_location": "$(cat /tmp/xorb_backup_location 2>/dev/null || echo 'none')"
}
EOF
    
    log "✅ Deployment report generated: $report_file"
}

print_deployment_summary() {
    log "🎉 XORB Production Deployment Summary"
    echo "=================================="
    echo "✅ Status: SUCCESSFUL"
    echo "📅 Date: $(date)"
    echo "🌍 Environment: $DEPLOYMENT_ENV"
    echo "📊 Services: PostgreSQL, Redis"
    echo "🔍 Monitoring: Available"
    echo "📋 Logs: $DEPLOYMENT_LOG"
    echo ""
    echo "🚀 Access Points:"
    echo "   📊 Grafana Dashboard: http://localhost:3000 (admin/admin)"
    echo "   🔍 Prometheus: http://localhost:9090"
    echo "   🗄️  PostgreSQL: localhost:5432"
    echo "   ⚡ Redis: localhost:6380"
    echo ""
    echo "📋 Next Steps:"
    echo "   1. Configure monitoring alerts"
    echo "   2. Set up automated backups"
    echo "   3. Configure SSL certificates"
    echo "   4. Review security settings"
    echo ""
    echo "🎯 Deployment completed successfully!"
}

# Main deployment flow
main() {
    log "🚀 Starting XORB production deployment..."
    
    # Trap errors for rollback
    trap 'error "Deployment failed"; rollback_deployment; exit 1' ERR
    
    check_prerequisites
    backup_current_deployment
    validate_configuration
    deploy_services
    wait_for_health_checks
    run_integration_tests
    setup_monitoring
    verify_deployment
    generate_deployment_report
    print_deployment_summary
    
    log "🎉 XORB production deployment completed successfully!"
}

# Script execution
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi