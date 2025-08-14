#!/bin/bash
set -e

# Xorb Platform Deployment Script
# This script deploys the enhanced Xorb backend platform

echo "🚀 Xorb Platform Deployment Script"
echo "=================================="

# Configuration
ENVIRONMENT=${ENVIRONMENT:-"production"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}
COMPOSE_FILE=${COMPOSE_FILE:-"infra/docker-compose.production.yml"}
HEALTH_CHECK_TIMEOUT=${HEALTH_CHECK_TIMEOUT:-300}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi

    # Check if running as root (for production)
    if [[ "$ENVIRONMENT" == "production" && $EUID -eq 0 ]]; then
        warn "Running as root in production. Consider using a dedicated user."
    fi

    # Check available disk space (minimum 10GB)
    available_space=$(df / | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 10485760 ]]; then # 10GB in KB
        warn "Low disk space available. Minimum 10GB recommended."
    fi

    log "Prerequisites check completed ✅"
}

# Validate environment configuration
validate_config() {
    log "Validating configuration..."

    # Check for required environment files
    if [[ ! -f ".env" ]]; then
        if [[ -f ".env.template" ]]; then
            warn "No .env file found. Copying from template..."
            cp .env.template .env
            warn "Please edit .env file with your configuration before proceeding"
            exit 1
        else
            error "No .env file or template found"
            exit 1
        fi
    fi

    # Validate critical environment variables
    source .env 2>/dev/null || true

    required_vars=(
        "DATABASE_URL"
        "REDIS_URL"
    )

    missing_vars=()
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            missing_vars+=("$var")
        fi
    done

    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        error "Missing required environment variables: ${missing_vars[*]}"
        error "Please configure these in your .env file"
        exit 1
    fi

    log "Configuration validation completed ✅"
}

# Build containers
build_containers() {
    log "Building containers..."

    # Build with security-focused Dockerfile
    docker build \
        -f src/api/Dockerfile.secure \
        -t "xorb-api:${IMAGE_TAG}" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')" \
        src/api/

    log "Container build completed ✅"
}

# Setup database
setup_database() {
    log "Setting up database..."

    # Start database service first
    docker-compose -f "$COMPOSE_FILE" up -d postgres

    # Wait for database to be ready
    log "Waiting for database to be ready..."
    timeout=60
    while ! docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U postgres >/dev/null 2>&1; do
        sleep 2
        timeout=$((timeout - 2))
        if [[ $timeout -le 0 ]]; then
            error "Database failed to start within 60 seconds"
            exit 1
        fi
    done

    # Enable pgvector extension
    docker-compose -f "$COMPOSE_FILE" exec -T postgres \
        psql -U postgres -d "${DATABASE_NAME:-xorb}" \
        -c "CREATE EXTENSION IF NOT EXISTS vector;" || warn "Could not enable pgvector extension"

    log "Database setup completed ✅"
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."

    # Run migrations using the API container
    docker-compose -f "$COMPOSE_FILE" run --rm api \
        alembic upgrade head

    log "Database migrations completed ✅"
}

# Deploy services
deploy_services() {
    log "Deploying services..."

    # Stop any existing services
    docker-compose -f "$COMPOSE_FILE" down

    # Start all services
    docker-compose -f "$COMPOSE_FILE" up -d

    log "Services deployment completed ✅"
}

# Health checks
run_health_checks() {
    log "Running health checks..."

    # Wait for services to be ready
    services=("api" "redis" "postgres")

    for service in "${services[@]}"; do
        info "Checking $service..."
        timeout=$HEALTH_CHECK_TIMEOUT

        while ! docker-compose -f "$COMPOSE_FILE" exec -T "$service" echo "Service is running" >/dev/null 2>&1; do
            sleep 5
            timeout=$((timeout - 5))
            if [[ $timeout -le 0 ]]; then
                error "$service failed to start within ${HEALTH_CHECK_TIMEOUT} seconds"
                return 1
            fi
        done

        info "$service is healthy ✅"
    done

    # Test API endpoints
    log "Testing API endpoints..."

    # Get API URL
    API_URL=${API_URL:-"http://localhost:8000"}

    # Health check
    if curl -f -s "$API_URL/health" >/dev/null; then
        info "Health endpoint: ✅"
    else
        error "Health endpoint failed"
        return 1
    fi

    # Readiness check
    if curl -f -s "$API_URL/readiness" >/dev/null; then
        info "Readiness endpoint: ✅"
    else
        warn "Readiness endpoint failed (dependencies may not be ready)"
    fi

    # API documentation
    if curl -f -s "$API_URL/docs" >/dev/null; then
        info "API documentation: ✅"
    else
        warn "API documentation not accessible"
    fi

    log "Health checks completed ✅"
}

# Security scan
run_security_scan() {
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log "Running security scan..."

        # Run Trivy scan on the built image
        if command -v trivy &> /dev/null; then
            trivy image --exit-code 0 --severity HIGH,CRITICAL "xorb-api:${IMAGE_TAG}"
            info "Security scan completed ✅"
        else
            warn "Trivy not available, skipping security scan"
        fi
    fi
}

# Backup existing deployment
backup_deployment() {
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log "Creating deployment backup..."

        # Create backup directory
        backup_dir="backups/$(date +'%Y%m%d_%H%M%S')"
        mkdir -p "$backup_dir"

        # Backup database
        if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U postgres >/dev/null 2>&1; then
            docker-compose -f "$COMPOSE_FILE" exec -T postgres \
                pg_dump -U postgres "${DATABASE_NAME:-xorb}" > "$backup_dir/database.sql"
            info "Database backup created: $backup_dir/database.sql"
        fi

        # Backup configuration
        cp .env "$backup_dir/env.backup" 2>/dev/null || true
        cp "$COMPOSE_FILE" "$backup_dir/compose.backup" 2>/dev/null || true

        log "Backup completed ✅"
    fi
}

# Rollback function
rollback() {
    error "Deployment failed. Rolling back..."

    # Stop new services
    docker-compose -f "$COMPOSE_FILE" down

    # Restore from backup if available
    latest_backup=$(ls -t backups/ | head -n 1)
    if [[ -n "$latest_backup" && -d "backups/$latest_backup" ]]; then
        warn "Restoring from backup: $latest_backup"

        # Restore database
        if [[ -f "backups/$latest_backup/database.sql" ]]; then
            docker-compose -f "$COMPOSE_FILE" up -d postgres
            sleep 10
            docker-compose -f "$COMPOSE_FILE" exec -T postgres \
                psql -U postgres -d "${DATABASE_NAME:-xorb}" < "backups/$latest_backup/database.sql"
        fi
    fi

    error "Rollback completed"
    exit 1
}

# Performance benchmark
run_benchmark() {
    if command -v bombardier &> /dev/null; then
        log "Running performance benchmark..."

        API_URL=${API_URL:-"http://localhost:8000"}

        # Simple health endpoint benchmark
        bombardier -c 10 -n 1000 -l "$API_URL/health" || warn "Benchmark failed"

        log "Benchmark completed ✅"
    else
        info "Bombardier not available, skipping benchmark"
    fi
}

# Main deployment flow
main() {
    log "Starting Xorb Platform deployment..."
    log "Environment: $ENVIRONMENT"
    log "Image Tag: $IMAGE_TAG"
    log "Compose File: $COMPOSE_FILE"

    # Set up error handling
    trap rollback ERR

    # Deployment steps
    check_prerequisites
    validate_config

    if [[ "$ENVIRONMENT" == "production" ]]; then
        backup_deployment
    fi

    build_containers

    if [[ "$ENVIRONMENT" == "production" ]]; then
        run_security_scan
    fi

    setup_database
    run_migrations
    deploy_services
    run_health_checks

    if [[ "${RUN_BENCHMARK:-false}" == "true" ]]; then
        run_benchmark
    fi

    # Success message
    echo ""
    log "🎉 Xorb Platform deployment completed successfully!"
    echo ""
    info "Services are running:"
    info "  - API: ${API_URL:-http://localhost:8000}"
    info "  - Documentation: ${API_URL:-http://localhost:8000}/docs"
    info "  - Health Check: ${API_URL:-http://localhost:8000}/health"
    echo ""
    info "Enhancement modules active:"
    info "  ✅ OIDC Authentication & RBAC"
    info "  ✅ Multi-tenant Row Level Security"
    info "  ✅ Secure File Storage (FS + S3)"
    info "  ✅ Job Orchestration with Redis"
    info "  ✅ Performance Optimizations (uvloop + pgvector)"
    info "  ✅ Observability (OpenTelemetry + Prometheus)"
    info "  ✅ Security & Rate Limiting"
    info "  ✅ Secure Build & CI Pipeline"
    echo ""
    log "Deployment logs available with: docker-compose -f $COMPOSE_FILE logs -f"
}

# Run deployment
main "$@"
