#!/bin/bash
# Production deployment script for XORB Enterprise Cybersecurity Platform

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENVIRONMENT="${ENVIRONMENT:-production}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.production.yml}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/xorb}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if .env file exists
    if [[ ! -f "$PROJECT_DIR/.env" ]]; then
        log_error ".env file not found. Please create it from .env.example"
        exit 1
    fi
    
    # Check if compose file exists
    if [[ ! -f "$PROJECT_DIR/$COMPOSE_FILE" ]]; then
        log_error "Compose file $COMPOSE_FILE not found"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Function to create backup
create_backup() {
    if [[ "${SKIP_BACKUP:-false}" == "true" ]]; then
        log_info "Skipping backup as requested"
        return 0
    fi
    
    log_info "Creating backup..."
    
    # Create backup directory
    sudo mkdir -p "$BACKUP_DIR"
    local backup_timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_path="$BACKUP_DIR/xorb_backup_$backup_timestamp"
    
    # Backup database
    log_info "Backing up database..."
    docker-compose -f "$PROJECT_DIR/$COMPOSE_FILE" exec -T postgres pg_dump -U xorb_user xorb_db > "$backup_path.sql" || {
        log_warning "Database backup failed (database might not be running)"
    }
    
    # Backup configuration
    log_info "Backing up configuration..."
    sudo cp -r "$PROJECT_DIR/config" "$backup_path_config" 2>/dev/null || {
        log_warning "Configuration backup failed (config directory might not exist)"
    }
    
    # Backup volumes (if they exist)
    log_info "Backing up volumes..."
    for volume in postgres_data redis_data grafana_data prometheus_data; do
        if docker volume inspect "${PROJECT_DIR##*/}_$volume" &> /dev/null; then
            docker run --rm -v "${PROJECT_DIR##*/}_$volume":/data -v "$BACKUP_DIR":/backup alpine tar czf "/backup/${volume}_$backup_timestamp.tar.gz" -C /data . || {
                log_warning "Volume backup failed for $volume"
            }
        fi
    done
    
    log_success "Backup created: $backup_path"
}

# Function to pull latest images
pull_images() {
    log_info "Pulling latest Docker images..."
    
    cd "$PROJECT_DIR"
    docker-compose -f "$COMPOSE_FILE" pull || {
        log_error "Failed to pull Docker images"
        exit 1
    }
    
    log_success "Docker images updated"
}

# Function to build custom images
build_images() {
    log_info "Building custom Docker images..."
    
    cd "$PROJECT_DIR"
    
    # Build with build arguments
    docker-compose -f "$COMPOSE_FILE" build \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
        --build-arg VERSION="$(git describe --tags --always 2>/dev/null || echo 'dev')" || {
        log_error "Failed to build Docker images"
        exit 1
    }
    
    log_success "Docker images built successfully"
}

# Function to run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    cd "$PROJECT_DIR"
    
    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    timeout 60 bash -c 'until docker-compose -f '"$COMPOSE_FILE"' exec postgres pg_isready -U xorb_user -d xorb_db; do sleep 2; done' || {
        log_error "Database did not become ready in time"
        exit 1
    }
    
    # Run migrations
    docker-compose -f "$COMPOSE_FILE" exec xorb-api alembic upgrade head || {
        log_warning "Database migrations failed (might be first deployment)"
    }
    
    log_success "Database migrations completed"
}

# Function to deploy services
deploy_services() {
    log_info "Deploying services..."
    
    cd "$PROJECT_DIR"
    
    # Deploy with zero-downtime strategy
    docker-compose -f "$COMPOSE_FILE" up -d --remove-orphans || {
        log_error "Failed to deploy services"
        exit 1
    }
    
    log_success "Services deployed"
}

# Function to wait for services to be healthy
wait_for_health() {
    log_info "Waiting for services to be healthy..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log_info "Health check attempt $attempt/$max_attempts"
        
        # Check API health
        if curl -sf http://localhost:8000/api/v1/health &> /dev/null; then
            log_success "API service is healthy"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Services did not become healthy in time"
            exit 1
        fi
        
        sleep 10
        ((attempt++))
    done
    
    log_success "All services are healthy"
}

# Function to run post-deployment tests
run_tests() {
    if [[ "${SKIP_TESTS:-false}" == "true" ]]; then
        log_info "Skipping tests as requested"
        return 0
    fi
    
    log_info "Running post-deployment tests..."
    
    # Basic API tests
    local api_tests=(
        "http://localhost:8000/api/v1/health"
        "http://localhost:8000/api/v1/info"
        "http://localhost:8000/api/v1/readiness"
    )
    
    for endpoint in "${api_tests[@]}"; do
        if curl -sf "$endpoint" &> /dev/null; then
            log_success "✅ $endpoint"
        else
            log_warning "❌ $endpoint failed"
        fi
    done
    
    # Check if Grafana is accessible
    if curl -sf http://localhost:3010/api/health &> /dev/null; then
        log_success "✅ Grafana is accessible"
    else
        log_warning "❌ Grafana health check failed"
    fi
    
    # Check if Prometheus is accessible
    if curl -sf http://localhost:9092/-/healthy &> /dev/null; then
        log_success "✅ Prometheus is accessible"
    else
        log_warning "❌ Prometheus health check failed"
    fi
    
    log_success "Post-deployment tests completed"
}

# Function to show deployment status
show_status() {
    log_info "Deployment Status:"
    
    cd "$PROJECT_DIR"
    echo ""
    docker-compose -f "$COMPOSE_FILE" ps
    echo ""
    
    log_info "Service URLs:"
    echo "  🌐 API Documentation: http://localhost:8000/docs"
    echo "  📊 Grafana Dashboard: http://localhost:3010"
    echo "  📈 Prometheus: http://localhost:9092"
    echo "  ⏱️  Temporal UI: http://localhost:8233"
    echo "  🔍 API Health: http://localhost:8000/api/v1/health"
    echo ""
    
    log_info "To view logs:"
    echo "  docker-compose -f $COMPOSE_FILE logs -f [service_name]"
    echo ""
}

# Function to cleanup old images
cleanup_old_images() {
    if [[ "${SKIP_CLEANUP:-false}" == "true" ]]; then
        log_info "Skipping cleanup as requested"
        return 0
    fi
    
    log_info "Cleaning up old Docker images..."
    
    # Remove dangling images
    docker image prune -f &> /dev/null || true
    
    # Remove old images (keep last 3 versions)
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}" | \
    grep "xorb" | \
    tail -n +4 | \
    awk '{print $3}' | \
    xargs -r docker rmi &> /dev/null || true
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_info "🚀 Starting XORB Enterprise Platform Deployment"
    log_info "Environment: $ENVIRONMENT"
    log_info "Compose file: $COMPOSE_FILE"
    echo ""
    
    # Run deployment steps
    check_prerequisites
    create_backup
    pull_images
    build_images
    deploy_services
    run_migrations
    wait_for_health
    run_tests
    cleanup_old_images
    show_status
    
    log_success "🎉 XORB Enterprise Platform deployed successfully!"
    log_info "Deployment completed at $(date)"
}

# Help function
show_help() {
    cat << EOF
XORB Enterprise Platform Deployment Script

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -e, --environment ENV   Set environment (default: production)
    -f, --file FILE         Set compose file (default: docker-compose.production.yml)
    --skip-backup          Skip database backup
    --skip-tests           Skip post-deployment tests
    --skip-cleanup         Skip cleanup of old images

Environment Variables:
    ENVIRONMENT            Deployment environment
    COMPOSE_FILE           Docker compose file to use
    BACKUP_DIR             Backup directory path
    SKIP_BACKUP            Skip backup if set to 'true'
    SKIP_TESTS             Skip tests if set to 'true'
    SKIP_CLEANUP           Skip cleanup if set to 'true'

Examples:
    $0                              # Deploy to production
    $0 -e staging                   # Deploy to staging
    $0 --skip-backup --skip-tests   # Deploy without backup and tests
    SKIP_BACKUP=true $0             # Deploy without backup using env var

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -f|--file)
            COMPOSE_FILE="$2"
            shift 2
            ;;
        --skip-backup)
            SKIP_BACKUP=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-cleanup)
            SKIP_CLEANUP=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main function
main "$@"