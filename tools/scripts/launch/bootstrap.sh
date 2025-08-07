#!/bin/bash
# XORB Ecosystem Bootstrap Script
# Detects environment and launches appropriate configuration

set -euo pipefail

# Color output functions
info() { echo -e "\033[1;34m[INFO]\033[0m $1"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $1"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $1"; }
success() { echo -e "\033[1;32m[SUCCESS]\033[0m $1"; }

# Environment detection
detect_environment() {
    local env="dev"
    
    # Check for production indicators
    if [[ -f /etc/xorb/production.flag ]] || [[ "${XORB_ENV:-}" == "production" ]]; then
        env="production"
    elif [[ -f /etc/xorb/staging.flag ]] || [[ "${XORB_ENV:-}" == "staging" ]]; then
        env="staging"
    elif [[ $(uname -m) == "aarch64" ]] && [[ -f /proc/device-tree/model ]] && grep -q "Raspberry Pi" /proc/device-tree/model; then
        env="rpi"
    fi
    
    echo "$env"
}

# Resource configuration based on environment
configure_resources() {
    local env=$1
    
    case $env in
        "production")
            export XORB_MAX_AGENTS=32
            export XORB_WORKER_CONCURRENCY=16
            export XORB_MEMORY_LIMIT="32Gi"
            export XORB_CPU_LIMIT="64"
            ;;
        "staging")
            export XORB_MAX_AGENTS=8
            export XORB_WORKER_CONCURRENCY=4
            export XORB_MEMORY_LIMIT="8Gi"
            export XORB_CPU_LIMIT="8"
            ;;
        "rpi")
            export XORB_MAX_AGENTS=2
            export XORB_WORKER_CONCURRENCY=1
            export XORB_MEMORY_LIMIT="4Gi"
            export XORB_CPU_LIMIT="4"
            ;;
        *)
            export XORB_MAX_AGENTS=4
            export XORB_WORKER_CONCURRENCY=2
            export XORB_MEMORY_LIMIT="4Gi"
            export XORB_CPU_LIMIT="4"
            ;;
    esac
    
    info "Configured for environment: $env"
    info "Max agents: $XORB_MAX_AGENTS, Worker concurrency: $XORB_WORKER_CONCURRENCY"
}

# Docker compose configuration selection
select_compose_config() {
    local env=$1
    local compose_file="docker-compose.yml"
    
    case $env in
        "production")
            compose_file="docker-compose.production.yml"
            ;;
        "staging")
            compose_file="docker-compose.yml"
            ;;
        "rpi")
            compose_file="docker-compose.yml"
            ;;
    esac
    
    echo "$compose_file"
}

# Health check function
health_check() {
    info "Performing health checks..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check available memory
    local available_mem=$(free -m | awk 'NR==2{printf "%.1f", $7/1024}')
    if (( $(echo "$available_mem < 2" | bc -l) )); then
        warn "Low available memory: ${available_mem}GB. XORB may not function optimally."
    fi
    
    success "Health checks passed"
}

# Main bootstrap function
main() {
    info "🚀 XORB Ecosystem Bootstrap Starting..."
    
    # Detect environment
    local env=$(detect_environment)
    info "Detected environment: $env"
    
    # Health checks
    health_check
    
    # Configure resources
    configure_resources "$env"
    
    # Select compose configuration
    local compose_file=$(select_compose_config "$env")
    info "Using compose file: $compose_file"
    
    # Create .env file if it doesn't exist
    if [[ ! -f .env ]]; then
        info "Creating .env file from template..."
        cp config/.xorb.env .env 2>/dev/null || warn ".env template not found, creating minimal .env"
        cat > .env << EOF
XORB_ENV=$env
XORB_MAX_AGENTS=$XORB_MAX_AGENTS
XORB_WORKER_CONCURRENCY=$XORB_WORKER_CONCURRENCY
XORB_MEMORY_LIMIT=$XORB_MEMORY_LIMIT
XORB_CPU_LIMIT=$XORB_CPU_LIMIT
EOF
    fi
    
    # Start services
    info "Starting XORB ecosystem..."
    if command -v docker-compose &> /dev/null; then
        docker-compose -f "$compose_file" up -d
    else
        docker compose -f "$compose_file" up -d
    fi
    
    # Wait for services to be ready
    info "Waiting for services to be ready..."
    sleep 30
    
    # Verify deployment
    info "Verifying deployment..."
    if curl -sf http://localhost:8000/health &> /dev/null; then
        success "✅ XORB API is responding"
    else
        warn "⚠️  XORB API may not be ready yet"
    fi
    
    success "🎉 XORB Ecosystem bootstrap complete!"
    info "Access the API at: http://localhost:8000"
    info "View logs with: docker-compose -f $compose_file logs -f"
}

# Run main function
main "$@"