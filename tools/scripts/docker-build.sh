#!/bin/bash

# XORB Docker Multi-stage Build and Management Script
# Optimized container building, caching, and deployment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." &> /dev/null && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
ENVIRONMENT="${ENVIRONMENT:-development}"
BUILD_TARGET="${BUILD_TARGET:-development}"
PUSH_IMAGES="${PUSH_IMAGES:-false}"
USE_CACHE="${USE_CACHE:-true}"
PARALLEL_BUILDS="${PARALLEL_BUILDS:-true}"
REGISTRY="${REGISTRY:-xorb}"
VERSION="${VERSION:-latest}"
SERVICES=""
DRY_RUN=false
VERBOSE=false

log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >&2
}

log_info() {
    log "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    log "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [[ "$VERBOSE" == "true" ]]; then
        log "${BLUE}[DEBUG]${NC} $1"
    fi
}

usage() {
    cat << EOF
XORB Docker Build Management Tool

Usage: $0 [OPTIONS] COMMAND [SERVICES...]

Commands:
  build                    Build all services or specified services
  push                     Push images to registry
  pull                     Pull images from registry
  clean                    Clean up Docker images and containers
  prune                    Prune unused Docker resources
  test                     Run container tests
  validate                 Validate Docker configurations
  security-scan            Run security scans on images
  size-report              Generate image size report
  deploy                   Deploy services using Docker Compose
  logs                     Show logs for services

Services:
  api                      API service
  orchestrator             Orchestrator service
  worker                   Worker service
  all                      All services (default)

Options:
  -e, --environment ENV    Target environment (development|staging|production)
  -t, --target TARGET      Build target (development|production|testing)
  -r, --registry REGISTRY  Docker registry (default: xorb)
  -v, --version VERSION    Image version tag (default: latest)
  -p, --push              Push images after building
  -n, --no-cache          Disable Docker build cache
  -j, --parallel          Enable parallel builds (default: true)
  -d, --dry-run           Show what would be done without executing
  -V, --verbose           Verbose output
  -h, --help              Show this help message

Examples:
  $0 build --environment production --target production api orchestrator
  $0 build --push --version v1.2.3 all
  $0 deploy --environment development
  $0 clean --environment staging
  $0 security-scan api

EOF
}

check_dependencies() {
    local deps=("docker" "docker-compose" "jq")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [[ ${#missing_deps[@]} -ne 0 ]]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Please install missing dependencies before proceeding"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_debug "All dependencies are available"
}

get_services() {
    if [[ ${#SERVICES[@]} -eq 0 || " ${SERVICES[*]} " =~ " all " ]]; then
        echo "api orchestrator worker"
    else
        echo "${SERVICES[*]}"
    fi
}

build_service() {
    local service="$1"
    local dockerfile="src/$service/Dockerfile"
    local image_name="$REGISTRY/$service"
    local full_tag="$image_name:$VERSION-$BUILD_TARGET"
    local cache_tag="$image_name:${BUILD_TARGET}-cache"
    
    log_info "Building $service service (target: $BUILD_TARGET)"
    
    if [[ ! -f "$PROJECT_ROOT/$dockerfile" ]]; then
        log_error "Dockerfile not found: $dockerfile"
        return 1
    fi
    
    local build_args=""
    
    # Add cache arguments if enabled
    if [[ "$USE_CACHE" == "true" ]]; then
        build_args="--cache-from $cache_tag"
    else
        build_args="--no-cache"
    fi
    
    # Build command
    local build_cmd="docker build \
        --target $BUILD_TARGET \
        --tag $full_tag \
        --tag $image_name:latest \
        --tag $cache_tag \
        --file $dockerfile \
        --label org.opencontainers.image.created=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
        --label org.opencontainers.image.version=$VERSION \
        --label org.opencontainers.image.revision=$(git rev-parse HEAD 2>/dev/null || echo 'unknown') \
        --label xorb.environment=$ENVIRONMENT \
        --label xorb.build-target=$BUILD_TARGET \
        $build_args \
        $PROJECT_ROOT"
    
    log_debug "Build command: $build_cmd"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        if eval "$build_cmd"; then
            log_info "✅ Successfully built $service ($full_tag)"
            return 0
        else
            log_error "❌ Failed to build $service"
            return 1
        fi
    else
        log_info "[DRY RUN] Would build $service with: $build_cmd"
    fi
}

build_services() {
    local services=($(get_services))
    local failed_builds=()
    local build_pids=()
    
    log_info "Building services: ${services[*]}"
    log_info "Environment: $ENVIRONMENT, Target: $BUILD_TARGET"
    
    # Build services
    if [[ "$PARALLEL_BUILDS" == "true" && ${#services[@]} -gt 1 ]]; then
        log_info "Building services in parallel..."
        
        for service in "${services[@]}"; do
            if [[ "$DRY_RUN" == "false" ]]; then
                build_service "$service" &
                build_pids+=($!)
            else
                build_service "$service"
            fi
        done
        
        # Wait for all builds to complete
        if [[ "$DRY_RUN" == "false" ]]; then
            for pid in "${build_pids[@]}"; do
                if ! wait "$pid"; then
                    failed_builds+=("pid_$pid")
                fi
            done
        fi
    else
        log_info "Building services sequentially..."
        
        for service in "${services[@]}"; do
            if ! build_service "$service"; then
                failed_builds+=("$service")
            fi
        done
    fi
    
    # Report results
    if [[ ${#failed_builds[@]} -eq 0 ]]; then
        log_info "✅ All service builds completed successfully"
        return 0
    else
        log_error "❌ Failed builds: ${failed_builds[*]}"
        return 1
    fi
}

push_services() {
    local services=($(get_services))
    
    log_info "Pushing services to registry: $REGISTRY"
    
    for service in "${services[@]}"; do
        local image_name="$REGISTRY/$service"
        local full_tag="$image_name:$VERSION-$BUILD_TARGET"
        local cache_tag="$image_name:${BUILD_TARGET}-cache"
        
        if [[ "$DRY_RUN" == "false" ]]; then
            log_info "Pushing $service..."
            
            # Push main tag
            if docker push "$full_tag"; then
                log_info "✅ Pushed $full_tag"
            else
                log_error "❌ Failed to push $full_tag"
                return 1
            fi
            
            # Push cache tag for build optimization
            if docker push "$cache_tag"; then
                log_debug "✅ Pushed cache tag $cache_tag"
            else
                log_warn "⚠️ Failed to push cache tag $cache_tag"
            fi
            
            # Push latest tag if this is a production build
            if [[ "$BUILD_TARGET" == "production" ]]; then
                if docker push "$image_name:latest"; then
                    log_debug "✅ Pushed latest tag for $service"
                fi
            fi
        else
            log_info "[DRY RUN] Would push: $full_tag"
        fi
    done
    
    log_info "✅ All services pushed successfully"
}

pull_services() {
    local services=($(get_services))
    
    log_info "Pulling services from registry: $REGISTRY"
    
    for service in "${services[@]}"; do
        local image_name="$REGISTRY/$service"
        local full_tag="$image_name:$VERSION-$BUILD_TARGET"
        
        if [[ "$DRY_RUN" == "false" ]]; then
            log_info "Pulling $service..."
            
            if docker pull "$full_tag"; then
                log_info "✅ Pulled $full_tag"
            else
                log_error "❌ Failed to pull $full_tag"
                return 1
            fi
        else
            log_info "[DRY RUN] Would pull: $full_tag"
        fi
    done
}

clean_services() {
    log_info "Cleaning up Docker images and containers for environment: $ENVIRONMENT"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        # Stop and remove containers
        log_info "Stopping containers..."
        docker-compose -f "docker-compose.$ENVIRONMENT.yml" down || true
        
        # Remove images
        log_info "Removing images..."
        local services=($(get_services))
        for service in "${services[@]}"; do
            local image_name="$REGISTRY/$service"
            docker rmi "$image_name:$VERSION-$BUILD_TARGET" 2>/dev/null || true
            docker rmi "$image_name:latest" 2>/dev/null || true
        done
        
        log_info "✅ Cleanup completed"
    else
        log_info "[DRY RUN] Would clean up containers and images for $ENVIRONMENT"
    fi
}

security_scan() {
    local services=($(get_services))
    
    log_info "Running security scans on images..."
    
    # Check if trivy is available
    if ! command -v trivy &> /dev/null; then
        log_warn "Trivy not found - installing..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y wget apt-transport-https gnupg lsb-release
            wget -qO - https://aquasecurity.github.io/trivy-repo/deb/public.key | sudo apt-key add -
            echo "deb https://aquasecurity.github.io/trivy-repo/deb $(lsb_release -sc) main" | sudo tee -a /etc/apt/sources.list.d/trivy.list
            sudo apt-get update && sudo apt-get install -y trivy
        else
            log_error "Cannot install trivy automatically. Please install manually."
            return 1
        fi
    fi
    
    for service in "${services[@]}"; do
        local image_name="$REGISTRY/$service:$VERSION-$BUILD_TARGET"
        
        log_info "Scanning $service for vulnerabilities..."
        
        if [[ "$DRY_RUN" == "false" ]]; then
            # Run Trivy scan
            trivy image \
                --format table \
                --severity HIGH,CRITICAL \
                --exit-code 1 \
                "$image_name" || {
                log_error "❌ Security vulnerabilities found in $service"
                return 1
            }
        else
            log_info "[DRY RUN] Would scan: $image_name"
        fi
    done
    
    log_info "✅ Security scans completed successfully"
}

size_report() {
    local services=($(get_services))
    
    log_info "Generating image size report..."
    
    printf "%-20s %-30s %-10s\n" "SERVICE" "IMAGE" "SIZE"
    printf "%s\n" "$(printf '=%.0s' {1..70})"
    
    for service in "${services[@]}"; do
        local image_name="$REGISTRY/$service:$VERSION-$BUILD_TARGET"
        
        if docker image inspect "$image_name" >/dev/null 2>&1; then
            local size=$(docker image inspect "$image_name" --format='{{.Size}}' | numfmt --to=iec)
            printf "%-20s %-30s %-10s\n" "$service" "$image_name" "$size"
        else
            printf "%-20s %-30s %-10s\n" "$service" "$image_name" "NOT_FOUND"
        fi
    done
}

deploy_services() {
    local compose_file="docker-compose.$ENVIRONMENT.yml"
    
    log_info "Deploying services using $compose_file"
    
    if [[ ! -f "$PROJECT_ROOT/$compose_file" ]]; then
        log_error "Docker Compose file not found: $compose_file"
        return 1
    fi
    
    if [[ "$DRY_RUN" == "false" ]]; then
        cd "$PROJECT_ROOT"
        
        # Pull latest images if not building locally
        if [[ "$BUILD_TARGET" != "development" ]]; then
            log_info "Pulling latest images..."
            docker-compose -f "$compose_file" pull || true
        fi
        
        # Deploy services
        log_info "Starting services..."
        docker-compose -f "$compose_file" up -d
        
        # Show status
        log_info "Service status:"
        docker-compose -f "$compose_file" ps
        
        log_info "✅ Deployment completed"
    else
        log_info "[DRY RUN] Would deploy using: $compose_file"
    fi
}

show_logs() {
    local compose_file="docker-compose.$ENVIRONMENT.yml"
    local services=($(get_services))
    
    if [[ ${#services[@]} -eq 1 ]]; then
        docker-compose -f "$compose_file" logs -f "${services[0]}"
    else
        docker-compose -f "$compose_file" logs -f "${services[@]}"
    fi
}

validate_configs() {
    log_info "Validating Docker configurations..."
    
    local configs=("docker-compose.development.yml" "docker-compose.production.yml")
    
    for config in "${configs[@]}"; do
        if [[ -f "$PROJECT_ROOT/$config" ]]; then
            log_info "Validating $config..."
            
            if docker-compose -f "$config" config >/dev/null 2>&1; then
                log_info "✅ $config is valid"
            else
                log_error "❌ $config has validation errors"
                return 1
            fi
        fi
    done
    
    log_info "✅ All configurations are valid"
}

main() {
    local command=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--target)
                BUILD_TARGET="$2"
                shift 2
                ;;
            -r|--registry)
                REGISTRY="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -p|--push)
                PUSH_IMAGES=true
                shift
                ;;
            -n|--no-cache)
                USE_CACHE=false
                shift
                ;;
            -j|--parallel)
                PARALLEL_BUILDS=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -V|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            build|push|pull|clean|prune|test|validate|security-scan|size-report|deploy|logs)
                command="$1"
                shift
                break
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
    
    # Remaining arguments are services
    SERVICES=("$@")
    
    if [[ -z "$command" ]]; then
        log_error "No command specified"
        usage
        exit 1
    fi
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Check dependencies
    check_dependencies
    
    log_debug "Command: $command"
    log_debug "Environment: $ENVIRONMENT"
    log_debug "Build Target: $BUILD_TARGET"
    log_debug "Services: $(get_services)"
    log_debug "DRY_RUN: $DRY_RUN"
    
    # Execute command
    case "$command" in
        build)
            build_services
            if [[ "$PUSH_IMAGES" == "true" ]]; then
                push_services
            fi
            ;;
        push)
            push_services
            ;;
        pull)
            pull_services
            ;;
        clean)
            clean_services
            ;;
        prune)
            if [[ "$DRY_RUN" == "false" ]]; then
                docker system prune -f
                docker volume prune -f
            else
                log_info "[DRY RUN] Would prune Docker resources"
            fi
            ;;
        test)
            # Run tests in containers
            build_services
            # TODO: Implement container testing
            log_info "Container testing not yet implemented"
            ;;
        validate)
            validate_configs
            ;;
        security-scan)
            security_scan
            ;;
        size-report)
            size_report
            ;;
        deploy)
            deploy_services
            ;;
        logs)
            show_logs
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

main "$@"