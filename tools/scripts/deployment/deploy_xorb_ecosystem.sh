#!/bin/bash
# XORB Ecosystem Production Deployment Script
# Deploys the complete XORB autonomous security intelligence platform

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_MODE="${DEPLOYMENT_MODE:-production}"
COMPOSE_FILES=()

echo -e "${BLUE}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║    ██╗  ██╗ ██████╗ ██████╗ ██████╗     ███████╗ ██████╗ ██████╗ ███████╗   ║
║    ╚██╗██╔╝██╔═══██╗██╔══██╗██╔══██╗    ██╔════╝██╔═══██╗██╔══██╗██╔════╝   ║
║     ╚███╔╝ ██║   ██║██████╔╝██████╔╝    █████╗  ██║   ██║██████╔╝█████╗     ║
║     ██╔██╗ ██║   ██║██╔══██╗██╔══██╗    ██╔══╝  ██║   ██║██╔══██╗██╔══╝     ║
║    ██╔╝ ██╗╚██████╔╝██║  ██║██████╔╝    ███████╗╚██████╔╝██║  ██║███████╗   ║
║    ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝     ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝   ║
║                                                                               ║
║              🚀 AUTONOMOUS SECURITY INTELLIGENCE PLATFORM 🚀                 ║
║                          Production Deployment                                ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

check_prerequisites() {
    log "🔍 Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi

    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        error "Docker Compose v2 is not available. Please install Docker Compose v2."
    fi

    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker."
    fi

    # Check system resources
    local memory_gb=$(free -g | awk 'NR==2{printf "%.0f", $2}')
    local cpu_cores=$(nproc)

    log "System resources: ${memory_gb}GB RAM, ${cpu_cores} CPU cores"

    if [ "$memory_gb" -lt 8 ]; then
        warn "Less than 8GB RAM detected. XORB may not perform optimally."
    fi

    if [ "$cpu_cores" -lt 4 ]; then
        warn "Less than 4 CPU cores detected. XORB may not perform optimally."
    fi

    log "✅ Prerequisites check completed"
}

setup_environment() {
    log "🔧 Setting up environment..."

    cd "$PROJECT_ROOT"

    # Create necessary directories
    mkdir -p logs data/{postgres,redis,prometheus,grafana,tempo,neo4j,qdrant} monitoring/rules

    # Set proper permissions
    chmod 755 logs data monitoring

    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        log "Creating .env file from template..."
        cp .env.example .env
        warn "Please review and update .env file with your configuration!"
    fi

    # Source environment variables
    if [ -f .env ]; then
        set -a
        source .env
        set +a
    fi

    log "✅ Environment setup completed"
}

select_deployment_mode() {
    log "🎯 Selecting deployment mode: $DEPLOYMENT_MODE"

    case "$DEPLOYMENT_MODE" in
        "production")
            COMPOSE_FILES=("-f" "docker-compose.unified.yml")
            log "Production mode: Full stack with monitoring"
            ;;
        "development")
            COMPOSE_FILES=("-f" "docker-compose.simple.yml")
            log "Development mode: Simplified stack"
            ;;
        "minimal")
            COMPOSE_FILES=("-f" "docker-compose.yml")
            log "Minimal mode: Core services only"
            ;;
        *)
            error "Unknown deployment mode: $DEPLOYMENT_MODE. Use: production, development, or minimal"
            ;;
    esac
}

deploy_infrastructure() {
    log "🏗️ Deploying infrastructure services..."

    # Pull required images first
    log "Pulling Docker images..."
    docker compose "${COMPOSE_FILES[@]}" pull --ignore-buildable

    # Deploy infrastructure services first
    log "Starting infrastructure services..."
    docker compose "${COMPOSE_FILES[@]}" up -d postgres redis temporal nats neo4j qdrant

    # Wait for infrastructure to be ready
    log "⏳ Waiting for infrastructure services to be healthy..."

    local max_wait=300  # 5 minutes
    local elapsed=0
    local interval=10

    while [ $elapsed -lt $max_wait ]; do
        if docker compose "${COMPOSE_FILES[@]}" ps --format json | jq -r '.[].Health' | grep -v "healthy\|" | grep -q "unhealthy\|starting"; then
            log "Waiting for services to become healthy... (${elapsed}s/${max_wait}s)"
            sleep $interval
            elapsed=$((elapsed + interval))
        else
            break
        fi
    done

    if [ $elapsed -ge $max_wait ]; then
        error "Infrastructure services failed to become healthy within ${max_wait} seconds"
    fi

    log "✅ Infrastructure services are healthy"
}

deploy_core_services() {
    log "🚀 Deploying XORB core services..."

    # Build and deploy core services
    docker compose "${COMPOSE_FILES[@]}" up -d --build api worker orchestrator scanner-go

    # Wait for core services
    log "⏳ Waiting for core services to start..."
    sleep 30

    # Verify core services
    local services=("api:8000" "worker:9000" "orchestrator:8080" "scanner-go:8004")
    for service in "${services[@]}"; do
        local name="${service%:*}"
        local port="${service#*:}"

        if curl -f -s --max-time 5 "http://localhost:$port/health" > /dev/null 2>&1; then
            log "✅ $name service is healthy"
        else
            warn "⚠️ $name service health check failed (may still be starting)"
        fi
    done

    log "✅ Core services deployed"
}

deploy_monitoring() {
    if [[ " ${COMPOSE_FILES[*]} " =~ "unified" ]]; then
        log "📊 Deploying monitoring stack..."

        # Deploy monitoring services
        docker compose "${COMPOSE_FILES[@]}" up -d prometheus grafana tempo alertmanager

        # Wait for monitoring stack
        log "⏳ Waiting for monitoring services..."
        sleep 20

        local monitoring_services=("prometheus:9090" "grafana:3000" "tempo:3200")
        for service in "${monitoring_services[@]}"; do
            local name="${service%:*}"
            local port="${service#*:}"

            if curl -f -s --max-time 5 "http://localhost:$port" > /dev/null 2>&1; then
                log "✅ $name is accessible"
            else
                warn "⚠️ $name may still be starting"
            fi
        done

        log "✅ Monitoring stack deployed"
    else
        log "📊 Skipping monitoring stack (not in production mode)"
    fi
}

verify_deployment() {
    log "🔍 Verifying deployment..."

    # Show running containers
    echo -e "\n${BLUE}=== Running Containers ===${NC}"
    docker compose "${COMPOSE_FILES[@]}" ps

    # Check container health
    echo -e "\n${BLUE}=== Container Health Status ===${NC}"
    local unhealthy_count=0
    while IFS= read -r container; do
        local name=$(echo "$container" | jq -r '.Name')
        local health=$(echo "$container" | jq -r '.Health // "N/A"')
        local state=$(echo "$container" | jq -r '.State')

        if [ "$health" = "healthy" ] || [ "$state" = "running" ]; then
            echo -e "${GREEN}✅ $name: $state ($health)${NC}"
        else
            echo -e "${RED}❌ $name: $state ($health)${NC}"
            unhealthy_count=$((unhealthy_count + 1))
        fi
    done < <(docker compose "${COMPOSE_FILES[@]}" ps --format json | jq -c '.[]')

    if [ $unhealthy_count -gt 0 ]; then
        warn "$unhealthy_count services are not healthy"
    fi

    # Test API endpoints
    echo -e "\n${BLUE}=== API Endpoint Tests ===${NC}"
    local endpoints=(
        "API Health:http://localhost:8000/health"
        "Worker Metrics:http://localhost:9000/metrics"
        "Orchestrator:http://localhost:8080/health"
        "Scanner:http://localhost:8004/health"
    )

    if [[ " ${COMPOSE_FILES[*]} " =~ "unified" ]]; then
        endpoints+=(
            "Prometheus:http://localhost:9090/-/healthy"
            "Grafana:http://localhost:3000/api/health"
            "Tempo:http://localhost:3200/ready"
        )
    fi

    for endpoint in "${endpoints[@]}"; do
        local name="${endpoint%:*}"
        local url="${endpoint#*:}"

        if curl -f -s --max-time 5 "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $name${NC}"
        else
            echo -e "${RED}❌ $name${NC}"
        fi
    done

    log "✅ Deployment verification completed"
}

show_access_info() {
    log "🎉 XORB Ecosystem deployed successfully!"

    echo -e "\n${BLUE}=== 🌐 Service Access Information ===${NC}"
    echo -e "${GREEN}Core Services:${NC}"
    echo "  🔌 API Service:          http://localhost:8000"
    echo "  🔧 Worker Metrics:       http://localhost:9000/metrics"
    echo "  🎯 Orchestrator:         http://localhost:8080"
    echo "  🔍 Scanner Service:      http://localhost:8004"

    echo -e "\n${GREEN}Infrastructure:${NC}"
    echo "  🗄️  PostgreSQL:          localhost:5432"
    echo "  📝 Redis:                localhost:6379"
    echo "  ⏰ Temporal UI:          http://localhost:8233"
    echo "  📡 NATS:                 http://localhost:8222"
    echo "  🕸️  Neo4j Browser:       http://localhost:7474"
    echo "  🔍 Qdrant:               http://localhost:6333"

    if [[ " ${COMPOSE_FILES[*]} " =~ "unified" ]]; then
        echo -e "\n${GREEN}Monitoring Stack:${NC}"
        echo "  📊 Prometheus:           http://localhost:9090"
        echo "  📈 Grafana:              http://localhost:3000 (admin/xorb_admin_2024)"
        echo "  🔍 Tempo:                http://localhost:3200"
        echo "  🚨 AlertManager:         http://localhost:9093"
    fi

    echo -e "\n${BLUE}=== 🔧 Management Commands ===${NC}"
    echo "  View logs:       docker compose ${COMPOSE_FILES[*]} logs -f [service]"
    echo "  Stop services:   docker compose ${COMPOSE_FILES[*]} down"
    echo "  Restart:         docker compose ${COMPOSE_FILES[*]} restart [service]"
    echo "  Scale service:   docker compose ${COMPOSE_FILES[*]} up -d --scale worker=3"

    echo -e "\n${BLUE}=== 🛠️ Useful Commands ===${NC}"
    echo "  Enter API shell: docker compose ${COMPOSE_FILES[*]} exec api bash"
    echo "  Check health:    curl http://localhost:8000/health"
    echo "  echo "  View metrics:    curl http://localhost:9001/metrics""

    echo -e "\n${GREEN}🚀 XORB is now running autonomously! 🤖${NC}"
    echo -e "${YELLOW}💡 Tip: Monitor the orchestrator logs to see autonomous mission execution${NC}"
    echo -e "${YELLOW}   docker compose ${COMPOSE_FILES[*]} logs -f orchestrator${NC}"
}

cleanup_on_error() {
    error "Deployment failed. Cleaning up..."
    docker compose "${COMPOSE_FILES[@]}" down --remove-orphans 2>/dev/null || true
    exit 1
}

# Main execution
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                DEPLOYMENT_MODE="$2"
                shift 2
                ;;
            --help|-h)
                echo "XORB Ecosystem Deployment Script"
                echo ""
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --mode MODE        Deployment mode (production|development|minimal) [default: production]"
                echo "  --help, -h         Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0                          # Deploy in production mode"
                echo "  $0 --mode development       # Deploy in development mode"
                echo "  $0 --mode minimal           # Deploy minimal stack"
                exit 0
                ;;
            *)
                error "Unknown option: $1. Use --help for usage information."
                ;;
        esac
    done

    # Set up error handling
    trap cleanup_on_error ERR

    # Deployment steps
    log "🚀 Starting XORB Ecosystem Deployment (Mode: $DEPLOYMENT_MODE)"

    check_prerequisites
    setup_environment
    select_deployment_mode
    deploy_infrastructure
    deploy_core_services
    deploy_monitoring
    verify_deployment
    show_access_info

    log "✨ Deployment completed successfully! XORB is ready for autonomous operations."
}

# Run main function
main "$@"
