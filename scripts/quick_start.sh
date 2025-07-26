#!/bin/bash
# XORB Ecosystem Quick Start Script
# 
# This script provides one-command deployment of the complete XORB ecosystem
# with intelligent environment detection and automated setup.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${ENVIRONMENT:-dev}"
PHASES="${PHASES:-all}"
MONITORING="${MONITORING:-true}"


# Banner
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
║           Autonomous Security Intelligence Platform - Quick Start             ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Functions
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

check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose (v1 or v2) first."
        exit 1
    fi
    
    # Preflight check for Docker and Docker Compose versions
    log_info "Verifying Docker and Docker Compose versions..."
    if ! docker version &> /dev/null; then
        log_error "Docker command failed. Please check your Docker installation."
        exit 1
    fi
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose command failed. Please check your Docker Compose installation."
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    
    log_success "All requirements met!"
}

detect_environment() {
    log_info "Detecting deployment environment..."
    
    # Check if we're on Raspberry Pi
    if grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
        log_info "Raspberry Pi detected - optimizing for Pi 5"
        export PI5_OPTIMIZATION=true
        export MAX_CONCURRENT_MISSIONS=5
        export ORCHESTRATION_CYCLE_TIME=400
    fi
    
    # Check available resources
    MEMORY_GB=$(free -g | awk 'NR==2{printf "%.0f", $2}')
    CPU_CORES=$(nproc)
    
    log_info "System resources: ${MEMORY_GB}GB RAM, ${CPU_CORES} CPU cores"
    
    # Adjust environment based on resources
    if [ "$MEMORY_GB" -lt 4 ]; then
        log_warning "Low memory detected. Using minimal configuration."
        ENVIRONMENT="dev"
        export SCALE_FACTOR=1
    elif [ "$MEMORY_GB" -lt 8 ]; then
        log_info "Medium resources detected. Using staging configuration."
        ENVIRONMENT="staging"
        export SCALE_FACTOR=2
    else
        log_info "High resources detected. Production configuration available."
        export SCALE_FACTOR=3
    fi
}

setup_environment() {
    log_info "Setting up environment variables..."
    
    # Create .env file if it doesn't exist
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        log_info "Creating environment configuration..."
        cat > "$PROJECT_ROOT/.env" << EOF
# XORB Environment Configuration
ENVIRONMENT=$ENVIRONMENT
COMPOSE_PROJECT_NAME=xorb

# Database Configuration
POSTGRES_DB=xorb
POSTGRES_USER=xorb_prod
POSTGRES_PASSWORD=xorb_secure_2024

# Monitoring Configuration
GRAFANA_ADMIN_PASSWORD=xorb_admin_2024

# Phase 11 Configuration
XORB_PHASE_11_ENABLED=true
XORB_PI5_OPTIMIZATION=${PI5_OPTIMIZATION:-false}
XORB_ORCHESTRATION_CYCLE_TIME=${ORCHESTRATION_CYCLE_TIME:-500}
XORB_MAX_CONCURRENT_MISSIONS=${MAX_CONCURRENT_MISSIONS:-10}
XORB_PLUGIN_DISCOVERY_ENABLED=true

# API Keys (set these for full functionality)
# OPENROUTER_API_KEY=your_key_here
# CEREBRAS_API_KEY=your_key_here
EOF
        log_success "Environment configuration created"
    else
        log_info "Using existing environment configuration"
    fi
}

create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/data/postgres"
    mkdir -p "$PROJECT_ROOT/data/redis"
    mkdir -p "$PROJECT_ROOT/data/prometheus"
    mkdir -p "$PROJECT_ROOT/data/grafana"
    mkdir -p "$PROJECT_ROOT/backups"
    mkdir -p "$PROJECT_ROOT/plugins"
    
    log_success "Directories created"
}

create_docker_network() {
    log_info "Creating Docker network..."
    
    if ! docker network ls | grep -q xorb-network; then
        docker network create xorb-network
        log_success "Docker network 'xorb-network' created"
    else
        log_info "Docker network 'xorb-network' already exists"
    fi
}

cleanup() {
    log_info "Cleaning up XORB deployment..."
    
    cd "$PROJECT_ROOT" 2>/dev/null || cd "$(dirname "$SCRIPT_DIR")"
    
    # Try to stop with all available compose files
    for compose_file in docker-compose.yml docker-compose.dev.yml docker-compose.enhanced.yml docker-compose.production.yml; do
        if [ -f "$compose_file" ]; then
            log_info "Stopping services from $compose_file"
            docker compose -f "$compose_file" down -v --remove-orphans 2>/dev/null || true
        fi
    done
    
    # Remove network
    docker network rm xorb-network 2>/dev/null || true
    
    # Clean up volumes with XORB prefix
    docker volume ls -q | grep -E "^xorb_|^[a-f0-9]+_.*_(postgres|redis|prometheus|grafana)" | xargs -r docker volume rm 2>/dev/null || true
    
    # Remove unused images
    docker image prune -f 2>/dev/null || true
    
    log_success "Cleanup completed"
}

show_completion_info() {
    log_success "XORB Ecosystem deployment completed!"
    
    echo ""
    echo -e "${GREEN}🎉 XORB is now running! 🎉${NC}"
    echo ""
    echo -e "${BLUE}Available Services:${NC}"
    echo "  🔍 Scanning Services:"
    echo "    - Scanner (Go): http://localhost:8080"
    echo "    - Vulnerability Scanner: http://localhost:8081"
    echo ""
    echo "  🤖 AI Services:"
    echo "    - Campaign Engine: http://localhost:8082"
    echo "    - Learning Engine: http://localhost:8083"
    echo "    - Prioritization Engine: http://localhost:8084"
    echo ""
    echo "  🎯 Orchestration:"
    echo "    - Main Orchestrator: http://localhost:8085"
    echo "    - Enhanced Orchestrator (Phase 11): http://localhost:8089"
    echo "    - API Service: http://localhost:8000"
    echo ""
    echo "  📊 Monitoring:"
    echo "    - Prometheus: http://localhost:9090"
    echo "    - Grafana: http://localhost:3000 (admin/xorb_admin_2024)"
    echo "    - AlertManager: http://localhost:9093"
    echo ""
    echo -e "${BLUE}Management Commands:${NC}"
    echo "  📊 View status: python3 scripts/xorb_ecosystem_manager.py status"
    echo "  🔄 Real-time monitor: python3 scripts/xorb_ecosystem_manager.py monitor"
    echo "  📜 View logs: python3 scripts/xorb_ecosystem_manager.py logs --service <service>"
    echo "  ⚖️ Scale service: python3 scripts/xorb_ecosystem_manager.py scale --service <service> --replicas <n>"
    echo "  💾 Create backup: python3 scripts/xorb_ecosystem_manager.py backup"
    echo "  🧹 Cleanup: python3 scripts/deploy_full_ecosystem.py --mode destroy"
    echo ""
    echo -e "${BLUE}Phase 11 Features:${NC}"
    echo "  🎯 Temporal Signal Pattern Recognition"
    echo "  🤝 Multi-Agent Role Dynamic Assignment"
    echo "  🔄 Fault-Tolerant Mission Recycling"
    echo "  📊 Per-Signal KPI Instrumentation"
    echo "  🔍 Redundancy & Conflict Detection"
    echo "  🔌 Plugin Registry System"
    echo ""
    echo -e "${YELLOW}Note:${NC} For full AI functionality, set API keys in .env file:"
    echo "  OPENROUTER_API_KEY=your_key"
    echo "  CEREBRAS_API_KEY=your_key"
    echo ""
    echo -e "${GREEN}Happy hunting! 🕵️‍♂️${NC}"
}



cleanup_on_error() {
    log_error "Deployment failed. Initiating cleanup..."
    
    # Try Python cleanup first
    if [ -f "$SCRIPT_DIR/deploy_full_ecosystem.py" ]; then
        python3 "$SCRIPT_DIR/deploy_full_ecosystem.py" --mode destroy 2>/dev/null || true
    fi
    
    # Fallback to Docker cleanup
    cleanup
    
    log_info "Cleanup after error completed. Check logs for errors."
}

# Main execution
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --phases)
                PHASES="$2"
                shift 2
                ;;
            --no-monitoring)
                MONITORING="false"
                shift
                ;;
            --help|-h)
                echo "XORB Quick Start Script"
                echo ""
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --env ENV          Deployment environment (dev|staging|prod) [default: dev]"
                echo "  --phases PHASES    Phases to deploy (1-11|all) [default: all]"
                echo "  --no-monitoring    Skip monitoring stack deployment"
                echo "  --cleanup          Stop and remove all Docker services and network"
                echo "  --help, -h         Show this help message"
                echo ""
                echo "Environment Variables:"
                echo "  ENVIRONMENT        Override deployment environment"
                echo "  PHASES            Override phases to deploy"
                echo "  MONITORING        Enable/disable monitoring (true|false)"
                echo "  DOCKER_COMPOSE_FILE Override default docker-compose.yml (e.g., docker-compose.prod.yml)"
                exit 0
                ;;
            --cleanup)
                cleanup
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Set up error handling
    trap cleanup_on_error ERR
    
    log_info "Starting XORB ecosystem deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Phases: $PHASES"
    log_info "Monitoring: $MONITORING"
    echo ""
    
    # Deployment steps
    log_info "--- Phase 1: Checking Requirements ---"
    check_requirements
    log_info "--- Phase 2: Detecting Environment ---"
    detect_environment
    log_info "--- Phase 3: Setting Up Environment ---"
    setup_environment
    log_info "--- Phase 4: Creating Directories ---"
    create_directories
    log_info "--- Phase 5: Creating Docker Network ---"
    create_docker_network

    log_info "--- Phase 6: Deploying Full Ecosystem ---"
    
    # Use docker-compose for quick deployment
    cd "$PROJECT_ROOT"
    
    # Select appropriate compose file based on environment
    COMPOSE_FILE="docker-compose.simple.yml"
    case "$ENVIRONMENT" in
        "dev")
            COMPOSE_FILE="docker-compose.simple.yml"  # Use simple compose for dev
            ;;
        "staging")
            COMPOSE_FILE="docker-compose.enhanced.yml"
            ;;
        "prod")
            COMPOSE_FILE="docker-compose.production.yml"
            ;;
    esac
    
    if [ -f "$COMPOSE_FILE" ]; then
        log_info "Using compose file: $COMPOSE_FILE"
        docker compose -f "$COMPOSE_FILE" up -d --build
        
        # Wait for services to be ready
        log_info "Waiting for services to start..."
        sleep 30
        
        # Basic health check
        log_info "Performing basic health checks..."
        for service in api orchestrator; do
            if docker compose -f "$COMPOSE_FILE" ps | grep -q "$service.*Up"; then
                log_success "$service is running"
            else
                log_warning "$service may not be ready yet"
            fi
        done
    else
        log_warning "Compose file $COMPOSE_FILE not found, falling back to Python deployment"
        
        MONITORING_FLAG=""
        if [ "$MONITORING" = "false" ]; then
            MONITORING_FLAG="--skip-monitoring"
        fi
        
        if [ -f "$SCRIPT_DIR/deploy_full_ecosystem.py" ]; then
            python3 "$SCRIPT_DIR/deploy_full_ecosystem.py" --mode deploy --env "$ENVIRONMENT" $MONITORING_FLAG
        else
            log_error "No deployment method available"
            exit 1
        fi
    fi
    
    log_info "--- Phase 7: Displaying Completion Info ---"
    show_completion_info
}

# Run main function
main "$@"