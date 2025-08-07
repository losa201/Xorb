#!/bin/bash

# XORB PTaaS Frontend Deployment Script
# Version: 1.0.0
# Description: Deploy the XORB PTaaS frontend with full stack integration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose-ptaas-with-frontend.yml"
PROJECT_NAME="xorb-ptaas"
FRONTEND_PORT=3000
API_PORT=8080
AUTH_PORT=8001
GRAFANA_PORT=3001

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

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if ports are available
    for port in $FRONTEND_PORT $API_PORT $AUTH_PORT $GRAFANA_PORT; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
            log_warning "Port $port is already in use. This may cause conflicts."
        fi
    done
    
    log_success "Prerequisites check completed"
}

# Build frontend
build_frontend() {
    log_info "Building XORB PTaaS frontend..."
    
    cd frontend
    
    # Install dependencies
    if [ -f "package-lock.json" ]; then
        npm ci --silent
    else
        npm install --silent
    fi
    
    # Build the application
    npm run build
    
    cd ..
    log_success "Frontend build completed"
}

# Deploy services
deploy_services() {
    log_info "Deploying XORB PTaaS services..."
    
    # Create necessary directories
    mkdir -p logs/nginx logs/nginx-lb
    
    # Set environment variables
    export QWEN_API_KEY=${QWEN_API_KEY:-""}
    export OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-""}
    export CEREBRAS_API_KEY=${CEREBRAS_API_KEY:-""}
    
    # Deploy with docker-compose
    if command -v docker-compose &> /dev/null; then
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d --build
    else
        docker compose -f $COMPOSE_FILE -p $PROJECT_NAME up -d --build
    fi
    
    log_success "Services deployment initiated"
}

# Wait for services to be healthy
wait_for_services() {
    log_info "Waiting for services to be healthy..."
    
    local max_attempts=60
    local attempt=0
    
    local services=("xorb-ptaas-frontend:3000" "xorb-ptaas-core:8080" "xorb-ptaas-company-api:8001")
    
    for service in "${services[@]}"; do
        local container_name=${service%:*}
        local port=${service#*:}
        
        attempt=0
        while [ $attempt -lt $max_attempts ]; do
            if curl -f http://localhost:$port/health >/dev/null 2>&1; then
                log_success "$container_name is healthy"
                break
            fi
            
            attempt=$((attempt + 1))
            if [ $attempt -eq $max_attempts ]; then
                log_error "$container_name failed to become healthy after $max_attempts attempts"
                return 1
            fi
            
            sleep 5
        done
    done
    
    log_success "All services are healthy"
}

# Display deployment information
show_deployment_info() {
    echo
    echo "=========================================="
    echo "🚀 XORB PTaaS Deployment Complete!"
    echo "=========================================="
    echo
    echo "🌐 Frontend Application:"
    echo "   Direct Access: http://localhost:$FRONTEND_PORT"
    echo "   Load Balanced: http://localhost:80"
    echo
    echo "🔧 API Endpoints:"
    echo "   Main API: http://localhost:$API_PORT"
    echo "   Auth API: http://localhost:$AUTH_PORT"
    echo
    echo "📊 Monitoring:"
    echo "   Grafana: http://localhost:$GRAFANA_PORT"
    echo "   Prometheus: http://localhost:9090"
    echo
    echo "🗄️ Databases:"
    echo "   PostgreSQL: localhost:5432"
    echo "   Neo4j: http://localhost:7474"
    echo "   Redis: localhost:6379"
    echo "   Qdrant: http://localhost:6333"
    echo
    echo "🔐 Demo Credentials:"
    echo "   Admin: admin / xorb_admin_2025"
    echo "   Analyst: analyst / analyst123"
    echo "   Demo: demo / demo123"
    echo
    echo "📋 Management Commands:"
    echo "   Check Status: docker ps --filter name=$PROJECT_NAME"
    echo "   View Logs: docker logs [container-name]"
    echo "   Stop All: docker-compose -f $COMPOSE_FILE down"
    echo
    echo "=========================================="
}

# Cleanup function
cleanup() {
    log_info "Cleaning up deployment artifacts..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -f $COMPOSE_FILE -p $PROJECT_NAME down --remove-orphans
    else
        docker compose -f $COMPOSE_FILE -p $PROJECT_NAME down --remove-orphans
    fi
    
    log_success "Cleanup completed"
}

# Main deployment flow
main() {
    echo "🛡️ XORB PTaaS Frontend Deployment Script"
    echo "=========================================="
    
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            build_frontend
            deploy_services
            wait_for_services
            show_deployment_info
            ;;
        "cleanup")
            cleanup
            ;;
        "status")
            docker ps --filter name=$PROJECT_NAME --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
            ;;
        "logs")
            if [ -n "$2" ]; then
                docker logs -f "${PROJECT_NAME}_$2_1" 2>/dev/null || docker logs -f "xorb-ptaas-$2"
            else
                log_error "Please specify a service name: ./deploy_frontend.sh logs [service-name]"
                echo "Available services: frontend, postgres, redis, neo4j, qdrant, prometheus, grafana, ptaas-core, researcher-api, company-api, nginx-lb"
            fi
            ;;
        "help")
            echo "Usage: $0 [command]"
            echo
            echo "Commands:"
            echo "  deploy    - Deploy the full XORB PTaaS platform (default)"
            echo "  cleanup   - Stop and remove all services"
            echo "  status    - Show status of all services"
            echo "  logs      - Show logs for a specific service"
            echo "  help      - Show this help message"
            echo
            echo "Examples:"
            echo "  $0 deploy"
            echo "  $0 status"
            echo "  $0 logs frontend"
            echo "  $0 cleanup"
            ;;
        *)
            log_error "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"