#!/bin/bash
set -e

# XORB Monitoring Stack Setup Script
# This script sets up a comprehensive monitoring stack with Prometheus, Grafana, and AlertManager

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
MONITORING_DIR="$PROJECT_ROOT/infra/monitoring"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    log "Prerequisites check passed ✅"
}

# Create monitoring directories
create_directories() {
    log "Creating monitoring directories..."
    
    mkdir -p "$MONITORING_DIR/grafana/dashboards/json"
    mkdir -p "$MONITORING_DIR/grafana/datasources"
    mkdir -p "$MONITORING_DIR/prometheus/rules"
    mkdir -p "$MONITORING_DIR/alertmanager/templates"
    mkdir -p "$PROJECT_ROOT/data/prometheus"
    mkdir -p "$PROJECT_ROOT/data/grafana"
    mkdir -p "$PROJECT_ROOT/data/alertmanager"
    
    log "Directories created ✅"
}

# Set proper permissions
set_permissions() {
    log "Setting proper file permissions..."
    
    # Grafana needs specific permissions
    sudo chown -R 472:472 "$PROJECT_ROOT/data/grafana" 2>/dev/null || {
        warn "Could not set Grafana permissions. You may need to run this script with sudo."
    }
    
    # Prometheus needs specific permissions
    sudo chown -R 65534:65534 "$PROJECT_ROOT/data/prometheus" 2>/dev/null || {
        warn "Could not set Prometheus permissions. You may need to run this script with sudo."
    }
    
    # AlertManager needs specific permissions
    sudo chown -R 65534:65534 "$PROJECT_ROOT/data/alertmanager" 2>/dev/null || {
        warn "Could not set AlertManager permissions. You may need to run this script with sudo."
    }
    
    log "Permissions set ✅"
}

# Generate environment file
generate_env_file() {
    log "Generating monitoring environment file..."
    
    ENV_FILE="$PROJECT_ROOT/.env.monitoring"
    
    cat > "$ENV_FILE" << EOF
# XORB Monitoring Stack Environment Variables
# Generated on $(date)

# Grafana Configuration
GRAFANA_PASSWORD=SecureAdminPass123!
GRAFANA_SECRET_KEY=$(openssl rand -base64 32)

# Database Configuration (for exporters)
DB_USER=xorb_user
DB_PASSWORD=password
DB_HOST=postgres
DB_PORT=5432
DB_NAME=xorb_db

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=

# SMTP Configuration for AlertManager
SMTP_PASSWORD=your_smtp_password_here

# Webhook URLs for alerts
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
PAGERDUTY_WEBHOOK_URL=https://events.pagerduty.com/integration/YOUR_INTEGRATION_KEY/enqueue

# Environment
ENVIRONMENT=production
EOF

    log "Environment file created: $ENV_FILE"
    info "Please update the environment variables in $ENV_FILE before starting the monitoring stack"
}

# Start monitoring stack
start_monitoring() {
    log "Starting XORB monitoring stack..."
    
    cd "$PROJECT_ROOT"
    
    # Load environment variables
    if [ -f ".env.monitoring" ]; then
        source .env.monitoring
    fi
    
    # Start the monitoring services
    docker-compose -f docker-compose.monitoring.yml up -d
    
    log "Monitoring stack started ✅"
}

# Check service health
check_services() {
    log "Checking service health..."
    
    local services=("prometheus:9090" "grafana:3000" "alertmanager:9093" "node-exporter:9100")
    local max_attempts=30
    local attempt=1
    
    for service in "${services[@]}"; do
        local name="${service%%:*}"
        local port="${service##*:}"
        
        info "Checking $name on port $port..."
        
        while [ $attempt -le $max_attempts ]; do
            if curl -s "http://localhost:$port" &> /dev/null; then
                log "$name is healthy ✅"
                break
            else
                if [ $attempt -eq $max_attempts ]; then
                    warn "$name health check failed after $max_attempts attempts"
                else
                    sleep 2
                    ((attempt++))
                fi
            fi
        done
        
        attempt=1
    done
}

# Display access information
show_access_info() {
    log "XORB Monitoring Stack is ready! 🚀"
    echo ""
    echo "📊 Access URLs:"
    echo "  • Prometheus:    http://localhost:9090"
    echo "  • Grafana:       http://localhost:3000 (admin / SecureAdminPass123!)"
    echo "  • AlertManager:  http://localhost:9093"
    echo "  • Node Exporter: http://localhost:9100"
    echo "  • cAdvisor:      http://localhost:8080"
    echo ""
    echo "📈 Grafana Dashboards:"
    echo "  • XORB Overview: http://localhost:3000/d/xorb-overview"
    echo ""
    echo "🔧 Management Commands:"
    echo "  • View logs:     docker-compose -f docker-compose.monitoring.yml logs -f"
    echo "  • Stop stack:    docker-compose -f docker-compose.monitoring.yml down"
    echo "  • Restart:       docker-compose -f docker-compose.monitoring.yml restart"
    echo ""
    echo "⚙️  Configuration Files:"
    echo "  • Prometheus:    $MONITORING_DIR/prometheus.yml"
    echo "  • Alert Rules:   $MONITORING_DIR/prometheus-rules.yml"
    echo "  • AlertManager:  $MONITORING_DIR/alertmanager.yml"
    echo "  • Grafana:       $MONITORING_DIR/grafana/"
    echo ""
}

# Main function
main() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════╗"
    echo "║              XORB Monitoring Setup               ║"
    echo "║          Prometheus + Grafana + AlertManager     ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    check_prerequisites
    create_directories
    set_permissions
    generate_env_file
    start_monitoring
    
    # Wait a bit for services to start
    info "Waiting for services to start..."
    sleep 10
    
    check_services
    show_access_info
}

# Handle script arguments
case "${1:-}" in
    "start")
        log "Starting monitoring stack..."
        start_monitoring
        ;;
    "stop")
        log "Stopping monitoring stack..."
        cd "$PROJECT_ROOT"
        docker-compose -f docker-compose.monitoring.yml down
        ;;
    "restart")
        log "Restarting monitoring stack..."
        cd "$PROJECT_ROOT"
        docker-compose -f docker-compose.monitoring.yml restart
        ;;
    "logs")
        log "Showing monitoring stack logs..."
        cd "$PROJECT_ROOT"
        docker-compose -f docker-compose.monitoring.yml logs -f
        ;;
    "status")
        log "Checking monitoring stack status..."
        cd "$PROJECT_ROOT"
        docker-compose -f docker-compose.monitoring.yml ps
        ;;
    "")
        main
        ;;
    *)
        echo "Usage: $0 [start|stop|restart|logs|status]"
        echo ""
        echo "Commands:"
        echo "  start    - Start the monitoring stack"
        echo "  stop     - Stop the monitoring stack"
        echo "  restart  - Restart the monitoring stack"
        echo "  logs     - Show monitoring stack logs"
        echo "  status   - Show monitoring stack status"
        echo "  (no arg) - Full setup and start"
        exit 1
        ;;
esac