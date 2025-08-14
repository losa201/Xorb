#!/bin/bash

# XORB Learning Engine Integration Deployment Script
# Comprehensive deployment of PTaaS to Learning Engine integration

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
CERT_DIR="${SCRIPT_DIR}/certs"
KEYS_DIR="${SCRIPT_DIR}/keys"
DATA_DIR="${SCRIPT_DIR}/data"
MODELS_DIR="${SCRIPT_DIR}/models"

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
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    mkdir -p "${LOG_DIR}"
    mkdir -p "${CERT_DIR}"
    mkdir -p "${KEYS_DIR}"
    mkdir -p "${DATA_DIR}"
    mkdir -p "${MODELS_DIR}"
    mkdir -p "${SCRIPT_DIR}/monitoring/grafana/dashboards"
    mkdir -p "${SCRIPT_DIR}/monitoring/prometheus"
    mkdir -p "${SCRIPT_DIR}/monitoring/rules"
    mkdir -p "${SCRIPT_DIR}/nginx"
    mkdir -p "${SCRIPT_DIR}/test-results"
}

# Generate SSL certificates
generate_certificates() {
    log "Generating SSL certificates..."

    if [[ ! -f "${CERT_DIR}/ca.crt" ]]; then
        # Generate CA private key
        openssl genrsa -out "${CERT_DIR}/ca.key" 4096

        # Generate CA certificate
        openssl req -new -x509 -days 3650 -key "${CERT_DIR}/ca.key" -out "${CERT_DIR}/ca.crt" \
            -subj "/C=US/ST=CA/L=San Francisco/O=XORB Security/CN=XORB Root CA"

        log "Generated CA certificate"
    fi

    if [[ ! -f "${CERT_DIR}/api.crt" ]]; then
        # Generate API server private key
        openssl genrsa -out "${CERT_DIR}/api.key" 2048

        # Generate certificate signing request
        openssl req -new -key "${CERT_DIR}/api.key" -out "${CERT_DIR}/api.csr" \
            -subj "/C=US/ST=CA/L=San Francisco/O=XORB Security/CN=localhost"

        # Generate API server certificate
        openssl x509 -req -days 365 -in "${CERT_DIR}/api.csr" -CA "${CERT_DIR}/ca.crt" \
            -CAkey "${CERT_DIR}/ca.key" -CAcreateserial -out "${CERT_DIR}/api.crt" \
            -extensions v3_req -extfile <(cat <<EOF
[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = learning-api
DNS.3 = adaptive-orchestrator
DNS.4 = security-framework
IP.1 = 127.0.0.1
IP.2 = 172.20.0.1
EOF
)

        # Clean up CSR
        rm "${CERT_DIR}/api.csr"

        log "Generated API server certificate"
    fi

    # Set appropriate permissions
    chmod 600 "${CERT_DIR}"/*.key
    chmod 644 "${CERT_DIR}"/*.crt
}

# Create configuration files
create_configs() {
    log "Creating configuration files..."

    # Database initialization script
    cat > "${SCRIPT_DIR}/init-learning-dbs.sql" << 'EOF'
-- Initialize databases for XORB Learning Engine
CREATE DATABASE IF NOT EXISTS xorb_learning;
CREATE DATABASE IF NOT EXISTS xorb_orchestration;
CREATE DATABASE IF NOT EXISTS xorb_security;
CREATE DATABASE IF NOT EXISTS xorb_test;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE xorb_learning TO xorb;
GRANT ALL PRIVILEGES ON DATABASE xorb_orchestration TO xorb;
GRANT ALL PRIVILEGES ON DATABASE xorb_security TO xorb;
GRANT ALL PRIVILEGES ON DATABASE xorb_test TO xorb;
EOF

    # Prometheus configuration
    cat > "${SCRIPT_DIR}/monitoring/prometheus-learning.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "/etc/prometheus/rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'xorb-learning-engine'
    static_configs:
      - targets: ['learning-engine:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'xorb-learning-api'
    static_configs:
      - targets: ['learning-api:8002']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'xorb-adaptive-orchestrator'
    static_configs:
      - targets: ['adaptive-orchestrator:8003']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'xorb-security-framework'
    static_configs:
      - targets: ['security-framework:8004']
    metrics_path: '/metrics'
    scrape_interval: 15s

  - job_name: 'xorb-integration-bridge'
    static_configs:
      - targets: ['ptaas-integration-bridge:8005']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
EOF

    # Nginx load balancer configuration
    cat > "${SCRIPT_DIR}/nginx/learning-lb.conf" << 'EOF'
events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log warn;

    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=telemetry:10m rate=100r/s;

    # Upstream definitions
    upstream learning_api {
        server learning-api:8002 max_fails=3 fail_timeout=30s;
    }

    upstream orchestrator {
        server adaptive-orchestrator:8003 max_fails=3 fail_timeout=30s;
    }

    upstream security {
        server security-framework:8004 max_fails=3 fail_timeout=30s;
    }

    upstream integration_bridge {
        server ptaas-integration-bridge:8005 max_fails=3 fail_timeout=30s;
    }

    # Main server configuration
    server {
        listen 80;
        server_name localhost;

        # Health check endpoint
        location /health {
            access_log off;
            return 200 "OK\n";
            add_header Content-Type text/plain;
        }

        # Learning API endpoints
        location /api/v1/learning/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://learning_api/api/v1/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_timeout 30s;
        }

        # Telemetry endpoints (higher rate limit)
        location /api/v1/telemetry {
            limit_req zone=telemetry burst=200 nodelay;
            proxy_pass http://learning_api/api/v1/telemetry;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_timeout 10s;
        }

        # Orchestrator endpoints
        location /api/v1/orchestration/ {
            limit_req zone=api burst=10 nodelay;
            proxy_pass http://orchestrator/api/v1/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_timeout 60s;
        }

        # Security endpoints
        location /api/v1/security/ {
            limit_req zone=api burst=5 nodelay;
            proxy_pass http://security/api/v1/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_timeout 30s;
        }

        # Integration bridge endpoints
        location /api/v1/bridge/ {
            limit_req zone=api burst=50 nodelay;
            proxy_pass http://integration_bridge/api/v1/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_timeout 30s;
        }
    }
}
EOF

    # Loki configuration
    cat > "${SCRIPT_DIR}/monitoring/loki-config.yml" << 'EOF'
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
    final_sleep: 0s
  chunk_idle_period: 1h
  max_chunk_age: 1h
  chunk_target_size: 1048576
  chunk_retain_period: 30s
  max_transfer_retries: 0

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    cache_ttl: 24h
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

compactor:
  working_directory: /loki/boltdb-shipper-compactor
  shared_store: filesystem

limits_config:
  reject_old_samples: true
  reject_old_samples_max_age: 168h

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: false
  retention_period: 0s

ruler:
  storage:
    type: local
    local:
      directory: /loki/rules
  rule_path: /loki/rules-temp
  alertmanager_url: http://alertmanager:9093
  ring:
    kvstore:
      store: inmemory
  enable_api: true
EOF

    # Promtail configuration
    cat > "${SCRIPT_DIR}/monitoring/promtail-config.yml" << 'EOF'
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: xorb-learning-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: xorb-learning
          __path__: /var/log/xorb/*.log

  - job_name: docker-logs
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ['__meta_docker_container_name']
        regex: '/xorb-(.*)'
        target_label: 'container'
      - source_labels: ['__meta_docker_container_log_stream']
        target_label: 'stream'
EOF

    log "Configuration files created"
}

# Build Docker images
build_images() {
    log "Building Docker images..."

    # Create Dockerfiles
    cat > "${SCRIPT_DIR}/Dockerfile.learning-engine" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-learning.txt .
RUN pip install --no-cache-dir -r requirements-learning.txt

# Copy application code
COPY xorb_learning_engine/ ./xorb_learning_engine/
COPY models/ ./models/
COPY certs/ ./certs/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "xorb_learning_engine.core.autonomous_learning_engine"]
EOF

    cat > "${SCRIPT_DIR}/Dockerfile.learning-api" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-learning.txt .
RUN pip install --no-cache-dir -r requirements-learning.txt

# Copy application code
COPY xorb_learning_engine/ ./xorb_learning_engine/
COPY certs/ ./certs/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

CMD ["python", "-m", "xorb_learning_engine.api.learning_api"]
EOF

    cat > "${SCRIPT_DIR}/Dockerfile.orchestrator" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-learning.txt .
RUN pip install --no-cache-dir -r requirements-learning.txt

# Copy application code
COPY xorb_learning_engine/ ./xorb_learning_engine/
COPY data/ ./data/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8003/health')" || exit 1

CMD ["python", "-m", "xorb_learning_engine.orchestration.adaptive_orchestrator"]
EOF

    cat > "${SCRIPT_DIR}/Dockerfile.security" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-learning.txt .
RUN pip install --no-cache-dir -r requirements-learning.txt

# Copy application code
COPY xorb_learning_engine/ ./xorb_learning_engine/
COPY certs/ ./certs/
COPY keys/ ./keys/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8004/health')" || exit 1

CMD ["python", "-m", "xorb_learning_engine.security.security_framework"]
EOF

    cat > "${SCRIPT_DIR}/Dockerfile.integration-tests" << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-learning.txt .
COPY requirements-test.txt .
RUN pip install --no-cache-dir -r requirements-learning.txt
RUN pip install --no-cache-dir -r requirements-test.txt

# Copy application code and tests
COPY xorb_learning_engine/ ./xorb_learning_engine/
COPY tests/ ./tests/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "xorb_learning_engine.tests.integration_test_suite"]
EOF

    # Create requirements files
    cat > "${SCRIPT_DIR}/requirements-learning.txt" << 'EOF'
# Core dependencies
asyncio-mqtt==0.13.0
aioredis==2.0.1
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
asyncpg==0.29.0
redis==5.0.1

# Machine Learning
torch==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
pandas==2.0.3

# Security
cryptography==41.0.7
pyjwt==2.8.0
httpx==0.25.0

# Monitoring
prometheus-client==0.19.0
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0

# Utilities
pydantic==2.5.0
python-multipart==0.0.6
websockets==12.0
aiofiles==23.2.1
EOF

    cat > "${SCRIPT_DIR}/requirements-test.txt" << 'EOF'
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0
pytest-cov==4.1.0
httpx==0.25.0
websockets==12.0
EOF

    log "Docker images built successfully"
}

# Deploy services
deploy_services() {
    log "Deploying XORB Learning Engine Integration services..."

    # Check if Docker and Docker Compose are available
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi

    # Use docker compose or docker-compose based on availability
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi

    # Pull base images
    log "Pulling base Docker images..."
    docker pull python:3.11-slim
    docker pull postgres:15-alpine
    docker pull redis:7-alpine
    docker pull nginx:alpine
    docker pull prom/prometheus:v2.45.0
    docker pull grafana/grafana:10.0.0
    docker pull grafana/loki:2.9.0
    docker pull grafana/promtail:2.9.0
    docker pull qdrant/qdrant:v1.7.0
    docker pull nats:2.10-alpine

    # Deploy services
    log "Starting XORB Learning Integration services..."
    $COMPOSE_CMD -f docker-compose-learning-integration.yml up -d --build

    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 30

    # Check service health
    check_service_health
}

# Check service health
check_service_health() {
    log "Checking service health..."

    services=(
        "postgres:5432"
        "redis:6379"
        "learning-api:8002"
        "adaptive-orchestrator:8003"
        "security-framework:8004"
        "ptaas-integration-bridge:8005"
        "prometheus:9090"
        "grafana:3000"
        "qdrant:6333"
        "nats:4222"
    )

    for service in "${services[@]}"; do
        name=$(echo $service | cut -d: -f1)
        port=$(echo $service | cut -d: -f2)

        if timeout 30 bash -c "until docker exec xorb-${name} nc -z localhost ${port}; do sleep 1; done" 2>/dev/null; then
            log "✓ Service ${name} is healthy"
        else
            warn "✗ Service ${name} health check failed"
        fi
    done
}

# Run integration tests
run_tests() {
    log "Running integration tests..."

    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi

    # Run quick validation tests
    log "Running quick validation tests..."
    $COMPOSE_CMD -f docker-compose-learning-integration.yml run --rm integration-tests python -m xorb_learning_engine.tests.integration_test_suite quick

    # Run comprehensive tests if quick tests pass
    if [[ $? -eq 0 ]]; then
        log "Quick tests passed. Running comprehensive test suite..."
        $COMPOSE_CMD -f docker-compose-learning-integration.yml run --rm integration-tests
    else
        warn "Quick tests failed. Skipping comprehensive tests."
    fi
}

# Show deployment status
show_status() {
    log "XORB Learning Engine Integration Deployment Status"
    echo "=================================================="

    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi

    # Show running services
    $COMPOSE_CMD -f docker-compose-learning-integration.yml ps

    echo ""
    log "Service Access URLs:"
    echo "  Learning API (Direct):          http://localhost:8002"
    echo "  Adaptive Orchestrator:          http://localhost:8003"
    echo "  Security Framework:             http://localhost:8004"
    echo "  Integration Bridge:             http://localhost:8005"
    echo "  Load Balanced API:              http://localhost:8000"
    echo "  Prometheus:                     http://localhost:9090"
    echo "  Grafana:                        http://localhost:3001"
    echo "  Qdrant Vector DB:               http://localhost:6333"
    echo "  NATS JetStream:                 http://localhost:8222"

    echo ""
    log "Default Credentials:"
    echo "  Grafana:                        admin / xorb_grafana_admin_2025"
    echo "  PostgreSQL:                     xorb / xorb_pass"

    echo ""
    log "API Authentication:"
    echo "  Orchestrator API Key:           xorb_orchestrator_key_2025_secure"
    echo "  Admin Dashboard Key:            xorb_admin_dashboard_key_2025"
    echo "  Integration Bridge Key:         xorb_bridge_key_2025_secure"
}

# Show logs
show_logs() {
    local service=$1

    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi

    if [[ -n "$service" ]]; then
        log "Showing logs for service: $service"
        $COMPOSE_CMD -f docker-compose-learning-integration.yml logs -f "$service"
    else
        log "Showing logs for all services"
        $COMPOSE_CMD -f docker-compose-learning-integration.yml logs -f
    fi
}

# Stop services
stop_services() {
    log "Stopping XORB Learning Engine Integration services..."

    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi

    $COMPOSE_CMD -f docker-compose-learning-integration.yml down
}

# Cleanup deployment
cleanup() {
    log "Cleaning up XORB Learning Engine Integration deployment..."

    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi

    # Stop and remove containers, networks, and volumes
    $COMPOSE_CMD -f docker-compose-learning-integration.yml down -v --remove-orphans

    # Remove custom images
    docker images | grep xorb | awk '{print $3}' | xargs -r docker rmi -f

    # Clean up temporary files
    rm -f "${SCRIPT_DIR}/init-learning-dbs.sql"

    log "Cleanup completed"
}

# Main script logic
main() {
    case "${1:-deploy}" in
        "deploy")
            log "Starting XORB Learning Engine Integration deployment..."
            create_directories
            generate_certificates
            create_configs
            build_images
            deploy_services
            show_status
            ;;
        "test")
            run_tests
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs "$2"
            ;;
        "stop")
            stop_services
            ;;
        "cleanup")
            cleanup
            ;;
        "restart")
            stop_services
            sleep 5
            deploy_services
            show_status
            ;;
        *)
            echo "Usage: $0 {deploy|test|status|logs [service]|stop|cleanup|restart}"
            echo ""
            echo "Commands:"
            echo "  deploy    - Deploy complete learning integration (default)"
            echo "  test      - Run integration tests"
            echo "  status    - Show deployment status"
            echo "  logs      - Show logs (optionally for specific service)"
            echo "  stop      - Stop all services"
            echo "  cleanup   - Stop services and remove all data"
            echo "  restart   - Restart all services"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
