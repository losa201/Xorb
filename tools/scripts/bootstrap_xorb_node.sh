#!/bin/bash

# XORB Federated Node Bootstrap Script
# Secure, idempotent deployment and management of XORB nodes
# Zero-trust, compliance-ready, enterprise-grade automation

set -euo pipefail
IFS=$'\n\t'

# Script metadata
readonly SCRIPT_VERSION="2.1.0"
readonly SCRIPT_NAME="XORB Node Bootstrap"
readonly SCRIPT_AUTHOR="XORB Platform Team"
readonly MIN_BASH_VERSION="4.4"

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly WHITE='\033[1;37m'
readonly NC='\033[0m' # No Color

# Configuration paths
readonly XORB_HOME="/opt/xorb"
readonly XORB_CONFIG="${XORB_HOME}/config"
readonly XORB_DATA="${XORB_HOME}/encrypted-data"
readonly XORB_LOGS="${XORB_HOME}/logs"
readonly XORB_CERTS="${XORB_HOME}/certs"
readonly XORB_USER="xorb-service"
readonly COMPOSE_FILE="${XORB_HOME}/compose/xorb-node-stack.yml"
readonly NODE_CONFIG="${XORB_CONFIG}/node.json"
readonly PID_FILE="${XORB_HOME}/xorb.pid"

# Default configuration
DEFAULT_NODE_REGION="eu-central"
DEFAULT_NODE_TIER="enterprise"
DEFAULT_COMPLIANCE_FRAMEWORKS='["GDPR","ISO27001","SOC2"]'
DEFAULT_FEDERATION_ENABLED="true"
DEFAULT_QUANTUM_CRYPTO_ENABLED="true"

# Logging function with structured output
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    case "$level" in
        "INFO")  echo -e "${GREEN}[INFO]${NC}  ${timestamp} - ${message}" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC}  ${timestamp} - ${message}" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} ${timestamp} - ${message}" ;;
        "DEBUG") [[ "${DEBUG:-false}" == "true" ]] && echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - ${message}" ;;
        "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - ${message}" ;;
        *)       echo -e "${WHITE}[${level}]${NC} ${timestamp} - ${message}" ;;
    esac
    
    # Also log to system journal
    echo "${timestamp} [${level}] ${message}" | systemd-cat -t xorb-bootstrap
}

# Error handling
error_exit() {
    log "ERROR" "$1"
    cleanup
    exit 1
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "WARN" "Script exited with error code: $exit_code"
        log "INFO" "Cleaning up any incomplete operations..."
    fi
    
    # Remove temporary files
    rm -f /tmp/xorb-*
    return $exit_code
}

# Signal handling
trap cleanup EXIT
trap 'error_exit "Script interrupted by user"' INT TERM

# Validation functions
check_prerequisites() {
    log "INFO" "Checking system prerequisites..."
    
    # Check bash version
    if [[ "${BASH_VERSION%%.*}" -lt "${MIN_BASH_VERSION%%.*}" ]]; then
        error_exit "Bash version ${MIN_BASH_VERSION} or higher required. Current: ${BASH_VERSION}"
    fi
    
    # Check if running as root initially (will drop privileges later)
    if [[ $EUID -ne 0 ]]; then
        error_exit "This script must be run as root initially for system setup"
    fi
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "curl" "jq" "openssl" "systemctl")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error_exit "Required command not found: $cmd"
        fi
    done
    
    # Check disk space (minimum 10GB free)
    local available_space=$(df /opt | awk 'NR==2 {print $4}')
    if [[ $available_space -lt 10485760 ]]; then  # 10GB in KB
        error_exit "Insufficient disk space. Minimum 10GB required in /opt"
    fi
    
    # Check memory (minimum 4GB)
    local total_memory=$(free -m | awk 'NR==2{print $2}')
    if [[ $total_memory -lt 4096 ]]; then
        error_exit "Insufficient memory. Minimum 4GB RAM required"
    fi
    
    log "SUCCESS" "All prerequisites met"
}

# Load and validate node configuration
load_node_config() {
    log "INFO" "Loading node configuration..."
    
    if [[ ! -f "$NODE_CONFIG" ]]; then
        log "WARN" "Node config not found. Creating default configuration..."
        create_default_config
    fi
    
    # Validate JSON format
    if ! jq empty "$NODE_CONFIG" 2>/dev/null; then
        error_exit "Invalid JSON in node configuration: $NODE_CONFIG"
    fi
    
    # Load configuration variables
    export NODE_ID=$(jq -r '.node_id // "unknown"' "$NODE_CONFIG")
    export NODE_REGION=$(jq -r '.node_region // "'"$DEFAULT_NODE_REGION"'"' "$NODE_CONFIG")
    export NODE_TIER=$(jq -r '.node_tier // "'"$DEFAULT_NODE_TIER"'"' "$NODE_CONFIG")
    export COMPLIANCE_FRAMEWORKS=$(jq -c '.compliance_frameworks // '"$DEFAULT_COMPLIANCE_FRAMEWORKS"'' "$NODE_CONFIG")
    export FEDERATION_ENABLED=$(jq -r '.federation_enabled // '"$DEFAULT_FEDERATION_ENABLED"'' "$NODE_CONFIG")
    export QUANTUM_CRYPTO_ENABLED=$(jq -r '.quantum_crypto_enabled // '"$DEFAULT_QUANTUM_CRYPTO_ENABLED"'' "$NODE_CONFIG")
    
    # Generate secure passwords if not exists
    export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$(generate_secure_password 32)}"
    export REDIS_PASSWORD="${REDIS_PASSWORD:-$(generate_secure_password 32)}"
    export NEO4J_PASSWORD="${NEO4J_PASSWORD:-$(generate_secure_password 32)}"
    export NATS_PASSWORD="${NATS_PASSWORD:-$(generate_secure_password 32)}"
    export GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-$(generate_secure_password 16)}"
    export GRAFANA_SECRET_KEY="${GRAFANA_SECRET_KEY:-$(generate_secure_password 32)}"
    
    log "SUCCESS" "Configuration loaded for node: $NODE_ID (Region: $NODE_REGION, Tier: $NODE_TIER)"
}

# Generate secure password
generate_secure_password() {
    local length="${1:-32}"
    openssl rand -base64 "$length" | tr -d "=+/" | cut -c1-"$length"
}

# Create default configuration
create_default_config() {
    local node_id=$(openssl rand -hex 8)
    
    mkdir -p "$XORB_CONFIG"
    cat > "$NODE_CONFIG" << EOF
{
  "node_id": "$node_id",
  "node_region": "$DEFAULT_NODE_REGION",
  "node_tier": "$DEFAULT_NODE_TIER",
  "compliance_frameworks": $DEFAULT_COMPLIANCE_FRAMEWORKS,
  "federation_enabled": $DEFAULT_FEDERATION_ENABLED,
  "quantum_crypto_enabled": $DEFAULT_QUANTUM_CRYPTO_ENABLED,
  "deployment_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hardening_level": "enterprise",
  "zero_trust_enabled": true,
  "bootstrap_version": "$SCRIPT_VERSION"
}
EOF
    
    chown "$XORB_USER:$XORB_USER" "$NODE_CONFIG"
    chmod 640 "$NODE_CONFIG"
    
    log "SUCCESS" "Default configuration created with node ID: $node_id"
}

# Setup directory structure
setup_directories() {
    log "INFO" "Setting up XORB directory structure..."
    
    local directories=(
        "$XORB_HOME"
        "$XORB_CONFIG"
        "$XORB_DATA"
        "$XORB_LOGS"
        "$XORB_CERTS"
        "$XORB_HOME/compose"
        "$XORB_HOME/backups"
        "$XORB_HOME/monitoring"
        "$XORB_DATA/postgres"
        "$XORB_DATA/redis"
        "$XORB_DATA/neo4j"
        "$XORB_DATA/qdrant"
        "$XORB_DATA/prometheus"
        "$XORB_DATA/grafana"
        "$XORB_DATA/loki"
        "$XORB_DATA/nats"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            chown "$XORB_USER:$XORB_USER" "$dir"
            chmod 750 "$dir"
        fi
    done
    
    log "SUCCESS" "Directory structure configured"
}

# Generate TLS certificates
generate_certificates() {
    log "INFO" "Generating TLS certificates..."
    
    local cert_config="${XORB_CERTS}/openssl.conf"
    local ca_key="${XORB_CERTS}/ca-key.pem"
    local ca_cert="${XORB_CERTS}/ca-cert.pem"
    local server_key="${XORB_CERTS}/server-key.pem"
    local server_cert="${XORB_CERTS}/server-cert.pem"
    
    # Skip if certificates already exist and are valid
    if [[ -f "$server_cert" ]] && openssl x509 -in "$server_cert" -noout -checkend 2592000 &>/dev/null; then
        log "INFO" "Valid certificates already exist, skipping generation"
        return 0
    fi
    
    # Create OpenSSL configuration
    cat > "$cert_config" << EOF
[req]
default_bits = 4096
prompt = no
distinguished_name = dn
req_extensions = v3_req

[dn]
C=DE
ST=Germany
L=Frankfurt
O=XORB Platform
OU=Security Operations
CN=xorb-node-${NODE_ID}

[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = xorb-node-${NODE_ID}
DNS.3 = *.xorb.local
IP.1 = 127.0.0.1
IP.2 = ::1
EOF
    
    # Generate CA key and certificate
    openssl genrsa -out "$ca_key" 4096
    openssl req -new -x509 -days 365 -key "$ca_key" -out "$ca_cert" \
        -subj "/C=DE/ST=Germany/L=Frankfurt/O=XORB Platform/OU=Certificate Authority/CN=XORB CA"
    
    # Generate server key and certificate
    openssl genrsa -out "$server_key" 4096
    openssl req -new -key "$server_key" -out "${XORB_CERTS}/server.csr" -config "$cert_config"
    openssl x509 -req -in "${XORB_CERTS}/server.csr" -CA "$ca_cert" -CAkey "$ca_key" \
        -CAcreateserial -out "$server_cert" -days 365 -extensions v3_req -extfile "$cert_config"
    
    # Set proper permissions
    chmod 600 "$ca_key" "$server_key"
    chmod 644 "$ca_cert" "$server_cert"
    chown "$XORB_USER:$XORB_USER" "${XORB_CERTS}"/*
    
    # Clean up
    rm -f "${XORB_CERTS}/server.csr" "$cert_config"
    
    log "SUCCESS" "TLS certificates generated and configured"
}

# Download and prepare Docker Compose stack
prepare_compose_stack() {
    log "INFO" "Preparing Docker Compose stack..."
    
    # Copy the hardened stack configuration
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        # This would typically download from a secure repository
        # For now, we'll create a reference to the existing stack
        ln -sf "${XORB_HOME}/../compose/xorb-node-stack.yml" "$COMPOSE_FILE" || {
            error_exit "Failed to link Docker Compose stack"
        }
    fi
    
    # Create environment file
    local env_file="${XORB_HOME}/.env"
    cat > "$env_file" << EOF
# XORB Node Environment Configuration
NODE_ID=${NODE_ID}
NODE_REGION=${NODE_REGION}
NODE_TIER=${NODE_TIER}
COMPLIANCE_FRAMEWORKS=${COMPLIANCE_FRAMEWORKS}
FEDERATION_ENABLED=${FEDERATION_ENABLED}
QUANTUM_CRYPTO_ENABLED=${QUANTUM_CRYPTO_ENABLED}

# Database Credentials
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
REDIS_PASSWORD=${REDIS_PASSWORD}
NEO4J_PASSWORD=${NEO4J_PASSWORD}
NATS_PASSWORD=${NATS_PASSWORD}

# Monitoring Credentials
GRAFANA_PASSWORD=${GRAFANA_PASSWORD}
GRAFANA_SECRET_KEY=${GRAFANA_SECRET_KEY}

# API Keys (if configured)
NVIDIA_API_KEY=${NVIDIA_API_KEY:-}

# Domain Configuration
DOMAIN_NAME=${DOMAIN_NAME:-localhost}
EOF
    
    chmod 600 "$env_file"
    chown "$XORB_USER:$XORB_USER" "$env_file"
    
    log "SUCCESS" "Docker Compose stack prepared"
}

# Health check function
health_check() {
    local service="${1:-all}"
    local timeout="${2:-300}"
    local interval="${3:-10}"
    
    log "INFO" "Performing health check for: $service"
    
    local start_time=$(date +%s)
    local end_time=$((start_time + timeout))
    
    while [[ $(date +%s) -lt $end_time ]]; do
        local healthy=true
        
        case "$service" in
            "all"|"unified-orchestrator")
                if ! curl -sf http://localhost:9000/health >/dev/null 2>&1; then
                    healthy=false
                fi
                ;;
            "ai-engine")
                if ! curl -sf http://localhost:9003/health >/dev/null 2>&1; then
                    healthy=false
                fi
                ;;
            "postgres")
                if ! docker exec xorb-postgres pg_isready -U xorb >/dev/null 2>&1; then
                    healthy=false
                fi
                ;;
            "redis")
                if ! docker exec xorb-redis redis-cli -a "$REDIS_PASSWORD" ping >/dev/null 2>&1; then
                    healthy=false
                fi
                ;;
        esac
        
        if [[ "$healthy" == "true" ]]; then
            log "SUCCESS" "Health check passed for: $service"
            return 0
        fi
        
        log "DEBUG" "Waiting for $service to become healthy..."
        sleep "$interval"
    done
    
    log "ERROR" "Health check failed for $service after ${timeout}s timeout"
    return 1
}

# Start XORB services
start_services() {
    log "INFO" "Starting XORB federated node services..."
    
    cd "$XORB_HOME"
    
    # Pull latest images
    sudo -u "$XORB_USER" docker-compose -f "$COMPOSE_FILE" pull
    
    # Start services with dependency order
    local service_groups=(
        "postgres redis neo4j qdrant nats"
        "unified-orchestrator ai-engine quantum-crypto threat-intel-fusion"
        "auto-scaler federated-learning compliance-audit"
        "nginx prometheus grafana loki"
    )
    
    for group in "${service_groups[@]}"; do
        log "INFO" "Starting service group: $group"
        sudo -u "$XORB_USER" docker-compose -f "$COMPOSE_FILE" up -d $group
        
        # Wait for services in this group to be healthy
        for service in $group; do
            if ! health_check "$service" 120 5; then
                error_exit "Failed to start service: $service"
            fi
        done
        
        sleep 10  # Brief pause between groups
    done
    
    # Store PID for monitoring
    echo $$ > "$PID_FILE"
    chown "$XORB_USER:$XORB_USER" "$PID_FILE"
    
    log "SUCCESS" "All XORB services started successfully"
}

# Stop XORB services
stop_services() {
    log "INFO" "Stopping XORB federated node services..."
    
    cd "$XORB_HOME"
    
    # Graceful shutdown
    sudo -u "$XORB_USER" docker-compose -f "$COMPOSE_FILE" stop
    
    # Remove containers but keep volumes
    sudo -u "$XORB_USER" docker-compose -f "$COMPOSE_FILE" rm -f
    
    # Clean up PID file
    rm -f "$PID_FILE"
    
    log "SUCCESS" "XORB services stopped"
}

# Reload configuration
reload_services() {
    log "INFO" "Reloading XORB services with new configuration..."
    
    load_node_config
    prepare_compose_stack
    
    cd "$XORB_HOME"
    
    # Restart services that need configuration reload
    local reload_services="unified-orchestrator ai-engine quantum-crypto threat-intel-fusion"
    
    sudo -u "$XORB_USER" docker-compose -f "$COMPOSE_FILE" restart $reload_services
    
    # Health check after reload
    for service in $reload_services; do
        health_check "$service" 60 5
    done
    
    log "SUCCESS" "Services reloaded successfully"
}

# Status check
check_status() {
    log "INFO" "Checking XORB node status..."
    
    # Check if PID file exists and process is running
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log "SUCCESS" "XORB bootstrap process is running (PID: $pid)"
        else
            log "WARN" "Stale PID file found, removing..."
            rm -f "$PID_FILE"
        fi
    fi
    
    # Check Docker services
    cd "$XORB_HOME"
    local running_services=$(sudo -u "$XORB_USER" docker-compose -f "$COMPOSE_FILE" ps --services --filter status=running 2>/dev/null | wc -l)
    local total_services=$(sudo -u "$XORB_USER" docker-compose -f "$COMPOSE_FILE" config --services 2>/dev/null | wc -l)
    
    log "INFO" "Docker services: $running_services/$total_services running"
    
    # Overall health check
    if health_check "all" 30 5; then
        log "SUCCESS" "XORB node is healthy and operational"
        return 0
    else
        log "WARN" "XORB node health check failed"
        return 1
    fi
}

# Backup function
backup_node() {
    log "INFO" "Creating XORB node backup..."
    
    local backup_dir="${XORB_HOME}/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup configuration
    cp -r "$XORB_CONFIG" "${backup_dir}/"
    
    # Backup certificates
    cp -r "$XORB_CERTS" "${backup_dir}/"
    
    # Export database backups
    if docker exec xorb-postgres pg_isready -U xorb >/dev/null 2>&1; then
        docker exec xorb-postgres pg_dump -U xorb xorb > "${backup_dir}/postgres_backup.sql"
    fi
    
    # Create archive
    tar -czf "${backup_dir}.tar.gz" -C "$backup_dir" .
    rm -rf "$backup_dir"
    
    # Keep only last 7 backups
    find "${XORB_HOME}/backups" -name "*.tar.gz" -mtime +7 -delete
    
    log "SUCCESS" "Backup created: ${backup_dir}.tar.gz"
}

# Main execution function
main() {
    local action="${1:-start}"
    
    # Display banner
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    XORB Federated Node                       ║"
    echo "║                  Bootstrap & Management                      ║"
    echo "║                                                              ║"
    echo "║  Version: $SCRIPT_VERSION                                    ║"
    echo "║  Zero-Trust • Quantum-Safe • Enterprise-Grade               ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    case "$action" in
        "start")
            log "INFO" "Starting XORB node bootstrap process..."
            check_prerequisites
            load_node_config
            setup_directories
            generate_certificates
            prepare_compose_stack
            start_services
            log "SUCCESS" "XORB federated node is operational!"
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            stop_services
            sleep 5
            main "start"
            ;;
        "reload")
            reload_services
            ;;
        "status")
            check_status
            ;;
        "health")
            health_check "all" 60 5
            ;;
        "backup")
            backup_node
            ;;
        "version")
            echo "$SCRIPT_VERSION"
            ;;
        *)
            echo "Usage: $0 {start|stop|restart|reload|status|health|backup|version}"
            echo ""
            echo "Commands:"
            echo "  start   - Initialize and start XORB node"
            echo "  stop    - Stop all XORB services"
            echo "  restart - Stop and start XORB node"
            echo "  reload  - Reload configuration without full restart"
            echo "  status  - Check node status"
            echo "  health  - Perform comprehensive health check"
            echo "  backup  - Create node backup"
            echo "  version - Show script version"
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"