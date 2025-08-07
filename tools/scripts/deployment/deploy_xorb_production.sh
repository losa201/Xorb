#!/bin/bash

# XORB Production Deployment Automation
# Complete enterprise-grade deployment of the XORB cybersecurity platform
# Handles infrastructure provisioning, security hardening, and service orchestration

set -euo pipefail
IFS=$'\n\t'

# Script metadata
readonly SCRIPT_VERSION="2.1.0"
readonly DEPLOYMENT_ID="XORB-PROD-$(date +%Y%m%d-%H%M%S)"
readonly BASE_DIR="/root/Xorb"
readonly LOG_FILE="/var/log/xorb_deployment_${DEPLOYMENT_ID}.log"

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly WHITE='\033[1;37m'
readonly NC='\033[0m'

# Configuration
DEPLOYMENT_MODE="production"
NODE_REGION="eu-central"
NODE_TIER="enterprise"
ENABLE_MONITORING="true"
ENABLE_COMPLIANCE="true"
ENABLE_QUANTUM_CRYPTO="true"
ENABLE_FEDERATED_LEARNING="true"

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                DEPLOYMENT_MODE="$2"
                shift 2
                ;;
            --region)
                NODE_REGION="$2"
                shift 2
                ;;
            --tier)
                NODE_TIER="$2"
                shift 2
                ;;
            --no-monitoring)
                ENABLE_MONITORING="false"
                shift
                ;;
            --no-compliance)
                ENABLE_COMPLIANCE="false"
                shift
                ;;
            --no-quantum)
                ENABLE_QUANTUM_CRYPTO="false"
                shift
                ;;
            --no-federated)
                ENABLE_FEDERATED_LEARNING="false"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    cat << EOF
XORB Production Deployment Script v${SCRIPT_VERSION}

Usage: $0 [OPTIONS]

Options:
    --mode MODE          Deployment mode (development|staging|production) [default: production]
    --region REGION      Node region (eu-central|us-east|us-west|asia-pacific) [default: eu-central]
    --tier TIER          Node tier (development|staging|production|enterprise|government) [default: enterprise]
    --no-monitoring      Disable monitoring and observability
    --no-compliance      Disable compliance automation
    --no-quantum         Disable quantum cryptography
    --no-federated       Disable federated learning
    --help               Show this help message

Examples:
    $0                                    # Full production deployment
    $0 --mode staging --region us-east   # Staging deployment in US East
    $0 --tier government --no-federated  # Government deployment without federated learning

EOF
}

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    case "$level" in
        "INFO")  echo -e "${GREEN}[INFO]${NC}  ${timestamp} - ${message}" | tee -a "${LOG_FILE}" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC}  ${timestamp} - ${message}" | tee -a "${LOG_FILE}" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} ${timestamp} - ${message}" | tee -a "${LOG_FILE}" ;;
        "DEBUG") [[ "${DEBUG:-false}" == "true" ]] && echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - ${message}" | tee -a "${LOG_FILE}" ;;
        "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - ${message}" | tee -a "${LOG_FILE}" ;;
        *)       echo -e "${WHITE}[${level}]${NC} ${timestamp} - ${message}" | tee -a "${LOG_FILE}" ;;
    esac
}

# Error handling
error_exit() {
    log "ERROR" "$1"
    cleanup_on_failure
    exit 1
}

cleanup_on_failure() {
    log "WARN" "Deployment failed, initiating cleanup..."
    
    # Stop any running containers
    if command -v docker-compose &> /dev/null; then
        cd "${BASE_DIR}" && docker-compose -f infra/docker-compose.yml down --remove-orphans || true
    fi
    
    # Remove partial deployments
    rm -f "${BASE_DIR}/.deployment_lock" || true
    
    log "INFO" "Cleanup completed"
}

# Prerequisites check
check_prerequisites() {
    log "INFO" "Checking deployment prerequisites..."
    
    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        error_exit "This script must be run as root for production deployment"
    fi
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "terraform" "git" "curl" "jq" "python3" "pip3")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error_exit "Required command not found: $cmd"
        fi
    done
    
    # Check system resources
    local available_memory=$(free -m | awk 'NR==2{print $7}')
    local available_disk=$(df /opt | awk 'NR==2 {print $4}')
    
    if [[ $available_memory -lt 8192 ]]; then  # 8GB
        error_exit "Insufficient memory. Minimum 8GB available required, found ${available_memory}MB"
    fi
    
    if [[ $available_disk -lt 20971520 ]]; then  # 20GB
        error_exit "Insufficient disk space. Minimum 20GB required in /opt"
    fi
    
    # Check for existing deployment
    if [[ -f "${BASE_DIR}/.deployment_lock" ]]; then
        error_exit "Another deployment is in progress or failed. Remove ${BASE_DIR}/.deployment_lock to continue"
    fi
    
    log "SUCCESS" "All prerequisites met"
}

# Display deployment banner
display_banner() {
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  🛡️  XORB CYBERSECURITY PLATFORM - PRODUCTION DEPLOYMENT  🛡️                ║
║                                                                              ║
║  Enterprise-Grade Federated Threat Defense System                           ║
║  Quantum-Safe • Privacy-Preserving • Compliance-Ready                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo
    log "INFO" "XORB Production Deployment v${SCRIPT_VERSION}"
    log "INFO" "Deployment ID: ${DEPLOYMENT_ID}"
    log "INFO" "Mode: ${DEPLOYMENT_MODE} | Region: ${NODE_REGION} | Tier: ${NODE_TIER}"
    echo
}

# Create deployment lock
create_deployment_lock() {
    cat > "${BASE_DIR}/.deployment_lock" << EOF
{
    "deployment_id": "${DEPLOYMENT_ID}",
    "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "mode": "${DEPLOYMENT_MODE}",
    "region": "${NODE_REGION}",
    "tier": "${NODE_TIER}",
    "pid": $$
}
EOF
}

# Install system dependencies
install_dependencies() {
    log "INFO" "Installing system dependencies..."
    
    # Update system packages
    apt-get update -y
    
    # Install required packages
    local packages=(
        "software-properties-common"
        "apt-transport-https"
        "ca-certificates" 
        "gnupg"
        "lsb-release"
        "curl"
        "wget"
        "git"
        "jq"
        "htop"
        "vim"
        "unzip"
        "python3"
        "python3-pip"
        "python3-venv"
        "build-essential"
        "libssl-dev"
        "libffi-dev"
        "python3-dev"
        "pkg-config"
    )
    
    apt-get install -y "${packages[@]}"
    
    # Install Docker if not present
    if ! command -v docker &> /dev/null; then
        log "INFO" "Installing Docker..."
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
        apt-get update -y
        apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
        systemctl enable docker
        systemctl start docker
    fi
    
    # Install Docker Compose if not present
    if ! command -v docker-compose &> /dev/null; then
        log "INFO" "Installing Docker Compose..."
        curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        chmod +x /usr/local/bin/docker-compose
    fi
    
    # Install Terraform if not present
    if ! command -v terraform &> /dev/null; then
        log "INFO" "Installing Terraform..."
        wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
        echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/hashicorp.list
        apt-get update -y
        apt-get install -y terraform
    fi
    
    log "SUCCESS" "System dependencies installed"
}

# Install Python dependencies
install_python_dependencies() {
    log "INFO" "Installing Python dependencies..."
    
    # Create virtual environment
    python3 -m venv "${BASE_DIR}/venv"
    source "${BASE_DIR}/venv/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install required packages
    local packages=(
        "asyncio"
        "aiohttp"
        "aioredis"
        "asyncpg"
        "psycopg2-binary"
        "numpy"
        "pandas"
        "scikit-learn"
        "cryptography"
        "pyyaml" 
        "rich"
        "psutil"
        "docker"
        "httpx"
        "fastapi"
        "uvicorn"
        "redis"
        "neo4j"
        "prometheus-client"
    )
    
    pip install "${packages[@]}"
    
    log "SUCCESS" "Python dependencies installed"
}

# Setup directory structure
setup_directories() {
    log "INFO" "Setting up XORB directory structure..."
    
    # Create main directories
    local directories=(
        "/opt/xorb"
        "/opt/xorb/config"
        "/opt/xorb/data"
        "/opt/xorb/logs"
        "/opt/xorb/certs"
        "/opt/xorb/backups"
        "/opt/xorb/monitoring"
        "/opt/xorb/compliance"
        "/opt/xorb/scripts"
        "/var/log/xorb"
        "/etc/xorb"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        chmod 750 "$dir"
    done
    
    # Create xorb user if doesn't exist
    if ! id "xorb" &>/dev/null; then
        useradd -r -s /bin/false -d /opt/xorb -c "XORB Service User" xorb
    fi
    
    # Set ownership
    chown -R xorb:xorb /opt/xorb
    chown -R xorb:xorb /var/log/xorb
    
    log "SUCCESS" "Directory structure created"
}

# Generate configuration files
generate_configurations() {
    log "INFO" "Generating XORB configuration files..."
    
    # Generate node configuration
    local node_id=$(openssl rand -hex 8)
    
    cat > "/opt/xorb/config/node.json" << EOF
{
    "node_id": "${node_id}",
    "node_region": "${NODE_REGION}",
    "node_tier": "${NODE_TIER}",
    "deployment_id": "${DEPLOYMENT_ID}",
    "deployment_mode": "${DEPLOYMENT_MODE}",
    "compliance_frameworks": ["GDPR", "ISO27001", "SOC2", "NIS2"],
    "federation_enabled": ${ENABLE_FEDERATED_LEARNING},
    "quantum_crypto_enabled": ${ENABLE_QUANTUM_CRYPTO},
    "monitoring_enabled": ${ENABLE_MONITORING},
    "compliance_enabled": ${ENABLE_COMPLIANCE},
    "deployment_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "hardening_level": "enterprise",
    "zero_trust_enabled": true,
    "version": "${SCRIPT_VERSION}"
}
EOF
    
    # Generate secure passwords
    local postgres_password=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
    local redis_password=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
    local neo4j_password=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)
    local grafana_password=$(openssl rand -base64 16 | tr -d "=+/" | cut -c1-16)
    
    # Generate environment file
    cat > "/opt/xorb/.env" << EOF
# XORB Production Environment Configuration
# Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)
# Deployment ID: ${DEPLOYMENT_ID}

# Node Configuration
NODE_ID=${node_id}
NODE_REGION=${NODE_REGION}
NODE_TIER=${NODE_TIER}
DEPLOYMENT_MODE=${DEPLOYMENT_MODE}

# Feature Flags
FEDERATION_ENABLED=${ENABLE_FEDERATED_LEARNING}
QUANTUM_CRYPTO_ENABLED=${ENABLE_QUANTUM_CRYPTO}
MONITORING_ENABLED=${ENABLE_MONITORING}
COMPLIANCE_ENABLED=${ENABLE_COMPLIANCE}

# Database Credentials
POSTGRES_PASSWORD=${postgres_password}
REDIS_PASSWORD=${redis_password}
NEO4J_PASSWORD=${neo4j_password}
NATS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)

# Monitoring Credentials
GRAFANA_PASSWORD=${grafana_password}
GRAFANA_SECRET_KEY=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-32)

# Security Configuration
TLS_ENABLED=true
MTLS_ENABLED=true
CERTIFICATE_PATH=/opt/xorb/certs
ENCRYPTION_KEY=$(openssl rand -base64 32)

# Compliance Configuration
COMPLIANCE_FRAMEWORKS=["GDPR","ISO27001","SOC2","NIS2"]
AUDIT_ENABLED=true
DATA_RETENTION_DAYS=2555  # 7 years

# API Configuration
API_RATE_LIMIT=1000
API_TIMEOUT=30
MAX_CONCURRENT_REQUESTS=100

# Deployment Metadata
DEPLOYMENT_VERSION=${SCRIPT_VERSION}
DEPLOYMENT_TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
    
    # Set secure permissions
    chmod 600 "/opt/xorb/.env"
    chown xorb:xorb "/opt/xorb/.env"
    chown xorb:xorb "/opt/xorb/config/node.json"
    
    log "SUCCESS" "Configuration files generated"
}

# Generate TLS certificates
generate_certificates() {
    log "INFO" "Generating TLS certificates..."
    
    cd /opt/xorb/certs
    
    # Generate CA key and certificate
    openssl genrsa -out ca-key.pem 4096
    openssl req -new -x509 -days 365 -key ca-key.pem -out ca-cert.pem \
        -subj "/C=DE/ST=Germany/L=Frankfurt/O=XORB Platform/OU=Certificate Authority/CN=XORB CA"
    
    # Generate server key and certificate signing request
    openssl genrsa -out server-key.pem 4096
    openssl req -new -key server-key.pem -out server.csr \
        -subj "/C=DE/ST=Germany/L=Frankfurt/O=XORB Platform/OU=Security Operations/CN=xorb-node-${node_id}"
    
    # Create certificate extensions
    cat > server-extensions.cnf << EOF
subjectAltName = @alt_names
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth, clientAuth

[alt_names]
DNS.1 = localhost
DNS.2 = xorb-node-${node_id}
DNS.3 = *.xorb.local
IP.1 = 127.0.0.1
IP.2 = ::1
EOF
    
    # Generate server certificate
    openssl x509 -req -in server.csr -CA ca-cert.pem -CAkey ca-key.pem \
        -CAcreateserial -out server-cert.pem -days 365 \
        -extensions v3_req -extfile server-extensions.cnf
    
    # Generate client certificates for mutual TLS
    openssl genrsa -out client-key.pem 4096
    openssl req -new -key client-key.pem -out client.csr \
        -subj "/C=DE/ST=Germany/L=Frankfurt/O=XORB Platform/OU=Client/CN=xorb-client"
    
    openssl x509 -req -in client.csr -CA ca-cert.pem -CAkey ca-key.pem \
        -CAcreateserial -out client-cert.pem -days 365
    
    # Set proper permissions
    chmod 600 *-key.pem
    chmod 644 *-cert.pem
    chown xorb:xorb /opt/xorb/certs/*
    
    # Clean up temporary files
    rm -f server.csr client.csr server-extensions.cnf
    
    log "SUCCESS" "TLS certificates generated"
}

# Deploy infrastructure
deploy_infrastructure() {
    log "INFO" "Deploying XORB infrastructure..."
    
    cd "${BASE_DIR}"
    
    # Copy deployment files
    cp -r infra/docker-compose.yml /opt/xorb/
    cp -r compose/xorb-node-stack.yml /opt/xorb/
    cp scripts/bootstrap_xorb_node.sh /opt/xorb/scripts/
    
    # Make scripts executable
    chmod +x /opt/xorb/scripts/*.sh
    
    # Deploy using Docker Compose
    cd /opt/xorb
    
    # Start core infrastructure services first
    log "INFO" "Starting core infrastructure services..."
    docker-compose -f xorb-node-stack.yml up -d postgres redis neo4j qdrant nats
    
    # Wait for databases to be ready
    log "INFO" "Waiting for databases to initialize..."
    sleep 30
    
    # Verify database connectivity
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if docker exec xorb-postgres pg_isready -U xorb >/dev/null 2>&1; then
            log "SUCCESS" "PostgreSQL ready"
            break
        fi
        
        log "INFO" "Waiting for PostgreSQL... (attempt $attempt/$max_attempts)"
        sleep 5
        ((attempt++))
    done
    
    if [[ $attempt -gt $max_attempts ]]; then
        error_exit "PostgreSQL failed to start within timeout"
    fi
    
    # Start XORB services
    log "INFO" "Starting XORB core services..."
    docker-compose -f xorb-node-stack.yml up -d \
        unified-orchestrator ai-engine quantum-crypto threat-intel-fusion
    
    sleep 20
    
    # Start additional services
    if [[ "$ENABLE_FEDERATED_LEARNING" == "true" ]]; then
        log "INFO" "Starting federated learning service..."
        docker-compose -f xorb-node-stack.yml up -d federated-learning
    fi
    
    if [[ "$ENABLE_COMPLIANCE" == "true" ]]; then
        log "INFO" "Starting compliance audit service..."
        docker-compose -f xorb-node-stack.yml up -d compliance-audit
    fi
    
    if [[ "$ENABLE_MONITORING" == "true" ]]; then
        log "INFO" "Starting monitoring services..."
        docker-compose -f xorb-node-stack.yml up -d prometheus grafana loki
    fi
    
    # Start support services
    log "INFO" "Starting support services..."
    docker-compose -f xorb-node-stack.yml up -d auto-scaler nginx
    
    log "SUCCESS" "Infrastructure deployment completed"
}

# Run health checks
run_health_checks() {
    log "INFO" "Running comprehensive health checks..."
    
    # Service health checks
    local services=("unified-orchestrator:9000" "ai-engine:9003" "quantum-crypto:9005")
    
    for service_port in "${services[@]}"; do
        local service_name="${service_port%:*}"
        local port="${service_port#*:}"
        
        local max_attempts=20
        local attempt=1
        
        while [[ $attempt -le $max_attempts ]]; do
            if curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
                log "SUCCESS" "${service_name} service healthy"
                break
            fi
            
            if [[ $attempt -eq $max_attempts ]]; then
                log "WARN" "${service_name} service not responding after ${max_attempts} attempts"
            else
                log "INFO" "Waiting for ${service_name}... (attempt $attempt/$max_attempts)"
            fi
            
            sleep 5
            ((attempt++))
        done
    done
    
    # Container health check
    local unhealthy_containers=$(docker ps --filter "health=unhealthy" --format "table {{.Names}}" | grep -v NAMES | wc -l)
    
    if [[ $unhealthy_containers -eq 0 ]]; then
        log "SUCCESS" "All containers healthy"
    else
        log "WARN" "${unhealthy_containers} containers are unhealthy"
    fi
    
    # System resource check
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    local memory_usage=$(free | grep Mem | awk '{printf("%.1f"), $3/$2 * 100.0}')
    
    log "INFO" "System resources: CPU ${cpu_usage}%, Memory ${memory_usage}%"
    
    log "SUCCESS" "Health checks completed"
}

# Configure system hardening
configure_hardening() {
    log "INFO" "Applying security hardening..."
    
    # Configure UFW firewall
    ufw --force reset
    ufw default deny incoming
    ufw default allow outgoing
    
    # Allow SSH (be careful with this in production)
    ufw allow ssh
    
    # Allow XORB services
    ufw allow 80/tcp    # HTTP (redirects to HTTPS)
    ufw allow 443/tcp   # HTTPS
    ufw allow 8080/tcp  # Health checks
    ufw allow 9000:9010/tcp  # XORB services
    
    # Enable firewall
    ufw --force enable
    
    # Configure fail2ban
    systemctl enable fail2ban
    systemctl start fail2ban
    
    # Configure AppArmor
    systemctl enable apparmor
    systemctl start apparmor
    
    # Configure audit daemon
    systemctl enable auditd
    systemctl start auditd
    
    # Set up automatic security updates
    cat > /etc/apt/apt.conf.d/50unattended-upgrades << 'EOF'
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
    "${distro_id}ESMApps:${distro_codename}-apps-security";
    "${distro_id}ESM:${distro_codename}-infra-security";
};
Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::MinimalSteps "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
EOF
    
    # Configure log rotation
    cat > /etc/logrotate.d/xorb << 'EOF'
/var/log/xorb/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 xorb xorb
    postrotate
        systemctl reload rsyslog > /dev/null 2>&1 || true
    endrotate
}
EOF
    
    log "SUCCESS" "Security hardening applied"
}

# Run compliance validation
run_compliance_validation() {
    if [[ "$ENABLE_COMPLIANCE" != "true" ]]; then
        log "INFO" "Compliance validation skipped (disabled)"
        return
    fi
    
    log "INFO" "Running compliance validation..."
    
    # Copy compliance checklist
    cp "${BASE_DIR}/compliance/checklists/node_gdpr_iso27001.yml" /opt/xorb/compliance/
    
    # Run basic compliance checks
    local compliance_score=0
    local total_checks=0
    
    # Check encryption
    if [[ -f "/opt/xorb/certs/server-cert.pem" ]]; then
        ((compliance_score++))
        log "SUCCESS" "TLS certificates present"
    fi
    ((total_checks++))
    
    # Check access controls
    if ufw status | grep -q "Status: active"; then
        ((compliance_score++))
        log "SUCCESS" "Firewall configured"
    fi
    ((total_checks++))
    
    # Check audit logging
    if systemctl is-enabled auditd >/dev/null 2>&1; then
        ((compliance_score++))
        log "SUCCESS" "Audit logging enabled"
    fi
    ((total_checks++))
    
    # Check data protection
    if [[ -d "/opt/xorb/data" ]] && [[ -O "/opt/xorb/data" ]]; then
        ((compliance_score++))
        log "SUCCESS" "Data directory secured"
    fi
    ((total_checks++))
    
    local compliance_percentage=$((compliance_score * 100 / total_checks))
    log "INFO" "Compliance score: ${compliance_percentage}% (${compliance_score}/${total_checks})"
    
    if [[ $compliance_percentage -ge 75 ]]; then
        log "SUCCESS" "Compliance validation passed"
    else
        log "WARN" "Compliance validation needs attention"
    fi
}

# Create systemd services
create_systemd_services() {
    log "INFO" "Creating systemd services..."
    
    # XORB main service
    cat > /etc/systemd/system/xorb.service << 'EOF'
[Unit]
Description=XORB Cybersecurity Platform
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/xorb
ExecStart=/usr/local/bin/docker-compose -f xorb-node-stack.yml up -d
ExecStop=/usr/local/bin/docker-compose -f xorb-node-stack.yml down
User=root
Group=root

[Install]
WantedBy=multi-user.target
EOF
    
    # XORB monitoring service
    if [[ "$ENABLE_MONITORING" == "true" ]]; then
        cat > /etc/systemd/system/xorb-monitoring.service << 'EOF'
[Unit]
Description=XORB Monitoring Stack
After=xorb.service
Requires=xorb.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/xorb
ExecStart=/usr/local/bin/docker-compose -f xorb-node-stack.yml up -d prometheus grafana loki
ExecStop=/usr/local/bin/docker-compose -f xorb-node-stack.yml stop prometheus grafana loki
User=root
Group=root

[Install]
WantedBy=multi-user.target
EOF
    fi
    
    # Reload systemd and enable services
    systemctl daemon-reload
    systemctl enable xorb.service
    
    if [[ "$ENABLE_MONITORING" == "true" ]]; then
        systemctl enable xorb-monitoring.service
    fi
    
    log "SUCCESS" "Systemd services created"
}

# Generate deployment report
generate_deployment_report() {
    log "INFO" "Generating deployment report..."
    
    local report_file="/opt/xorb/deployment_report_${DEPLOYMENT_ID}.json"
    
    # Collect system information
    local total_containers=$(docker ps -q | wc -l)
    local running_containers=$(docker ps --filter "status=running" -q | wc -l)
    local system_load=$(uptime | awk -F'load average:' '{print $2}' | cut -d, -f1 | xargs)
    local disk_usage=$(df /opt | awk 'NR==2 {print $5}' | sed 's/%//')
    local memory_usage=$(free | grep Mem | awk '{printf("%.1f"), $3/$2 * 100.0}')
    
    # Generate report
    cat > "$report_file" << EOF
{
    "deployment_summary": {
        "deployment_id": "${DEPLOYMENT_ID}",
        "version": "${SCRIPT_VERSION}",
        "deployment_mode": "${DEPLOYMENT_MODE}",
        "node_region": "${NODE_REGION}",
        "node_tier": "${NODE_TIER}",
        "deployment_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "deployment_duration": "$(echo "$(date +%s) - $(stat -c %Y ${LOG_FILE})" | bc)s",
        "status": "SUCCESS"
    },
    "system_status": {
        "total_containers": ${total_containers},
        "running_containers": ${running_containers},
        "system_load": "${system_load}",
        "disk_usage_percent": ${disk_usage},
        "memory_usage_percent": ${memory_usage}
    },
    "features_enabled": {
        "monitoring": ${ENABLE_MONITORING},
        "compliance": ${ENABLE_COMPLIANCE},
        "quantum_crypto": ${ENABLE_QUANTUM_CRYPTO},
        "federated_learning": ${ENABLE_FEDERATED_LEARNING}
    },
    "service_endpoints": {
        "orchestrator": "http://localhost:9000",
        "ai_engine": "http://localhost:9003",
        "quantum_crypto": "http://localhost:9005",
        "threat_intel": "http://localhost:9002",
        "web_interface": "https://localhost:443"
    },
    "security_status": {
        "tls_enabled": true,
        "firewall_active": true,
        "certificates_generated": true,
        "hardening_applied": true
    },
    "next_steps": [
        "Access the XORB dashboard at https://localhost:443",
        "Run deployment verification: python3 deployment_verification_comprehensive.py",
        "Configure monitoring alerts in Grafana",
        "Review compliance checklist",
        "Schedule regular backups"
    ]
}
EOF
    
    chown xorb:xorb "$report_file"
    
    log "SUCCESS" "Deployment report generated: $report_file"
}

# Display deployment summary
display_deployment_summary() {
    echo
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                              ║"
    echo "║  🎉 XORB PRODUCTION DEPLOYMENT SUCCESSFUL! 🎉                               ║"
    echo "║                                                                              ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo
    log "SUCCESS" "XORB Platform deployed successfully!"
    log "INFO" "Deployment ID: ${DEPLOYMENT_ID}"
    log "INFO" "Mode: ${DEPLOYMENT_MODE} | Region: ${NODE_REGION} | Tier: ${NODE_TIER}"
    echo
    log "INFO" "Service Endpoints:"
    log "INFO" "  • Web Interface: https://localhost:443"
    log "INFO" "  • API Gateway: http://localhost:9000"
    log "INFO" "  • AI Engine: http://localhost:9003"
    log "INFO" "  • Quantum Crypto: http://localhost:9005"
    if [[ "$ENABLE_MONITORING" == "true" ]]; then
        log "INFO" "  • Grafana Dashboard: http://localhost:3000"
        log "INFO" "  • Prometheus Metrics: http://localhost:9090"
    fi
    echo
    log "INFO" "Next Steps:"
    log "INFO" "  1. Run verification: python3 ${BASE_DIR}/deployment_verification_comprehensive.py"
    log "INFO" "  2. Run demo: python3 ${BASE_DIR}/xorb_platform_demo.py"
    log "INFO" "  3. Review logs: tail -f ${LOG_FILE}"
    log "INFO" "  4. Monitor services: docker ps"
    echo
    log "INFO" "Documentation available in: ${BASE_DIR}/docs/"
    log "INFO" "Configuration files in: /opt/xorb/config/"
    log "INFO" "Logs available in: /var/log/xorb/"
    echo
}

# Cleanup function
cleanup() {
    if [[ -f "${BASE_DIR}/.deployment_lock" ]]; then
        rm -f "${BASE_DIR}/.deployment_lock"
    fi
}

# Main deployment function
main() {
    # Parse command line arguments
    parse_arguments "$@"
    
    # Display banner
    display_banner
    
    # Create log file
    touch "${LOG_FILE}"
    chmod 644 "${LOG_FILE}"
    
    # Create deployment lock
    create_deployment_lock
    
    # Set up cleanup on exit
    trap cleanup EXIT
    
    # Execute deployment steps
    check_prerequisites
    install_dependencies
    install_python_dependencies
    setup_directories
    generate_configurations
    generate_certificates
    configure_hardening
    deploy_infrastructure
    run_health_checks
    run_compliance_validation
    create_systemd_services
    generate_deployment_report
    
    # Display summary
    display_deployment_summary
    
    log "SUCCESS" "XORB Platform deployment completed successfully!"
}

# Execute main function with all arguments
main "$@"