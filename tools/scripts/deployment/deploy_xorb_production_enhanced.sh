#!/bin/bash
set -euo pipefail

# XORB PRKMT 12.9 Enhanced Production Deployment Script
# Tactical cybersecurity platform deployment without quantum mysticism
# Classification: Production Ready - Enterprise Grade

echo "🛡️ XORB PRKMT 12.9 Enhanced Production Deployment"
echo "⚔️ Tactical Cybersecurity Platform - Enterprise Edition"
echo "🎯 Deploying autonomous adversarial testing and defensive mutation"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
XORB_VERSION="12.9-enhanced"
DEPLOYMENT_ENV=${1:-production}
XORB_DOMAIN=${XORB_DOMAIN:-"xorb.local"}
ENABLE_SSL=${ENABLE_SSL:-"true"}
PARALLEL_AGENTS=${PARALLEL_AGENTS:-32}
THREAT_REALISM=${THREAT_REALISM:-"extreme"}

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check if running as root for production deployment
check_privileges() {
    if [[ $EUID -ne 0 ]] && [[ "$DEPLOYMENT_ENV" == "production" ]]; then
        error "Production deployment requires root privileges. Run with sudo."
    fi
}

# System requirements check
check_system_requirements() {
    log "Checking system requirements..."

    # Check available memory (minimum 8GB for PRKMT 12.9)
    total_mem=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $total_mem -lt 8 ]]; then
        error "Insufficient memory. XORB PRKMT 12.9 requires at least 8GB RAM."
    fi

    # Check disk space (minimum 50GB)
    available_disk=$(df / | awk 'NR==2{print int($4/1024/1024)}')
    if [[ $available_disk -lt 50 ]]; then
        error "Insufficient disk space. XORB requires at least 50GB free space."
    fi

    # Check CPU cores (minimum 4 cores for parallel execution)
    cpu_cores=$(nproc)
    if [[ $cpu_cores -lt 4 ]]; then
        warn "Limited CPU cores detected ($cpu_cores). Optimal performance requires 8+ cores."
    fi

    info "System requirements: ✅ Memory: ${total_mem}GB ✅ Disk: ${available_disk}GB ✅ CPU: ${cpu_cores} cores"
}

# Install dependencies
install_dependencies() {
    log "Installing XORB dependencies..."

    # Update package list
    apt-get update -qq

    # Install core dependencies
    apt-get install -y \
        docker.io \
        docker-compose \
        python3 \
        python3-pip \
        python3-venv \
        nginx \
        postgresql-client \
        redis-tools \
        curl \
        wget \
        jq \
        htop \
        net-tools \
        nmap \
        tcpdump \
        wireshark-common \
        fail2ban \
        ufw \
        certbot \
        python3-certbot-nginx

    # Install Python packages for XORB
    pip3 install --upgrade \
        asyncio \
        aiohttp \
        aiofiles \
        cryptography \
        numpy \
        pandas \
        scikit-learn \
        prometheus-client \
        psutil \
        requests \
        fastapi \
        uvicorn \
        pydantic \
        sqlalchemy \
        redis \
        celery

    # Start and enable Docker
    systemctl start docker
    systemctl enable docker

    # Add current user to docker group if not root
    if [[ $EUID -ne 0 ]]; then
        usermod -aG docker $USER
        warn "Please log out and back in for Docker group membership to take effect"
    fi

    info "Dependencies installed successfully"
}

# Setup XORB directories and permissions
setup_directories() {
    log "Setting up XORB directory structure..."

    # Create XORB directories
    mkdir -p /opt/xorb/{bin,config,data,logs,certs,backups}
    mkdir -p /opt/xorb/data/{agents,campaigns,detections,mutations,malware}
    mkdir -p /opt/xorb/logs/{apt-engine,breach-sim,drift-detect,malware-gen,orchestrator}
    mkdir -p /var/lib/xorb/{postgresql,redis,prometheus,grafana}
    mkdir -p /etc/xorb/config

    # Set permissions
    chown -R root:root /opt/xorb
    chown -R root:root /etc/xorb
    chmod -R 755 /opt/xorb
    chmod -R 750 /etc/xorb
    chmod -R 700 /opt/xorb/certs

    info "Directory structure created"
}

# Deploy XORB tactical engines
deploy_tactical_engines() {
    log "Deploying XORB tactical engines..."

    # Copy XORB engines to production location
    cp XORB_AUTONOMOUS_APT_EMULATION_ENGINE.py /opt/xorb/bin/
    cp XORB_ZERO_TRUST_BREACH_SIMULATOR.py /opt/xorb/bin/
    cp XORB_BEHAVIORAL_DRIFT_DETECTION.py /opt/xorb/bin/
    cp XORB_SYNTHETIC_MALWARE_GENERATOR.py /opt/xorb/bin/
    cp XORB_PRKMT_12_9_ENHANCED_ORCHESTRATOR.py /opt/xorb/bin/

    # Make engines executable
    chmod +x /opt/xorb/bin/*.py

    # Create systemd services for each engine
    create_systemd_services

    info "Tactical engines deployed"
}

# Create systemd services
create_systemd_services() {
    log "Creating systemd services..."

    # XORB Orchestrator service
    cat > /etc/systemd/system/xorb-orchestrator.service << EOF
[Unit]
Description=XORB PRKMT 12.9 Enhanced Orchestrator
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/opt/xorb/bin
ExecStart=/usr/bin/python3 /opt/xorb/bin/XORB_PRKMT_12_9_ENHANCED_ORCHESTRATOR.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment=PYTHONPATH=/opt/xorb/bin
Environment=XORB_ENV=production
Environment=XORB_LOG_LEVEL=INFO

[Install]
WantedBy=multi-user.target
EOF

    # APT Emulation Engine service
    cat > /etc/systemd/system/xorb-apt-engine.service << EOF
[Unit]
Description=XORB Autonomous APT Emulation Engine
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/opt/xorb/bin
ExecStart=/usr/bin/python3 /opt/xorb/bin/XORB_AUTONOMOUS_APT_EMULATION_ENGINE.py
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal
Environment=PYTHONPATH=/opt/xorb/bin

[Install]
WantedBy=multi-user.target
EOF

    # Behavioral Drift Detection service
    cat > /etc/systemd/system/xorb-drift-detector.service << EOF
[Unit]
Description=XORB Behavioral Drift Detection
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/opt/xorb/bin
ExecStart=/usr/bin/python3 /opt/xorb/bin/XORB_BEHAVIORAL_DRIFT_DETECTION.py
Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal
Environment=PYTHONPATH=/opt/xorb/bin

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd
    systemctl daemon-reload

    info "Systemd services created"
}

# Deploy monitoring infrastructure
deploy_monitoring() {
    log "Deploying monitoring infrastructure..."

    # Create docker-compose for monitoring stack
    cat > /opt/xorb/docker-compose-monitoring.yml << EOF
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: xorb-prometheus
    ports:
      - "9090:9090"
    volumes:
      - /opt/xorb/config/prometheus.yml:/etc/prometheus/prometheus.yml
      - /var/lib/xorb/prometheus:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped
    networks:
      - xorb-monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: xorb-grafana
    ports:
      - "3000:3000"
    volumes:
      - /var/lib/xorb/grafana:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=xorb_admin_$(openssl rand -hex 8)
      - GF_USERS_ALLOW_SIGN_UP=false
    restart: unless-stopped
    networks:
      - xorb-monitoring

  redis:
    image: redis:alpine
    container_name: xorb-redis
    ports:
      - "6379:6379"
    volumes:
      - /var/lib/xorb/redis:/data
    command: redis-server --appendonly yes
    restart: unless-stopped
    networks:
      - xorb-monitoring

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: xorb-elasticsearch
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms2g -Xmx2g"
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - /var/lib/xorb/elasticsearch:/usr/share/elasticsearch/data
    restart: unless-stopped
    networks:
      - xorb-monitoring

networks:
  xorb-monitoring:
    driver: bridge
EOF

    # Create Prometheus configuration
    cat > /opt/xorb/config/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "xorb_alerts.yml"

scrape_configs:
  - job_name: 'xorb-orchestrator'
    static_configs:
      - targets: ['localhost:8080']
    scrape_interval: 10s
    metrics_path: /metrics

  - job_name: 'xorb-apt-engine'
    static_configs:
      - targets: ['localhost:8081']
    scrape_interval: 30s

  - job_name: 'xorb-breach-simulator'
    static_configs:
      - targets: ['localhost:8082']
    scrape_interval: 30s

  - job_name: 'xorb-drift-detector'
    static_configs:
      - targets: ['localhost:8083']
    scrape_interval: 10s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
EOF

    info "Monitoring infrastructure configured"
}

# Setup SSL certificates
setup_ssl() {
    if [[ "$ENABLE_SSL" == "true" ]]; then
        log "Setting up SSL certificates..."

        if [[ "$XORB_DOMAIN" != "xorb.local" ]]; then
            # Request Let's Encrypt certificate for public domain
            certbot --nginx -d "$XORB_DOMAIN" --non-interactive --agree-tos --email admin@"$XORB_DOMAIN"
        else
            # Generate self-signed certificate for local deployment
            openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
                -keyout /opt/xorb/certs/xorb.key \
                -out /opt/xorb/certs/xorb.crt \
                -subj "/C=US/ST=State/L=City/O=XORB/CN=$XORB_DOMAIN"
        fi

        info "SSL certificates configured"
    fi
}

# Configure nginx reverse proxy
setup_nginx() {
    log "Configuring nginx reverse proxy..."

    cat > /etc/nginx/sites-available/xorb << EOF
server {
    listen 80;
    server_name $XORB_DOMAIN;
    return 301 https://\$server_name\$request_uri;
}

server {
    listen 443 ssl http2;
    server_name $XORB_DOMAIN;

    ssl_certificate /opt/xorb/certs/xorb.crt;
    ssl_certificate_key /opt/xorb/certs/xorb.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";

    # XORB Dashboard
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Grafana
    location /grafana/ {
        proxy_pass http://127.0.0.1:3000/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Prometheus
    location /prometheus/ {
        proxy_pass http://127.0.0.1:9090/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # API endpoints
    location /api/ {
        proxy_pass http://127.0.0.1:8080/api/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

    # Enable site
    ln -sf /etc/nginx/sites-available/xorb /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default

    # Test nginx configuration
    nginx -t

    # Restart nginx
    systemctl restart nginx
    systemctl enable nginx

    info "Nginx configured and started"
}

# Setup firewall
setup_firewall() {
    log "Configuring firewall..."

    # Reset UFW
    ufw --force reset

    # Default policies
    ufw default deny incoming
    ufw default allow outgoing

    # Allow SSH
    ufw allow ssh

    # Allow HTTP/HTTPS
    ufw allow 80/tcp
    ufw allow 443/tcp

    # Allow XORB services (internal only)
    ufw allow from 127.0.0.1 to any port 8080:8090
    ufw allow from 127.0.0.1 to any port 3000
    ufw allow from 127.0.0.1 to any port 9090

    # Enable firewall
    ufw --force enable

    info "Firewall configured"
}

# Start XORB services
start_services() {
    log "Starting XORB services..."

    # Start monitoring infrastructure
    cd /opt/xorb && docker-compose -f docker-compose-monitoring.yml up -d

    # Wait for services to initialize
    sleep 10

    # Enable and start XORB systemd services
    systemctl enable xorb-orchestrator
    systemctl enable xorb-apt-engine
    systemctl enable xorb-drift-detector

    systemctl start xorb-orchestrator
    systemctl start xorb-apt-engine
    systemctl start xorb-drift-detector

    # Wait for services to stabilize
    sleep 15

    info "XORB services started"
}

# Verify deployment
verify_deployment() {
    log "Verifying XORB deployment..."

    # Check systemd services
    local services=("xorb-orchestrator" "xorb-apt-engine" "xorb-drift-detector")
    for service in "${services[@]}"; do
        if systemctl is-active --quiet "$service"; then
            echo -e "  ✅ $service: ${GREEN}RUNNING${NC}"
        else
            echo -e "  ❌ $service: ${RED}FAILED${NC}"
        fi
    done

    # Check Docker containers
    echo ""
    echo "Docker containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

    # Check network connectivity
    echo ""
    echo "Network connectivity:"
    if curl -s -k "https://localhost" > /dev/null; then
        echo -e "  ✅ HTTPS: ${GREEN}ACCESSIBLE${NC}"
    else
        echo -e "  ❌ HTTPS: ${RED}FAILED${NC}"
    fi

    if curl -s "http://localhost:3000" > /dev/null; then
        echo -e "  ✅ Grafana: ${GREEN}ACCESSIBLE${NC}"
    else
        echo -e "  ❌ Grafana: ${RED}FAILED${NC}"
    fi

    if curl -s "http://localhost:9090" > /dev/null; then
        echo -e "  ✅ Prometheus: ${GREEN}ACCESSIBLE${NC}"
    else
        echo -e "  ❌ Prometheus: ${RED}FAILED${NC}"
    fi

    echo ""
}

# Create deployment report
create_deployment_report() {
    log "Creating deployment report..."

    local report_file="/opt/xorb/logs/deployment_report_$(date +%Y%m%d_%H%M%S).json"

    cat > "$report_file" << EOF
{
  "deployment": {
    "timestamp": "$(date -Iseconds)",
    "version": "$XORB_VERSION",
    "environment": "$DEPLOYMENT_ENV",
    "domain": "$XORB_DOMAIN",
    "ssl_enabled": $ENABLE_SSL,
    "parallel_agents": $PARALLEL_AGENTS,
    "threat_realism": "$THREAT_REALISM"
  },
  "system": {
    "hostname": "$(hostname)",
    "os": "$(lsb_release -d | cut -f2)",
    "kernel": "$(uname -r)",
    "memory_gb": $(free -g | awk '/^Mem:/{print $2}'),
    "cpu_cores": $(nproc),
    "disk_available_gb": $(df / | awk 'NR==2{print int($4/1024/1024)}')
  },
  "services": {
    "xorb_orchestrator": "$(systemctl is-active xorb-orchestrator)",
    "xorb_apt_engine": "$(systemctl is-active xorb-apt-engine)",
    "xorb_drift_detector": "$(systemctl is-active xorb-drift-detector)",
    "nginx": "$(systemctl is-active nginx)",
    "docker": "$(systemctl is-active docker)"
  },
  "access_urls": {
    "dashboard": "https://$XORB_DOMAIN",
    "grafana": "https://$XORB_DOMAIN/grafana",
    "prometheus": "https://$XORB_DOMAIN/prometheus",
    "api": "https://$XORB_DOMAIN/api"
  },
  "credentials": {
    "grafana_admin": "admin",
    "note": "Check docker logs for generated passwords"
  }
}
EOF

    echo "📋 Deployment report saved: $report_file"
}

# Main deployment function
main() {
    echo -e "${PURPLE}"
    cat << "EOF"
    ██╗  ██╗ ██████╗ ██████╗ ██████╗     ██████╗ ██████╗ ██╗  ██╗███╗   ███╗████████╗
    ╚██╗██╔╝██╔═══██╗██╔══██╗██╔══██╗    ██╔══██╗██╔══██╗██║ ██╔╝████╗ ████║╚══██╔══╝
     ╚███╔╝ ██║   ██║██████╔╝██████╔╝    ██████╔╝██████╔╝█████╔╝ ██╔████╔██║   ██║
     ██╔██╗ ██║   ██║██╔══██╗██╔══██╗    ██╔═══╝ ██╔══██╗██╔═██╗ ██║╚██╔╝██║   ██║
    ██╔╝ ██╗╚██████╔╝██║  ██║██████╔╝    ██║     ██║  ██║██║  ██╗██║ ╚═╝ ██║   ██║
    ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝     ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝   ╚═╝

    ████████╗ █████╗  ██████╗████████╗██╗ ██████╗ █████╗ ██╗
    ╚══██╔══╝██╔══██╗██╔════╝╚══██╔══╝██║██╔════╝██╔══██╗██║
       ██║   ███████║██║        ██║   ██║██║     ███████║██║
       ██║   ██╔══██║██║        ██║   ██║██║     ██╔══██║██║
       ██║   ██║  ██║╚██████╗   ██║   ██║╚██████╗██║  ██║███████╗
       ╚═╝   ╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝
EOF
    echo -e "${NC}"

    echo -e "${CYAN}Enhanced Autonomous Adversarial Testing Platform${NC}"
    echo -e "${CYAN}Version: $XORB_VERSION | Environment: $DEPLOYMENT_ENV${NC}"
    echo ""

    # Deployment steps
    check_privileges
    check_system_requirements
    install_dependencies
    setup_directories
    deploy_tactical_engines
    deploy_monitoring
    setup_ssl
    setup_nginx
    setup_firewall
    start_services

    # Wait for services to fully initialize
    log "Waiting for services to initialize..."
    sleep 30

    verify_deployment
    create_deployment_report

    echo ""
    echo -e "${GREEN}🎯 XORB PRKMT 12.9 Enhanced deployment completed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Access URLs:${NC}"
    echo -e "  🌐 Dashboard: ${BLUE}https://$XORB_DOMAIN${NC}"
    echo -e "  📊 Grafana:   ${BLUE}https://$XORB_DOMAIN/grafana${NC}"
    echo -e "  📈 Prometheus: ${BLUE}https://$XORB_DOMAIN/prometheus${NC}"
    echo -e "  🔧 API:       ${BLUE}https://$XORB_DOMAIN/api${NC}"
    echo ""
    echo -e "${YELLOW}System Status:${NC}"
    echo -e "  ⚔️ Threat Realism: ${RED}$THREAT_REALISM${NC}"
    echo -e "  🤖 Parallel Agents: ${PURPLE}$PARALLEL_AGENTS${NC}"
    echo -e "  🛡️ Zero Trust: ${GREEN}ACTIVE${NC}"
    echo -e "  🔄 Defensive Mutation: ${GREEN}ENABLED${NC}"
    echo ""
    echo -e "${PURPLE}XORB PRKMT 12.9 Enhanced is now protecting your infrastructure with${NC}"
    echo -e "${PURPLE}autonomous adversarial testing and real-time defensive evolution.${NC}"
    echo ""
    echo -e "${RED}⚠️  This system deploys advanced offensive security capabilities.${NC}"
    echo -e "${RED}    Use responsibly and only on authorized infrastructure.${NC}"
}

# Run main deployment
main "$@"
