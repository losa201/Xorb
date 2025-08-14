#!/bin/bash
set -euo pipefail

# XORB Red Team Infrastructure Deployment Script
# Containerized adversarial testing environment

echo "🔴 XORB Red Team Infrastructure Deployment"
echo "⚔️ Containerized Adversarial Testing Environment"
echo "🎯 Tactical cybersecurity testing without quantum mysticism"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
REDTEAM_ENV=${1:-development}
REDTEAM_DOMAIN=${REDTEAM_DOMAIN:-"redteam.local"}
ISOLATION_LEVEL=${ISOLATION_LEVEL:-"high"}
THREAT_INTENSITY=${THREAT_INTENSITY:-"extreme"}

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

# Check Docker and Docker Compose
check_docker() {
    log "Checking Docker installation..."

    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker first."
    fi

    info "Docker environment: ✅ Ready"
}

# Create required directories
create_directories() {
    log "Creating red team infrastructure directories..."

    mkdir -p {data,logs,config,reports,targets}/{campaigns,techniques,malware,signatures,breaches,network-maps,behavioral,baselines,pcaps,analysis,c2}
    mkdir -p targets/{web,api}
    mkdir -p config/{grafana-redteam-dashboards,prometheus}
    mkdir -p compose

    # Set appropriate permissions
    chmod -R 755 data logs config reports targets
    chmod 700 data/malware data/c2  # Extra security for sensitive data

    info "Directory structure created"
}

# Generate configuration files
generate_configs() {
    log "Generating configuration files..."

    # Prometheus configuration for red team
    cat > config/prometheus-redteam.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'xorb-red-orchestrator'
    static_configs:
      - targets: ['xorb-red-orchestrator:8080']
    scrape_interval: 10s

  - job_name: 'xorb-apt-emulator'
    static_configs:
      - targets: ['xorb-apt-emulator:8080']
    scrape_interval: 30s

  - job_name: 'xorb-breach-simulator'
    static_configs:
      - targets: ['xorb-breach-simulator:8080']
    scrape_interval: 30s

  - job_name: 'xorb-drift-monitor'
    static_configs:
      - targets: ['xorb-drift-monitor:8080']
    scrape_interval: 10s

  - job_name: 'red-team-metrics'
    static_configs:
      - targets: ['reporting-engine:8080']
    scrape_interval: 60s
EOF

    # Fluentd configuration
    cat > config/fluentd.conf << EOF
<source>
  @type forward
  port 24224
  bind 0.0.0.0
</source>

<match xorb.redteam.**>
  @type file
  path /opt/xorb/logs/redteam
  append true
  time_slice_format %Y%m%d
  time_slice_wait 10m
  time_format %Y%m%dT%H%M%S%z
  format json
</match>

<match **>
  @type stdout
</match>
EOF

    # Red team database initialization
    cat > compose/init-red-db.sql << EOF
-- XORB Red Team Database Schema
CREATE DATABASE xorb_redteam;

\c xorb_redteam;

CREATE TABLE campaigns (
    id SERIAL PRIMARY KEY,
    campaign_id VARCHAR(50) UNIQUE NOT NULL,
    apt_group VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    techniques_count INTEGER DEFAULT 0,
    success_rate FLOAT DEFAULT 0.0
);

CREATE TABLE attack_techniques (
    id SERIAL PRIMARY KEY,
    campaign_id VARCHAR(50) REFERENCES campaigns(campaign_id),
    technique_id VARCHAR(20) NOT NULL,
    technique_name VARCHAR(255) NOT NULL,
    target_system VARCHAR(100) NOT NULL,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN DEFAULT FALSE,
    detected BOOLEAN DEFAULT FALSE,
    response_time FLOAT,
    artifacts TEXT
);

CREATE TABLE detection_events (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(50) UNIQUE NOT NULL,
    agent_id VARCHAR(100) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    confidence FLOAT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    details TEXT,
    resolved BOOLEAN DEFAULT FALSE
);

CREATE TABLE defensive_mutations (
    id SERIAL PRIMARY KEY,
    mutation_id VARCHAR(50) UNIQUE NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    target_system VARCHAR(100) NOT NULL,
    trigger_event VARCHAR(100) NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    effectiveness_score FLOAT,
    status VARCHAR(20) DEFAULT 'pending'
);

-- Create indexes for performance
CREATE INDEX idx_campaigns_status ON campaigns(status);
CREATE INDEX idx_techniques_campaign ON attack_techniques(campaign_id);
CREATE INDEX idx_events_timestamp ON detection_events(timestamp);
CREATE INDEX idx_mutations_status ON defensive_mutations(status);

-- Insert sample data
INSERT INTO campaigns (campaign_id, apt_group, status) VALUES
    ('CAMPAIGN-APT28-DEMO', 'apt28', 'active'),
    ('CAMPAIGN-LAZARUS-DEMO', 'lazarus', 'completed');

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO xorb_red;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO xorb_red;
EOF

    info "Configuration files generated"
}

# Create Dockerfiles
create_dockerfiles() {
    log "Creating Dockerfiles for red team services..."

    # Red Team Orchestrator Dockerfile
    cat > compose/Dockerfile.red-orchestrator << EOF
FROM python:3.11-slim

WORKDIR /opt/xorb

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    nmap \\
    netcat-traditional \\
    tcpdump \\
    && rm -rf /var/lib/apt/lists/*

# Copy XORB red team files
COPY XORB_PRKMT_12_9_ENHANCED_ORCHESTRATOR.py .
COPY XORB_AUTONOMOUS_APT_EMULATION_ENGINE.py .
COPY XORB_ZERO_TRUST_BREACH_SIMULATOR.py .
COPY XORB_BEHAVIORAL_DRIFT_DETECTION.py .
COPY XORB_SYNTHETIC_MALWARE_GENERATOR.py .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN useradd -m -u 1000 xorb && chown -R xorb:xorb /opt/xorb
USER xorb

EXPOSE 8080

CMD ["python", "XORB_PRKMT_12_9_ENHANCED_ORCHESTRATOR.py"]
EOF

    # APT Emulator Dockerfile
    cat > compose/Dockerfile.apt-emulator << EOF
FROM python:3.11-slim

WORKDIR /opt/xorb

RUN apt-get update && apt-get install -y \\
    nmap \\
    masscan \\
    sqlmap \\
    nikto \\
    && rm -rf /var/lib/apt/lists/*

COPY XORB_AUTONOMOUS_APT_EMULATION_ENGINE.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 xorb && chown -R xorb:xorb /opt/xorb
USER xorb

CMD ["python", "XORB_AUTONOMOUS_APT_EMULATION_ENGINE.py"]
EOF

    # Malware Lab Dockerfile
    cat > compose/Dockerfile.malware-lab << EOF
FROM python:3.11-slim

WORKDIR /opt/xorb

# WARNING: This container generates synthetic malware for testing
# Do not use in production environments
ENV MALWARE_LAB=true
ENV CONTAINER_PURPOSE=defensive_testing

RUN apt-get update && apt-get install -y \\
    yara \\
    clamav \\
    && rm -rf /var/lib/apt/lists/*

COPY XORB_SYNTHETIC_MALWARE_GENERATOR.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 xorb && chown -R xorb:xorb /opt/xorb
USER xorb

CMD ["python", "XORB_SYNTHETIC_MALWARE_GENERATOR.py"]
EOF

    # Breach Simulator Dockerfile
    cat > compose/Dockerfile.breach-simulator << EOF
FROM python:3.11-slim

WORKDIR /opt/xorb

RUN apt-get update && apt-get install -y \\
    nmap \\
    netcat-traditional \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

COPY XORB_ZERO_TRUST_BREACH_SIMULATOR.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 xorb && chown -R xorb:xorb /opt/xorb
USER xorb

CMD ["python", "XORB_ZERO_TRUST_BREACH_SIMULATOR.py"]
EOF

    # Drift Monitor Dockerfile
    cat > compose/Dockerfile.drift-monitor << EOF
FROM python:3.11-slim

WORKDIR /opt/xorb

COPY XORB_BEHAVIORAL_DRIFT_DETECTION.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 xorb && chown -R xorb:xorb /opt/xorb
USER xorb

CMD ["python", "XORB_BEHAVIORAL_DRIFT_DETECTION.py"]
EOF

    # Traffic Analyzer Dockerfile
    cat > compose/Dockerfile.traffic-analyzer << EOF
FROM python:3.11-slim

WORKDIR /opt/xorb

RUN apt-get update && apt-get install -y \\
    tcpdump \\
    wireshark-common \\
    tshark \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt scapy

RUN useradd -m -u 1000 xorb && chown -R xorb:xorb /opt/xorb
USER xorb

CMD ["python", "-c", "import time; time.sleep(3600)"]
EOF

    # C2 Server Dockerfile
    cat > compose/Dockerfile.c2-server << EOF
FROM python:3.11-alpine

WORKDIR /opt/xorb

RUN apk add --no-cache curl

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt flask

RUN adduser -D -u 1000 xorb && chown -R xorb:xorb /opt/xorb
USER xorb

EXPOSE 80 443

CMD ["python", "-c", "import time; time.sleep(3600)"]
EOF

    # Reporting Engine Dockerfile
    cat > compose/Dockerfile.reporting << EOF
FROM python:3.11-slim

WORKDIR /opt/xorb

RUN apt-get update && apt-get install -y \\
    wkhtmltopdf \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt jinja2 weasyprint

RUN useradd -m -u 1000 xorb && chown -R xorb:xorb /opt/xorb
USER xorb

EXPOSE 8080

CMD ["python", "-c", "import time; time.sleep(3600)"]
EOF

    info "Dockerfiles created"
}

# Create target applications
create_targets() {
    log "Creating target applications for testing..."

    # Vulnerable web application
    cat > targets/web/index.html << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Vulnerable Web App - XORB Target</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .warning { background: #ffeb3b; padding: 20px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>🎯 XORB Test Target Application</h1>
    <div class="warning">
        <strong>⚠️ WARNING:</strong> This is a deliberately vulnerable application for testing purposes only.
        Do not deploy in production environments.
    </div>

    <h2>Vulnerable Features:</h2>
    <ul>
        <li><a href="/login.php">SQL Injection Login</a></li>
        <li><a href="/upload.php">File Upload</a></li>
        <li><a href="/search.php">XSS Search</a></li>
        <li><a href="/admin.php">Weak Authentication</a></li>
    </ul>

    <h2>System Information:</h2>
    <p>Server: nginx/1.21.0</p>
    <p>PHP: 7.4.0 (Vulnerable)</p>
    <p>Database: PostgreSQL 13</p>

    <p><em>This target is monitored by XORB for adversarial testing.</em></p>
</body>
</html>
EOF

    # Nginx configuration for target
    cat > targets/nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    server {
        listen 80;
        server_name target-web;
        root /usr/share/nginx/html;
        index index.html;

        # Deliberately weak security headers for testing
        add_header X-Frame-Options "ALLOWALL";
        add_header X-XSS-Protection "0";
        add_header X-Content-Type-Options "";

        location / {
            try_files \$uri \$uri/ =404;
        }

        # Log all requests for analysis
        access_log /var/log/nginx/access.log;
        error_log /var/log/nginx/error.log;
    }
}
EOF

    # Target database initialization
    cat > targets/db-init.sql << EOF
-- Vulnerable database for XORB testing
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    password VARCHAR(50) NOT NULL,  -- Stored in plaintext (vulnerable)
    email VARCHAR(100),
    role VARCHAR(20) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE sensitive_data (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    data_type VARCHAR(50),
    content TEXT,
    classification VARCHAR(20) DEFAULT 'confidential'
);

-- Insert vulnerable test data
INSERT INTO users (username, password, email, role) VALUES
    ('admin', 'admin123', 'admin@target.local', 'admin'),
    ('user1', 'password', 'user1@target.local', 'user'),
    ('testuser', 'test123', 'test@target.local', 'user'),
    ('guest', 'guest', 'guest@target.local', 'guest');

INSERT INTO sensitive_data (user_id, data_type, content) VALUES
    (1, 'credit_card', '4532-1234-5678-9012'),
    (1, 'ssn', '123-45-6789'),
    (2, 'personal_info', 'Sensitive personal information'),
    (3, 'financial_data', 'Bank account: 987654321');

-- Deliberately weak permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO webapp;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO webapp;
EOF

    # Simple Flask API target
    mkdir -p targets/api
    cat > targets/api/app.py << EOF
#!/usr/bin/env python3
"""
Vulnerable API Target for XORB Testing
WARNING: Contains deliberate vulnerabilities for testing only
"""

from flask import Flask, request, jsonify
import os
import subprocess

app = Flask(__name__)

@app.route('/')
def home():
    return {"message": "Vulnerable API Target", "warning": "For testing only"}

@app.route('/api/users')
def get_users():
    # SQL injection vulnerability
    user_id = request.args.get('id', '')
    query = f"SELECT * FROM users WHERE id = {user_id}"  # Vulnerable
    return {"query": query, "warning": "SQL injection possible"}

@app.route('/api/exec')
def exec_command():
    # Command injection vulnerability
    cmd = request.args.get('cmd', 'whoami')
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return {"command": cmd, "output": result.stdout, "error": result.stderr}
    except Exception as e:
        return {"error": str(e)}

@app.route('/api/file')
def read_file():
    # Path traversal vulnerability
    filename = request.args.get('file', '/etc/passwd')
    try:
        with open(filename, 'r') as f:
            content = f.read()
        return {"file": filename, "content": content}
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
EOF

    cat > targets/api/Dockerfile << EOF
FROM python:3.11-slim

WORKDIR /app

RUN pip install flask

COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
EOF

    cat > targets/api/requirements.txt << EOF
flask==2.3.3
EOF

    info "Target applications created"
}

# Deploy infrastructure
deploy_infrastructure() {
    log "Deploying XORB Red Team Infrastructure..."

    # Build and start services
    docker-compose -f docker-compose-redteam-infrastructure.yml build --parallel
    docker-compose -f docker-compose-redteam-infrastructure.yml up -d

    # Wait for services to start
    log "Waiting for services to initialize..."
    sleep 30

    # Check service health
    check_service_health

    info "Red Team Infrastructure deployed successfully"
}

# Check service health
check_service_health() {
    log "Checking service health..."

    local services=(
        "xorb-red-orchestrator"
        "xorb-apt-emulator"
        "xorb-malware-lab"
        "xorb-breach-simulator"
        "xorb-drift-monitor"
        "target-web-server"
        "target-database"
        "target-api-server"
    )

    for service in "${services[@]}"; do
        if docker ps --filter "name=$service" --filter "status=running" | grep -q "$service"; then
            echo -e "  ✅ $service: ${GREEN}RUNNING${NC}"
        else
            echo -e "  ❌ $service: ${RED}FAILED${NC}"
        fi
    done
}

# Generate deployment report
generate_report() {
    log "Generating deployment report..."

    local report_file="reports/redteam_deployment_$(date +%Y%m%d_%H%M%S).json"

    cat > "$report_file" << EOF
{
  "deployment": {
    "timestamp": "$(date -Iseconds)",
    "environment": "$REDTEAM_ENV",
    "domain": "$REDTEAM_DOMAIN",
    "isolation_level": "$ISOLATION_LEVEL",
    "threat_intensity": "$THREAT_INTENSITY"
  },
  "services": {
    "orchestrator": "http://localhost:8100",
    "target_web": "http://localhost:8200",
    "target_api": "http://localhost:8201",
    "c2_server": "http://localhost:8300",
    "reporting": "http://localhost:8400",
    "metrics": "http://localhost:9091",
    "grafana": "http://localhost:3001"
  },
  "networks": {
    "red_team": "172.30.0.0/16",
    "isolated": "172.31.0.0/16",
    "targets": "172.32.0.0/16"
  },
  "warnings": [
    "This infrastructure contains active adversarial testing tools",
    "Malware lab generates synthetic malware samples",
    "Use only in controlled, isolated environments",
    "Do not deploy on production networks"
  ]
}
EOF

    echo "📋 Deployment report saved: $report_file"
}

# Main deployment function
main() {
    echo -e "${RED}"
    cat << "EOF"
    ██╗  ██╗ ██████╗ ██████╗ ██████╗     ██████╗ ███████╗██████╗
    ╚██╗██╔╝██╔═══██╗██╔══██╗██╔══██╗    ██╔══██╗██╔════╝██╔══██╗
     ╚███╔╝ ██║   ██║██████╔╝██████╔╝    ██████╔╝█████╗  ██║  ██║
     ██╔██╗ ██║   ██║██╔══██╗██╔══██╗    ██╔══██╗██╔══╝  ██║  ██║
    ██╔╝ ██╗╚██████╔╝██║  ██║██████╔╝    ██║  ██║███████╗██████╔╝
    ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝     ╚═╝  ╚═╝╚══════╝╚═════╝

    ████████╗███████╗ █████╗ ███╗   ███╗
    ╚══██╔══╝██╔════╝██╔══██╗████╗ ████║
       ██║   █████╗  ███████║██╔████╔██║
       ██║   ██╔══╝  ██╔══██║██║╚██╔╝██║
       ██║   ███████╗██║  ██║██║ ╚═╝ ██║
       ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝
EOF
    echo -e "${NC}"

    echo -e "${CYAN}Containerized Adversarial Testing Infrastructure${NC}"
    echo -e "${CYAN}Environment: $REDTEAM_ENV | Isolation: $ISOLATION_LEVEL | Threat: $THREAT_INTENSITY${NC}"
    echo ""

    # Deployment steps
    check_docker
    create_directories
    generate_configs
    create_dockerfiles
    create_targets
    deploy_infrastructure
    generate_report

    echo ""
    echo -e "${GREEN}🎯 XORB Red Team Infrastructure deployment completed!${NC}"
    echo ""
    echo -e "${YELLOW}Access URLs:${NC}"
    echo -e "  🔴 Red Team Dashboard: ${BLUE}http://localhost:8100${NC}"
    echo -e "  🎯 Target Web App:     ${BLUE}http://localhost:8200${NC}"
    echo -e "  🎯 Target API:         ${BLUE}http://localhost:8201${NC}"
    echo -e "  💀 C2 Server:          ${BLUE}http://localhost:8300${NC}"
    echo -e "  📊 Reporting:          ${BLUE}http://localhost:8400${NC}"
    echo -e "  📈 Metrics:            ${BLUE}http://localhost:9091${NC}"
    echo -e "  📊 Grafana:            ${BLUE}http://localhost:3001${NC}"
    echo ""
    echo -e "${YELLOW}Management Commands:${NC}"
    echo -e "  Stop all services:     ${BLUE}docker-compose -f docker-compose-redteam-infrastructure.yml down${NC}"
    echo -e "  View logs:             ${BLUE}docker-compose -f docker-compose-redteam-infrastructure.yml logs -f${NC}"
    echo -e "  Scale services:        ${BLUE}docker-compose -f docker-compose-redteam-infrastructure.yml up --scale xorb-apt-emulator=3${NC}"
    echo ""
    echo -e "${RED}⚠️  SECURITY WARNING:${NC}"
    echo -e "${RED}    This infrastructure contains active adversarial testing tools${NC}"
    echo -e "${RED}    Use only in controlled, isolated environments${NC}"
    echo -e "${RED}    Do not deploy on production networks${NC}"
    echo ""
    echo -e "${PURPLE}XORB Red Team Infrastructure is now operational.${NC}"
    echo -e "${PURPLE}Autonomous adversarial testing capabilities active.${NC}"
}

# Run main deployment
main "$@"
