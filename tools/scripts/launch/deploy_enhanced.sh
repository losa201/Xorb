#!/bin/bash

# XORB Supreme Enhanced Edition - Deployment Script
# Automated deployment for Ubuntu 24.04 LTS systems

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
XORB_DIR="/opt/xorb"
PYTHON_VERSION="3.12"
REDIS_PORT="6379"
QDRANT_PORT="6333"

print_banner() {
    echo -e "${BLUE}"
    cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════╗
║                        XORB SUPREME ENHANCED EDITION                      ║
║                 AI-Augmented Red Team & Bug Bounty Platform              ║
║                                                                          ║
║  🎯 ML-Powered Target Prioritization    🧠 Hybrid LLM Architecture      ║
║  📡 Event-Driven Architecture          🛡️  Production Security Hardening ║
║  🔍 Vector-Enhanced Knowledge Fabric   💰 ROI-Optimized Bounty Engine   ║
║  🤖 Multi-Engine Agent System          📊 Real-Time Resource Optimization ║
║                                                                          ║
║                          Automated Deployment Script                     ║
╚══════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

check_requirements() {
    log "Checking system requirements..."

    # Check OS
    if ! grep -q "Ubuntu" /etc/os-release; then
        warn "This script is optimized for Ubuntu. Proceeding anyway..."
    fi

    # Check Python version
    if ! command -v python3.12 &> /dev/null; then
        warn "Python 3.12 not found. Will attempt to install..."
    fi

    # Check available memory
    MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$MEM_GB" -lt 4 ]; then
        warn "System has less than 4GB RAM. Performance may be limited."
    fi

    # Check disk space
    DISK_GB=$(df / | awk 'NR==2{printf "%.0f", $4/1024/1024}')
    if [ "$DISK_GB" -lt 10 ]; then
        error "Insufficient disk space. At least 10GB free space required."
        exit 1
    fi

    log "Requirements check completed"
}

install_system_dependencies() {
    log "Installing system dependencies..."

    # Update package list
    sudo apt update

    # Install Python 3.12 and essential packages
    sudo apt install -y \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip \
        build-essential \
        libssl-dev \
        libffi-dev \
        redis-server \
        postgresql-client \
        git \
        curl \
        wget \
        unzip \
        htop \
        vim

    # Install Docker (optional but recommended)
    if ! command -v docker &> /dev/null; then
        log "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
    fi

    log "System dependencies installed"
}

setup_services() {
    log "Setting up services..."

    # Configure and start Redis
    sudo systemctl enable redis-server
    sudo systemctl start redis-server

    # Check Redis status
    if systemctl is-active --quiet redis-server; then
        log "Redis server is running on port $REDIS_PORT"
    else
        error "Failed to start Redis server"
        exit 1
    fi

    # Start Qdrant vector database (optional)
    if command -v docker &> /dev/null; then
        log "Starting Qdrant vector database..."
        docker run -d --name qdrant -p $QDRANT_PORT:6333 qdrant/qdrant || true
        if docker ps | grep -q qdrant; then
            log "Qdrant is running on port $QDRANT_PORT"
        else
            warn "Qdrant failed to start. Vector search will be disabled."
        fi
    else
        warn "Docker not available. Qdrant vector database will not be started."
    fi

    log "Services setup completed"
}

create_xorb_user() {
    log "Creating XORB user and directories..."

    # Create xorb user if it doesn't exist
    if ! id "xorb" &>/dev/null; then
        sudo useradd -m -s /bin/bash -d /home/xorb xorb
        sudo usermod -aG docker xorb || true
    fi

    # Create XORB directory
    sudo mkdir -p $XORB_DIR
    sudo chown -R xorb:xorb $XORB_DIR

    # Create log directory
    sudo mkdir -p /var/log/xorb
    sudo chown -R xorb:xorb /var/log/xorb

    # Create config directory
    sudo mkdir -p /etc/xorb
    sudo chown -R xorb:xorb /etc/xorb

    log "XORB user and directories created"
}

install_xorb() {
    log "Installing XORB Supreme Enhanced Edition..."

    # Switch to xorb user for installation
    sudo -u xorb bash << 'EOF'
        cd /opt/xorb

        # Create virtual environment
        python3.12 -m venv venv
        source venv/bin/activate

        # Upgrade pip
        pip install --upgrade pip setuptools wheel

        # Install core dependencies
        pip install -r requirements.txt

        # Install recommended ML dependencies
        echo "Installing ML dependencies (this may take a while)..."
        pip install xgboost scikit-learn pandas numpy || {
            echo "Warning: Failed to install some ML dependencies. ML features may be limited."
        }

        # Install vector search dependencies (optional)
        echo "Installing vector search dependencies..."
        pip install qdrant-client sentence-transformers torch transformers || {
            echo "Warning: Failed to install vector search dependencies. Semantic search will be disabled."
        }

        # Install security dependencies
        pip install bcrypt PyJWT cryptography

        # Install Playwright browsers
        playwright install chromium || {
            echo "Warning: Failed to install Playwright browsers. Web testing may be limited."
        }

        echo "XORB installation completed"
EOF

    log "XORB Supreme installed successfully"
}

create_configuration() {
    log "Creating XORB configuration..."

    # Create default configuration
    sudo -u xorb tee /etc/xorb/config.json > /dev/null << 'EOF'
{
    "redis_url": "redis://localhost:6379/0",
    "database_url": "sqlite+aiosqlite:///var/lib/xorb/xorb_enhanced.db",
    "openrouter_api_key": "your_openrouter_api_key_here",
    "hackerone_api_key": "your_hackerone_api_key_here",
    "security_level": "production",
    "deployment_mode": "production",
    "enable_monitoring": true,
    "enable_ml": true,
    "enable_vector_search": true,
    "enable_bounty_intelligence": true,
    "log_level": "INFO",
    "components": {
        "orchestrator": {"enabled": true, "ml_enabled": true},
        "knowledge_fabric": {"enabled": true, "vector_enabled": true},
        "agent_manager": {"enabled": true, "multi_engine": true},
        "llm_client": {"enabled": true, "hybrid_mode": true},
        "bounty_intelligence": {"enabled": true},
        "security_manager": {"enabled": true},
        "deployment_optimizer": {"enabled": true},
        "event_bus": {"enabled": true},
        "dashboard": {"enabled": true}
    }
}
EOF

    # Create data directory
    sudo mkdir -p /var/lib/xorb
    sudo chown -R xorb:xorb /var/lib/xorb

    # Create systemd service
    sudo tee /etc/systemd/system/xorb.service > /dev/null << EOF
[Unit]
Description=XORB Supreme Enhanced Edition
After=network.target redis.service
Requires=redis.service

[Service]
Type=simple
User=xorb
Group=xorb
WorkingDirectory=$XORB_DIR
Environment=PATH=$XORB_DIR/venv/bin
ExecStart=$XORB_DIR/venv/bin/python main.py --config /etc/xorb/config.json --mode production
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=xorb

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/lib/xorb /var/log/xorb /tmp

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd
    sudo systemctl daemon-reload

    log "Configuration created"
}

setup_security() {
    log "Setting up security configurations..."

    # Configure firewall (if ufw is available)
    if command -v ufw &> /dev/null; then
        sudo ufw allow ssh
        sudo ufw allow 6379/tcp  # Redis (restrict in production)
        sudo ufw allow 6333/tcp  # Qdrant (restrict in production)
        sudo ufw --force enable
        log "Firewall configured"
    fi

    # Set secure permissions
    sudo chmod 600 /etc/xorb/config.json
    sudo chmod 755 $XORB_DIR

    log "Security setup completed"
}

initialize_database() {
    log "Initializing XORB database..."

    sudo -u xorb bash << 'EOF'
        cd /opt/xorb
        source venv/bin/activate

        # Initialize knowledge base
        python -c "
import asyncio
from knowledge_fabric.core import KnowledgeFabric
from knowledge_fabric.vector_fabric import VectorKnowledgeFabric

async def init_db():
    try:
        fabric = VectorKnowledgeFabric()
        await fabric.initialize()
        print('Vector knowledge fabric initialized successfully')
        await fabric.shutdown()
    except Exception as e:
        print(f'Vector fabric failed, falling back to basic fabric: {e}')
        fabric = KnowledgeFabric()
        await fabric.initialize()
        print('Basic knowledge fabric initialized successfully')
        await fabric.shutdown()

asyncio.run(init_db())
" || echo "Database initialization completed with warnings"
EOF

    log "Database initialized"
}

start_xorb() {
    log "Starting XORB Supreme Enhanced Edition..."

    # Enable and start XORB service
    sudo systemctl enable xorb
    sudo systemctl start xorb

    # Wait a moment for startup
    sleep 5

    # Check status
    if systemctl is-active --quiet xorb; then
        log "XORB Supreme is running successfully!"

        # Show service status
        sudo systemctl status xorb --no-pager -l

        # Show logs
        echo -e "\n${BLUE}Recent logs:${NC}"
        sudo journalctl -u xorb --no-pager -l -n 10

    else
        error "XORB failed to start. Check logs with: sudo journalctl -u xorb"
        exit 1
    fi
}

show_summary() {
    echo -e "\n${GREEN}✅ XORB Supreme Enhanced Edition Deployment Complete!${NC}\n"

    echo -e "${BLUE}📋 Deployment Summary:${NC}"
    echo -e "   Installation Directory: $XORB_DIR"
    echo -e "   Configuration File: /etc/xorb/config.json"
    echo -e "   Database Location: /var/lib/xorb/"
    echo -e "   Log Files: /var/log/xorb/"
    echo -e "   Service Status: $(systemctl is-active xorb)"

    echo -e "\n${BLUE}🔧 Next Steps:${NC}"
    echo -e "   1. Edit configuration: sudo nano /etc/xorb/config.json"
    echo -e "   2. Add your API keys (OpenRouter, HackerOne)"
    echo -e "   3. Restart service: sudo systemctl restart xorb"
    echo -e "   4. View logs: sudo journalctl -u xorb -f"
    echo -e "   5. Monitor status: sudo systemctl status xorb"

    echo -e "\n${BLUE}🚀 Management Commands:${NC}"
    echo -e "   Start:   sudo systemctl start xorb"
    echo -e "   Stop:    sudo systemctl stop xorb"
    echo -e "   Restart: sudo systemctl restart xorb"
    echo -e "   Status:  sudo systemctl status xorb"
    echo -e "   Logs:    sudo journalctl -u xorb -f"

    echo -e "\n${BLUE}🌐 Service Ports:${NC}"
    echo -e "   Redis:   localhost:$REDIS_PORT"
    echo -e "   Qdrant:  localhost:$QDRANT_PORT (if Docker available)"

    echo -e "\n${BLUE}📚 Documentation:${NC}"
    echo -e "   README:     $XORB_DIR/README.md"
    echo -e "   Config:     $XORB_DIR/config.example.json"

    echo -e "\n${YELLOW}⚠️  Important Security Notes:${NC}"
    echo -e "   • Update API keys in /etc/xorb/config.json"
    echo -e "   • Restrict Redis/Qdrant access in production"
    echo -e "   • Review firewall rules for your environment"
    echo -e "   • Monitor system resources and performance"

    echo -e "\n${GREEN}🎯 XORB Supreme is ready for cybersecurity operations!${NC}"
}

# Main deployment flow
main() {
    print_banner

    # Check if running as root or with sudo
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root or with sudo"
        exit 1
    fi

    log "Starting XORB Supreme Enhanced Edition deployment..."

    check_requirements
    install_system_dependencies
    setup_services
    create_xorb_user
    install_xorb
    create_configuration
    setup_security
    initialize_database
    start_xorb
    show_summary

    log "Deployment completed successfully!"
}

# Handle script interruption
trap 'error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"
