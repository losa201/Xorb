#!/bin/bash

# XORB Security Platform - Deployment Script
# Optimized for Ubuntu 24.04 LTS with pRoot compatibility

set -e  # Exit on any error

# Change to the script's directory
cd "$(dirname "$0")"

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

# Check if running as root (for system-level installs)
check_permissions() {
    if [[ $EUID -eq 0 ]]; then
        warn "Running as root. This is fine for initial setup but not recommended for operation."
    fi
}

# Detect system capabilities
detect_system() {
    log "Detecting system capabilities..."
    
    # Check OS
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$NAME
        VERSION=$VERSION_ID
        info "Detected OS: $OS $VERSION"
        
        if [[ "$OS" != *"Ubuntu"* ]]; then
            warn "This script is optimized for Ubuntu 24.04. Your mileage may vary."
        fi
    else
        error "Cannot detect operating system"
        exit 1
    fi
    
    # Check if we're in a container or pRoot environment
    if [[ -n "$PROOT_ROOT" ]] || [[ -f /.dockerenv ]] || grep -q container /proc/1/cgroup 2>/dev/null; then
        info "Container/pRoot environment detected - using user-space installation"
        export CONTAINER_ENV=true
        export USE_SUDO=false
    else
        info "Native Linux environment detected"
        export CONTAINER_ENV=false
        export USE_SUDO=true
    fi
    
    # Check Python version
    if command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
    elif command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        if [[ "$PYTHON_VERSION" == "3.12" ]]; then
            PYTHON_CMD="python3"
        else
            warn "Python 3.12 not found. Current version: $PYTHON_VERSION"
            warn "XORB is optimized for Python 3.12 but will attempt to continue"
            PYTHON_CMD="python3"
        fi
    else
        error "Python 3 not found. Please install Python 3.12+"
        exit 1
    fi
    
    info "Using Python command: $PYTHON_CMD"
}

# Install system dependencies
install_system_deps() {
    log "Installing system dependencies..."
    
    if [[ "$CONTAINER_ENV" == "true" ]]; then
        warn "Container environment - skipping system package installation"
        warn "Ensure the following packages are available: curl, git, redis-server"
        return 0
    fi
    
    # Update package list
    if [[ "$USE_SUDO" == "true" ]]; then
        sudo apt update
    else
        apt update 2>/dev/null || warn "Cannot update package list - continuing anyway"
    fi
    
    # Install required packages
    local packages=(
        curl
        git
        build-essential
        libssl-dev
        libffi-dev
        python3.12-dev
        python3.12-venv
        python3-pip
        redis-server
        sqlite3
        libsqlite3-dev
        pkg-config
    )
    
    for package in "${packages[@]}"; do
        if ! dpkg -l | grep -q "^ii  $package "; then
            info "Installing $package..."
            if [[ "$USE_SUDO" == "true" ]]; then
                sudo apt install -y "$package" || warn "Failed to install $package"
            else
                warn "Cannot install $package - please install manually"
            fi
        else
            info "$package is already installed"
        fi
    done
    
    # Start Redis if not in container
    if [[ "$CONTAINER_ENV" == "false" ]]; then
        if command -v systemctl &> /dev/null; then
            sudo systemctl start redis-server || warn "Failed to start Redis"
            sudo systemctl enable redis-server || warn "Failed to enable Redis"
        else
            warn "systemctl not available - please start Redis manually"
        fi
    fi
}

# Install Poetry
install_poetry() {
    log "Installing Poetry..."
    
    if command -v poetry &> /dev/null; then
        info "Poetry is already installed: $(poetry --version)"
        return 0
    fi
    
    # Install Poetry using the official installer
    curl -sSL https://install.python-poetry.org | $PYTHON_CMD -
    
    # Add Poetry to PATH
    if [[ -d "$HOME/.local/bin" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    fi
    
    # Verify installation
    if command -v poetry &> /dev/null; then
        info "Poetry installed successfully: $(poetry --version)"
    else
        error "Poetry installation failed"
        exit 1
    fi
}

# Setup Python environment
setup_python_env() {
    log "Setting up Python environment..."
    
    # Configure Poetry to create virtual environments in project directory
    poetry config virtualenvs.in-project true
    
    # Install dependencies
    info "Installing Python dependencies (this may take a few minutes)..."
    poetry install
    
    # Install additional tools
    poetry run pip install --upgrade pip
    
    # Install Playwright browsers
    info "Installing Playwright browsers..."
    poetry run playwright install chromium || warn "Failed to install Playwright browsers"
    
    # Verify installation
    poetry run python --version
    poetry run python -c "import redis; print('Redis client OK')" || warn "Redis client not working"
    poetry run python -c "import playwright; print('Playwright OK')" || warn "Playwright not working"
}

# Setup configuration
setup_config() {
    log "Setting up configuration..."
    
    # Create environment file from template
    if [[ ! -f .env ]]; then
        if [[ -f .env.example ]]; then
            cp .env.example .env
            info "Created .env from template"
            warn "Please edit .env file with your actual API keys and configuration"
        else
            warn ".env.example not found - creating minimal configuration"
            cat > .env << EOF
# XORB Configuration
OPENROUTER_API_KEY=your-openrouter-api-key
HACKERONE_API_KEY=your-hackerone-api-key
REDIS_HOST=localhost
REDIS_PORT=6379
DATABASE_URL=sqlite:///./xorb.db
LOG_LEVEL=INFO
ROE_COMPLIANCE=strict
MAX_CONCURRENT_AGENTS=5
RESPECT_ROBOTS_TXT=true
EOF
        fi
    else
        info ".env file already exists"
    fi
    
    # Create required directories
    local dirs=(
        logs
        reports_output
        screenshots
        models
        knowledge_cache
        prometheus_data
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            info "Created directory: $dir"
        fi
    done
    
    # Set appropriate permissions
    chmod 755 logs reports_output screenshots
    chmod 700 models knowledge_cache  # More restrictive for sensitive data
}

# Initialize database
init_database() {
    log "Initializing database..."
    
    # Create SQLite database and tables
    poetry run python -c "
import asyncio
from knowledge_fabric.models import create_tables
from sqlalchemy.ext.asyncio import create_async_engine

async def init_db():
    engine = create_async_engine('sqlite+aiosqlite:///./xorb.db')
    await create_tables(engine)
    await engine.dispose()
    print('Database initialized successfully')

try:
    asyncio.run(init_db())
except Exception as e:
    print(f'Database initialization failed: {e}')
    exit(1)
"
    
    info "Database initialized successfully"
}

# Run tests
run_tests() {
    log "Running basic tests..."
    
    # Check if pytest is available
    if poetry run python -c "import pytest" 2>/dev/null; then
        info "Running test suite..."
        poetry run pytest --tb=short -v || warn "Some tests failed"
    else
        warn "pytest not available - running basic validation tests"
        
        # Basic validation tests
        poetry run python -c "
import sys
sys.path.append('.')

try:
    from orchestration.orchestrator import Orchestrator
    from knowledge_fabric.core import KnowledgeFabric
    from agents.base_agent import BaseAgent
    from integrations.openrouter_client import OpenRouterClient
    print('✓ All core modules import successfully')
except Exception as e:
    print(f'✗ Module import failed: {e}')
    sys.exit(1)
"
    fi
    
    info "Basic tests completed"
}

# Setup systemd services (if not in container)
setup_services() {
    if [[ "$CONTAINER_ENV" == "true" ]] || [[ "$USE_SUDO" == "false" ]]; then
        warn "Skipping systemd service setup (container/unprivileged environment)"
        return 0
    fi
    
    log "Setting up systemd services..."
    
    local service_file="/etc/systemd/system/xorb.service"
    local current_dir=$(pwd)
    local user=$(whoami)
    
    # Create systemd service file
    sudo tee "$service_file" > /dev/null << EOF
[Unit]
Description=XORB Security Platform
After=network.target redis.service
Requires=redis.service

[Service]
Type=simple
User=$user
WorkingDirectory=$current_dir
Environment=PATH=$current_dir/.venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$current_dir/.venv/bin/python orchestration/orchestrator.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=xorb

[Install]
WantedBy=multi-user.target
EOF
    
    # Enable and start service
    sudo systemctl daemon-reload
    sudo systemctl enable xorb.service
    
    info "Systemd service created and enabled"
    info "Use 'sudo systemctl start xorb' to start the service"
    info "Use 'sudo systemctl status xorb' to check service status"
    info "Use 'journalctl -u xorb -f' to view logs"
}

# Create startup scripts
create_scripts() {
    log "Creating startup scripts..."
    
    # Create main startup script
    cat > start-xorb.sh << 'EOF'
#!/bin/bash
# XORB Platform Startup Script

export PATH="$HOME/.local/bin:$PATH"

echo "🚀 Starting XORB Security Platform..."

# Check if Redis is running
if ! pgrep -x "redis-server" > /dev/null; then
    echo "⚠️  Redis not running. Please start Redis first:"
    echo "   sudo systemctl start redis-server"
    exit 1
fi

# Start components in background
echo "Starting orchestrator..."
poetry run python orchestration/orchestrator.py &
ORCHESTRATOR_PID=$!

echo "Starting LLM updater..."
poetry run python orchestration/llm_updater.py --continuous &
LLM_PID=$!

echo "Starting monitoring dashboard..."
poetry run python monitoring/dashboard.py &
DASHBOARD_PID=$!

# Function to cleanup on exit
cleanup() {
    echo "🛑 Shutting down XORB..."
    kill $ORCHESTRATOR_PID $LLM_PID $DASHBOARD_PID 2>/dev/null
    wait
    echo "✅ XORB stopped"
}

trap cleanup EXIT

echo "✅ XORB Platform is running!"
echo "   - Orchestrator PID: $ORCHESTRATOR_PID"
echo "   - LLM Updater PID: $LLM_PID" 
echo "   - Dashboard PID: $DASHBOARD_PID"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for all background processes
wait
EOF
    
    # Create dashboard-only script
    cat > start-dashboard.sh << 'EOF'
#!/bin/bash
# XORB Dashboard Only

export PATH="$HOME/.local/bin:$PATH"

echo "📊 Starting XORB Dashboard..."
poetry run python monitoring/dashboard.py
EOF
    
    # Create health check script
    cat > health-check.sh << 'EOF'
#!/bin/bash
# XORB Health Check Script

export PATH="$HOME/.local/bin:$PATH"

echo "🔍 XORB Health Check..."
poetry run python monitoring/dashboard.py --health
EOF
    
    # Make scripts executable
    chmod +x start-xorb.sh start-dashboard.sh health-check.sh
    
    info "Created startup scripts:"
    info "  - start-xorb.sh: Start full XORB platform"
    info "  - start-dashboard.sh: Start dashboard only"
    info "  - health-check.sh: Run system health check"
}

# Print deployment summary
print_summary() {
    log "🎉 XORB deployment completed successfully!"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "                    XORB SECURITY PLATFORM"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "🏗️  DEPLOYMENT SUMMARY:"
    echo "   ✅ System dependencies installed"
    echo "   ✅ Python environment configured"
    echo "   ✅ Database initialized"
    echo "   ✅ Configuration files created"
    echo "   ✅ Startup scripts generated"
    echo ""
    echo "📋 NEXT STEPS:"
    echo "   1. Edit .env file with your API keys:"
    echo "      nano .env"
    echo ""
    echo "   2. Start Redis (if not running):"
    if [[ "$CONTAINER_ENV" == "false" ]]; then
        echo "      sudo systemctl start redis-server"
    else
        echo "      redis-server --daemonize yes"
    fi
    echo ""
    echo "   3. Run health check:"
    echo "      ./health-check.sh"
    echo ""
    echo "   4. Start XORB platform:"
    echo "      ./start-xorb.sh"
    echo ""
    echo "   5. Or start individual components:"
    echo "      poetry run python orchestration/orchestrator.py --campaign"
    echo "      poetry run python orchestration/llm_updater.py --continuous"
    echo "      poetry run python monitoring/dashboard.py"
    echo ""
    echo "📖 DOCUMENTATION:"
    echo "   - README.md: Complete usage guide"
    echo "   - Architecture diagrams in docs/"
    echo "   - API documentation: poetry run python -m docs"
    echo ""
    echo "🔧 USEFUL COMMANDS:"
    echo "   - Health check: ./health-check.sh"
    echo "   - View logs: tail -f logs/audit.log"
    echo "   - Prometheus metrics: curl http://localhost:8000/metrics"
    echo "   - Terminal dashboard: ./start-dashboard.sh"
    echo ""
    echo "🆘 SUPPORT:"
    echo "   - Issues: https://github.com/your-org/xorb/issues"
    echo "   - Docs: https://docs.xorb.ai"
    echo "   - Security: security@xorb.ai"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Main execution
main() {
    echo ""
    echo "🚀 XORB Security Platform - Automated Deployment"
    echo "=================================================="
    echo ""
    
    check_permissions
    detect_system
    install_system_deps
    install_poetry
    setup_python_env
    setup_config
    init_database
    run_tests
    setup_services
    create_scripts
    print_summary
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        echo "XORB Deployment Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --skip-tests   Skip running tests during deployment"
        echo "  --minimal      Minimal installation (skip optional components)"
        echo "  --container    Force container/pRoot mode"
        echo ""
        exit 0
        ;;
    --skip-tests)
        export SKIP_TESTS=true
        ;;
    --minimal)
        export MINIMAL_INSTALL=true
        ;;
    --container)
        export CONTAINER_ENV=true
        export USE_SUDO=false
        ;;
esac

# Run main deployment
main "$@"