#!/bin/bash

# Post-create script for Xorb 2.0 development container
# This script runs after the container is created

set -e

echo "🚀 Running Xorb 2.0 post-create setup..."

# Install Python dependencies
if [ -f "pyproject.toml" ]; then
    echo "📦 Installing Python dependencies with Poetry..."
    poetry install --with dev
elif [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies with pip..."
    pip install -r requirements.txt
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
    fi
fi

# Set up Git hooks if pre-commit config exists
if [ -f ".pre-commit-config.yaml" ]; then
    echo "🔧 Setting up pre-commit hooks..."
    pre-commit install
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p ~/.config/xorb
mkdir -p ~/.local/share/xorb
mkdir -p ~/workspace/.vscode
mkdir -p ~/workspace/logs

# Set up shell aliases and functions
echo "🐚 Setting up shell aliases..."
cat << 'EOF' >> ~/.bashrc

# Xorb development aliases
alias xorb-api="cd /workspace && python -m services.api.app.main"
alias xorb-worker="cd /workspace && python -m services.worker.app.run_worker"
alias xorb-orchestrator="cd /workspace && python -m services.orchestrator.main"
alias xorb-test="cd /workspace && python -m pytest tests/ -v"
alias xorb-lint="cd /workspace && flake8 . && mypy ."
alias xorb-format="cd /workspace && black . && isort ."

# Kubernetes aliases
alias k="kubectl"
alias kgp="kubectl get pods"
alias kgs="kubectl get services"
alias kgd="kubectl get deployments"
alias kdesc="kubectl describe"
alias klogs="kubectl logs"

# Docker aliases
alias dc="docker-compose"
alias dcup="docker-compose up"
alias dcdown="docker-compose down"
alias dcbuild="docker-compose build"

# Git aliases
alias gst="git status"
alias gco="git checkout"
alias gcb="git checkout -b"
alias gaa="git add ."
alias gcm="git commit -m"
alias gp="git push"
alias gl="git log --oneline -10"

# Utility functions
function xorb-logs() {
    if [ -z "$1" ]; then
        echo "Usage: xorb-logs <service>"
        echo "Available services: api, worker, orchestrator"
        return 1
    fi
    
    case $1 in
        api)
            kubectl logs -f deployment/xorb-api -n xorb-dev
            ;;
        worker)
            kubectl logs -f deployment/xorb-worker -n xorb-dev
            ;;
        orchestrator)
            kubectl logs -f deployment/xorb-orchestrator -n xorb-dev
            ;;
        *)
            echo "Unknown service: $1"
            ;;
    esac
}

function xorb-port-forward() {
    echo "Setting up port forwards for Xorb services..."
    kubectl port-forward -n xorb-dev svc/xorb-api 8000:8000 &
    kubectl port-forward -n xorb-infra-dev svc/temporal-frontend 7233:7233 &
    kubectl port-forward -n xorb-infra-dev svc/temporal-web 8233:8233 &
    kubectl port-forward -n xorb-infra-dev svc/postgresql 5432:5432 &
    kubectl port-forward -n xorb-infra-dev svc/redis-master 6379:6379 &
    echo "Port forwards started in background"
}

function xorb-status() {
    echo "📊 Xorb Development Environment Status"
    echo "======================================="
    
    echo "🐳 Docker services:"
    docker-compose ps 2>/dev/null || echo "  No local Docker services running"
    
    echo ""
    echo "☸️  Kubernetes services (if available):"
    kubectl get pods -n xorb-dev 2>/dev/null || echo "  Kubernetes not available or no xorb-dev namespace"
    
    echo ""
    echo "🔗 Available endpoints:"
    echo "  API: http://localhost:8000"
    echo "  Temporal Web: http://localhost:8233"
    echo "  Neo4j Browser: http://localhost:7474"
    echo "  Grafana: http://localhost:3001"
}

EOF

# Set up zsh if it's available
if [ -f ~/.zshrc ]; then
    echo "🐚 Setting up zsh configuration..."
    cat << 'EOF' >> ~/.zshrc

# Xorb development aliases (same as bash)
source ~/.bashrc

EOF
fi

# Create sample configuration files
echo "⚙️  Creating sample configuration files..."

# Create .env.development
cat << 'EOF' > ~/.config/xorb/.env.development
# Xorb 2.0 Development Configuration
XORB_ENVIRONMENT=development
XORB_LOG_LEVEL=DEBUG
XORB_DEBUG=true

# Database
DATABASE_URL=postgresql://xorb_dev:xorb_dev_password@localhost:5432/xorb_dev

# Redis
REDIS_URL=redis://localhost:6379

# Temporal
TEMPORAL_HOST=localhost:7233

# NATS
NATS_URL=nats://localhost:4222

# Qdrant
QDRANT_URL=http://localhost:6333

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=xorb_dev_password

# API Keys (replace with your actual keys)
# OPENROUTER_API_KEY=your_openrouter_key_here
# HACKERONE_API_KEY=your_hackerone_key_here

# Development settings
MAX_CONCURRENT_AGENTS=8
MAX_CONCURRENT_CAMPAIGNS=3
WORKER_CONCURRENCY=4
AGENT_POOL_SIZE=8
EOF

# Create VSCode workspace settings
echo "🔧 Creating VSCode workspace settings..."
cat << 'EOF' > /workspace/.vscode/settings.json
{
    "python.pythonPath": "/usr/local/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.provider": "isort",
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/venv": true,
        "**/.venv": true,
        "**/node_modules": true,
        "**/.git": true
    },
    "yaml.schemas": {
        "https://json.schemastore.org/kustomization": "kustomization.yaml",
        "https://json.schemastore.org/chart": "Chart.yaml",
        "kubernetes": "*.k8s.yaml"
    }
}
EOF

# Install VSCode extensions if code-server is available
if command -v code &> /dev/null; then
    echo "🔌 Installing VSCode extensions..."
    code --install-extension ms-python.python
    code --install-extension ms-kubernetes-tools.vscode-kubernetes-tools
    code --install-extension redhat.vscode-yaml
    code --install-extension ms-vscode.vscode-docker
fi

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
timeout=60
while [ $timeout -gt 0 ]; do
    if nc -z localhost 5432 2>/dev/null; then
        echo "✅ PostgreSQL is ready"
        break
    fi
    echo "⏳ Waiting for PostgreSQL... ($timeout seconds remaining)"
    sleep 2
    timeout=$((timeout-2))
done

if [ $timeout -le 0 ]; then
    echo "⚠️  PostgreSQL not ready within timeout, continuing anyway..."
fi

# Test database connection
echo "🔍 Testing database connection..."
if command -v psql &> /dev/null; then
    if PGPASSWORD=xorb_dev_password psql -h localhost -U xorb_dev -d xorb_dev -c "SELECT 1;" &> /dev/null; then
        echo "✅ Database connection successful"
    else
        echo "⚠️  Database connection failed"
    fi
fi

echo "✅ Post-create setup complete!"
echo ""
echo "🎉 Welcome to Xorb 2.0 Development Environment!"
echo ""
echo "Quick start commands:"
echo "  xorb-status    - Show environment status"
echo "  xorb-test      - Run tests"
echo "  xorb-format    - Format code"
echo "  xorb-api       - Start API server"
echo "  xorb-worker    - Start worker"
echo ""
echo "Configuration files:"
echo "  ~/.config/xorb/.env.development - Environment variables"
echo "  /workspace/.vscode/settings.json - VSCode settings"
echo ""
echo "Happy coding! 🚀"