# Justfile for Xorb PTaaS Development
# Simplified command runner for common development tasks

# Default recipe to display available commands
default:
    @just --list

# Setup development environment
setup:
    #!/usr/bin/env bash
    echo "🚀 Setting up Xorb PTaaS development environment..."
    
    # Install Poetry if not present  
    if ! command -v poetry &> /dev/null; then
        curl -sSL https://install.python-poetry.org | python3 -
    fi
    
    # Install dependencies
    poetry install --with dev
    
    # Install pre-commit hooks
    poetry run pre-commit install --install-hooks
    
    # Generate environment file
    if [ ! -f compose/.env ]; then
        cp compose/.env.example compose/.env
        echo "📝 Please edit compose/.env with your configuration"
    fi
    
    # Generate TLS certificates
    cd compose/security && ./generate-certs.sh
    
    echo "✅ Development environment setup complete!"

# Start all services
up:
    #!/usr/bin/env bash
    echo "🐳 Starting Xorb PTaaS services..."
    cd compose && docker-compose up -d
    echo "⏳ Waiting for services to be ready..."
    timeout 300 bash -c 'until curl -f http://localhost:8000/health; do sleep 5; done'
    echo "✅ All services are running!"
    echo "🌐 API: http://localhost:8000"
    echo "🔧 Orchestrator: http://localhost:8001"
    echo "🧑‍💻 Researcher Portal: http://localhost:3000"
    echo "📊 Grafana: http://localhost:3001 (admin/admin)"
    echo "🔍 Prometheus: http://localhost:9090"

# Start services with observability stack
up-full:
    #!/usr/bin/env bash
    echo "🚀 Starting full Xorb PTaaS stack with observability..."
    cd compose
    docker-compose -f docker-compose.yml -f docker-compose.observability.yml up -d
    echo "⏳ Waiting for services to be ready..."
    timeout 300 bash -c 'until curl -f http://localhost:8000/health; do sleep 5; done'
    echo "✅ Full stack is running!"

# Stop all services
down:
    cd compose && docker-compose down

# Stop all services and remove volumes
down-clean:
    cd compose && docker-compose down -v --remove-orphans

# Restart services
restart:
    just down && just up

# View service logs
logs service="":
    #!/usr/bin/env bash
    cd compose
    if [ -n "{{service}}" ]; then
        docker-compose logs -f {{service}}
    else
        docker-compose logs -f
    fi

# Open shell in a service container
shell service="api":
    cd compose && docker-compose exec {{service}} /bin/bash

# Run tests
test:
    poetry run pytest -v

# Run tests with coverage
test-cov:
    poetry run pytest --cov=xorb_common --cov=services --cov-report=html --cov-report=term

# Run linting
lint:
    poetry run ruff check .
    poetry run mypy xorb_common/ services/

# Format code
format:
    poetry run ruff format .
    poetry run isort xorb_common/ services/

# Run security scan
security:
    poetry run bandit -r xorb_common/ services/ -f json

# Run pre-commit on all files  
pre-commit:
    poetry run pre-commit run --all-files

# Build Docker images
build service="":
    #!/usr/bin/env bash
    cd compose
    if [ -n "{{service}}" ]; then
        docker-compose build {{service}}
    else
        docker-compose build
    fi

# Pull latest images
pull:
    cd compose && docker-compose pull

# Clean Docker resources
clean:
    docker system prune -f
    docker volume prune -f

# Database operations
db-migrate:
    cd compose && docker-compose exec api alembic upgrade head

db-reset:
    cd compose && docker-compose exec api alembic downgrade base && docker-compose exec api alembic upgrade head

db-shell:
    cd compose && docker-compose exec postgres psql -U xorb -d xorb_ptaas

# Run load tests
load-test:
    #!/usr/bin/env bash
    if ! command -v k6 &> /dev/null; then
        echo "❌ k6 not found, please install it first"
        exit 1
    fi
    k6 run scripts/load-test.js

# Performance profiling
profile service="api":
    #!/usr/bin/env bash
    echo "🔬 Starting profiling for {{service}} service..."
    echo "📊 Visit http://localhost:4040 to view Pyroscope dashboard"
    cd compose && docker-compose -f docker-compose.yml -f docker-compose.observability.yml up -d pyroscope

# Check service health
health:
    #!/usr/bin/env bash
    echo "🏥 Checking service health..."
    services=("api:8000" "orchestrator:8001" "payments:8002" "researcher-portal:3000")
    for service in "${services[@]}"; do
        name=$(echo $service | cut -d: -f1)
        port=$(echo $service | cut -d: -f2)
        if curl -f -s http://localhost:$port/health > /dev/null; then
            echo "✅ $name is healthy"
        else
            echo "❌ $name is unhealthy"
        fi
    done

# Monitor system resources
monitor:
    #!/usr/bin/env bash
    echo "📊 System Resource Monitoring"
    echo "EPYC CPU Usage:"
    top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"% used"}'
    echo ""
    echo "Memory Usage:"
    free -h
    echo ""
    echo "Docker Container Status:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Generate development SSL certificates
certs:
    cd compose/security && ./generate-certs.sh

# Update dependencies
update-deps:
    poetry update
    poetry run pre-commit autoupdate

# Create database backup
backup:
    cd compose && docker-compose -f docker-compose.backups.yml run --rm postgres-backup

# Restore from backup
restore backup-file:
    echo "🔄 Restoring from backup: {{backup-file}}"
    # Implementation depends on backup format

# Deploy to production
deploy:
    #!/usr/bin/env bash
    echo "🚀 Deploying to production..."
    if [ ! -f deploy.env ]; then
        echo "❌ deploy.env file not found"
        exit 1
    fi
    source deploy.env
    # Run deployment script
    ./scripts/deploy.sh

# Run EPYC optimization checks
epyc-tune:
    #!/usr/bin/env bash
    echo "⚡ Applying EPYC optimizations..."
    if [ -f compose/epyc-tuning.conf ]; then
        sudo cp compose/epyc-tuning.conf /etc/sysctl.d/99-xorb.conf
        sudo sysctl --system
        echo "✅ EPYC optimizations applied"
    else
        echo "❌ EPYC tuning config not found"
    fi

# Show system information
info:
    #!/usr/bin/env bash
    echo "🔍 Xorb PTaaS System Information"
    echo "================================="
    echo "OS: $(uname -s -r)"
    echo "CPU: $(cat /proc/cpuinfo | grep 'model name' | head -1 | cut -d: -f2 | xargs)"
    echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
    echo "Docker: $(docker --version)"
    echo "Python: $(python3 --version)"
    echo "Poetry: $(poetry --version)"
    echo ""
    echo "Services Status:"
    cd compose && docker-compose ps

# Troubleshooting helpers
troubleshoot:
    #!/usr/bin/env bash
    echo "🔧 Xorb PTaaS Troubleshooting"
    echo "=============================="
    echo ""
    echo "1. Check Docker daemon:"
    systemctl is-active docker
    echo ""
    echo "2. Check disk space:"
    df -h
    echo ""
    echo "3. Check service logs for errors:"
    cd compose && docker-compose logs --tail=50 | grep -i error
    echo ""
    echo "4. Check container resource usage:"
    docker stats --no-stream

# Generate project documentation
docs:
    #!/usr/bin/env bash
    echo "📚 Generating project documentation..."
    if command -v mkdocs &> /dev/null; then
        mkdocs build
        echo "✅ Documentation generated in site/"
    else
        echo "❌ mkdocs not found, install with: pip install mkdocs mkdocs-material"
    fi

# Run integration tests
test-integration:
    #!/usr/bin/env bash
    echo "🧪 Running integration tests..."
    just up
    sleep 30  # Wait for services to be ready
    poetry run pytest tests/integration/ -v
    just down

# Quick health check and restart if needed
heal:
    #!/usr/bin/env bash
    echo "🩺 Running health check and auto-healing..."
    if ! curl -f -s http://localhost:8000/health > /dev/null; then
        echo "⚕️ API unhealthy, restarting..."
        cd compose && docker-compose restart api
        sleep 10
    fi
    
    if ! curl -f -s http://localhost:8001/health > /dev/null; then
        echo "⚕️ Orchestrator unhealthy, restarting..."
        cd compose && docker-compose restart orchestrator
        sleep 10
    fi
    
    echo "✅ Health check complete"