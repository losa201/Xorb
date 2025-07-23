#!/bin/bash

# Post-start script for Xorb 2.0 development container
# This script runs every time the container starts

set -e

echo "🔄 Running Xorb 2.0 post-start setup..."

# Check if required services are running
echo "🔍 Checking service availability..."

# Function to check if a service is responding
check_service() {
    local service_name=$1
    local host=$2
    local port=$3
    local max_attempts=${4:-30}
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z $host $port 2>/dev/null; then
            echo "✅ $service_name is ready"
            return 0
        fi
        
        if [ $attempt -eq 1 ]; then
            echo "⏳ Waiting for $service_name ($host:$port)..."
        fi
        
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "⚠️  $service_name is not responding after $max_attempts attempts"
    return 1
}

# Check core services
check_service "PostgreSQL" "localhost" "5432" 15
check_service "Redis" "localhost" "6379" 15
check_service "NATS" "localhost" "4222" 15

# Check if Temporal is available (may take longer to start)
if check_service "Temporal" "localhost" "7233" 30; then
    echo "✅ All core services are ready"
else
    echo "⚠️  Some services may not be fully ready"
fi

# Update Python path
export PYTHONPATH="/workspace:$PYTHONPATH"

# Source development environment
if [ -f ~/.config/xorb/.env.development ]; then
    echo "📋 Loading development environment variables..."
    set -a
    source ~/.config/xorb/.env.development
    set +a
fi

# Initialize database if needed
echo "🗄️  Checking database initialization..."
if command -v psql &> /dev/null; then
    if PGPASSWORD=xorb_dev_password psql -h localhost -U xorb_dev -d xorb_dev -c "SELECT 1;" &> /dev/null; then
        echo "✅ Database connection verified"
        
        # Check if tables exist, if not, run migrations
        table_count=$(PGPASSWORD=xorb_dev_password psql -h localhost -U xorb_dev -d xorb_dev -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null | xargs)
        
        if [ "$table_count" -eq "0" ]; then
            echo "🔧 Database appears empty, checking for migration scripts..."
            if [ -f "/workspace/migrations/init.sql" ]; then
                echo "📄 Running database initialization..."
                PGPASSWORD=xorb_dev_password psql -h localhost -U xorb_dev -d xorb_dev -f /workspace/migrations/init.sql
                echo "✅ Database initialized"
            else
                echo "ℹ️  No migration scripts found, database will be initialized by application"
            fi
        else
            echo "✅ Database already initialized ($table_count tables found)"
        fi
    else
        echo "⚠️  Could not connect to database"
    fi
fi

# Check for and run any pending migrations
if [ -f "/workspace/manage.py" ]; then
    echo "🔄 Checking for Django migrations..."
    cd /workspace && python manage.py migrate --check &> /dev/null || {
        echo "🔧 Running Django migrations..."
        python manage.py migrate
    }
elif [ -f "/workspace/alembic.ini" ]; then
    echo "🔄 Checking for Alembic migrations..."
    cd /workspace && alembic current &> /dev/null || {
        echo "🔧 Running Alembic migrations..."
        alembic upgrade head
    }
fi

# Start background services if in development mode
if [ "$XORB_ENVIRONMENT" = "development" ]; then
    echo "🔧 Starting development background services..."
    
    # Start file watcher for auto-reload (if available)
    if command -v watchdog &> /dev/null; then
        echo "👀 Starting file watcher..."
        nohup watchdog /workspace --patterns="*.py" --command="echo 'File changed: {}'" > /dev/null 2>&1 &
    fi
    
    # Start local metrics collection (if Prometheus is configured)
    if [ -f "/workspace/prometheus.yml" ]; then
        echo "📊 Starting local Prometheus..."
        nohup prometheus --config.file=/workspace/prometheus.yml --storage.tsdb.path=/tmp/prometheus > /dev/null 2>&1 &
    fi
fi

# Display useful information
echo ""
echo "🎯 Development Environment Ready!"
echo "================================="
echo ""
echo "📍 Current directory: $(pwd)"
echo "🐍 Python version: $(python --version)"
echo "📦 Poetry version: $(poetry --version 2>/dev/null || echo 'Not available')"
echo "☸️  Kubectl version: $(kubectl version --client --short 2>/dev/null || echo 'Not available')"
echo "⎈  Helm version: $(helm version --short 2>/dev/null || echo 'Not available')"
echo ""

echo "🌐 Available endpoints:"
echo "  📡 API Server: http://localhost:8000"
echo "  ⏰ Temporal Web: http://localhost:8233"
echo "  🧠 Neo4j Browser: http://localhost:7474"
echo "  📊 Grafana: http://localhost:3001"
echo ""

echo "🔧 Quick commands:"
echo "  xorb-status     - Show detailed status"
echo "  xorb-test       - Run tests"
echo "  xorb-api        - Start API server"
echo "  xorb-worker     - Start worker"
echo "  dc up           - Start all services"
echo "  k get pods      - List Kubernetes pods"
echo ""

# Show git status if we're in a git repository
if [ -d ".git" ]; then
    echo "🔀 Git status:"
    git status --porcelain | head -5
    if [ $(git status --porcelain | wc -l) -gt 5 ]; then
        echo "   ... and $(( $(git status --porcelain | wc -l) - 5 )) more files"
    fi
    echo ""
fi

# Show recent logs if available
if [ -d "/workspace/logs" ] && [ "$(ls -A /workspace/logs 2>/dev/null)" ]; then
    echo "📝 Recent logs:"
    ls -la /workspace/logs/ | tail -3
    echo ""
fi

# Final setup complete message
echo "✅ Post-start setup complete - Happy coding! 🚀"

# Change to workspace directory
cd /workspace