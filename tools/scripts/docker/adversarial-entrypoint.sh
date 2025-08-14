#!/bin/bash
set -e

# XORB Adversarial Engine Entrypoint Script
# Handles initialization, health checks, and graceful startup

echo "🔴 Starting XORB Adversarial Engine..."

# Function for logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function for error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Validate environment
log "Validating environment..."

required_vars=(
    "DATABASE_URL"
    "REDIS_URL"
    "ENVIRONMENT"
)

for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        error_exit "Required environment variable $var is not set"
    fi
done

# Wait for dependencies
log "Waiting for dependencies..."

# Wait for PostgreSQL
if command -v pg_isready &> /dev/null; then
    until pg_isready -h "$(echo $DATABASE_URL | sed 's/.*@\([^:]*\).*/\1/')" -p 5432; do
        log "Waiting for PostgreSQL..."
        sleep 2
    done
    log "PostgreSQL is ready"
fi

# Wait for Redis
redis_host=$(echo $REDIS_URL | sed 's/redis:\/\/\([^:]*\).*/\1/')
until timeout 5 bash -c "</dev/tcp/$redis_host/6379"; do
    log "Waiting for Redis..."
    sleep 2
done
log "Redis is ready"

# Initialize application
log "Initializing application..."

# Create necessary directories
mkdir -p /app/{logs,data,models,checkpoints}

# Set proper permissions
chmod 755 /app/{logs,data,models,checkpoints}

# Initialize database schema if needed
if [[ "$ENVIRONMENT" == "development" ]] || [[ "$INIT_DB" == "true" ]]; then
    log "Initializing database schema..."
    python -c "
from adversarial.database import init_db
import asyncio
asyncio.run(init_db())
" || log "Database initialization failed (might already exist)"
fi

# Warm up models
log "Warming up adversarial models..."
python -c "
from adversarial import initialize_enhanced_adversarial_systems
import asyncio
asyncio.run(initialize_enhanced_adversarial_systems())
" || log "Model warmup failed"

# Health check endpoint setup
log "Setting up health monitoring..."
cat > /app/health_server.py << 'EOF'
import asyncio
import json
from datetime import datetime
from aiohttp import web

async def health_check(request):
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "xorb-adversarial-engine",
        "version": "1.0.0"
    }
    return web.json_response(health_status)

async def ready_check(request):
    # Add actual readiness checks here
    return web.json_response({"status": "ready"})

async def create_health_app():
    app = web.Application()
    app.router.add_get('/health', health_check)
    app.router.add_get('/ready', ready_check)
    return app

async def run_health_server():
    app = await create_health_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8002)
    await site.start()
    print("Health server running on port 8002")

if __name__ == "__main__":
    asyncio.run(run_health_server())
EOF

# Start health server in background
python /app/health_server.py &
HEALTH_PID=$!

# Function to handle shutdown
cleanup() {
    log "Shutting down gracefully..."
    if [[ -n "$HEALTH_PID" ]]; then
        kill $HEALTH_PID 2>/dev/null || true
    fi
    if [[ -n "$MAIN_PID" ]]; then
        kill $MAIN_PID 2>/dev/null || true
        wait $MAIN_PID 2>/dev/null || true
    fi
    log "Shutdown complete"
    exit 0
}

# Set trap for graceful shutdown
trap cleanup SIGTERM SIGINT

# Performance optimizations
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

# Start main application
log "Starting main application..."
log "Configuration:"
log "  Environment: $ENVIRONMENT"
log "  Adversarial Mode: ${ADVERSARIAL_MODE:-standard}"
log "  ML Defense: ${ML_DEFENSE_ENABLED:-false}"
log "  Dynamic Adaptation: ${DYNAMIC_ADAPTATION:-false}"

# Execute the main command
exec "$@" &
MAIN_PID=$!

# Wait for main process
wait $MAIN_PID
