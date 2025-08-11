#!/bin/bash

# XORB API Service Production Entrypoint
# Handles configuration, health checks, and graceful startup

set -euo pipefail

# Color output for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >&2
}

log_info() {
    log "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    log "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        log "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Configuration validation using centralized config manager
validate_config() {
    log_info "Validating configuration using XORB config manager..."
    
    python3 -c "
import sys
sys.path.append('/app/src')
try:
    from common.config_manager import get_config
    config = get_config()
    print('✅ Configuration validation passed')
    print(f'Environment: {config.environment.value}')
    print(f'API Port: {config.api_service.port}')
    print(f'Debug Mode: {config.debug}')
    print(f'Database: {config.database.host}:{config.database.port}/{config.database.name}')
    print(f'Redis: {config.redis.host}:{config.redis.port}')
except Exception as e:
    print(f'❌ Configuration validation failed: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        log_info "Configuration validation completed successfully"
    else
        log_error "Configuration validation failed"
        exit 1
    fi
}

# Health check dependencies using config manager
check_dependencies() {
    log_info "Checking service dependencies..."
    
    # Check PostgreSQL and Redis connectivity using config manager
    python3 -c "
import asyncio
import sys
sys.path.append('/app/src')

async def check_postgres():
    try:
        from common.config_manager import get_config
        config = get_config()
        db_url = config.database.get_url()
        
        import asyncpg
        conn = await asyncpg.connect(db_url)
        await conn.execute('SELECT 1')
        await conn.close()
        print('✅ PostgreSQL connection successful')
        return True
    except Exception as e:
        print(f'❌ PostgreSQL connection failed: {e}')
        return False

async def check_redis():
    try:
        from common.config_manager import get_config
        config = get_config()
        
        import aioredis
        redis = aioredis.from_url(config.redis.get_url())
        await redis.ping()
        await redis.close()
        print('✅ Redis connection successful')
        return True
    except Exception as e:
        print(f'❌ Redis connection failed: {e}')
        return False

async def main():
    postgres_ok = await check_postgres()
    redis_ok = await check_redis()
    return postgres_ok and redis_ok

result = asyncio.run(main())
sys.exit(0 if result else 1)
"
    
    if [ $? -eq 0 ]; then
        log_info "All dependency checks passed"
        return 0
    else
        log_warn "Some dependencies are not available - will continue with degraded mode"
        return 1
    fi
}

# Function to run database migrations
run_migrations() {
    if [[ "${SKIP_MIGRATIONS:-false}" != "true" ]]; then
        log "Running database migrations..."
        if alembic upgrade head; then
            log "Database migrations completed successfully"
        else
            log "ERROR: Database migration failed"
            exit 1
        fi
    else
        log "Skipping database migrations (SKIP_MIGRATIONS=true)"
    fi
}

# Function to validate configuration
validate_config() {
    log "Validating configuration..."
    
    # Check Python modules can be imported
    python -c "
import sys
try:
    from app.main import app
    from app.infrastructure.database import check_database_connection
    print('✓ Application modules imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"
    
    # Check file permissions
    if [[ ! -r /app/app/main.py ]]; then
        log "ERROR: Cannot read application files"
        exit 1
    fi
    
    log "Configuration validation passed"
}

# Function to setup ClamAV (if enabled)
setup_clamav() {
    if [[ "${ENABLE_CLAMAV:-false}" == "true" ]]; then
        log "Setting up ClamAV..."
        
        # Update virus definitions if needed
        if [[ ! -f /var/lib/clamav/daily.cvd ]] || [[ $(find /var/lib/clamav/daily.cvd -mtime +1) ]]; then
            log "Updating ClamAV virus definitions..."
            freshclam --quiet || log "WARNING: Could not update virus definitions"
        fi
        
        # Start ClamAV daemon
        log "Starting ClamAV daemon..."
        clamd || log "WARNING: Could not start ClamAV daemon"
    fi
}

# Function to setup directories and permissions
setup_directories() {
    log "Setting up directories..."
    
    # Create necessary directories
    mkdir -p /app/logs /app/data /app/tmp
    
    # Ensure proper ownership (should already be set in Dockerfile)
    if [[ "$(id -u)" == "0" ]]; then
        chown -R xorb:xorb /app/logs /app/data /app/tmp
    fi
}

# Function to cleanup on exit
cleanup() {
    log "Cleaning up..."
    # Add any cleanup tasks here
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Graceful shutdown handler
shutdown_handler() {
    log_info "Received shutdown signal - performing graceful shutdown..."
    
    # Send SIGTERM to application process
    if [ ! -z "$APP_PID" ]; then
        kill -TERM "$APP_PID" 2>/dev/null || true
        wait "$APP_PID" 2>/dev/null || true
    fi
    
    log_info "Graceful shutdown completed"
    exit 0
}

# Main execution
main() {
    # Set up signal handlers
    trap shutdown_handler SIGTERM SIGINT
    
    # Validate environment
    if [ -z "${ENVIRONMENT:-}" ]; then
        export ENVIRONMENT="production"
    fi
    
    log_info "Starting XORB API Service in $ENVIRONMENT environment"
    log_info "User: $(id)"
    log_info "Working directory: $(pwd)"
    log_info "Python version: $(python3 --version)"
    
    # Pre-startup validation
    validate_config
    check_dependencies
    
    # Run migrations if not skipped
    if [[ "${SKIP_MIGRATIONS:-false}" != "true" ]]; then
        log_info "Running database migrations..."
        if [ -f "alembic.ini" ]; then
            python3 -m alembic upgrade head
            log_info "Database migrations completed"
        else
            log_warn "No alembic.ini found - skipping migrations"
        fi
    else
        log_info "Skipping migrations (SKIP_MIGRATIONS=true)"
    fi
    
    # Determine startup command based on environment and provided arguments
    if [ $# -eq 0 ]; then
        # No arguments provided, use default production command
        if [ "$ENVIRONMENT" = "development" ]; then
            log_info "Starting development server with hot-reload"
            set -- uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --reload --log-level debug
        else
            log_info "Starting production server with Gunicorn"
            
            # Calculate worker processes
            WORKERS=${WORKERS:-$(($(nproc) * 2 + 1))}
            MAX_WORKERS=${MAX_WORKERS:-16}
            
            if [ "$WORKERS" -gt "$MAX_WORKERS" ]; then
                WORKERS=$MAX_WORKERS
            fi
            
            log_info "Starting with $WORKERS worker processes"
            
            set -- gunicorn app.main:app \
                -k uvicorn.workers.UvicornWorker \
                -w "$WORKERS" \
                -b 0.0.0.0:${PORT:-8000} \
                --timeout ${TIMEOUT:-30} \
                --keepalive ${KEEPALIVE:-5} \
                --max-requests ${MAX_REQUESTS:-1000} \
                --max-requests-jitter ${MAX_REQUESTS_JITTER:-100} \
                --preload \
                --log-level info \
                --access-logfile - \
                --error-logfile -
        fi
    fi
    
    log_info "Starting application with command: $*"
    log_info "Health endpoint will be available at http://localhost:${PORT:-8000}/health"
    
    # Execute the main command in background to handle signals
    "$@" &
    APP_PID=$!
    
    # Wait for the application process
    wait "$APP_PID"
}

# Run main function with all arguments
main "$@"