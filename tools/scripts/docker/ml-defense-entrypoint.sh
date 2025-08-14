#!/bin/bash
set -e

# XORB ML Defense Engine Entrypoint Script
# Handles ML model initialization, GPU detection, and service startup

echo "🛡️ Starting XORB ML Defense Engine..."

# Function for logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function for error handling
error_exit() {
    log "ERROR: $1"
    exit 1
}

# Configure CPU optimization for AMD EPYC
configure_cpu() {
    log "Configuring CPU optimization for AMD EPYC 7002 (16 cores, 32GB RAM)"
    export USE_GPU=false
    export OMP_NUM_THREADS=16
    export MKL_NUM_THREADS=16
    export OPENBLAS_NUM_THREADS=16
    export BLIS_NUM_THREADS=16
    export TORCH_NUM_THREADS=16
    export NUMEXPR_NUM_THREADS=16
    log "CPU threads configured for maximum EPYC performance"
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

# Hardware configuration

# Memory optimization for AMD EPYC 7002 (32GB RAM)
setup_memory_optimization() {
    local total_mem=$(free -m | awk 'NR==2{print $2}')
    local available_mem=$(free -m | awk 'NR==2{print $7}')

    log "AMD EPYC System memory: ${total_mem}MB total, ${available_mem}MB available"
    log "Optimizing for 32GB RAM and 16 CPU cores"

    # Already configured in configure_cpu(), but ensure consistency
    export TORCH_MEMORY_FRACTION=0.8
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export MALLOC_TRIM_THRESHOLD_=65536
    log "Memory optimizations applied for high-memory EPYC deployment"
}

configure_cpu
setup_memory_optimization

# Wait for dependencies
log "Waiting for dependencies..."

# Wait for PostgreSQL
redis_host=$(echo $REDIS_URL | sed 's/redis:\/\/\([^:]*\).*/\1/')
until timeout 5 bash -c "</dev/tcp/$redis_host/6379"; do
    log "Waiting for Redis..."
    sleep 2
done
log "Redis is ready"

# Initialize ML environment
log "Initializing ML environment..."

# Create necessary directories
mkdir -p /app/{logs,models,data,checkpoints,cache}
chmod 755 /app/{logs,models,data,checkpoints,cache}

# Set up model cache
export TRANSFORMERS_CACHE=/app/cache/transformers
export HF_HOME=/app/cache/huggingface
mkdir -p $TRANSFORMERS_CACHE $HF_HOME

# Validate ML dependencies
log "Validating ML dependencies..."
python -c "
import torch
import sklearn
import numpy as np
import pandas as pd

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'Current CUDA device: {torch.cuda.current_device()}')
print(f'Scikit-learn version: {sklearn.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'Pandas version: {pd.__version__}')
" || error_exit "ML dependencies validation failed"

# Initialize ML models
if [[ "$ENVIRONMENT" != "test" ]]; then
    log "Initializing ML models..."
    python -c "
from ml_defense.ml_defense_engine import initialize_ml_defense
import asyncio
try:
    asyncio.run(initialize_ml_defense())
    print('ML Defense Engine initialized successfully')
except Exception as e:
    print(f'ML initialization warning: {e}')
    # Don't exit - models can be lazy loaded
" || log "ML model initialization completed with warnings"
fi

# Model warmup and validation
log "Performing model warmup..."
python -c "
import torch
import numpy as np

# Test tensor operations
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.randn(100, 100).to(device)
    y = torch.mm(x, x.t())
    print('GPU warmup completed')
else:
    x = torch.randn(100, 100)
    y = torch.mm(x, x.t())
    print('CPU warmup completed')

# Test numpy operations
a = np.random.rand(1000, 1000)
b = np.dot(a, a.T)
print('NumPy operations validated')
" || log "Model warmup completed with warnings"

# Health check endpoint setup
log "Setting up health monitoring..."
cat > /app/ml_health_server.py << 'EOF'
import asyncio
import json
import torch
from datetime import datetime
from aiohttp import web

async def health_check(request):
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "xorb-ml-defense-engine",
        "version": "1.0.0",
        "gpu_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    return web.json_response(health_status)

async def ready_check(request):
    try:
        # Basic ML readiness check
        import sklearn
        import numpy as np

        # Test basic operations
        test_array = np.random.rand(10, 10)
        test_result = np.dot(test_array, test_array.T)

        return web.json_response({
            "status": "ready",
            "ml_frameworks": "operational"
        })
    except Exception as e:
        return web.json_response({
            "status": "not_ready",
            "error": str(e)
        }, status=503)

async def models_status(request):
    try:
        # Check model loading status
        status = {
            "models_loaded": True,  # Update based on actual model status
            "model_count": 0,       # Update with actual count
            "memory_usage": "N/A"   # Add memory monitoring
        }
        return web.json_response(status)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def create_health_app():
    app = web.Application()
    app.router.add_get('/health', health_check)
    app.router.add_get('/ready', ready_check)
    app.router.add_get('/models', models_status)
    return app

async def run_health_server():
    app = await create_health_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8004)
    await site.start()
    print("ML Health server running on port 8004")

if __name__ == "__main__":
    asyncio.run(run_health_server())
EOF

# Start health server in background
python /app/ml_health_server.py &
HEALTH_PID=$!

# Function to handle shutdown
cleanup() {
    log "Shutting down ML Defense Engine gracefully..."
    if [[ -n "$HEALTH_PID" ]]; then
        kill $HEALTH_PID 2>/dev/null || true
    fi
    if [[ -n "$MAIN_PID" ]]; then
        kill $MAIN_PID 2>/dev/null || true
        wait $MAIN_PID 2>/dev/null || true
    fi
    log "ML Defense shutdown complete"
    exit 0
}

# Set trap for graceful shutdown
trap cleanup SIGTERM SIGINT

# Start main application
log "Starting ML Defense Engine..."
log "Configuration:"
log "  Environment: $ENVIRONMENT"
log "  GPU Support: $USE_GPU"
log "  ML Training: ${ML_TRAINING_ENABLED:-false}"
log "  Adversarial Training: ${ADVERSARIAL_TRAINING:-false}"
log "  Threat Correlation: ${THREAT_CORRELATION:-false}"
log "  OMP Threads: $OMP_NUM_THREADS"

# Execute the main command
exec "$@" &
MAIN_PID=$!

# Wait for main process
wait $MAIN_PID
