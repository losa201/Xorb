#!/bin/bash
# Health check script for XORB API service

# Exit on any error
set -e

# Configuration
API_HOST=${API_HOST:-localhost}
API_PORT=${API_PORT:-8000}
HEALTH_ENDPOINT="/api/v1/health"
TIMEOUT=${HEALTHCHECK_TIMEOUT:-10}

# Function to check API health
check_api_health() {
    local url="http://${API_HOST}:${API_PORT}${HEALTH_ENDPOINT}"
    
    # Check if the health endpoint responds
    if curl -sf --max-time "$TIMEOUT" "$url" > /dev/null 2>&1; then
        echo "✅ API health check passed"
        return 0
    else
        echo "❌ API health check failed"
        return 1
    fi
}

# Function to check process
check_process() {
    if pgrep -f "uvicorn" > /dev/null; then
        echo "✅ Uvicorn process is running"
        return 0
    else
        echo "❌ Uvicorn process not found"
        return 1
    fi
}

# Function to check memory usage
check_memory() {
    local memory_limit=${MEMORY_LIMIT_MB:-2048}
    local memory_usage
    
    # Get memory usage in MB
    memory_usage=$(ps -o pid,ppid,cmd,%mem,%cpu --sort=-%mem -C python | awk 'NR==2{print $4}' | cut -d. -f1)
    
    if [[ -n "$memory_usage" && "$memory_usage" -lt "$memory_limit" ]]; then
        echo "✅ Memory usage OK: ${memory_usage}%"
        return 0
    else
        echo "⚠️ High memory usage: ${memory_usage}%"
        return 1
    fi
}

# Function to check disk space
check_disk_space() {
    local disk_threshold=${DISK_THRESHOLD:-90}
    local disk_usage
    
    # Get disk usage percentage
    disk_usage=$(df / | awk 'NR==2{gsub(/%/,"",$5); print $5}')
    
    if [[ "$disk_usage" -lt "$disk_threshold" ]]; then
        echo "✅ Disk space OK: ${disk_usage}% used"
        return 0
    else
        echo "⚠️ Low disk space: ${disk_usage}% used"
        return 1
    fi
}

# Main health check function
main() {
    echo "🔍 Running XORB API health check..."
    
    local exit_code=0
    
    # Check API health (critical)
    if ! check_api_health; then
        exit_code=1
    fi
    
    # Check process (critical)
    if ! check_process; then
        exit_code=1
    fi
    
    # Check memory (warning only)
    check_memory || true
    
    # Check disk space (warning only)
    check_disk_space || true
    
    if [[ $exit_code -eq 0 ]]; then
        echo "✅ Overall health check passed"
    else
        echo "❌ Health check failed"
    fi
    
    exit $exit_code
}

# Run main function
main "$@"