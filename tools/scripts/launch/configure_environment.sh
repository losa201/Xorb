#!/bin/bash
# XORB Environment Auto-Configuration Script
# Alternative shell-based configurator for systems without Python dependencies

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# System detection variables
OS_TYPE=""
ARCHITECTURE=""
CPU_CORES=0
RAM_GB=0
IS_ARM=false
IS_VIRTUALIZED=false
DOCKER_VERSION=""
SYSTEM_PROFILE=""
XORB_MODE=""

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

print_header() {
    echo -e "${BLUE}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║    ██╗  ██╗ ██████╗ ██████╗ ██████╗     ██████╗ ██████╗ ███╗   ██╗███████╗   ║
║    ╚██╗██╔╝██╔═══██╗██╔══██╗██╔══██╗   ██╔════╝██╔═══██╗████╗  ██║██╔════╝   ║
║     ╚███╔╝ ██║   ██║██████╔╝██████╔╝   ██║     ██║   ██║██╔██╗ ██║█████╗     ║
║     ██╔██╗ ██║   ██║██╔══██╗██╔══██╗   ██║     ██║   ██║██║╚██╗██║██╔══╝     ║
║    ██╔╝ ██╗╚██████╔╝██║  ██║██████╔╝   ╚██████╗╚██████╔╝██║ ╚████║██║        ║
║    ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝     ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝        ║
║                                                                               ║
║              🤖 AUTONOMOUS SELF-CONFIGURATION SYSTEM 🤖                      ║
║                    Detecting and Optimizing Environment                      ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}\n"
}

detect_system() {
    log "🔍 Detecting system capabilities..."
    
    # OS Detection
    OS_TYPE=$(uname -s)
    OS_VERSION=$(uname -r)
    ARCHITECTURE=$(uname -m)
    
    # CPU Detection
    if command -v nproc &> /dev/null; then
        CPU_CORES=$(nproc)
    elif [ -f /proc/cpuinfo ]; then
        CPU_CORES=$(grep -c ^processor /proc/cpuinfo)
    else
        CPU_CORES=1
    fi
    
    # Memory Detection
    if [ -f /proc/meminfo ]; then
        RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        RAM_GB=$((RAM_KB / 1024 / 1024))
    elif command -v free &> /dev/null; then
        RAM_GB=$(free -g | awk 'NR==2{print $2}')
    else
        RAM_GB=1
    fi
    
    # Architecture Detection
    case "$ARCHITECTURE" in
        arm64|aarch64|armv7l|armv6l)
            IS_ARM=true
            ;;
        *)
            IS_ARM=false
            ;;
    esac
    
    # Virtualization Detection
    if [ -f /.dockerenv ] || [ -n "${container:-}" ]; then
        IS_VIRTUALIZED=true
    elif [ -d /proc/vz ] || [ -d /proc/xen ]; then
        IS_VIRTUALIZED=true
    elif [ -f /sys/class/dmi/id/sys_vendor ]; then
        VENDOR=$(cat /sys/class/dmi/id/sys_vendor 2>/dev/null | tr '[:upper:]' '[:lower:]')
        if echo "$VENDOR" | grep -qE "(vmware|virtualbox|qemu|kvm)"; then
            IS_VIRTUALIZED=true
        else
            IS_VIRTUALIZED=false
        fi
    else
        IS_VIRTUALIZED=false
    fi
    
    # Docker Detection
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
    else
        DOCKER_VERSION="not_installed"
    fi
    
    log "System detected:"
    log "   OS: $OS_TYPE $OS_VERSION"
    log "   Architecture: $ARCHITECTURE"
    log "   CPU Cores: $CPU_CORES"
    log "   RAM: ${RAM_GB}GB"
    log "   ARM: $IS_ARM"
    log "   Virtualized: $IS_VIRTUALIZED"
    log "   Docker: $DOCKER_VERSION"
}

classify_system() {
    log "🎯 Classifying system profile..."
    
    # System classification logic
    if [ "$IS_ARM" = true ]; then
        SYSTEM_PROFILE="RPI"
    elif [ "$CPU_CORES" -ge 32 ] && [ "$RAM_GB" -ge 64 ]; then
        SYSTEM_PROFILE="EPYC_SERVER"
    elif [ "$CPU_CORES" -ge 16 ] && [ "$RAM_GB" -ge 32 ]; then
        SYSTEM_PROFILE="BARE_METAL"
    elif [ "$IS_VIRTUALIZED" = true ] || [ "$CPU_CORES" -le 8 ]; then
        if [ "$CPU_CORES" -le 2 ] && [ "$RAM_GB" -le 4 ]; then
            SYSTEM_PROFILE="CLOUD_MICRO"
        elif [ "$CPU_CORES" -le 4 ] && [ "$RAM_GB" -le 8 ]; then
            SYSTEM_PROFILE="CLOUD_SMALL"
        else
            SYSTEM_PROFILE="CLOUD_MEDIUM"
        fi
    else
        SYSTEM_PROFILE="BARE_METAL"
    fi
    
    # Determine XORB mode
    case "$SYSTEM_PROFILE" in
        "RPI"|"CLOUD_MICRO")
            XORB_MODE="SIMPLE"
            ;;
        "CLOUD_SMALL"|"CLOUD_MEDIUM")
            XORB_MODE="ENHANCED"
            ;;
        *)
            XORB_MODE="FULL"
            ;;
    esac
    
    log "✅ System classified as: $SYSTEM_PROFILE"
    log "   XORB Mode: $XORB_MODE"
}

generate_env_file() {
    log "📝 Generating .xorb.env configuration..."
    
    # Calculate configuration values based on profile
    case "$SYSTEM_PROFILE" in
        "RPI")
            AGENT_CONCURRENCY=2
            MAX_MISSIONS=1
            WORKER_THREADS=2
            MONITORING_ENABLED=false
            MEMORY_LIMIT=$((RAM_GB * 1024 * 8 / 10))  # 80% of RAM
            ;;
        "CLOUD_MICRO")
            AGENT_CONCURRENCY=4
            MAX_MISSIONS=2
            WORKER_THREADS=2
            MONITORING_ENABLED=false
            MEMORY_LIMIT=$((RAM_GB * 1024 * 8 / 10))
            ;;
        "CLOUD_SMALL")
            AGENT_CONCURRENCY=8
            MAX_MISSIONS=3
            WORKER_THREADS=4
            MONITORING_ENABLED=true
            MEMORY_LIMIT=$((RAM_GB * 1024 * 8 / 10))
            ;;
        "CLOUD_MEDIUM")
            AGENT_CONCURRENCY=16
            MAX_MISSIONS=5
            WORKER_THREADS=6
            MONITORING_ENABLED=true
            MEMORY_LIMIT=$((RAM_GB * 1024 * 8 / 10))
            ;;
        "BARE_METAL")
            AGENT_CONCURRENCY=32
            MAX_MISSIONS=10
            WORKER_THREADS=8
            MONITORING_ENABLED=true
            MEMORY_LIMIT=$((RAM_GB * 1024 * 8 / 10))
            ;;
        "EPYC_SERVER")
            AGENT_CONCURRENCY=64
            MAX_MISSIONS=20
            WORKER_THREADS=16
            MONITORING_ENABLED=true
            MEMORY_LIMIT=$((RAM_GB * 1024 * 8 / 10))
            ;;
    esac
    
    # Generate .xorb.env file
    cat > "$PROJECT_ROOT/.xorb.env" << EOF
# XORB Auto-Generated Environment Configuration
# Generated: $(date -Iseconds)
# System Profile: $SYSTEM_PROFILE
# XORB Mode: $XORB_MODE
#
# This file was automatically generated by XORB configure_environment.sh
# Modify with caution - regenerate with: ./configure_environment.sh

# Core XORB Configuration
XORB_MODE=$SYSTEM_PROFILE
XORB_AGENT_CONCURRENCY=$AGENT_CONCURRENCY
XORB_MAX_CONCURRENT_MISSIONS=$MAX_MISSIONS
XORB_WORKER_THREADS=$WORKER_THREADS
XORB_MONITORING_ENABLED=$MONITORING_ENABLED
XORB_SYSTEM_PROFILE=$SYSTEM_PROFILE
XORB_MEMORY_LIMIT_MB=$MEMORY_LIMIT
XORB_CPU_LIMIT=$CPU_CORES

# System-specific optimizations
XORB_IS_ARM=$IS_ARM
XORB_IS_VIRTUALIZED=$IS_VIRTUALIZED
XORB_CPU_CORES=$CPU_CORES
XORB_RAM_GB=$RAM_GB

# Docker configuration
DOCKER_BUILDKIT=1
COMPOSE_DOCKER_CLI_BUILD=1

# Performance tuning
XORB_ORCHESTRATION_CYCLE_TIME=$(get_cycle_time)
XORB_DATABASE_POOL_SIZE=$(get_db_pool_size)
XORB_REDIS_POOL_SIZE=$(get_redis_pool_size)

# Feature flags
XORB_PHASE_11_ENABLED=$([ "$SYSTEM_PROFILE" != "RPI" ] && echo "true" || echo "false")
XORB_PLUGIN_DISCOVERY_ENABLED=$([ "$SYSTEM_PROFILE" != "RPI" ] && [ "$SYSTEM_PROFILE" != "CLOUD_MICRO" ] && echo "true" || echo "false")
XORB_PI5_OPTIMIZATION=$([ "$IS_ARM" = true ] && [ "$CPU_CORES" -ge 4 ] && echo "true" || echo "false")

# Database Configuration
POSTGRES_USER=xorb
POSTGRES_PASSWORD=xorb_secure_2024
POSTGRES_DB=xorb
NEO4J_PASSWORD=xorb_neo4j_2024
GRAFANA_ADMIN_PASSWORD=xorb_admin_2024

# API Keys (configure as needed)
OPENROUTER_API_KEY=
CEREBRAS_API_KEY=

# Container Configuration
COMPOSE_PROJECT_NAME=xorb
EOF

    log "✅ Environment file generated: .xorb.env"
}

get_cycle_time() {
    case "$SYSTEM_PROFILE" in
        "RPI") echo "800" ;;
        "CLOUD_MICRO") echo "600" ;;
        "CLOUD_SMALL") echo "400" ;;
        "CLOUD_MEDIUM") echo "300" ;;
        "BARE_METAL") echo "200" ;;
        "EPYC_SERVER") echo "100" ;;
        *) echo "400" ;;
    esac
}

get_db_pool_size() {
    case "$SYSTEM_PROFILE" in
        "RPI") echo "5" ;;
        "CLOUD_MICRO") echo "10" ;;
        "CLOUD_SMALL") echo "15" ;;
        "CLOUD_MEDIUM") echo "20" ;;
        "BARE_METAL") echo "30" ;;
        "EPYC_SERVER") echo "50" ;;
        *) echo "20" ;;
    esac
}

get_redis_pool_size() {
    case "$SYSTEM_PROFILE" in
        "RPI") echo "5" ;;
        "CLOUD_MICRO") echo "10" ;;
        "CLOUD_SMALL") echo "15" ;;
        "CLOUD_MEDIUM") echo "20" ;;
        "BARE_METAL") echo "25" ;;
        "EPYC_SERVER") echo "40" ;;
        *) echo "20" ;;
    esac
}

select_compose_file() {
    log "🐳 Selecting appropriate Docker Compose configuration..."
    
    # Select compose file based on profile
    case "$SYSTEM_PROFILE" in
        "RPI"|"CLOUD_MICRO")
            COMPOSE_FILE="docker-compose.simple.yml"
            ;;
        "CLOUD_SMALL"|"CLOUD_MEDIUM")
            COMPOSE_FILE="docker-compose.yml"
            ;;
        *)
            COMPOSE_FILE="docker-compose.unified.yml"
            ;;
    esac
    
    # Create symlink for easy deployment
    if [ -f "$PROJECT_ROOT/$COMPOSE_FILE" ]; then
        ln -sf "$COMPOSE_FILE" "$PROJECT_ROOT/docker-compose.auto.yml"
        log "✅ Selected compose file: $COMPOSE_FILE -> docker-compose.auto.yml"
    else
        warn "Compose file $COMPOSE_FILE not found, using docker-compose.unified.yml"
        ln -sf "docker-compose.unified.yml" "$PROJECT_ROOT/docker-compose.auto.yml"
    fi
}

generate_bootstrap_report() {
    log "📊 Generating bootstrap report..."
    
    mkdir -p "$PROJECT_ROOT/logs"
    
    cat > "$PROJECT_ROOT/logs/bootstrap_report.json" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "version": "2.0.0",
  "system_capabilities": {
    "os_type": "$OS_TYPE",
    "architecture": "$ARCHITECTURE",
    "cpu_cores": $CPU_CORES,
    "ram_gb": $RAM_GB,
    "is_arm": $IS_ARM,
    "is_virtualized": $IS_VIRTUALIZED,
    "docker_version": "$DOCKER_VERSION",
    "profile": "$SYSTEM_PROFILE"
  },
  "generated_configuration": {
    "mode": "$XORB_MODE",
    "system_profile": "$SYSTEM_PROFILE",
    "agent_concurrency": $AGENT_CONCURRENCY,
    "max_concurrent_missions": $MAX_MISSIONS,
    "worker_threads": $WORKER_THREADS,
    "monitoring_enabled": $MONITORING_ENABLED,
    "memory_limit_mb": $MEMORY_LIMIT
  },
  "deployment_readiness": {
    "docker_available": $([ "$DOCKER_VERSION" != "not_installed" ] && echo "true" || echo "false"),
    "sufficient_resources": $([ "$RAM_GB" -ge 2 ] && echo "true" || echo "false")
  },
  "recommendations": $(generate_recommendations)
}
EOF

    log "✅ Bootstrap report generated: logs/bootstrap_report.json"
}

generate_recommendations() {
    local recommendations="["
    local first=true
    
    if [ "$RAM_GB" -lt 4 ]; then
        [ "$first" = false ] && recommendations+=","
        recommendations+="\"Consider increasing RAM for better performance\""
        first=false
    fi
    
    if [ "$CPU_CORES" -lt 4 ]; then
        [ "$first" = false ] && recommendations+=","
        recommendations+="\"Limited CPU cores may affect concurrent agent execution\""
        first=false
    fi
    
    if [ "$DOCKER_VERSION" = "not_installed" ]; then
        [ "$first" = false ] && recommendations+=","
        recommendations+="\"Docker not found - please install Docker\""
        first=false
    fi
    
    if [ "$SYSTEM_PROFILE" = "RPI" ]; then
        [ "$first" = false ] && recommendations+=","
        recommendations+="\"ARM optimization enabled - some features disabled for stability\""
        first=false
    fi
    
    if [ "$MONITORING_ENABLED" = false ]; then
        [ "$first" = false ] && recommendations+=","
        recommendations+="\"Monitoring disabled due to resource constraints\""
        first=false
    fi
    
    recommendations+="]"
    echo "$recommendations"
}

print_summary() {
    echo
    echo "="*80
    echo "🚀 XORB AUTO-CONFIGURATION COMPLETE"
    echo "="*80
    
    echo
    echo "📊 SYSTEM PROFILE: $SYSTEM_PROFILE"
    echo "   CPU: $CPU_CORES cores"
    echo "   RAM: ${RAM_GB}GB"
    echo "   Architecture: $ARCHITECTURE"
    echo "   OS: $OS_TYPE"
    
    echo
    echo "⚙️ XORB CONFIGURATION:"
    echo "   Mode: $XORB_MODE"
    echo "   Agent Concurrency: $AGENT_CONCURRENCY"
    echo "   Max Concurrent Missions: $MAX_MISSIONS"
    echo "   Worker Threads: $WORKER_THREADS"
    echo "   Monitoring: $([ "$MONITORING_ENABLED" = true ] && echo "Enabled" || echo "Disabled")"
    echo "   Memory Limit: ${MEMORY_LIMIT}MB"
    
    echo
    echo "📁 FILES GENERATED:"
    echo "   • .xorb.env"
    echo "   • docker-compose.auto.yml -> $COMPOSE_FILE"
    echo "   • logs/bootstrap_report.json"
    
    if [ "$SYSTEM_PROFILE" = "RPI" ]; then
        echo
        echo "🍓 RASPBERRY PI OPTIMIZATIONS:"
        echo "   • ARM-specific container images"
        echo "   • Reduced resource allocation"
        echo "   • Simplified service stack"
    fi
    
    echo
    echo "="*80
}

validate_environment() {
    log "✅ Validating environment..."
    
    # Check Docker
    if [ "$DOCKER_VERSION" = "not_installed" ]; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check minimum resources
    if [ "$RAM_GB" -lt 1 ]; then
        error "Insufficient RAM (minimum 1GB required)"
    fi
    
    # Check Docker Compose
    if ! command -v docker &> /dev/null || ! docker compose version &> /dev/null; then
        if ! command -v docker-compose &> /dev/null; then
            error "Docker Compose not found. Please install Docker Compose."
        fi
    fi
    
    log "✅ Environment validation passed"
}

main() {
    print_header
    
    # Detection phase
    detect_system
    classify_system
    validate_environment
    
    # Configuration generation
    generate_env_file
    select_compose_file
    generate_bootstrap_report
    
    # Summary and confirmation
    print_summary
    
    echo
    read -p "🚀 Ready to deploy XORB with optimized configuration? [Y/n]: " -r response
    
    case "$response" in
        ""|[Yy]|[Yy][Ee][Ss])
            echo
            echo "✅ Configuration complete! Use the following command to deploy:"
            echo "   docker compose -f docker-compose.auto.yml --env-file .xorb.env up -d"
            echo
            echo "📊 Monitor deployment with:"
            echo "   docker compose -f docker-compose.auto.yml logs -f"
            ;;
        *)
            echo
            echo "⏸️ Configuration saved. Deploy when ready with:"
            echo "   docker compose -f docker-compose.auto.yml --env-file .xorb.env up -d"
            ;;
    esac
}

# Run main function
main "$@"