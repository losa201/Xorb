#!/bin/bash
set -euo pipefail

# AMD EPYC 7002 System Optimization Script
# Optimizes system for ML workloads on 16-core, 32GB RAM configuration

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root for system optimizations"
    fi
}

# Detect hardware configuration
detect_hardware() {
    log "Detecting hardware configuration..."
    
    local cpu_cores=$(nproc)
    local total_memory=$(free -g | awk '/^Mem:/{print $2}')
    local cpu_model=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
    
    info "CPU: $cpu_model"
    info "Cores: $cpu_cores"
    info "Memory: ${total_memory}GB"
    
    # Verify EPYC configuration
    if [[ $cpu_cores -ne 16 ]]; then
        warn "Expected 16 cores, found $cpu_cores"
    fi
    
    if [[ $total_memory -lt 30 ]]; then
        warn "Expected ~32GB memory, found ${total_memory}GB"
    fi
    
    # Check for GPU (should be none)
    if command -v nvidia-smi &> /dev/null; then
        warn "NVIDIA drivers detected - this is a CPU-only deployment"
    fi
    
    log "Hardware detection completed"
}

# Optimize CPU settings
optimize_cpu() {
    log "Optimizing CPU settings for AMD EPYC..."
    
    # Set CPU governor to performance
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        if [[ -w $cpu ]]; then
            echo "performance" > "$cpu"
        fi
    done
    
    # Disable CPU frequency scaling
    if [[ -w /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
        echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo
    fi
    
    # Enable NUMA balancing
    echo 1 > /proc/sys/kernel/numa_balancing
    
    # Optimize scheduler
    echo 1 > /proc/sys/kernel/sched_migration_cost_ns
    
    # Set CPU affinity optimization
    echo 2 > /proc/sys/kernel/sched_domain/cpu0/domain1/flags
    
    log "CPU optimizations applied"
}

# Optimize memory settings
optimize_memory() {
    log "Optimizing memory settings for 32GB RAM..."
    
    # Disable swap for ML workloads
    swapoff -a
    
    # Update sysctl settings
    cat >> /etc/sysctl.conf << EOF

# XORB EPYC Memory Optimizations
vm.swappiness=1
vm.dirty_ratio=15
vm.dirty_background_ratio=5
vm.overcommit_memory=0
vm.overcommit_ratio=80
vm.max_map_count=262144

# Transparent Huge Pages
vm.nr_hugepages=1024
EOF
    
    # Apply transparent huge pages
    echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
    echo madvise > /sys/kernel/mm/transparent_hugepage/defrag
    
    # Apply sysctl settings
    sysctl -p
    
    log "Memory optimizations applied"
}

# Optimize I/O and storage
optimize_io() {
    log "Optimizing I/O and storage settings..."
    
    # Set I/O scheduler to deadline for SSDs
    for disk in /sys/block/sd*/queue/scheduler; do
        if [[ -w $disk ]]; then
            echo deadline > "$disk"
        fi
    done
    
    for disk in /sys/block/nvme*/queue/scheduler; do
        if [[ -w $disk ]]; then
            echo none > "$disk"
        fi
    done
    
    # Optimize read-ahead for ML workloads
    for disk in /sys/block/sd*/queue/read_ahead_kb; do
        if [[ -w $disk ]]; then
            echo 512 > "$disk"
        fi
    done
    
    for disk in /sys/block/nvme*/queue/read_ahead_kb; do
        if [[ -w $disk ]]; then
            echo 512 > "$disk"
        fi
    done
    
    log "I/O optimizations applied"
}

# Optimize network settings
optimize_network() {
    log "Optimizing network settings..."
    
    cat >> /etc/sysctl.conf << EOF

# XORB Network Optimizations
net.core.rmem_max=134217728
net.core.wmem_max=134217728
net.core.rmem_default=262144
net.core.wmem_default=262144
net.ipv4.tcp_rmem=4096 262144 134217728
net.ipv4.tcp_wmem=4096 262144 134217728
net.ipv4.tcp_congestion_control=bbr
net.ipv4.tcp_window_scaling=1
net.ipv4.tcp_timestamps=1
net.ipv4.tcp_sack=1
net.core.netdev_max_backlog=5000
EOF
    
    sysctl -p
    
    log "Network optimizations applied"
}

# Configure container runtime
configure_docker() {
    log "Configuring Docker for EPYC optimization..."
    
    # Create Docker daemon configuration
    mkdir -p /etc/docker
    cat > /etc/docker/daemon.json << EOF
{
    "exec-opts": ["native.cgroupdriver=systemd"],
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "100m",
        "max-file": "3"
    },
    "storage-driver": "overlay2",
    "default-ulimits": {
        "memlock": {
            "Hard": -1,
            "Name": "memlock",
            "Soft": -1
        },
        "nofile": {
            "Hard": 65536,
            "Name": "nofile",
            "Soft": 65536
        }
    },
    "default-runtime": "runc",
    "runtimes": {
        "runc": {
            "path": "runc"
        }
    },
    "features": {
        "cri": false
    }
}
EOF
    
    # Restart Docker service
    systemctl restart docker
    
    log "Docker configuration updated"
}

# Create systemd service for optimizations
create_optimization_service() {
    log "Creating optimization service..."
    
    cat > /etc/systemd/system/xorb-epyc-optimization.service << EOF
[Unit]
Description=XORB EPYC System Optimizations
After=multi-user.target

[Service]
Type=oneshot
ExecStart=$SCRIPT_DIR/epyc-tuning.sh apply-runtime
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    systemctl enable xorb-epyc-optimization.service
    
    log "Optimization service created and enabled"
}

# Apply runtime optimizations (called by systemd service)
apply_runtime_optimizations() {
    log "Applying runtime optimizations..."
    
    # Set CPU performance mode
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        if [[ -w $cpu ]]; then
            echo "performance" > "$cpu"
        fi
    done
    
    # Configure IRQ affinity for better performance
    if command -v irqbalance &> /dev/null; then
        systemctl stop irqbalance
        systemctl disable irqbalance
    fi
    
    # Set process priorities for XORB services
    if pgrep -f "xorb-ml-defense" > /dev/null; then
        renice -10 $(pgrep -f "xorb-ml-defense")
    fi
    
    if pgrep -f "xorb-adversarial" > /dev/null; then
        renice -5 $(pgrep -f "xorb-adversarial")
    fi
    
    log "Runtime optimizations applied"
}

# Create monitoring script
create_monitoring() {
    log "Creating performance monitoring script..."
    
    cat > "$PROJECT_ROOT/scripts/monitoring/epyc-monitor.sh" << 'EOF'
#!/bin/bash

# EPYC Performance Monitoring Script
INTERVAL=${1:-5}

echo "EPYC Performance Monitor (interval: ${INTERVAL}s)"
echo "================================================"

while true; do
    clear
    echo "XORB EPYC Performance Dashboard - $(date)"
    echo "=========================================="
    
    # CPU Usage
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)" | awk '{print "  Total: " $2 " user, " $4 " system, " $8 " idle"}'
    
    # Memory Usage
    echo "Memory Usage:"
    free -h | grep "Mem:" | awk '{print "  Used: " $3 "/" $2 " (" int($3/$2*100) "%)"}'
    
    # Top processes by CPU
    echo "Top CPU Consumers:"
    ps aux --sort=-%cpu | head -6 | tail -5 | awk '{printf "  %-15s %5s%% %s\n", $11, $3, $2}'
    
    # Top processes by Memory
    echo "Top Memory Consumers:"
    ps aux --sort=-%mem | head -6 | tail -5 | awk '{printf "  %-15s %5s%% %s\n", $11, $4, $2}'
    
    # Load Average
    echo "Load Average:"
    uptime | awk -F'load average:' '{print "  " $2}'
    
    # Container Stats (if Docker is running)
    if command -v docker &> /dev/null && docker info &> /dev/null; then
        echo "Container Resources:"
        docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -6
    fi
    
    sleep $INTERVAL
done
EOF
    
    chmod +x "$PROJECT_ROOT/scripts/monitoring/epyc-monitor.sh"
    
    log "Performance monitoring script created"
}

# Validate optimizations
validate_optimizations() {
    log "Validating optimizations..."
    
    local errors=0
    
    # Check CPU governor
    local governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
    if [[ "$governor" != "performance" ]]; then
        warn "CPU governor is not set to performance: $governor"
        ((errors++))
    fi
    
    # Check swap
    local swap_usage=$(free | grep Swap | awk '{print $3}')
    if [[ $swap_usage -gt 0 ]]; then
        warn "Swap is being used: ${swap_usage}KB"
        ((errors++))
    fi
    
    # Check memory settings
    local swappiness=$(cat /proc/sys/vm/swappiness)
    if [[ $swappiness -ne 1 ]]; then
        warn "Swappiness not optimized: $swappiness"
        ((errors++))
    fi
    
    # Check Docker status
    if ! systemctl is-active --quiet docker; then
        warn "Docker service is not running"
        ((errors++))
    fi
    
    if [[ $errors -eq 0 ]]; then
        log "All optimizations validated successfully"
    else
        warn "Found $errors optimization issues"
    fi
    
    return $errors
}

# Print optimization summary
print_summary() {
    log "EPYC Optimization Summary"
    echo "========================"
    echo "CPU Configuration:"
    echo "  - Cores: $(nproc)"
    echo "  - Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'N/A')"
    echo "  - Frequency: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo 'N/A') kHz"
    
    echo "Memory Configuration:"
    echo "  - Total: $(free -h | grep Mem | awk '{print $2}')"
    echo "  - Available: $(free -h | grep Mem | awk '{print $7}')"
    echo "  - Swappiness: $(cat /proc/sys/vm/swappiness)"
    
    echo "Performance Tuning:"
    echo "  - Transparent Huge Pages: $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
    echo "  - NUMA Balancing: $(cat /proc/sys/kernel/numa_balancing)"
    
    echo "Services:"
    echo "  - Docker: $(systemctl is-active docker)"
    echo "  - XORB Optimization: $(systemctl is-active xorb-epyc-optimization.service 2>/dev/null || echo 'not installed')"
}

# Main execution
main() {
    case "${1:-full}" in
        "full")
            log "Starting full EPYC optimization..."
            check_root
            detect_hardware
            optimize_cpu
            optimize_memory
            optimize_io
            optimize_network
            configure_docker
            create_optimization_service
            create_monitoring
            validate_optimizations
            print_summary
            log "EPYC optimization completed successfully!"
            ;;
        "apply-runtime")
            apply_runtime_optimizations
            ;;
        "validate")
            validate_optimizations
            ;;
        "summary")
            print_summary
            ;;
        "monitor")
            exec "$PROJECT_ROOT/scripts/monitoring/epyc-monitor.sh" "${2:-5}"
            ;;
        *)
            echo "Usage: $0 {full|apply-runtime|validate|summary|monitor [interval]}"
            echo "  full         - Apply all optimizations (requires root)"
            echo "  apply-runtime - Apply runtime optimizations only"
            echo "  validate     - Validate current optimizations"
            echo "  summary      - Show optimization summary"
            echo "  monitor      - Start performance monitor"
            exit 1
            ;;
    esac
}

main "$@"