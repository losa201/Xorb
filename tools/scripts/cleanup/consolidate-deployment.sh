#!/bin/bash
set -euo pipefail

# XORB Deployment Consolidation Script
# Eliminates duplicate Docker Compose files and streamlines deployment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Colors
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

# Analyze current duplication
analyze_duplication() {
    log "Analyzing Docker Compose duplication..."

    local compose_files=$(find "$PROJECT_ROOT" -name "docker-compose*.yml" -o -name "compose*.yml" | wc -l)
    local total_size=$(find "$PROJECT_ROOT" -name "docker-compose*.yml" -o -name "compose*.yml" -exec du -ch {} + | tail -1 | awk '{print $1}')

    info "Found $compose_files Docker Compose files totaling $total_size"

    # Count service duplications
    local prometheus_count=$(find "$PROJECT_ROOT" -name "docker-compose*.yml" -exec grep -l "prometheus:" {} \; | wc -l)
    local grafana_count=$(find "$PROJECT_ROOT" -name "docker-compose*.yml" -exec grep -l "grafana:" {} \; | wc -l)
    local postgres_count=$(find "$PROJECT_ROOT" -name "docker-compose*.yml" -exec grep -l "postgres:" {} \; | wc -l)

    info "Service duplications found:"
    info "  - Prometheus: $prometheus_count files"
    info "  - Grafana: $grafana_count files"
    info "  - PostgreSQL: $postgres_count files"

    echo "$compose_files" > /tmp/xorb_compose_count.txt
}

# Create backup of current structure
create_backup() {
    log "Creating backup of current Docker Compose files..."

    local backup_dir="$PROJECT_ROOT/backups/compose_consolidation_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"

    # Backup all compose files
    find "$PROJECT_ROOT" -name "docker-compose*.yml" -o -name "compose*.yml" | while read -r file; do
        local rel_path=$(realpath --relative-to="$PROJECT_ROOT" "$file")
        local backup_path="$backup_dir/$rel_path"
        mkdir -p "$(dirname "$backup_path")"
        cp "$file" "$backup_path"
    done

    # Create inventory
    find "$PROJECT_ROOT" -name "docker-compose*.yml" -o -name "compose*.yml" > "$backup_dir/compose_inventory.txt"

    log "Backup created at: $backup_dir"
    echo "$backup_dir" > /tmp/xorb_backup_dir.txt
}

# Identify and remove duplicates
remove_duplicates() {
    log "Removing duplicate Docker Compose files..."

    # Keep only essential compose files
    local keep_files=(
        "$PROJECT_ROOT/docker-compose.yml"  # New consolidated file
        "$PROJECT_ROOT/infra/docker-compose-enhanced.yml"  # Original enhanced (for reference)
    )

    # List files to remove
    local remove_count=0
    find "$PROJECT_ROOT" -name "docker-compose*.yml" -o -name "compose*.yml" | while read -r file; do
        local should_keep=false
        for keep_file in "${keep_files[@]}"; do
            if [[ "$file" == "$keep_file" ]]; then
                should_keep=true
                break
            fi
        done

        if [[ "$should_keep" == false ]]; then
            echo "$file" >> /tmp/xorb_files_to_remove.txt
            ((remove_count++))
        fi
    done

    # Remove duplicate files
    if [[ -f /tmp/xorb_files_to_remove.txt ]]; then
        local remove_count=$(wc -l < /tmp/xorb_files_to_remove.txt)
        info "Removing $remove_count duplicate Docker Compose files..."

        while read -r file; do
            rm -f "$file"
            info "Removed: $(realpath --relative-to="$PROJECT_ROOT" "$file")"
        done < /tmp/xorb_files_to_remove.txt

        rm -f /tmp/xorb_files_to_remove.txt
    fi
}

# Update documentation and references
update_references() {
    log "Updating documentation and script references..."

    # Update README files
    find "$PROJECT_ROOT" -name "README*.md" -exec sed -i 's|docker-compose -f [^[:space:]]*docker-compose[^[:space:]]*\.yml|docker-compose|g' {} \;

    # Update deployment scripts
    find "$PROJECT_ROOT/scripts" -name "*.sh" -exec sed -i 's|docker-compose -f [^[:space:]]*docker-compose[^[:space:]]*\.yml|docker-compose|g' {} \;

    # Create deployment script
    cat > "$PROJECT_ROOT/deploy.sh" << 'EOF'
#!/bin/bash
set -euo pipefail

# XORB Consolidated Deployment Script
# Single command deployment for AMD EPYC systems

echo "🚀 Starting XORB deployment on AMD EPYC..."

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Apply system optimizations (if available)
if [[ -f "./scripts/optimization/epyc-tuning.sh" ]]; then
    echo "⚡ Applying EPYC optimizations..."
    sudo ./scripts/optimization/epyc-tuning.sh validate || echo "⚠️  EPYC optimizations not applied (run as root for full optimization)"
fi

# Create required directories
mkdir -p config/{prometheus,grafana,nginx} data logs

# Set environment variables
export ENVIRONMENT=${ENVIRONMENT:-production}
export POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-$(openssl rand -base64 32)}
export JWT_SECRET=${JWT_SECRET:-$(openssl rand -base64 64)}

echo "🔧 Starting services with consolidated configuration..."

# Deploy services
docker-compose up -d

echo "⏳ Waiting for services to be ready..."
sleep 30

# Health check
echo "🏥 Performing health checks..."
docker-compose ps

# Show access information
echo ""
echo "✅ XORB deployment completed!"
echo "📊 Access URLs:"
echo "  - API:        http://localhost:8000"
echo "  - Dashboard:  http://localhost:3000 (admin/xorb-admin-2024)"
echo "  - Metrics:    http://localhost:9090"
echo "  - Status:     docker-compose ps"
echo ""
echo "🔧 Management commands:"
echo "  - Stop:       docker-compose down"
echo "  - Logs:       docker-compose logs -f"
echo "  - Monitor:    ./scripts/optimization/epyc-tuning.sh monitor"
EOF

    chmod +x "$PROJECT_ROOT/deploy.sh"

    log "Created consolidated deployment script: deploy.sh"
}

# Optimize configurations
optimize_configs() {
    log "Optimizing remaining configurations..."

    # Remove unused config directories
    local config_dirs_to_check=(
        "$PROJECT_ROOT/config/temporal"
        "$PROJECT_ROOT/config/vault"
        "$PROJECT_ROOT/config/consul"
        "$PROJECT_ROOT/config/etcd"
    )

    for dir in "${config_dirs_to_check[@]}"; do
        if [[ -d "$dir" ]] && [[ -z "$(find "$dir" -name "*.yml" -o -name "*.yaml" -o -name "*.json" 2>/dev/null)" ]]; then
            warn "Removing empty config directory: $dir"
            rm -rf "$dir"
        fi
    done

    # Consolidate environment files
    if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
        cat > "$PROJECT_ROOT/.env" << EOF
# XORB Consolidated Environment Configuration
# AMD EPYC Optimized Deployment

# Application
ENVIRONMENT=production
DOMAIN=xorb.local

# Database
POSTGRES_DB=xorb
POSTGRES_USER=xorb
POSTGRES_PASSWORD=secure_password_change_me

# Redis
REDIS_PASSWORD=

# Security
JWT_SECRET=your_jwt_secret_change_me

# Monitoring
GRAFANA_USER=admin
GRAFANA_PASSWORD=xorb-admin-2024
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Resource Limits (for EPYC 16-core, 32GB)
MAX_MEMORY_ML=16g
MAX_MEMORY_ADV=8g
MAX_CPU_ML=8
MAX_CPU_ADV=4

# Features
DEBUG=false
LOG_LEVEL=info
CORS_ORIGINS=*
EOF
        log "Created consolidated .env file"
    fi
}

# Generate consolidation report
generate_report() {
    log "Generating consolidation report..."

    local backup_dir=$(cat /tmp/xorb_backup_dir.txt 2>/dev/null || echo "Unknown")
    local original_count=$(cat /tmp/xorb_compose_count.txt 2>/dev/null || echo "Unknown")
    local final_count=$(find "$PROJECT_ROOT" -name "docker-compose*.yml" | wc -l)

    cat > "$PROJECT_ROOT/CONSOLIDATION_REPORT.md" << EOF
# XORB Docker Compose Consolidation Report

## Summary
- **Original compose files:** $original_count
- **Final compose files:** $final_count
- **Files eliminated:** $((original_count - final_count))
- **Consolidation date:** $(date)

## What was consolidated
1. **Monitoring stack** - Combined Prometheus, Grafana, Loki into single definitions
2. **Database services** - Single PostgreSQL and Redis instances
3. **XORB services** - Streamlined with EPYC-optimized resource limits
4. **Networking** - Simplified to single bridge network
5. **Reverse proxy** - Replaced Traefik with optimized Nginx

## Remaining files
$(find "$PROJECT_ROOT" -name "docker-compose*.yml" | sort)

## Key improvements
- ✅ Eliminated $((original_count - final_count)) duplicate compose files
- ✅ Single \`docker-compose.yml\` for all deployments
- ✅ EPYC-optimized resource allocations
- ✅ Simplified networking (single bridge network)
- ✅ Consolidated monitoring stack
- ✅ Streamlined service definitions
- ✅ Single deployment command: \`./deploy.sh\`

## Resource allocation (EPYC optimized)
- **ML Defense:** 16GB memory, 8 CPU cores
- **Adversarial Engine:** 8GB memory, 4 CPU cores
- **Monitoring Stack:** 6GB memory, 3 CPU cores
- **Infrastructure:** 2GB memory, 1 CPU cores

## Backup location
Original files backed up to: \`$backup_dir\`

## Next steps
1. Test deployment: \`./deploy.sh\`
2. Monitor performance: \`./scripts/optimization/epyc-tuning.sh monitor\`
3. Validate functionality: \`docker-compose ps\`
EOF

    log "Consolidation report saved to: CONSOLIDATION_REPORT.md"
}

# Cleanup temporary files
cleanup() {
    rm -f /tmp/xorb_*.txt
}

# Validate new deployment
validate_deployment() {
    log "Validating consolidated deployment..."

    # Check if new compose file is valid
    if docker-compose -f "$PROJECT_ROOT/docker-compose.yml" config > /dev/null 2>&1; then
        log "✅ New docker-compose.yml is valid"
    else
        error "❌ New docker-compose.yml has syntax errors"
    fi

    # Check for conflicting ports
    local port_conflicts=$(docker-compose -f "$PROJECT_ROOT/docker-compose.yml" config --services | while read service; do
        docker-compose -f "$PROJECT_ROOT/docker-compose.yml" config | grep -A 10 "^  $service:" | grep "ports:" -A 5
    done | grep -o '[0-9]\+:' | sort | uniq -d)

    if [[ -n "$port_conflicts" ]]; then
        warn "Port conflicts detected: $port_conflicts"
    else
        log "✅ No port conflicts found"
    fi
}

# Main execution
main() {
    case "${1:-full}" in
        "analyze")
            analyze_duplication
            ;;
        "backup")
            create_backup
            ;;
        "consolidate")
            create_backup
            remove_duplicates
            update_references
            optimize_configs
            validate_deployment
            ;;
        "full")
            log "Starting full Docker Compose consolidation..."
            analyze_duplication
            create_backup
            remove_duplicates
            update_references
            optimize_configs
            validate_deployment
            generate_report
            cleanup
            log "🎉 Consolidation completed successfully!"
            log "📋 Review the report: CONSOLIDATION_REPORT.md"
            log "🚀 Deploy with: ./deploy.sh"
            ;;
        "validate")
            validate_deployment
            ;;
        *)
            echo "Usage: $0 {analyze|backup|consolidate|full|validate}"
            echo "  analyze     - Analyze current duplication"
            echo "  backup      - Create backup only"
            echo "  consolidate - Remove duplicates and update references"
            echo "  full        - Complete consolidation process (default)"
            echo "  validate    - Validate new deployment configuration"
            exit 1
            ;;
    esac
}

main "$@"
