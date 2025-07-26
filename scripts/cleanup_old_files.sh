#!/bin/bash
# XORB Repository Cleanup Script
# Removes old/duplicate files after refactoring

set -euo pipefail

info() { echo -e "\033[1;34m[INFO]\033[0m $1"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $1"; }
success() { echo -e "\033[1;32m[SUCCESS]\033[0m $1"; }

info "🧹 Starting XORB repository cleanup..."

# Backup important files before cleanup
BACKUP_DIR="backups/pre_cleanup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
info "Creating backup in: $BACKUP_DIR"

# Remove old xorb_common directory (replaced by xorb_core)
if [ -d "xorb_common" ]; then
    warn "Removing old xorb_common directory..."
    mv xorb_common "$BACKUP_DIR/"
fi

# Remove duplicate docker-compose files (keep main one)
for file in docker-compose.*.yml; do
    if [[ "$file" != "docker-compose.yml" && "$file" != "docker-compose.production.yml" && "$file" != "docker-compose.monitoring.yml" ]]; then
        if [ -f "$file" ]; then
            warn "Moving duplicate compose file: $file"
            mv "$file" "$BACKUP_DIR/"
        fi
    fi
done

# Remove old/duplicate Makefiles
for file in Makefile_*; do
    if [ -f "$file" ]; then
        warn "Moving old Makefile: $file"
        mv "$file" "$BACKUP_DIR/"
    fi
done

# Clean up result/log files
info "Cleaning up result and log files..."
find . -name "campaign_results_*.json" -exec mv {} "$BACKUP_DIR/" \; 2>/dev/null || true
find . -name "*.log" -not -path "./logs/*" -exec mv {} "$BACKUP_DIR/" \; 2>/dev/null || true
find . -name "deployment_verification_*.json" -exec mv {} "$BACKUP_DIR/" \; 2>/dev/null || true

# Remove Python cache files
info "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Remove temporary files
info "Removing temporary files..."
find . -name "*.tmp" -delete 2>/dev/null || true
find . -name "*.swp" -delete 2>/dev/null || true
find . -name ".DS_Store" -delete 2>/dev/null || true

# Clean up old environment files
if [ -f ".env" ] && [ ! -f "config/.xorb.env" ]; then
    warn "Moving old .env file to backup"
    mv .env "$BACKUP_DIR/"
fi

# Remove duplicate/old documentation files
for file in README_*.md; do
    if [[ "$file" != "README_REFACTORED.md" ]]; then
        if [ -f "$file" ]; then
            warn "Moving old documentation: $file"
            mv "$file" "$BACKUP_DIR/"
        fi
    fi
done

# Clean up old data directories if empty
for dir in prometheus_data data/*/; do
    if [ -d "$dir" ] && [ -z "$(ls -A "$dir" 2>/dev/null)" ]; then
        info "Removing empty directory: $dir"
        rmdir "$dir" 2>/dev/null || true
    fi
done

# Set proper permissions on scripts
info "Setting proper permissions on scripts..."
find scripts/ -name "*.sh" -exec chmod +x {} \; 2>/dev/null || true
find scripts/ -name "*.py" -exec chmod +x {} \; 2>/dev/null || true

# Create .gitignore for organized repo
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
.venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/
*.log.*

# Data directories
data/
prometheus_data/
grafana_data/

# Environment files
.env
.env.local
.env.production

# Test coverage
htmlcov/
.coverage
.pytest_cache/

# Results and reports
campaign_results_*.json
deployment_verification_*.json
*.report

# Temporary files
*.tmp
*.temp

# Docker
.docker/

# Secrets
secrets/
certs/
*.pem
*.key
*.crt

# Backups
backups/
EOF

success "✅ Repository cleanup completed!"
info "Backup created in: $BACKUP_DIR"
info "Updated .gitignore for clean repository management"