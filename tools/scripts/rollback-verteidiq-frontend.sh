#!/bin/bash

# =============================================================================
# Verteidiq.com PTaaS Frontend Rollback Script
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOMAIN="verteidiq.com"
FRONTEND_DIR="/root/Xorb/ptaas-frontend"
SERVICE_NAME="ptaas-frontend"
PRODUCTION_PORT="3005"
LOG_FILE="/root/Xorb/logs/rollback-$(date +%Y%m%d_%H%M%S).log"

# Ensure log directory exists
mkdir -p /root/Xorb/logs

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR $(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARN $(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO $(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
    fi
}

# Stop services
stop_services() {
    log "🛑 Stopping services..."
    
    # Stop PM2 processes
    pm2 stop "$SERVICE_NAME" 2>/dev/null || true
    pm2 delete "$SERVICE_NAME" 2>/dev/null || true
    
    # Stop systemd service
    systemctl stop "${SERVICE_NAME}.service" 2>/dev/null || true
    systemctl disable "${SERVICE_NAME}.service" 2>/dev/null || true
    
    # Kill any processes on production port
    if lsof -ti:$PRODUCTION_PORT > /dev/null 2>&1; then
        warn "Killing existing process on port $PRODUCTION_PORT"
        kill -9 $(lsof -ti:$PRODUCTION_PORT) 2>/dev/null || true
    fi
    
    info "✅ Services stopped"
}

# Remove configurations
remove_configurations() {
    log "🗑️ Removing configurations..."
    
    # Remove nginx site
    rm -f "/etc/nginx/sites-enabled/$DOMAIN"
    rm -f "/etc/nginx/sites-available/$DOMAIN"
    
    # Remove systemd service
    rm -f "/etc/systemd/system/${SERVICE_NAME}.service"
    systemctl daemon-reload
    
    # Remove monitoring cron job
    crontab -l 2>/dev/null | grep -v "monitor-frontend.sh" | crontab - 2>/dev/null || true
    
    # Remove monitoring script
    rm -f "/root/Xorb/scripts/monitor-frontend.sh"
    
    info "✅ Configurations removed"
}

# Clean up PM2
cleanup_pm2() {
    log "🧹 Cleaning up PM2..."
    
    # Remove PM2 ecosystem file
    rm -f "$FRONTEND_DIR/ecosystem.config.js"
    
    # Reset PM2
    pm2 kill 2>/dev/null || true
    
    info "✅ PM2 cleaned up"
}

# Restore nginx default
restore_nginx_default() {
    log "🔄 Restoring nginx default..."
    
    # Create a simple default page
    mkdir -p /var/www/html
    cat > /var/www/html/index.html <<EOF
<!DOCTYPE html>
<html>
<head>
    <title>Service Temporarily Unavailable</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
        .container { max-width: 600px; margin: 0 auto; }
        .status { color: #e74c3c; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="status">Service Temporarily Unavailable</h1>
        <p>The PTaaS frontend service has been rolled back and is currently unavailable.</p>
        <p>Please contact your system administrator for more information.</p>
        <small>Rollback completed at $(date)</small>
    </div>
</body>
</html>
EOF
    
    # Create basic nginx config
    cat > /etc/nginx/sites-available/default <<EOF
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    
    root /var/www/html;
    index index.html index.htm index.nginx-debian.html;
    
    server_name _;
    
    location / {
        try_files \$uri \$uri/ =404;
    }
}
EOF
    
    ln -sf /etc/nginx/sites-available/default /etc/nginx/sites-enabled/default
    
    # Test and reload nginx
    nginx -t && systemctl reload nginx
    
    info "✅ Nginx default restored"
}

# Generate rollback report
generate_report() {
    log "📋 Generating rollback report..."
    
    REPORT_FILE="/root/Xorb/logs/verteidiq-rollback-report-$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$REPORT_FILE" <<EOF
{
  "rollback": {
    "timestamp": "$(date -Iseconds)",
    "domain": "$DOMAIN",
    "service_name": "$SERVICE_NAME",
    "actions_performed": [
      "stopped_pm2_processes",
      "removed_nginx_configuration",
      "removed_systemd_service",
      "cleaned_pm2_ecosystem",
      "removed_monitoring",
      "restored_nginx_default"
    ]
  },
  "status": {
    "nginx": "$(systemctl is-active nginx)",
    "pm2_processes": "$(pm2 list --porcelain | wc -l)",
    "port_${PRODUCTION_PORT}_free": "$(! netstat -tuln | grep -q ":$PRODUCTION_PORT " && echo "true" || echo "false")"
  },
  "logs": {
    "rollback_log": "$LOG_FILE"
  }
}
EOF
    
    log "📋 Rollback report saved to: $REPORT_FILE"
}

# Display final status
display_status() {
    echo -e "\n${YELLOW}⏪ Rollback Completed!${NC}\n"
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo -e "${BLUE}PTaaS Frontend Rollback Status${NC}"
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo -e "📍 Domain: ${YELLOW}$DOMAIN${NC}"
    echo -e "🛑 Status: ${RED}Offline${NC}"
    echo -e "🔧 PM2: ${YELLOW}Cleaned${NC}"
    echo -e "🌐 Nginx: ${GREEN}Default restored${NC}"
    echo -e "📊 Monitoring: ${YELLOW}Removed${NC}"
    echo -e "\n${GREEN}Actions taken:${NC}"
    echo -e "• Stopped all PTaaS frontend services"
    echo -e "• Removed nginx configuration"
    echo -e "• Cleaned up PM2 processes"
    echo -e "• Removed monitoring scripts"
    echo -e "• Restored nginx default page"
    echo -e "\n${YELLOW}To redeploy:${NC}"
    echo -e "• Run: /root/Xorb/scripts/deploy-verteidiq-frontend.sh"
    echo -e "\n${GREEN}Rollback completed at $(date)${NC}\n"
}

# Main rollback function
main() {
    log "⏪ Starting Verteidiq.com PTaaS Frontend Rollback"
    log "================================================="
    
    check_root
    stop_services
    remove_configurations
    cleanup_pm2
    restore_nginx_default
    generate_report
    display_status
    
    log "✅ Rollback completed successfully!"
}

# Handle script interruption
trap 'error "Rollback interrupted"' INT TERM

# Run main function
main "$@"