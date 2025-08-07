#!/bin/bash

# =============================================================================
# Verteidiq.com PTaaS Frontend Deployment Script
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
NGINX_CONFIG="/root/Xorb/infra/nginx/verteidiq.conf"
SSL_CERT="/root/Xorb/ssl/verteidiq.crt"
SSL_KEY="/root/Xorb/ssl/verteidiq.key"
PRODUCTION_PORT="3005"
SERVICE_NAME="ptaas-frontend"
LOG_FILE="/root/Xorb/logs/deployment-$(date +%Y%m%d_%H%M%S).log"

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

# Check prerequisites
check_prerequisites() {
    log "🔍 Checking prerequisites..."
    
    # Check if Node.js is installed
    if ! command -v node &> /dev/null; then
        error "Node.js is not installed. Please install Node.js first."
    fi
    
    # Check if npm is installed
    if ! command -v npm &> /dev/null; then
        error "npm is not installed. Please install npm first."
    fi
    
    # Check if nginx is installed
    if ! command -v nginx &> /dev/null; then
        warn "Nginx is not installed. Installing nginx..."
        apt update && apt install -y nginx
    fi
    
    # Check if PM2 is installed globally
    if ! command -v pm2 &> /dev/null; then
        log "Installing PM2 process manager..."
        npm install -g pm2
    fi
    
    # Check frontend directory exists
    if [[ ! -d "$FRONTEND_DIR" ]]; then
        error "Frontend directory not found: $FRONTEND_DIR"
    fi
    
    # Check SSL certificates exist
    if [[ ! -f "$SSL_CERT" ]] || [[ ! -f "$SSL_KEY" ]]; then
        error "SSL certificates not found. Expected: $SSL_CERT and $SSL_KEY"
    fi
    
    info "✅ All prerequisites met"
}

# Stop existing services
stop_existing_services() {
    log "🛑 Stopping existing services..."
    
    # Stop PM2 processes
    pm2 stop "$SERVICE_NAME" 2>/dev/null || true
    pm2 delete "$SERVICE_NAME" 2>/dev/null || true
    
    # Kill any processes on production port
    if lsof -ti:$PRODUCTION_PORT > /dev/null 2>&1; then
        warn "Killing existing process on port $PRODUCTION_PORT"
        kill -9 $(lsof -ti:$PRODUCTION_PORT) 2>/dev/null || true
    fi
    
    info "✅ Services stopped"
}

# Build frontend application
build_frontend() {
    log "🏗️ Building frontend application..."
    
    cd "$FRONTEND_DIR"
    
    # Clean install with legacy peer deps
    info "Installing dependencies..."
    rm -rf node_modules package-lock.json
    npm install --legacy-peer-deps
    
    # Build for production
    info "Building for production..."
    npm run build
    
    # Check if build was successful
    if [[ ! -d ".next" ]]; then
        error "Build failed - .next directory not found"
    fi
    
    info "✅ Frontend built successfully"
}

# Configure environment
configure_environment() {
    log "⚙️ Configuring environment..."
    
    cd "$FRONTEND_DIR"
    
    # Create or update .env.production
    cat > .env.production <<EOF
NODE_ENV=production
PORT=$PRODUCTION_PORT
HOSTNAME=0.0.0.0
NEXT_TELEMETRY_DISABLED=1

# API Configuration
API_BASE_URL=https://$DOMAIN/api
XORB_API_URL=https://$DOMAIN/api/v1

# Security Configuration
NEXTAUTH_URL=https://$DOMAIN
NEXTAUTH_SECRET=$(openssl rand -base64 32)

# Performance Configuration
NEXT_PUBLIC_APP_ENV=production
NEXT_PUBLIC_DOMAIN=$DOMAIN
EOF
    
    # Create PM2 ecosystem file
    cat > ecosystem.config.js <<EOF
module.exports = {
  apps: [{
    name: '$SERVICE_NAME',
    script: 'npm',
    args: 'start',
    cwd: '$FRONTEND_DIR',
    instances: 1,
    exec_mode: 'fork',
    env: {
      NODE_ENV: 'production',
      PORT: $PRODUCTION_PORT,
      HOSTNAME: '0.0.0.0'
    },
    error_file: '/root/Xorb/logs/${SERVICE_NAME}-error.log',
    out_file: '/root/Xorb/logs/${SERVICE_NAME}-out.log',
    log_file: '/root/Xorb/logs/${SERVICE_NAME}.log',
    time: true,
    max_memory_restart: '1G',
    node_args: '--max_old_space_size=1024'
  }]
};
EOF
    
    info "✅ Environment configured"
}

# Configure Nginx
configure_nginx() {
    log "🌐 Configuring Nginx..."
    
    # Backup existing nginx config
    if [[ -f "/etc/nginx/sites-available/$DOMAIN" ]]; then
        cp "/etc/nginx/sites-available/$DOMAIN" "/etc/nginx/sites-available/$DOMAIN.backup.$(date +%s)"
    fi
    
    # Add rate limiting to nginx.conf if not present
    if ! grep -q "limit_req_zone.*frontend" /etc/nginx/nginx.conf; then
        sed -i '/http {/a\\tlimit_req_zone $binary_remote_addr zone=frontend:10m rate=30r/s;' /etc/nginx/nginx.conf
    fi
    
    # Copy nginx configuration
    cp "$NGINX_CONFIG" "/etc/nginx/sites-available/$DOMAIN"
    
    # Enable site
    ln -sf "/etc/nginx/sites-available/$DOMAIN" "/etc/nginx/sites-enabled/$DOMAIN"
    
    # Remove default site if it exists
    rm -f /etc/nginx/sites-enabled/default
    
    # Test nginx configuration
    if ! nginx -t; then
        error "Nginx configuration test failed"
    fi
    
    info "✅ Nginx configured"
}

# Start services
start_services() {
    log "🚀 Starting services..."
    
    cd "$FRONTEND_DIR"
    
    # Start application with PM2
    pm2 start ecosystem.config.js
    
    # Save PM2 configuration
    pm2 save
    
    # Setup PM2 startup script
    pm2 startup systemd -u root --hp /root
    
    # Restart nginx
    systemctl restart nginx
    systemctl enable nginx
    
    # Wait for application to start
    info "Waiting for application to start..."
    sleep 5
    
    # Check if application is running
    if ! pm2 show "$SERVICE_NAME" | grep -q "online"; then
        error "Application failed to start"
    fi
    
    info "✅ Services started"
}

# Setup SSL and security
setup_ssl() {
    log "🔒 Setting up SSL and security..."
    
    # Ensure SSL certificates have correct permissions
    chmod 644 "$SSL_CERT"
    chmod 600 "$SSL_KEY"
    chown root:root "$SSL_CERT" "$SSL_KEY"
    
    # Setup firewall rules
    if command -v ufw &> /dev/null; then
        ufw allow 80/tcp
        ufw allow 443/tcp
        ufw allow 22/tcp
        ufw --force enable
    fi
    
    info "✅ SSL and security configured"
}

# Health checks
perform_health_checks() {
    log "🏥 Performing health checks..."
    
    # Check if port is listening
    if ! netstat -tuln | grep -q ":$PRODUCTION_PORT "; then
        error "Application not listening on port $PRODUCTION_PORT"
    fi
    
    # Check nginx status
    if ! systemctl is-active --quiet nginx; then
        error "Nginx is not running"
    fi
    
    # Check PM2 process
    if ! pm2 show "$SERVICE_NAME" | grep -q "online"; then
        error "PM2 process is not online"
    fi
    
    # Test HTTP redirect
    info "Testing HTTP redirect..."
    if ! curl -s -o /dev/null -w "%{http_code}" "http://$DOMAIN" | grep -q "301"; then
        warn "HTTP redirect test failed"
    fi
    
    # Test HTTPS
    info "Testing HTTPS connection..."
    sleep 2
    if ! curl -k -s -o /dev/null "https://$DOMAIN/health"; then
        warn "HTTPS health check failed"
    fi
    
    info "✅ Health checks completed"
}

# Setup monitoring
setup_monitoring() {
    log "📊 Setting up monitoring..."
    
    # Create monitoring script
    cat > /root/Xorb/scripts/monitor-frontend.sh <<EOF
#!/bin/bash
# Frontend monitoring script

SERVICE_NAME="$SERVICE_NAME"
DOMAIN="$DOMAIN"
PORT="$PRODUCTION_PORT"

# Check PM2 process
if ! pm2 show \$SERVICE_NAME | grep -q "online"; then
    echo "WARNING: \$SERVICE_NAME is not running" | logger -t frontend-monitor
    pm2 restart \$SERVICE_NAME
fi

# Check port
if ! netstat -tuln | grep -q ":\$PORT "; then
    echo "WARNING: Port \$PORT is not listening" | logger -t frontend-monitor
fi

# Check SSL certificate expiry
if openssl x509 -in "$SSL_CERT" -noout -checkend 604800; then
    echo "SSL certificate expires within 7 days" | logger -t frontend-monitor
fi
EOF
    
    chmod +x /root/Xorb/scripts/monitor-frontend.sh
    
    # Add cron job for monitoring
    (crontab -l 2>/dev/null; echo "*/5 * * * * /root/Xorb/scripts/monitor-frontend.sh") | crontab -
    
    info "✅ Monitoring configured"
}

# Create systemd service as backup
create_systemd_service() {
    log "🔧 Creating systemd service..."
    
    cat > /etc/systemd/system/${SERVICE_NAME}.service <<EOF
[Unit]
Description=PTaaS Frontend Application
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$FRONTEND_DIR
ExecStart=/usr/bin/npm start
Restart=always
RestartSec=10
Environment=NODE_ENV=production
Environment=PORT=$PRODUCTION_PORT

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    systemctl enable ${SERVICE_NAME}.service
    
    info "✅ Systemd service created"
}

# Generate deployment report
generate_report() {
    log "📋 Generating deployment report..."
    
    REPORT_FILE="/root/Xorb/logs/verteidiq-deployment-report-$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$REPORT_FILE" <<EOF
{
  "deployment": {
    "timestamp": "$(date -Iseconds)",
    "domain": "$DOMAIN",
    "frontend_directory": "$FRONTEND_DIR",
    "production_port": "$PRODUCTION_PORT",
    "service_name": "$SERVICE_NAME",
    "ssl_configured": true,
    "nginx_configured": true,
    "pm2_configured": true,
    "monitoring_enabled": true
  },
  "services": {
    "nginx": "$(systemctl is-active nginx)",
    "pm2_process": "$(pm2 show $SERVICE_NAME --porcelain | head -n1 | cut -d'|' -f13)",
    "ssl_expires": "$(openssl x509 -in $SSL_CERT -noout -enddate | cut -d= -f2)"
  },
  "urls": {
    "production": "https://$DOMAIN",
    "health_check": "https://$DOMAIN/health",
    "api_base": "https://$DOMAIN/api"
  },
  "logs": {
    "deployment_log": "$LOG_FILE",
    "application_log": "/root/Xorb/logs/${SERVICE_NAME}.log",
    "error_log": "/root/Xorb/logs/${SERVICE_NAME}-error.log"
  }
}
EOF
    
    log "📋 Deployment report saved to: $REPORT_FILE"
}

# Display final status
display_status() {
    echo -e "\n${GREEN}🎉 Deployment Completed Successfully!${NC}\n"
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo -e "${BLUE}PTaaS Frontend Deployment Status${NC}"
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo -e "📍 Domain: ${GREEN}https://$DOMAIN${NC}"
    echo -e "🚀 Status: ${GREEN}Online${NC}"
    echo -e "🔒 SSL: ${GREEN}Enabled${NC}"
    echo -e "⚡ Port: ${GREEN}$PRODUCTION_PORT${NC}"
    echo -e "🔧 Process Manager: ${GREEN}PM2${NC}"
    echo -e "🌐 Web Server: ${GREEN}Nginx${NC}"
    echo -e "📊 Monitoring: ${GREEN}Enabled${NC}"
    echo -e "\n${YELLOW}Next Steps:${NC}"
    echo -e "• Visit https://$DOMAIN to access your PTaaS platform"
    echo -e "• Monitor logs: pm2 logs $SERVICE_NAME"
    echo -e "• Check status: pm2 status"
    echo -e "• View deployment log: $LOG_FILE"
    echo -e "\n${GREEN}Deployment completed at $(date)${NC}\n"
}

# Main deployment function
main() {
    log "🚀 Starting Verteidiq.com PTaaS Frontend Deployment"
    log "=================================================="
    
    check_root
    check_prerequisites
    stop_existing_services
    build_frontend
    configure_environment
    configure_nginx
    setup_ssl
    start_services
    setup_monitoring
    create_systemd_service
    perform_health_checks
    generate_report
    display_status
    
    log "🎉 Deployment completed successfully!"
}

# Handle script interruption
trap 'error "Deployment interrupted"' INT TERM

# Run main function
main "$@"