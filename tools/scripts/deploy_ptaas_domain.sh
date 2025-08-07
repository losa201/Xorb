#!/bin/bash

# XORB PTaaS Domain Deployment Script
# Domain: ptaas.verteidiq.com
# This script automates the deployment of XORB PTaaS platform on the specified domain

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOMAIN="ptaas.verteidiq.com"
ADMIN_DOMAIN="admin.ptaas.verteidiq.com"
PROJECT_DIR="/root/Xorb"
SSL_DIR="${PROJECT_DIR}/ssl"
NGINX_CONFIG_DIR="${PROJECT_DIR}/nginx"
WEB_ROOT="/var/www"
COMPOSE_FILE="${PROJECT_DIR}/infra/docker-compose-ptaas.yml"
ENV_FILE="${PROJECT_DIR}/infra/.env.ptaas.verteidiq"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if nginx is available
    if ! command -v nginx &> /dev/null; then
        warning "Nginx is not installed. Installing nginx..."
        apt-get update
        apt-get install -y nginx apache2-utils
    fi
    
    success "Prerequisites check completed"
}

# Create web directories
setup_web_directories() {
    log "Setting up web directories..."
    
    # Create main domain directory
    mkdir -p "${WEB_ROOT}/${DOMAIN}"
    mkdir -p "${WEB_ROOT}/${ADMIN_DOMAIN}"
    
    # Set proper ownership
    chown -R www-data:www-data "${WEB_ROOT}/${DOMAIN}"
    chown -R www-data:www-data "${WEB_ROOT}/${ADMIN_DOMAIN}"
    
    # Create basic index files
    cat > "${WEB_ROOT}/${DOMAIN}/index.html" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XORB PTaaS Platform</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 40px; 
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .container {
            text-align: center;
            max-width: 600px;
        }
        h1 { 
            font-size: 3em; 
            margin-bottom: 0.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        p { 
            font-size: 1.2em; 
            margin-bottom: 2em;
            opacity: 0.9;
        }
        .status {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            margin-top: 2em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ XORB PTaaS Platform</h1>
        <p>Enterprise Penetration Testing as a Service</p>
        <div class="status">
            <p>Platform is initializing...</p>
            <p>AI-Enhanced Security Testing • German Legal Compliance • Real-Time Threat Intelligence</p>
        </div>
    </div>
</body>
</html>
EOF

    cat > "${WEB_ROOT}/${ADMIN_DOMAIN}/index.html" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XORB PTaaS Admin</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 40px; 
            background: linear-gradient(135deg, #2c1810 0%, #8b4513 100%);
            color: white;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .container {
            text-align: center;
            max-width: 600px;
        }
        h1 { 
            font-size: 2.5em; 
            margin-bottom: 0.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .warning {
            background: rgba(255,0,0,0.2);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid rgba(255,0,0,0.3);
            margin-top: 2em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔐 XORB PTaaS Admin</h1>
        <div class="warning">
            <h3>⚠️ Administrative Access Required</h3>
            <p>This interface requires proper authentication and authorization.</p>
            <p>Unauthorized access is strictly prohibited and logged.</p>
        </div>
    </div>
</body>
</html>
EOF
    
    success "Web directories created and configured"
}

# Setup SSL certificates
setup_ssl() {
    log "Setting up SSL certificates..."
    
    if [[ -f "${SSL_DIR}/verteidiq.com.crt" && -f "${SSL_DIR}/verteidiq.com.key" ]]; then
        success "SSL certificates already exist"
        
        # Verify certificate
        if openssl x509 -in "${SSL_DIR}/verteidiq.com.crt" -text -noout | grep -q "${DOMAIN}"; then
            success "SSL certificate is valid for ${DOMAIN}"
        else
            warning "SSL certificate may not be valid for ${DOMAIN}"
        fi
    else
        warning "SSL certificates not found. Please ensure certificates are available at:"
        echo "  ${SSL_DIR}/verteidiq.com.crt"
        echo "  ${SSL_DIR}/verteidiq.com.key"
        echo ""
        echo "You can:"
        echo "1. Use existing wildcard certificate for *.verteidiq.com"
        echo "2. Generate Let's Encrypt certificate"
        echo "3. Use CloudFlare Origin certificate"
        echo ""
        echo "See /root/Xorb/config/ptaas_domain_setup.md for detailed instructions"
        exit 1
    fi
}

# Setup authentication
setup_authentication() {
    log "Setting up authentication files..."
    
    # Create nginx directory if it doesn't exist
    mkdir -p /etc/nginx
    
    # Create password file for PTaaS metrics if it doesn't exist
    if [[ ! -f "/etc/nginx/.htpasswd_ptaas" ]]; then
        log "Creating PTaaS metrics authentication file..."
        echo "Enter password for PTaaS metrics access:"
        htpasswd -c /etc/nginx/.htpasswd_ptaas ptaas_metrics
        chmod 600 /etc/nginx/.htpasswd_ptaas
    fi
    
    # Create password file for PTaaS admin if it doesn't exist
    if [[ ! -f "/etc/nginx/.htpasswd_ptaas_admin" ]]; then
        log "Creating PTaaS admin authentication file..."
        echo "Enter password for PTaaS admin access:"
        htpasswd -c /etc/nginx/.htpasswd_ptaas_admin ptaas_admin
        chmod 600 /etc/nginx/.htpasswd_ptaas_admin
    fi
    
    success "Authentication setup completed"
}

# Validate nginx configuration
validate_nginx_config() {
    log "Validating nginx configuration..."
    
    if nginx -t; then
        success "Nginx configuration is valid"
    else
        error "Nginx configuration is invalid"
        return 1
    fi
}

# Deploy Docker services
deploy_services() {
    log "Deploying PTaaS Docker services..."
    
    cd "${PROJECT_DIR}/infra"
    
    # Check if environment file exists
    if [[ ! -f "${ENV_FILE}" ]]; then
        error "Environment file not found: ${ENV_FILE}"
        exit 1
    fi
    
    # Copy environment file
    cp "${ENV_FILE}" .env
    
    log "Starting PTaaS services..."
    docker-compose -f docker-compose-ptaas.yml up -d
    
    # Wait for services to start
    sleep 30
    
    # Check service health
    log "Checking service health..."
    local services_healthy=true
    
    if ! curl -f http://localhost:8080/health &>/dev/null; then
        error "PTaaS Core service health check failed"
        services_healthy=false
    fi
    
    if ! curl -f http://localhost:8081/api/v1/health &>/dev/null; then
        error "Researcher API service health check failed"
        services_healthy=false
    fi
    
    if ! curl -f http://localhost:8082/api/v1/health &>/dev/null; then
        error "Company API service health check failed"
        services_healthy=false
    fi
    
    if [[ "$services_healthy" == true ]]; then
        success "All PTaaS services are healthy"
    else
        warning "Some services may not be fully ready. Check logs with: docker-compose -f docker-compose-ptaas.yml logs"
    fi
}

# Test external access
test_external_access() {
    log "Testing external access..."
    
    # Test HTTPS access
    if curl -f -k "https://${DOMAIN}/health" &>/dev/null; then
        success "HTTPS access to ${DOMAIN} is working"
    else
        warning "HTTPS access to ${DOMAIN} may not be working yet"
    fi
    
    # Test admin domain
    if curl -f -k "https://${ADMIN_DOMAIN}/" &>/dev/null; then
        success "HTTPS access to ${ADMIN_DOMAIN} is working"
    else
        warning "HTTPS access to ${ADMIN_DOMAIN} may not be working yet"
    fi
}

# Setup firewall rules
setup_firewall() {
    log "Setting up firewall rules..."
    
    if command -v ufw &> /dev/null; then
        # Allow HTTP and HTTPS
        ufw allow 80/tcp comment "HTTP for PTaaS"
        ufw allow 443/tcp comment "HTTPS for PTaaS"
        
        # Enable firewall if not already enabled
        if ! ufw status | grep -q "Status: active"; then
            echo "y" | ufw enable
        fi
        
        success "Firewall rules configured"
    else
        warning "UFW not installed, skipping firewall configuration"
    fi
}

# Generate deployment report
generate_report() {
    log "Generating deployment report..."
    
    local report_file="${PROJECT_DIR}/logs/ptaas_deployment_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$report_file" << EOF
{
  "deployment_timestamp": "$(date -Iseconds)",
  "domain": "${DOMAIN}",
  "admin_domain": "${ADMIN_DOMAIN}",
  "ssl_certificates": {
    "cert_file": "${SSL_DIR}/verteidiq.com.crt",
    "key_file": "${SSL_DIR}/verteidiq.com.key",
    "exists": $([ -f "${SSL_DIR}/verteidiq.com.crt" ] && echo "true" || echo "false")
  },
  "services": {
    "ptaas_core": "$(curl -s http://localhost:8080/health || echo 'unavailable')",
    "researcher_api": "$(curl -s http://localhost:8081/api/v1/health || echo 'unavailable')",
    "company_api": "$(curl -s http://localhost:8082/api/v1/health || echo 'unavailable')"
  },
  "docker_containers": $(docker ps --format "table {{.Names}}\t{{.Status}}" | grep ptaas | wc -l),
  "deployment_status": "completed",
  "access_urls": {
    "main_platform": "https://${DOMAIN}",
    "admin_interface": "https://${ADMIN_DOMAIN}",
    "api_docs": "https://${DOMAIN}/api/docs",
    "grafana": "https://${DOMAIN}/grafana/",
    "metrics": "https://${DOMAIN}/metrics/"
  }
}
EOF
    
    success "Deployment report saved to: $report_file"
}

# Main deployment function
main() {
    echo
    echo "🚀 XORB PTaaS Domain Deployment"
    echo "==============================="
    echo "Domain: ${DOMAIN}"
    echo "Admin: ${ADMIN_DOMAIN}"
    echo
    
    check_root
    check_prerequisites
    setup_web_directories
    setup_ssl
    setup_authentication
    validate_nginx_config
    deploy_services
    test_external_access
    setup_firewall
    generate_report
    
    echo
    success "🎉 PTaaS deployment completed successfully!"
    echo
    echo "Access your PTaaS platform at:"
    echo "  Main Platform: https://${DOMAIN}"
    echo "  Admin Interface: https://${ADMIN_DOMAIN}"
    echo "  API Documentation: https://${DOMAIN}/api/docs"
    echo "  Grafana Dashboard: https://${DOMAIN}/grafana/"
    echo "  System Metrics: https://${DOMAIN}/metrics/"
    echo
    echo "Next steps:"
    echo "1. Verify DNS propagation"
    echo "2. Test all endpoints"
    echo "3. Configure monitoring alerts"
    echo "4. Set up backup procedures"
    echo "5. Review security settings"
    echo
    echo "For troubleshooting, see: /root/Xorb/config/ptaas_domain_setup.md"
    echo
}

# Run main function
main "$@"