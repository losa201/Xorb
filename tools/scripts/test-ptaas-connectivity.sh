#!/bin/bash

# PTaaS Frontend Connectivity Test Script
# Tests connectivity to ptaas.verteidiq.com and local services

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                       🌐 PTaaS Connectivity Test                            ║${NC}"
echo -e "${BLUE}║                   Testing ptaas.verteidiq.com access                        ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════════════╝${NC}"

# Test external domain connectivity
log_info "Testing external domain connectivity..."

# Test DNS resolution
if nslookup ptaas.verteidiq.com >/dev/null 2>&1; then
    log_success "DNS resolution: ptaas.verteidiq.com resolves correctly"
else
    log_error "DNS resolution: ptaas.verteidiq.com failed to resolve"
fi

# Test HTTPS connectivity (port 443)
log_info "Testing HTTPS connectivity to ptaas.verteidiq.com:443..."
if timeout 10 bash -c "</dev/tcp/ptaas.verteidiq.com/443" 2>/dev/null; then
    log_success "Port 443: HTTPS port is accessible"
else
    log_error "Port 443: HTTPS port is not accessible"
fi

# Test HTTP connectivity (port 80)
log_info "Testing HTTP connectivity to ptaas.verteidiq.com:80..."
if timeout 10 bash -c "</dev/tcp/ptaas.verteidiq.com/80" 2>/dev/null; then
    log_success "Port 80: HTTP port is accessible"
else
    log_warning "Port 80: HTTP port may not be accessible (common for HTTPS-only sites)"
fi

# Test HTTPS request
log_info "Testing HTTPS request to ptaas.verteidiq.com..."
if curl -k -s --connect-timeout 10 --max-time 30 -I https://ptaas.verteidiq.com >/dev/null 2>&1; then
    log_success "HTTPS request: Successfully connected to ptaas.verteidiq.com"

    # Get response details
    response=$(curl -k -s --connect-timeout 10 --max-time 30 -I https://ptaas.verteidiq.com | head -1)
    log_info "Response: $response"
else
    log_error "HTTPS request: Failed to connect to ptaas.verteidiq.com"
fi

# Test API endpoint if accessible
log_info "Testing API endpoint access..."
if curl -k -s --connect-timeout 10 --max-time 30 -I https://ptaas.verteidiq.com/api >/dev/null 2>&1; then
    log_success "API endpoint: https://ptaas.verteidiq.com/api is accessible"
else
    log_warning "API endpoint: https://ptaas.verteidiq.com/api may not be accessible or may require authentication"
fi

# Test local Docker network connectivity
echo ""
log_info "Testing local Docker network configuration..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    log_error "Docker is not running"
    exit 1
fi

# Check if xorb network exists
if docker network ls | grep -q "xorb-net"; then
    log_success "Docker network: xorb-net exists"
else
    log_warning "Docker network: xorb-net does not exist - run docker-compose up first"
fi

# Test container connectivity (if containers are running)
log_info "Testing container connectivity..."

containers=(
    "xorb-ptaas-frontend:80"
    "xorb-nginx:80"
    "xorb-api-gateway:8000"
)

for container_port in "${containers[@]}"; do
    container=$(echo $container_port | cut -d: -f1)
    port=$(echo $container_port | cut -d: -f2)

    if docker ps | grep -q "$container"; then
        if docker exec "$container" sh -c "nc -z localhost $port" 2>/dev/null; then
            log_success "Container $container is running and port $port is accessible"
        else
            log_warning "Container $container is running but port $port is not accessible"
        fi
    else
        log_warning "Container $container is not running"
    fi
done

# Test local port accessibility
echo ""
log_info "Testing local port accessibility..."

local_ports=(
    "80:HTTP (nginx)"
    "443:HTTPS (nginx)"
    "3000:PTaaS Frontend"
    "8000:API Gateway"
)

for port_desc in "${local_ports[@]}"; do
    port=$(echo $port_desc | cut -d: -f1)
    desc=$(echo $port_desc | cut -d: -f2)

    if nc -z localhost $port 2>/dev/null; then
        log_success "Port $port ($desc) is accessible locally"
    else
        log_warning "Port $port ($desc) is not accessible locally"
    fi
done

# Test firewall configuration
echo ""
log_info "Testing firewall configuration..."

if command -v ufw >/dev/null 2>&1; then
    ufw_status=$(ufw status | grep Status | awk '{print $2}')
    log_info "UFW status: $ufw_status"

    if ufw status | grep -q "80/tcp"; then
        log_success "Firewall: Port 80 is allowed"
    else
        log_warning "Firewall: Port 80 may not be explicitly allowed"
    fi

    if ufw status | grep -q "443/tcp"; then
        log_success "Firewall: Port 443 is allowed"
    else
        log_warning "Firewall: Port 443 may not be explicitly allowed"
    fi
else
    log_info "UFW is not installed - using system default firewall"
fi

# Test nginx configuration
echo ""
log_info "Testing nginx configuration..."

if docker ps | grep -q "xorb-nginx"; then
    if docker exec xorb-nginx nginx -t 2>/dev/null; then
        log_success "Nginx configuration is valid"
    else
        log_error "Nginx configuration has errors"
    fi
else
    log_warning "Nginx container is not running"
fi

# Final summary
echo ""
log_info "🔍 Connectivity Test Complete"
echo ""
log_info "📋 Summary:"
log_info "   External Domain: ptaas.verteidiq.com"
log_info "   Local Frontend: http://localhost:3000"
log_info "   Main Nginx: http://localhost:80"
log_info "   API Gateway: http://localhost:8000"
echo ""
log_info "🚀 To start the services:"
log_info "   docker-compose up -d"
echo ""
log_info "🌐 To access PTaaS frontend:"
log_info "   http://localhost (will proxy to PTaaS frontend)"
log_info "   http://localhost:3000 (direct PTaaS frontend access)"
echo ""
log_info "🔧 Configuration files:"
log_info "   Frontend config: /root/Xorb/ptaas-frontend/src/config.ts"
log_info "   Nginx config: /root/Xorb/legacy/config/nginx/nginx.conf"
log_info "   Docker Compose: /root/Xorb/docker-compose.yml"
