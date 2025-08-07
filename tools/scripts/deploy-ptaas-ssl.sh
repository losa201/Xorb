#!/bin/bash

# PTaaS Platform SSL Deployment Script
# Deploys PTaaS platform with Cloudflare Origin SSL certificates

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
echo -e "${BLUE}║                     🛡️  PTaaS Platform SSL Deployment                       ║${NC}"
echo -e "${BLUE}║                     https://ptaas.verteidiq.com                             ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════════════╝${NC}"

# Check prerequisites
log_info "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    log_error "Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    log_error "Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

log_success "Docker and Docker Compose found"

# Check SSL certificates
log_info "Verifying SSL certificates..."

if [[ ! -f "./ssl/verteidiq.crt" ]]; then
    log_error "SSL certificate not found at ./ssl/verteidiq.crt"
    exit 1
fi

if [[ ! -f "./ssl/verteidiq.key" ]]; then
    log_error "SSL private key not found at ./ssl/verteidiq.key"
    exit 1
fi

# Verify certificate details
cert_info=$(openssl x509 -in ./ssl/verteidiq.crt -text -noout)
if echo "$cert_info" | grep -q "ptaas.verteidiq.com\|*.verteidiq.com"; then
    log_success "SSL certificate is valid for ptaas.verteidiq.com"
else
    log_warning "SSL certificate may not include ptaas.verteidiq.com"
fi

# Check certificate expiration
expiry_date=$(openssl x509 -in ./ssl/verteidiq.crt -enddate -noout | cut -d= -f2)
log_info "SSL certificate expires: $expiry_date"

# Set secure permissions on SSL files
chmod 600 ./ssl/verteidiq.key
chmod 644 ./ssl/verteidiq.crt
log_success "SSL file permissions secured"

# Validate nginx configuration
log_info "Validating nginx configuration..."

if docker run --rm -v "$(pwd)/legacy/config/nginx/nginx.conf:/etc/nginx/nginx.conf:ro" nginx:alpine nginx -t; then
    log_success "Nginx configuration is valid"
else
    log_error "Nginx configuration validation failed"
    exit 1
fi

# Set environment variables for production
export ENVIRONMENT=production
export VITE_API_URL=https://ptaas.verteidiq.com/api
export NODE_ENV=production

log_info "Environment configured for production deployment"

# Stop existing services
log_info "Stopping existing services..."
docker-compose down --remove-orphans || true

# Pull latest images
log_info "Pulling latest Docker images..."
docker-compose pull

# Build custom images
log_info "Building PTaaS frontend..."
docker-compose build ptaas-frontend

# Start services with SSL support
log_info "Starting services with SSL support..."
docker-compose up -d

# Wait for services to be ready
log_info "Waiting for services to start..."
sleep 20

# Health checks
log_info "Performing health checks..."

# Check nginx is running
if docker ps | grep -q "xorb-nginx"; then
    log_success "Nginx container is running"
else
    log_error "Nginx container failed to start"
    docker-compose logs nginx
    exit 1
fi

# Check PTaaS frontend is running
if docker ps | grep -q "xorb-ptaas-frontend"; then
    log_success "PTaaS frontend container is running"
else
    log_error "PTaaS frontend container failed to start"
    docker-compose logs ptaas-frontend
    exit 1
fi

# Test HTTP to HTTPS redirect
log_info "Testing HTTP to HTTPS redirect..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost/ | grep -q "301"; then
    log_success "HTTP to HTTPS redirect is working"
else
    log_warning "HTTP to HTTPS redirect may not be working properly"
fi

# Test HTTPS endpoint (if certificates are properly configured)
log_info "Testing HTTPS endpoint..."
if curl -k -s -o /dev/null -w "%{http_code}" https://localhost/ | grep -q "200"; then
    log_success "HTTPS endpoint is responding"
else
    log_warning "HTTPS endpoint may not be responding (expected if not accessible externally)"
fi

# Test API endpoint
log_info "Testing API endpoint..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health | grep -q "200"; then
    log_success "API endpoint is healthy"
else
    log_warning "API endpoint may not be responding"
fi

# Display service status
log_info "Service status:"
docker-compose ps

# Display logs summary
log_info "Recent nginx logs:"
docker-compose logs --tail=10 nginx

# Final deployment summary
echo ""
log_success "🎉 PTaaS Platform SSL deployment complete!"
echo ""
log_info "📋 Deployment Summary:"
log_info "   Domain: https://ptaas.verteidiq.com"
log_info "   SSL: Cloudflare Origin Certificate"
log_info "   Frontend: React SPA with Vite"
log_info "   Backend: FastAPI with clean architecture"
log_info "   Reverse Proxy: Nginx with SSL termination"
echo ""
log_info "🌐 Access Points:"
log_info "   Production: https://ptaas.verteidiq.com"
log_info "   Local Development: http://localhost"
log_info "   API Health Check: http://localhost:8000/health"
log_info "   Frontend Direct: http://localhost:3000"
echo ""
log_info "🔧 Management Commands:"
log_info "   View logs: docker-compose logs -f"
log_info "   Restart: docker-compose restart"
log_info "   Stop: docker-compose down"
log_info "   Update: ./scripts/deploy-ptaas-ssl.sh"
echo ""
log_info "📊 Monitoring:"
log_info "   Nginx logs: docker-compose logs nginx"
log_info "   Frontend logs: docker-compose logs ptaas-frontend"
log_info "   API logs: docker-compose logs xorb-api-gateway"
echo ""
log_info "🔒 Security Features:"
log_info "   - HTTPS with Cloudflare Origin SSL"
log_info "   - HTTP to HTTPS redirect"
log_info "   - Security headers (HSTS, XSS, etc.)"
log_info "   - CORS configuration for API access"
echo ""

# Check if running behind Cloudflare
if curl -s -I https://ptaas.verteidiq.com 2>/dev/null | grep -q "cloudflare"; then
    log_success "✅ Domain is properly configured with Cloudflare"
    log_info "   Cloudflare will handle:"
    log_info "   - DDoS protection"
    log_info "   - CDN and caching"
    log_info "   - SSL/TLS termination for clients"
    log_info "   - Origin certificate validation"
else
    log_warning "⚠️  Unable to verify Cloudflare configuration"
    log_info "   Ensure ptaas.verteidiq.com is properly configured in Cloudflare"
fi

echo ""
log_info "🚀 PTaaS Platform is ready for production use!"
log_info "   Monitor the deployment and ensure DNS is properly configured"

# Optional: Display certificate information
if command -v openssl &> /dev/null; then
    echo ""
    log_info "📋 SSL Certificate Information:"
    openssl x509 -in ./ssl/verteidiq.crt -subject -issuer -dates -noout | sed 's/^/   /'
fi