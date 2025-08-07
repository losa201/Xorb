#!/bin/bash

# XORB Telemetry Authentication Credentials Generator
# Generates secure passwords and API keys for telemetry access

set -euo pipefail

# Configuration
AUTH_DIR="/root/Xorb/auth"
BCRYPT_ROUNDS=12

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

# Generate secure random password
generate_password() {
    local length=${1:-32}
    openssl rand -base64 ${length} | tr -d "=+/" | cut -c1-${length}
}

# Generate bcrypt hash
generate_bcrypt_hash() {
    local password=$1
    python3 -c "
import bcrypt
password = '${password}'.encode('utf-8')
hashed = bcrypt.hashpw(password, bcrypt.gensalt(rounds=${BCRYPT_ROUNDS}))
print(hashed.decode('utf-8'))
"
}

# Create authentication directory
create_auth_directory() {
    log_info "Creating authentication directory..."
    mkdir -p "${AUTH_DIR}"
    chmod 700 "${AUTH_DIR}"
    log_success "Authentication directory created: ${AUTH_DIR}"
}

# Generate credentials
generate_credentials() {
    log_info "Generating authentication credentials..."
    
    # Dashboard credentials
    local dashboard_admin_pass=$(generate_password 24)
    local dashboard_user_pass=$(generate_password 24)
    
    # Metrics credentials
    local metrics_admin_pass=$(generate_password 24)
    local metrics_user_pass=$(generate_password 24)
    
    # API Keys
    local neural_api_key="xorb_neural_$(generate_password 32)"
    local learning_api_key="xorb_learning_$(generate_password 32)"
    
    # Database passwords
    local postgres_password=$(generate_password 24)
    local redis_password=$(generate_password 24)
    local grafana_secret_key=$(generate_password 48)
    
    # Generate bcrypt hashes for Caddy Basic Auth
    local dashboard_admin_hash=$(generate_bcrypt_hash "${dashboard_admin_pass}")
    local dashboard_user_hash=$(generate_bcrypt_hash "${dashboard_user_pass}")
    local metrics_admin_hash=$(generate_bcrypt_hash "${metrics_admin_pass}")
    local metrics_user_hash=$(generate_bcrypt_hash "${metrics_user_pass}")
    
    # Create credentials file
    cat > "${AUTH_DIR}/credentials.txt" <<EOF
# XORB Telemetry Authentication Credentials
# Generated: $(date)
# 
# IMPORTANT: Store these credentials securely and do not commit to version control!

# Dashboard Access (Grafana)
DASHBOARD_ADMIN_USER=admin
DASHBOARD_ADMIN_PASS=${dashboard_admin_pass}
DASHBOARD_USER_USER=xorb_user
DASHBOARD_USER_PASS=${dashboard_user_pass}

# Metrics Access (Prometheus)
METRICS_ADMIN_USER=admin
METRICS_ADMIN_PASS=${metrics_admin_pass}
METRICS_USER_USER=metrics_user
METRICS_USER_PASS=${metrics_user_pass}

# API Keys
NEURAL_API_KEY=${neural_api_key}
LEARNING_API_KEY=${learning_api_key}

# Database Credentials
POSTGRES_PASSWORD=${postgres_password}
REDIS_PASSWORD=${redis_password}
GRAFANA_SECRET_KEY=${grafana_secret_key}

# Bcrypt Hashes for Caddy (use these in Caddyfile)
DASHBOARD_ADMIN_HASH=${dashboard_admin_hash}
DASHBOARD_USER_HASH=${dashboard_user_hash}
METRICS_ADMIN_HASH=${metrics_admin_hash}
METRICS_USER_HASH=${metrics_user_hash}
EOF
    
    chmod 600 "${AUTH_DIR}/credentials.txt"
    
    # Create environment file for Docker Compose
    cat > "${AUTH_DIR}/.env" <<EOF
# XORB Telemetry Environment Variables
# Generated: $(date)

# Grafana
GRAFANA_ADMIN_PASSWORD=${dashboard_admin_pass}
GRAFANA_SECRET_KEY=${grafana_secret_key}

# API Keys
NEURAL_API_KEY=${neural_api_key}
LEARNING_API_KEY=${learning_api_key}

# Database
POSTGRES_PASSWORD=${postgres_password}
REDIS_PASSWORD=${redis_password}
EOF
    
    chmod 600 "${AUTH_DIR}/.env"
    
    # Create updated Caddyfile with real hashes
    create_updated_caddyfile "${dashboard_admin_hash}" "${dashboard_user_hash}" "${metrics_admin_hash}" "${metrics_user_hash}"
    
    log_success "Credentials generated successfully!"
    log_info "Files created:"
    log_info "  📋 Credentials: ${AUTH_DIR}/credentials.txt"
    log_info "  🔧 Environment: ${AUTH_DIR}/.env"
    log_info "  🌐 Caddyfile:   /root/Xorb/Caddyfile.secure"
}

# Create updated Caddyfile with real bcrypt hashes
create_updated_caddyfile() {
    local dashboard_admin_hash=$1
    local dashboard_user_hash=$2
    local metrics_admin_hash=$3
    local metrics_user_hash=$4
    
    log_info "Creating secure Caddyfile with real authentication hashes..."
    
    cat > "/root/Xorb/Caddyfile.secure" <<EOF
# XORB Secure Telemetry Exposure Configuration
# Caddy reverse proxy with TLS and real authentication

# AI Intelligence Dashboard (Grafana)
dashboard.xorb.local:443 {
    # TLS configuration
    tls internal {
        on_demand
    }
    
    # Basic authentication with real hashes
    basicauth {
        admin ${dashboard_admin_hash}
        xorb_user ${dashboard_user_hash}
    }
    
    # Reverse proxy to Grafana
    reverse_proxy localhost:3002 {
        header_up Host {upstream_hostport}
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-For {remote_host}
        header_up X-Forwarded-Proto {scheme}
    }
    
    # Security headers
    header {
        X-Frame-Options DENY
        X-Content-Type-Options nosniff
        X-XSS-Protection "1; mode=block"
        Strict-Transport-Security "max-age=31536000; includeSubDomains"
        Content-Security-Policy "default-src 'self' 'unsafe-inline' 'unsafe-eval'"
    }
    
    # Rate limiting
    rate_limit {
        zone dashboard_ip {
            key {remote_host}
            events 100
            window 1m
        }
    }
    
    # Logging
    log {
        output file /var/log/caddy/dashboard.log
        format json
    }
}

# Prometheus AI Metrics
metrics.xorb.local:443 {
    # TLS configuration
    tls internal {
        on_demand
    }
    
    # Basic authentication with real hashes
    basicauth {
        admin ${metrics_admin_hash}
        metrics_user ${metrics_user_hash}
    }
    
    # Reverse proxy to Prometheus
    reverse_proxy localhost:9092 {
        header_up Host {upstream_hostport}
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-For {remote_host}
        header_up X-Forwarded-Proto {scheme}
    }
    
    # Security headers
    header {
        X-Frame-Options DENY
        X-Content-Type-Options nosniff
        X-XSS-Protection "1; mode=block"
        Strict-Transport-Security "max-age=31536000; includeSubDomains"
    }
    
    # Rate limiting for metrics scraping
    rate_limit {
        zone metrics_ip {
            key {remote_host}
            events 1000
            window 1m
        }
    }
    
    # Logging
    log {
        output file /var/log/caddy/metrics.log
        format json
    }
}

# Neural Orchestration API
orchestrator.xorb.local:443 {
    # TLS configuration
    tls internal {
        on_demand
    }
    
    # API key authentication middleware
    @api_auth {
        header Authorization "Bearer $(cat ${AUTH_DIR}/.env | grep NEURAL_API_KEY | cut -d= -f2)"
    }
    
    handle @api_auth {
        # Reverse proxy to Neural Orchestrator
        reverse_proxy localhost:8003 {
            header_up Host {upstream_hostport}
            header_up X-Real-IP {remote_host}
            header_up X-Forwarded-For {remote_host}
            header_up X-Forwarded-Proto {scheme}
        }
    }
    
    handle {
        respond "Unauthorized: Valid API key required" 401
    }
    
    # Security headers
    header {
        X-Frame-Options DENY
        X-Content-Type-Options nosniff
        X-XSS-Protection "1; mode=block"
        Strict-Transport-Security "max-age=31536000; includeSubDomains"
    }
    
    # Rate limiting for API calls
    rate_limit {
        zone orchestrator_ip {
            key {remote_host}
            events 500
            window 1m
        }
    }
    
    # Logging
    log {
        output file /var/log/caddy/orchestrator.log
        format json
    }
}

# Autonomous Learning API
learning.xorb.local:443 {
    # TLS configuration
    tls internal {
        on_demand
    }
    
    # API key authentication middleware
    @api_auth {
        header Authorization "Bearer $(cat ${AUTH_DIR}/.env | grep LEARNING_API_KEY | cut -d= -f2)"
    }
    
    handle @api_auth {
        # Reverse proxy to Learning Service
        reverse_proxy localhost:8004 {
            header_up Host {upstream_hostport}
            header_up X-Real-IP {remote_host}
            header_up X-Forwarded-For {remote_host}
            header_up X-Forwarded-Proto {scheme}
        }
    }
    
    handle {
        respond "Unauthorized: Valid API key required" 401
    }
    
    # Security headers
    header {
        X-Frame-Options DENY
        X-Content-Type-Options nosniff
        X-XSS-Protection "1; mode=block"
        Strict-Transport-Security "max-age=31536000; includeSubDomains"
    }
    
    # Rate limiting for API calls
    rate_limit {
        zone learning_ip {
            key {remote_host}
            events 300
            window 1m
        }
    }
    
    # Logging
    log {
        output file /var/log/caddy/learning.log
        format json
    }
}

# Health check endpoint (no auth required)
health.xorb.local:443 {
    tls internal {
        on_demand
    }
    
    respond /health \`{
        "status": "healthy",
        "services": {
            "dashboard": "https://dashboard.xorb.local",
            "metrics": "https://metrics.xorb.local", 
            "orchestrator": "https://orchestrator.xorb.local",
            "learning": "https://learning.xorb.local"
        },
        "timestamp": "{{.now.Format \"2006-01-02T15:04:05Z\"}}"
    }\`
    
    log {
        output file /var/log/caddy/health.log
        format json
    }
}

# Global configuration
{
    # Security settings
    auto_https disable_redirects
    
    # Admin endpoint (local only)
    admin localhost:2019
    
    # Global logging
    log {
        level INFO
        output file /var/log/caddy/access.log {
            roll_size 100mb
            roll_keep 5
            roll_keep_for 720h
        }
        format json
    }
}
EOF
    
    log_success "Secure Caddyfile created: /root/Xorb/Caddyfile.secure"
}

# Create authentication test script
create_auth_test_script() {
    log_info "Creating authentication test script..."
    
    cat > "${AUTH_DIR}/test-auth.sh" <<'EOF'
#!/bin/bash

# XORB Authentication Test Script
# Tests all authentication mechanisms

AUTH_DIR="/root/Xorb/auth"
source "${AUTH_DIR}/.env"

echo "🔐 XORB Authentication Test"
echo "=========================="

# Test health endpoint (no auth)
echo ""
echo "🏥 Testing health endpoint (no auth required)..."
curl -k -s "https://health.xorb.local/health" | jq . 2>/dev/null || echo "❌ Health endpoint failed"

# Test dashboard with basic auth
echo ""
echo "📊 Testing dashboard with basic auth..."
curl -k -s -u "admin:${GRAFANA_ADMIN_PASSWORD}" "https://dashboard.xorb.local/api/health" | jq . 2>/dev/null && echo "✅ Dashboard auth successful" || echo "❌ Dashboard auth failed"

# Test metrics with basic auth
echo ""
echo "📈 Testing metrics with basic auth..."
curl -k -s -u "admin:${GRAFANA_ADMIN_PASSWORD}" "https://metrics.xorb.local/api/v1/query?query=up" | jq . 2>/dev/null && echo "✅ Metrics auth successful" || echo "❌ Metrics auth failed"

# Test neural orchestrator with API key
echo ""
echo "🧠 Testing neural orchestrator with API key..."
curl -k -s -H "Authorization: Bearer ${NEURAL_API_KEY}" "https://orchestrator.xorb.local/health" | jq . 2>/dev/null && echo "✅ Neural API auth successful" || echo "❌ Neural API auth failed"

# Test learning service with API key
echo ""
echo "🎓 Testing learning service with API key..."
curl -k -s -H "Authorization: Bearer ${LEARNING_API_KEY}" "https://learning.xorb.local/health" | jq . 2>/dev/null && echo "✅ Learning API auth successful" || echo "❌ Learning API auth failed"

echo ""
echo "🔍 Authentication testing complete"
EOF
    
    chmod +x "${AUTH_DIR}/test-auth.sh"
    log_success "Authentication test script created: ${AUTH_DIR}/test-auth.sh"
}

# Create credential rotation script
create_rotation_script() {
    log_info "Creating credential rotation script..."
    
    cat > "${AUTH_DIR}/rotate-credentials.sh" <<'EOF'
#!/bin/bash

# XORB Credential Rotation Script
# Rotates all authentication credentials

AUTH_DIR="/root/Xorb/auth"
BACKUP_DIR="${AUTH_DIR}/backup_$(date +%Y%m%d_%H%M%S)"

echo "🔄 XORB Credential Rotation"
echo "========================="

# Create backup
echo "📦 Creating backup of existing credentials..."
mkdir -p "${BACKUP_DIR}"
cp "${AUTH_DIR}"/{credentials.txt,.env} "${BACKUP_DIR}/" 2>/dev/null || true

# Regenerate credentials
echo "🔐 Regenerating credentials..."
cd /root/Xorb
./scripts/generate-auth-credentials.sh

# Update Docker Compose services
echo "🔄 Restarting services with new credentials..."
docker-compose -f docker-compose.telemetry-secure.yml down
docker-compose -f docker-compose.telemetry-secure.yml --env-file="${AUTH_DIR}/.env" up -d

echo "✅ Credential rotation complete"
echo "📦 Backup stored in: ${BACKUP_DIR}"
EOF
    
    chmod +x "${AUTH_DIR}/rotate-credentials.sh"
    log_success "Credential rotation script created: ${AUTH_DIR}/rotate-credentials.sh"
}

# Install required Python packages
install_dependencies() {
    log_info "Installing required Python packages..."
    
    if ! python3 -c "import bcrypt" 2>/dev/null; then
        pip3 install bcrypt || {
            log_warning "Failed to install bcrypt via pip, trying apt..."
            apt-get update && apt-get install -y python3-bcrypt
        }
    fi
    
    log_success "Dependencies installed"
}

# Main execution
main() {
    log_info "🔐 XORB Telemetry Authentication Setup"
    log_info "====================================="
    
    install_dependencies
    create_auth_directory
    generate_credentials
    create_auth_test_script
    create_rotation_script
    
    log_success "🎉 Authentication setup complete!"
    log_info ""
    log_info "📋 Files created:"
    log_info "   Credentials: ${AUTH_DIR}/credentials.txt"
    log_info "   Environment: ${AUTH_DIR}/.env"
    log_info "   Secure Caddyfile: /root/Xorb/Caddyfile.secure"
    log_info "   Test script: ${AUTH_DIR}/test-auth.sh"
    log_info "   Rotation script: ${AUTH_DIR}/rotate-credentials.sh"
    log_info ""
    log_warning "⚠️  IMPORTANT: Store credentials securely and do not commit to version control!"
    log_info ""
    log_info "📋 Next steps:"
    log_info "1. Review credentials: cat ${AUTH_DIR}/credentials.txt"
    log_info "2. Copy secure Caddyfile: cp /root/Xorb/Caddyfile.secure /root/Xorb/Caddyfile"
    log_info "3. Start services: docker-compose -f docker-compose.telemetry-secure.yml --env-file=${AUTH_DIR}/.env up -d"
    log_info "4. Test authentication: ${AUTH_DIR}/test-auth.sh"
}

# Run main function
main "$@"