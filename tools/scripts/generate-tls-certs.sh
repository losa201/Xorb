#!/bin/bash

# XORB Telemetry TLS Certificate Generation Script
# Generates self-signed certificates for secure telemetry access

set -euo pipefail

# Configuration
CERT_DIR="/root/Xorb/certs"
DOMAINS=("dashboard.xorb.local" "metrics.xorb.local" "orchestrator.xorb.local" "learning.xorb.local" "health.xorb.local")
CA_KEY="${CERT_DIR}/ca-key.pem"
CA_CERT="${CERT_DIR}/ca-cert.pem"
CERT_VALIDITY_DAYS=365

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

# Create certificate directory
create_cert_directory() {
    log_info "Creating certificate directory..."
    mkdir -p "${CERT_DIR}"
    chmod 700 "${CERT_DIR}"
    log_success "Certificate directory created: ${CERT_DIR}"
}

# Generate CA certificate
generate_ca_certificate() {
    log_info "Generating Certificate Authority (CA)..."
    
    # Generate CA private key
    openssl genrsa -out "${CA_KEY}" 4096
    chmod 600 "${CA_KEY}"
    
    # Generate CA certificate
    openssl req -new -x509 -days ${CERT_VALIDITY_DAYS} -key "${CA_KEY}" -out "${CA_CERT}" \
        -subj "/C=US/ST=Security/L=XORB/O=XORB AI Platform/OU=Telemetry/CN=XORB Root CA"
    
    chmod 644 "${CA_CERT}"
    log_success "CA certificate generated: ${CA_CERT}"
}

# Generate domain certificates
generate_domain_certificates() {
    log_info "Generating domain certificates..."
    
    for domain in "${DOMAINS[@]}"; do
        log_info "Processing domain: ${domain}"
        
        local domain_key="${CERT_DIR}/${domain}-key.pem"
        local domain_csr="${CERT_DIR}/${domain}.csr"
        local domain_cert="${CERT_DIR}/${domain}-cert.pem"
        local domain_config="${CERT_DIR}/${domain}.conf"
        
        # Create domain configuration file
        cat > "${domain_config}" <<EOF
[req]
default_bits = 2048
prompt = no
distinguished_name = req_distinguished_name
req_extensions = v3_req

[req_distinguished_name]
C = US
ST = Security
L = XORB
O = XORB AI Platform
OU = Telemetry
CN = ${domain}

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = ${domain}
DNS.2 = localhost
IP.1 = 127.0.0.1
IP.2 = ::1
EOF
        
        # Generate domain private key
        openssl genrsa -out "${domain_key}" 2048
        chmod 600 "${domain_key}"
        
        # Generate certificate signing request (CSR)
        openssl req -new -key "${domain_key}" -out "${domain_csr}" -config "${domain_config}"
        
        # Generate domain certificate signed by CA
        openssl x509 -req -in "${domain_csr}" -CA "${CA_CERT}" -CAkey "${CA_KEY}" \
            -CAcreateserial -out "${domain_cert}" -days ${CERT_VALIDITY_DAYS} \
            -extensions v3_req -extfile "${domain_config}"
        
        chmod 644 "${domain_cert}"
        
        # Clean up CSR and config files
        rm -f "${domain_csr}" "${domain_config}"
        
        log_success "Certificate generated for ${domain}"
    done
}

# Create PEM bundles for Caddy
create_caddy_bundles() {
    log_info "Creating certificate bundles for Caddy..."
    
    for domain in "${DOMAINS[@]}"; do
        local domain_key="${CERT_DIR}/${domain}-key.pem"
        local domain_cert="${CERT_DIR}/${domain}-cert.pem"
        local bundle_file="${CERT_DIR}/${domain}.pem"
        
        # Create bundle with certificate and key
        cat "${domain_cert}" "${domain_key}" > "${bundle_file}"
        chmod 600 "${bundle_file}"
        
        log_success "Bundle created for ${domain}: ${bundle_file}"
    done
}

# Generate DH parameters for enhanced security
generate_dh_params() {
    log_info "Generating Diffie-Hellman parameters (this may take a while)..."
    local dh_file="${CERT_DIR}/dhparam.pem"
    
    openssl dhparam -out "${dh_file}" 2048
    chmod 644 "${dh_file}"
    
    log_success "DH parameters generated: ${dh_file}"
}

# Update /etc/hosts for local testing
update_hosts_file() {
    log_info "Updating /etc/hosts for local domain resolution..."
    
    # Backup original hosts file
    cp /etc/hosts /etc/hosts.backup.$(date +%Y%m%d_%H%M%S)
    
    # Remove existing XORB entries
    sed -i '/# XORB Telemetry Domains/,/# End XORB Telemetry Domains/d' /etc/hosts
    
    # Add new entries
    {
        echo "# XORB Telemetry Domains"
        for domain in "${DOMAINS[@]}"; do
            echo "127.0.0.1 ${domain}"
        done
        echo "# End XORB Telemetry Domains"
    } >> /etc/hosts
    
    log_success "/etc/hosts updated with XORB telemetry domains"
}

# Create certificate verification script
create_verification_script() {
    log_info "Creating certificate verification script..."
    
    cat > "${CERT_DIR}/verify-certs.sh" <<'EOF'
#!/bin/bash

# XORB Certificate Verification Script

CERT_DIR="/root/Xorb/certs"
DOMAINS=("dashboard.xorb.local" "metrics.xorb.local" "orchestrator.xorb.local" "learning.xorb.local" "health.xorb.local")

echo "🔐 XORB Certificate Verification Report"
echo "======================================"

# Check CA certificate
echo ""
echo "📋 Certificate Authority:"
openssl x509 -in "${CERT_DIR}/ca-cert.pem" -text -noout | grep -E "(Subject:|Not Before|Not After)"

# Check domain certificates
for domain in "${DOMAINS[@]}"; do
    echo ""
    echo "🌐 Domain: ${domain}"
    echo "   Certificate: ${CERT_DIR}/${domain}-cert.pem"
    
    if [[ -f "${CERT_DIR}/${domain}-cert.pem" ]]; then
        openssl x509 -in "${CERT_DIR}/${domain}-cert.pem" -text -noout | grep -E "(Subject:|Not Before|Not After|DNS:|IP Address:)"
        
        # Verify certificate chain
        if openssl verify -CAfile "${CERT_DIR}/ca-cert.pem" "${CERT_DIR}/${domain}-cert.pem" > /dev/null 2>&1; then
            echo "   ✅ Certificate chain valid"
        else
            echo "   ❌ Certificate chain invalid"
        fi
    else
        echo "   ❌ Certificate file not found"
    fi
done

echo ""
echo "📁 Certificate files:"
ls -la "${CERT_DIR}/"*.pem 2>/dev/null || echo "   No certificate files found"

echo ""
echo "🔍 Certificate verification complete"
EOF
    
    chmod +x "${CERT_DIR}/verify-certs.sh"
    log_success "Certificate verification script created: ${CERT_DIR}/verify-certs.sh"
}

# Create certificate renewal script
create_renewal_script() {
    log_info "Creating certificate renewal script..."
    
    cat > "${CERT_DIR}/renew-certs.sh" <<'EOF'
#!/bin/bash

# XORB Certificate Renewal Script
# Run this script to renew certificates before they expire

CERT_DIR="/root/Xorb/certs"
BACKUP_DIR="${CERT_DIR}/backup_$(date +%Y%m%d_%H%M%S)"

echo "🔄 XORB Certificate Renewal"
echo "=========================="

# Create backup
echo "📦 Creating backup of existing certificates..."
mkdir -p "${BACKUP_DIR}"
cp "${CERT_DIR}"/*.pem "${BACKUP_DIR}/" 2>/dev/null || true

# Regenerate certificates
echo "🔐 Regenerating certificates..."
cd /root/Xorb
./scripts/generate-tls-certs.sh

# Restart Caddy to pick up new certificates
echo "🔄 Restarting Caddy proxy..."
docker-compose -f docker-compose.telemetry-secure.yml restart xorb-caddy-proxy

echo "✅ Certificate renewal complete"
echo "📦 Backup stored in: ${BACKUP_DIR}"
EOF
    
    chmod +x "${CERT_DIR}/renew-certs.sh"
    log_success "Certificate renewal script created: ${CERT_DIR}/renew-certs.sh"
}

# Create certificate monitoring script
create_monitoring_script() {
    log_info "Creating certificate monitoring script..."
    
    cat > "${CERT_DIR}/monitor-certs.sh" <<'EOF'
#!/bin/bash

# XORB Certificate Monitoring Script
# Checks certificate expiration and sends alerts

CERT_DIR="/root/Xorb/certs"
DOMAINS=("dashboard.xorb.local" "metrics.xorb.local" "orchestrator.xorb.local" "learning.xorb.local" "health.xorb.local")
WARNING_DAYS=30
CRITICAL_DAYS=7

echo "🔍 XORB Certificate Expiration Monitor"
echo "====================================="

for domain in "${DOMAINS[@]}"; do
    cert_file="${CERT_DIR}/${domain}-cert.pem"
    
    if [[ -f "${cert_file}" ]]; then
        # Get certificate expiration date
        exp_date=$(openssl x509 -in "${cert_file}" -noout -dates | grep notAfter | cut -d= -f2)
        exp_timestamp=$(date -d "${exp_date}" +%s)
        current_timestamp=$(date +%s)
        days_until_expiry=$(( (exp_timestamp - current_timestamp) / 86400 ))
        
        echo ""
        echo "🌐 ${domain}:"
        echo "   Expires: ${exp_date}"
        echo "   Days until expiry: ${days_until_expiry}"
        
        if [[ ${days_until_expiry} -le ${CRITICAL_DAYS} ]]; then
            echo "   🚨 CRITICAL: Certificate expires in ${days_until_expiry} days!"
        elif [[ ${days_until_expiry} -le ${WARNING_DAYS} ]]; then
            echo "   ⚠️  WARNING: Certificate expires in ${days_until_expiry} days"
        else
            echo "   ✅ Certificate is valid"
        fi
    else
        echo ""
        echo "🌐 ${domain}:"
        echo "   ❌ Certificate file not found: ${cert_file}"
    fi
done

echo ""
echo "🔍 Certificate monitoring complete"
EOF
    
    chmod +x "${CERT_DIR}/monitor-certs.sh"
    log_success "Certificate monitoring script created: ${CERT_DIR}/monitor-certs.sh"
}

# Main execution
main() {
    log_info "🔐 XORB Telemetry TLS Certificate Generation"
    log_info "==========================================="
    
    create_cert_directory
    generate_ca_certificate
    generate_domain_certificates
    create_caddy_bundles
    generate_dh_params
    update_hosts_file
    create_verification_script
    create_renewal_script
    create_monitoring_script
    
    log_success "🎉 TLS certificate generation complete!"
    log_info ""
    log_info "📋 Next steps:"
    log_info "1. Verify certificates: ${CERT_DIR}/verify-certs.sh"
    log_info "2. Start secure telemetry: docker-compose -f docker-compose.telemetry-secure.yml up -d"
    log_info "3. Test access: curl -k https://health.xorb.local/health"
    log_info "4. Monitor certificates: ${CERT_DIR}/monitor-certs.sh"
    log_info ""
    log_info "🌐 Telemetry endpoints:"
    log_info "   Dashboard: https://dashboard.xorb.local (admin/xorb_user)"
    log_info "   Metrics:   https://metrics.xorb.local (admin/metrics_user)"
    log_info "   Neural:    https://orchestrator.xorb.local (API Key required)"
    log_info "   Learning:  https://learning.xorb.local (API Key required)"
    log_info "   Health:    https://health.xorb.local (no auth)"
}

# Run main function
main "$@"