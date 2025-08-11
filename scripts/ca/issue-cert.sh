#!/bin/bash
# Certificate Issuance Script for XORB Platform Services
# Issues server and client certificates with proper SANs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRETS_DIR="${SCRIPT_DIR}/../../secrets/tls"
CA_DIR="${SECRETS_DIR}/ca"

# Default certificate validity (30 days for short-lived certs)
CERT_DAYS=30

usage() {
    cat << EOF
Usage: $0 [OPTIONS] <service_name> <cert_type>

OPTIONS:
    -d, --days DAYS     Certificate validity in days (default: 30)
    -h, --help          Show this help message

ARGUMENTS:
    service_name        Name of the service (api, orchestrator, agent, redis, dind, etc.)
    cert_type          Type of certificate (server, client, both)

EXAMPLES:
    $0 api server
    $0 orchestrator both
    $0 redis-client client
    $0 -d 90 api server
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--days)
            CERT_DAYS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Error: Unknown option $1" >&2
            usage
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -lt 2 ]]; then
    echo "Error: Missing required arguments" >&2
    usage
    exit 1
fi

SERVICE_NAME="$1"
CERT_TYPE="$2"

# Validate certificate type
case "$CERT_TYPE" in
    server|client|both)
        ;;
    *)
        echo "Error: Invalid cert_type. Must be 'server', 'client', or 'both'" >&2
        exit 1
        ;;
esac

echo "🔐 Issuing ${CERT_TYPE} certificate for ${SERVICE_NAME}..."

# Create service directory
SERVICE_DIR="${SECRETS_DIR}/${SERVICE_NAME}"
mkdir -p "${SERVICE_DIR}"

# Generate private key
echo "📝 Generating private key for ${SERVICE_NAME}..."
openssl genrsa -out "${SERVICE_DIR}/key.pem" 2048
chmod 400 "${SERVICE_DIR}/key.pem"

# Define Subject Alternative Names based on service
get_san_config() {
    local service="$1"
    case "$service" in
        api)
            echo "DNS:api,DNS:api.xorb.local,DNS:localhost,IP:127.0.0.1,IP:0.0.0.0"
            ;;
        orchestrator)
            echo "DNS:orchestrator,DNS:orchestrator.xorb.local,DNS:localhost,IP:127.0.0.1"
            ;;
        agent|agent-*)
            echo "DNS:${service},DNS:${service}.xorb.local,DNS:localhost,IP:127.0.0.1"
            ;;
        redis)
            echo "DNS:redis,DNS:redis.xorb.local,DNS:localhost,IP:127.0.0.1"
            ;;
        dind)
            echo "DNS:dind,DNS:dind.xorb.local,DNS:localhost,IP:127.0.0.1"
            ;;
        envoy|envoy-*)
            echo "DNS:${service},DNS:${service}.xorb.local,DNS:localhost,IP:127.0.0.1"
            ;;
        *)
            echo "DNS:${service},DNS:${service}.xorb.local,DNS:localhost,IP:127.0.0.1"
            ;;
    esac
}

# Update OpenSSL config with service-specific alt names
update_ca_config() {
    local service="$1"
    local san_config="$2"
    
    # Create temporary config file with service-specific SAN
    cp "${CA_DIR}/intermediate/openssl.cnf" "${SERVICE_DIR}/temp_openssl.cnf"
    
    # Update alt_names section with service-specific values
    sed -i '/\[ alt_names \]/,$d' "${SERVICE_DIR}/temp_openssl.cnf"
    
    cat >> "${SERVICE_DIR}/temp_openssl.cnf" << EOF
[ alt_names ]
EOF
    
    # Parse SAN config and add to alt_names
    local counter=1
    IFS=',' read -ra SANS <<< "$san_config"
    for san in "${SANS[@]}"; do
        san=$(echo "$san" | xargs) # trim whitespace
        if [[ "$san" =~ ^DNS: ]]; then
            echo "DNS.${counter} = ${san#DNS:}" >> "${SERVICE_DIR}/temp_openssl.cnf"
        elif [[ "$san" =~ ^IP: ]]; then
            echo "IP.${counter} = ${san#IP:}" >> "${SERVICE_DIR}/temp_openssl.cnf"
        fi
        ((counter++))
    done
}

# Generate Certificate Signing Request
echo "📋 Creating certificate signing request for ${SERVICE_NAME}..."
SUBJECT="/C=US/ST=CA/L=San Francisco/O=XORB Platform/OU=Services/CN=${SERVICE_NAME}.xorb.local"

openssl req -new -key "${SERVICE_DIR}/key.pem" \
    -subj "${SUBJECT}" \
    -out "${SERVICE_DIR}/csr.pem"

# Get SAN configuration for the service
SAN_CONFIG=$(get_san_config "$SERVICE_NAME")

# Update CA config with service-specific SAN
update_ca_config "$SERVICE_NAME" "$SAN_CONFIG"

# Determine extension type based on cert type
case "$CERT_TYPE" in
    "server")
        EXTENSIONS="server_cert"
        ;;
    "client")
        EXTENSIONS="client_cert"
        ;;
    "both")
        EXTENSIONS="both_cert"
        ;;
esac

# Sign the certificate with Intermediate CA using proper extensions
echo "🏛️ Signing certificate with Intermediate CA..."
openssl ca -config "${SERVICE_DIR}/temp_openssl.cnf" \
    -extensions "${EXTENSIONS}" \
    -days "${CERT_DAYS}" \
    -notext -md sha256 -batch \
    -passin pass:xorb-intermediate-ca-key \
    -in "${SERVICE_DIR}/csr.pem" \
    -out "${SERVICE_DIR}/cert.pem"

chmod 444 "${SERVICE_DIR}/cert.pem"

# Clean up temporary config
rm -f "${SERVICE_DIR}/temp_openssl.cnf"

# Copy CA chain
cp "${CA_DIR}/ca-chain.cert.pem" "${SERVICE_DIR}/ca.pem"

# Create full certificate chain (cert + intermediate + root)
cat "${SERVICE_DIR}/cert.pem" "${CA_DIR}/ca-chain.cert.pem" > "${SERVICE_DIR}/fullchain.pem"

# Create PKCS#12 bundle for Java/Windows compatibility
echo "📦 Creating PKCS#12 bundle..."
openssl pkcs12 -export -password pass:xorb-${SERVICE_NAME} \
    -out "${SERVICE_DIR}/${SERVICE_NAME}.p12" \
    -inkey "${SERVICE_DIR}/key.pem" \
    -in "${SERVICE_DIR}/cert.pem" \
    -certfile "${CA_DIR}/ca-chain.cert.pem" \
    -name "${SERVICE_NAME}.xorb.local"

# Generate separate client certificate if needed (only for client-only certs)
if [[ "$CERT_TYPE" == "client" ]]; then
    echo "🔐 Generating client certificate..."
    
    # Client-specific key and cert
    openssl genrsa -out "${SERVICE_DIR}/client-key.pem" 2048
    chmod 400 "${SERVICE_DIR}/client-key.pem"
    
    CLIENT_SUBJECT="/C=US/ST=CA/L=San Francisco/O=XORB Platform/OU=Clients/CN=${SERVICE_NAME}-client.xorb.local"
    
    openssl req -new -key "${SERVICE_DIR}/client-key.pem" \
        -subj "${CLIENT_SUBJECT}" \
        -out "${SERVICE_DIR}/client-csr.pem"
    
    # Client extensions
    cat > "${SERVICE_DIR}/client-extensions.cnf" << EOF
basicConstraints = CA:FALSE
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid,issuer:always
keyUsage = critical, digitalSignature
extendedKeyUsage = clientAuth
nsCertType = client
nsComment = "XORB ${SERVICE_NAME} Client Certificate"
EOF
    
    openssl ca -config "${CA_DIR}/intermediate/openssl.cnf" \
        -extensions client_cert \
        -days "${CERT_DAYS}" \
        -notext -md sha256 -batch \
        -passin pass:xorb-intermediate-ca-key \
        -extfile "${SERVICE_DIR}/client-extensions.cnf" \
        -in "${SERVICE_DIR}/client-csr.pem" \
        -out "${SERVICE_DIR}/client-cert.pem"
    
    chmod 444 "${SERVICE_DIR}/client-cert.pem"
    
    # Client PKCS#12 bundle
    openssl pkcs12 -export -password pass:xorb-${SERVICE_NAME}-client \
        -out "${SERVICE_DIR}/${SERVICE_NAME}-client.p12" \
        -inkey "${SERVICE_DIR}/client-key.pem" \
        -in "${SERVICE_DIR}/client-cert.pem" \
        -certfile "${CA_DIR}/ca-chain.cert.pem" \
        -name "${SERVICE_NAME}-client.xorb.local"
fi

# Set proper permissions
chown -R root:root "${SERVICE_DIR}"
chmod 400 "${SERVICE_DIR}"/*.pem "${SERVICE_DIR}"/*.p12 2>/dev/null || true
chmod 444 "${SERVICE_DIR}/cert.pem" "${SERVICE_DIR}/ca.pem" "${SERVICE_DIR}/fullchain.pem" 2>/dev/null || true
if [[ -f "${SERVICE_DIR}/client-cert.pem" ]]; then
    chmod 444 "${SERVICE_DIR}/client-cert.pem"
fi

# Cleanup temporary files
rm -f "${SERVICE_DIR}/csr.pem" "${SERVICE_DIR}/extensions.cnf" "${SERVICE_DIR}/client-csr.pem" "${SERVICE_DIR}/client-extensions.cnf"

echo "✅ Certificate issued successfully for ${SERVICE_NAME}!"
echo "📁 Certificate: ${SERVICE_DIR}/cert.pem"
echo "📁 Private Key: ${SERVICE_DIR}/key.pem"
echo "📁 CA Chain: ${SERVICE_DIR}/ca.pem"
echo "📁 Full Chain: ${SERVICE_DIR}/fullchain.pem"
echo "📁 PKCS#12: ${SERVICE_DIR}/${SERVICE_NAME}.p12"

if [[ "$CERT_TYPE" == "client" || "$CERT_TYPE" == "both" ]]; then
    echo "📁 Client Certificate: ${SERVICE_DIR}/client-cert.pem"
    echo "📁 Client Private Key: ${SERVICE_DIR}/client-key.pem"
    echo "📁 Client PKCS#12: ${SERVICE_DIR}/${SERVICE_NAME}-client.p12"
fi

echo ""
echo "🔍 Certificate details:"
openssl x509 -noout -text -in "${SERVICE_DIR}/cert.pem" | grep -A 1 "Subject:"
openssl x509 -noout -text -in "${SERVICE_DIR}/cert.pem" | grep -A 3 "X509v3 Subject Alternative Name:"
openssl x509 -noout -dates -in "${SERVICE_DIR}/cert.pem"