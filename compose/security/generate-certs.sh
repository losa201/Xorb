#!/bin/bash
# Generate TLS certificates for Xorb PTaaS services
# Self-signed certificates for development environment

set -euo pipefail

CERT_DIR="/tmp/xorb/certs"
mkdir -p "$CERT_DIR"/{certs,private}

# Certificate configuration
COUNTRY="US"
STATE="CA"
LOCALITY="San Francisco"
ORG="Xorb Security"
ORG_UNIT="PTaaS Platform"
VALIDITY_DAYS=365

# Generate CA private key
openssl genrsa -out "$CERT_DIR/private/xorb-ca.key" 4096

# Generate CA certificate
openssl req -new -x509 -days $VALIDITY_DAYS -key "$CERT_DIR/private/xorb-ca.key" \
    -out "$CERT_DIR/certs/xorb-ca.crt" \
    -subj "/C=$COUNTRY/ST=$STATE/L=$LOCALITY/O=$ORG/OU=$ORG_UNIT/CN=Xorb-CA"

# Function to generate service certificates
generate_service_cert() {
    local service=$1
    local common_name=${2:-$service.xorb.local}
    
    echo "Generating certificate for $service..."
    
    # Generate private key
    openssl genrsa -out "$CERT_DIR/private/$service.key" 2048
    
    # Generate certificate signing request
    openssl req -new -key "$CERT_DIR/private/$service.key" \
        -out "$CERT_DIR/$service.csr" \
        -subj "/C=$COUNTRY/ST=$STATE/L=$LOCALITY/O=$ORG/OU=$ORG_UNIT/CN=$common_name"
    
    # Create certificate extensions
    cat > "$CERT_DIR/$service.ext" << EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = $common_name
DNS.2 = $service
DNS.3 = localhost
IP.1 = 127.0.0.1
IP.2 = 172.20.0.0/16
EOF
    
    # Sign certificate with CA
    openssl x509 -req -in "$CERT_DIR/$service.csr" \
        -CA "$CERT_DIR/certs/xorb-ca.crt" \
        -CAkey "$CERT_DIR/private/xorb-ca.key" \
        -CAcreateserial \
        -out "$CERT_DIR/certs/$service.crt" \
        -days $VALIDITY_DAYS \
        -extensions v3_req \
        -extfile "$CERT_DIR/$service.ext"
    
    # Clean up temporary files
    rm "$CERT_DIR/$service.csr" "$CERT_DIR/$service.ext"
}

# Generate certificates for all services
generate_service_cert "api" "api.xorb.local"
generate_service_cert "worker" "worker.xorb.local"
generate_service_cert "orchestrator" "orchestrator.xorb.local"
generate_service_cert "scanner" "scanner.xorb.local"
generate_service_cert "triage" "triage.xorb.local"
generate_service_cert "payments" "payments.xorb.local"
generate_service_cert "researcher-portal" "portal.xorb.local"
generate_service_cert "scheduler" "scheduler.xorb.local"

# Generate main server certificate
generate_service_cert "xorb" "xorb.local"

# Set proper permissions
chmod 600 "$CERT_DIR/private"/*.key
chmod 644 "$CERT_DIR/certs"/*.crt

echo "TLS certificates generated successfully in $CERT_DIR"
echo "CA certificate: $CERT_DIR/certs/xorb-ca.crt"
echo "Service certificates generated for all Xorb services"

# Create Docker volume for certificates
docker volume create xorb_certs
docker run --rm -v xorb_certs:/certs -v "$CERT_DIR":/tmp/certs alpine \
    sh -c "cp -r /tmp/certs/* /certs/ && chown -R 1001:1001 /certs"

echo "Certificates copied to Docker volume 'xorb_certs'"