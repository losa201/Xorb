#!/bin/bash

# XORB SSL/TLS Certificate Automation System
# Automated certificate generation and deployment for production

set -euo pipefail

echo "🔒 XORB SSL/TLS Certificate Automation"
echo "====================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_step() {
    echo -e "${BLUE}🔧 $1${NC}"
}

# Configuration
DOMAIN_NAME="${XORB_DOMAIN:-localhost}"
SSL_DIR="/root/Xorb/ssl"
CERTS_DIR="$SSL_DIR/certs"
PRIVATE_DIR="$SSL_DIR/private"
CSR_DIR="$SSL_DIR/csr"

# Create SSL directory structure
log_step "Creating SSL directory structure..."
mkdir -p "$CERTS_DIR" "$PRIVATE_DIR" "$CSR_DIR"
chmod 700 "$PRIVATE_DIR"

# Generate CA private key and certificate
log_step "Generating Certificate Authority (CA)..."
if [ ! -f "$PRIVATE_DIR/ca-key.pem" ]; then
    openssl genrsa -out "$PRIVATE_DIR/ca-key.pem" 4096
    log_info "CA private key generated"
else
    log_info "CA private key already exists"
fi

if [ ! -f "$CERTS_DIR/ca-cert.pem" ]; then
    cat > "$SSL_DIR/ca.conf" << EOF
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_ca
prompt = no

[req_distinguished_name]
C = US
ST = Production
L = Enterprise
O = XORB Cybersecurity Platform
OU = Security Operations
CN = XORB Root CA

[v3_ca]
basicConstraints = critical,CA:TRUE
keyUsage = critical,keyCertSign,cRLSign
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid:always,issuer:always
EOF

    openssl req -new -x509 -days 3650 -key "$PRIVATE_DIR/ca-key.pem" \
        -out "$CERTS_DIR/ca-cert.pem" -config "$SSL_DIR/ca.conf"
    log_info "CA certificate generated (valid for 10 years)"
else
    log_info "CA certificate already exists"
fi

# Generate server private key
log_step "Generating server private key..."
if [ ! -f "$PRIVATE_DIR/server-key.pem" ]; then
    openssl genrsa -out "$PRIVATE_DIR/server-key.pem" 4096
    log_info "Server private key generated"
else
    log_info "Server private key already exists"
fi

# Generate server certificate signing request (CSR)
log_step "Generating server certificate signing request..."
cat > "$SSL_DIR/server.conf" << EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = Production
L = Enterprise
O = XORB Cybersecurity Platform
OU = Production Services
CN = $DOMAIN_NAME

[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = $DOMAIN_NAME
DNS.2 = localhost
DNS.3 = api.xorb.local
DNS.4 = orchestrator.xorb.local
DNS.5 = worker.xorb.local
IP.1 = 127.0.0.1
IP.2 = ::1
EOF

openssl req -new -key "$PRIVATE_DIR/server-key.pem" \
    -out "$CSR_DIR/server.csr" -config "$SSL_DIR/server.conf"
log_info "Server CSR generated"

# Sign server certificate with CA
log_step "Signing server certificate with CA..."
cat > "$SSL_DIR/server_cert.conf" << EOF
[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth, clientAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = $DOMAIN_NAME
DNS.2 = localhost
DNS.3 = api.xorb.local
DNS.4 = orchestrator.xorb.local
DNS.5 = worker.xorb.local
IP.1 = 127.0.0.1
IP.2 = ::1
EOF

openssl x509 -req -in "$CSR_DIR/server.csr" -CA "$CERTS_DIR/ca-cert.pem" \
    -CAkey "$PRIVATE_DIR/ca-key.pem" -CAcreateserial \
    -out "$CERTS_DIR/server-cert.pem" -days 365 \
    -extensions v3_req -extfile "$SSL_DIR/server_cert.conf"
log_info "Server certificate signed (valid for 1 year)"

# Generate client certificates for mutual TLS
log_step "Generating client certificates for mutual TLS..."
CLIENT_NAMES=("api-client" "orchestrator-client" "worker-client" "admin-client")

for client in "${CLIENT_NAMES[@]}"; do
    if [ ! -f "$PRIVATE_DIR/${client}-key.pem" ]; then
        # Generate client private key
        openssl genrsa -out "$PRIVATE_DIR/${client}-key.pem" 2048

        # Generate client CSR
        cat > "$SSL_DIR/${client}.conf" << EOF
[req]
distinguished_name = req_distinguished_name
prompt = no

[req_distinguished_name]
C = US
ST = Production
L = Enterprise
O = XORB Cybersecurity Platform
OU = Client Authentication
CN = $client
EOF

        openssl req -new -key "$PRIVATE_DIR/${client}-key.pem" \
            -out "$CSR_DIR/${client}.csr" -config "$SSL_DIR/${client}.conf"

        # Sign client certificate
        openssl x509 -req -in "$CSR_DIR/${client}.csr" \
            -CA "$CERTS_DIR/ca-cert.pem" -CAkey "$PRIVATE_DIR/ca-key.pem" \
            -CAcreateserial -out "$CERTS_DIR/${client}-cert.pem" -days 365

        log_info "Client certificate generated for: $client"
    else
        log_info "Client certificate already exists for: $client"
    fi
done

# Create certificate bundles
log_step "Creating certificate bundles..."
cat "$CERTS_DIR/server-cert.pem" "$CERTS_DIR/ca-cert.pem" > "$CERTS_DIR/server-bundle.pem"
log_info "Server certificate bundle created"

# Create NGINX SSL configuration template
log_step "Creating NGINX SSL configuration..."
mkdir -p "$SSL_DIR/nginx"
cat > "$SSL_DIR/nginx/ssl.conf" << EOF
# XORB SSL/TLS Configuration for NGINX
ssl_certificate /etc/ssl/certs/server-bundle.pem;
ssl_certificate_key /etc/ssl/private/server-key.pem;
ssl_client_certificate /etc/ssl/certs/ca-cert.pem;

# SSL Configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-SHA384;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;
ssl_session_tickets off;

# HSTS
add_header Strict-Transport-Security "max-age=63072000" always;

# Security headers
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
add_header Referrer-Policy "strict-origin-when-cross-origin";

# Optional: Enable mutual TLS (uncomment for mTLS)
# ssl_verify_client on;
# ssl_verify_depth 2;
EOF

# Create Apache SSL configuration template
cat > "$SSL_DIR/apache/ssl.conf" << EOF
# XORB SSL/TLS Configuration for Apache
SSLEngine on
SSLCertificateFile /etc/ssl/certs/server-cert.pem
SSLCertificateKeyFile /etc/ssl/private/server-key.pem
SSLCertificateChainFile /etc/ssl/certs/ca-cert.pem
SSLCACertificateFile /etc/ssl/certs/ca-cert.pem

# SSL Configuration
SSLProtocol all -SSLv3 -TLSv1 -TLSv1.1
SSLCipherSuite ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305
SSLHonorCipherOrder off
SSLSessionCache shmcb:logs/ssl_cache(512000)
SSLSessionCacheTimeout 300

# Security headers
Header set Strict-Transport-Security "max-age=63072000; includeSubDomains; preload"
Header set X-Frame-Options DENY
Header set X-Content-Type-Options nosniff
Header set X-XSS-Protection "1; mode=block"
Header set Referrer-Policy "strict-origin-when-cross-origin"

# Optional: Enable mutual TLS (uncomment for mTLS)
# SSLVerifyClient require
# SSLVerifyDepth 2
EOF

# Create Docker Compose SSL configuration
log_step "Creating Docker Compose SSL configuration..."
cat > "$SSL_DIR/docker-compose.ssl.yml" << EOF
version: '3.8'

services:
  nginx-ssl:
    image: nginx:alpine
    container_name: xorb_nginx_ssl
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./ssl/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl/certs:/etc/ssl/certs:ro
      - ./ssl/private:/etc/ssl/private:ro
    networks:
      - xorb_network
    depends_on:
      - api
      - orchestrator
      - worker

networks:
  xorb_network:
    external: true
EOF

# Create NGINX main configuration with SSL
cat > "$SSL_DIR/nginx/nginx.conf" << EOF
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        server host.docker.internal:8000;
    }

    upstream orchestrator_backend {
        server host.docker.internal:8080;
    }

    upstream worker_backend {
        server host.docker.internal:9000;
    }

    # Redirect HTTP to HTTPS
    server {
        listen 80;
        server_name _;
        return 301 https://\$server_name\$request_uri;
    }

    # Main HTTPS server
    server {
        listen 443 ssl http2;
        server_name $DOMAIN_NAME;

        include /etc/ssl/ssl.conf;

        # API routes
        location /api/ {
            proxy_pass http://api_backend/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }

        # Orchestrator routes
        location /orchestrator/ {
            proxy_pass http://orchestrator_backend/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }

        # Worker routes
        location /worker/ {
            proxy_pass http://worker_backend/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }

        # Default route to API docs
        location / {
            proxy_pass http://api_backend/;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
    }
}
EOF

# Create certificate validation script
log_step "Creating certificate validation script..."
cat > "$SSL_DIR/validate-certificates.sh" << 'EOF'
#!/bin/bash

echo "🔍 XORB Certificate Validation"
echo "============================="

SSL_DIR="/root/Xorb/ssl"
CERTS_DIR="$SSL_DIR/certs"
PRIVATE_DIR="$SSL_DIR/private"

# Check CA certificate
echo "📋 CA Certificate:"
openssl x509 -in "$CERTS_DIR/ca-cert.pem" -text -noout | grep -E "(Subject|Issuer|Not Before|Not After)"

echo ""
echo "📋 Server Certificate:"
openssl x509 -in "$CERTS_DIR/server-cert.pem" -text -noout | grep -E "(Subject|Issuer|Not Before|Not After|DNS:|IP Address:)"

echo ""
echo "🔗 Certificate Chain Validation:"
openssl verify -CAfile "$CERTS_DIR/ca-cert.pem" "$CERTS_DIR/server-cert.pem"

echo ""
echo "🔐 Private Key Validation:"
SERVER_KEY_HASH=$(openssl rsa -in "$PRIVATE_DIR/server-key.pem" -modulus -noout | openssl md5)
SERVER_CERT_HASH=$(openssl x509 -in "$CERTS_DIR/server-cert.pem" -modulus -noout | openssl md5)

if [ "$SERVER_KEY_HASH" = "$SERVER_CERT_HASH" ]; then
    echo "✅ Private key matches certificate"
else
    echo "❌ Private key does not match certificate"
fi

echo ""
echo "🎯 Certificate Expiry Check:"
EXPIRY_DATE=$(openssl x509 -in "$CERTS_DIR/server-cert.pem" -enddate -noout | cut -d= -f2)
EXPIRY_EPOCH=$(date -d "$EXPIRY_DATE" +%s)
CURRENT_EPOCH=$(date +%s)
DAYS_UNTIL_EXPIRY=$(( (EXPIRY_EPOCH - CURRENT_EPOCH) / 86400 ))

echo "Certificate expires: $EXPIRY_DATE"
echo "Days until expiry: $DAYS_UNTIL_EXPIRY"

if [ $DAYS_UNTIL_EXPIRY -lt 30 ]; then
    echo "⚠️  Certificate expires in less than 30 days!"
elif [ $DAYS_UNTIL_EXPIRY -lt 90 ]; then
    echo "⚠️  Certificate expires in less than 90 days"
else
    echo "✅ Certificate has sufficient validity period"
fi
EOF

chmod +x "$SSL_DIR/validate-certificates.sh"

# Create certificate renewal script
log_step "Creating certificate renewal script..."
cat > "$SSL_DIR/renew-certificates.sh" << 'EOF'
#!/bin/bash

echo "🔄 XORB Certificate Renewal"
echo "=========================="

SSL_DIR="/root/Xorb/ssl"
CERTS_DIR="$SSL_DIR/certs"
PRIVATE_DIR="$SSL_DIR/private"
CSR_DIR="$SSL_DIR/csr"

# Backup existing certificates
BACKUP_DIR="$SSL_DIR/backup/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp "$CERTS_DIR/server-cert.pem" "$BACKUP_DIR/" 2>/dev/null || true
echo "✅ Certificates backed up to: $BACKUP_DIR"

# Generate new server certificate
openssl req -new -key "$PRIVATE_DIR/server-key.pem" \
    -out "$CSR_DIR/server.csr" -config "$SSL_DIR/server.conf"

openssl x509 -req -in "$CSR_DIR/server.csr" -CA "$CERTS_DIR/ca-cert.pem" \
    -CAkey "$PRIVATE_DIR/ca-key.pem" -CAcreateserial \
    -out "$CERTS_DIR/server-cert.pem" -days 365 \
    -extensions v3_req -extfile "$SSL_DIR/server_cert.conf"

# Recreate certificate bundle
cat "$CERTS_DIR/server-cert.pem" "$CERTS_DIR/ca-cert.pem" > "$CERTS_DIR/server-bundle.pem"

echo "✅ Server certificate renewed"
echo "🔄 Please restart NGINX/Apache to load new certificates"
EOF

chmod +x "$SSL_DIR/renew-certificates.sh"

# Set proper permissions
log_step "Setting certificate permissions..."
chmod 644 "$CERTS_DIR"/*.pem
chmod 600 "$PRIVATE_DIR"/*.pem
chown -R root:root "$SSL_DIR"

# Validate certificates
log_step "Validating generated certificates..."
"$SSL_DIR/validate-certificates.sh"

echo ""
log_info "SSL/TLS automation setup complete!"
echo ""
echo "📋 Generated certificates:"
echo "   - CA Certificate: $CERTS_DIR/ca-cert.pem"
echo "   - Server Certificate: $CERTS_DIR/server-cert.pem"
echo "   - Server Private Key: $PRIVATE_DIR/server-key.pem"
echo "   - Certificate Bundle: $CERTS_DIR/server-bundle.pem"
echo ""
echo "🔧 Available configurations:"
echo "   - NGINX: $SSL_DIR/nginx/"
echo "   - Apache: $SSL_DIR/apache/"
echo "   - Docker Compose: $SSL_DIR/docker-compose.ssl.yml"
echo ""
echo "🛠️  Management scripts:"
echo "   - Validate: $SSL_DIR/validate-certificates.sh"
echo "   - Renew: $SSL_DIR/renew-certificates.sh"
echo ""
echo "🚀 To deploy SSL:"
echo "   docker-compose -f $SSL_DIR/docker-compose.ssl.yml up -d"
