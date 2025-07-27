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
