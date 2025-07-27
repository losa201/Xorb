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
