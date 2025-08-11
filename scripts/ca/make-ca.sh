#!/bin/bash
# Certificate Authority Setup for XORB Platform
# Creates root CA and intermediate CA for internal mTLS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRETS_DIR="${SCRIPT_DIR}/../../secrets/tls"
CA_DIR="${SECRETS_DIR}/ca"

# Certificate validity periods
ROOT_CA_DAYS=3650  # 10 years for root CA
INTERMEDIATE_DAYS=1825  # 5 years for intermediate CA

# Certificate subject information
ROOT_SUBJECT="/C=US/ST=CA/L=San Francisco/O=XORB Platform/OU=Security/CN=XORB Root CA"
INTERMEDIATE_SUBJECT="/C=US/ST=CA/L=San Francisco/O=XORB Platform/OU=Security/CN=XORB Intermediate CA"

echo "🔐 Setting up XORB Certificate Authority..."

# Create directory structure
mkdir -p "${CA_DIR}"/{root,intermediate}/{certs,crl,newcerts,private}
chmod 700 "${CA_DIR}"/{root,intermediate}/private

# Initialize CA databases
touch "${CA_DIR}/root/index.txt"
touch "${CA_DIR}/intermediate/index.txt"
echo 1000 > "${CA_DIR}/root/serial"
echo 1000 > "${CA_DIR}/intermediate/serial"
echo 1000 > "${CA_DIR}/intermediate/crlnumber"

# Create OpenSSL configuration for Root CA
cat > "${CA_DIR}/root/openssl.cnf" << 'EOF'
[ ca ]
default_ca = CA_default

[ CA_default ]
dir               = REPLACE_CA_DIR/root
certs             = $dir/certs
crl_dir           = $dir/crl
new_certs_dir     = $dir/newcerts
database          = $dir/index.txt
serial            = $dir/serial
RANDFILE          = $dir/private/.rand

private_key       = $dir/private/ca.key.pem
certificate       = $dir/certs/ca.cert.pem

crlnumber         = $dir/crlnumber
crl               = $dir/crl/ca.crl.pem
crl_extensions    = crl_ext
default_crl_days  = 30

default_md        = sha256
name_opt          = ca_default
cert_opt          = ca_default
default_days      = 375
preserve          = no
policy            = policy_strict

[ policy_strict ]
countryName             = match
stateOrProvinceName     = match
organizationName        = match
organizationalUnitName  = optional
commonName              = supplied
emailAddress            = optional

[ req ]
default_bits        = 4096
distinguished_name  = req_distinguished_name
string_mask         = utf8only
default_md          = sha256
x509_extensions     = v3_ca

[ req_distinguished_name ]
countryName                     = Country Name (2 letter code)
stateOrProvinceName             = State or Province Name
localityName                    = Locality Name
0.organizationName              = Organization Name
organizationalUnitName          = Organizational Unit Name
commonName                      = Common Name
emailAddress                    = Email Address

[ v3_ca ]
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid:always,issuer
basicConstraints = critical, CA:true
keyUsage = critical, digitalSignature, cRLSign, keyCertSign

[ v3_intermediate_ca ]
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid:always,issuer
basicConstraints = critical, CA:true, pathlen:0
keyUsage = critical, digitalSignature, cRLSign, keyCertSign

[ crl_ext ]
authorityKeyIdentifier=keyid:always
EOF

# Create OpenSSL configuration for Intermediate CA
cat > "${CA_DIR}/intermediate/openssl.cnf" << 'EOF'
[ ca ]
default_ca = CA_default

[ CA_default ]
dir               = REPLACE_CA_DIR/intermediate
certs             = $dir/certs
crl_dir           = $dir/crl
new_certs_dir     = $dir/newcerts
database          = $dir/index.txt
serial            = $dir/serial
RANDFILE          = $dir/private/.rand

private_key       = $dir/private/intermediate.key.pem
certificate       = $dir/certs/intermediate.cert.pem

crlnumber         = $dir/crlnumber
crl               = $dir/crl/intermediate.crl.pem
crl_extensions    = crl_ext
default_crl_days  = 30

default_md        = sha256
name_opt          = ca_default
cert_opt          = ca_default
default_days      = 30
preserve          = no
policy            = policy_loose

[ policy_loose ]
countryName             = optional
stateOrProvinceName     = optional
localityName            = optional
organizationName        = optional
organizationalUnitName  = optional
commonName              = supplied
emailAddress            = optional

[ req ]
default_bits        = 2048
distinguished_name  = req_distinguished_name
string_mask         = utf8only
default_md          = sha256

[ req_distinguished_name ]
countryName                     = Country Name (2 letter code)
stateOrProvinceName             = State or Province Name
localityName                    = Locality Name
0.organizationName              = Organization Name
organizationalUnitName          = Organizational Unit Name
commonName                      = Common Name
emailAddress                    = Email Address

[ server_cert ]
basicConstraints = CA:FALSE
nsCertType = server
nsComment = "XORB Server Certificate"
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid,issuer:always
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth

[ client_cert ]
basicConstraints = CA:FALSE
nsCertType = client, email
nsComment = "XORB Client Certificate"
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid,issuer:always
keyUsage = critical, digitalSignature
extendedKeyUsage = clientAuth, emailProtection

[ crl_ext ]
authorityKeyIdentifier=keyid:always
EOF

# Replace placeholder paths
sed -i "s|REPLACE_CA_DIR|${CA_DIR}|g" "${CA_DIR}/root/openssl.cnf"
sed -i "s|REPLACE_CA_DIR|${CA_DIR}|g" "${CA_DIR}/intermediate/openssl.cnf"

echo "📝 Generating Root CA private key..."
openssl genrsa -aes256 -passout pass:xorb-root-ca-key \
    -out "${CA_DIR}/root/private/ca.key.pem" 4096
chmod 400 "${CA_DIR}/root/private/ca.key.pem"

echo "🏛️ Creating Root CA certificate..."
openssl req -config "${CA_DIR}/root/openssl.cnf" \
    -key "${CA_DIR}/root/private/ca.key.pem" \
    -new -x509 -days ${ROOT_CA_DAYS} -sha256 -extensions v3_ca \
    -passin pass:xorb-root-ca-key \
    -subj "${ROOT_SUBJECT}" \
    -out "${CA_DIR}/root/certs/ca.cert.pem"
chmod 444 "${CA_DIR}/root/certs/ca.cert.pem"

echo "🔍 Verifying Root CA certificate..."
openssl x509 -noout -text -in "${CA_DIR}/root/certs/ca.cert.pem"

echo "📝 Generating Intermediate CA private key..."
openssl genrsa -aes256 -passout pass:xorb-intermediate-ca-key \
    -out "${CA_DIR}/intermediate/private/intermediate.key.pem" 4096
chmod 400 "${CA_DIR}/intermediate/private/intermediate.key.pem"

echo "📋 Creating Intermediate CA certificate signing request..."
openssl req -config "${CA_DIR}/intermediate/openssl.cnf" -new -sha256 \
    -key "${CA_DIR}/intermediate/private/intermediate.key.pem" \
    -passin pass:xorb-intermediate-ca-key \
    -subj "${INTERMEDIATE_SUBJECT}" \
    -out "${CA_DIR}/intermediate/csr/intermediate.csr.pem"

echo "🏛️ Signing Intermediate CA certificate with Root CA..."
openssl ca -config "${CA_DIR}/root/openssl.cnf" -extensions v3_intermediate_ca \
    -days ${INTERMEDIATE_DAYS} -notext -md sha256 -batch \
    -passin pass:xorb-root-ca-key \
    -in "${CA_DIR}/intermediate/csr/intermediate.csr.pem" \
    -out "${CA_DIR}/intermediate/certs/intermediate.cert.pem"
chmod 444 "${CA_DIR}/intermediate/certs/intermediate.cert.pem"

echo "🔗 Creating certificate chain..."
cat "${CA_DIR}/intermediate/certs/intermediate.cert.pem" \
    "${CA_DIR}/root/certs/ca.cert.pem" > "${CA_DIR}/ca-chain.cert.pem"
chmod 444 "${CA_DIR}/ca-chain.cert.pem"

# Create convenient symlinks
ln -sf "${CA_DIR}/ca-chain.cert.pem" "${CA_DIR}/ca.pem"
ln -sf "${CA_DIR}/root/certs/ca.cert.pem" "${CA_DIR}/root-ca.pem"
ln -sf "${CA_DIR}/intermediate/certs/intermediate.cert.pem" "${CA_DIR}/intermediate-ca.pem"

echo "🔍 Verifying Intermediate CA certificate..."
openssl x509 -noout -text -in "${CA_DIR}/intermediate/certs/intermediate.cert.pem"

echo "✅ Certificate Authority setup complete!"
echo "📁 Root CA: ${CA_DIR}/root/certs/ca.cert.pem"
echo "📁 Intermediate CA: ${CA_DIR}/intermediate/certs/intermediate.cert.pem"
echo "📁 CA Chain: ${CA_DIR}/ca-chain.cert.pem"
echo ""
echo "⚠️  IMPORTANT: Store the CA private keys securely!"
echo "   Root CA Key: ${CA_DIR}/root/private/ca.key.pem"
echo "   Intermediate CA Key: ${CA_DIR}/intermediate/private/intermediate.key.pem"
echo "   Passwords: xorb-root-ca-key / xorb-intermediate-ca-key"