# Vault Production Configuration
# Secure configuration for production environments

ui = true
disable_mlock = false

# File storage (consider Consul for HA)
storage "file" {
  path = "/vault/data"
}

# TLS-enabled listener for production
listener "tcp" {
  address       = "0.0.0.0:8200"
  tls_disable   = false
  tls_cert_file = "/vault/tls/vault.crt"
  tls_key_file  = "/vault/tls/vault.key"
  
  # Security headers
  tls_min_version     = "tls12"
  tls_cipher_suites   = "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384"
  tls_prefer_server_cipher_suites = true
  
  # Client certificate authentication (optional)
  tls_require_and_verify_client_cert = false
}

# API and cluster addresses
api_addr     = "https://vault.xorb-security.com:8200"
cluster_addr = "https://vault.xorb-security.com:8201"

# TTL values
default_lease_ttl = "24h"
max_lease_ttl = "168h"  # 1 week

# Disable raw endpoint
raw_storage_endpoint = false

# Enable audit logging
audit {
  enabled = true
}

# Seal configuration (use auto-unseal in production)
# seal "awskms" {
#   region     = "us-west-2"
#   kms_key_id = "alias/vault-unseal-key"
# }

# Log level
log_level = "INFO"
log_format = "json"

# Performance and limits
cluster_name = "xorb-vault-cluster"

# Plugins directory
plugin_directory = "/vault/plugins"