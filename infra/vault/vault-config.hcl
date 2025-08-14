ui = true

storage "file" {
  path = "/vault/data"
}

# HTTP listener (development mode - use vault-production.hcl for production)
listener "tcp" {
  address = "127.0.0.1:8200"
  tls_disable = true
}

# Production listener (commented out for dev)
# listener "tcp" {
#   address = "0.0.0.0:8200"
#   tls_disable = false
#   tls_cert_file = "/vault/tls/vault.crt"
#   tls_key_file = "/vault/tls/vault.key"
# }

# Enable API addr
api_addr = "http://0.0.0.0:8200"
cluster_addr = "http://0.0.0.0:8201"

# Disable mlock for development
disable_mlock = true

# Default TTL values
default_lease_ttl = "168h"
max_lease_ttl = "720h"

# Enable secret engines
path "secret/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "database/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "auth/jwt/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Transit engine for encryption
path "transit/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
