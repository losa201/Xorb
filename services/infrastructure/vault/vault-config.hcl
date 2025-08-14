ui = true

storage "file" {
  path = "/vault/data"
}

# HTTP listener
listener "tcp" {
  address = "0.0.0.0:8200"
  tls_disable = true
}

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
