// Development Vault configuration for XORB
// This is for development only - production should use proper Vault deployment

storage "file" {
  path = "./vault-data"
}

listener "tcp" {
  address     = "127.0.0.1:8200"
  tls_disable = 1
}

ui = true

// Development settings - DO NOT USE IN PRODUCTION
disable_mlock = true
api_addr = "http://127.0.0.1:8200"
cluster_addr = "https://127.0.0.1:8201"