#!/bin/bash
set -e

# Wait for Vault to be ready
until vault status; do
  echo "Waiting for Vault to start..."
  sleep 5
done

# Initialize Vault if not already initialized
if ! vault status | grep -q "Initialized.*true"; then
  echo "Initializing Vault..."
  vault operator init -key-shares=5 -key-threshold=3 > /vault/init-keys.txt

  echo "Vault initialized. Unsealing..."
  # Extract unseal keys and unseal
  UNSEAL_KEY_1=$(grep 'Unseal Key 1:' /vault/init-keys.txt | awk '{print $NF}')
  UNSEAL_KEY_2=$(grep 'Unseal Key 2:' /vault/init-keys.txt | awk '{print $NF}')
  UNSEAL_KEY_3=$(grep 'Unseal Key 3:' /vault/init-keys.txt | awk '{print $NF}')

  vault operator unseal $UNSEAL_KEY_1
  vault operator unseal $UNSEAL_KEY_2
  vault operator unseal $UNSEAL_KEY_3

  # Extract root token
  ROOT_TOKEN=$(grep 'Initial Root Token:' /vault/init-keys.txt | awk '{print $NF}')
  export VAULT_TOKEN=$ROOT_TOKEN

  echo "Setting up secret engines..."

  # Enable KV secrets engine
  vault secrets enable -path=secret kv-v2

  # Enable database secrets engine
  vault secrets enable database

  # Enable transit secrets engine for encryption
  vault secrets enable transit

  # Create encryption key for JWT signing
  vault write -f transit/keys/jwt-signing

  # Create database config
  vault write database/config/postgresql \
    plugin_name=postgresql-database-plugin \
    connection_url="postgresql://{{username}}:{{password}}@postgres:5432/xorb_db?sslmode=disable" \
    allowed_roles="xorb-app" \
    username="xorb_admin" \
    password="$POSTGRES_ADMIN_PASSWORD"

  # Create database role
  vault write database/roles/xorb-app \
    db_name=postgresql \
    creation_statements="CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}'; GRANT CONNECT ON DATABASE xorb_db TO \"{{name}}\"; GRANT USAGE ON SCHEMA public TO \"{{name}}\"; GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO \"{{name}}\"; ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO \"{{name}}\";" \
    default_ttl="1h" \
    max_ttl="24h"

  # Store initial secrets
  vault kv put secret/xorb/jwt secret="$(openssl rand -base64 32)"
  vault kv put secret/xorb/redis password="$(openssl rand -base64 32)"

  # Create policy for application
  vault policy write xorb-app - <<EOF
# Read application secrets
path "secret/data/xorb/*" {
  capabilities = ["read"]
}

# Generate database credentials
path "database/creds/xorb-app" {
  capabilities = ["read"]
}

# Use transit engine for JWT signing
path "transit/encrypt/jwt-signing" {
  capabilities = ["update"]
}

path "transit/decrypt/jwt-signing" {
  capabilities = ["update"]
}

path "transit/sign/jwt-signing" {
  capabilities = ["update"]
}

path "transit/verify/jwt-signing" {
  capabilities = ["update"]
}
EOF

  # Create auth method for applications
  vault auth enable jwt

  echo "Vault setup complete!"
  echo "Root token: $ROOT_TOKEN"
  echo "Save the init-keys.txt file securely!"

else
  echo "Vault already initialized"
fi
