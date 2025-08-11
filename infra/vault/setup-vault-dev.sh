#!/bin/bash
# Development Vault setup for XORB
# This script initializes a development Vault instance with secrets

set -e

echo "🔐 Setting up development Vault for XORB..."

# Check if vault is installed
if ! command -v vault &> /dev/null; then
    echo "❌ Vault not found. Please install Vault first:"
    echo "   wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg"
    echo "   echo \"deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com bookworm main\" | sudo tee /etc/apt/sources.list.d/hashicorp.list"
    echo "   sudo apt update && sudo apt install vault"
    exit 1
fi

# Create vault data directory
mkdir -p vault-data

# Start vault in development mode (background)
echo "🚀 Starting Vault in development mode..."
vault server -config=vault-dev-config.hcl &
VAULT_PID=$!

# Wait for vault to start
sleep 3

# Set vault address
export VAULT_ADDR='http://127.0.0.1:8200'

# Check if vault is already initialized
if vault status &> /dev/null; then
    echo "ℹ️  Vault already running"
else
    echo "⏳ Waiting for Vault to start..."
    sleep 5
fi

# Initialize vault (only in dev mode with known token)
echo "🔑 Initializing Vault..."
export VAULT_DEV_ROOT_TOKEN_ID="xorb-dev-token-$(date +%s)"

# Kill existing vault and restart in dev mode
kill $VAULT_PID 2>/dev/null || true
sleep 2

# Start in dev mode with known token
vault server -dev -dev-root-token-id="$VAULT_DEV_ROOT_TOKEN_ID" -dev-listen-address="127.0.0.1:8200" &
VAULT_PID=$!

sleep 5

# Set token for CLI
export VAULT_TOKEN="$VAULT_DEV_ROOT_TOKEN_ID"

echo "✅ Vault started with token: $VAULT_DEV_ROOT_TOKEN_ID"

# Enable KV secrets engine
echo "🔧 Enabling KV secrets engine..."
vault secrets enable -path=secret kv-v2

# Create XORB secrets
echo "📝 Creating XORB secrets..."

# Generate secure random secrets
JWT_SECRET=$(openssl rand -base64 64 | tr -d '\n')
API_KEY=$(openssl rand -hex 32)
DB_PASSWORD="SecureDB_$(openssl rand -base64 16 | tr -d '\n' | tr '/' '_')"

# Store secrets in Vault
vault kv put secret/xorb/config \
    JWT_SECRET="$JWT_SECRET" \
    XORB_API_KEY="$API_KEY" \
    DB_PASSWORD="$DB_PASSWORD" \
    DB_USER="xorb_user" \
    DB_NAME="xorb_secure" \
    DB_HOST="localhost" \
    DB_PORT="5432"

# Store additional secrets
vault kv put secret/xorb/external \
    NVIDIA_API_KEY="" \
    OPENROUTER_API_KEY="" \
    AZURE_CLIENT_SECRET="" \
    GOOGLE_CLIENT_SECRET="" \
    GITHUB_CLIENT_SECRET=""

echo "✅ Vault setup complete!"
echo ""
echo "🔐 Vault Information:"
echo "   Address: http://127.0.0.1:8200"
echo "   Root Token: $VAULT_DEV_ROOT_TOKEN_ID"
echo "   UI: http://127.0.0.1:8200/ui"
echo ""
echo "💾 Save this token to ~/.vault-token:"
echo "   echo '$VAULT_DEV_ROOT_TOKEN_ID' > ~/.vault-token"
echo ""
echo "🐳 To use with Docker, add to docker-compose.yml:"
echo "   environment:"
echo "     - VAULT_URL=http://vault:8200"
echo "     - VAULT_TOKEN=$VAULT_DEV_ROOT_TOKEN_ID"
echo ""

# Save token for development
echo "$VAULT_DEV_ROOT_TOKEN_ID" > ~/.vault-token

echo "🎉 Development Vault is ready!"
echo "   Vault PID: $VAULT_PID"
echo "   To stop: kill $VAULT_PID"