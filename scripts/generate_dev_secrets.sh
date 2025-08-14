#!/bin/bash
# Development secret generation script
# Generates secure secrets for local development

set -e

echo "🔐 Generating development secrets for XORB platform"

# Check if .env exists and warn user
if [ -f .env ]; then
    echo "⚠️  WARNING: .env file already exists"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Operation cancelled"
        exit 1
    fi
    echo "🗑️  Backing up existing .env to .env.backup"
    cp .env .env.backup
fi

# Generate secure secrets
echo "🔑 Generating cryptographically secure secrets..."

JWT_SECRET=$(openssl rand -hex 32)
DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
ENCRYPTION_SALT=$(openssl rand -base64 32)

# Create development .env
cat > .env << EOF
# XORB Platform - Development Environment
# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# WARNING: Never commit this file to version control

# Application Settings
APP_NAME="XORB Enterprise Cybersecurity Platform"
APP_VERSION="3.1.0"
ENVIRONMENT="development"
DEBUG=false

# Security Configuration (Generated)
JWT_SECRET="${JWT_SECRET}"
JWT_ALGORITHM="HS256"
JWT_EXPIRATION_MINUTES=15

# Database Configuration (Generated)
DATABASE_URL="postgresql://xorb_user:${DB_PASSWORD}@localhost:5432/xorb_enterprise"
DATABASE_URL_TEMPLATE="postgresql://xorb_user:PASSWORD@localhost:5432/xorb_enterprise"

# Redis Configuration (Generated)
REDIS_URL="redis://:${REDIS_PASSWORD}@localhost:6379/0"
REDIS_URL_TEMPLATE="redis://:PASSWORD@localhost:6379/0"

# Encryption
ENCRYPTION_SALT="${ENCRYPTION_SALT}"

# API Configuration
API_PREFIX="/api/v1"
API_HOST="0.0.0.0"
API_PORT=8000

# Security Settings (Secure defaults)
MIN_PASSWORD_LENGTH=12
REQUIRE_MFA=true
MAX_LOGIN_ATTEMPTS=5

# Rate Limiting (Development)
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# CORS Configuration (Development)
CORS_ALLOW_ORIGINS="http://localhost:3000,http://localhost:8080"
CORS_ALLOW_CREDENTIALS=true

# Feature Flags (Development)
ENABLE_DEBUG_ENDPOINTS=false
ENABLE_ENTERPRISE_FEATURES=true

# External API Keys (Configure as needed)
NVIDIA_API_KEY=""
OPENROUTER_API_KEY=""
OPENAI_API_KEY=""

EOF

echo "✅ Development secrets generated successfully!"
echo ""
echo "📁 Created files:"
echo "   - .env (development configuration with secure secrets)"
echo "   - .env.backup (backup of previous .env if existed)"
echo ""
echo "🔐 Generated secrets:"
echo "   - JWT_SECRET: 32-character cryptographic key"
echo "   - DATABASE_PASSWORD: 25-character secure password"
echo "   - REDIS_PASSWORD: 25-character secure password"
echo "   - ENCRYPTION_SALT: Base64 cryptographic salt"
echo ""
echo "⚠️  SECURITY REMINDERS:"
echo "   1. Never commit .env to version control"
echo "   2. Rotate secrets regularly"
echo "   3. Use Vault for production secrets"
echo "   4. Backup .env securely if needed"
echo ""
echo "🚀 Next steps:"
echo "   1. Review configuration in .env"
echo "   2. Start services: docker-compose up"
echo "   3. Verify health: curl http://localhost:8000/api/v1/health"