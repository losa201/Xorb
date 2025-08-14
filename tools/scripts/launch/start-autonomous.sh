#!/bin/bash
# XORB Autonomous Mode Startup Script

echo "🧠 Starting XORB in Autonomous Mode..."

# Set environment variables
export XORB_ENVIRONMENT=autonomous
export DATABASE_URL=postgresql://xorb:xorb_secure_2024@postgres:5432/xorb_ptaas
export POSTGRES_PASSWORD=xorb_secure_2024
export ALLOW_EMPTY_PASSWORD=yes
export REDIS_URL=redis://redis:6379/0
export NATS_URL=nats://nats:4222
export TEMPORAL_HOST=temporal:7233
export NVIDIA_API_KEY=${NVIDIA_API_KEY:-"your_nvidia_api_key_here"}
export JWT_SECRET_KEY=autonomous-xorb-secret-key-2024
export OPENAI_API_KEY=dummy-key
export ANTHROPIC_API_KEY=dummy-key
export STRIPE_SECRET_KEY=dummy-key
export STRIPE_WEBHOOK_SECRET=dummy-key
export USDC_CONTRACT_ADDRESS=dummy-address
export STRIPE_GROWTH_PRICE_ID=dummy-id
export STRIPE_ELITE_PRICE_ID=dummy-id
export STRIPE_ENTERPRISE_PRICE_ID=dummy-id
export AWS_ACCESS_KEY_ID=dummy-key
export AWS_SECRET_ACCESS_KEY=dummy-key
export AWS_REGION=us-west-2

# Clean up any existing networks
echo "🧹 Cleaning up existing networks..."
docker network prune -f

# Start core infrastructure first
echo "🗄️ Starting core infrastructure..."
docker-compose -f compose/docker-compose.yml up -d postgres redis nats temporal

# Wait for services to be ready
echo "⏳ Waiting for services to initialize..."
sleep 30

# Check service health
echo "🏥 Checking service health..."
docker-compose -f compose/docker-compose.yml ps

echo "✅ XORB Autonomous Mode startup complete!"
