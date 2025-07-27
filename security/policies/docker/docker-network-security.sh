#!/bin/bash

# Docker Network Security Configuration
echo "🐳 Configuring Docker network security..."

# Create isolated networks for different environments
docker network create --driver bridge \
  --subnet=172.20.0.0/16 \
  --ip-range=172.20.1.0/24 \
  --gateway=172.20.0.1 \
  xorb-production-network 2>/dev/null || echo "Production network exists"

docker network create --driver bridge \
  --subnet=172.21.0.0/16 \
  --ip-range=172.21.1.0/24 \
  --gateway=172.21.0.1 \
  xorb-staging-network 2>/dev/null || echo "Staging network exists"

docker network create --driver bridge \
  --subnet=172.22.0.0/16 \
  --ip-range=172.22.1.0/24 \
  --gateway=172.22.0.1 \
  xorb-development-network 2>/dev/null || echo "Development network exists"

# Create DMZ network for external-facing services
docker network create --driver bridge \
  --subnet=172.30.0.0/16 \
  --ip-range=172.30.1.0/24 \
  --gateway=172.30.0.1 \
  xorb-dmz-network 2>/dev/null || echo "DMZ network exists"

echo "✅ Docker networks configured"
