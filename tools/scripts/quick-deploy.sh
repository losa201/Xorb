#!/bin/bash
# Quick deployment wrapper for XORB Autonomous Orchestrator
# This script provides a simple interface to the comprehensive deployment system

set -euo pipefail

# Default configuration
DEFAULT_POSTGRES_PASSWORD="xorb-postgres-$(date +%Y%m%d)"
DEFAULT_REDIS_PASSWORD="xorb-redis-$(date +%Y%m%d)"
DEFAULT_JWT_SECRET="$(openssl rand -hex 32)"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 XORB Autonomous Orchestrator - Quick Deploy${NC}"
echo "============================================="
echo ""

# Set environment variables if not already set
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$DEFAULT_POSTGRES_PASSWORD}"
export REDIS_PASSWORD="${REDIS_PASSWORD:-$DEFAULT_REDIS_PASSWORD}"
export JWT_SECRET="${JWT_SECRET:-$DEFAULT_JWT_SECRET}"

echo -e "${GREEN}✅ Environment configured:${NC}"
echo "  📊 Postgres Password: [HIDDEN]"
echo "  ⚡ Redis Password: [HIDDEN]"
echo "  🔐 JWT Secret: [HIDDEN]"
echo ""

echo -e "${YELLOW}⚠️  Important: Save these credentials securely!${NC}"
echo "  POSTGRES_PASSWORD=$POSTGRES_PASSWORD"
echo "  REDIS_PASSWORD=$REDIS_PASSWORD"
echo "  JWT_SECRET=$JWT_SECRET"
echo ""

read -p "Press Enter to continue with deployment or Ctrl+C to abort..."
echo ""

# Execute the main deployment script
exec "$(dirname "$0")/deploy-production.sh" "$@"
