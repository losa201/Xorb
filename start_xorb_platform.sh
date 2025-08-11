#!/bin/bash
# XORB Enterprise Cybersecurity Platform Startup Script
# Principal Auditor - Production Ready Implementation

set -e

echo "🛡️  XORB Enterprise Cybersecurity Platform"
echo "=========================================="
echo "Starting platform with production-ready configuration..."
echo

# Check if we're in the right directory
if [ ! -f "src/api/app/main.py" ]; then
    echo "❌ Error: Must run from Xorb project root directory"
    echo "Usage: cd Xorb && ./start_xorb_platform.sh"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $PYTHON_VERSION"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found - using default configuration"
else
    echo "✅ Environment configuration loaded from .env"
fi

# Check dependencies
echo "📦 Checking core dependencies..."
python3 -c "import fastapi, uvicorn, pydantic" 2>/dev/null && echo "✅ Core dependencies installed" || {
    echo "❌ Missing core dependencies. Installing..."
    pip install -r requirements.txt
}

# Validate configuration
echo "🔧 Validating configuration..."
python3 -c "
from src.api.app.core.config import get_config_manager
config = get_config_manager()
issues = config.validate_configuration()
if issues:
    print('⚠️  Configuration warnings:')
    for issue in issues:
        print(f'  - {issue}')
else:
    print('✅ Configuration validation passed')
print(f'🏢 Environment: {config.app_settings.environment}')
print(f'🔐 Security: MFA={config.app_settings.require_mfa}, Rate Limiting={config.app_settings.rate_limit_enabled}')
"

echo
echo "🚀 Starting XORB Platform..."
echo "   API Endpoint: http://localhost:8000"
echo "   Documentation: http://localhost:8000/docs"
echo "   Health Check: http://localhost:8000/api/v1/health"
echo

# Start the server
exec python3 -m uvicorn src.api.app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --access-log \
    --log-level info