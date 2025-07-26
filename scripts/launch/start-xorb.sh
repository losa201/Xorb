#!/bin/bash
# XORB Platform Startup Script

export PATH="$HOME/.local/bin:$PATH"

echo "🚀 Starting XORB Security Platform..."

# Check if Redis is running
if ! pgrep -x "redis-server" > /dev/null; then
    echo "⚠️  Redis not running. Please start Redis first:"
    echo "   sudo systemctl start redis-server"
    exit 1
fi

# Start components in background
echo "Starting orchestrator..."
poetry run python orchestration/orchestrator.py &
ORCHESTRATOR_PID=$!

echo "Starting LLM updater..."
poetry run python orchestration/llm_updater.py --continuous &
LLM_PID=$!

echo "Starting monitoring dashboard..."
poetry run python monitoring/dashboard.py &
DASHBOARD_PID=$!

# Function to cleanup on exit
cleanup() {
    echo "🛑 Shutting down XORB..."
    kill $ORCHESTRATOR_PID $LLM_PID $DASHBOARD_PID 2>/dev/null
    wait
    echo "✅ XORB stopped"
}

trap cleanup EXIT

echo "✅ XORB Platform is running!"
echo "   - Orchestrator PID: $ORCHESTRATOR_PID"
echo "   - LLM Updater PID: $LLM_PID" 
echo "   - Dashboard PID: $DASHBOARD_PID"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for all background processes
wait
