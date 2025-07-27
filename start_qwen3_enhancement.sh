#!/bin/bash
"""
XORB Qwen3-Coder Autonomous Enhancement Startup Script
Launches the continuous code improvement system
"""

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🧠 XORB QWEN3-CODER AUTONOMOUS ENHANCEMENT SYSTEM${NC}"
echo -e "${CYAN}================================================================${NC}"

# Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python 3 found${NC}"

# Check Git
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Git found${NC}"

# Check if we're in a Git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️ Not in a Git repository, initializing...${NC}"
    git init
    git add .
    git commit -m "Initial commit before Qwen3 enhancement"
fi
echo -e "${GREEN}✅ Git repository ready${NC}"

# Create necessary directories
echo -e "${BLUE}📁 Setting up directories...${NC}"
mkdir -p logs
mkdir -p backups/enhancements
echo -e "${GREEN}✅ Directories created${NC}"

# Install required packages if needed
echo -e "${BLUE}📦 Checking Python dependencies...${NC}"
python3 -c "import aiofiles" 2>/dev/null || pip install aiofiles
python3 -c "import asyncio" 2>/dev/null || echo -e "${GREEN}✅ asyncio built-in${NC}"
echo -e "${GREEN}✅ Dependencies ready${NC}"

# Set up environment variables if needed
echo -e "${BLUE}🔧 Setting up environment...${NC}"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export QWEN3_ENHANCEMENT_MODE="autonomous"
export QWEN3_LOG_LEVEL="INFO"
echo -e "${GREEN}✅ Environment configured${NC}"

# Backup current state
echo -e "${BLUE}💾 Creating backup...${NC}"
BACKUP_FILE="backups/pre_qwen3_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$BACKUP_FILE" --exclude='backups' --exclude='logs' --exclude='__pycache__' --exclude='.git' .
echo -e "${GREEN}✅ Backup created: $BACKUP_FILE${NC}"

echo -e "${CYAN}================================================================${NC}"
echo -e "${PURPLE}🚀 LAUNCHING QWEN3-CODER AUTONOMOUS ENHANCEMENT${NC}"
echo -e "${CYAN}================================================================${NC}"

echo -e "${YELLOW}📊 System Information:${NC}"
echo -e "   🖥️ Host: $(hostname)"
echo -e "   🐍 Python: $(python3 --version)"
echo -e "   📝 Git: $(git --version | head -n1)"
echo -e "   📁 Working Directory: $(pwd)"
echo -e "   ⏰ Start Time: $(date)"

echo -e "\n${BLUE}🧠 Qwen3-Coder Features:${NC}"
echo -e "   🔍 Comprehensive code analysis"
echo -e "   🐛 Automatic bug detection and fixing"
echo -e "   ⚡ Performance optimization"
echo -e "   🔒 Security vulnerability scanning"
echo -e "   🧪 Automated testing and validation"
echo -e "   📝 Git commit automation"
echo -e "   🔄 Continuous 10-minute cycles"

echo -e "\n${YELLOW}⚠️ IMPORTANT NOTES:${NC}"
echo -e "   • Qwen3-Coder will continuously analyze and modify your code"
echo -e "   • All changes are automatically committed to Git"
echo -e "   • Backups are created before each cycle"
echo -e "   • Press Ctrl+C to stop the enhancement loop"
echo -e "   • Monitor logs in the 'logs/' directory"

echo -e "\n${GREEN}🎯 Starting enhancement in 5 seconds...${NC}"
sleep 5

# Function to handle cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down Qwen3-Coder enhancement...${NC}"
    
    # Kill any background processes
    jobs -p | xargs -r kill
    
    # Final Git status
    echo -e "${BLUE}📝 Final Git status:${NC}"
    git status --short
    
    # Enhancement summary
    if [ -f "logs/qwen3_enhancement.log" ]; then
        echo -e "${BLUE}📊 Enhancement summary:${NC}"
        echo -e "   Cycles completed: $(grep -c "Enhancement cycle" logs/qwen3_enhancement.log || echo "0")"
        echo -e "   Files analyzed: $(grep -c "analyzed" logs/qwen3_enhancement.log || echo "0")"
        echo -e "   Enhancements applied: $(grep -c "Applied enhancement" logs/qwen3_enhancement.log || echo "0")"
    fi
    
    echo -e "${GREEN}✅ Qwen3-Coder enhancement stopped cleanly${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM EXIT

# Launch the Qwen3-Coder autonomous enhancement orchestrator
echo -e "${CYAN}🔥 QWEN3-CODER AUTONOMOUS ENHANCEMENT ACTIVE${NC}"
echo -e "${CYAN}================================================================${NC}\n"

# Run with output logging
python3 qwen3_autonomous_enhancement_orchestrator.py 2>&1 | tee -a logs/qwen3_enhancement.log

# This will only execute if the Python script exits normally
echo -e "\n${GREEN}🏁 Qwen3-Coder enhancement completed${NC}"