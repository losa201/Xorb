#!/bin/bash
# XORB Health Check Script

export PATH="$HOME/.local/bin:$PATH"

echo "🔍 XORB Health Check..."
poetry run python monitoring/dashboard.py --health
