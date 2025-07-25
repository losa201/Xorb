#!/bin/bash
# XORB Dashboard Only

export PATH="$HOME/.local/bin:$PATH"

echo "📊 Starting XORB Dashboard..."
poetry run python monitoring/dashboard.py
