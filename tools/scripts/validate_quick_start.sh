#!/bin/bash
# Validation script for quick_start.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔍 Validating quick_start.sh script..."

# Test 1: Syntax check
echo "✅ Test 1: Bash syntax check"
bash -n "$SCRIPT_DIR/quick_start.sh"

# Test 2: Help function
echo "✅ Test 2: Help function works"
"$SCRIPT_DIR/quick_start.sh" --help > /dev/null

# Test 3: Check key functions exist
echo "✅ Test 3: Key functions exist"
if grep -q "check_requirements()" "$SCRIPT_DIR/quick_start.sh"; then
    echo "  ✓ check_requirements function found"
else
    echo "  ❌ check_requirements function missing"
    exit 1
fi

if grep -q "cleanup()" "$SCRIPT_DIR/quick_start.sh"; then
    echo "  ✓ cleanup function found"
else
    echo "  ❌ cleanup function missing"
    exit 1
fi

# Test 4: Docker compose commands are correct
echo "✅ Test 4: Docker compose commands syntax"
if grep -q "docker compose" "$SCRIPT_DIR/quick_start.sh"; then
    echo "  ✓ Docker compose v2 commands used"
else
    echo "  ❌ Docker compose commands not found"
    exit 1
fi

# Test 5: Environment variables are properly handled
echo "✅ Test 5: Environment variables"
if grep -q "ENVIRONMENT=.*dev" "$SCRIPT_DIR/quick_start.sh"; then
    echo "  ✓ Default environment is dev"
else
    echo "  ❌ Default environment not set correctly"
    exit 1
fi

echo ""
echo "🎉 All validation tests passed! quick_start.sh is ready to use."
echo ""
echo "Usage examples:"
echo "  ./scripts/quick_start.sh                    # Deploy with defaults (dev environment)"
echo "  ./scripts/quick_start.sh --env staging      # Deploy to staging"
echo "  ./scripts/quick_start.sh --cleanup          # Clean up deployment"
echo "  ./scripts/quick_start.sh --help             # Show help"
