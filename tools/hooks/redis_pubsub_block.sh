#!/bin/sh
# Redis Pub/Sub Blocker - ADR-002 Guard
# Blocks Redis pub/sub usage outside of redis_*cache* files

# Find staged files that might contain Redis pub/sub usage
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACMR | grep -E '\.(py|js|ts|go)$' | grep -v 'redis_.*cache')

if [ -z "$STAGED_FILES" ]; then
    exit 0
fi

# Check for Redis pub/sub usage
VIOLATIONS=$(echo "$STAGED_FILES" | xargs -r grep -lE '\b(PUBLISH|SUBSCRIBE|psubscribe|pubsub)\b' 2>/dev/null || true)

if [ -n "$VIOLATIONS" ]; then
    echo "❌ ADR-002 Violation: Redis pub/sub usage detected in:"
    echo "$VIOLATIONS" | sed 's/^/  - /'
    echo ""
    echo "Only redis_*cache* files may use Redis as a bus per ADR-002."
    exit 1
fi

exit 0