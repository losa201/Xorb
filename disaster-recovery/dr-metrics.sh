#!/bin/bash

# XORB Disaster Recovery Metrics
# Track and report on disaster recovery capabilities

echo "📊 XORB Disaster Recovery Metrics"
echo "================================="
echo "Generated: $(date)"
echo ""

BACKUP_DIR="/root/Xorb/disaster-recovery/backups"
LOGS_DIR="/root/Xorb/disaster-recovery/logs"

# Backup metrics
echo "💾 Backup Metrics:"
TOTAL_BACKUPS=$(find "$BACKUP_DIR" -type f -name "*_*" | wc -l)
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1 || echo "0B")
LATEST_BACKUP=$(find "$BACKUP_DIR" -type f -name "*_*" | sort | tail -1)
BACKUP_AGE=$([ -n "$LATEST_BACKUP" ] && echo "$(($(date +%s) - $(stat -c %Y "$LATEST_BACKUP"))) seconds ago" || echo "No backups found")

echo "  - Total backups: $TOTAL_BACKUPS"
echo "  - Total size: $BACKUP_SIZE"
echo "  - Latest backup: $BACKUP_AGE"

# Test metrics
echo ""
echo "🧪 Test Metrics:"
TEST_LOGS=$(find "$LOGS_DIR" -name "dr-test_*" | wc -l)
LATEST_TEST=$(find "$LOGS_DIR" -name "dr-test_*" | sort | tail -1)
if [ -n "$LATEST_TEST" ]; then
    PASSED=$(grep "PASSED" "$LATEST_TEST" | wc -l)
    FAILED=$(grep "FAILED" "$LATEST_TEST" | wc -l)
    SUCCESS_RATE=$([ $((PASSED + FAILED)) -gt 0 ] && echo "$((PASSED * 100 / (PASSED + FAILED)))%" || echo "N/A")
    echo "  - Total test runs: $TEST_LOGS"
    echo "  - Latest test - Passed: $PASSED, Failed: $FAILED"
    echo "  - Success rate: $SUCCESS_RATE"
else
    echo "  - No test logs found"
fi

# RTO/RPO tracking
echo ""
echo "⏱️  RTO/RPO Metrics:"
echo "  - Target RTO: 30 minutes (critical), 2 hours (full)"
echo "  - Target RPO: 15 minutes (database), 24 hours (config)"
echo "  - Last backup: $BACKUP_AGE"

# Availability metrics
echo ""
echo "📈 Availability Metrics:"
UPTIME=$(uptime | awk '{print $3,$4}' | sed 's/,//')
echo "  - System uptime: $UPTIME"

if docker ps | grep -q xorb; then
    echo "  - XORB services: Running"
else
    echo "  - XORB services: Not running"
fi

# Recommendations
echo ""
echo "🎯 Recommendations:"
if [ "$TOTAL_BACKUPS" -lt 7 ]; then
    echo "  - Increase backup frequency"
fi

if [ -z "$LATEST_TEST" ] || [ $(find "$LOGS_DIR" -name "dr-test_*" -mtime -7 | wc -l) -eq 0 ]; then
    echo "  - Run disaster recovery tests"
fi

echo "  - Review and update recovery procedures monthly"
echo "  - Validate cross-region backup replication"
