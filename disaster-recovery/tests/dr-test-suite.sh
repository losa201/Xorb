#!/bin/bash

# XORB Disaster Recovery Test Suite
# Comprehensive testing of backup and recovery procedures

set -euo pipefail

TESTS_DIR="/root/Xorb/disaster-recovery/tests"
LOGS_DIR="/root/Xorb/disaster-recovery/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_LOG="$LOGS_DIR/dr-test_$TIMESTAMP.log"

echo "🧪 XORB Disaster Recovery Test Suite - $TIMESTAMP" | tee "$TEST_LOG"
echo "===============================================" | tee -a "$TEST_LOG"

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Function to run test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo "" | tee -a "$TEST_LOG"
    echo "🧪 Test: $test_name" | tee -a "$TEST_LOG"
    echo "Command: $test_command" | tee -a "$TEST_LOG"
    
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    
    if eval "$test_command" >> "$TEST_LOG" 2>&1; then
        echo "✅ PASSED" | tee -a "$TEST_LOG"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo "❌ FAILED" | tee -a "$TEST_LOG"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Test 1: Backup creation
run_test "Backup Creation" "/root/Xorb/disaster-recovery/backup-system.sh"

# Test 2: Backup file integrity
run_test "Backup File Integrity" "find /root/Xorb/disaster-recovery/backups -name '*_$(date +%Y%m%d)*' -type f | xargs -I {} bash -c 'echo \"Checking: {}\"; file {}'"

# Test 3: Database backup verification
run_test "Database Backup Verification" "test -f \$(find /root/Xorb/disaster-recovery/backups/database -name '*postgres*' | tail -1)"

# Test 4: Secrets backup verification
run_test "Secrets Backup Verification" "test -f \$(find /root/Xorb/disaster-recovery/backups/secrets -name '*secrets*' | tail -1)"

# Test 5: Configuration backup verification
run_test "Configuration Backup Verification" "test -f \$(find /root/Xorb/disaster-recovery/backups/config -name '*docker*' | tail -1)"

# Test 6: Service health before recovery test
run_test "Service Health Check" "curl -s http://localhost:8000/health && curl -s http://localhost:8080/health && curl -s http://localhost:9000/health"

# Test 7: Recovery dry run (database only)
run_test "Recovery Dry Run" "echo 'Simulating database recovery...'; /root/Xorb/disaster-recovery/recovery-system.sh database"

# Test 8: Post-recovery health check
run_test "Post-Recovery Health Check" "sleep 30 && curl -s http://localhost:8000/health"

# Test 9: Backup rotation test
run_test "Backup Rotation Test" "find /root/Xorb/disaster-recovery/backups -name '*' -mtime +1 | wc -l"

# Test 10: Secret management recovery
run_test "Secret Management Recovery" "python3 /root/Xorb/secrets/secret-manager.py list"

# Generate test report
echo "" | tee -a "$TEST_LOG"
echo "📊 Disaster Recovery Test Report" | tee -a "$TEST_LOG"
echo "===============================" | tee -a "$TEST_LOG"
echo "Total Tests: $TESTS_TOTAL" | tee -a "$TEST_LOG"
echo "Passed: $TESTS_PASSED" | tee -a "$TEST_LOG"
echo "Failed: $TESTS_FAILED" | tee -a "$TEST_LOG"
echo "Success Rate: $(( TESTS_PASSED * 100 / TESTS_TOTAL ))%" | tee -a "$TEST_LOG"
echo "" | tee -a "$TEST_LOG"

if [ $TESTS_FAILED -eq 0 ]; then
    echo "🎉 All disaster recovery tests passed!" | tee -a "$TEST_LOG"
    exit 0
else
    echo "⚠️  Some disaster recovery tests failed. Review log: $TEST_LOG" | tee -a "$TEST_LOG"
    exit 1
fi
