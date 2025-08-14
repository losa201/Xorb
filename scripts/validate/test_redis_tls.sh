#!/bin/bash
# Redis TLS Validation Script for XORB Platform
# Tests Redis TLS-only configuration and client certificate authentication

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_DIR="${SCRIPT_DIR}/../../reports/redis-tls"
SECRETS_DIR="${SCRIPT_DIR}/../../secrets/tls"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Redis configuration
REDIS_HOST="redis"
REDIS_TLS_PORT="6379"
REDIS_PASSWORD="xorb_redis_secure_pass"

log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message"
            ;;
        "DEBUG")
            if [[ "${VERBOSE:-false}" == "true" ]]; then
                echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message"
            fi
            ;;
    esac
}

test_plaintext_blocked() {
    log "INFO" "Testing that plaintext Redis connections are blocked"
    
    # Try to connect to Redis without TLS (should fail)
    if timeout 5 redis-cli -h "$REDIS_HOST" -p 6379 ping >/dev/null 2>&1; then
        log "ERROR" "❌ Plaintext Redis connection succeeded - TLS not enforced!"
        return 1
    else
        log "INFO" "✅ Plaintext Redis connection correctly blocked"
        return 0
    fi
}

test_tls_without_client_cert() {
    log "INFO" "Testing TLS connection without client certificate"
    
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    
    # Try TLS connection without client certificate (should fail with mTLS)
    if timeout 5 redis-cli \
        --tls \
        --cacert "$ca_file" \
        -h "$REDIS_HOST" \
        -p "$REDIS_TLS_PORT" \
        -a "$REDIS_PASSWORD" \
        ping >/dev/null 2>&1; then
        log "ERROR" "❌ Redis TLS connection without client cert succeeded - mTLS not enforced!"
        return 1
    else
        log "INFO" "✅ Redis TLS connection without client cert correctly blocked"
        return 0
    fi
}

test_tls_with_client_cert() {
    log "INFO" "Testing TLS connection with valid client certificate"
    
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    local client_cert="${SECRETS_DIR}/redis-client/cert.pem"
    local client_key="${SECRETS_DIR}/redis-client/key.pem"
    
    # Check if client certificate files exist
    for file in "$ca_file" "$client_cert" "$client_key"; do
        if [[ ! -f "$file" ]]; then
            log "ERROR" "Required file not found: $file"
            return 1
        fi
    done
    
    # Test TLS connection with client certificate
    log "DEBUG" "Attempting Redis TLS connection with client certificate"
    
    local redis_response
    if redis_response=$(timeout 10 redis-cli \
        --tls \
        --cert "$client_cert" \
        --key "$client_key" \
        --cacert "$ca_file" \
        -h "$REDIS_HOST" \
        -p "$REDIS_TLS_PORT" \
        -a "$REDIS_PASSWORD" \
        ping 2>&1); then
        
        if [[ "$redis_response" == "PONG" ]]; then
            log "INFO" "✅ Redis TLS connection with client cert successful"
            return 0
        else
            log "ERROR" "❌ Redis TLS connection with client cert failed: $redis_response"
            return 1
        fi
    else
        log "ERROR" "❌ Redis TLS connection with client cert timed out or failed: $redis_response"
        return 1
    fi
}

test_redis_commands() {
    log "INFO" "Testing Redis commands over TLS"
    
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    local client_cert="${SECRETS_DIR}/redis-client/cert.pem"
    local client_key="${SECRETS_DIR}/redis-client/key.pem"
    local test_key="xorb:tls:test:$(date +%s)"
    local test_value="TLS_TEST_$(openssl rand -hex 8)"
    
    # Test SET command
    log "DEBUG" "Testing Redis SET command"
    if redis-cli \
        --tls \
        --cert "$client_cert" \
        --key "$client_key" \
        --cacert "$ca_file" \
        -h "$REDIS_HOST" \
        -p "$REDIS_TLS_PORT" \
        -a "$REDIS_PASSWORD" \
        SET "$test_key" "$test_value" >/dev/null 2>&1; then
        log "INFO" "✅ Redis SET command successful"
    else
        log "ERROR" "❌ Redis SET command failed"
        return 1
    fi
    
    # Test GET command
    log "DEBUG" "Testing Redis GET command"
    local retrieved_value
    if retrieved_value=$(redis-cli \
        --tls \
        --cert "$client_cert" \
        --key "$client_key" \
        --cacert "$ca_file" \
        -h "$REDIS_HOST" \
        -p "$REDIS_TLS_PORT" \
        -a "$REDIS_PASSWORD" \
        GET "$test_key" 2>/dev/null); then
        
        if [[ "$retrieved_value" == "$test_value" ]]; then
            log "INFO" "✅ Redis GET command successful (value matches)"
        else
            log "ERROR" "❌ Redis GET command returned wrong value: expected '$test_value', got '$retrieved_value'"
            return 1
        fi
    else
        log "ERROR" "❌ Redis GET command failed"
        return 1
    fi
    
    # Test DEL command
    log "DEBUG" "Testing Redis DEL command"
    if redis-cli \
        --tls \
        --cert "$client_cert" \
        --key "$client_key" \
        --cacert "$ca_file" \
        -h "$REDIS_HOST" \
        -p "$REDIS_TLS_PORT" \
        -a "$REDIS_PASSWORD" \
        DEL "$test_key" >/dev/null 2>&1; then
        log "INFO" "✅ Redis DEL command successful"
    else
        log "ERROR" "❌ Redis DEL command failed"
        return 1
    fi
    
    return 0
}

test_redis_info() {
    log "INFO" "Testing Redis INFO command and TLS configuration"
    
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    local client_cert="${SECRETS_DIR}/redis-client/cert.pem"
    local client_key="${SECRETS_DIR}/redis-client/key.pem"
    
    # Get Redis server information
    log "DEBUG" "Retrieving Redis server information"
    
    local redis_info
    if redis_info=$(redis-cli \
        --tls \
        --cert "$client_cert" \
        --key "$client_key" \
        --cacert "$ca_file" \
        -h "$REDIS_HOST" \
        -p "$REDIS_TLS_PORT" \
        -a "$REDIS_PASSWORD" \
        INFO server 2>/dev/null); then
        
        log "INFO" "✅ Redis INFO command successful"
        
        # Extract useful information
        local redis_version=$(echo "$redis_info" | grep "redis_version:" | cut -d: -f2 | tr -d '\r')
        local redis_mode=$(echo "$redis_info" | grep "redis_mode:" | cut -d: -f2 | tr -d '\r')
        local tcp_port=$(echo "$redis_info" | grep "tcp_port:" | cut -d: -f2 | tr -d '\r')
        
        log "INFO" "Redis Version: $redis_version"
        log "INFO" "Redis Mode: $redis_mode"
        log "INFO" "TCP Port: $tcp_port"
        
        # Check if plaintext port is disabled
        if [[ "$tcp_port" == "0" ]]; then
            log "INFO" "✅ Plaintext TCP port disabled (port 0)"
        else
            log "WARN" "⚠️  Plaintext TCP port enabled: $tcp_port"
        fi
        
        return 0
    else
        log "ERROR" "❌ Redis INFO command failed"
        return 1
    fi
}

test_invalid_client_cert() {
    log "INFO" "Testing Redis with invalid client certificate"
    
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    
    # Create temporary invalid certificate
    local temp_dir=$(mktemp -d)
    local invalid_cert="${temp_dir}/invalid_cert.pem"
    local invalid_key="${temp_dir}/invalid_key.pem"
    
    # Generate self-signed invalid certificate
    openssl req -x509 -newkey rsa:2048 -keyout "${invalid_key}" -out "${invalid_cert}" \
        -days 1 -nodes -subj "/CN=invalid.redis.local" >/dev/null 2>&1
    
    log "DEBUG" "Testing Redis connection with invalid client certificate"
    
    # Try connection with invalid certificate (should fail)
    if timeout 5 redis-cli \
        --tls \
        --cert "$invalid_cert" \
        --key "$invalid_key" \
        --cacert "$ca_file" \
        -h "$REDIS_HOST" \
        -p "$REDIS_TLS_PORT" \
        -a "$REDIS_PASSWORD" \
        ping >/dev/null 2>&1; then
        log "ERROR" "❌ Redis accepts invalid client certificate!"
        rm -rf "$temp_dir"
        return 1
    else
        log "INFO" "✅ Redis correctly rejects invalid client certificate"
        rm -rf "$temp_dir"
        return 0
    fi
}

test_connection_security() {
    log "INFO" "Testing Redis connection security parameters"
    
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    local client_cert="${SECRETS_DIR}/redis-client/cert.pem"
    local client_key="${SECRETS_DIR}/redis-client/key.pem"
    
    # Test with openssl s_client to verify TLS parameters
    log "DEBUG" "Testing TLS parameters with openssl s_client"
    
    local tls_info
    if tls_info=$(timeout 10 openssl s_client \
        -connect "${REDIS_HOST}:${REDIS_TLS_PORT}" \
        -CAfile "$ca_file" \
        -cert "$client_cert" \
        -key "$client_key" \
        -servername "redis.xorb.local" \
        -quiet \
        </dev/null 2>&1); then
        
        # Check TLS version
        local tls_version=$(echo "$tls_info" | grep "Protocol" | head -1)
        if echo "$tls_version" | grep -E "(TLSv1\.2|TLSv1\.3)" >/dev/null; then
            log "INFO" "✅ Secure TLS version: $tls_version"
        else
            log "ERROR" "❌ Insecure TLS version: $tls_version"
            return 1
        fi
        
        # Check cipher suite
        local cipher=$(echo "$tls_info" | grep "Cipher" | head -1)
        if echo "$cipher" | grep -E "(ECDHE|AES|GCM|CHACHA20)" >/dev/null; then
            log "INFO" "✅ Secure cipher suite: $cipher"
        else
            log "WARN" "⚠️  Cipher suite: $cipher"
        fi
        
        # Check certificate verification
        if echo "$tls_info" | grep -q "Verification: OK"; then
            log "INFO" "✅ Certificate verification successful"
        else
            log "ERROR" "❌ Certificate verification failed"
            return 1
        fi
        
        return 0
    else
        log "ERROR" "❌ Failed to establish TLS connection for security testing"
        return 1
    fi
}

performance_test() {
    log "INFO" "Running Redis TLS performance test"
    
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    local client_cert="${SECRETS_DIR}/redis-client/cert.pem"
    local client_key="${SECRETS_DIR}/redis-client/key.pem"
    
    # Run simple benchmark
    log "DEBUG" "Running Redis benchmark over TLS"
    
    local benchmark_result
    if benchmark_result=$(timeout 30 redis-benchmark \
        --tls \
        --cert "$client_cert" \
        --key "$client_key" \
        --cacert "$ca_file" \
        -h "$REDIS_HOST" \
        -p "$REDIS_TLS_PORT" \
        -a "$REDIS_PASSWORD" \
        -t ping,set,get \
        -n 1000 \
        -c 10 \
        -q 2>&1); then
        
        log "INFO" "✅ Redis TLS performance test completed"
        log "DEBUG" "Benchmark results: $benchmark_result"
        
        # Extract performance metrics
        local ping_ops=$(echo "$benchmark_result" | grep "PING_INLINE" | awk '{print $3}')
        local set_ops=$(echo "$benchmark_result" | grep "SET" | awk '{print $3}')
        local get_ops=$(echo "$benchmark_result" | grep "GET" | awk '{print $3}')
        
        log "INFO" "Performance: PING=$ping_ops ops/sec, SET=$set_ops ops/sec, GET=$get_ops ops/sec"
        
        return 0
    else
        log "WARN" "⚠️  Redis TLS performance test failed or timed out"
        return 1
    fi
}

generate_redis_report() {
    log "INFO" "Generating Redis TLS validation report"
    
    local report_file="${REPORT_DIR}/redis_tls_report.html"
    
    cat > "$report_file" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Redis TLS Security Report - XORB Platform</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #c0392b; color: white; padding: 20px; border-radius: 5px; }
        .test-section { background: #ecf0f1; margin: 20px 0; padding: 15px; border-radius: 5px; }
        .pass { color: #27ae60; font-weight: bold; }
        .fail { color: #e74c3c; font-weight: bold; }
        .warn { color: #f39c12; font-weight: bold; }
        .details { margin-top: 10px; font-family: monospace; background: #2c3e50; color: #ecf0f1; padding: 10px; border-radius: 3px; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background: #bdc3c7; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔴 Redis TLS Security Report</h1>
        <p>Generated: $(date)</p>
        <p>Redis Host: ${REDIS_HOST}:${REDIS_TLS_PORT}</p>
    </div>
EOF

    # Read test results from the detailed report
    local detailed_report="${REPORT_DIR}/redis_detailed.txt"
    if [[ -f "$detailed_report" ]]; then
        local total_tests=$(grep -c "✅\|❌" "$detailed_report" 2>/dev/null || echo "0")
        local passed_tests=$(grep -c "✅" "$detailed_report" 2>/dev/null || echo "0")
        local success_rate=0
        
        if [[ $total_tests -gt 0 ]]; then
            success_rate=$(( passed_tests * 100 / total_tests ))
        fi
        
        cat >> "$report_file" << EOF
    <div class="test-section">
        <h2>📊 Test Summary</h2>
        <div class="metric">Total Tests: ${total_tests}</div>
        <div class="metric">Passed: <span class="pass">${passed_tests}</span></div>
        <div class="metric">Failed: <span class="fail">$((total_tests - passed_tests))</span></div>
        <div class="metric">Success Rate: ${success_rate}%</div>
    </div>
    
    <div class="test-section">
        <h2>🔍 Detailed Results</h2>
        <div class="details">
$(cat "$detailed_report" | sed 's/</\&lt;/g; s/>/\&gt;/g')
        </div>
    </div>
EOF
    fi
    
    cat >> "$report_file" << 'EOF'
    
    <div class="test-section">
        <h2>🔒 Security Recommendations</h2>
        <ul>
            <li><strong>Disable Plaintext:</strong> Ensure Redis port 6379 (plaintext) is disabled</li>
            <li><strong>Client Certificates:</strong> All clients must use valid mTLS certificates</li>
            <li><strong>Password Protection:</strong> Use strong Redis AUTH passwords</li>
            <li><strong>Network Security:</strong> Restrict Redis access to authorized networks only</li>
            <li><strong>Monitoring:</strong> Monitor for failed authentication attempts</li>
        </ul>
    </div>
</body>
</html>
EOF

    log "INFO" "Redis TLS report generated: ${report_file}"
}

# Create report directory
mkdir -p "$REPORT_DIR"

log "INFO" "Starting Redis TLS validation"
log "INFO" "Redis Host: ${REDIS_HOST}:${REDIS_TLS_PORT}"

# Create detailed report file
detailed_report="${REPORT_DIR}/redis_detailed.txt"
echo "Redis TLS Test Results" > "$detailed_report"
echo "Generated: $(date)" >> "$detailed_report"
echo "========================" >> "$detailed_report"

# Run tests
test_results=()
failed_tests=0

# Test 1: Plaintext connections blocked
if test_plaintext_blocked; then
    echo "✅ Plaintext connections blocked: PASS" >> "$detailed_report"
    test_results+=("plaintext_blocked:PASS")
else
    echo "❌ Plaintext connections blocked: FAIL" >> "$detailed_report"
    test_results+=("plaintext_blocked:FAIL")
    failed_tests=$((failed_tests + 1))
fi

# Test 2: TLS without client cert blocked
if test_tls_without_client_cert; then
    echo "✅ TLS without client cert blocked: PASS" >> "$detailed_report"
    test_results+=("tls_no_client_cert:PASS")
else
    echo "❌ TLS without client cert blocked: FAIL" >> "$detailed_report"
    test_results+=("tls_no_client_cert:FAIL")
    failed_tests=$((failed_tests + 1))
fi

# Test 3: TLS with valid client cert
if test_tls_with_client_cert; then
    echo "✅ TLS with valid client cert: PASS" >> "$detailed_report"
    test_results+=("tls_valid_cert:PASS")
else
    echo "❌ TLS with valid client cert: FAIL" >> "$detailed_report"
    test_results+=("tls_valid_cert:FAIL")
    failed_tests=$((failed_tests + 1))
fi

# Test 4: Redis commands over TLS
if test_redis_commands; then
    echo "✅ Redis commands over TLS: PASS" >> "$detailed_report"
    test_results+=("redis_commands:PASS")
else
    echo "❌ Redis commands over TLS: FAIL" >> "$detailed_report"
    test_results+=("redis_commands:FAIL")
    failed_tests=$((failed_tests + 1))
fi

# Test 5: Redis server info
if test_redis_info; then
    echo "✅ Redis server info: PASS" >> "$detailed_report"
    test_results+=("redis_info:PASS")
else
    echo "❌ Redis server info: FAIL" >> "$detailed_report"
    test_results+=("redis_info:FAIL")
    failed_tests=$((failed_tests + 1))
fi

# Test 6: Invalid client cert rejected
if test_invalid_client_cert; then
    echo "✅ Invalid client cert rejected: PASS" >> "$detailed_report"
    test_results+=("invalid_cert:PASS")
else
    echo "❌ Invalid client cert rejected: FAIL" >> "$detailed_report"
    test_results+=("invalid_cert:FAIL")
    failed_tests=$((failed_tests + 1))
fi

# Test 7: Connection security
if test_connection_security; then
    echo "✅ Connection security: PASS" >> "$detailed_report"
    test_results+=("connection_security:PASS")
else
    echo "❌ Connection security: FAIL" >> "$detailed_report"
    test_results+=("connection_security:FAIL")
    failed_tests=$((failed_tests + 1))
fi

# Test 8: Performance test (non-critical)
if performance_test; then
    echo "✅ Performance test: PASS" >> "$detailed_report"
    test_results+=("performance:PASS")
else
    echo "⚠️  Performance test: WARN" >> "$detailed_report"
    test_results+=("performance:WARN")
fi

# Generate report
generate_redis_report

# Final results
log "INFO" "Redis TLS validation completed"
log "INFO" "Results: $(printf '%s ' "${test_results[@]}")"

if [[ $failed_tests -eq 0 ]]; then
    log "INFO" "🎉 All Redis TLS tests PASSED!"
    exit 0
else
    log "ERROR" "💥 ${failed_tests} Redis TLS test(s) FAILED"
    exit 1
fi