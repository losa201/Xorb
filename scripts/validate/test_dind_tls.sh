#!/bin/bash
# Docker-in-Docker TLS Validation Script for XORB Platform
# Tests Docker daemon TLS configuration and client certificate authentication

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_DIR="${SCRIPT_DIR}/../../reports/dind-tls"
SECRETS_DIR="${SCRIPT_DIR}/../../secrets/tls"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Docker daemon configuration
DOCKER_HOST_URL="tcp://dind:2376"
DOCKER_CLIENT_CERT_DIR="${SECRETS_DIR}/dind-client"

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

test_plaintext_docker_blocked() {
    log "INFO" "Testing that plaintext Docker connections are blocked"
    
    # Try to connect without TLS (should fail)
    if timeout 5 docker -H "tcp://dind:2375" version >/dev/null 2>&1; then
        log "ERROR" "❌ Plaintext Docker connection succeeded - TLS not enforced!"
        return 1
    else
        log "INFO" "✅ Plaintext Docker connection correctly blocked"
        return 0
    fi
}

test_docker_tls_without_client_cert() {
    log "INFO" "Testing Docker TLS connection without client certificate"
    
    # Try TLS connection without client certificate (should fail)
    if timeout 5 docker \
        -H "$DOCKER_HOST_URL" \
        --tls \
        version >/dev/null 2>&1; then
        log "ERROR" "❌ Docker TLS connection without client cert succeeded - Client cert not enforced!"
        return 1
    else
        log "INFO" "✅ Docker TLS connection without client cert correctly blocked"
        return 0
    fi
}

test_docker_tls_with_client_cert() {
    log "INFO" "Testing Docker TLS connection with valid client certificate"
    
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    local client_cert="${DOCKER_CLIENT_CERT_DIR}/cert.pem"
    local client_key="${DOCKER_CLIENT_CERT_DIR}/key.pem"
    
    # Check if client certificate files exist
    for file in "$ca_file" "$client_cert" "$client_key"; do
        if [[ ! -f "$file" ]]; then
            log "ERROR" "Required file not found: $file"
            return 1
        fi
    done
    
    # Test Docker TLS connection with client certificate
    log "DEBUG" "Attempting Docker TLS connection with client certificate"
    
    local docker_response
    if docker_response=$(timeout 15 docker \
        -H "$DOCKER_HOST_URL" \
        --tlsverify \
        --tlscacert "$ca_file" \
        --tlscert "$client_cert" \
        --tlskey "$client_key" \
        version 2>&1); then
        
        if echo "$docker_response" | grep -q "Server Version"; then
            local server_version=$(echo "$docker_response" | grep "Server Version" | awk '{print $3}')
            log "INFO" "✅ Docker TLS connection with client cert successful (Server: $server_version)"
            return 0
        else
            log "ERROR" "❌ Docker TLS connection with client cert failed: $docker_response"
            return 1
        fi
    else
        log "ERROR" "❌ Docker TLS connection with client cert timed out or failed: $docker_response"
        return 1
    fi
}

test_docker_commands() {
    log "INFO" "Testing Docker commands over TLS"
    
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    local client_cert="${DOCKER_CLIENT_CERT_DIR}/cert.pem"
    local client_key="${DOCKER_CLIENT_CERT_DIR}/key.pem"
    
    # Set up Docker client environment
    export DOCKER_HOST="$DOCKER_HOST_URL"
    export DOCKER_TLS_VERIFY=1
    export DOCKER_CERT_PATH="$(dirname "$client_cert")"
    
    # Test docker info command
    log "DEBUG" "Testing 'docker info' command"
    local docker_info
    if docker_info=$(timeout 15 docker \
        --tlsverify \
        --tlscacert "$ca_file" \
        --tlscert "$client_cert" \
        --tlskey "$client_key" \
        info --format "{{.ServerVersion}}" 2>&1); then
        log "INFO" "✅ Docker info command successful (Version: $docker_info)"
    else
        log "ERROR" "❌ Docker info command failed: $docker_info"
        return 1
    fi
    
    # Test docker ps command
    log "DEBUG" "Testing 'docker ps' command"
    if docker \
        --tlsverify \
        --tlscacert "$ca_file" \
        --tlscert "$client_cert" \
        --tlskey "$client_key" \
        ps >/dev/null 2>&1; then
        log "INFO" "✅ Docker ps command successful"
    else
        log "ERROR" "❌ Docker ps command failed"
        return 1
    fi
    
    # Test running a simple container
    log "DEBUG" "Testing container execution"
    local container_output
    if container_output=$(timeout 30 docker \
        --tlsverify \
        --tlscacert "$ca_file" \
        --tlscert "$client_cert" \
        --tlskey "$client_key" \
        run --rm alpine:latest echo "TLS_TEST_SUCCESS" 2>&1); then
        
        if echo "$container_output" | grep -q "TLS_TEST_SUCCESS"; then
            log "INFO" "✅ Container execution over TLS successful"
        else
            log "ERROR" "❌ Container execution over TLS failed: $container_output"
            return 1
        fi
    else
        log "ERROR" "❌ Container execution over TLS timed out or failed: $container_output"
        return 1
    fi
    
    return 0
}

test_invalid_client_cert() {
    log "INFO" "Testing Docker with invalid client certificate"
    
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    
    # Create temporary invalid certificate
    local temp_dir=$(mktemp -d)
    local invalid_cert="${temp_dir}/invalid_cert.pem"
    local invalid_key="${temp_dir}/invalid_key.pem"
    
    # Generate self-signed invalid certificate
    openssl req -x509 -newkey rsa:2048 -keyout "${invalid_key}" -out "${invalid_cert}" \
        -days 1 -nodes -subj "/CN=invalid.docker.local" >/dev/null 2>&1
    
    log "DEBUG" "Testing Docker connection with invalid client certificate"
    
    # Try connection with invalid certificate (should fail)
    if timeout 10 docker \
        -H "$DOCKER_HOST_URL" \
        --tlsverify \
        --tlscacert "$ca_file" \
        --tlscert "$invalid_cert" \
        --tlskey "$invalid_key" \
        version >/dev/null 2>&1; then
        log "ERROR" "❌ Docker accepts invalid client certificate!"
        rm -rf "$temp_dir"
        return 1
    else
        log "INFO" "✅ Docker correctly rejects invalid client certificate"
        rm -rf "$temp_dir"
        return 0
    fi
}

test_docker_security_scan() {
    log "INFO" "Testing security scanning container execution"
    
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    local client_cert="${DOCKER_CLIENT_CERT_DIR}/cert.pem"
    local client_key="${DOCKER_CLIENT_CERT_DIR}/key.pem"
    
    # Test running a security scan container (Nmap)
    log "DEBUG" "Testing Nmap security scan container"
    
    local scan_result
    if scan_result=$(timeout 60 docker \
        --tlsverify \
        --tlscacert "$ca_file" \
        --tlscert "$client_cert" \
        --tlskey "$client_key" \
        run --rm \
        --cap-add=NET_RAW \
        --cap-add=NET_ADMIN \
        instrumentisto/nmap:latest \
        nmap -sn 127.0.0.1 2>&1); then
        
        if echo "$scan_result" | grep -q "Host is up"; then
            log "INFO" "✅ Security scan container execution successful"
        else
            log "WARN" "⚠️  Security scan container completed but may not have found targets"
            log "DEBUG" "Scan output: $scan_result"
        fi
    else
        log "ERROR" "❌ Security scan container execution failed: $scan_result"
        return 1
    fi
    
    return 0
}

test_docker_daemon_config() {
    log "INFO" "Testing Docker daemon TLS configuration"
    
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    local client_cert="${DOCKER_CLIENT_CERT_DIR}/cert.pem"
    local client_key="${DOCKER_CLIENT_CERT_DIR}/key.pem"
    
    # Get Docker daemon configuration
    log "DEBUG" "Retrieving Docker daemon information"
    
    local daemon_info
    if daemon_info=$(timeout 15 docker \
        --tlsverify \
        --tlscacert "$ca_file" \
        --tlscert "$client_cert" \
        --tlskey "$client_key" \
        info --format json 2>/dev/null); then
        
        log "INFO" "✅ Docker daemon info retrieved successfully"
        
        # Extract security information
        local security_options=$(echo "$daemon_info" | jq -r '.SecurityOptions[]? // empty' 2>/dev/null || echo "Unknown")
        local server_version=$(echo "$daemon_info" | jq -r '.ServerVersion // "Unknown"' 2>/dev/null)
        local architecture=$(echo "$daemon_info" | jq -r '.Architecture // "Unknown"' 2>/dev/null)
        
        log "INFO" "Docker Version: $server_version"
        log "INFO" "Architecture: $architecture"
        log "INFO" "Security Options: $security_options"
        
        # Check for important security features
        if echo "$security_options" | grep -q "apparmor\|seccomp"; then
            log "INFO" "✅ Security features enabled: $security_options"
        else
            log "WARN" "⚠️  Limited security features detected"
        fi
        
        return 0
    else
        log "ERROR" "❌ Failed to retrieve Docker daemon info"
        return 1
    fi
}

test_tls_connection_security() {
    log "INFO" "Testing Docker TLS connection security parameters"
    
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    local client_cert="${DOCKER_CLIENT_CERT_DIR}/cert.pem"
    local client_key="${DOCKER_CLIENT_CERT_DIR}/key.pem"
    
    # Test TLS connection with openssl s_client
    log "DEBUG" "Testing TLS parameters with openssl s_client"
    
    local tls_info
    if tls_info=$(timeout 10 openssl s_client \
        -connect "dind:2376" \
        -CAfile "$ca_file" \
        -cert "$client_cert" \
        -key "$client_key" \
        -servername "dind.xorb.local" \
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

generate_dind_report() {
    log "INFO" "Generating Docker-in-Docker TLS validation report"
    
    local report_file="${REPORT_DIR}/dind_tls_report.html"
    
    cat > "$report_file" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Docker-in-Docker TLS Security Report - XORB Platform</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #2980b9; color: white; padding: 20px; border-radius: 5px; }
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
        <h1>🐳 Docker-in-Docker TLS Security Report</h1>
        <p>Generated: $(date)</p>
        <p>Docker Host: ${DOCKER_HOST_URL}</p>
    </div>
EOF

    # Read test results from the detailed report
    local detailed_report="${REPORT_DIR}/dind_detailed.txt"
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
            <li><strong>TLS Only:</strong> Ensure Docker daemon only accepts TLS connections</li>
            <li><strong>Client Certificates:</strong> All clients must use valid mTLS certificates</li>
            <li><strong>Container Security:</strong> Use security options like AppArmor and seccomp</li>
            <li><strong>Network Isolation:</strong> Restrict Docker daemon access to authorized networks</li>
            <li><strong>Resource Limits:</strong> Apply CPU and memory limits to containers</li>
            <li><strong>Image Security:</strong> Scan container images for vulnerabilities</li>
        </ul>
    </div>
</body>
</html>
EOF

    log "INFO" "Docker-in-Docker TLS report generated: ${report_file}"
}

# Create report directory
mkdir -p "$REPORT_DIR"

log "INFO" "Starting Docker-in-Docker TLS validation"
log "INFO" "Docker Host: ${DOCKER_HOST_URL}"

# Create detailed report file
detailed_report="${REPORT_DIR}/dind_detailed.txt"
echo "Docker-in-Docker TLS Test Results" > "$detailed_report"
echo "Generated: $(date)" >> "$detailed_report"
echo "=================================" >> "$detailed_report"

# Run tests
test_results=()
failed_tests=0

# Test 1: Plaintext connections blocked
if test_plaintext_docker_blocked; then
    echo "✅ Plaintext Docker connections blocked: PASS" >> "$detailed_report"
    test_results+=("plaintext_blocked:PASS")
else
    echo "❌ Plaintext Docker connections blocked: FAIL" >> "$detailed_report"
    test_results+=("plaintext_blocked:FAIL")
    failed_tests=$((failed_tests + 1))
fi

# Test 2: TLS without client cert blocked
if test_docker_tls_without_client_cert; then
    echo "✅ TLS without client cert blocked: PASS" >> "$detailed_report"
    test_results+=("tls_no_client_cert:PASS")
else
    echo "❌ TLS without client cert blocked: FAIL" >> "$detailed_report"
    test_results+=("tls_no_client_cert:FAIL")
    failed_tests=$((failed_tests + 1))
fi

# Test 3: TLS with valid client cert
if test_docker_tls_with_client_cert; then
    echo "✅ TLS with valid client cert: PASS" >> "$detailed_report"
    test_results+=("tls_valid_cert:PASS")
else
    echo "❌ TLS with valid client cert: FAIL" >> "$detailed_report"
    test_results+=("tls_valid_cert:FAIL")
    failed_tests=$((failed_tests + 1))
fi

# Test 4: Docker commands over TLS
if test_docker_commands; then
    echo "✅ Docker commands over TLS: PASS" >> "$detailed_report"
    test_results+=("docker_commands:PASS")
else
    echo "❌ Docker commands over TLS: FAIL" >> "$detailed_report"
    test_results+=("docker_commands:FAIL")
    failed_tests=$((failed_tests + 1))
fi

# Test 5: Invalid client cert rejected
if test_invalid_client_cert; then
    echo "✅ Invalid client cert rejected: PASS" >> "$detailed_report"
    test_results+=("invalid_cert:PASS")
else
    echo "❌ Invalid client cert rejected: FAIL" >> "$detailed_report"
    test_results+=("invalid_cert:FAIL")
    failed_tests=$((failed_tests + 1))
fi

# Test 6: Security scan execution
if test_docker_security_scan; then
    echo "✅ Security scan execution: PASS" >> "$detailed_report"
    test_results+=("security_scan:PASS")
else
    echo "❌ Security scan execution: FAIL" >> "$detailed_report"
    test_results+=("security_scan:FAIL")
    failed_tests=$((failed_tests + 1))
fi

# Test 7: Docker daemon config
if test_docker_daemon_config; then
    echo "✅ Docker daemon config: PASS" >> "$detailed_report"
    test_results+=("daemon_config:PASS")
else
    echo "❌ Docker daemon config: FAIL" >> "$detailed_report"
    test_results+=("daemon_config:FAIL")
    failed_tests=$((failed_tests + 1))
fi

# Test 8: TLS connection security
if test_tls_connection_security; then
    echo "✅ TLS connection security: PASS" >> "$detailed_report"
    test_results+=("connection_security:PASS")
else
    echo "❌ TLS connection security: FAIL" >> "$detailed_report"
    test_results+=("connection_security:FAIL")
    failed_tests=$((failed_tests + 1))
fi

# Generate report
generate_dind_report

# Final results
log "INFO" "Docker-in-Docker TLS validation completed"
log "INFO" "Results: $(printf '%s ' "${test_results[@]}")"

if [[ $failed_tests -eq 0 ]]; then
    log "INFO" "🎉 All Docker-in-Docker TLS tests PASSED!"
    exit 0
else
    log "ERROR" "💥 ${failed_tests} Docker-in-Docker TLS test(s) FAILED"
    exit 1
fi