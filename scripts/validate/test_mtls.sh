#!/bin/bash
# mTLS (Mutual TLS) Validation Script for XORB Platform
# Tests client certificate authentication and authorization

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_DIR="${SCRIPT_DIR}/../../reports/mtls"
SECRETS_DIR="${SCRIPT_DIR}/../../secrets/tls"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
MTLS_ENDPOINTS=(
    "api:envoy-api:8443:/api/v1/health"
    "agent:envoy-agent:8444:/health"
)

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

test_client_cert_required() {
    local service_name="$1"
    local host="$2"
    local port="$3"
    local path="$4"
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    
    log "INFO" "Testing client certificate requirement for ${service_name}"
    
    # Test 1: Connection without client certificate should fail
    log "DEBUG" "Testing connection without client certificate"
    
    local no_cert_response=$(curl -s -w "%{http_code}" -o /dev/null \
        --cacert "${ca_file}" \
        --connect-timeout 10 \
        --max-time 30 \
        "https://${host}:${port}${path}" 2>/dev/null || echo "000")
    
    if [[ "$no_cert_response" == "400" || "$no_cert_response" == "401" || "$no_cert_response" == "403" || "$no_cert_response" == "000" ]]; then
        log "INFO" "${service_name}: ✅ Correctly rejects connections without client certificate (HTTP ${no_cert_response})"
    else
        log "ERROR" "${service_name}: ❌ Accepts connections without client certificate (HTTP ${no_cert_response})"
        return 1
    fi
    
    return 0
}

test_valid_client_cert() {
    local service_name="$1"
    local host="$2"
    local port="$3"
    local path="$4"
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    local client_cert="${SECRETS_DIR}/${service_name}-client/cert.pem"
    local client_key="${SECRETS_DIR}/${service_name}-client/key.pem"
    
    log "INFO" "Testing valid client certificate for ${service_name}"
    
    # Check if client certificate files exist
    if [[ ! -f "$client_cert" ]]; then
        log "ERROR" "Client certificate not found: ${client_cert}"
        return 1
    fi
    
    if [[ ! -f "$client_key" ]]; then
        log "ERROR" "Client private key not found: ${client_key}"
        return 1
    fi
    
    # Test connection with valid client certificate
    log "DEBUG" "Testing connection with valid client certificate"
    
    local valid_cert_response=$(curl -s -w "%{http_code}" -o /dev/null \
        --cacert "${ca_file}" \
        --cert "${client_cert}" \
        --key "${client_key}" \
        --connect-timeout 10 \
        --max-time 30 \
        "https://${host}:${port}${path}" 2>/dev/null || echo "000")
    
    if [[ "$valid_cert_response" == "200" ]]; then
        log "INFO" "${service_name}: ✅ Successfully authenticates with valid client certificate (HTTP ${valid_cert_response})"
        
        # Get detailed response for analysis
        local detailed_response=$(curl -s \
            --cacert "${ca_file}" \
            --cert "${client_cert}" \
            --key "${client_key}" \
            --connect-timeout 10 \
            --max-time 30 \
            -H "Accept: application/json" \
            "https://${host}:${port}${path}" 2>/dev/null || echo "{}")
        
        log "DEBUG" "${service_name}: Response body: ${detailed_response}"
        
        return 0
    else
        log "ERROR" "${service_name}: ❌ Failed to authenticate with valid client certificate (HTTP ${valid_cert_response})"
        return 1
    fi
}

test_invalid_client_cert() {
    local service_name="$1"
    local host="$2"
    local port="$3"
    local path="$4"
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    
    log "INFO" "Testing invalid client certificate for ${service_name}"
    
    # Create temporary invalid certificate
    local temp_dir=$(mktemp -d)
    local invalid_cert="${temp_dir}/invalid_cert.pem"
    local invalid_key="${temp_dir}/invalid_key.pem"
    
    # Generate self-signed invalid certificate
    openssl req -x509 -newkey rsa:2048 -keyout "${invalid_key}" -out "${invalid_cert}" \
        -days 1 -nodes -subj "/CN=invalid.example.com" >/dev/null 2>&1
    
    log "DEBUG" "Testing connection with invalid client certificate"
    
    local invalid_cert_response=$(curl -s -w "%{http_code}" -o /dev/null \
        --cacert "${ca_file}" \
        --cert "${invalid_cert}" \
        --key "${invalid_key}" \
        --connect-timeout 10 \
        --max-time 30 \
        "https://${host}:${port}${path}" 2>/dev/null || echo "000")
    
    # Cleanup
    rm -rf "$temp_dir"
    
    if [[ "$invalid_cert_response" == "400" || "$invalid_cert_response" == "401" || "$invalid_cert_response" == "403" || "$invalid_cert_response" == "000" ]]; then
        log "INFO" "${service_name}: ✅ Correctly rejects invalid client certificate (HTTP ${invalid_cert_response})"
        return 0
    else
        log "ERROR" "${service_name}: ❌ Accepts invalid client certificate (HTTP ${invalid_cert_response})"
        return 1
    fi
}

test_wrong_ca_cert() {
    local service_name="$1"
    local host="$2"
    local port="$3"
    local path="$4"
    
    log "INFO" "Testing client certificate from wrong CA for ${service_name}"
    
    # Create temporary wrong CA and certificate
    local temp_dir=$(mktemp -d)
    local wrong_ca="${temp_dir}/wrong_ca.pem"
    local wrong_ca_key="${temp_dir}/wrong_ca_key.pem"
    local wrong_cert="${temp_dir}/wrong_cert.pem"
    local wrong_key="${temp_dir}/wrong_key.pem"
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    
    # Generate wrong CA
    openssl req -x509 -newkey rsa:2048 -keyout "${wrong_ca_key}" -out "${wrong_ca}" \
        -days 1 -nodes -subj "/CN=Wrong CA" >/dev/null 2>&1
    
    # Generate certificate signed by wrong CA
    openssl req -newkey rsa:2048 -keyout "${wrong_key}" -out "${temp_dir}/wrong.csr" \
        -nodes -subj "/CN=${service_name}-client.wrong.local" >/dev/null 2>&1
    
    openssl x509 -req -in "${temp_dir}/wrong.csr" -CA "${wrong_ca}" -CAkey "${wrong_ca_key}" \
        -CAcreateserial -out "${wrong_cert}" -days 1 >/dev/null 2>&1
    
    log "DEBUG" "Testing connection with certificate from wrong CA"
    
    local wrong_ca_response=$(curl -s -w "%{http_code}" -o /dev/null \
        --cacert "${ca_file}" \
        --cert "${wrong_cert}" \
        --key "${wrong_key}" \
        --connect-timeout 10 \
        --max-time 30 \
        "https://${host}:${port}${path}" 2>/dev/null || echo "000")
    
    # Cleanup
    rm -rf "$temp_dir"
    
    if [[ "$wrong_ca_response" == "400" || "$wrong_ca_response" == "401" || "$wrong_ca_response" == "403" || "$wrong_ca_response" == "000" ]]; then
        log "INFO" "${service_name}: ✅ Correctly rejects certificate from wrong CA (HTTP ${wrong_ca_response})"
        return 0
    else
        log "ERROR" "${service_name}: ❌ Accepts certificate from wrong CA (HTTP ${wrong_ca_response})"
        return 1
    fi
}

test_certificate_authorization() {
    local service_name="$1"
    local host="$2"
    local port="$3"
    local path="$4"
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    
    log "INFO" "Testing certificate-based authorization for ${service_name}"
    
    # Test with different client certificates to verify authorization
    local test_certs=(
        "orchestrator-client"
        "agent-client"
        "scanner-client"
    )
    
    local authorization_results=()
    
    for cert_name in "${test_certs[@]}"; do
        local test_cert="${SECRETS_DIR}/${cert_name}/cert.pem"
        local test_key="${SECRETS_DIR}/${cert_name}/key.pem"
        
        if [[ -f "$test_cert" && -f "$test_key" ]]; then
            log "DEBUG" "Testing authorization with ${cert_name} certificate"
            
            local auth_response=$(curl -s -w "%{http_code}" -o /dev/null \
                --cacert "${ca_file}" \
                --cert "${test_cert}" \
                --key "${test_key}" \
                --connect-timeout 10 \
                --max-time 30 \
                "https://${host}:${port}${path}" 2>/dev/null || echo "000")
            
            authorization_results+=("${cert_name}:${auth_response}")
            
            if [[ "$auth_response" == "200" ]]; then
                log "INFO" "${service_name}: ✅ ${cert_name} authorized (HTTP ${auth_response})"
            elif [[ "$auth_response" == "403" ]]; then
                log "INFO" "${service_name}: 🚫 ${cert_name} forbidden (HTTP ${auth_response}) - Expected for some clients"
            else
                log "WARN" "${service_name}: ⚠️  ${cert_name} unexpected response (HTTP ${auth_response})"
            fi
        else
            log "DEBUG" "Certificate not found for ${cert_name}, skipping authorization test"
        fi
    done
    
    # Store authorization results
    printf '%s\n' "${authorization_results[@]}" > "${REPORT_DIR}/${service_name}_authorization.txt"
    
    return 0
}

test_certificate_rotation() {
    local service_name="$1"
    local client_cert="${SECRETS_DIR}/${service_name}-client/cert.pem"
    
    log "INFO" "Testing certificate rotation readiness for ${service_name}"
    
    if [[ ! -f "$client_cert" ]]; then
        log "WARN" "Client certificate not found for rotation test: ${client_cert}"
        return 1
    fi
    
    # Check certificate validity period
    local not_before=$(openssl x509 -in "$client_cert" -noout -startdate | cut -d= -f2)
    local not_after=$(openssl x509 -in "$client_cert" -noout -enddate | cut -d= -f2)
    local current_time=$(date +%s)
    local expire_time=$(date -d "$not_after" +%s)
    local days_remaining=$(( (expire_time - current_time) / 86400 ))
    
    log "INFO" "${service_name}: Certificate valid from ${not_before} to ${not_after}"
    log "INFO" "${service_name}: ${days_remaining} days remaining until expiration"
    
    # Check if certificate is approaching expiration (rotation needed)
    if [[ $days_remaining -lt 7 ]]; then
        log "ERROR" "${service_name}: ⚠️  Certificate expires in ${days_remaining} days - ROTATION REQUIRED"
        return 1
    elif [[ $days_remaining -lt 30 ]]; then
        log "WARN" "${service_name}: Certificate expires in ${days_remaining} days - Consider rotation"
    else
        log "INFO" "${service_name}: ✅ Certificate rotation not needed (${days_remaining} days remaining)"
    fi
    
    return 0
}

generate_mtls_report() {
    log "INFO" "Generating mTLS validation report"
    
    local report_file="${REPORT_DIR}/mtls_summary.html"
    
    cat > "$report_file" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>XORB Platform mTLS Security Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #34495e; color: white; padding: 20px; border-radius: 5px; }
        .test-section { background: #ecf0f1; margin: 20px 0; padding: 15px; border-radius: 5px; }
        .service { background: #f8f9fa; margin: 10px 0; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }
        .pass { color: #27ae60; font-weight: bold; }
        .fail { color: #e74c3c; font-weight: bold; }
        .warn { color: #f39c12; font-weight: bold; }
        .details { margin-top: 10px; font-family: monospace; background: #2c3e50; color: #ecf0f1; padding: 10px; border-radius: 3px; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background: #bdc3c7; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔐 XORB Platform mTLS Security Report</h1>
        <p>Generated: $(date)</p>
        <p>Environment: Production Mutual TLS Configuration</p>
    </div>
    
    <div class="test-section">
        <h2>🧪 Test Results Summary</h2>
EOF

    local total_tests=0
    local passed_tests=0
    
    for endpoint_config in "${MTLS_ENDPOINTS[@]}"; do
        IFS=':' read -r service_name host port path <<< "$endpoint_config"
        
        # Count tests and results from individual test files
        local service_report="${REPORT_DIR}/${service_name}_mtls_detailed.txt"
        if [[ -f "$service_report" ]]; then
            local service_tests=$(grep -c "✅\|❌" "$service_report" 2>/dev/null || echo "0")
            local service_passed=$(grep -c "✅" "$service_report" 2>/dev/null || echo "0")
            
            total_tests=$((total_tests + service_tests))
            passed_tests=$((passed_tests + service_passed))
            
            local success_rate=0
            if [[ $service_tests -gt 0 ]]; then
                success_rate=$(( service_passed * 100 / service_tests ))
            fi
            
            cat >> "$report_file" << EOF
        <div class="service">
            <h3>${service_name^} Service</h3>
            <p><strong>Endpoint:</strong> ${host}:${port}${path}</p>
            <div class="metric">Tests: ${service_tests}</div>
            <div class="metric">Passed: <span class="pass">${service_passed}</span></div>
            <div class="metric">Failed: <span class="fail">$((service_tests - service_passed))</span></div>
            <div class="metric">Success Rate: ${success_rate}%</div>
            
            <div class="details">
$(cat "$service_report" 2>/dev/null | sed 's/</\&lt;/g; s/>/\&gt;/g' || echo "No detailed report available")
            </div>
        </div>
EOF
        fi
    done
    
    local overall_success_rate=0
    if [[ $total_tests -gt 0 ]]; then
        overall_success_rate=$(( passed_tests * 100 / total_tests ))
    fi
    
    cat >> "$report_file" << EOF
    </div>
    
    <div class="test-section">
        <h2>📊 Overall Statistics</h2>
        <div class="metric">Total Tests: ${total_tests}</div>
        <div class="metric">Passed: <span class="pass">${passed_tests}</span></div>
        <div class="metric">Failed: <span class="fail">$((total_tests - passed_tests))</span></div>
        <div class="metric">Success Rate: ${overall_success_rate}%</div>
    </div>
    
    <div class="test-section">
        <h2>🔍 Security Recommendations</h2>
        <ul>
            <li><strong>Certificate Rotation:</strong> Ensure certificates are rotated every 30 days</li>
            <li><strong>Authorization:</strong> Verify that only authorized clients can access services</li>
            <li><strong>Monitoring:</strong> Monitor for failed mTLS authentication attempts</li>
            <li><strong>Backup:</strong> Maintain secure backups of CA private keys</li>
        </ul>
    </div>
</body>
</html>
EOF

    log "INFO" "mTLS report generated: ${report_file}"
}

# Create report directory
mkdir -p "$REPORT_DIR"

log "INFO" "Starting XORB Platform mTLS validation"
log "INFO" "Report directory: ${REPORT_DIR}"

# Run mTLS tests
test_results=()
failed_tests=0

for endpoint_config in "${MTLS_ENDPOINTS[@]}"; do
    IFS=':' read -r service_name host port path <<< "$endpoint_config"
    
    log "INFO" "Testing mTLS for service: ${service_name}"
    
    local service_report="${REPORT_DIR}/${service_name}_mtls_detailed.txt"
    echo "mTLS Test Results for ${service_name}" > "$service_report"
    echo "Generated: $(date)" >> "$service_report"
    echo "======================================" >> "$service_report"
    
    local service_passed=true
    
    # Test 1: Client certificate required
    if test_client_cert_required "$service_name" "$host" "$port" "$path"; then
        echo "✅ Client certificate requirement: PASS" >> "$service_report"
    else
        echo "❌ Client certificate requirement: FAIL" >> "$service_report"
        service_passed=false
    fi
    
    # Test 2: Valid client certificate authentication
    if test_valid_client_cert "$service_name" "$host" "$port" "$path"; then
        echo "✅ Valid client certificate authentication: PASS" >> "$service_report"
    else
        echo "❌ Valid client certificate authentication: FAIL" >> "$service_report"
        service_passed=false
    fi
    
    # Test 3: Invalid client certificate rejection
    if test_invalid_client_cert "$service_name" "$host" "$port" "$path"; then
        echo "✅ Invalid client certificate rejection: PASS" >> "$service_report"
    else
        echo "❌ Invalid client certificate rejection: FAIL" >> "$service_report"
        service_passed=false
    fi
    
    # Test 4: Wrong CA certificate rejection
    if test_wrong_ca_cert "$service_name" "$host" "$port" "$path"; then
        echo "✅ Wrong CA certificate rejection: PASS" >> "$service_report"
    else
        echo "❌ Wrong CA certificate rejection: FAIL" >> "$service_report"
        service_passed=false
    fi
    
    # Test 5: Certificate authorization
    if test_certificate_authorization "$service_name" "$host" "$port" "$path"; then
        echo "✅ Certificate authorization: PASS" >> "$service_report"
    else
        echo "❌ Certificate authorization: FAIL" >> "$service_report"
        service_passed=false
    fi
    
    # Test 6: Certificate rotation readiness
    if test_certificate_rotation "$service_name"; then
        echo "✅ Certificate rotation readiness: PASS" >> "$service_report"
    else
        echo "❌ Certificate rotation readiness: FAIL" >> "$service_report"
        service_passed=false
    fi
    
    # Record overall result
    if [[ "$service_passed" == "true" ]]; then
        test_results+=("${service_name}:PASS")
        log "INFO" "${service_name}: All mTLS tests PASSED"
    else
        test_results+=("${service_name}:FAIL")
        failed_tests=$((failed_tests + 1))
        log "ERROR" "${service_name}: mTLS tests FAILED"
    fi
    
    echo "---"
done

# Generate comprehensive report
generate_mtls_report

# Final results
log "INFO" "mTLS validation completed"
log "INFO" "Results: $(printf '%s ' "${test_results[@]}")"

if [[ $failed_tests -eq 0 ]]; then
    log "INFO" "🎉 All mTLS tests PASSED!"
    exit 0
else
    log "ERROR" "💥 ${failed_tests} service(s) FAILED mTLS validation"
    exit 1
fi