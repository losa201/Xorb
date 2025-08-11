#!/bin/bash
# TLS Configuration Validation Script for XORB Platform
# Tests TLS protocols, cipher suites, and certificate validity

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_DIR="${SCRIPT_DIR}/../../reports/tls"
SECRETS_DIR="${SCRIPT_DIR}/../../secrets/tls"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
SERVICES=(
    "api:envoy-api:8443"
    "agent:envoy-agent:8444"
    "redis:redis:6380"
    "temporal:temporal:7234"
    "prometheus:prometheus:9093"
    "grafana:grafana:3001"
)

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help          Show this help message
    -s, --service NAME  Test specific service only
    -v, --verbose       Verbose output
    -r, --report-only   Generate report without running tests

EXAMPLES:
    $0                  # Test all services
    $0 -s api          # Test API service only
    $0 -v              # Verbose output
EOF
}

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

test_tls_connection() {
    local service_name="$1"
    local host="$2"
    local port="$3"
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    local client_cert="${SECRETS_DIR}/${service_name}-client/cert.pem"
    local client_key="${SECRETS_DIR}/${service_name}-client/key.pem"
    
    log "INFO" "Testing TLS connection to ${service_name} (${host}:${port})"
    
    # Check if service is responding
    if ! timeout 5 bash -c "</dev/tcp/${host}/${port}" 2>/dev/null; then
        log "ERROR" "Service ${service_name} is not responding on ${host}:${port}"
        return 1
    fi
    
    # Test TLS protocols
    local protocols=("tls1_2" "tls1_3")
    local protocol_results=()
    
    for protocol in "${protocols[@]}"; do
        log "DEBUG" "Testing ${protocol} for ${service_name}"
        
        local result=$(openssl s_client \
            -connect "${host}:${port}" \
            -${protocol} \
            -verify_return_error \
            -CAfile "${ca_file}" \
            -cert "${client_cert}" \
            -key "${client_key}" \
            -servername "${service_name}.xorb.local" \
            -quiet \
            </dev/null 2>&1 | grep -E "(Verification|Protocol|Cipher)" || true)
        
        if echo "$result" | grep -q "Verification: OK"; then
            protocol_results+=("${protocol}:PASS")
            log "INFO" "${service_name}: ${protocol} - PASS"
        else
            protocol_results+=("${protocol}:FAIL")
            log "WARN" "${service_name}: ${protocol} - FAIL"
        fi
    done
    
    # Test cipher suites
    local secure_ciphers=(
        "ECDHE-ECDSA-AES256-GCM-SHA384"
        "ECDHE-ECDSA-CHACHA20-POLY1305"
        "ECDHE-ECDSA-AES128-GCM-SHA256"
        "ECDHE-RSA-AES256-GCM-SHA384"
        "ECDHE-RSA-CHACHA20-POLY1305"
        "ECDHE-RSA-AES128-GCM-SHA256"
    )
    
    local cipher_results=()
    for cipher in "${secure_ciphers[@]}"; do
        log "DEBUG" "Testing cipher ${cipher} for ${service_name}"
        
        local cipher_test=$(openssl s_client \
            -connect "${host}:${port}" \
            -cipher "${cipher}" \
            -CAfile "${ca_file}" \
            -cert "${client_cert}" \
            -key "${client_key}" \
            -servername "${service_name}.xorb.local" \
            -quiet \
            </dev/null 2>&1 | grep "Cipher is" || true)
        
        if [[ -n "$cipher_test" ]]; then
            cipher_results+=("${cipher}:PASS")
            log "DEBUG" "${service_name}: Cipher ${cipher} - PASS"
        else
            cipher_results+=("${cipher}:FAIL")
            log "DEBUG" "${service_name}: Cipher ${cipher} - FAIL"
        fi
    done
    
    # Test certificate validation
    log "DEBUG" "Validating certificate for ${service_name}"
    
    local cert_info=$(openssl s_client \
        -connect "${host}:${port}" \
        -CAfile "${ca_file}" \
        -cert "${client_cert}" \
        -key "${client_key}" \
        -servername "${service_name}.xorb.local" \
        -showcerts \
        </dev/null 2>&1)
    
    local cert_subject=$(echo "$cert_info" | openssl x509 -noout -subject 2>/dev/null | sed 's/subject=//')
    local cert_issuer=$(echo "$cert_info" | openssl x509 -noout -issuer 2>/dev/null | sed 's/issuer=//')
    local cert_dates=$(echo "$cert_info" | openssl x509 -noout -dates 2>/dev/null)
    local cert_san=$(echo "$cert_info" | openssl x509 -noout -text 2>/dev/null | grep -A1 "Subject Alternative Name" | tail -1 || echo "None")
    
    # Check certificate expiration
    local cert_end_date=$(echo "$cert_dates" | grep "notAfter" | cut -d= -f2)
    local expiry_days=$(( ($(date -d "$cert_end_date" +%s) - $(date +%s)) / 86400 ))
    
    # Store results
    cat > "${REPORT_DIR}/${service_name}_tls_report.txt" << EOF
TLS Test Report for ${service_name}
Generated: $(date)
Host: ${host}:${port}

=== TLS PROTOCOLS ===
$(printf '%s\n' "${protocol_results[@]}")

=== CIPHER SUITES ===
$(printf '%s\n' "${cipher_results[@]}")

=== CERTIFICATE INFORMATION ===
Subject: ${cert_subject}
Issuer: ${cert_issuer}
${cert_dates}
Subject Alternative Names: ${cert_san}
Days until expiration: ${expiry_days}

=== CERTIFICATE VALIDATION ===
$(echo "$cert_info" | grep -E "(Verification|Verify return code)" || echo "Verification status unclear")

=== FULL OPENSSL OUTPUT ===
${cert_info}
EOF

    if [[ $expiry_days -lt 7 ]]; then
        log "ERROR" "${service_name}: Certificate expires in ${expiry_days} days!"
        return 1
    elif [[ $expiry_days -lt 30 ]]; then
        log "WARN" "${service_name}: Certificate expires in ${expiry_days} days"
    else
        log "INFO" "${service_name}: Certificate valid for ${expiry_days} days"
    fi
    
    log "INFO" "${service_name}: TLS test completed"
    return 0
}

test_weak_protocols() {
    local service_name="$1"
    local host="$2"
    local port="$3"
    
    log "INFO" "Testing for weak TLS protocols on ${service_name}"
    
    local weak_protocols=("ssl2" "ssl3" "tls1" "tls1_1")
    local weak_found=false
    
    for protocol in "${weak_protocols[@]}"; do
        log "DEBUG" "Testing weak protocol ${protocol} for ${service_name}"
        
        if timeout 5 openssl s_client \
            -connect "${host}:${port}" \
            -${protocol} \
            -quiet \
            </dev/null >/dev/null 2>&1; then
            log "ERROR" "${service_name}: Weak protocol ${protocol} is ENABLED!"
            weak_found=true
        else
            log "DEBUG" "${service_name}: Weak protocol ${protocol} is disabled"
        fi
    done
    
    if [[ "$weak_found" == "false" ]]; then
        log "INFO" "${service_name}: No weak protocols detected"
        return 0
    else
        return 1
    fi
}

test_hsts_headers() {
    local service_name="$1"
    local host="$2"
    local port="$3"
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    local client_cert="${SECRETS_DIR}/${service_name}-client/cert.pem"
    local client_key="${SECRETS_DIR}/${service_name}-client/key.pem"
    
    log "INFO" "Testing HSTS headers for ${service_name}"
    
    # Skip HSTS test for non-HTTP services
    if [[ "$service_name" == "redis" || "$service_name" == "temporal" ]]; then
        log "DEBUG" "Skipping HSTS test for non-HTTP service ${service_name}"
        return 0
    fi
    
    local response=$(curl -s -I \
        --cacert "${ca_file}" \
        --cert "${client_cert}" \
        --key "${client_key}" \
        --connect-timeout 10 \
        "https://${host}:${port}/health" 2>/dev/null || echo "")
    
    if echo "$response" | grep -qi "strict-transport-security"; then
        local hsts_header=$(echo "$response" | grep -i "strict-transport-security" | tr -d '\r')
        log "INFO" "${service_name}: HSTS header present - ${hsts_header}"
        
        if echo "$hsts_header" | grep -qi "includeSubDomains"; then
            log "INFO" "${service_name}: HSTS includes subdomains"
        else
            log "WARN" "${service_name}: HSTS does not include subdomains"
        fi
        
        return 0
    else
        log "WARN" "${service_name}: HSTS header missing"
        return 1
    fi
}

generate_summary_report() {
    log "INFO" "Generating summary report"
    
    local summary_file="${REPORT_DIR}/tls_summary.html"
    
    cat > "$summary_file" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>XORB Platform TLS Security Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .service { background: #ecf0f1; margin: 10px 0; padding: 15px; border-radius: 5px; }
        .pass { color: #27ae60; font-weight: bold; }
        .fail { color: #e74c3c; font-weight: bold; }
        .warn { color: #f39c12; font-weight: bold; }
        .details { margin-top: 10px; font-family: monospace; background: #f8f9fa; padding: 10px; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔐 XORB Platform TLS Security Report</h1>
        <p>Generated: $(date)</p>
        <p>Environment: Production TLS/mTLS Configuration</p>
    </div>
EOF

    local total_services=0
    local passed_services=0
    
    for service_config in "${SERVICES[@]}"; do
        IFS=':' read -r service_name host port <<< "$service_config"
        total_services=$((total_services + 1))
        
        local report_file="${REPORT_DIR}/${service_name}_tls_report.txt"
        
        if [[ -f "$report_file" ]]; then
            local expiry_days=$(grep "Days until expiration" "$report_file" | cut -d: -f2 | tr -d ' ')
            local protocols_pass=$(grep -c ":PASS" "$report_file" | head -1 || echo "0")
            local verification=$(grep "Verification" "$report_file" || echo "Unknown")
            
            if [[ "$expiry_days" -gt 7 ]] && echo "$verification" | grep -q "OK"; then
                passed_services=$((passed_services + 1))
                local status_class="pass"
                local status="✅ PASS"
            else
                local status_class="fail"
                local status="❌ FAIL"
            fi
            
            cat >> "$summary_file" << EOF
    <div class="service">
        <h3>${service_name^} Service <span class="${status_class}">${status}</span></h3>
        <p><strong>Endpoint:</strong> ${host}:${port}</p>
        <p><strong>Certificate Expiry:</strong> ${expiry_days} days</p>
        <p><strong>TLS Protocols:</strong> ${protocols_pass} passed</p>
        <div class="details">
            <strong>Verification Status:</strong><br>
            ${verification}
        </div>
    </div>
EOF
        fi
    done
    
    cat >> "$summary_file" << EOF
    <div class="service">
        <h3>📊 Summary</h3>
        <p><strong>Total Services:</strong> ${total_services}</p>
        <p><strong>Passed:</strong> <span class="pass">${passed_services}</span></p>
        <p><strong>Failed:</strong> <span class="fail">$((total_services - passed_services))</span></p>
        <p><strong>Success Rate:</strong> $(( passed_services * 100 / total_services ))%</p>
    </div>
</body>
</html>
EOF

    log "INFO" "Summary report generated: ${summary_file}"
}

# Parse command line arguments
SPECIFIC_SERVICE=""
VERBOSE=false
REPORT_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--service)
            SPECIFIC_SERVICE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -r|--report-only)
            REPORT_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1" >&2
            usage
            exit 1
            ;;
    esac
done

# Create report directory
mkdir -p "$REPORT_DIR"

log "INFO" "Starting XORB Platform TLS validation"
log "INFO" "Report directory: ${REPORT_DIR}"

if [[ "$REPORT_ONLY" == "true" ]]; then
    generate_summary_report
    exit 0
fi

# Test services
test_results=()
failed_tests=0

for service_config in "${SERVICES[@]}"; do
    IFS=':' read -r service_name host port <<< "$service_config"
    
    # Skip if specific service requested and this isn't it
    if [[ -n "$SPECIFIC_SERVICE" && "$service_name" != "$SPECIFIC_SERVICE" ]]; then
        continue
    fi
    
    log "INFO" "Testing service: ${service_name}"
    
    # Run tests
    test_passed=true
    
    if ! test_tls_connection "$service_name" "$host" "$port"; then
        test_passed=false
    fi
    
    if ! test_weak_protocols "$service_name" "$host" "$port"; then
        test_passed=false
    fi
    
    if ! test_hsts_headers "$service_name" "$host" "$port"; then
        log "WARN" "HSTS test failed for ${service_name} (non-critical)"
    fi
    
    if [[ "$test_passed" == "true" ]]; then
        test_results+=("${service_name}:PASS")
        log "INFO" "${service_name}: All TLS tests PASSED"
    else
        test_results+=("${service_name}:FAIL")
        failed_tests=$((failed_tests + 1))
        log "ERROR" "${service_name}: TLS tests FAILED"
    fi
    
    echo "---"
done

# Generate summary report
generate_summary_report

# Final results
log "INFO" "TLS validation completed"
log "INFO" "Results: $(printf '%s ' "${test_results[@]}")"

if [[ $failed_tests -eq 0 ]]; then
    log "INFO" "🎉 All TLS tests PASSED!"
    exit 0
else
    log "ERROR" "💥 ${failed_tests} service(s) FAILED TLS validation"
    exit 1
fi