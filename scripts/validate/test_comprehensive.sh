#!/bin/bash
# Comprehensive TLS/mTLS Security Validation Suite for XORB Platform
# Orchestrates all security tests and generates unified reports

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORT_DIR="${SCRIPT_DIR}/../../reports/comprehensive"
LOG_DIR="${SCRIPT_DIR}/../../logs/validation"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test suite configuration
TESTS=(
    "tls:TLS Protocol & Cipher Validation:test_tls.sh"
    "mtls:Mutual TLS Authentication:test_mtls.sh"
    "redis:Redis TLS Configuration:test_redis_tls.sh"
    "dind:Docker-in-Docker TLS:test_dind_tls.sh"
)

PARALLEL_TESTS=true
GENERATE_REPORT=true
SEND_NOTIFICATIONS=false

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
        "SUCCESS")
            echo -e "${CYAN}[SUCCESS]${NC} ${timestamp} - $message"
            ;;
        "HEADER")
            echo -e "${PURPLE}[SUITE]${NC} ${timestamp} - $message"
            ;;
    esac
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -t, --test TYPE         Run specific test type (tls, mtls, redis, dind)
    -s, --sequential        Run tests sequentially instead of parallel
    -n, --no-report         Skip report generation
    -a, --alert             Send notifications on completion
    -v, --verbose           Verbose output
    -h, --help              Show this help message

EXAMPLES:
    $0                      # Run all tests in parallel
    $0 -t tls              # Run only TLS protocol tests
    $0 -s                  # Run all tests sequentially
    $0 -a                  # Run tests and send alerts
EOF
}

setup_environment() {
    # Create necessary directories
    mkdir -p "$REPORT_DIR" "$LOG_DIR"
    
    # Setup logging
    local log_file="${LOG_DIR}/comprehensive-$(date +%Y%m%d-%H%M%S).log"
    exec > >(tee -a "$log_file")
    exec 2>&1
    
    log "INFO" "Comprehensive TLS validation started - Log: $log_file"
    log "INFO" "Report directory: $REPORT_DIR"
}

check_prerequisites() {
    log "INFO" "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    for tool in openssl curl docker redis-cli; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log "ERROR" "Missing required tools: ${missing_tools[*]}"
        return 1
    fi
    
    # Check if services are running
    log "DEBUG" "Checking service availability..."
    
    local services_check=(
        "envoy-api:8443"
        "envoy-agent:8444"
        "redis:6380"
        "dind:2376"
    )
    
    for service_port in "${services_check[@]}"; do
        IFS=':' read -r service port <<< "$service_port"
        if ! timeout 5 bash -c "</dev/tcp/${service}/${port}" 2>/dev/null; then
            log "WARN" "Service $service not responding on port $port"
        else
            log "DEBUG" "Service $service responding on port $port"
        fi
    done
    
    # Check certificate files
    local cert_files=(
        "secrets/tls/ca/ca.pem"
        "secrets/tls/api/cert.pem"
        "secrets/tls/redis-client/cert.pem"
    )
    
    for cert_file in "${cert_files[@]}"; do
        if [[ ! -f "$cert_file" ]]; then
            log "ERROR" "Certificate file missing: $cert_file"
            return 1
        fi
    done
    
    log "SUCCESS" "Prerequisites check completed"
    return 0
}

run_test_suite() {
    local test_type="$1"
    local test_name="$2"
    local test_script="$3"
    
    log "HEADER" "Starting $test_name"
    
    local test_start_time=$(date +%s)
    local test_log="${LOG_DIR}/${test_type}-$(date +%Y%m%d-%H%M%S).log"
    local test_report="${REPORT_DIR}/${test_type}-results.json"
    
    # Run the specific test
    local exit_code=0
    if "${SCRIPT_DIR}/${test_script}" -v > "$test_log" 2>&1; then
        log "SUCCESS" "$test_name completed successfully"
    else
        exit_code=$?
        log "ERROR" "$test_name failed with exit code $exit_code"
    fi
    
    local test_end_time=$(date +%s)
    local test_duration=$((test_end_time - test_start_time))
    
    # Generate test result JSON
    cat > "$test_report" << EOF
{
  "test_type": "$test_type",
  "test_name": "$test_name", 
  "test_script": "$test_script",
  "start_time": "$test_start_time",
  "end_time": "$test_end_time",
  "duration_seconds": $test_duration,
  "exit_code": $exit_code,
  "status": $([ $exit_code -eq 0 ] && echo '"PASS"' || echo '"FAIL"'),
  "log_file": "$test_log",
  "timestamp": "$(date -Iseconds)"
}
EOF

    return $exit_code
}

run_parallel_tests() {
    log "INFO" "Running tests in parallel mode"
    
    local pids=()
    local test_results=()
    
    for test_config in "${TESTS[@]}"; do
        IFS=':' read -r test_type test_name test_script <<< "$test_config"
        
        # Skip if specific test type requested
        if [[ -n "${SPECIFIC_TEST:-}" && "$test_type" != "$SPECIFIC_TEST" ]]; then
            continue
        fi
        
        # Start test in background
        (run_test_suite "$test_type" "$test_name" "$test_script") &
        local pid=$!
        pids+=($pid)
        test_results+=("$test_type:$pid")
        
        log "DEBUG" "Started $test_name (PID: $pid)"
    done
    
    # Wait for all tests to complete
    local failed_tests=0
    for result in "${test_results[@]}"; do
        IFS=':' read -r test_type pid <<< "$result"
        
        if wait "$pid"; then
            log "SUCCESS" "$test_type test completed successfully"
        else
            log "ERROR" "$test_type test failed"
            failed_tests=$((failed_tests + 1))
        fi
    done
    
    return $failed_tests
}

run_sequential_tests() {
    log "INFO" "Running tests in sequential mode"
    
    local failed_tests=0
    
    for test_config in "${TESTS[@]}"; do
        IFS=':' read -r test_type test_name test_script <<< "$test_config"
        
        # Skip if specific test type requested
        if [[ -n "${SPECIFIC_TEST:-}" && "$test_type" != "$SPECIFIC_TEST" ]]; then
            continue
        fi
        
        if ! run_test_suite "$test_type" "$test_name" "$test_script"; then
            failed_tests=$((failed_tests + 1))
        fi
        
        echo "---"
    done
    
    return $failed_tests
}

generate_comprehensive_report() {
    if [[ "$GENERATE_REPORT" != "true" ]]; then
        return 0
    fi
    
    log "INFO" "Generating comprehensive security report"
    
    local report_file="${REPORT_DIR}/security-assessment-$(date +%Y%m%d-%H%M%S).html"
    local json_report="${REPORT_DIR}/security-assessment-$(date +%Y%m%d-%H%M%S).json"
    
    # Collect all test results
    local total_tests=0
    local passed_tests=0
    local test_results_json="["
    
    for result_file in "$REPORT_DIR"/*-results.json; do
        if [[ -f "$result_file" ]]; then
            if [[ "$test_results_json" != "[" ]]; then
                test_results_json+=","
            fi
            test_results_json+=$(cat "$result_file")
            
            total_tests=$((total_tests + 1))
            if grep -q '"status": "PASS"' "$result_file"; then
                passed_tests=$((passed_tests + 1))
            fi
        fi
    done
    test_results_json+="]"
    
    # Calculate metrics
    local success_rate=0
    if [[ $total_tests -gt 0 ]]; then
        success_rate=$(( passed_tests * 100 / total_tests ))
    fi
    
    # Generate JSON report
    cat > "$json_report" << EOF
{
  "assessment_type": "TLS/mTLS Security Validation",
  "platform": "XORB Platform",
  "timestamp": "$(date -Iseconds)",
  "summary": {
    "total_tests": $total_tests,
    "passed_tests": $passed_tests,
    "failed_tests": $((total_tests - passed_tests)),
    "success_rate": $success_rate
  },
  "test_results": $test_results_json,
  "environment": {
    "hostname": "$(hostname)",
    "user": "$(whoami)",
    "working_directory": "$(pwd)",
    "docker_compose_version": "$(docker-compose --version 2>/dev/null || echo 'Not available')",
    "openssl_version": "$(openssl version 2>/dev/null || echo 'Not available')"
  }
}
EOF

    # Generate HTML report
    cat > "$report_file" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XORB Platform - Comprehensive Security Assessment</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; color: #2c3e50; line-height: 1.6; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { background: white; padding: 25px; border-radius: 10px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1); transition: transform 0.3s ease; }
        .metric-card:hover { transform: translateY(-5px); }
        .metric-value { font-size: 3em; font-weight: bold; margin-bottom: 10px; }
        .metric-label { font-size: 1.1em; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }
        .success { color: #27ae60; }
        .warning { color: #f39c12; }
        .danger { color: #e74c3c; }
        .info { color: #3498db; }
        .test-results { background: white; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); overflow: hidden; }
        .test-results h2 { background: #34495e; color: white; padding: 20px; margin: 0; }
        .test-item { padding: 20px; border-bottom: 1px solid #ecf0f1; display: flex; justify-content: space-between; align-items: center; }
        .test-item:last-child { border-bottom: none; }
        .test-info h3 { margin-bottom: 5px; }
        .test-info p { color: #7f8c8d; }
        .test-status { font-weight: bold; font-size: 1.2em; }
        .status-pass { color: #27ae60; }
        .status-fail { color: #e74c3c; }
        .details { background: #ecf0f1; margin-top: 30px; padding: 20px; border-radius: 10px; }
        .timestamp { text-align: center; margin-top: 30px; color: #7f8c8d; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔐 XORB Platform Security Assessment</h1>
            <p>Comprehensive TLS/mTLS Validation Report</p>
            <p>Generated: $(date)</p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value info">$total_tests</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric-card">
                <div class="metric-value success">$passed_tests</div>
                <div class="metric-label">Passed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value danger">$((total_tests - passed_tests))</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value $([ $success_rate -ge 90 ] && echo 'success' || [ $success_rate -ge 70 ] && echo 'warning' || echo 'danger')">$success_rate%</div>
                <div class="metric-label">Success Rate</div>
            </div>
        </div>
        
        <div class="test-results">
            <h2>🧪 Test Results</h2>
EOF

    # Add test results to HTML
    for result_file in "$REPORT_DIR"/*-results.json; do
        if [[ -f "$result_file" ]]; then
            local test_type=$(jq -r '.test_type' "$result_file" 2>/dev/null || echo "unknown")
            local test_name=$(jq -r '.test_name' "$result_file" 2>/dev/null || echo "Unknown Test")
            local status=$(jq -r '.status' "$result_file" 2>/dev/null || echo "UNKNOWN")
            local duration=$(jq -r '.duration_seconds' "$result_file" 2>/dev/null || echo "0")
            
            local status_class="status-pass"
            local status_icon="✅"
            if [[ "$status" == "FAIL" ]]; then
                status_class="status-fail"
                status_icon="❌"
            fi
            
            cat >> "$report_file" << EOF
            <div class="test-item">
                <div class="test-info">
                    <h3>$test_name</h3>
                    <p>Type: $test_type | Duration: ${duration}s</p>
                </div>
                <div class="test-status $status_class">$status_icon $status</div>
            </div>
EOF
        fi
    done
    
    cat >> "$report_file" << EOF
        </div>
        
        <div class="details">
            <h3>📋 Security Assessment Summary</h3>
            <p><strong>Platform:</strong> XORB Platform TLS/mTLS Implementation</p>
            <p><strong>Scope:</strong> End-to-end transport security validation</p>
            <p><strong>Standards:</strong> TLS 1.2+, mTLS authentication, certificate management</p>
            <p><strong>Environment:</strong> $(hostname) | User: $(whoami)</p>
            
            <h3 style="margin-top: 20px;">🔍 Test Coverage</h3>
            <ul style="margin-left: 20px; margin-top: 10px;">
                <li><strong>TLS Protocol Validation:</strong> Version enforcement, cipher suite security</li>
                <li><strong>Mutual TLS Authentication:</strong> Client certificate verification</li>
                <li><strong>Redis TLS Configuration:</strong> Database encryption and authentication</li>
                <li><strong>Docker-in-Docker TLS:</strong> Container runtime security</li>
            </ul>
            
            <h3 style="margin-top: 20px;">📊 Recommendations</h3>
            <ul style="margin-left: 20px; margin-top: 10px;">
                $([ $success_rate -ge 95 ] && echo '<li style="color: #27ae60;">✅ Excellent security posture - maintain current practices</li>' || echo '<li style="color: #e74c3c;">⚠️ Review failed tests and implement recommended fixes</li>')
                <li>🔄 Continue automated certificate rotation every 30 days</li>
                <li>📈 Monitor TLS metrics and certificate expiry dates</li>
                <li>🛡️ Conduct quarterly security assessments</li>
            </ul>
        </div>
        
        <div class="timestamp">
            Report generated on $(date) | XORB Platform Security Team
        </div>
    </div>
</body>
</html>
EOF

    log "SUCCESS" "Comprehensive report generated: $report_file"
    log "INFO" "JSON report: $json_report"
}

send_notifications() {
    if [[ "$SEND_NOTIFICATIONS" != "true" ]]; then
        return 0
    fi
    
    log "INFO" "Sending notification alerts"
    
    # Collect results summary
    local total_tests=0
    local failed_tests=0
    
    for result_file in "$REPORT_DIR"/*-results.json; do
        if [[ -f "$result_file" ]]; then
            total_tests=$((total_tests + 1))
            if grep -q '"status": "FAIL"' "$result_file"; then
                failed_tests=$((failed_tests + 1))
            fi
        fi
    done
    
    local status_emoji="✅"
    local status_text="PASSED"
    local color="good"
    
    if [[ $failed_tests -gt 0 ]]; then
        status_emoji="❌"
        status_text="FAILED"
        color="danger"
    fi
    
    # Slack notification (if webhook configured)
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -s -X POST "$SLACK_WEBHOOK_URL" \
             -H "Content-Type: application/json" \
             -d "{
                \"attachments\": [{
                    \"color\": \"$color\",
                    \"title\": \"$status_emoji XORB Platform Security Assessment\",
                    \"fields\": [
                        {\"title\": \"Status\", \"value\": \"$status_text\", \"short\": true},
                        {\"title\": \"Tests\", \"value\": \"$((total_tests - failed_tests))/$total_tests passed\", \"short\": true},
                        {\"title\": \"Environment\", \"value\": \"$(hostname)\", \"short\": true},
                        {\"title\": \"Timestamp\", \"value\": \"$(date)\", \"short\": true}
                    ]
                }]
             }" >/dev/null 2>&1
        
        log "INFO" "Slack notification sent"
    fi
    
    # Email notification (if configured)
    if [[ -n "${EMAIL_RECIPIENT:-}" ]] && command -v mail >/dev/null 2>&1; then
        local subject="XORB Security Assessment - $status_text"
        local body="Security assessment completed on $(hostname) at $(date)\n\nResults: $((total_tests - failed_tests))/$total_tests tests passed\n\nFailed tests: $failed_tests"
        
        echo -e "$body" | mail -s "$subject" "$EMAIL_RECIPIENT"
        log "INFO" "Email notification sent to $EMAIL_RECIPIENT"
    fi
}

cleanup() {
    log "INFO" "Cleaning up temporary files"
    
    # Remove old log files (keep last 10)
    find "$LOG_DIR" -name "comprehensive-*.log" -type f | sort -r | tail -n +11 | xargs rm -f 2>/dev/null || true
    
    # Remove old report files (keep last 5)
    find "$REPORT_DIR" -name "*-results.json" -type f | sort -r | tail -n +6 | xargs rm -f 2>/dev/null || true
}

# Parse command line arguments
SPECIFIC_TEST=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--test)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        -s|--sequential)
            PARALLEL_TESTS=false
            shift
            ;;
        -n|--no-report)
            GENERATE_REPORT=false
            shift
            ;;
        -a|--alert)
            SEND_NOTIFICATIONS=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
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

# Main execution
main() {
    local start_time=$(date +%s)
    
    setup_environment
    
    log "HEADER" "🚀 Starting XORB Platform Comprehensive Security Assessment"
    
    # Check prerequisites
    if ! check_prerequisites; then
        log "ERROR" "Prerequisites check failed - aborting"
        exit 1
    fi
    
    # Run tests
    local failed_tests=0
    if [[ "$PARALLEL_TESTS" == "true" ]]; then
        if ! run_parallel_tests; then
            failed_tests=$?
        fi
    else
        if ! run_sequential_tests; then
            failed_tests=$?
        fi
    fi
    
    # Generate reports
    generate_comprehensive_report
    
    # Send notifications
    send_notifications
    
    # Cleanup
    cleanup
    
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    
    # Final summary
    log "HEADER" "🏁 Security Assessment Completed"
    log "INFO" "Total Duration: ${total_duration}s"
    log "INFO" "Reports: $REPORT_DIR"
    
    if [[ $failed_tests -eq 0 ]]; then
        log "SUCCESS" "🎉 All security tests PASSED!"
        exit 0
    else
        log "ERROR" "💥 $failed_tests test suite(s) FAILED"
        exit 1
    fi
}

# Execute main function
main "$@"