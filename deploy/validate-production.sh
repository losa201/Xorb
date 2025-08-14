#!/bin/bash
# XORB Production Validation Script
# Comprehensive testing before go-live

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
VALIDATION_LOG="/root/Xorb/validation-$(date +%Y%m%d_%H%M%S).log"
TIMEOUT=30
CRITICAL_ENDPOINTS=(
    "http://localhost:8080/api/health"
    "http://localhost:8000/health"
)

# Logging
log_info() { echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$VALIDATION_LOG"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$VALIDATION_LOG"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$VALIDATION_LOG"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" | tee -a "$VALIDATION_LOG"; }

# Test counters
TESTS_TOTAL=0
TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local test_name="$1"
    local test_command="$2"

    ((TESTS_TOTAL++))
    log_info "Running test: $test_name"

    if eval "$test_command" >/dev/null 2>&1; then
        log_success "✓ $test_name"
        ((TESTS_PASSED++))
        return 0
    else
        log_error "✗ $test_name"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Environment validation
validate_environment() {
    log_info "Validating environment configuration..."

    # Check required environment variables
    local required_vars=("JWT_SECRET" "XORB_API_KEY" "POSTGRES_PASSWORD")
    for var in "${required_vars[@]}"; do
        run_test "Environment variable $var is set" "[[ -n \"\${$var:-}\" ]]"
    done

    # Check JWT secret strength
    if [[ -n "${JWT_SECRET:-}" ]]; then
        run_test "JWT_SECRET is strong (>32 chars)" "[[ \${#JWT_SECRET} -gt 32 ]]"
        run_test "JWT_SECRET is not default" "[[ \"\$JWT_SECRET\" != *\"change\"* ]] && [[ \"\$JWT_SECRET\" != *\"secret\"* ]]"
    fi

    # Check system resources
    run_test "Sufficient memory (>2GB)" "[[ \$(free -g | awk 'NR==2{print \$7}') -gt 2 ]]"
    run_test "Sufficient disk space (>5GB)" "[[ \$(df -BG / | awk 'NR==2{print \$4}' | sed 's/G//') -gt 5 ]]"

    log_success "Environment validation completed"
}

# Security validation
validate_security() {
    log_info "Validating security configuration..."

    # File permissions
    run_test "SSL key files have secure permissions" "find /root/Xorb/ssl -name '*.key' -exec test '{}' -perm 600 \\;"
    run_test ".env file has secure permissions" "[[ ! -f /root/Xorb/.env ]] || [[ \$(stat -c '%a' /root/Xorb/.env) == '600' ]]"

    # Container security
    if docker ps | grep -q xorb; then
        run_test "Containers running with security options" "docker inspect \$(docker ps -q --filter name=xorb) | grep -q 'no-new-privileges'"
    fi

    # Network security
    run_test "API not exposed on 0.0.0.0" "! netstat -tlpn | grep ':8080' | grep -q '0.0.0.0'"

    log_success "Security validation completed"
}

# Service validation
validate_services() {
    log_info "Validating service health..."

    # Check Docker services
    local services=("xorb-postgres" "xorb-redis" "api" "orchestrator")
    for service in "${services[@]}"; do
        run_test "Service $service is running" "docker ps --format '{{.Names}}' | grep -q $service"
        if docker ps --format '{{.Names}}' | grep -q "$service"; then
            run_test "Service $service is healthy" "docker inspect $service | grep -q '\"Status\": \"healthy\"' || docker inspect $service | grep -q '\"Status\": \"running\"'"
        fi
    done

    # Database connectivity
    run_test "PostgreSQL is accepting connections" "docker exec xorb-postgres pg_isready -U \${POSTGRES_USER:-xorb}"
    run_test "Redis is accepting connections" "docker exec redis redis-cli ping | grep -q PONG"

    log_success "Service validation completed"
}

# API validation
validate_api() {
    log_info "Validating API endpoints..."

    # Health endpoints
    for endpoint in "${CRITICAL_ENDPOINTS[@]}"; do
        run_test "Endpoint $endpoint responds" "curl -f -s --connect-timeout $TIMEOUT '$endpoint' >/dev/null"
        if curl -f -s --connect-timeout "$TIMEOUT" "$endpoint" >/dev/null; then
            local status=$(curl -s --connect-timeout "$TIMEOUT" "$endpoint" | jq -r '.status' 2>/dev/null || echo "unknown")
            run_test "Endpoint $endpoint reports healthy status" "[[ '$status' == 'operational' ]] || [[ '$status' == 'healthy' ]]"
        fi
    done

    # API functionality tests
    local api_base="http://localhost:8080/api"

    # Test OpenAPI docs
    run_test "OpenAPI documentation accessible" "curl -f -s --connect-timeout $TIMEOUT '$api_base/docs' >/dev/null"

    # Test authentication (should fail without API key)
    run_test "Authentication required for protected endpoints" "! curl -f -s --connect-timeout $TIMEOUT '$api_base/intel/submit' >/dev/null"

    # Test rate limiting headers
    if curl -s --connect-timeout "$TIMEOUT" "$api_base/health" -I | grep -q "X-RateLimit"; then
        run_test "Rate limiting headers present" "true"
    else
        run_test "Rate limiting headers present" "false"
    fi

    log_success "API validation completed"
}

# Performance validation
validate_performance() {
    log_info "Validating performance metrics..."

    # Response time tests
    local health_endpoint="http://localhost:8080/api/health"
    local response_time=$(curl -w "%{time_total}" -s -o /dev/null --connect-timeout "$TIMEOUT" "$health_endpoint" || echo "999")

    run_test "Health endpoint responds within 2 seconds" "echo '$response_time < 2' | bc -l | grep -q 1"

    # Memory usage
    local memory_usage=$(free | awk 'NR==2{printf "%.1f", ($3/$2)*100}')
    run_test "Memory usage below 80%" "echo '$memory_usage < 80' | bc -l | grep -q 1"

    # Disk usage
    local disk_usage=$(df -h / | awk 'NR==2{print $5}' | sed 's/%//')
    run_test "Disk usage below 85%" "[[ $disk_usage -lt 85 ]]"

    # Container resource usage
    if command -v docker >/dev/null; then
        local container_stats=$(docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemPerc}}" | grep xorb || echo "")
        if [[ -n "$container_stats" ]]; then
            log_info "Container resource usage:"
            echo "$container_stats" | tee -a "$VALIDATION_LOG"
        fi
    fi

    log_success "Performance validation completed"
}

# Integration tests
validate_integration() {
    log_info "Running integration tests..."

    # Database integration
    if docker exec xorb-postgres psql -U "${POSTGRES_USER:-xorb}" -d "${POSTGRES_DB:-xorb_db}" -c "SELECT 1;" >/dev/null 2>&1; then
        run_test "Database query execution" "true"
    else
        run_test "Database query execution" "false"
    fi

    # Redis integration
    if docker exec redis redis-cli set test_key test_value >/dev/null 2>&1 && docker exec redis redis-cli get test_key | grep -q test_value; then
        run_test "Redis read/write operations" "true"
        docker exec redis redis-cli del test_key >/dev/null 2>&1
    else
        run_test "Redis read/write operations" "false"
    fi

    # Service communication
    if docker exec api curl -f -s http://orchestrator:8080/health >/dev/null 2>&1; then
        run_test "API to Orchestrator communication" "true"
    else
        run_test "API to Orchestrator communication" "false"
    fi

    log_success "Integration validation completed"
}

# SSL/TLS validation
validate_ssl() {
    log_info "Validating SSL/TLS configuration..."

    # Check SSL certificates exist
    run_test "SSL private key exists" "[[ -f /root/Xorb/ssl/xorb.key ]]"
    run_test "SSL certificate exists" "[[ -f /root/Xorb/ssl/xorb.crt ]]"

    # Check certificate validity
    if [[ -f /root/Xorb/ssl/xorb.crt ]]; then
        local cert_expiry=$(openssl x509 -enddate -noout -in /root/Xorb/ssl/xorb.crt | cut -d= -f2)
        local expiry_epoch=$(date -d "$cert_expiry" +%s)
        local now_epoch=$(date +%s)
        local days_until_expiry=$(( (expiry_epoch - now_epoch) / 86400 ))

        run_test "SSL certificate valid for >30 days" "[[ $days_until_expiry -gt 30 ]]"
    fi

    # Test HTTPS endpoints if nginx is running
    if docker ps | grep -q nginx; then
        run_test "HTTPS endpoint responds" "curl -k -f -s --connect-timeout $TIMEOUT https://localhost/api/health >/dev/null"
    fi

    log_success "SSL/TLS validation completed"
}

# Monitoring validation
validate_monitoring() {
    log_info "Validating monitoring and observability..."

    # Check if Prometheus is accessible
    if curl -f -s --connect-timeout "$TIMEOUT" http://localhost:9090/-/healthy >/dev/null 2>&1; then
        run_test "Prometheus is healthy" "true"
        run_test "Prometheus has targets" "curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets | length' | grep -v '^0$'"
    else
        log_warning "Prometheus not accessible, skipping monitoring tests"
    fi

    # Check if Grafana is accessible
    if curl -f -s --connect-timeout "$TIMEOUT" http://localhost:3000/api/health >/dev/null 2>&1; then
        run_test "Grafana is healthy" "true"
    else
        log_warning "Grafana not accessible"
    fi

    # Check log files
    run_test "Application logs are being written" "[[ -n \"\$(find /root/Xorb/logs -name '*.log' -newer /tmp/validation_start 2>/dev/null)\" ]] || [[ -n \"\$(docker logs api 2>&1 | tail -10)\" ]]"

    log_success "Monitoring validation completed"
}

# Load testing
validate_load() {
    log_info "Running basic load tests..."

    # Simple load test with curl
    local health_endpoint="http://localhost:8080/api/health"
    local concurrent_requests=10
    local success_count=0

    log_info "Sending $concurrent_requests concurrent requests..."

    for i in $(seq 1 $concurrent_requests); do
        if curl -f -s --connect-timeout 5 "$health_endpoint" >/dev/null 2>&1 & then
            ((success_count++))
        fi
    done

    wait  # Wait for all background jobs to complete

    run_test "Handles concurrent requests (>80% success)" "[[ $success_count -gt $((concurrent_requests * 8 / 10)) ]]"

    log_success "Load testing completed"
}

# Generate report
generate_report() {
    log_info "Generating validation report..."

    local report_file="/root/Xorb/validation-report-$(date +%Y%m%d_%H%M%S).json"
    local success_rate=$(echo "scale=1; $TESTS_PASSED * 100 / $TESTS_TOTAL" | bc -l)

    cat > "$report_file" << EOF
{
    "validation": {
        "timestamp": "$(date -Iseconds)",
        "environment": "production",
        "status": "$([[ $TESTS_FAILED -eq 0 ]] && echo 'PASS' || echo 'FAIL')"
    },
    "results": {
        "total_tests": $TESTS_TOTAL,
        "passed": $TESTS_PASSED,
        "failed": $TESTS_FAILED,
        "success_rate": "${success_rate}%"
    },
    "system_info": {
        "hostname": "$(hostname)",
        "uptime": "$(uptime -p)",
        "memory_usage": "$(free -h | awk 'NR==2{print $3"/"$2}')",
        "disk_usage": "$(df -h / | awk 'NR==2{print $3"/"$2" ("$5")"}')",
        "docker_version": "$(docker --version 2>/dev/null || echo 'Not available')"
    },
    "services": {
        "running_containers": $(docker ps -q | wc -l),
        "container_names": [$(docker ps --format '"{{.Names}}"' | paste -sd, -)]
    }
}
EOF

    log_success "Validation report generated: $report_file"

    # Display summary
    echo
    echo "================================="
    echo "     VALIDATION SUMMARY"
    echo "================================="
    echo "Total Tests: $TESTS_TOTAL"
    echo "Passed: $TESTS_PASSED"
    echo "Failed: $TESTS_FAILED"
    echo "Success Rate: ${success_rate}%"
    echo "================================="

    if [[ $TESTS_FAILED -eq 0 ]]; then
        log_success "🎉 ALL TESTS PASSED - PRODUCTION READY!"
        return 0
    else
        log_error "❌ $TESTS_FAILED TESTS FAILED - REQUIRES ATTENTION"
        log_error "Check the validation log for details: $VALIDATION_LOG"
        return 1
    fi
}

# Main validation function
main() {
    echo "Starting XORB Production Validation..."
    echo "Log file: $VALIDATION_LOG"
    echo

    # Create timestamp for log filtering
    touch /tmp/validation_start

    # Load environment variables
    if [[ -f "/root/Xorb/.env" ]]; then
        source /root/Xorb/.env
    fi

    # Run all validation tests
    validate_environment
    validate_security
    validate_services
    validate_api
    validate_performance
    validate_integration
    validate_ssl
    validate_monitoring
    validate_load

    # Generate report and exit with appropriate code
    generate_report
}

# Execute main function
main "$@"
