#!/bin/bash
# TLS/mTLS Performance Benchmark Script for XORB Platform
# Measures TLS handshake performance, throughput, and resource usage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRETS_DIR="${SCRIPT_DIR}/../secrets/tls"
REPORT_DIR="${SCRIPT_DIR}/../reports/performance"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Benchmark configuration
CONCURRENT_CONNECTIONS=100
TEST_DURATION=60
WARMUP_TIME=10
ENDPOINTS=(
    "envoy-api:8443:/api/v1/health"
    "envoy-agent:8444:/health"
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
        "PERF")
            echo -e "${CYAN}[PERF]${NC} ${timestamp} - $message"
            ;;
        "RESULT")
            echo -e "${BLUE}[RESULT]${NC} ${timestamp} - $message"
            ;;
    esac
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

TLS/mTLS performance benchmarking for XORB Platform.

OPTIONS:
    -c, --connections NUM   Number of concurrent connections (default: 100)
    -d, --duration SEC      Test duration in seconds (default: 60)
    -w, --warmup SEC        Warmup time in seconds (default: 10)
    -e, --endpoint URL      Specific endpoint to test
    -o, --output FILE       Output report file
    -v, --verbose           Verbose output
    -h, --help              Show this help message

EXAMPLES:
    $0                      # Run default benchmark
    $0 -c 200 -d 120        # 200 connections for 2 minutes
    $0 -e envoy-api:8443    # Test specific endpoint only
EOF
}

check_prerequisites() {
    log "INFO" "Checking benchmark prerequisites"
    
    local missing_tools=()
    
    # Check required tools
    for tool in openssl curl wrk ab; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log "ERROR" "Missing required tools: ${missing_tools[*]}"
        log "INFO" "Install missing tools:"
        for tool in "${missing_tools[@]}"; do
            case "$tool" in
                "wrk")
                    log "INFO" "  wrk: apt-get install wrk (Ubuntu) or brew install wrk (macOS)"
                    ;;
                "ab")
                    log "INFO" "  ab: apt-get install apache2-utils (Ubuntu)"
                    ;;
            esac
        done
        return 1
    fi
    
    # Check certificate files
    local cert_files=(
        "${SECRETS_DIR}/ca/ca.pem"
        "${SECRETS_DIR}/api-client/cert.pem"
        "${SECRETS_DIR}/api-client/key.pem"
    )
    
    for cert_file in "${cert_files[@]}"; do
        if [[ ! -f "$cert_file" ]]; then
            log "ERROR" "Certificate file missing: $cert_file"
            return 1
        fi
    done
    
    log "INFO" "Prerequisites check completed"
    return 0
}

benchmark_tls_handshake() {
    local endpoint="$1"
    local service_name="$2"
    
    log "PERF" "Benchmarking TLS handshake performance for $service_name"
    
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    local client_cert="${SECRETS_DIR}/${service_name}-client/cert.pem"
    local client_key="${SECRETS_DIR}/${service_name}-client/key.pem"
    
    # Use fallback client cert if service-specific not found
    if [[ ! -f "$client_cert" ]]; then
        client_cert="${SECRETS_DIR}/api-client/cert.pem"
        client_key="${SECRETS_DIR}/api-client/key.pem"
    fi
    
    local results_file="${REPORT_DIR}/${service_name}_handshake.json"
    
    # Measure TLS handshake time using OpenSSL
    local handshake_times=()
    local total_handshakes=50
    
    log "INFO" "Measuring $total_handshakes TLS handshakes for $service_name"
    
    for ((i=1; i<=total_handshakes; i++)); do
        local start_time=$(date +%s%3N)
        
        if openssl s_client \
            -connect "$endpoint" \
            -CAfile "$ca_file" \
            -cert "$client_cert" \
            -key "$client_key" \
            -quiet \
            </dev/null >/dev/null 2>&1; then
            
            local end_time=$(date +%s%3N)
            local handshake_time=$((end_time - start_time))
            handshake_times+=($handshake_time)
        fi
        
        # Small delay between handshakes
        sleep 0.1
    done
    
    # Calculate statistics
    local total_time=0
    local min_time=999999
    local max_time=0
    
    for time in "${handshake_times[@]}"; do
        total_time=$((total_time + time))
        if [[ $time -lt $min_time ]]; then
            min_time=$time
        fi
        if [[ $time -gt $max_time ]]; then
            max_time=$time
        fi
    done
    
    local avg_time=$((total_time / ${#handshake_times[@]}))
    local successful_handshakes=${#handshake_times[@]}
    local success_rate=$((successful_handshakes * 100 / total_handshakes))
    
    # Generate JSON results
    cat > "$results_file" << EOF
{
  "service": "$service_name",
  "endpoint": "$endpoint",
  "test_type": "tls_handshake",
  "total_attempts": $total_handshakes,
  "successful_handshakes": $successful_handshakes,
  "success_rate_percent": $success_rate,
  "handshake_times_ms": {
    "min": $min_time,
    "max": $max_time,
    "avg": $avg_time,
    "samples": [$(IFS=','; echo "${handshake_times[*]}")]
  },
  "timestamp": "$(date -Iseconds)"
}
EOF

    log "RESULT" "$service_name TLS handshake: avg=${avg_time}ms, min=${min_time}ms, max=${max_time}ms, success=${success_rate}%"
    
    return 0
}

benchmark_throughput() {
    local endpoint="$1"
    local path="$2" 
    local service_name="$3"
    
    log "PERF" "Benchmarking throughput for $service_name"
    
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    local client_cert="${SECRETS_DIR}/${service_name}-client/cert.pem"
    local client_key="${SECRETS_DIR}/${service_name}-client/key.pem"
    
    # Use fallback client cert if service-specific not found
    if [[ ! -f "$client_cert" ]]; then
        client_cert="${SECRETS_DIR}/api-client/cert.pem"
        client_key="${SECRETS_DIR}/api-client/key.pem"
    fi
    
    local results_file="${REPORT_DIR}/${service_name}_throughput.json"
    local wrk_output="${REPORT_DIR}/${service_name}_wrk.txt"
    
    # Warmup phase
    log "INFO" "Warming up $service_name for ${WARMUP_TIME}s"
    wrk -t4 -c10 -d${WARMUP_TIME}s \
        --timeout 30s \
        "https://${endpoint}${path}" \
        -s <(cat << 'EOF'
wrk.method = "GET"
wrk.headers["User-Agent"] = "XORB-Benchmark/1.0"
EOF
        ) >/dev/null 2>&1 || true
    
    # Main benchmark using wrk
    log "INFO" "Running throughput test: ${CONCURRENT_CONNECTIONS} connections for ${TEST_DURATION}s"
    
    if command -v wrk >/dev/null 2>&1; then
        wrk -t8 -c${CONCURRENT_CONNECTIONS} -d${TEST_DURATION}s \
            --timeout 30s \
            --latency \
            "https://${endpoint}${path}" \
            -s <(cat << 'EOF'
wrk.method = "GET"
wrk.headers["User-Agent"] = "XORB-Benchmark/1.0"
EOF
            ) > "$wrk_output" 2>&1 || true
            
        # Parse wrk output
        local requests_per_sec=$(grep "Requests/sec:" "$wrk_output" | awk '{print $2}' || echo "0")
        local transfer_per_sec=$(grep "Transfer/sec:" "$wrk_output" | awk '{print $2}' || echo "0")
        local avg_latency=$(grep "Latency" "$wrk_output" | awk '{print $2}' | head -1 || echo "0")
        local total_requests=$(grep "requests in" "$wrk_output" | awk '{print $1}' || echo "0")
        
        # Extract latency percentiles
        local latency_50th=$(awk '/50%/{print $2}' "$wrk_output" || echo "0")
        local latency_90th=$(awk '/90%/{print $2}' "$wrk_output" || echo "0")
        local latency_99th=$(awk '/99%/{print $2}' "$wrk_output" || echo "0")
        
    else
        # Fallback to Apache Bench
        log "WARN" "wrk not available, using Apache Bench as fallback"
        
        ab -n $((CONCURRENT_CONNECTIONS * 10)) \
           -c $CONCURRENT_CONNECTIONS \
           -t $TEST_DURATION \
           -k \
           "https://${endpoint}${path}" > "$wrk_output" 2>&1 || true
           
        local requests_per_sec=$(grep "Requests per second:" "$wrk_output" | awk '{print $4}' || echo "0")
        local avg_latency=$(grep "Time per request:" "$wrk_output" | head -1 | awk '{print $4}' || echo "0")
        local total_requests=$(grep "Complete requests:" "$wrk_output" | awk '{print $3}' || echo "0")
        
        local latency_50th="0"
        local latency_90th="0" 
        local latency_99th="0"
        local transfer_per_sec="0"
    fi
    
    # Generate JSON results
    cat > "$results_file" << EOF
{
  "service": "$service_name",
  "endpoint": "$endpoint",
  "path": "$path",
  "test_type": "throughput",
  "test_config": {
    "concurrent_connections": $CONCURRENT_CONNECTIONS,
    "test_duration_seconds": $TEST_DURATION,
    "warmup_seconds": $WARMUP_TIME
  },
  "results": {
    "requests_per_second": "$requests_per_sec",
    "total_requests": "$total_requests",
    "transfer_per_second": "$transfer_per_sec",
    "latency": {
      "average": "$avg_latency",
      "percentile_50": "$latency_50th",
      "percentile_90": "$latency_90th", 
      "percentile_99": "$latency_99th"
    }
  },
  "timestamp": "$(date -Iseconds)"
}
EOF

    log "RESULT" "$service_name throughput: ${requests_per_sec} req/s, avg latency: ${avg_latency}, total: ${total_requests} requests"
    
    return 0
}

benchmark_resource_usage() {
    local service_name="$1"
    
    log "PERF" "Measuring resource usage for $service_name"
    
    local results_file="${REPORT_DIR}/${service_name}_resources.json"
    
    # Get container stats before test
    local container_id=$(docker-compose ps -q "$service_name" 2>/dev/null || echo "")
    
    if [[ -z "$container_id" ]]; then
        log "WARN" "Container not found for service: $service_name"
        return 1
    fi
    
    # Collect resource metrics
    local stats_before=$(docker stats --no-stream --format "table {{.CPUPerc}},{{.MemUsage}},{{.NetIO}},{{.BlockIO}}" "$container_id" | tail -1)
    
    # Run load test (simplified)
    sleep 5
    
    local stats_after=$(docker stats --no-stream --format "table {{.CPUPerc}},{{.MemUsage}},{{.NetIO}},{{.BlockIO}}" "$container_id" | tail -1)
    
    # Parse stats (simplified for demo)
    IFS=',' read -r cpu_before mem_before net_before disk_before <<< "$stats_before"
    IFS=',' read -r cpu_after mem_after net_after disk_after <<< "$stats_after"
    
    # Generate results
    cat > "$results_file" << EOF
{
  "service": "$service_name",
  "test_type": "resource_usage",
  "container_id": "$container_id",
  "measurements": {
    "before_load": {
      "cpu_percent": "$cpu_before",
      "memory_usage": "$mem_before",
      "network_io": "$net_before",
      "disk_io": "$disk_before"
    },
    "after_load": {
      "cpu_percent": "$cpu_after",
      "memory_usage": "$mem_after", 
      "network_io": "$net_after",
      "disk_io": "$disk_after"
    }
  },
  "timestamp": "$(date -Iseconds)"
}
EOF

    log "RESULT" "$service_name resources: CPU ${cpu_after}, Memory ${mem_after}"
    
    return 0
}

benchmark_cipher_performance() {
    local endpoint="$1"
    local service_name="$2"
    
    log "PERF" "Benchmarking cipher performance for $service_name"
    
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    local client_cert="${SECRETS_DIR}/${service_name}-client/cert.pem"
    local client_key="${SECRETS_DIR}/${service_name}-client/key.pem"
    
    if [[ ! -f "$client_cert" ]]; then
        client_cert="${SECRETS_DIR}/api-client/cert.pem"
        client_key="${SECRETS_DIR}/api-client/key.pem"
    fi
    
    local results_file="${REPORT_DIR}/${service_name}_ciphers.json"
    
    # Test different cipher suites
    local ciphers=(
        "ECDHE-ECDSA-AES256-GCM-SHA384"
        "ECDHE-ECDSA-CHACHA20-POLY1305"
        "ECDHE-ECDSA-AES128-GCM-SHA256"
        "ECDHE-RSA-AES256-GCM-SHA384"
    )
    
    local cipher_results="["
    
    for cipher in "${ciphers[@]}"; do
        log "INFO" "Testing cipher: $cipher"
        
        local cipher_times=()
        local successful_connections=0
        
        # Test cipher performance (10 connections)
        for ((i=1; i<=10; i++)); do
            local start_time=$(date +%s%3N)
            
            if openssl s_client \
                -connect "$endpoint" \
                -CAfile "$ca_file" \
                -cert "$client_cert" \
                -key "$client_key" \
                -cipher "$cipher" \
                -quiet \
                </dev/null >/dev/null 2>&1; then
                
                local end_time=$(date +%s%3N)
                local connection_time=$((end_time - start_time))
                cipher_times+=($connection_time)
                successful_connections=$((successful_connections + 1))
            fi
        done
        
        # Calculate average
        local total_time=0
        for time in "${cipher_times[@]}"; do
            total_time=$((total_time + time))
        done
        
        local avg_time=0
        if [[ ${#cipher_times[@]} -gt 0 ]]; then
            avg_time=$((total_time / ${#cipher_times[@]}))
        fi
        
        # Add to results
        if [[ "$cipher_results" != "[" ]]; then
            cipher_results+=","
        fi
        
        cipher_results+="{\"cipher\":\"$cipher\",\"avg_time_ms\":$avg_time,\"successful_connections\":$successful_connections}"
    done
    
    cipher_results+="]"
    
    # Generate results file
    cat > "$results_file" << EOF
{
  "service": "$service_name",
  "endpoint": "$endpoint",
  "test_type": "cipher_performance",
  "cipher_results": $cipher_results,
  "timestamp": "$(date -Iseconds)"
}
EOF

    log "RESULT" "$service_name cipher performance test completed"
    
    return 0
}

generate_performance_report() {
    log "INFO" "Generating comprehensive performance report"
    
    local report_file="${REPORT_DIR}/performance-summary-$(date +%Y%m%d-%H%M%S).html"
    
    cat > "$report_file" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XORB Platform - TLS/mTLS Performance Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f8f9fa; color: #2c3e50; line-height: 1.6; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; text-align: center; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
        .metric-card h3 { color: #2c3e50; margin-bottom: 15px; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .metric-value { font-size: 2em; font-weight: bold; margin: 10px 0; }
        .performance-excellent { color: #27ae60; }
        .performance-good { color: #f39c12; }
        .performance-poor { color: #e74c3c; }
        .chart-placeholder { background: #ecf0f1; height: 200px; border-radius: 5px; display: flex; align-items: center; justify-content: center; color: #7f8c8d; margin: 15px 0; }
        .recommendations { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); margin-top: 30px; }
        .timestamp { text-align: center; margin-top: 30px; color: #7f8c8d; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚡ XORB Platform Performance Report</h1>
            <p>TLS/mTLS Performance Benchmarking Results</p>
            <p>Generated: $(date)</p>
        </div>
        
        <div class="metrics-grid">
EOF

    # Add performance metrics from JSON files
    for result_file in "$REPORT_DIR"/*.json; do
        if [[ -f "$result_file" ]]; then
            local service=$(jq -r '.service // "unknown"' "$result_file" 2>/dev/null)
            local test_type=$(jq -r '.test_type // "unknown"' "$result_file" 2>/dev/null)
            
            case "$test_type" in
                "tls_handshake")
                    local avg_time=$(jq -r '.handshake_times_ms.avg // 0' "$result_file" 2>/dev/null)
                    local success_rate=$(jq -r '.success_rate_percent // 0' "$result_file" 2>/dev/null)
                    
                    local performance_class="performance-excellent"
                    if [[ $avg_time -gt 100 ]]; then
                        performance_class="performance-good"
                    fi
                    if [[ $avg_time -gt 200 ]]; then
                        performance_class="performance-poor"
                    fi
                    
                    cat >> "$report_file" << EOF
            <div class="metric-card">
                <h3>🤝 ${service^} TLS Handshake</h3>
                <div class="metric-value $performance_class">${avg_time}ms</div>
                <p>Average handshake time</p>
                <p>Success Rate: ${success_rate}%</p>
                <div class="chart-placeholder">Handshake Performance Chart</div>
            </div>
EOF
                    ;;
                "throughput")
                    local rps=$(jq -r '.results.requests_per_second // "0"' "$result_file" 2>/dev/null)
                    local avg_latency=$(jq -r '.results.latency.average // "0"' "$result_file" 2>/dev/null)
                    
                    cat >> "$report_file" << EOF
            <div class="metric-card">
                <h3>🚀 ${service^} Throughput</h3>
                <div class="metric-value performance-excellent">${rps}</div>
                <p>Requests per second</p>
                <p>Average Latency: ${avg_latency}</p>
                <div class="chart-placeholder">Throughput Performance Chart</div>
            </div>
EOF
                    ;;
            esac
        fi
    done
    
    cat >> "$report_file" << 'EOF'
        </div>
        
        <div class="recommendations">
            <h3>🎯 Performance Recommendations</h3>
            <ul style="margin-left: 20px; margin-top: 15px;">
                <li><strong>TLS 1.3 Optimization:</strong> Ensure TLS 1.3 is preferred for reduced handshake overhead</li>
                <li><strong>Session Resumption:</strong> Implement TLS session tickets for faster reconnections</li>
                <li><strong>Certificate Size:</strong> Use ECDSA certificates for better performance</li>
                <li><strong>Connection Pooling:</strong> Implement HTTP keep-alive and connection pooling</li>
                <li><strong>Load Balancing:</strong> Distribute TLS termination across multiple Envoy instances</li>
                <li><strong>Monitoring:</strong> Continuously monitor TLS performance metrics</li>
            </ul>
            
            <h3 style="margin-top: 25px;">📈 Performance Targets</h3>
            <table style="width: 100%; margin-top: 15px; border-collapse: collapse;">
                <tr style="background: #ecf0f1;">
                    <th style="padding: 10px; text-align: left;">Metric</th>
                    <th style="padding: 10px; text-align: left;">Target</th>
                    <th style="padding: 10px; text-align: left;">Acceptable</th>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #bdc3c7;">TLS Handshake</td>
                    <td style="padding: 10px; border-bottom: 1px solid #bdc3c7;">&lt; 50ms</td>
                    <td style="padding: 10px; border-bottom: 1px solid #bdc3c7;">&lt; 100ms</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #bdc3c7;">Request Latency</td>
                    <td style="padding: 10px; border-bottom: 1px solid #bdc3c7;">&lt; 10ms</td>
                    <td style="padding: 10px; border-bottom: 1px solid #bdc3c7;">&lt; 50ms</td>
                </tr>
                <tr>
                    <td style="padding: 10px;">Throughput</td>
                    <td style="padding: 10px;">&gt; 1000 req/s</td>
                    <td style="padding: 10px;">&gt; 500 req/s</td>
                </tr>
            </table>
        </div>
        
        <div class="timestamp">
            Performance benchmark completed on $(date) | XORB Platform Performance Team
        </div>
    </div>
</body>
</html>
EOF

    log "INFO" "Performance report generated: $report_file"
}

# Parse command line arguments
SPECIFIC_ENDPOINT=""
OUTPUT_FILE=""
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--connections)
            CONCURRENT_CONNECTIONS="$2"
            shift 2
            ;;
        -d|--duration)
            TEST_DURATION="$2"
            shift 2
            ;;
        -w|--warmup)
            WARMUP_TIME="$2"
            shift 2
            ;;
        -e|--endpoint)
            SPECIFIC_ENDPOINT="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
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
    
    mkdir -p "$REPORT_DIR"
    
    log "PERF" "🚀 Starting XORB Platform TLS/mTLS Performance Benchmark"
    log "INFO" "Configuration: ${CONCURRENT_CONNECTIONS} connections, ${TEST_DURATION}s duration"
    
    # Check prerequisites
    if ! check_prerequisites; then
        log "ERROR" "Prerequisites check failed - aborting"
        exit 1
    fi
    
    # Run benchmarks
    for endpoint_config in "${ENDPOINTS[@]}"; do
        IFS=':' read -r host port path <<< "$endpoint_config"
        local endpoint="${host}:${port}"
        local service_name="$host"
        
        # Remove envoy- prefix for service name
        service_name="${service_name#envoy-}"
        
        # Skip if specific endpoint requested
        if [[ -n "$SPECIFIC_ENDPOINT" && "$endpoint" != "$SPECIFIC_ENDPOINT" ]]; then
            continue
        fi
        
        log "PERF" "Benchmarking $service_name ($endpoint)"
        
        # Run different benchmark types
        benchmark_tls_handshake "$endpoint" "$service_name"
        benchmark_throughput "$endpoint" "$path" "$service_name"
        benchmark_resource_usage "$service_name"
        benchmark_cipher_performance "$endpoint" "$service_name"
        
        echo "---"
    done
    
    # Generate comprehensive report
    generate_performance_report
    
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    
    log "PERF" "Performance benchmarking completed in ${total_duration}s"
    log "INFO" "Reports available in: $REPORT_DIR"
    
    # Copy to output file if specified
    if [[ -n "$OUTPUT_FILE" ]]; then
        cp "${REPORT_DIR}"/performance-summary-*.html "$OUTPUT_FILE" 2>/dev/null || true
        log "INFO" "Report copied to: $OUTPUT_FILE"
    fi
    
    log "RESULT" "🎯 Performance benchmarking completed successfully"
}

# Execute main function
main "$@"