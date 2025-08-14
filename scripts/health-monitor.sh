#!/bin/bash
# TLS Health Monitoring Script for XORB Platform
# Continuous monitoring of certificate health and TLS connectivity

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRETS_DIR="${SCRIPT_DIR}/../secrets/tls"
LOG_DIR="${SCRIPT_DIR}/../logs/monitoring"
CONFIG_FILE="${SCRIPT_DIR}/../config/health-monitor.conf"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
CHECK_INTERVAL=300  # 5 minutes
CERT_WARNING_DAYS=7
CERT_CRITICAL_DAYS=2
ENABLE_SLACK_ALERTS=false
ENABLE_EMAIL_ALERTS=false
SLACK_WEBHOOK_URL=""
EMAIL_RECIPIENTS=""
SYSLOG_FACILITY="local0"

# Services to monitor
SERVICES=(
    "envoy-api:8443"
    "envoy-agent:8444"
    "redis:6380"
    "postgres:5433"
    "temporal:7234"
    "prometheus:9093"
    "grafana:3001"
)

log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message"
            logger -t "xorb-health-monitor" -p "${SYSLOG_FACILITY}.info" "INFO: $message"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message"
            logger -t "xorb-health-monitor" -p "${SYSLOG_FACILITY}.warning" "WARN: $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message"
            logger -t "xorb-health-monitor" -p "${SYSLOG_FACILITY}.err" "ERROR: $message"
            ;;
        "CRITICAL")
            echo -e "${RED}[CRITICAL]${NC} ${timestamp} - $message"
            logger -t "xorb-health-monitor" -p "${SYSLOG_FACILITY}.crit" "CRITICAL: $message"
            ;;
        "DEBUG")
            if [[ "${VERBOSE:-false}" == "true" ]]; then
                echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message"
            fi
            ;;
    esac
}

load_config() {
    if [[ -f "$CONFIG_FILE" ]]; then
        source "$CONFIG_FILE"
        log "INFO" "Configuration loaded from $CONFIG_FILE"
    else
        log "WARN" "No configuration file found, using defaults"
    fi
}

create_default_config() {
    mkdir -p "$(dirname "$CONFIG_FILE")"
    
    cat > "$CONFIG_FILE" << 'EOF'
# XORB Platform Health Monitor Configuration

# Monitoring interval in seconds
CHECK_INTERVAL=300

# Certificate expiry warning thresholds (days)
CERT_WARNING_DAYS=7
CERT_CRITICAL_DAYS=2

# Alert configuration
ENABLE_SLACK_ALERTS=false
ENABLE_EMAIL_ALERTS=false
SLACK_WEBHOOK_URL=""
EMAIL_RECIPIENTS=""

# Syslog configuration
SYSLOG_FACILITY="local0"

# Service endpoints to monitor (format: "service:port")
SERVICES=(
    "envoy-api:8443"
    "envoy-agent:8444"
    "redis:6380"
    "postgres:5433"
    "temporal:7234"
)

# Custom health check commands (optional)
# CUSTOM_HEALTH_CHECKS=(
#     "redis:redis-cli --tls --cert /path/to/cert ping"
# )
EOF

    log "INFO" "Default configuration created at $CONFIG_FILE"
}

check_certificate_expiry() {
    local service_name="$1"
    local cert_file="${SECRETS_DIR}/${service_name}/cert.pem"
    
    if [[ ! -f "$cert_file" ]]; then
        log "ERROR" "Certificate file not found for $service_name: $cert_file"
        return 2
    fi
    
    # Get certificate expiry date
    local expire_date=$(openssl x509 -in "$cert_file" -noout -enddate | cut -d= -f2)
    local expire_timestamp=$(date -d "$expire_date" +%s)
    local current_timestamp=$(date +%s)
    local days_remaining=$(( (expire_timestamp - current_timestamp) / 86400 ))
    
    log "DEBUG" "$service_name certificate expires in $days_remaining days"
    
    # Check expiry status
    if [[ $days_remaining -le $CERT_CRITICAL_DAYS ]]; then
        log "CRITICAL" "$service_name certificate expires in $days_remaining days - CRITICAL"
        send_alert "CRITICAL" "$service_name certificate expires in $days_remaining days"
        return 1
    elif [[ $days_remaining -le $CERT_WARNING_DAYS ]]; then
        log "WARN" "$service_name certificate expires in $days_remaining days - WARNING"
        send_alert "WARNING" "$service_name certificate expires in $days_remaining days"
        return 1
    else
        log "DEBUG" "$service_name certificate valid for $days_remaining days"
        return 0
    fi
}

check_tls_connectivity() {
    local service_endpoint="$1"
    IFS=':' read -r service_name port <<< "$service_endpoint"
    
    log "DEBUG" "Checking TLS connectivity for $service_name:$port"
    
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    local client_cert="${SECRETS_DIR}/${service_name}-client/cert.pem"
    local client_key="${SECRETS_DIR}/${service_name}-client/key.pem"
    
    # Use fallback client cert if service-specific not found
    if [[ ! -f "$client_cert" ]]; then
        client_cert="${SECRETS_DIR}/api-client/cert.pem"
        client_key="${SECRETS_DIR}/api-client/key.pem"
    fi
    
    # Test TLS connection
    if timeout 10 openssl s_client \
        -connect "${service_name}:${port}" \
        -CAfile "$ca_file" \
        -cert "$client_cert" \
        -key "$client_key" \
        -verify_return_error \
        -quiet \
        </dev/null >/dev/null 2>&1; then
        
        log "DEBUG" "TLS connectivity OK for $service_name:$port"
        return 0
    else
        log "ERROR" "TLS connectivity FAILED for $service_name:$port"
        send_alert "ERROR" "TLS connectivity failed for $service_name:$port"
        return 1
    fi
}

check_service_health() {
    local service_endpoint="$1"
    IFS=':' read -r service_name port <<< "$service_endpoint"
    
    log "DEBUG" "Checking service health for $service_name"
    
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    local client_cert="${SECRETS_DIR}/${service_name}-client/cert.pem"
    local client_key="${SECRETS_DIR}/${service_name}-client/key.pem"
    
    # Use fallback client cert if service-specific not found
    if [[ ! -f "$client_cert" ]]; then
        client_cert="${SECRETS_DIR}/api-client/cert.pem"
        client_key="${SECRETS_DIR}/api-client/key.pem"
    fi
    
    # Service-specific health checks
    case "$service_name" in
        "envoy-api"|"envoy-agent")
            # Check HTTP health endpoint
            local health_path="/health"
            if [[ "$service_name" == "envoy-api" ]]; then
                health_path="/api/v1/health"
            fi
            
            if timeout 10 curl -s -f \
                --cacert "$ca_file" \
                --cert "$client_cert" \
                --key "$client_key" \
                "https://${service_name}:${port}${health_path}" >/dev/null 2>&1; then
                
                log "DEBUG" "Service health OK for $service_name"
                return 0
            else
                log "ERROR" "Service health FAILED for $service_name"
                send_alert "ERROR" "Service health check failed for $service_name"
                return 1
            fi
            ;;
        "redis")
            # Check Redis ping
            if timeout 10 redis-cli \
                --tls \
                --cert "$client_cert" \
                --key "$client_key" \
                --cacert "$ca_file" \
                -h "$service_name" \
                -p "$port" \
                ping >/dev/null 2>&1; then
                
                log "DEBUG" "Redis health OK"
                return 0
            else
                log "ERROR" "Redis health FAILED"
                send_alert "ERROR" "Redis health check failed"
                return 1
            fi
            ;;
        "postgres")
            # Check PostgreSQL connection
            if timeout 10 pg_isready \
                -h "$service_name" \
                -p "$port" \
                -U xorb >/dev/null 2>&1; then
                
                log "DEBUG" "PostgreSQL health OK"
                return 0
            else
                log "ERROR" "PostgreSQL health FAILED"
                send_alert "ERROR" "PostgreSQL health check failed"
                return 1
            fi
            ;;
        *)
            # Generic TLS connectivity check
            return $(check_tls_connectivity "$service_endpoint")
            ;;
    esac
}

send_alert() {
    local severity="$1"
    local message="$2"
    local timestamp=$(date -Iseconds)
    
    # Slack alerts
    if [[ "$ENABLE_SLACK_ALERTS" == "true" && -n "$SLACK_WEBHOOK_URL" ]]; then
        local color="warning"
        local emoji="⚠️"
        
        case "$severity" in
            "CRITICAL")
                color="danger"
                emoji="🚨"
                ;;
            "ERROR")
                color="danger"
                emoji="❌"
                ;;
            "WARNING")
                color="warning"
                emoji="⚠️"
                ;;
        esac
        
        curl -s -X POST "$SLACK_WEBHOOK_URL" \
             -H "Content-Type: application/json" \
             -d "{
                \"channel\": \"#xorb-alerts\",
                \"username\": \"XORB Health Monitor\",
                \"icon_emoji\": \":shield:\",
                \"attachments\": [{
                    \"color\": \"$color\",
                    \"title\": \"$emoji XORB Platform Alert - $severity\",
                    \"text\": \"$message\",
                    \"fields\": [
                        {\"title\": \"Severity\", \"value\": \"$severity\", \"short\": true},
                        {\"title\": \"Timestamp\", \"value\": \"$timestamp\", \"short\": true},
                        {\"title\": \"Host\", \"value\": \"$(hostname)\", \"short\": true}
                    ],
                    \"ts\": $(date +%s)
                }]
             }" >/dev/null 2>&1
        
        log "DEBUG" "Slack alert sent: $severity - $message"
    fi
    
    # Email alerts
    if [[ "$ENABLE_EMAIL_ALERTS" == "true" && -n "$EMAIL_RECIPIENTS" ]] && command -v mail >/dev/null 2>&1; then
        local subject="XORB Platform Alert - $severity"
        local body="XORB Platform Health Monitor Alert

Severity: $severity
Message: $message
Timestamp: $timestamp
Host: $(hostname)

This is an automated alert from the XORB Platform health monitoring system.
Please investigate the issue immediately.

---
XORB Platform Security Team"
        
        echo "$body" | mail -s "$subject" "$EMAIL_RECIPIENTS"
        log "DEBUG" "Email alert sent to $EMAIL_RECIPIENTS"
    fi
}

generate_health_report() {
    local report_file="${LOG_DIR}/health-report-$(date +%Y%m%d-%H%M%S).json"
    local timestamp=$(date -Iseconds)
    
    log "INFO" "Generating health report: $report_file"
    
    # Collect system information
    local hostname=$(hostname)
    local uptime=$(uptime | awk '{print $3,$4}' | sed 's/,//')
    local load_avg=$(uptime | awk -F'load average:' '{ print $2 }')
    local disk_usage=$(df -h / | awk 'NR==2 {print $5}')
    local memory_usage=$(free | awk 'NR==2{printf "%.2f%%", $3*100/$2}')
    
    # Start JSON report
    cat > "$report_file" << EOF
{
  "report_type": "health_monitoring",
  "timestamp": "$timestamp",
  "system_info": {
    "hostname": "$hostname",
    "uptime": "$uptime",
    "load_average": "$load_avg",
    "disk_usage": "$disk_usage",
    "memory_usage": "$memory_usage"
  },
  "certificate_status": [
EOF

    # Certificate status
    local cert_count=0
    local cert_warnings=0
    local cert_criticals=0
    
    for service_dir in "$SECRETS_DIR"/*; do
        if [[ -d "$service_dir" && -f "$service_dir/cert.pem" ]]; then
            local service_name=$(basename "$service_dir")
            
            if [[ "$service_name" == "ca" || "$service_name" == "backups" ]]; then
                continue
            fi
            
            local cert_file="$service_dir/cert.pem"
            local expire_date=$(openssl x509 -in "$cert_file" -noout -enddate | cut -d= -f2)
            local expire_timestamp=$(date -d "$expire_date" +%s)
            local current_timestamp=$(date +%s)
            local days_remaining=$(( (expire_timestamp - current_timestamp) / 86400 ))
            
            local status="OK"
            if [[ $days_remaining -le $CERT_CRITICAL_DAYS ]]; then
                status="CRITICAL"
                cert_criticals=$((cert_criticals + 1))
            elif [[ $days_remaining -le $CERT_WARNING_DAYS ]]; then
                status="WARNING"
                cert_warnings=$((cert_warnings + 1))
            fi
            
            if [[ $cert_count -gt 0 ]]; then
                echo "," >> "$report_file"
            fi
            
            cat >> "$report_file" << EOF
    {
      "service": "$service_name",
      "certificate_file": "$cert_file",
      "expiry_date": "$expire_date",
      "days_remaining": $days_remaining,
      "status": "$status"
    }
EOF
            cert_count=$((cert_count + 1))
        fi
    done
    
    # Service connectivity status
    cat >> "$report_file" << EOF
  ],
  "service_connectivity": [
EOF

    local service_count=0
    local service_failures=0
    
    for service_endpoint in "${SERVICES[@]}"; do
        IFS=':' read -r service_name port <<< "$service_endpoint"
        
        if [[ $service_count -gt 0 ]]; then
            echo "," >> "$report_file"
        fi
        
        # Test connectivity
        local connectivity_status="OK"
        local health_status="OK"
        
        if ! check_tls_connectivity "$service_endpoint" >/dev/null 2>&1; then
            connectivity_status="FAILED"
            service_failures=$((service_failures + 1))
        fi
        
        if ! check_service_health "$service_endpoint" >/dev/null 2>&1; then
            health_status="FAILED"
            if [[ "$connectivity_status" == "OK" ]]; then
                service_failures=$((service_failures + 1))
            fi
        fi
        
        cat >> "$report_file" << EOF
    {
      "service": "$service_name",
      "endpoint": "$service_endpoint",
      "tls_connectivity": "$connectivity_status",
      "health_status": "$health_status"
    }
EOF
        service_count=$((service_count + 1))
    done
    
    # Summary
    cat >> "$report_file" << EOF
  ],
  "summary": {
    "total_certificates": $cert_count,
    "certificate_warnings": $cert_warnings,
    "certificate_criticals": $cert_criticals,
    "total_services": $service_count,
    "service_failures": $service_failures,
    "overall_status": "$( [[ $cert_criticals -eq 0 && $service_failures -eq 0 ]] && echo "HEALTHY" || echo "UNHEALTHY" )"
  }
}
EOF

    log "INFO" "Health report generated: $report_file"
}

run_health_check() {
    log "INFO" "Running health monitoring cycle"
    
    local check_failures=0
    local total_checks=0
    
    # Check certificate expiry
    for service_dir in "$SECRETS_DIR"/*; do
        if [[ -d "$service_dir" && -f "$service_dir/cert.pem" ]]; then
            local service_name=$(basename "$service_dir")
            
            if [[ "$service_name" == "ca" || "$service_name" == "backups" ]]; then
                continue
            fi
            
            total_checks=$((total_checks + 1))
            if ! check_certificate_expiry "$service_name"; then
                check_failures=$((check_failures + 1))
            fi
        fi
    done
    
    # Check service connectivity and health
    for service_endpoint in "${SERVICES[@]}"; do
        total_checks=$((total_checks + 1))
        if ! check_service_health "$service_endpoint"; then
            check_failures=$((check_failures + 1))
        fi
    done
    
    # Log summary
    if [[ $check_failures -eq 0 ]]; then
        log "INFO" "Health check completed: $total_checks checks passed"
    else
        log "WARN" "Health check completed: $check_failures/$total_checks checks failed"
    fi
    
    return $check_failures
}

daemon_mode() {
    log "INFO" "Starting health monitor daemon (interval: ${CHECK_INTERVAL}s)"
    
    # Create PID file
    local pid_file="${LOG_DIR}/health-monitor.pid"
    echo $$ > "$pid_file"
    
    # Trap signals for graceful shutdown
    trap 'log "INFO" "Health monitor daemon stopping"; rm -f "$pid_file"; exit 0' TERM INT
    
    while true; do
        local cycle_start=$(date +%s)
        
        # Run health checks
        if ! run_health_check; then
            log "WARN" "Health check cycle completed with failures"
        fi
        
        # Generate periodic report (every hour)
        local current_minute=$(date +%M)
        if [[ "$current_minute" == "00" ]]; then
            generate_health_report
        fi
        
        local cycle_end=$(date +%s)
        local cycle_duration=$((cycle_end - cycle_start))
        
        log "DEBUG" "Health check cycle completed in ${cycle_duration}s"
        
        # Sleep until next check
        sleep $CHECK_INTERVAL
    done
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS] [COMMAND]

Health monitoring for XORB Platform TLS/mTLS infrastructure.

COMMANDS:
    check           Run single health check cycle
    daemon          Run continuous monitoring daemon
    report          Generate health report
    config          Create default configuration file

OPTIONS:
    -c, --config FILE   Configuration file path
    -i, --interval SEC  Check interval for daemon mode (default: 300)
    -v, --verbose       Verbose output
    -h, --help          Show this help message

EXAMPLES:
    $0 check                    # Run single health check
    $0 daemon                   # Start monitoring daemon
    $0 -i 60 daemon            # Start daemon with 1-minute intervals
    $0 report                   # Generate health report
EOF
}

# Parse command line arguments
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -i|--interval)
            CHECK_INTERVAL="$2"
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
        check|daemon|report|config)
            COMMAND="$1"
            shift
            ;;
        *)
            echo "Error: Unknown option or command $1" >&2
            usage
            exit 1
            ;;
    esac
done

# Default command
if [[ -z "$COMMAND" ]]; then
    COMMAND="check"
fi

# Main execution
main() {
    mkdir -p "$LOG_DIR"
    
    # Load configuration
    load_config
    
    case "$COMMAND" in
        "check")
            run_health_check
            ;;
        "daemon")
            daemon_mode
            ;;
        "report")
            generate_health_report
            ;;
        "config")
            create_default_config
            ;;
        *)
            echo "Error: Unknown command $COMMAND" >&2
            usage
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"