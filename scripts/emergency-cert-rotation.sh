#!/bin/bash
# Emergency Certificate Rotation Script for XORB Platform
# Immediate certificate rotation for security incidents and compromises

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CA_SCRIPT_DIR="${SCRIPT_DIR}/ca"
SECRETS_DIR="${SCRIPT_DIR}/../secrets/tls"
LOG_DIR="${SCRIPT_DIR}/../logs/emergency"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Emergency response configuration
INCIDENT_ID=""
AFFECTED_SERVICES=()
COMPROMISE_TYPE=""
EMERGENCY_CONTACT=""
AUTO_RESTART=true
SKIP_VALIDATION=false

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
        "CRITICAL")
            echo -e "${RED}[CRITICAL]${NC} ${timestamp} - $message"
            ;;
        "EMERGENCY")
            echo -e "${PURPLE}[EMERGENCY]${NC} ${timestamp} - $message"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - $message"
            ;;
    esac
    
    # Also log to syslog for audit trail
    logger -t "xorb-emergency-rotation" -p local0.info "[$level] $message"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Emergency certificate rotation for security incidents.

OPTIONS:
    -i, --incident-id ID     Incident tracking ID
    -s, --service NAME       Specific service to rotate (can be repeated)
    -t, --type TYPE          Compromise type (key_leak, cert_compromise, ca_breach)
    -c, --contact EMAIL      Emergency contact for notifications
    -n, --no-restart         Skip automatic service restart
    -f, --force              Skip validation checks
    -h, --help               Show this help message

EXAMPLES:
    $0 -i INC-2024-001 -s api -t key_leak
    $0 -i INC-2024-002 -t ca_breach -c security@xorb.local
    $0 -s redis -s postgres -t cert_compromise
EOF
}

setup_emergency_logging() {
    mkdir -p "$LOG_DIR"
    
    local log_file="${LOG_DIR}/emergency-$(date +%Y%m%d-%H%M%S)-${INCIDENT_ID:-unknown}.log"
    exec > >(tee -a "$log_file")
    exec 2>&1
    
    log "EMERGENCY" "Emergency certificate rotation initiated"
    log "INFO" "Incident ID: ${INCIDENT_ID:-Not specified}"
    log "INFO" "Compromise Type: ${COMPROMISE_TYPE:-Not specified}"
    log "INFO" "Log file: $log_file"
}

validate_emergency_response() {
    if [[ "$SKIP_VALIDATION" == "true" ]]; then
        log "WARN" "Validation checks skipped due to emergency mode"
        return 0
    fi
    
    log "INFO" "Validating emergency response prerequisites"
    
    # Check if CA is available
    if [[ ! -f "${SECRETS_DIR}/ca/ca.pem" ]]; then
        log "CRITICAL" "CA certificate not found - cannot issue new certificates"
        return 1
    fi
    
    # Check if CA private key is accessible
    if [[ ! -f "${SECRETS_DIR}/ca/intermediate/private/intermediate.key.pem" ]]; then
        log "CRITICAL" "CA private key not accessible - certificate issuance will fail"
        return 1
    fi
    
    # Verify CA scripts are available
    if [[ ! -x "${CA_SCRIPT_DIR}/issue-cert.sh" ]]; then
        log "CRITICAL" "Certificate issuance script not found or not executable"
        return 1
    fi
    
    # Check Docker daemon availability
    if ! docker info >/dev/null 2>&1; then
        log "CRITICAL" "Docker daemon not available - service restart will fail"
        if [[ "$AUTO_RESTART" == "true" ]]; then
            log "WARN" "Disabling automatic restart due to Docker unavailability"
            AUTO_RESTART=false
        fi
    fi
    
    log "SUCCESS" "Emergency response validation completed"
    return 0
}

revoke_compromised_certificate() {
    local service_name="$1"
    local cert_file="${SECRETS_DIR}/${service_name}/cert.pem"
    
    if [[ ! -f "$cert_file" ]]; then
        log "WARN" "Certificate file not found for $service_name: $cert_file"
        return 1
    fi
    
    log "CRITICAL" "Revoking compromised certificate for $service_name"
    
    # Add certificate to CRL
    local ca_config="${SECRETS_DIR}/ca/intermediate/openssl.cnf"
    
    if openssl ca -config "$ca_config" \
                  -revoke "$cert_file" \
                  -passin pass:xorb-intermediate-ca-key \
                  >/dev/null 2>&1; then
        log "SUCCESS" "Certificate revoked for $service_name"
    else
        log "ERROR" "Failed to revoke certificate for $service_name"
        return 1
    fi
    
    # Generate updated CRL
    local crl_file="${SECRETS_DIR}/ca/intermediate/crl/intermediate.crl.pem"
    
    if openssl ca -config "$ca_config" \
                  -gencrl \
                  -passin pass:xorb-intermediate-ca-key \
                  -out "$crl_file" \
                  >/dev/null 2>&1; then
        log "SUCCESS" "Updated CRL generated"
    else
        log "ERROR" "Failed to generate updated CRL"
        return 1
    fi
    
    return 0
}

isolate_compromised_service() {
    local service_name="$1"
    
    log "CRITICAL" "Isolating compromised service: $service_name"
    
    # Block service in Envoy (if applicable)
    if [[ "$service_name" == "api" || "$service_name" == "agent" ]]; then
        local admin_port
        case "$service_name" in
            "api") admin_port="9901" ;;
            "agent") admin_port="9902" ;;
        esac
        
        # Add service to blocked list via Envoy admin API
        if curl -s -X POST "http://localhost:${admin_port}/runtime_modify" \
                -d "blocked_services=${service_name}" >/dev/null 2>&1; then
            log "SUCCESS" "Service $service_name blocked in Envoy"
        else
            log "WARN" "Failed to block service $service_name in Envoy"
        fi
    fi
    
    # Stop service container (if running)
    if docker-compose ps "$service_name" >/dev/null 2>&1; then
        if docker-compose stop "$service_name" >/dev/null 2>&1; then
            log "SUCCESS" "Service $service_name stopped"
        else
            log "ERROR" "Failed to stop service $service_name"
        fi
    fi
    
    return 0
}

emergency_certificate_generation() {
    local service_name="$1"
    
    log "EMERGENCY" "Generating emergency certificate for $service_name"
    
    # Backup existing certificate
    local backup_dir="${SECRETS_DIR}/emergency-backup/${service_name}/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    
    if [[ -d "${SECRETS_DIR}/${service_name}" ]]; then
        cp -r "${SECRETS_DIR}/${service_name}"/* "$backup_dir/" 2>/dev/null || true
        log "INFO" "Existing certificates backed up to $backup_dir"
    fi
    
    # Determine certificate type
    local cert_type="server"
    case "$service_name" in
        *-client|orchestrator|scanner)
            cert_type="client"
            ;;
        api|agent)
            cert_type="both"
            ;;
    esac
    
    # Generate new certificate with emergency flag
    log "INFO" "Issuing new $cert_type certificate for $service_name"
    
    if "${CA_SCRIPT_DIR}/issue-cert.sh" "$service_name" "$cert_type" -d 7; then
        log "SUCCESS" "Emergency certificate generated for $service_name"
        
        # Set emergency metadata
        echo "$(date -Iseconds)" > "${SECRETS_DIR}/${service_name}/emergency_issued"
        echo "$INCIDENT_ID" > "${SECRETS_DIR}/${service_name}/incident_id"
        echo "$COMPROMISE_TYPE" > "${SECRETS_DIR}/${service_name}/compromise_type"
        
        return 0
    else
        log "CRITICAL" "Failed to generate emergency certificate for $service_name"
        
        # Restore backup if generation failed
        if [[ -d "$backup_dir" && -n "$(ls -A "$backup_dir")" ]]; then
            cp -r "$backup_dir"/* "${SECRETS_DIR}/${service_name}/" 2>/dev/null || true
            log "WARN" "Restored backup certificates for $service_name"
        fi
        
        return 1
    fi
}

restart_service_with_new_certificate() {
    local service_name="$1"
    
    if [[ "$AUTO_RESTART" != "true" ]]; then
        log "INFO" "Skipping automatic restart for $service_name (disabled)"
        return 0
    fi
    
    log "INFO" "Restarting $service_name with new certificate"
    
    # Unblock service in Envoy first (if applicable)
    if [[ "$service_name" == "api" || "$service_name" == "agent" ]]; then
        local admin_port
        case "$service_name" in
            "api") admin_port="9901" ;;
            "agent") admin_port="9902" ;;
        esac
        
        curl -s -X POST "http://localhost:${admin_port}/runtime_modify" \
             -d "blocked_services=" >/dev/null 2>&1 || true
    fi
    
    # Restart service
    if docker-compose restart "$service_name" >/dev/null 2>&1; then
        log "SUCCESS" "Service $service_name restarted"
        
        # Wait for service to be healthy
        local retries=0
        while [[ $retries -lt 30 ]]; do
            if docker-compose ps "$service_name" | grep -q "healthy\|Up"; then
                log "SUCCESS" "Service $service_name is healthy"
                return 0
            fi
            sleep 2
            retries=$((retries + 1))
        done
        
        log "ERROR" "Service $service_name failed health check after restart"
        return 1
    else
        log "ERROR" "Failed to restart service $service_name"
        return 1
    fi
}

validate_emergency_certificates() {
    local service_name="$1"
    
    log "INFO" "Validating emergency certificate for $service_name"
    
    local cert_file="${SECRETS_DIR}/${service_name}/cert.pem"
    local ca_file="${SECRETS_DIR}/ca/ca.pem"
    
    # Basic certificate validation
    if ! openssl x509 -in "$cert_file" -noout -text >/dev/null 2>&1; then
        log "ERROR" "Invalid certificate format for $service_name"
        return 1
    fi
    
    # Verify certificate chain
    if ! openssl verify -CAfile "$ca_file" "$cert_file" >/dev/null 2>&1; then
        log "ERROR" "Certificate chain verification failed for $service_name"
        return 1
    fi
    
    # Check if certificate is not expired
    if ! openssl x509 -in "$cert_file" -checkend 0 -noout >/dev/null 2>&1; then
        log "ERROR" "Certificate is expired for $service_name"
        return 1
    fi
    
    # Verify certificate was issued recently (emergency context)
    local cert_date=$(openssl x509 -in "$cert_file" -noout -startdate | cut -d= -f2)
    local cert_timestamp=$(date -d "$cert_date" +%s)
    local current_timestamp=$(date +%s)
    local age_minutes=$(( (current_timestamp - cert_timestamp) / 60 ))
    
    if [[ $age_minutes -gt 60 ]]; then
        log "WARN" "Certificate for $service_name is older than expected for emergency rotation"
    fi
    
    log "SUCCESS" "Emergency certificate validation passed for $service_name"
    return 0
}

send_emergency_notifications() {
    local summary="$1"
    
    log "INFO" "Sending emergency notifications"
    
    # Create incident summary
    local notification_body=$(cat << EOF
🚨 XORB Platform Security Incident - Emergency Certificate Rotation

Incident ID: ${INCIDENT_ID:-Unknown}
Compromise Type: ${COMPROMISE_TYPE:-Unknown}
Affected Services: ${AFFECTED_SERVICES[*]:-All services}
Timestamp: $(date -Iseconds)
Hostname: $(hostname)

Actions Taken:
$summary

Status: Certificate rotation completed
Next Steps: Validate service functionality and conduct incident analysis

Emergency Contact: ${EMERGENCY_CONTACT:-Not specified}
EOF
)
    
    # Send Slack notification (critical channel)
    if [[ -n "${SLACK_EMERGENCY_WEBHOOK:-}" ]]; then
        curl -s -X POST "$SLACK_EMERGENCY_WEBHOOK" \
             -H "Content-Type: application/json" \
             -d "{
                \"channel\": \"#security-critical\",
                \"username\": \"XORB Security Bot\",
                \"icon_emoji\": \":rotating_light:\",
                \"attachments\": [{
                    \"color\": \"danger\",
                    \"title\": \"🚨 Emergency Certificate Rotation Completed\",
                    \"text\": \"$(echo "$notification_body" | sed 's/"/\\"/g' | tr '\n' ' ')\",
                    \"ts\": $(date +%s)
                }]
             }" >/dev/null 2>&1
        
        log "SUCCESS" "Emergency Slack notification sent"
    fi
    
    # Send email to emergency contact
    if [[ -n "$EMERGENCY_CONTACT" ]] && command -v mail >/dev/null 2>&1; then
        echo "$notification_body" | mail -s "🚨 XORB Security Incident - Emergency Certificate Rotation" "$EMERGENCY_CONTACT"
        log "SUCCESS" "Emergency email notification sent to $EMERGENCY_CONTACT"
    fi
    
    # Log to security audit log
    echo "$notification_body" >> "${LOG_DIR}/security-incidents.log"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--incident-id)
            INCIDENT_ID="$2"
            shift 2
            ;;
        -s|--service)
            AFFECTED_SERVICES+=("$2")
            shift 2
            ;;
        -t|--type)
            COMPROMISE_TYPE="$2"
            shift 2
            ;;
        -c|--contact)
            EMERGENCY_CONTACT="$2"
            shift 2
            ;;
        -n|--no-restart)
            AUTO_RESTART=false
            shift
            ;;
        -f|--force)
            SKIP_VALIDATION=true
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

# Validate required parameters
if [[ -z "$INCIDENT_ID" ]]; then
    log "WARN" "No incident ID provided - generating automatic ID"
    INCIDENT_ID="EMERGENCY-$(date +%Y%m%d-%H%M%S)"
fi

if [[ ${#AFFECTED_SERVICES[@]} -eq 0 ]]; then
    log "WARN" "No specific services provided - rotating all service certificates"
    AFFECTED_SERVICES=(
        "api" "orchestrator" "agent" "redis" "postgres" "temporal" "dind"
        "redis-client" "postgres-client" "temporal-client" "dind-client" "scanner"
    )
fi

# Main emergency response execution
main() {
    local start_time=$(date +%s)
    
    setup_emergency_logging
    
    log "EMERGENCY" "🚨 XORB Platform Emergency Certificate Rotation Started"
    log "CRITICAL" "Incident: ${INCIDENT_ID} | Type: ${COMPROMISE_TYPE:-Unknown}"
    log "CRITICAL" "Affected Services: ${AFFECTED_SERVICES[*]}"
    
    # Validate emergency response capability
    if ! validate_emergency_response; then
        log "CRITICAL" "Emergency response validation failed - aborting"
        exit 1
    fi
    
    local successful_rotations=0
    local failed_rotations=0
    local actions_summary=""
    
    # Process each affected service
    for service in "${AFFECTED_SERVICES[@]}"; do
        log "EMERGENCY" "Processing emergency rotation for $service"
        
        local service_success=true
        
        # Step 1: Revoke compromised certificate
        if ! revoke_compromised_certificate "$service"; then
            log "ERROR" "Failed to revoke certificate for $service"
            service_success=false
        fi
        
        # Step 2: Isolate service
        if ! isolate_compromised_service "$service"; then
            log "ERROR" "Failed to isolate service $service"
            service_success=false
        fi
        
        # Step 3: Generate new certificate
        if ! emergency_certificate_generation "$service"; then
            log "CRITICAL" "Failed to generate new certificate for $service"
            service_success=false
        fi
        
        # Step 4: Restart service
        if [[ "$service_success" == "true" ]]; then
            if ! restart_service_with_new_certificate "$service"; then
                log "ERROR" "Failed to restart service $service"
                service_success=false
            fi
        fi
        
        # Step 5: Validate new certificate
        if [[ "$service_success" == "true" ]]; then
            if ! validate_emergency_certificates "$service"; then
                log "ERROR" "Certificate validation failed for $service"
                service_success=false
            fi
        fi
        
        # Update counters and summary
        if [[ "$service_success" == "true" ]]; then
            successful_rotations=$((successful_rotations + 1))
            actions_summary+="✅ $service: Certificate rotated and service restored\n"
            log "SUCCESS" "Emergency rotation completed successfully for $service"
        else
            failed_rotations=$((failed_rotations + 1))
            actions_summary+="❌ $service: Emergency rotation failed\n"
            log "CRITICAL" "Emergency rotation failed for $service"
        fi
        
        echo "---"
    done
    
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    
    # Generate final summary
    local final_summary=$(cat << EOF
Emergency Certificate Rotation Summary:
- Total Services: ${#AFFECTED_SERVICES[@]}
- Successful: $successful_rotations
- Failed: $failed_rotations
- Duration: ${total_duration}s

Service Actions:
$(echo -e "$actions_summary")
EOF
)
    
    log "EMERGENCY" "Emergency response completed in ${total_duration}s"
    log "INFO" "$final_summary"
    
    # Send notifications
    send_emergency_notifications "$final_summary"
    
    # Final status
    if [[ $failed_rotations -eq 0 ]]; then
        log "SUCCESS" "🎉 Emergency certificate rotation completed successfully for all services"
        exit 0
    else
        log "CRITICAL" "💥 Emergency certificate rotation failed for $failed_rotations service(s)"
        log "CRITICAL" "Manual intervention required for failed services"
        exit 1
    fi
}

# Execute main function
main "$@"