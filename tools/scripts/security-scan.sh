#!/bin/bash
set -e

# XORB Security Scanning Script
# Comprehensive security scanning for development and CI/CD

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check if running in CI
is_ci() {
    [[ -n "${CI:-}" ]] || [[ -n "${GITHUB_ACTIONS:-}" ]]
}

# Install security tools
install_security_tools() {
    log "Installing security scanning tools..."

    # Python security tools
    pip install --quiet bandit safety semgrep

    # Trivy for container scanning
    if ! command -v trivy &> /dev/null; then
        info "Installing Trivy..."
        curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
    fi

    # Grype for vulnerability scanning
    if ! command -v grype &> /dev/null; then
        info "Installing Grype..."
        curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin
    fi

    # Gitleaks for secret scanning
    if ! command -v gitleaks &> /dev/null; then
        info "Installing Gitleaks..."
        git clone https://github.com/gitleaks/gitleaks.git /tmp/gitleaks
        cd /tmp/gitleaks && make build && sudo mv gitleaks /usr/local/bin/
        cd "$PROJECT_ROOT"
        rm -rf /tmp/gitleaks
    fi

    log "Security tools installation completed ✅"
}

# Secret scanning with Gitleaks
run_secret_scan() {
    log "Running secret scanning with Gitleaks..."

    local output_dir="$PROJECT_ROOT/security-reports"
    mkdir -p "$output_dir"

    cd "$PROJECT_ROOT"

    if gitleaks detect --config src/api/.gitleaks.toml --report-path "$output_dir/gitleaks-report.json" --report-format json; then
        log "✅ No secrets detected"
    else
        error "❌ Secrets detected! Check $output_dir/gitleaks-report.json"
        return 1
    fi
}

# Static Application Security Testing (SAST)
run_sast_scan() {
    log "Running Static Application Security Testing (SAST)..."

    local output_dir="$PROJECT_ROOT/security-reports"
    mkdir -p "$output_dir"

    # Bandit for Python security issues
    log "Running Bandit security scan..."
    cd "$PROJECT_ROOT/src/api"

    if bandit -r app/ -f json -o "$output_dir/bandit-report.json" --severity-level medium; then
        log "✅ Bandit scan completed"
    else
        warn "⚠️ Bandit found security issues"
    fi

    # Display summary
    bandit -r app/ -f txt --severity-level medium || true

    # Semgrep for additional SAST
    if command -v semgrep &> /dev/null; then
        log "Running Semgrep SAST scan..."
        semgrep --config=auto app/ --json --output="$output_dir/semgrep-report.json" || true
        semgrep --config=auto app/ --text || true
    fi
}

# Dependency vulnerability scanning
run_dependency_scan() {
    log "Running dependency vulnerability scanning..."

    local output_dir="$PROJECT_ROOT/security-reports"
    mkdir -p "$output_dir"

    cd "$PROJECT_ROOT/src/api"

    # Safety for Python package vulnerabilities
    log "Running Safety check for Python dependencies..."
    if safety check --json --output "$output_dir/safety-report.json"; then
        log "✅ No known vulnerabilities in dependencies"
    else
        warn "⚠️ Vulnerabilities found in dependencies"
    fi

    safety check --short-report || true

    # License compliance check
    log "Checking license compliance..."
    if command -v pip-licenses &> /dev/null || pip install pip-licenses; then
        pip-licenses --format=json --output-file="$output_dir/licenses.json"
        pip-licenses --allow-only="MIT;BSD;Apache;ISC;PSF;MPL" || warn "License compliance issues found"
    fi
}

# Container security scanning
run_container_scan() {
    log "Running container security scanning..."

    local output_dir="$PROJECT_ROOT/security-reports"
    mkdir -p "$output_dir"

    cd "$PROJECT_ROOT"

    # Build container for scanning
    if [[ -f "src/api/Dockerfile.secure" ]]; then
        log "Building container for security scanning..."
        docker build -t xorb-api-security-scan -f src/api/Dockerfile.secure src/api/

        # Trivy container vulnerability scan
        if command -v trivy &> /dev/null; then
            log "Running Trivy container vulnerability scan..."
            trivy image --format json --output "$output_dir/trivy-report.json" xorb-api-security-scan
            trivy image --severity HIGH,CRITICAL xorb-api-security-scan || true
        fi

        # Grype container vulnerability scan
        if command -v grype &> /dev/null; then
            log "Running Grype container vulnerability scan..."
            grype xorb-api-security-scan -o json > "$output_dir/grype-report.json" || true
            grype xorb-api-security-scan || true
        fi

        # Cleanup
        docker rmi xorb-api-security-scan || true
    else
        warn "Dockerfile.secure not found, skipping container scan"
    fi
}

# Infrastructure security scanning
run_infrastructure_scan() {
    log "Running infrastructure security scanning..."

    local output_dir="$PROJECT_ROOT/security-reports"
    mkdir -p "$output_dir"

    cd "$PROJECT_ROOT"

    # Checkov for Infrastructure as Code
    if command -v checkov &> /dev/null || pip install checkov; then
        log "Running Checkov IaC security scan..."
        checkov --framework terraform,dockerfile,docker_compose,yaml \
               --output json --output-file "$output_dir/checkov-report.json" \
               --directory . || true

        checkov --framework terraform,dockerfile,docker_compose,yaml \
               --directory . --compact || true
    fi

    # Hadolint for Dockerfile security
    if command -v hadolint &> /dev/null; then
        log "Running Hadolint Dockerfile security scan..."
        find . -name "Dockerfile*" -exec hadolint {} \; || true
    fi
}

# Generate security report
generate_security_report() {
    log "Generating comprehensive security report..."

    local output_dir="$PROJECT_ROOT/security-reports"
    local report_file="$output_dir/security-summary-report.md"

    cat > "$report_file" << EOF
# XORB Security Scan Report

**Generated:** $(date)
**Scan Type:** Comprehensive Security Assessment
**Repository:** XORB Cybersecurity Platform

## Executive Summary

This report contains the results of comprehensive security scanning including:

- Secret detection and credential scanning
- Static Application Security Testing (SAST)
- Dependency vulnerability assessment
- Container security analysis
- Infrastructure as Code security review

## Scan Results

### 🔍 Secret Scanning
- **Tool:** Gitleaks
- **Status:** $([ -f "$output_dir/gitleaks-report.json" ] && echo "✅ Completed" || echo "❌ Failed")
- **Report:** \`gitleaks-report.json\`

### 🛡️ Static Code Analysis (SAST)
- **Tools:** Bandit, Semgrep
- **Status:** $([ -f "$output_dir/bandit-report.json" ] && echo "✅ Completed" || echo "❌ Failed")
- **Reports:** \`bandit-report.json\`, \`semgrep-report.json\`

### 📦 Dependency Scanning
- **Tool:** Safety
- **Status:** $([ -f "$output_dir/safety-report.json" ] && echo "✅ Completed" || echo "❌ Failed")
- **Report:** \`safety-report.json\`

### 🐳 Container Security
- **Tools:** Trivy, Grype
- **Status:** $([ -f "$output_dir/trivy-report.json" ] && echo "✅ Completed" || echo "❌ Failed")
- **Reports:** \`trivy-report.json\`, \`grype-report.json\`

### 🏗️ Infrastructure Security
- **Tool:** Checkov
- **Status:** $([ -f "$output_dir/checkov-report.json" ] && echo "✅ Completed" || echo "❌ Failed")
- **Report:** \`checkov-report.json\`

## Recommendations

1. **Regular Scanning:** Run security scans on every commit and pull request
2. **Automated Remediation:** Set up automated dependency updates with tools like Dependabot
3. **Security Training:** Provide security awareness training for development team
4. **Monitoring:** Implement runtime security monitoring and anomaly detection
5. **Incident Response:** Maintain updated incident response procedures

## Next Steps

1. Review detailed findings in individual report files
2. Prioritize and remediate high and critical severity issues
3. Implement security fixes and re-scan
4. Update security policies and procedures as needed
5. Schedule regular security assessments

---
*Report generated by XORB Security Scanner v1.0*
EOF

    log "Security report generated: $report_file"
}

# Clean up old reports
cleanup_reports() {
    local output_dir="$PROJECT_ROOT/security-reports"

    if [[ -d "$output_dir" ]]; then
        find "$output_dir" -name "*.json" -mtime +7 -delete 2>/dev/null || true
        find "$output_dir" -name "*.txt" -mtime +7 -delete 2>/dev/null || true
    fi
}

# Main execution function
run_security_scan() {
    local scan_type="${1:-full}"

    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════╗"
    echo "║           XORB Security Scanner v1.0             ║"
    echo "║     Comprehensive Security Assessment Tool       ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo -e "${NC}"

    log "Starting security scan (type: $scan_type)..."

    # Install tools if not in CI
    if ! is_ci; then
        install_security_tools
    fi

    # Clean up old reports
    cleanup_reports

    case $scan_type in
        "secrets")
            run_secret_scan
            ;;
        "sast")
            run_sast_scan
            ;;
        "dependencies")
            run_dependency_scan
            ;;
        "container")
            run_container_scan
            ;;
        "infrastructure")
            run_infrastructure_scan
            ;;
        "full"|*)
            run_secret_scan
            run_sast_scan
            run_dependency_scan
            run_container_scan
            run_infrastructure_scan
            generate_security_report
            ;;
    esac

    log "Security scan completed! 🚀"
    log "Reports available in: $PROJECT_ROOT/security-reports/"
}

# Handle script arguments
case "${1:-}" in
    "secrets"|"sast"|"dependencies"|"container"|"infrastructure"|"full")
        run_security_scan "$1"
        ;;
    "install")
        install_security_tools
        ;;
    "clean")
        cleanup_reports
        log "Old security reports cleaned up"
        ;;
    "")
        run_security_scan "full"
        ;;
    *)
        echo "Usage: $0 [secrets|sast|dependencies|container|infrastructure|full|install|clean]"
        echo ""
        echo "Scan types:"
        echo "  secrets        - Secret and credential scanning"
        echo "  sast          - Static Application Security Testing"
        echo "  dependencies  - Dependency vulnerability scanning"
        echo "  container     - Container security scanning"
        echo "  infrastructure - Infrastructure as Code security"
        echo "  full          - Complete security assessment (default)"
        echo ""
        echo "Utility commands:"
        echo "  install       - Install security scanning tools"
        echo "  clean         - Clean up old security reports"
        exit 1
        ;;
esac
