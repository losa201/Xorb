#!/bin/bash
# Production Environment Setup for XORB PTaaS
# Installs and configures enterprise-grade security tools

set -euo pipefail

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

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root for security tool installation"
        exit 1
    fi
}

# System requirements check
check_system_requirements() {
    log "Checking system requirements..."

    # Check OS
    if [[ ! -f /etc/os-release ]]; then
        error "Cannot determine OS version"
        exit 1
    fi

    . /etc/os-release
    log "Detected OS: $NAME $VERSION"

    # Check architecture
    ARCH=$(uname -m)
    log "Architecture: $ARCH"

    # Check available memory (minimum 4GB)
    MEM_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $MEM_GB -lt 4 ]]; then
        warn "System has only ${MEM_GB}GB memory. Recommended: 8GB+"
    fi

    # Check disk space (minimum 20GB)
    DISK_GB=$(df / | awk 'NR==2{print int($4/1024/1024)}')
    if [[ $DISK_GB -lt 20 ]]; then
        error "Insufficient disk space: ${DISK_GB}GB available. Need 20GB+"
        exit 1
    fi

    log "System requirements check: PASSED"
}

# Update system packages
update_system() {
    log "Updating system packages..."

    if command -v apt-get &> /dev/null; then
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get upgrade -y -qq
        apt-get install -y -qq \
            curl \
            wget \
            git \
            build-essential \
            python3 \
            python3-pip \
            python3-dev \
            golang-go \
            nodejs \
            npm \
            docker.io \
            docker-compose \
            unzip \
            jq \
            ca-certificates \
            gnupg \
            lsb-release
    elif command -v yum &> /dev/null; then
        yum update -y
        yum groupinstall -y "Development Tools"
        yum install -y \
            curl \
            wget \
            git \
            python3 \
            python3-pip \
            python3-devel \
            golang \
            nodejs \
            npm \
            docker \
            docker-compose \
            unzip \
            jq
    else
        error "Unsupported package manager. Please install manually."
        exit 1
    fi

    log "System packages updated successfully"
}

# Install Nmap
install_nmap() {
    log "Installing Nmap..."

    if command -v apt-get &> /dev/null; then
        apt-get install -y nmap
    elif command -v yum &> /dev/null; then
        yum install -y nmap
    fi

    # Verify installation
    if command -v nmap &> /dev/null; then
        NMAP_VERSION=$(nmap --version | head -1 | awk '{print $3}')
        log "Nmap $NMAP_VERSION installed successfully"
    else
        error "Nmap installation failed"
        exit 1
    fi
}

# Install Nuclei
install_nuclei() {
    log "Installing Nuclei..."

    # Install using Go
    if command -v go &> /dev/null; then
        export GO111MODULE=on
        go install -v github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest

        # Move to system path
        cp ~/go/bin/nuclei /usr/local/bin/
    else
        # Download binary release
        NUCLEI_VERSION="v3.1.0"
        wget -q "https://github.com/projectdiscovery/nuclei/releases/download/${NUCLEI_VERSION}/nuclei_${NUCLEI_VERSION#v}_linux_amd64.zip"
        unzip -q "nuclei_${NUCLEI_VERSION#v}_linux_amd64.zip"
        mv nuclei /usr/local/bin/
        chmod +x /usr/local/bin/nuclei
        rm -f "nuclei_${NUCLEI_VERSION#v}_linux_amd64.zip"
    fi

    # Update nuclei templates
    nuclei -update-templates -silent

    # Verify installation
    if command -v nuclei &> /dev/null; then
        NUCLEI_VERSION=$(nuclei -version 2>&1 | grep -o 'v[0-9.]*')
        log "Nuclei $NUCLEI_VERSION installed successfully"
    else
        error "Nuclei installation failed"
        exit 1
    fi
}

# Install Nikto
install_nikto() {
    log "Installing Nikto..."

    if command -v apt-get &> /dev/null; then
        apt-get install -y nikto
    elif command -v yum &> /dev/null; then
        # Install from source for CentOS/RHEL
        cd /opt
        git clone https://github.com/sullo/nikto.git
        ln -sf /opt/nikto/program/nikto.pl /usr/local/bin/nikto
        chmod +x /usr/local/bin/nikto
    fi

    # Verify installation
    if command -v nikto &> /dev/null; then
        log "Nikto installed successfully"
    else
        error "Nikto installation failed"
        exit 1
    fi
}

# Install SSLScan
install_sslscan() {
    log "Installing SSLScan..."

    if command -v apt-get &> /dev/null; then
        apt-get install -y sslscan
    else
        # Build from source
        cd /tmp
        git clone https://github.com/rbsec/sslscan.git
        cd sslscan
        make static
        cp sslscan /usr/local/bin/
        cd /
        rm -rf /tmp/sslscan
    fi

    # Verify installation
    if command -v sslscan &> /dev/null; then
        log "SSLScan installed successfully"
    else
        error "SSLScan installation failed"
        exit 1
    fi
}

# Install DIRB
install_dirb() {
    log "Installing DIRB..."

    if command -v apt-get &> /dev/null; then
        apt-get install -y dirb
    else
        # Build from source
        cd /tmp
        wget -q http://dirb.sourceforge.net/releases/dirb222.tar.gz
        tar -xzf dirb222.tar.gz
        cd dirb222
        ./configure
        make
        cp dirb /usr/local/bin/
        cp -r wordlists /usr/share/dirb/
        cd /
        rm -rf /tmp/dirb222*
    fi

    # Verify installation
    if command -v dirb &> /dev/null; then
        log "DIRB installed successfully"
    else
        error "DIRB installation failed"
        exit 1
    fi
}

# Install Masscan
install_masscan() {
    log "Installing Masscan..."

    if command -v apt-get &> /dev/null; then
        apt-get install -y masscan
    else
        # Build from source
        cd /tmp
        git clone https://github.com/robertdavidgraham/masscan.git
        cd masscan
        make
        cp bin/masscan /usr/local/bin/
        cd /
        rm -rf /tmp/masscan
    fi

    # Verify installation
    if command -v masscan &> /dev/null; then
        log "Masscan installed successfully"
    else
        warn "Masscan installation failed (optional tool)"
    fi
}

# Install additional security tools
install_additional_tools() {
    log "Installing additional security tools..."

    # Install subfinder for subdomain discovery
    if command -v go &> /dev/null; then
        go install -v github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest
        cp ~/go/bin/subfinder /usr/local/bin/ 2>/dev/null || true
    fi

    # Install httpx for HTTP probing
    if command -v go &> /dev/null; then
        go install -v github.com/projectdiscovery/httpx/cmd/httpx@latest
        cp ~/go/bin/httpx /usr/local/bin/ 2>/dev/null || true
    fi

    # Install naabu for port scanning
    if command -v go &> /dev/null; then
        go install -v github.com/projectdiscovery/naabu/v2/cmd/naabu@latest
        cp ~/go/bin/naabu /usr/local/bin/ 2>/dev/null || true
    fi

    log "Additional tools installation completed"
}

# Configure security tools
configure_tools() {
    log "Configuring security tools..."

    # Create tools configuration directory
    mkdir -p /opt/xorb/config
    mkdir -p /opt/xorb/wordlists
    mkdir -p /opt/xorb/templates

    # Download additional wordlists
    cd /opt/xorb/wordlists
    wget -q https://github.com/danielmiessler/SecLists/archive/master.zip -O seclists.zip
    unzip -q seclists.zip
    mv SecLists-master/* .
    rm -rf SecLists-master seclists.zip

    # Create Nmap configuration
    cat > /opt/xorb/config/nmap.conf << 'EOF'
# XORB Nmap Configuration
timing_template = T4
max_rate = 1000
min_rate = 100
max_retries = 1
host_timeout = 300
scan_delay = 0
EOF

    # Create Nuclei configuration
    cat > /opt/xorb/config/nuclei.yaml << 'EOF'
# XORB Nuclei Configuration
templates-directory: /root/nuclei-templates
rate-limit: 150
bulk-size: 25
timeout: 10
retries: 1
max-host-error: 30
severity:
  - critical
  - high
  - medium
  - low
  - info
EOF

    # Set proper permissions
    chmod 755 /opt/xorb/config
    chmod 644 /opt/xorb/config/*

    log "Security tools configuration completed"
}

# Create scanning user and environment
setup_scanning_environment() {
    log "Setting up scanning environment..."

    # Create dedicated scanning user
    if ! id "xorb-scanner" &>/dev/null; then
        useradd -r -s /bin/bash -d /opt/xorb xorb-scanner
        usermod -aG docker xorb-scanner
    fi

    # Create scanning directories
    mkdir -p /opt/xorb/{scans,reports,logs,temp}
    chown -R xorb-scanner:xorb-scanner /opt/xorb

    # Set up logging
    mkdir -p /var/log/xorb
    touch /var/log/xorb/scanner.log
    chown xorb-scanner:xorb-scanner /var/log/xorb/scanner.log

    # Create systemd service for scanner
    cat > /etc/systemd/system/xorb-scanner.service << 'EOF'
[Unit]
Description=XORB PTaaS Scanner Service
After=network.target

[Service]
Type=simple
User=xorb-scanner
Group=xorb-scanner
WorkingDirectory=/opt/xorb
ExecStart=/usr/bin/python3 /opt/xorb/scanner_daemon.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/xorb/scanner.log
StandardError=append:/var/log/xorb/scanner.log

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload

    log "Scanning environment setup completed"
}

# Performance and security hardening
harden_system() {
    log "Applying security hardening..."

    # Increase file descriptor limits for scanning
    cat >> /etc/security/limits.conf << 'EOF'
xorb-scanner soft nofile 65536
xorb-scanner hard nofile 65536
xorb-scanner soft nproc 32768
xorb-scanner hard nproc 32768
EOF

    # Configure kernel parameters for network scanning
    cat > /etc/sysctl.d/99-xorb-scanner.conf << 'EOF'
# XORB Scanner Network Optimizations
net.core.rmem_default = 262144
net.core.rmem_max = 16777216
net.core.wmem_default = 262144
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 65536 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_window_scaling = 1
net.ipv4.tcp_congestion_control = bbr
EOF

    sysctl -p /etc/sysctl.d/99-xorb-scanner.conf

    # Configure firewall rules
    if command -v ufw &> /dev/null; then
        ufw --force enable
        ufw default deny incoming
        ufw default allow outgoing
        ufw allow ssh
        ufw allow 8000/tcp  # XORB API
        ufw allow 3000/tcp  # Frontend
    fi

    log "Security hardening completed"
}

# Validate installation
validate_installation() {
    log "Validating security tools installation..."

    TOOLS=("nmap" "nuclei" "nikto" "sslscan" "dirb")
    FAILED_TOOLS=()

    for tool in "${TOOLS[@]}"; do
        if command -v "$tool" &> /dev/null; then
            log "✓ $tool is installed and accessible"
        else
            error "✗ $tool is not installed or not in PATH"
            FAILED_TOOLS+=("$tool")
        fi
    done

    # Test tool functionality
    log "Testing tool functionality..."

    # Test Nmap
    if nmap --version &>/dev/null; then
        log "✓ Nmap functionality test passed"
    else
        error "✗ Nmap functionality test failed"
        FAILED_TOOLS+=("nmap-test")
    fi

    # Test Nuclei
    if nuclei -version &>/dev/null; then
        log "✓ Nuclei functionality test passed"
    else
        error "✗ Nuclei functionality test failed"
        FAILED_TOOLS+=("nuclei-test")
    fi

    if [[ ${#FAILED_TOOLS[@]} -eq 0 ]]; then
        log "All security tools validation: PASSED"
        return 0
    else
        error "Failed tools: ${FAILED_TOOLS[*]}"
        return 1
    fi
}

# Generate installation report
generate_report() {
    log "Generating installation report..."

    REPORT_FILE="/opt/xorb/installation-report-$(date +%Y%m%d-%H%M%S).txt"

    cat > "$REPORT_FILE" << EOF
XORB PTaaS Production Environment Installation Report
Generated: $(date)
Hostname: $(hostname)
OS: $(lsb_release -d | cut -f2)
Architecture: $(uname -m)

Installed Security Tools:
$(for tool in nmap nuclei nikto sslscan dirb masscan; do
    if command -v "$tool" &>/dev/null; then
        echo "✓ $tool - $(command -v "$tool")"
    else
        echo "✗ $tool - Not installed"
    fi
done)

System Configuration:
- User: xorb-scanner created
- Directories: /opt/xorb structure created
- Firewall: Configured and enabled
- Kernel parameters: Optimized for scanning
- File descriptors: Increased limits

Next Steps:
1. Configure XORB application settings
2. Set up database connections
3. Configure API authentication
4. Run initial system tests
5. Deploy XORB services

For support, contact: devops@xorb-security.com
EOF

    chown xorb-scanner:xorb-scanner "$REPORT_FILE"
    log "Installation report saved to: $REPORT_FILE"
}

# Main installation function
main() {
    log "Starting XORB PTaaS Production Environment Setup"
    log "=================================================="

    check_root
    check_system_requirements
    update_system
    install_nmap
    install_nuclei
    install_nikto
    install_sslscan
    install_dirb
    install_masscan
    install_additional_tools
    configure_tools
    setup_scanning_environment
    harden_system

    if validate_installation; then
        generate_report
        log "=================================================="
        log "XORB PTaaS Production Environment Setup: COMPLETED"
        log "=================================================="
        log "Next steps:"
        log "1. Start XORB services: systemctl start xorb-scanner"
        log "2. Check status: systemctl status xorb-scanner"
        log "3. View logs: tail -f /var/log/xorb/scanner.log"
        log "4. Configuration files: /opt/xorb/config/"
    else
        error "Installation completed with errors. Check the logs above."
        exit 1
    fi
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
