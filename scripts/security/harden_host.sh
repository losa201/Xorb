#!/bin/bash

# XORB Host Security Hardening Script
# Applies security controls for XORB deployment environments

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Check if running as root
check_privileges() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi
}

# Detect OS and package manager
detect_os() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$ID
        VER=$VERSION_ID
    else
        log_error "Cannot detect OS"
        exit 1
    fi
    
    log_info "Detected OS: $OS $VER"
}

# Configure firewall
configure_firewall() {
    log_info "Configuring firewall..."
    
    case $OS in
        ubuntu|debian)
            # Install ufw if not present
            apt-get update
            apt-get install -y ufw
            
            # Reset to defaults
            ufw --force reset
            
            # Default policies
            ufw default deny incoming
            ufw default allow outgoing
            
            # Allow SSH (preserve access)
            ufw allow ssh
            
            # Allow XORB services
            ufw allow 8000/tcp   # API
            ufw allow 8080/tcp   # Orchestrator
            ufw allow 9090/tcp   # Worker/Prometheus
            ufw allow 3000/tcp   # Grafana
            
            # Enable firewall
            ufw --force enable
            
            log_success "UFW firewall configured"
            ;;
        
        centos|rhel|fedora)
            # Configure firewalld
            systemctl enable firewalld
            systemctl start firewalld
            
            # XORB services
            firewall-cmd --permanent --add-port=8000/tcp
            firewall-cmd --permanent --add-port=8080/tcp
            firewall-cmd --permanent --add-port=9090/tcp
            firewall-cmd --permanent --add-port=3000/tcp
            
            # Reload configuration
            firewall-cmd --reload
            
            log_success "Firewalld configured"
            ;;
        
        *)
            log_warning "Firewall configuration not implemented for $OS"
            ;;
    esac
}

# Configure SELinux (RHEL/CentOS)
configure_selinux() {
    if [[ $OS == "centos" || $OS == "rhel" ]]; then
        log_info "Configuring SELinux..."
        
        # Install SELinux tools
        yum install -y policycoreutils-python-utils setools-console
        
        # Set SELinux to enforcing mode
        setenforce 1
        sed -i 's/SELINUX=.*/SELINUX=enforcing/' /etc/selinux/config
        
        # Allow container management
        setsebool -P container_manage_cgroup true
        setsebool -P virt_use_execmem true
        
        log_success "SELinux configured"
    fi
}

# Configure AppArmor (Ubuntu/Debian)
configure_apparmor() {
    if [[ $OS == "ubuntu" || $OS == "debian" ]]; then
        log_info "Configuring AppArmor..."
        
        # Install AppArmor
        apt-get install -y apparmor apparmor-utils
        
        # Enable AppArmor
        systemctl enable apparmor
        systemctl start apparmor
        
        # Docker profile
        if [[ -f /etc/apparmor.d/docker ]]; then
            aa-enforce /etc/apparmor.d/docker
        fi
        
        log_success "AppArmor configured"
    fi
}

# Configure audit logging
configure_audit() {
    log_info "Configuring audit logging..."
    
    case $OS in
        ubuntu|debian)
            apt-get install -y auditd audispd-plugins
            ;;
        centos|rhel|fedora)
            yum install -y audit audit-libs
            ;;
    esac
    
    # Create XORB audit rules
    cat > /etc/audit/rules.d/xorb.rules << 'EOF'
# XORB Security Audit Rules

# Monitor XORB configuration changes
-w /opt/xorb/config/ -p wa -k xorb_config
-w /etc/systemd/system/xorb*.service -p wa -k xorb_systemd

# Monitor Docker operations
-w /var/lib/docker/ -p wa -k docker_data
-w /etc/docker/ -p wa -k docker_config

# Monitor privileged operations
-a always,exit -F arch=b64 -S mount -F auid>=1000 -F auid!=4294967295 -k privileged_mount
-a always,exit -F arch=b64 -S unshare -F auid>=1000 -F auid!=4294967295 -k privileged_unshare

# Monitor network configuration
-w /etc/hosts -p wa -k network_config
-w /etc/resolv.conf -p wa -k network_config

# Monitor user management
-w /etc/passwd -p wa -k user_mgmt
-w /etc/group -p wa -k user_mgmt
-w /etc/shadow -p wa -k user_mgmt

# Monitor sudo usage
-w /var/log/sudo.log -p wa -k sudo_log
EOF
    
    # Enable and start auditd
    systemctl enable auditd
    systemctl restart auditd
    
    log_success "Audit logging configured"
}

# Harden Docker daemon
harden_docker() {
    log_info "Hardening Docker daemon..."
    
    # Create Docker daemon configuration
    mkdir -p /etc/docker
    
    cat > /etc/docker/daemon.json << 'EOF'
{
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "live-restore": true,
    "userland-proxy": false,
    "no-new-privileges": true,
    "seccomp-profile": "/etc/docker/seccomp.json",
    "selinux-enabled": true,
    "storage-driver": "overlay2",
    "storage-opts": [
        "overlay2.override_kernel_check=true"
    ]
}
EOF
    
    # Create seccomp profile
    cat > /etc/docker/seccomp.json << 'EOF'
{
    "defaultAction": "SCMP_ACT_ERRNO",
    "syscalls": [
        {
            "names": [
                "accept",
                "accept4",
                "access",
                "alarm",
                "bind",
                "brk",
                "chdir",
                "chmod",
                "chown",
                "clock_getres",
                "clock_gettime",
                "clone",
                "close",
                "connect",
                "dup",
                "dup2",
                "execve",
                "exit",
                "exit_group",
                "fchmod",
                "fchown",
                "fcntl",
                "fork",
                "fstat",
                "futex",
                "getcwd",
                "getdents",
                "getegid",
                "geteuid",
                "getgid",
                "getpid",
                "getppid",
                "gettid",
                "getuid",
                "listen",
                "lseek",
                "lstat",
                "madvise",
                "mkdir",
                "mmap",
                "mprotect",
                "munmap",
                "nanosleep",
                "open",
                "openat",
                "pipe",
                "poll",
                "read",
                "readlink",
                "recvfrom",
                "rename",
                "rmdir",
                "select",
                "sendto",
                "setgid",
                "setuid",
                "socket",
                "stat",
                "statfs",
                "symlink",
                "umask",
                "unlink",
                "wait4",
                "write"
            ],
            "action": "SCMP_ACT_ALLOW"
        }
    ]
}
EOF
    
    # Restart Docker
    systemctl daemon-reload
    systemctl restart docker
    
    log_success "Docker daemon hardened"
}

# Configure log rotation
configure_logging() {
    log_info "Configuring log rotation..."
    
    # XORB log rotation
    cat > /etc/logrotate.d/xorb << 'EOF'
/var/log/xorb/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    postrotate
        systemctl reload rsyslog > /dev/null 2>&1 || true
    endscript
}

/opt/xorb/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF
    
    # Docker log rotation (handled by daemon.json)
    
    log_success "Log rotation configured"
}

# Set file permissions
set_permissions() {
    log_info "Setting secure file permissions..."
    
    # Create XORB directories if they don't exist
    mkdir -p /opt/xorb/{config,logs,data}
    mkdir -p /var/log/xorb
    
    # Set ownership
    chown -R root:docker /opt/xorb
    chown -R root:root /var/log/xorb
    
    # Set permissions
    chmod 750 /opt/xorb
    chmod 640 /opt/xorb/config/*
    chmod 755 /opt/xorb/logs
    chmod 750 /opt/xorb/data
    chmod 755 /var/log/xorb
    
    # Secure systemd files
    if [[ -d /etc/systemd/system ]]; then
        chmod 644 /etc/systemd/system/xorb*.service
        chmod 644 /etc/systemd/system/xorb*.target
    fi
    
    log_success "File permissions configured"
}

# Configure fail2ban
configure_fail2ban() {
    log_info "Configuring fail2ban..."
    
    case $OS in
        ubuntu|debian)
            apt-get install -y fail2ban
            ;;
        centos|rhel|fedora)
            yum install -y fail2ban
            ;;
    esac
    
    # Create XORB jail configuration
    cat > /etc/fail2ban/jail.d/xorb.conf << 'EOF'
[xorb-api]
enabled = true
port = 8000
filter = xorb-api
logpath = /opt/xorb/logs/api.log
maxretry = 5
bantime = 3600
findtime = 600

[xorb-auth]
enabled = true
port = 8000,8080
filter = xorb-auth
logpath = /opt/xorb/logs/*.log
maxretry = 3
bantime = 7200
findtime = 300
EOF
    
    # Create filter for XORB API
    cat > /etc/fail2ban/filter.d/xorb-api.conf << 'EOF'
[Definition]
failregex = ^.*\[.*\] ".*" 40[13] \d+ ".*" ".*" <HOST>.*$
            ^.*\[.*\] client <HOST> denied.*$
ignoreregex =
EOF
    
    # Create filter for XORB auth
    cat > /etc/fail2ban/filter.d/xorb-auth.conf << 'EOF'
[Definition]
failregex = ^.*authentication failed.*from <HOST>.*$
            ^.*invalid credentials.*from <HOST>.*$
            ^.*unauthorized access.*from <HOST>.*$
ignoreregex =
EOF
    
    # Enable and start fail2ban
    systemctl enable fail2ban
    systemctl restart fail2ban
    
    log_success "Fail2ban configured"
}

# Configure network security
configure_network_security() {
    log_info "Configuring network security..."
    
    # Disable IPv6 if not needed
    if ! grep -q "net.ipv6.conf.all.disable_ipv6" /etc/sysctl.conf; then
        echo "net.ipv6.conf.all.disable_ipv6 = 1" >> /etc/sysctl.conf
        echo "net.ipv6.conf.default.disable_ipv6 = 1" >> /etc/sysctl.conf
    fi
    
    # Network security settings
    cat >> /etc/sysctl.conf << 'EOF'

# XORB Network Security Settings
net.ipv4.ip_forward = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv4.conf.all.secure_redirects = 0
net.ipv4.conf.default.secure_redirects = 0
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.tcp_synack_retries = 2
net.ipv4.tcp_syn_retries = 5
EOF
    
    # Apply sysctl settings
    sysctl -p
    
    log_success "Network security configured"
}

# Generate security report
generate_security_report() {
    log_info "Generating security report..."
    
    REPORT_FILE="/opt/xorb/security_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$REPORT_FILE" << EOF
XORB Security Hardening Report
Generated: $(date)
Host: $(hostname)
OS: $OS $VER

=== Firewall Status ===
$(ufw status 2>/dev/null || firewall-cmd --list-all 2>/dev/null || echo "Firewall not configured")

=== SELinux/AppArmor Status ===
$(sestatus 2>/dev/null || aa-status 2>/dev/null || echo "Not applicable")

=== Audit Status ===
$(systemctl is-active auditd 2>/dev/null || echo "Audit not running")

=== Docker Security ===
$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "Docker not running")

=== Fail2ban Status ===
$(systemctl is-active fail2ban 2>/dev/null || echo "Fail2ban not running")

=== Network Configuration ===
$(sysctl net.ipv4.ip_forward net.ipv4.tcp_syncookies 2>/dev/null)

=== File Permissions ===
$(ls -la /opt/xorb/ 2>/dev/null || echo "XORB directory not found")

=== Services Status ===
$(systemctl status xorb-* 2>/dev/null | grep -E "(Active|Loaded)" || echo "XORB services not found")

=== Security Recommendations ===
1. Regularly update system packages
2. Monitor audit logs for suspicious activity
3. Review firewall rules monthly
4. Rotate secrets and passwords
5. Backup configuration files
6. Monitor resource usage
7. Update Docker and containers regularly

=== Hardening Completed ===
Date: $(date)
Status: SUCCESS
EOF
    
    log_success "Security report generated: $REPORT_FILE"
}

# Main hardening function
main() {
    echo -e "${BLUE}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════╗
║              XORB SECURITY HARDENING                ║
║                                                      ║
║  🛡️  Host Security Configuration                     ║
║  🔒  Container Security Hardening                    ║  
║  📊  Audit and Monitoring Setup                      ║
║  🔥  Firewall and Network Security                   ║
╚══════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    
    log_info "Starting XORB security hardening..."
    
    check_privileges
    detect_os
    
    # Core security configurations
    configure_firewall
    configure_selinux
    configure_apparmor
    configure_audit
    harden_docker
    configure_logging
    set_permissions
    configure_fail2ban
    configure_network_security
    
    # Generate final report
    generate_security_report
    
    echo
    log_success "🎉 XORB security hardening completed!"
    echo
    echo -e "${GREEN}Security features enabled:${NC}"
    echo "  ✅ Firewall configured with XORB ports"
    echo "  ✅ SELinux/AppArmor hardening applied"
    echo "  ✅ Audit logging enabled"
    echo "  ✅ Docker daemon hardened"
    echo "  ✅ Log rotation configured"
    echo "  ✅ File permissions secured"
    echo "  ✅ Fail2ban protection enabled"
    echo "  ✅ Network security hardened"
    echo
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Review security report: $(ls /opt/xorb/security_report_*.txt | tail -1)"
    echo "  2. Test XORB functionality"
    echo "  3. Monitor audit logs: journalctl -f _TRANSPORT=audit"
    echo "  4. Verify firewall: ufw status (or firewall-cmd --list-all)"
    echo
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "XORB Security Hardening Script"
        echo
        echo "Usage: $0 [options]"
        echo
        echo "Options:"
        echo "  --help, -h          Show this help message"
        echo "  --audit-only        Only configure audit logging"
        echo "  --firewall-only     Only configure firewall"
        echo "  --docker-only       Only harden Docker"
        echo
        exit 0
        ;;
    --audit-only)
        check_privileges
        detect_os
        configure_audit
        ;;
    --firewall-only)
        check_privileges
        detect_os
        configure_firewall
        ;;
    --docker-only)
        check_privileges
        detect_os
        harden_docker
        ;;
    "")
        main
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac