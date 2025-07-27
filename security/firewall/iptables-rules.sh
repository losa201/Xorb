#!/bin/bash

# XORB Advanced Firewall Rules
# Comprehensive iptables configuration for XORB security

echo "🔥 Configuring advanced firewall rules..."

# Backup existing rules
iptables-save > /root/Xorb/security/firewall/iptables-backup-$(date +%Y%m%d_%H%M%S).rules

# Flush existing rules
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X
iptables -t mangle -F
iptables -t mangle -X

# Set default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow established and related connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (rate limited)
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m limit --limit 5/min --limit-burst 10 -j ACCEPT

# Allow HTTP/HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow XORB services (restricted to specific source IPs in production)
iptables -A INPUT -p tcp --dport 8000 -s 172.0.0.0/8 -j ACCEPT  # API
iptables -A INPUT -p tcp --dport 8080 -s 172.0.0.0/8 -j ACCEPT  # Orchestrator
iptables -A INPUT -p tcp --dport 9000 -s 172.0.0.0/8 -j ACCEPT  # Worker

# Allow monitoring services (restricted)
iptables -A INPUT -p tcp --dport 9090 -s 172.0.0.0/8 -j ACCEPT  # Prometheus
iptables -A INPUT -p tcp --dport 3000 -s 172.0.0.0/8 -j ACCEPT  # Grafana

# Allow database connections (very restricted)
iptables -A INPUT -p tcp --dport 5432 -s 172.20.0.0/16 -j ACCEPT  # PostgreSQL
iptables -A INPUT -p tcp --dport 6379 -s 172.20.0.0/16 -j ACCEPT  # Redis

# DDoS protection
iptables -A INPUT -p tcp --dport 80 -m limit --limit 100/minute --limit-burst 200 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -m limit --limit 100/minute --limit-burst 200 -j ACCEPT

# Log dropped packets
iptables -A INPUT -m limit --limit 5/min -j LOG --log-prefix "iptables-dropped: " --log-level 4

# Drop everything else
iptables -A INPUT -j DROP

# Save rules
iptables-save > /etc/iptables/rules.v4 2>/dev/null || iptables-save > /root/Xorb/security/firewall/iptables-active.rules

echo "✅ Firewall rules configured"
