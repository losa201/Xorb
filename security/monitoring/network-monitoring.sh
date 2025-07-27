#!/bin/bash

# XORB Network Intrusion Detection System
# Real-time network monitoring and threat detection

echo "🔍 Starting XORB network monitoring..."

# Function to monitor suspicious activity
monitor_connections() {
    echo "📡 Monitoring network connections..."
    
    # Monitor for port scanning
    netstat -tuln | grep LISTEN | while read line; do
        port=$(echo "$line" | awk '{print $4}' | cut -d: -f2)
        if [ "$port" -gt 1024 ] && [ "$port" -lt 65535 ]; then
            echo "$(date): Listening on port $port" >> /var/log/xorb/network-monitor.log
        fi
    done
    
    # Monitor for unusual traffic patterns
    ss -tuln | grep -E "(8000|8080|9000|5432|6379)" | while read line; do
        echo "$(date): XORB service connection: $line" >> /var/log/xorb/network-monitor.log
    done
}

# Function to detect anomalies
detect_anomalies() {
    echo "🚨 Detecting network anomalies..."
    
    # Check for excessive connections
    CONN_COUNT=$(netstat -an | grep ESTABLISHED | wc -l)
    if [ "$CONN_COUNT" -gt 1000 ]; then
        echo "$(date): WARNING: High connection count: $CONN_COUNT" >> /var/log/xorb/security-alerts.log
    fi
    
    # Check for suspicious IPs
    netstat -an | grep ESTABLISHED | awk '{print $5}' | cut -d: -f1 | sort | uniq -c | sort -nr | head -10 | while read count ip; do
        if [ "$count" -gt 50 ]; then
            echo "$(date): WARNING: Suspicious IP activity: $ip ($count connections)" >> /var/log/xorb/security-alerts.log
        fi
    done
}

# Function to analyze traffic
analyze_traffic() {
    echo "📊 Analyzing network traffic..."
    
    # Monitor XORB service ports
    for port in 8000 8080 9000; do
        connections=$(netstat -an | grep ":$port " | grep ESTABLISHED | wc -l)
        echo "$(date): Port $port active connections: $connections" >> /var/log/xorb/traffic-analysis.log
    done
}

# Create log directories
mkdir -p /var/log/xorb

# Main monitoring loop
while true; do
    monitor_connections
    detect_anomalies
    analyze_traffic
    sleep 60
done
