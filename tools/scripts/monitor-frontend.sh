#!/bin/bash
# Frontend monitoring script

SERVICE_NAME="ptaas-frontend"
DOMAIN="verteidiq.com"
PORT="3005"

# Check PM2 process
if ! pm2 show $SERVICE_NAME | grep -q "online"; then
    echo "WARNING: $SERVICE_NAME is not running" | logger -t frontend-monitor
    pm2 restart $SERVICE_NAME
fi

# Check port
if ! netstat -tuln | grep -q ":$PORT "; then
    echo "WARNING: Port $PORT is not listening" | logger -t frontend-monitor
fi

# Check SSL certificate expiry
if openssl x509 -in "/root/Xorb/ssl/verteidiq.crt" -noout -checkend 604800; then
    echo "SSL certificate expires within 7 days" | logger -t frontend-monitor
fi
