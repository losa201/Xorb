# Certificate Authority Docker Image for XORB Platform
# Provides runtime certificate generation and management

FROM alpine:3.18

# Install OpenSSL and required tools
RUN apk add --no-cache \
    openssl \
    bash \
    curl \
    jq \
    ca-certificates

# Create CA user for security
RUN addgroup -g 1001 ca && \
    adduser -D -u 1001 -G ca ca

# Create directories
RUN mkdir -p /run/tls /scripts /var/log/ca && \
    chown -R ca:ca /run/tls /var/log/ca

# Copy CA scripts
COPY scripts/ca/ /scripts/
RUN chmod +x /scripts/*.sh && \
    chown -R ca:ca /scripts

# Create entrypoint script
RUN cat > /entrypoint.sh << 'EOF'
#!/bin/bash
set -euo pipefail

echo "🔐 XORB Certificate Authority starting..."

# Initialize CA if not already done
if [[ ! -f /run/tls/ca/ca.pem ]]; then
    echo "📋 Initializing Certificate Authority..."
    /scripts/make-ca.sh
    
    echo "🏭 Generating service certificates..."
    services=(
        "api:both"
        "orchestrator:client" 
        "agent:both"
        "redis:server"
        "redis-client:client"
        "postgres:server"
        "postgres-client:client"
        "temporal:server"
        "temporal-client:client"
        "dind:server"
        "dind-client:client"
        "scanner:client"
        "prometheus:server"
        "grafana:server"
        "envoy-api:server"
        "envoy-agent:server"
    )
    
    for service_spec in "${services[@]}"; do
        IFS=':' read -r service cert_type <<< "$service_spec"
        echo "📜 Issuing certificate for ${service} (${cert_type})"
        /scripts/issue-cert.sh "$service" "$cert_type"
    done
    
    echo "✅ All certificates generated successfully"
else
    echo "📋 Certificate Authority already initialized"
fi

# Set proper permissions
chown -R ca:ca /run/tls
find /run/tls -name "*.pem" -exec chmod 444 {} \;
find /run/tls -name "*key*" -exec chmod 400 {} \;

echo "🔒 CA initialization complete. Monitoring for certificate requests..."

# Keep container running and monitor for cert requests
while true; do
    sleep 300
    # Check certificate expiration and log warnings
    for cert in /run/tls/*/cert.pem; do
        if [[ -f "$cert" ]]; then
            days_left=$(openssl x509 -in "$cert" -noout -checkend $((7*24*3600)) 2>/dev/null && echo "7+" || echo "7-")
            if [[ "$days_left" == "7-" ]]; then
                service_name=$(basename "$(dirname "$cert")")
                echo "⚠️  Certificate for $service_name expires within 7 days"
            fi
        fi
    done
done
EOF

RUN chmod +x /entrypoint.sh

# Switch to CA user
USER ca

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD test -f /run/tls/ca/ca.pem

ENTRYPOINT ["/entrypoint.sh"]