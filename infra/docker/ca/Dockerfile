# Certificate Authority Service Dockerfile
# Simple Alpine container for CA operations

FROM alpine:3.19

# Install OpenSSL and required tools
RUN apk add --no-cache \
    openssl \
    bash \
    curl \
    ca-certificates

# Create CA user
RUN addgroup -g 1000 ca && \
    adduser -u 1000 -G ca -s /bin/bash -D ca

# Create directories
RUN mkdir -p /run/tls/ca && \
    chown -R ca:ca /run/tls

# Switch to CA user
USER ca

# Health check to verify CA certificate exists
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD test -f /run/tls/ca/ca.pem || exit 1

# Default command to keep container running
CMD ["tail", "-f", "/dev/null"]