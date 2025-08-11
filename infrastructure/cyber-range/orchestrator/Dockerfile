# XORB PTaaS Cyber Range Orchestrator Docker Image
FROM python:3.11-slim

# Set metadata
LABEL maintainer="XORB Security Team" \
      description="XORB PTaaS Cyber Range Orchestrator" \
      version="1.0.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Create non-root user
RUN groupadd -r xorb && useradd -r -g xorb xorb

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    gnupg \
    lsb-release \
    jq \
    iptables \
    iputils-ping \
    net-tools \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

# Install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl \
    && rm kubectl

# Create application directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY orchestrator/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy orchestrator code
COPY orchestrator/cyber-range-orchestrator.py .
COPY scripts/ ./scripts/
COPY firewall/ ./firewall/
COPY k8s/ ./k8s/

# Create necessary directories
RUN mkdir -p /app/config /app/logs /app/scenarios /var/log/cyber-range/campaigns

# Copy configuration files
COPY orchestrator/config/ ./config/ 2>/dev/null || true

# Create default configuration if not provided
RUN echo 'cyber_range:\n  mode: "staging"\n  exercise_duration: "4h"\n  auto_reset: true\n  kill_switch_enabled: true\nsecurity:\n  network_isolation: true\n  traffic_monitoring: true\n  malware_detection: true\n  geographic_restrictions: true\nteams:\n  red_team:\n    namespace: "cyber-range-red"\n    max_concurrent_attacks: 10\n    rate_limiting: true\n  blue_team:\n    namespace: "cyber-range-blue"\n    monitoring_enabled: true\n    alert_thresholds:\n      critical: 5\n      warning: 20\ntargets:\n  auto_restore: true\n  snapshot_interval: "15m"\n  backup_retention: "7d"\nscenarios:\n  available:\n    - "web_app_pentest"\n    - "network_lateral_movement"\n    - "apt_simulation"\n    - "insider_threat"\n    - "ransomware_defense"\n  default_scenario: "web_app_pentest"' > /app/config/config.yaml

# Make scripts executable
RUN chmod +x /app/scripts/*.sh /app/firewall/*.sh

# Set ownership and permissions
RUN chown -R xorb:xorb /app /var/log/cyber-range && \
    chmod 755 /app/cyber-range-orchestrator.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import asyncio; import aiohttp; asyncio.run(aiohttp.ClientSession().get('http://localhost:8080/health').close())" || exit 1

# Switch to non-root user
USER xorb

# Expose ports
EXPOSE 8080 9090

# Set default command
CMD ["python", "cyber-range-orchestrator.py"]