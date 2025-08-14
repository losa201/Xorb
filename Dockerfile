# Multi-stage Dockerfile for XORB Platform
# Supports development, production, and secure deployment modes
ARG BUILD_MODE=production

# Base stage with common dependencies
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    libpq-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development
ENV ENVIRONMENT=development
COPY . .
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "src.api.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production
ENV ENVIRONMENT=production

# Create non-root user
RUN groupadd -r xorb && useradd --no-log-init -r -g xorb xorb \
    && mkdir -p /app/logs /app/data \
    && chown -R xorb:xorb /app

# Copy application code
COPY --chown=xorb:xorb . .

# Set Python path
ENV PYTHONPATH=/app/src

# Switch to non-root user
USER xorb

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "src.api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Secure stage - hardened for production
FROM python:3.11-slim as secure

# Security labels
LABEL maintainer="xorb-security-team" \
      version="1.0.0" \
      description="XORB API - Secure Production Build" \
      security.scan.required="true" \
      security.non-root="true"

# Install minimal runtime packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libmagic1 \
    curl \
    tini \
    ca-certificates \
    && apt-get upgrade -y \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create application user with specific UID/GID
RUN groupadd -r -g 1000 xorb && \
    useradd --no-log-init -r -g xorb -u 1000 -m -d /home/xorb xorb && \
    mkdir -p /app /app/logs /app/data /app/tmp && \
    chown -R xorb:xorb /app /home/xorb

# Security environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/src" \
    PATH="/home/xorb/.local/bin:$PATH" \
    USER=xorb \
    HOME=/home/xorb \
    TMPDIR=/app/tmp \
    PYTHONSAFEPATH=1 \
    PYTHONHASHSEED=random

WORKDIR /app

# Copy requirements and install as xorb user
COPY --chown=xorb:xorb requirements.txt .
USER xorb
RUN pip install --user --no-warn-script-location -r requirements.txt

# Copy application code with proper ownership
COPY --chown=xorb:xorb . /app/

# Security: Remove development files and set strict permissions
RUN find /app -name "*.pyc" -delete && \
    find /app -name "__pycache__" -type d -exec rm -rf {} + && \
    find /app -name ".git*" -delete 2>/dev/null || true && \
    find /app -name "test_*" -delete 2>/dev/null || true && \
    chmod 700 /app/logs /app/data /app/tmp

# Security: Health check with timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f -A "HealthCheck/1.0" --max-time 10 --connect-timeout 5 \
        http://localhost:8000/health || exit 1

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

EXPOSE 8000
CMD ["uvicorn", "src.api.app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--access-log", \
     "--log-level", "info"]

# Volume for logs and data (external mount)
VOLUME ["/app/logs", "/app/data"]
