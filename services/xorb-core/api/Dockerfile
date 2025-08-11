# XORB API Service - Multi-stage Production Dockerfile
# Optimized for security, performance, and minimal attack surface

# ============================================================================
# Build Stage - Dependencies compilation and build artifacts
# ============================================================================
FROM python:3.11-slim-bookworm as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
WORKDIR /build
COPY ../../requirements.lock .
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir -r requirements.lock

# ============================================================================
# Runtime Base - Minimal runtime environment
# ============================================================================
FROM python:3.11-slim-bookworm as runtime-base

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    dumb-init \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r xorb && useradd -r -g xorb -d /app -s /bin/false -c "XORB API User" xorb

# Create application directory structure
WORKDIR /app
RUN mkdir -p /app/config /app/logs /app/tmp \
    && chown -R xorb:xorb /app

# ============================================================================
# Development Stage - Full development environment
# ============================================================================
FROM runtime-base as development

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    less \
    htop \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy source code with proper ownership
COPY --chown=xorb:xorb . .
COPY --chown=xorb:xorb ../../config ./config
COPY --chown=xorb:xorb ../../src/common ./src/common

# Development environment variables
ENV ENVIRONMENT=development \
    DEBUG=true \
    LOG_LEVEL=DEBUG \
    PYTHONPATH=/app:/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER xorb

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Development startup command
ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]

# ============================================================================
# Production Stage - Minimal secure runtime
# ============================================================================
FROM runtime-base as production

# Copy only necessary source code
COPY --chown=xorb:xorb app ./app
COPY --chown=xorb:xorb ../../config ./config
COPY --chown=xorb:xorb ../../src/common ./src/common
COPY --chown=xorb:xorb entrypoint.sh ./

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

# Production environment variables
ENV ENVIRONMENT=production \
    DEBUG=false \
    LOG_LEVEL=INFO \
    PYTHONPATH=/app:/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WORKERS=4 \
    MAX_WORKERS=8 \
    TIMEOUT=30 \
    KEEPALIVE=5

# Remove unnecessary files and set strict permissions
RUN find /app -type f -name "*.pyc" -delete \
    && find /app -type d -name "__pycache__" -delete \
    && chmod -R 755 /app \
    && chmod 644 /app/app/*.py /app/config/*.json

# Switch to non-root user
USER xorb

# Health check with timeout
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Security labels
LABEL org.opencontainers.image.title="XORB API Service" \
      org.opencontainers.image.description="XORB Platform API Gateway - Production Container" \
      org.opencontainers.image.vendor="XORB Security" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.source="https://github.com/xorb/platform" \
      security.non-root="true" \
      security.no-new-privileges="true"

# Production startup with dumb-init for proper signal handling
ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["./entrypoint.sh"]

# ============================================================================
# Testing Stage - Unit and integration testing
# ============================================================================
FROM development as testing

# Install additional testing dependencies
RUN pip install --no-cache-dir pytest-xdist pytest-benchmark

# Copy test files
COPY --chown=xorb:xorb tests ./tests

# Set testing environment
ENV ENVIRONMENT=test \
    TESTING=true

# Run tests by default
CMD ["pytest", "-xvs", "--tb=short", "./tests/"]