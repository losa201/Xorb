# XORB Worker Service - Multi-stage Production Dockerfile
# Background job processing and task execution

# ============================================================================
# Build Stage - Dependencies compilation
# ============================================================================
FROM python:3.11-slim-bookworm as builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
WORKDIR /build
COPY ../../requirements.lock .
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir -r requirements.lock

# ============================================================================
# Runtime Base
# ============================================================================
FROM python:3.11-slim-bookworm as runtime-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    dumb-init \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create user and directories
RUN groupadd -r xorb && useradd -r -g xorb -d /app -s /bin/false xorb
WORKDIR /app
RUN mkdir -p logs tmp && chown -R xorb:xorb /app

# ============================================================================
# Development Stage
# ============================================================================
FROM runtime-base as development

COPY --chown=xorb:xorb . .
COPY --chown=xorb:xorb ../../config ./config
COPY --chown=xorb:xorb ../../src/common ./src/common

ENV ENVIRONMENT=development \
    PYTHONPATH=/app:/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=DEBUG \
    WORKER_CONCURRENCY=4

USER xorb

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app/src'); from common.config_manager import get_config; print('healthy')" || exit 1

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["python", "-u", "worker.py"]

# ============================================================================
# Production Stage
# ============================================================================
FROM runtime-base as production

COPY --chown=xorb:xorb . .
COPY --chown=xorb:xorb ../../config ./config
COPY --chown=xorb:xorb ../../src/common ./src/common

# Production environment variables
ENV ENVIRONMENT=production \
    PYTHONPATH=/app:/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO \
    WORKER_CONCURRENCY=8 \
    WORKER_MAX_MEMORY=512m

# Security hardening
RUN find /app -type f -name "*.pyc" -delete \
    && find /app -type d -name "__pycache__" -delete \
    && chmod -R 755 /app

USER xorb

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import sys; sys.path.append('/app/src'); from common.config_manager import get_config; print('healthy')" || exit 1

# Security labels
LABEL org.opencontainers.image.title="XORB Worker Service" \
      org.opencontainers.image.description="XORB Background Job Processing Service" \
      org.opencontainers.image.vendor="XORB Security" \
      security.non-root="true"

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["python", "-u", "worker.py"]
