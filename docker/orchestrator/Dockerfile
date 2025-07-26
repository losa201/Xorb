FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    build-essential \
    libpq-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install additional orchestrator dependencies
RUN pip install --no-cache-dir \
    temporalio \
    structlog \
    sqlalchemy \
    asyncpg \
    redis \
    prometheus-client \
    psutil \
    asyncio

# Copy application code
COPY packages/ ./packages/
COPY xorb_common/ ./xorb_common/
COPY xorb_core/ ./xorb_core/
COPY services/ ./services/
COPY pyproject.toml README.md ./

# Install in development mode
RUN pip install --no-cache-dir -e .

# Create logs directory
RUN mkdir -p logs

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1002 orchestrator
RUN chown -R orchestrator:orchestrator /app
USER orchestrator

# Set Python path
ENV PYTHONPATH=/app:/app/packages:/app/xorb_common:/app/xorb_core

# Expose port for orchestrator
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || python -c "import sys; sys.exit(0)"

# Start the orchestrator service
CMD ["python", "services/orchestrator/main.py"]