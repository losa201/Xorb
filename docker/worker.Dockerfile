FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies including Temporal
RUN pip install --no-cache-dir -r requirements.txt temporalio

# Copy the entire project structure to maintain imports
COPY . .

# Set Python path to include packages
ENV PYTHONPATH=/app:/app/packages/xorb_core

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash worker
RUN chown -R worker:worker /app
USER worker

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9001/metrics || exit 1

# Default command runs the worker entry point
CMD ["python", "-m", "xorb_core.workflows.worker_entry"]