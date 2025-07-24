FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from root project
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt uvicorn

# Copy the entire project structure to maintain imports
COPY . .

# Set Python path to include packages
ENV PYTHONPATH=/app:/app/packages/xorb_core

# Expose API port
EXPOSE 8000

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash api
RUN chown -R api:api /app
USER api

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the API
CMD ["uvicorn", "services.api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
