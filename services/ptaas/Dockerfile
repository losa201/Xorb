FROM python:3.12-slim

# Security hardening
RUN groupadd -r ptaas && useradd -r -g ptaas ptaas

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs && chown -R ptaas:ptaas /app

# Switch to non-root user
USER ptaas

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8010/health || exit 1

# Expose port
EXPOSE 8010

# Run the service
CMD ["python", "ptaas_service.py"]