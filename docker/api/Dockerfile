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

# Install additional API dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    prometheus-fastapi-instrumentator \
    structlog \
    openai \
    numpy \
    scikit-learn \
    sqlalchemy \
    alembic \
    asyncpg \
    redis \
    temporalio \
    python-jose[cryptography] \
    passlib[bcrypt] \
    python-multipart \
    pydantic[email] \
    pydantic-settings

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
RUN useradd --create-home --shell /bin/bash --uid 1000 api
RUN chown -R api:api /app
USER api

# Set Python path
ENV PYTHONPATH=/app:/app/packages:/app/xorb_common:/app/xorb_core

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the API service
CMD ["uvicorn", "services.api.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]