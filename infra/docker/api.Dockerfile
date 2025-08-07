# API Service Dockerfile
# Optimized for production deployment

# Build stage
FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN adduser --disabled-password --gecos '' xorb

# Set up Python environment
ENV POETRY_VENV_IN_PROJECT=1
WORKDIR /app

# Install Poetry
RUN pip install poetry==1.6.1

# Copy project files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.in-project true && \
    poetry install --only=main

# Final stage
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN adduser --disabled-password --gecos '' xorb

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /root/.cache/pypoetry /root/.cache/pypoetry
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --chown=xorb:xorb src/api/ ./src/api/

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Switch to non-root user
USER xorb

# Run application
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]