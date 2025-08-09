
# Multi-stage build for production optimization
FROM python:3.11-slim as base

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development
COPY . .
ENV TEMPORAL_HOST=temporal:7233
CMD ["python", "-u", "main.py"]

# Production stage
FROM base as production
COPY . .
RUN useradd -m -u 1000 xorb && chown -R xorb:xorb /app
USER xorb
CMD ["python", "-u", "main.py"]
