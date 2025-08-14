# Final Orchestrator Dockerfile
# /root/Xorb/infra/docker/orchestrator.Dockerfile
FROM python:3.12-slim as builder
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry==1.6.1
ENV POETRY_VENV_IN_PROJECT=1

WORKDIR /app
COPY --chown=1001:1001 requirements/orchestrator/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos '' xorb
WORKDIR /app
COPY --from=builder /root/Xorb/src/orchestrator /app
COPY --chown=xorb:xorb src/orchestrator/ ./src/orchestrator/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1

EXPOSE 8001
USER xorb
CMD ["python", "src/orchestrator/main.py"]

# Security hardening
RUN apt-get update && apt-get install -y \
    libcap2-bin \
    && setcap CAP_NET_BIND_SERVICE=+eip /usr/local/bin/python3.12 \
    && apt-get purge -y --auto-remove libcap2-bin \
    && rm -rf /var/lib/apt/lists/*

# Resource limits
RUN mkdir -p /etc/security/limits.d/ && \
    echo "xorb hard nofile 8192\nxorb soft nofile 8192" > /etc/security/limits.d/xorb.conf
