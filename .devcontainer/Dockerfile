# Development container for Xorb 2.0
# Based on Microsoft's Python devcontainer with additional tools

FROM mcr.microsoft.com/devcontainers/python:3.12-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.1

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        curl \
        wget \
        git \
        vim \
        nano \
        htop \
        tree \
        jq \
        unzip \
        build-essential \
        libpq-dev \
        postgresql-client \
        redis-tools \
        netcat \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && chmod +x kubectl \
    && mv kubectl /usr/local/bin/

# Install Helm
RUN curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install Docker CLI (for Docker-in-Docker)
RUN curl -fsSL https://get.docker.com | sh

# Install Linkerd CLI
RUN curl -sL https://run.linkerd.io/install | sh \
    && mv ~/.linkerd2/bin/linkerd /usr/local/bin/

# Install ArgoCD CLI
RUN curl -sSL -o argocd-linux-amd64 https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-amd64 \
    && chmod +x argocd-linux-amd64 \
    && mv argocd-linux-amd64 /usr/local/bin/argocd

# Install Kustomize
RUN curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash \
    && mv kustomize /usr/local/bin/

# Install Poetry
RUN pip install poetry==${POETRY_VERSION}

# Configure Poetry
RUN poetry config virtualenvs.create false \
    && poetry config virtualenvs.in-project false \
    && poetry config cache-dir /tmp/poetry_cache

# Create development user
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid 1000 -m vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd

# Switch to vscode user
USER vscode

# Set up shell environment
RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc \
    && echo 'export PYTHONPATH="/workspace:$PYTHONPATH"' >> ~/.bashrc

# Install common Python development tools
RUN pip install --user \
    black \
    flake8 \
    mypy \
    isort \
    pytest \
    pytest-asyncio \
    pytest-cov \
    pre-commit \
    structlog \
    httpx

# Create workspace directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Copy requirements and install dependencies (if available)
COPY --chown=vscode:vscode requirements.txt* pyproject.toml* poetry.lock* ./

# Install Python dependencies if pyproject.toml exists
RUN if [ -f pyproject.toml ]; then poetry install --no-dev; fi

# Copy shell scripts
COPY --chown=vscode:vscode .devcontainer/post-*.sh /tmp/
RUN sudo chmod +x /tmp/post-*.sh

# Set default command
CMD ["sleep", "infinity"]