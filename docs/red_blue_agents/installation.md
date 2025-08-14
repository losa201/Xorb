# Installation Guide

This guide provides comprehensive instructions for installing and configuring the XORB Red/Blue Agent Framework across different environments.

##  📋 Prerequisites

###  System Requirements

- **Minimum Requirements**:
- CPU: 4 cores (8 recommended)
- RAM: 8GB (16GB recommended)
- Storage: 50GB SSD (100GB recommended)
- Network: 1Gbps connection

- **Operating System Support**:
- Ubuntu 20.04 LTS or later
- CentOS 8 or later
- RHEL 8 or later
- Debian 11 or later
- macOS 12+ (development only)

###  Software Dependencies

- **Core Dependencies**:
```bash
# Docker Engine 20.10+
docker --version

# Docker Compose 2.0+
docker-compose --version

# Python 3.9+
python3 --version

# Git 2.25+
git --version
```

- **Optional Dependencies**:
```bash
# Kata Containers (enhanced isolation)
kata-runtime --version

# Kubernetes CLI (production deployments)
kubectl version --client

# Terraform (infrastructure as code)
terraform --version

# Helm (Kubernetes package manager)
helm version
```

##  🚀 Quick Installation

###  One-Line Installer

For development and testing environments:

```bash
curl -fsSL https://install.xorb-security.com/agents | sh
```

This script will:
1. Check system requirements
2. Install Docker and Docker Compose if needed
3. Clone the repository
4. Set up Python virtual environment
5. Start infrastructure services
6. Initialize the framework
7. Run basic health checks

###  Manual Installation

For production environments or custom configurations:

```bash
# 1. Clone repository
git clone https://github.com/xorb-security/red-blue-agents.git
cd red-blue-agents

# 2. Run installation script
./scripts/install.sh

# 3. Configure environment
cp .env.example .env
vim .env

# 4. Start services
docker-compose up -d

# 5. Initialize framework
python -m src.services.red_blue_agents.cli init
```

##  🔧 Detailed Installation Steps

###  1. Environment Preparation

####  Ubuntu/Debian Setup

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Log out and back in for group changes to take effect
newgrp docker
```

####  CentOS/RHEL Setup

```bash
# Update system packages
sudo yum update -y

# Install EPEL repository
sudo yum install -y epel-release

# Install required packages
sudo yum install -y \
    curl \
    git \
    python39 \
    python39-pip \
    gcc \
    openssl-devel \
    libffi-devel \
    python39-devel

# Install Docker
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install -y docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

####  macOS Setup

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install git python@3.9

# Install Docker Desktop
brew install --cask docker

# Start Docker Desktop and complete setup
open /Applications/Docker.app
```

###  2. Kata Containers Setup (Optional)

For enhanced security isolation using Kata Containers:

####  Ubuntu/Debian

```bash
# Add Kata repository
sudo apt install -y apt-transport-https ca-certificates gnupg lsb-release
curl -fsSL https://download.opensuse.org/repositories/home:/katacontainers:/releases:/$(lsb_release -rs):/$(lsb_release -is | tr '[:upper:]' '[:lower:]')/Release.key | sudo gpg --dearmor -o /usr/share/keyrings/kata-containers-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/kata-containers-archive-keyring.gpg] https://download.opensuse.org/repositories/home:/katacontainers:/releases:/$(lsb_release -rs):/$(lsb_release -is | tr '[:upper:]' '[:lower:]')/ /" | sudo tee /etc/apt/sources.list.d/kata-containers.list

# Install Kata runtime
sudo apt update
sudo apt install -y kata-containers

# Configure Docker to use Kata runtime
sudo mkdir -p /etc/docker
cat <<EOF | sudo tee /etc/docker/daemon.json
{
    "runtimes": {
        "kata-runtime": {
            "path": "/usr/bin/kata-runtime"
        }
    }
}
EOF

# Restart Docker
sudo systemctl restart docker

# Verify installation
sudo docker run --rm --runtime=kata-runtime hello-world
```

####  CentOS/RHEL

```bash
# Add Kata repository
sudo yum-config-manager --add-repo https://download.opensuse.org/repositories/home:/katacontainers:/releases:/$(cat /etc/os-release | grep VERSION_ID | cut -d'"' -f2):/CentOS_$(cat /etc/os-release | grep VERSION_ID | cut -d'"' -f2)/home:katacontainers:releases:$(cat /etc/os-release | grep VERSION_ID | cut -d'"' -f2):CentOS_$(cat /etc/os-release | grep VERSION_ID | cut -d'"' -f2).repo

# Install Kata runtime
sudo yum install -y kata-containers

# Configure Docker (same as Ubuntu)
# ... rest of configuration
```

###  3. Repository Setup

```bash
# Clone the repository
git clone https://github.com/xorb-security/red-blue-agents.git
cd red-blue-agents

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Set up pre-commit hooks (development)
pre-commit install
```

###  4. Configuration

####  Environment Variables

Create and configure your environment file:

```bash
cp .env.example .env
```

Edit `.env` with your specific configuration:

```bash
# Application Configuration
XORB_ENV=production
DEBUG=false
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql://xorb:secure_password@postgres:5432/xorb_agents
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis Configuration
REDIS_URL=redis://redis:6379/0
REDIS_PASSWORD=secure_redis_password
REDIS_CLUSTER_ENABLED=false

# Temporal Configuration
TEMPORAL_HOST=temporal:7233
TEMPORAL_NAMESPACE=xorb-agents

# Security Configuration
JWT_SECRET=your-super-secure-jwt-secret-key-here
ENCRYPTION_KEY=your-32-character-encryption-key
API_KEY=your-api-key-for-external-access

# Sandbox Configuration
DOCKER_HOST=unix:///var/run/docker.sock
KATA_RUNTIME_ENABLED=false
MAX_SANDBOX_TTL=86400
DEFAULT_SANDBOX_TTL=3600

# Learning Configuration
ML_MODEL_PATH=/app/models
LEARNING_ENABLED=true
AUTO_LEARNING_INTERVAL=3600

# Monitoring Configuration
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
GRAFANA_ENABLED=true
GRAFANA_PORT=3000
GRAFANA_ADMIN_PASSWORD=secure_grafana_password

# External Integrations
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK
EMAIL_SMTP_HOST=smtp.your-domain.com
EMAIL_SMTP_PORT=587
EMAIL_SMTP_USER=alerts@your-domain.com
EMAIL_SMTP_PASSWORD=your-smtp-password

# Compliance and Audit
AUDIT_LOG_ENABLED=true
SOC2_COMPLIANCE=true
GDPR_COMPLIANCE=true
DATA_RETENTION_DAYS=365
```

####  Capability Configuration

Configure technique manifests and environment policies:

```bash
# Copy default configurations
cp -r configs/examples/* src/services/red_blue_agents/configs/

# Customize environment policies
vim src/services/red_blue_agents/configs/environment_policies.json

# Add custom techniques (optional)
vim src/services/red_blue_agents/configs/techniques/custom_techniques.json
```

####  Network Configuration

For production deployments, configure network policies:

```bash
# Create network configuration
cat <<EOF > configs/network.yaml
networks:
  agent_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
    driver_opts:
      com.docker.network.bridge.enable_ip_masquerade: "true"
      com.docker.network.bridge.enable_icc: "false"

  sandbox_network:
    driver: bridge
    internal: true
    ipam:
      config:
        - subnet: 172.21.0.0/16
EOF
```

###  5. Infrastructure Services

####  Start Core Services

```bash
# Start infrastructure services
docker-compose up -d postgres redis temporal

# Wait for services to be ready
./scripts/wait-for-services.sh

# Verify services
docker-compose ps
```

####  Database Initialization

```bash
# Run database migrations
python -m alembic upgrade head

# Create initial data
python -m src.services.red_blue_agents.cli init-db

# Verify database
psql $DATABASE_URL -c "SELECT COUNT(*) FROM techniques;"
```

####  Redis Initialization

```bash
# Test Redis connection
redis-cli -u $REDIS_URL ping

# Initialize Redis with default data
python -m src.services.red_blue_agents.cli init-redis

# Verify Redis
redis-cli -u $REDIS_URL keys "*"
```

###  6. Application Services

####  Start Agent Framework

```bash
# Start all agent framework services
docker-compose up -d

# Check service health
curl http://localhost:8000/health

# View logs
docker-compose logs -f agent-scheduler
```

####  Service Configuration

Configure individual services:

```yaml
# docker-compose.override.yml
version: '3.8'

services:
  agent-scheduler:
    environment:
      - SCHEDULER_MAX_MISSIONS=50
      - SCHEDULER_CLEANUP_INTERVAL=300
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G

  sandbox-orchestrator:
    environment:
      - MAX_SANDBOXES=100
      - SANDBOX_CLEANUP_INTERVAL=60
    privileged: true
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 4G
```

###  7. Monitoring Setup

####  Prometheus Configuration

```bash
# Create Prometheus configuration
mkdir -p monitoring/prometheus
cat <<EOF > monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'agent-framework'
    static_configs:
      - targets:
        - agent-scheduler:8080
        - sandbox-orchestrator:8081
        - telemetry-collector:8082

  - job_name: 'infrastructure'
    static_configs:
      - targets:
        - postgres-exporter:9187
        - redis-exporter:9121
        - node-exporter:9100

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
EOF
```

####  Grafana Dashboards

```bash
# Import pre-built dashboards
./scripts/import-dashboards.sh

# Or manually copy dashboards
cp -r monitoring/grafana/dashboards/* /var/lib/grafana/dashboards/
```

####  Start Monitoring Stack

```bash
# Start monitoring services
docker-compose -f docker-compose.monitoring.yml up -d

# Access Grafana
open http://localhost:3000
# Default login: admin / admin (change on first login)
```

##  🧪 Testing Installation

###  Health Checks

```bash
# Run comprehensive health check
./scripts/health-check.sh

# Check individual services
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health

# Check database connectivity
python -c "
import asyncpg
import asyncio

async def test_db():
    conn = await asyncpg.connect('$DATABASE_URL')
    result = await conn.fetchval('SELECT 1')
    await conn.close()
    print(f'Database test: {result}')

asyncio.run(test_db())
"

# Check Redis connectivity
redis-cli -u $REDIS_URL ping
```

###  Integration Tests

```bash
# Run integration test suite
pytest tests/integration/ -v

# Run specific test categories
pytest tests/integration/test_scheduler.py -v
pytest tests/integration/test_sandbox.py -v
pytest tests/integration/test_agents.py -v

# Run end-to-end tests
pytest tests/e2e/ -v --slow
```

###  Sample Mission

```bash
# Create a test mission
python -m src.services.red_blue_agents.cli create-mission \
  --config configs/examples/test_mission.json

# Monitor mission progress
python -m src.services.red_blue_agents.cli list-missions

# View mission results
python -m src.services.red_blue_agents.cli get-mission <mission_id>
```

##  🐳 Container Images

###  Pre-built Images

Pull official images from Docker Hub:

```bash
# Core services
docker pull xorb/agent-scheduler:latest
docker pull xorb/sandbox-orchestrator:latest
docker pull xorb/telemetry-collector:latest

# Agent images
docker pull xorb/red-recon:latest
docker pull xorb/red-exploit:latest
docker pull xorb/blue-detect:latest
docker pull xorb/blue-hunt:latest

# Infrastructure
docker pull xorb/postgres-with-extensions:latest
docker pull redis:7-alpine
docker pull temporalio/auto-setup:latest
```

###  Build Custom Images

```bash
# Build all images
./scripts/build-images.sh

# Build specific service
docker build -t xorb/agent-scheduler:custom -f docker/Dockerfile.scheduler .

# Build agent images
docker build -t xorb/red-recon:custom -f docker/agents/Dockerfile.red-recon .
docker build -t xorb/blue-detect:custom -f docker/agents/Dockerfile.blue-detect .

# Push to registry
docker push your-registry.com/xorb/agent-scheduler:custom
```

###  Image Security Scanning

```bash
# Scan images for vulnerabilities
./scripts/scan-images.sh

# Or manually scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image xorb/agent-scheduler:latest

# Generate SBOM (Software Bill of Materials)
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  anchore/syft xorb/agent-scheduler:latest
```

##  ☸️ Kubernetes Deployment

###  Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://get.helm.sh/helm-v3.10.0-linux-amd64.tar.gz | tar -xzO linux-amd64/helm | sudo mv /usr/local/bin/

# Add XORB Helm repository
helm repo add xorb https://charts.xorb-security.com
helm repo update
```

###  Deploy with Helm

```bash
# Create namespace
kubectl create namespace xorb-agents

# Install chart
helm install xorb-agents xorb/red-blue-agents \
  --namespace xorb-agents \
  --values values.production.yaml

# Check deployment status
kubectl get pods -n xorb-agents
kubectl get services -n xorb-agents

# Port forward for local access
kubectl port-forward -n xorb-agents svc/agent-scheduler 8000:8000
```

###  Custom Values

```yaml
# values.production.yaml
global:
  environment: production
  registry: your-registry.com/xorb

agentScheduler:
  replicaCount: 3
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 2000m
      memory: 4Gi

sandboxOrchestrator:
  replicaCount: 2
  nodeSelector:
    sandbox-node: "true"

postgresql:
  enabled: true
  persistence:
    size: 100Gi
    storageClass: fast-ssd

redis:
  enabled: true
  cluster:
    enabled: true
    slaveCount: 2

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
    ingress:
      enabled: true
      host: grafana.your-domain.com

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: agents.your-domain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: agents-tls
      hosts:
        - agents.your-domain.com
```

##  🔧 Troubleshooting

###  Common Issues

####  Docker Permission Denied

```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Or run with sudo (not recommended for production)
sudo docker-compose up -d
```

####  Port Conflicts

```bash
# Check what's using ports
sudo netstat -tulpn | grep :8000
sudo lsof -i :8000

# Change ports in docker-compose.yml
sed -i 's/8000:8000/8001:8000/' docker-compose.yml
```

####  Database Connection Issues

```bash
# Check database logs
docker-compose logs postgres

# Test connection manually
docker-compose exec postgres psql -U xorb -d xorb_agents -c "SELECT 1;"

# Reset database
docker-compose down -v
docker-compose up -d postgres
./scripts/init-database.sh
```

####  Out of Memory Issues

```bash
# Check memory usage
docker stats

# Increase Docker memory limit (Docker Desktop)
# Settings > Resources > Memory > 8GB

# Add swap space (Linux)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

###  Log Analysis

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f agent-scheduler

# Filter logs by level
docker-compose logs | grep ERROR
docker-compose logs | grep WARNING

# Export logs for analysis
docker-compose logs --no-color > application.log

# Real-time log monitoring
tail -f application.log | grep -E "(ERROR|FATAL)"
```

###  Performance Tuning

```bash
# Monitor resource usage
docker stats
htop

# Optimize PostgreSQL
echo "shared_preload_libraries = 'pg_stat_statements'" >> postgresql.conf
echo "max_connections = 200" >> postgresql.conf
echo "shared_buffers = 256MB" >> postgresql.conf

# Optimize Redis
echo "maxmemory 2gb" >> redis.conf
echo "maxmemory-policy allkeys-lru" >> redis.conf

# Tune container resources
docker update --memory=2g --cpus=2 container_name
```

###  Backup and Recovery

```bash
# Backup database
docker-compose exec postgres pg_dump -U xorb xorb_agents > backup.sql

# Backup Redis
docker-compose exec redis redis-cli BGSAVE
docker-compose exec redis cp /data/dump.rdb /backup/

# Backup configuration
tar -czf config-backup.tar.gz configs/ .env

# Restore database
docker-compose exec -T postgres psql -U xorb xorb_agents < backup.sql

# Restore Redis
docker-compose exec redis cp /backup/dump.rdb /data/
docker-compose restart redis
```

##  📱 Management Tools

###  CLI Tool

```bash
# Install CLI tool globally
pip install xorb-agents-cli

# Or use from source
python -m src.services.red_blue_agents.cli --help

# Common commands
xorb-agents status                    # System status
xorb-agents missions list             # List missions
xorb-agents missions create config.json  # Create mission
xorb-agents agents list               # List agents
xorb-agents sandbox list              # List sandboxes
xorb-agents logs tail                 # Tail logs
```

###  Web UI

```bash
# Enable web UI in docker-compose.yml
web-ui:
  image: xorb/agents-web-ui:latest
  ports:
    - "3000:3000"
  environment:
    - API_URL=http://agent-scheduler:8000

# Access at http://localhost:3000
```

###  Mobile App

Download the XORB Agents mobile app for iOS/Android:
- Monitor mission status
- Receive push notifications
- View real-time dashboards
- Emergency stop capabilities

##  🎓 Next Steps

After successful installation:

1. **Read the [Architecture Guide](./architecture.md)** to understand system design
2. **Follow the [Configuration Guide](./configuration.md)** for advanced setup
3. **Try the [Quickstart Tutorial](./quickstart.md)** to run your first mission
4. **Explore [Agent Documentation](./agents/)** to understand capabilities
5. **Set up [Monitoring](./operations/monitoring.md)** for production use
6. **Configure [Security](./security/)** according to your requirements

##  📞 Support

If you encounter issues during installation:

1. Check the [Troubleshooting Guide](./operations/troubleshooting.md)
2. Search [GitHub Issues](https://github.com/xorb-security/red-blue-agents/issues)
3. Join our [Discord Community](https://discord.gg/xorb-security)
4. Contact [Enterprise Support](mailto:enterprise@xorb-security.com) for business customers

- --

- **⚡ Pro Tip**: Use the `--dry-run` flag with installation scripts to preview changes before applying them.