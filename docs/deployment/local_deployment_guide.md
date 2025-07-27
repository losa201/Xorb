# XORB Local Deployment Guide

Complete guide for deploying XORB Autonomous Security Platform in on-premises, edge, and airgapped environments.

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Hardware Support](#hardware-support)
4. [Quick Start](#quick-start)
5. [Detailed Installation](#detailed-installation)
6. [Configuration](#configuration)
7. [Monitoring & Management](#monitoring--management)
8. [Security Hardening](#security-hardening)
9. [Troubleshooting](#troubleshooting)
10. [Offline/Airgapped Deployment](#offlineairgapped-deployment)

## Overview

XORB 2.0 provides autonomous security intelligence capabilities designed for local deployment scenarios including:

- **On-Premises Deployment**: Full enterprise deployment on dedicated hardware
- **Edge Computing**: Optimized deployment for edge devices like Raspberry Pi 5
- **Airgapped Environments**: Completely offline deployment with no cloud dependencies
- **Hybrid Environments**: Mixed on-prem/edge with secure coordination

### Key Features

- ✅ **Zero Cloud Dependencies**: Fully autonomous operation
- ✅ **Hardware Auto-Detection**: Automatic optimization for different platforms
- ✅ **Container-Based**: Docker/Docker Compose deployment
- ✅ **Systemd Integration**: Persistent service management
- ✅ **Real-time Monitoring**: Prometheus + Grafana stack
- ✅ **Security Hardening**: Built-in security controls and audit logging

## System Requirements

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 2 cores | 4+ cores |
| **Memory** | 4GB RAM | 8+ GB RAM |
| **Storage** | 20GB free | 50+ GB free |
| **OS** | Linux (Ubuntu 20.04+, CentOS 8+) | Ubuntu 22.04 LTS |
| **Docker** | 20.0+ | Latest stable |
| **Docker Compose** | 1.28+ | Latest stable |

### Production Requirements

| Component | EPYC Server | Workstation | Edge Device |
|-----------|-------------|-------------|-------------|
| **CPU** | 32+ cores | 8+ cores | 4 cores |
| **Memory** | 64+ GB | 16+ GB | 8 GB |
| **Storage** | 500+ GB SSD | 100+ GB SSD | 64+ GB |
| **Network** | 10Gb Ethernet | 1Gb Ethernet | WiFi/Ethernet |

## Hardware Support

### Supported Architectures

- **x86_64 (AMD64)**: Intel/AMD processors
- **ARM64 (AArch64)**: Raspberry Pi 5, Apple Silicon, ARM servers
- **ARMv7**: Older ARM devices (limited support)

### Tested Platforms

#### Edge Devices
- **Raspberry Pi 5** (16GB): Optimized edge profile
- **NVIDIA Jetson**: GPU-accelerated processing
- **Intel NUC**: Compact workstation deployment

#### Servers
- **AMD EPYC 7702**: High-performance server optimization
- **Intel Xeon**: Enterprise server deployment
- **AWS Graviton**: ARM-based cloud instances

#### Workstations
- **Standard x86_64**: Desktop/laptop deployment
- **Apple Silicon**: M1/M2 Mac development

## Quick Start

### Automated Deployment

The fastest way to deploy XORB locally:

```bash
# Clone the repository
git clone https://github.com/your-org/xorb.git
cd xorb

# Run automated deployment
./autodeploy.sh

# Or using make
make bootstrap-local
```

The `autodeploy.sh` script will:
1. Detect your hardware configuration
2. Install missing dependencies
3. Generate optimized configuration
4. Deploy all XORB services
5. Set up monitoring and systemd services

### Manual Quick Start

```bash
# Check prerequisites
make check-config

# Start services
make up

# Monitor deployment
make monitor

# View status
make status
```

## Detailed Installation

### Step 1: System Preparation

#### Ubuntu/Debian
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y docker.io docker-compose git python3 python3-pip curl

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker
```

#### CentOS/RHEL
```bash
# Install Docker
sudo yum install -y docker docker-compose git python3 python3-pip

# Start services
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

### Step 2: Hardware Detection

Run hardware detection to see optimization profile:

```bash
./autodeploy.sh --check-only
```

Example output:
```
Hardware detection complete:
  Architecture: arm64
  CPU Cores: 4
  Memory: 8GB
  Device Type: raspberry_pi_5
  Optimization Profile: edge_optimized
```

### Step 3: Configuration Generation

Generate hardware-optimized configuration:

```bash
./autodeploy.sh --config-only
```

This creates:
- `config/local/.xorb.env`: Environment variables
- `config/local/prometheus.yml`: Monitoring configuration
- `docker-compose.local.yml`: Optimized container settings
- `systemd/`: Service files for persistence

### Step 4: Service Deployment

Deploy all XORB services:

```bash
make deploy-local
```

Or manually:
```bash
docker-compose -f docker-compose.local.yml up -d
```

### Step 5: Service Installation (Optional)

Install systemd services for automatic startup:

```bash
make install-systemd
```

This enables:
- `xorb-orchestrator.service`: Main XORB services
- `xorb-watchdog.service`: Self-healing monitoring
- `xorb-agents.target`: Security agents

## Configuration

### Environment Variables

Key configuration options in `config/local/.xorb.env`:

```bash
# Hardware configuration
ARCH=arm64
CPU_CORES=4
MEMORY_GB=8
DEVICE_TYPE=raspberry_pi_5
OPTIMIZATION_PROFILE=edge_optimized

# Service ports
XORB_API_PORT=8000
XORB_ORCHESTRATOR_PORT=8080
GRAFANA_PORT=3000
PROMETHEUS_PORT=9090

# Performance tuning
MAX_CONCURRENT_AGENTS=4
ORCHESTRATOR_WORKERS=2
CACHE_SIZE_MB=256
MEMORY_LIMIT_API=512m
CPU_LIMIT_API=1.0
```

### Optimization Profiles

#### Edge Optimized (Raspberry Pi 5)
```bash
MAX_CONCURRENT_AGENTS=4
ORCHESTRATOR_WORKERS=2
CACHE_SIZE_MB=256
MEMORY_LIMIT_API=512m
ENABLE_COMPRESSION=true
```

#### Server Optimized (EPYC)
```bash
MAX_CONCURRENT_AGENTS=32
ORCHESTRATOR_WORKERS=16
CACHE_SIZE_MB=2048
MEMORY_LIMIT_API=4g
NUMA_ENABLED=true
```

#### Workstation Optimized
```bash
MAX_CONCURRENT_AGENTS=8
ORCHESTRATOR_WORKERS=4
CACHE_SIZE_MB=1024
MEMORY_LIMIT_API=2g
```

### Database Configuration

PostgreSQL, Redis, Neo4j, and Qdrant are automatically configured with appropriate resource limits based on the optimization profile.

## Monitoring & Management

### Interactive Status Monitor

Launch the real-time status dashboard:

```bash
make monitor
# or
./scripts/monitor/status_tui.py
```

Features:
- Real-time container status
- System resource usage
- Recent log messages
- Service health checks

### Management Commands

```bash
# Service management
make start          # Start all services
make stop           # Stop all services
make restart        # Restart all services
make status         # Show service status

# Monitoring
make health-check   # Comprehensive health check
make metrics        # Show Prometheus metrics
make logs           # View all logs
make logs-api       # API service logs only

# Maintenance
make backup-data    # Backup XORB data
make clean          # Clean up old logs/images
make update         # Update XORB services
```

### Web Interfaces

Once deployed, access XORB through these interfaces:

- **API Documentation**: http://localhost:8000/docs
- **Grafana Dashboard**: http://localhost:3000 (admin/xorb_admin)
- **Prometheus Metrics**: http://localhost:9090
- **Neo4j Browser**: http://localhost:7474

### Systemd Management

If systemd services are installed:

```bash
# System service control
sudo systemctl start xorb-orchestrator
sudo systemctl stop xorb-orchestrator
sudo systemctl status xorb-orchestrator

# View logs
sudo journalctl -u xorb-orchestrator -f
sudo journalctl -u xorb-watchdog -f

# Enable/disable autostart
sudo systemctl enable xorb-orchestrator
sudo systemctl disable xorb-orchestrator
```

## Security Hardening

### Automated Hardening

Run the security hardening script:

```bash
make harden-host
```

This applies:
- Container security policies
- Network access controls
- Audit logging configuration
- File permission hardening

### Manual Security Steps

#### Container Security
```bash
# Enable AppArmor (Ubuntu)
sudo aa-enforce /etc/apparmor.d/docker

# Enable SELinux (CentOS/RHEL)
sudo setsebool -P container_manage_cgroup true
```

#### Network Security
```bash
# Configure firewall
sudo ufw allow 8000/tcp  # API
sudo ufw allow 3000/tcp  # Grafana
sudo ufw allow 9090/tcp  # Prometheus
sudo ufw enable
```

#### Audit Logging
```bash
# Enable audit logging
sudo systemctl enable auditd
sudo systemctl start auditd

# Configure XORB audit rules
sudo cp config/audit/xorb.rules /etc/audit/rules.d/
sudo systemctl restart auditd
```

### Security Scanning

Run security scans on the deployment:

```bash
make security-scan
```

## Troubleshooting

### Common Issues

#### Services Won't Start

**Problem**: Containers fail to start
```bash
# Check Docker status
sudo systemctl status docker

# Check container logs
docker-compose -f docker-compose.local.yml logs

# Check system resources
free -h
df -h
```

**Solution**: Ensure sufficient resources and correct configuration

#### Port Conflicts

**Problem**: Port already in use
```bash
# Check port usage
sudo netstat -tlnp | grep :8000

# Kill conflicting process
sudo kill -9 <PID>
```

**Solution**: Update port configuration in `.xorb.env`

#### Permission Errors

**Problem**: Permission denied errors
```bash
# Fix Docker permissions
sudo usermod -aG docker $USER
newgrp docker

# Fix file permissions
sudo chown -R $USER:$USER .
chmod +x scripts/*.sh
```

#### Memory Issues (Edge Devices)

**Problem**: Out of memory on Raspberry Pi
```bash
# Check memory usage
free -h

# Restart with lower limits
docker-compose -f docker-compose.local.yml down
# Edit .xorb.env to reduce memory limits
docker-compose -f docker-compose.local.yml up -d
```

### Log Analysis

#### Container Logs
```bash
# All services
make logs

# Specific service
docker-compose -f docker-compose.local.yml logs xorb-api

# Follow logs in real-time
docker-compose -f docker-compose.local.yml logs -f
```

#### System Logs
```bash
# XORB systemd services
sudo journalctl -u "xorb-*" -f

# System messages
sudo journalctl -p err -f

# Docker daemon logs
sudo journalctl -u docker -f
```

### Performance Optimization

#### Resource Monitoring
```bash
# CPU and memory usage
htop

# Docker stats
docker stats

# Disk usage
ncdu /
```

#### Optimization Tips

1. **Memory Optimization**:
   - Reduce `CACHE_SIZE_MB` for edge devices
   - Lower memory limits for containers
   - Enable compression for edge profiles

2. **CPU Optimization**:
   - Reduce `MAX_CONCURRENT_AGENTS` on low-end hardware
   - Adjust `ORCHESTRATOR_WORKERS` based on CPU cores
   - Enable NUMA for EPYC servers

3. **Storage Optimization**:
   - Use SSD storage when possible
   - Enable log rotation
   - Regular cleanup of old data

## Offline/Airgapped Deployment

### Creating Offline Bundle

For environments without internet access, create an offline deployment bundle:

```bash
# Create OCI bundle with all required images
python3 scripts/pack_oci_bundle.py --output-dir deployment

# Include sample data (optional)
python3 scripts/pack_oci_bundle.py --include-data --output-dir deployment
```

This creates:
- `deployment/xorb_oci_bundle_YYYYMMDD_HHMMSS.tar.gz`
- `deployment/xorb_oci_bundle_YYYYMMDD_HHMMSS.tar.gz.sha256`

### Bundle Contents

- All Docker images (PostgreSQL, Redis, Neo4j, Qdrant, Prometheus, Grafana, XORB services)
- Configuration templates
- Installation scripts
- Documentation
- Systemd service files
- Offline installer script

### Airgapped Installation

1. **Transfer Bundle**:
   ```bash
   # Copy bundle to target system via USB/secure transfer
   scp xorb_oci_bundle_*.tar.gz user@target-host:/tmp/
   ```

2. **Extract Bundle**:
   ```bash
   cd /tmp
   tar -xzf xorb_oci_bundle_*.tar.gz
   cd xorb_oci_bundle_*
   ```

3. **Run Offline Installer**:
   ```bash
   ./install_xorb_offline.sh
   ```

4. **Verify Installation**:
   ```bash
   docker-compose -f docker-compose.local.yml ps
   make health-check
   ```

### USB Installation

For completely disconnected systems:

1. Create bundle on connected system
2. Copy bundle to USB drive
3. Transfer USB to airgapped system
4. Extract and run installer

## Advanced Configuration

### Multi-Node Deployment

For distributed edge deployments:

```bash
# Configure node coordination
export XORB_NODE_TYPE=coordinator  # or worker
export XORB_CLUSTER_NODES="node1.local,node2.local,node3.local"
export XORB_NODE_ID=$(hostname)

# Deploy with cluster configuration
make deploy-local
```

### Custom Agents

Add custom security agents:

```python
# Place in packages/xorb_core/xorb_core/agents/custom/
class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.capabilities = [AgentCapability.CUSTOM]
        self.name = "CustomAgent"
    
    async def execute(self, target: str, **kwargs):
        # Custom logic here
        return {"status": "success", "findings": []}
```

### Integration APIs

XORB provides REST APIs for integration:

```bash
# Campaign management
curl -X POST http://localhost:8000/api/v1/campaigns \
  -H "Content-Type: application/json" \
  -d '{"name": "Security Assessment", "targets": ["192.168.1.0/24"]}'

# Agent status
curl http://localhost:8000/api/v1/agents

# Metrics endpoint
curl http://localhost:8000/metrics
```

## Support and Documentation

### Additional Resources

- **API Documentation**: http://localhost:8000/docs (when deployed)
- **Source Code**: GitHub repository
- **Issue Tracking**: GitHub Issues
- **Configuration Reference**: `CLAUDE.md`

### Community Support

- **Discussions**: GitHub Discussions
- **Bug Reports**: GitHub Issues
- **Feature Requests**: GitHub Issues

### Professional Support

For enterprise deployments and professional support, contact the XORB team.

---

**Note**: This deployment guide covers local/on-premises deployment scenarios. For cloud deployment options, refer to the main deployment documentation.