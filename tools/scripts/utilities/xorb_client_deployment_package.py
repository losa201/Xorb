#!/usr/bin/env python3
"""
🛡️ XORB Client Deployment Package Generator
Automated packaging system for XORB PRKMT 12.9 Enhanced distribution

This module creates comprehensive deployment packages for different client
environments including enterprise, development, and evaluation deployments.
"""

import os
import sys
import json
import tarfile
import zipfile
import shutil
import logging
import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XORBClientPackager:
    """XORB Client Deployment Package Generator"""

    def __init__(self):
        self.packager_id = f"CLIENT-PACKAGER-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.base_dir = Path.cwd()
        self.packages_dir = self.base_dir / "packages"
        self.packages_dir.mkdir(exist_ok=True)

        # Package configurations
        self.package_configs = {
            "enterprise": {
                "name": "XORB-Enterprise-Edition",
                "description": "Full enterprise deployment with all security features",
                "includes": [
                    "core_engines", "enterprise_apis", "monitoring",
                    "threat_intelligence", "red_team_infra", "documentation"
                ],
                "security_level": "maximum",
                "support_level": "enterprise"
            },
            "development": {
                "name": "XORB-Development-Kit",
                "description": "Development environment with debugging capabilities",
                "includes": [
                    "core_engines", "basic_monitoring", "development_tools",
                    "documentation", "examples"
                ],
                "security_level": "standard",
                "support_level": "community"
            },
            "evaluation": {
                "name": "XORB-Evaluation-Edition",
                "description": "Limited evaluation version for testing and demos",
                "includes": [
                    "core_engines_limited", "basic_monitoring", "documentation"
                ],
                "security_level": "basic",
                "support_level": "documentation"
            },
            "research": {
                "name": "XORB-Research-Platform",
                "description": "Academic and research deployment",
                "includes": [
                    "core_engines", "research_tools", "monitoring",
                    "documentation", "academic_license"
                ],
                "security_level": "high",
                "support_level": "academic"
            }
        }

        logger.info(f"🛡️ XORB Client Packager initialized - ID: {self.packager_id}")

    def create_all_packages(self) -> Dict[str, str]:
        """Create all deployment packages"""
        logger.info("📦 Creating XORB client deployment packages...")

        package_paths = {}

        for package_type, config in self.package_configs.items():
            try:
                package_path = self.create_package(package_type, config)
                package_paths[package_type] = package_path
                logger.info(f"✅ Created {package_type} package: {package_path}")
            except Exception as e:
                logger.error(f"❌ Failed to create {package_type} package: {e}")
                package_paths[package_type] = f"ERROR: {e}"

        return package_paths

    def create_package(self, package_type: str, config: Dict[str, Any]) -> str:
        """Create individual deployment package"""
        logger.info(f"📦 Creating {package_type} package...")

        # Create temporary directory for package assembly
        with tempfile.TemporaryDirectory() as temp_dir:
            package_dir = Path(temp_dir) / config["name"]
            package_dir.mkdir()

            # Create package structure
            self._create_package_structure(package_dir, package_type, config)

            # Copy core files based on package type
            self._copy_package_files(package_dir, package_type, config)

            # Generate package documentation
            self._generate_package_docs(package_dir, package_type, config)

            # Create deployment scripts
            self._create_deployment_scripts(package_dir, package_type, config)

            # Generate package manifest
            self._generate_manifest(package_dir, package_type, config)

            # Create compressed package
            package_path = self._compress_package(package_dir, package_type)

            return package_path

    def _create_package_structure(self, package_dir: Path, package_type: str, config: Dict[str, Any]):
        """Create package directory structure"""
        structure = [
            "bin",           # Executable scripts
            "config",        # Configuration files
            "docs",          # Documentation
            "examples",      # Example configurations
            "scripts",       # Deployment scripts
            "src",           # Source code
            "docker",        # Docker configurations
            "monitoring",    # Monitoring configurations
            "security",      # Security configurations
            "tests"          # Test suites
        ]

        for dir_name in structure:
            (package_dir / dir_name).mkdir(exist_ok=True)

    def _copy_package_files(self, package_dir: Path, package_type: str, config: Dict[str, Any]):
        """Copy files based on package configuration"""
        includes = config["includes"]

        # Core engine files
        if "core_engines" in includes or "core_engines_limited" in includes:
            self._copy_core_engines(package_dir, package_type == "evaluation")

        # Enterprise API files
        if "enterprise_apis" in includes:
            self._copy_enterprise_apis(package_dir)

        # Monitoring components
        if "monitoring" in includes or "basic_monitoring" in includes:
            self._copy_monitoring_files(package_dir, includes)

        # Documentation
        if "documentation" in includes:
            self._copy_documentation(package_dir)

        # Docker configurations
        if "red_team_infra" in includes:
            self._copy_docker_configs(package_dir)

        # Development tools
        if "development_tools" in includes:
            self._copy_development_tools(package_dir)

        # Research tools
        if "research_tools" in includes:
            self._copy_research_tools(package_dir)

    def _copy_core_engines(self, package_dir: Path, limited: bool = False):
        """Copy core XORB engine files"""
        core_files = [
            "XORB_AUTONOMOUS_APT_EMULATION_ENGINE.py",
            "XORB_ZERO_TRUST_BREACH_SIMULATOR.py",
            "XORB_BEHAVIORAL_DRIFT_DETECTION.py",
            "XORB_SYNTHETIC_MALWARE_GENERATOR.py",
            "XORB_PRKMT_12_9_ENHANCED_ORCHESTRATOR.py"
        ]

        src_dir = package_dir / "src"

        for file_name in core_files:
            if self.base_dir.joinpath(file_name).exists():
                if limited and "SYNTHETIC_MALWARE" in file_name:
                    # Create limited version for evaluation
                    self._create_limited_version(
                        self.base_dir / file_name,
                        src_dir / file_name
                    )
                else:
                    shutil.copy2(self.base_dir / file_name, src_dir / file_name)

        # Copy requirements
        if self.base_dir.joinpath("requirements.txt").exists():
            shutil.copy2(self.base_dir / "requirements.txt", src_dir / "requirements.txt")

    def _copy_enterprise_apis(self, package_dir: Path):
        """Copy enterprise API components"""
        api_files = [
            "xorb_enterprise_api.py",
            "xorb_threat_intelligence_feeds.py"
        ]

        src_dir = package_dir / "src"

        for file_name in api_files:
            if self.base_dir.joinpath(file_name).exists():
                shutil.copy2(self.base_dir / file_name, src_dir / file_name)

    def _copy_monitoring_files(self, package_dir: Path, includes: List[str]):
        """Copy monitoring and dashboard files"""
        monitoring_files = ["xorb_tactical_dashboard.py"]

        if "monitoring" in includes:
            # Full monitoring stack
            monitoring_files.extend([
                "config/prometheus.yml",
                "config/grafana_dashboard.json",
                "config/monitoring_config.json"
            ])

        src_dir = package_dir / "src"
        config_dir = package_dir / "config"

        for file_path in monitoring_files:
            source_path = self.base_dir / file_path
            if source_path.exists():
                if file_path.startswith("config/"):
                    dest_path = config_dir / source_path.name
                else:
                    dest_path = src_dir / source_path.name
                shutil.copy2(source_path, dest_path)

    def _copy_documentation(self, package_dir: Path):
        """Copy documentation files"""
        doc_files = [
            "XORB_PRKMT_12_9_DEPLOYMENT_COMPLETE.md",
            "docs/CLAUDE.md",
            "docs/README_COMPLETE.md",
            "docs/OPERATIONAL_GUIDE.md"
        ]

        docs_dir = package_dir / "docs"

        for file_path in doc_files:
            source_path = self.base_dir / file_path
            if source_path.exists():
                shutil.copy2(source_path, docs_dir / source_path.name)

    def _copy_docker_configs(self, package_dir: Path):
        """Copy Docker and infrastructure configurations"""
        docker_files = [
            "docker-compose-redteam-infrastructure.yml",
            "deploy_redteam_infrastructure.sh"
        ]

        docker_dir = package_dir / "docker"
        scripts_dir = package_dir / "scripts"

        for file_name in docker_files:
            source_path = self.base_dir / file_name
            if source_path.exists():
                if file_name.endswith(".yml"):
                    dest_path = docker_dir / file_name
                else:
                    dest_path = scripts_dir / file_name
                shutil.copy2(source_path, dest_path)

                # Make scripts executable
                if file_name.endswith(".sh"):
                    dest_path.chmod(0o755)

    def _copy_development_tools(self, package_dir: Path):
        """Copy development and debugging tools"""
        # Create development helper scripts
        self._create_dev_scripts(package_dir)

        # Copy test configurations
        test_files = []
        for test_file in test_files:
            source_path = self.base_dir / test_file
            if source_path.exists():
                shutil.copy2(source_path, package_dir / "tests" / source_path.name)

    def _copy_research_tools(self, package_dir: Path):
        """Copy research and academic tools"""
        # Create research-specific configurations
        research_config = {
            "research_mode": True,
            "data_collection": True,
            "academic_reporting": True,
            "anonymization": True
        }

        config_path = package_dir / "config" / "research_config.json"
        with open(config_path, 'w') as f:
            json.dump(research_config, f, indent=2)

    def _create_limited_version(self, source_path: Path, dest_path: Path):
        """Create limited version of a file for evaluation"""
        with open(source_path, 'r') as f:
            content = f.read()

        # Add evaluation limitations
        limited_content = f'''#!/usr/bin/env python3
"""
XORB EVALUATION VERSION - LIMITED FUNCTIONALITY
This is a limited evaluation version of {source_path.name}

Full functionality available in licensed versions.
Contact: enterprise@xorb-security.com
"""

# Evaluation limitations applied
EVALUATION_MODE = True
MAX_CAMPAIGNS = 5
MAX_AGENTS = 8
MAX_RUNTIME_MINUTES = 30

{content}

# Evaluation mode restrictions
if EVALUATION_MODE:
    print("⚠️  EVALUATION MODE: Limited to 5 campaigns, 8 agents, 30 minutes runtime")
'''

        with open(dest_path, 'w') as f:
            f.write(limited_content)

    def _create_dev_scripts(self, package_dir: Path):
        """Create development helper scripts"""
        scripts_dir = package_dir / "scripts"

        # Development setup script
        dev_setup = scripts_dir / "dev_setup.sh"
        with open(dev_setup, 'w') as f:
            f.write('''#!/bin/bash
# XORB Development Environment Setup

echo "🛡️ Setting up XORB development environment..."

# Install Python dependencies
pip install -r src/requirements.txt

# Create development configuration
mkdir -p dev_config
cp config/* dev_config/ 2>/dev/null || true

# Setup development database
echo "Setting up development database..."

# Create development Docker network
docker network create xorb-dev 2>/dev/null || true

echo "✅ Development environment ready!"
echo "Run: python src/XORB_PRKMT_12_9_ENHANCED_ORCHESTRATOR.py"
''')
        dev_setup.chmod(0o755)

        # Debug launcher
        debug_launcher = scripts_dir / "debug_launcher.py"
        with open(debug_launcher, 'w') as f:
            f.write('''#!/usr/bin/env python3
"""XORB Debug Launcher"""

import sys
import os
import logging

# Setup debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

if __name__ == "__main__":
    print("🛡️ XORB Debug Launcher")
    print("Available engines:")
    print("1. APT Emulation Engine")
    print("2. Zero Trust Breach Simulator")
    print("3. Behavioral Drift Detection")
    print("4. Enhanced Orchestrator")

    choice = input("Select engine (1-4): ")

    if choice == "1":
        from XORB_AUTONOMOUS_APT_EMULATION_ENGINE import main
        main()
    elif choice == "2":
        from XORB_ZERO_TRUST_BREACH_SIMULATOR import main
        main()
    elif choice == "3":
        from XORB_BEHAVIORAL_DRIFT_DETECTION import main
        main()
    elif choice == "4":
        from XORB_PRKMT_12_9_ENHANCED_ORCHESTRATOR import main
        main()
    else:
        print("Invalid selection")
''')
        debug_launcher.chmod(0o755)

    def _generate_package_docs(self, package_dir: Path, package_type: str, config: Dict[str, Any]):
        """Generate package-specific documentation"""
        docs_dir = package_dir / "docs"

        # Main README
        readme_path = package_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(f'''# {config["name"]}

## Overview
{config["description"]}

**Package Type:** {package_type.title()}
**Security Level:** {config["security_level"].title()}
**Support Level:** {config["support_level"].title()}

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- 8GB+ RAM
- 50GB+ disk space

### Installation

1. **Extract Package:**
   ```bash
   tar -xzf {config["name"]}.tar.gz
   cd {config["name"]}
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r src/requirements.txt
   ```

3. **Configure Environment:**
   ```bash
   cp config/example.env .env
   # Edit .env with your settings
   ```

4. **Deploy XORB:**
   ```bash
   ./scripts/deploy.sh
   ```

### Core Services

| Service | Port | Description |
|---------|------|-------------|
| Tactical Dashboard | 8080 | Main operations interface |
| Enterprise API | 9000 | REST API endpoints |
| Monitoring | 9091 | Prometheus metrics |
| Visualization | 3001 | Grafana dashboards |

### Access URLs
- **Dashboard:** http://localhost:8080
- **API Docs:** http://localhost:9000/api/docs
- **Monitoring:** http://localhost:9091

## Architecture

### Core Components
- **APT Emulation Engine:** Nation-state attack simulation
- **Zero Trust Simulator:** Breach testing framework
- **Drift Detection:** Behavioral anomaly monitoring
- **Orchestrator:** Multi-agent coordination

### Security Features
- Network isolation
- Encrypted communications
- Role-based access control
- Audit logging

## Configuration

See `docs/CONFIGURATION.md` for detailed configuration options.

## Support

**{config["support_level"].title()} Support**
- Documentation: `docs/`
- Examples: `examples/`
- Issue Tracking: GitHub Issues

## Security Notice

⚠️ **This package contains active security testing tools**
- Use only in isolated environments
- Ensure proper network segmentation
- Monitor all activities continuously

## License

XORB PRKMT 12.9 Enhanced - {package_type.title()} Edition
Copyright 2025 XORB Security Platform
''')

        # Installation guide
        install_guide = docs_dir / "INSTALLATION.md"
        with open(install_guide, 'w') as f:
            f.write(self._generate_installation_guide(package_type, config))

        # Configuration guide
        config_guide = docs_dir / "CONFIGURATION.md"
        with open(config_guide, 'w') as f:
            f.write(self._generate_configuration_guide(package_type, config))

    def _generate_installation_guide(self, package_type: str, config: Dict[str, Any]) -> str:
        """Generate detailed installation guide"""
        return f'''# XORB {package_type.title()} Installation Guide

## System Requirements

### Minimum Requirements
- **OS:** Linux (Ubuntu 20.04+, CentOS 8+, RHEL 8+)
- **CPU:** 4 cores
- **RAM:** 8GB
- **Storage:** 50GB available space
- **Network:** Isolated VLAN recommended

### Recommended Requirements
- **OS:** Ubuntu 22.04 LTS
- **CPU:** 8+ cores
- **RAM:** 16GB+
- **Storage:** 100GB+ SSD
- **Network:** Dedicated security lab network

## Pre-Installation Checklist

### Docker Installation
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Python Environment
```bash
# Install Python 3.11+
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv

# Create virtual environment
python3.11 -m venv xorb-env
source xorb-env/bin/activate
```

### Network Configuration
```bash
# Ensure Docker can create custom networks
sudo systemctl enable docker
sudo systemctl start docker

# Verify network capabilities
docker network create test-network
docker network rm test-network
```

## Installation Steps

### 1. Extract and Prepare
```bash
# Extract package
tar -xzf {config["name"]}.tar.gz
cd {config["name"]}

# Set permissions
chmod +x scripts/*.sh
```

### 2. Environment Configuration
```bash
# Copy environment template
cp config/example.env .env

# Edit configuration (required)
nano .env
```

### 3. Dependency Installation
```bash
# Install Python dependencies
pip install -r src/requirements.txt

# Verify installation
python -c "import asyncio, aiohttp, fastapi; print('Dependencies OK')"
```

### 4. Security Setup
```bash
# Generate security keys
./scripts/generate_keys.sh

# Setup SSL certificates (production)
./scripts/setup_ssl.sh
```

### 5. Deploy Services
```bash
# Deploy XORB platform
./scripts/deploy.sh

# Verify deployment
./scripts/health_check.sh
```

## Post-Installation Verification

### Service Health Check
```bash
# Check all services
docker-compose ps

# Check logs
docker-compose logs -f
```

### Connectivity Test
```bash
# Test dashboard
curl http://localhost:8080/api/health

# Test API
curl http://localhost:9000/api/health
```

### Security Validation
```bash
# Verify network isolation
./scripts/network_test.sh

# Check security configuration
./scripts/security_audit.sh
```

## Troubleshooting

### Common Issues

**Port Conflicts:**
```bash
# Check port usage
sudo netstat -tulpn | grep -E "(8080|9000|9091)"

# Modify ports in docker-compose.yml if needed
```

**Memory Issues:**
```bash
# Check available memory
free -h

# Increase Docker memory limit
# Edit /etc/docker/daemon.json
```

**Network Issues:**
```bash
# Reset Docker networks
docker network prune

# Restart Docker
sudo systemctl restart docker
```

## Next Steps

1. **Access Dashboard:** http://localhost:8080
2. **Review Documentation:** `docs/`
3. **Configure Monitoring:** `docs/MONITORING.md`
4. **Setup Integrations:** `docs/INTEGRATIONS.md`

## Support

For {config["support_level"]} support, refer to:
- Documentation in `docs/` directory
- Example configurations in `examples/`
- Troubleshooting guide in `docs/TROUBLESHOOTING.md`
'''

    def _generate_configuration_guide(self, package_type: str, config: Dict[str, Any]) -> str:
        """Generate configuration guide"""
        return f'''# XORB {package_type.title()} Configuration Guide

## Environment Variables

### Core Configuration
```bash
# Basic settings
XORB_MODE={package_type}
XORB_LOG_LEVEL=INFO
XORB_DATA_DIR=/opt/xorb/data

# Security settings
XORB_SECRET_KEY=your-secret-key-here
XORB_JWT_EXPIRY=3600
XORB_ENCRYPTION_KEY=your-encryption-key

# Network configuration
XORB_API_HOST=0.0.0.0
XORB_API_PORT=9000
XORB_DASHBOARD_PORT=8080

# Database settings
XORB_DB_HOST=localhost
XORB_DB_PORT=5432
XORB_DB_NAME=xorb
XORB_DB_USER=xorb
XORB_DB_PASSWORD=secure-password
```

### Advanced Configuration
```bash
# Performance tuning
XORB_MAX_AGENTS=32
XORB_PARALLEL_CAMPAIGNS=10
XORB_WORKER_THREADS=8

# Monitoring
XORB_METRICS_ENABLED=true
XORB_METRICS_PORT=9091
XORB_HEALTH_CHECK_INTERVAL=30

# Security features
XORB_NETWORK_ISOLATION=true
XORB_AUDIT_LOGGING=true
XORB_RATE_LIMITING=true
```

## Service Configuration

### Docker Compose Override
Create `docker-compose.override.yml`:
```yaml
version: '3.8'
services:
  xorb-orchestrator:
    environment:
      - CUSTOM_SETTING=value
    ports:
      - "custom_port:8080"
```

### Custom Networks
```yaml
networks:
  custom-network:
    driver: bridge
    ipam:
      config:
        - subnet: 192.168.100.0/24
```

## Security Configuration

### Authentication
```json
{{
  "auth": {{
    "method": "jwt",
    "secret_key": "your-jwt-secret",
    "token_expiry": 3600,
    "refresh_enabled": true
  }}
}}
```

### Network Security
```json
{{
  "network": {{
    "isolation_level": "high",
    "allowed_networks": ["192.168.0.0/16"],
    "firewall_rules": {{
      "inbound": "deny",
      "outbound": "allow"
    }}
  }}
}}
```

## Monitoring Configuration

### Prometheus Settings
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'xorb-services'
    static_configs:
      - targets: ['localhost:9091']
```

### Grafana Dashboard
```json
{{
  "dashboard": {{
    "title": "XORB Monitoring",
    "refresh": "5s",
    "panels": [
      {{
        "title": "System Health",
        "type": "stat",
        "targets": ["xorb_system_health"]
      }}
    ]
  }}
}}
```

## Integration Configuration

### SIEM Integration
```json
{{
  "siem": {{
    "enabled": true,
    "endpoint": "https://your-siem.com/api",
    "format": "cef",
    "auth_token": "your-token"
  }}
}}
```

### Threat Intelligence Feeds
```json
{{
  "threat_intel": {{
    "feeds": [
      {{
        "name": "abuse.ch",
        "url": "https://bazaar.abuse.ch/export/csv/recent/",
        "format": "csv",
        "update_interval": 3600
      }}
    ]
  }}
}}
```

## Package-Specific Settings

### {package_type.title()} Mode
'''

    def _create_deployment_scripts(self, package_dir: Path, package_type: str, config: Dict[str, Any]):
        """Create deployment scripts"""
        scripts_dir = package_dir / "scripts"

        # Main deployment script
        deploy_script = scripts_dir / "deploy.sh"
        with open(deploy_script, 'w') as f:
            f.write(f'''#!/bin/bash
# XORB {config["name"]} Deployment Script

set -euo pipefail

echo "🛡️ Deploying {config["name"]}..."

# Check prerequisites
./scripts/check_prereqs.sh

# Setup environment
echo "Setting up environment..."
if [ ! -f .env ]; then
    cp config/example.env .env
    echo "⚠️  Please edit .env file before continuing"
    exit 1
fi

# Deploy based on package type
if [ -f docker/docker-compose-redteam-infrastructure.yml ]; then
    echo "Deploying containerized infrastructure..."
    docker-compose -f docker/docker-compose-redteam-infrastructure.yml up -d
fi

# Wait for services
echo "Waiting for services to start..."
sleep 30

# Health check
./scripts/health_check.sh

echo "✅ {config["name"]} deployment complete!"
echo "📊 Dashboard: http://localhost:8080"
echo "🔧 API: http://localhost:9000/api/docs"
''')
        deploy_script.chmod(0o755)

        # Prerequisites check script
        prereq_script = scripts_dir / "check_prereqs.sh"
        with open(prereq_script, 'w') as f:
            f.write('''#!/bin/bash
# Prerequisites Check Script

echo "🔍 Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.11+ first."
    exit 1
fi

# Check memory
MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
if [ $MEMORY_GB -lt 8 ]; then
    echo "⚠️  Warning: Less than 8GB RAM available. XORB may not perform optimally."
fi

# Check disk space
DISK_GB=$(df -BG / | awk 'NR==2{gsub(/G/,"",$4); print $4}')
if [ $DISK_GB -lt 50 ]; then
    echo "⚠️  Warning: Less than 50GB disk space available."
fi

echo "✅ Prerequisites check complete."
''')
        prereq_script.chmod(0o755)

        # Health check script
        health_script = scripts_dir / "health_check.sh"
        with open(health_script, 'w') as f:
            f.write('''#!/bin/bash
# Health Check Script

echo "🏥 Performing health check..."

# Check dashboard
if curl -s http://localhost:8080/api/health > /dev/null; then
    echo "✅ Dashboard: HEALTHY"
else
    echo "❌ Dashboard: NOT RESPONDING"
fi

# Check API
if curl -s http://localhost:9000/api/health > /dev/null; then
    echo "✅ Enterprise API: HEALTHY"
else
    echo "❌ Enterprise API: NOT RESPONDING"
fi

# Check containers
echo "📦 Container Status:"
docker-compose ps

echo "🏥 Health check complete."
''')
        health_script.chmod(0o755)

    def _generate_manifest(self, package_dir: Path, package_type: str, config: Dict[str, Any]):
        """Generate package manifest"""
        manifest = {
            "package": {
                "name": config["name"],
                "type": package_type,
                "version": "12.9-enhanced",
                "description": config["description"],
                "created": datetime.now().isoformat(),
                "packager_id": self.packager_id
            },
            "configuration": {
                "security_level": config["security_level"],
                "support_level": config["support_level"],
                "includes": config["includes"]
            },
            "requirements": {
                "python": "3.11+",
                "docker": "20.10+",
                "docker_compose": "2.0+",
                "memory_gb": 8,
                "disk_gb": 50
            },
            "services": {
                "dashboard": {"port": 8080, "path": "/"},
                "api": {"port": 9000, "path": "/api"},
                "metrics": {"port": 9091, "path": "/metrics"}
            },
            "files": self._generate_file_manifest(package_dir),
            "checksums": self._generate_checksums(package_dir)
        }

        manifest_path = package_dir / "MANIFEST.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    def _generate_file_manifest(self, package_dir: Path) -> Dict[str, List[str]]:
        """Generate file manifest"""
        manifest = {}

        for root, dirs, files in os.walk(package_dir):
            rel_root = Path(root).relative_to(package_dir)
            for file in files:
                file_path = rel_root / file
                category = str(rel_root).split('/')[0] if '/' in str(rel_root) else 'root'

                if category not in manifest:
                    manifest[category] = []
                manifest[category].append(str(file_path))

        return manifest

    def _generate_checksums(self, package_dir: Path) -> Dict[str, str]:
        """Generate file checksums"""
        checksums = {}

        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(package_dir)

                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    checksums[str(rel_path)] = file_hash

        return checksums

    def _compress_package(self, package_dir: Path, package_type: str) -> str:
        """Compress package into distributable archive"""
        package_name = package_dir.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create tar.gz archive
        tar_path = self.packages_dir / f"{package_name}_{package_type}_{timestamp}.tar.gz"

        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(package_dir, arcname=package_name)

        # Create zip archive for Windows compatibility
        zip_path = self.packages_dir / f"{package_name}_{package_type}_{timestamp}.zip"

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(package_dir):
                for file in files:
                    file_path = Path(root) / file
                    arc_path = Path(package_name) / file_path.relative_to(package_dir)
                    zipf.write(file_path, arc_path)

        # Generate checksums for archives
        self._generate_archive_checksums(tar_path, zip_path)

        return str(tar_path)

    def _generate_archive_checksums(self, tar_path: Path, zip_path: Path):
        """Generate checksums for archive files"""
        checksums = {}

        for archive_path in [tar_path, zip_path]:
            with open(archive_path, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
                checksums[archive_path.name] = checksum

        # Save checksums
        checksum_file = tar_path.parent / f"{tar_path.stem}_checksums.txt"
        with open(checksum_file, 'w') as f:
            for filename, checksum in checksums.items():
                f.write(f"{checksum}  {filename}\n")

def main():
    """Generate XORB client deployment packages"""
    logger.info("🛡️ Starting XORB Client Package Generation")

    packager = XORBClientPackager()

    # Create all packages
    package_paths = packager.create_all_packages()

    # Summary report
    logger.info("📦 Package Generation Complete!")
    logger.info("Generated packages:")

    for package_type, path in package_paths.items():
        if path.startswith("ERROR"):
            logger.error(f"  ❌ {package_type}: {path}")
        else:
            logger.info(f"  ✅ {package_type}: {path}")

    # Create distribution summary
    summary = {
        "generation_timestamp": datetime.now().isoformat(),
        "packager_id": packager.packager_id,
        "packages": package_paths,
        "total_packages": len([p for p in package_paths.values() if not p.startswith("ERROR")]),
        "package_directory": str(packager.packages_dir)
    }

    summary_path = packager.packages_dir / "package_generation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"📋 Summary saved: {summary_path}")
    logger.info("🎯 XORB Client Packages Ready for Distribution!")

    return summary

if __name__ == "__main__":
    main()
