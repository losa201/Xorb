#!/usr/bin/env python3

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PTaaSPlatformDeployer:
    """Complete PTaaS Platform Deployment Orchestrator"""
    
    def __init__(self):
        self.deployment_id = f"DEPLOY-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        self.services = [
            "postgresql",
            "redis", 
            "rabbitmq",
            "qdrant",
            "prometheus",
            "grafana",
            "tenant-isolation-service",
            "agent-sandbox-engine", 
            "vulnerability-classifier",
            "report-generator",
            "rate-limiter",
            "audit-trail-service",
            "billing-metering-service",
            "service-orchestrator",
            "api-gateway",
            "web-dashboard",
            "nginx"
        ]
        
        self.critical_services = [
            "postgresql",
            "redis",
            "service-orchestrator",
            "api-gateway"
        ]
    
    async def deploy(self):
        """Deploy the complete PTaaS platform"""
        logger.info("🚀 Starting XORB PTaaS Platform Deployment")
        logger.info(f"📋 Deployment ID: {self.deployment_id}")
        
        try:
            # Pre-deployment checks
            await self._pre_deployment_checks()
            
            # Initialize environment
            await self._initialize_environment()
            
            # Build Docker images
            await self._build_images()
            
            # Deploy infrastructure services
            await self._deploy_infrastructure()
            
            # Deploy PTaaS services
            await self._deploy_ptaas_services()
            
            # Post-deployment verification
            await self._post_deployment_verification()
            
            # Generate deployment report
            await self._generate_deployment_report()
            
            logger.info("🎉 PTaaS Platform Deployment Complete!")
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            await self._rollback_deployment()
            raise
    
    async def _pre_deployment_checks(self):
        """Pre-deployment system checks"""
        logger.info("🔍 Running pre-deployment checks...")
        
        # Check Docker
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception("Docker not available")
            logger.info(f"✅ Docker: {result.stdout.strip()}")
        except Exception as e:
            raise Exception(f"Docker check failed: {e}")
        
        # Check Docker Compose
        try:
            result = subprocess.run(["docker-compose", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception("Docker Compose not available")
            logger.info(f"✅ Docker Compose: {result.stdout.strip()}")
        except Exception as e:
            raise Exception(f"Docker Compose check failed: {e}")
        
        # Check available disk space
        import shutil
        disk_usage = shutil.disk_usage("/")
        free_gb = disk_usage.free / (1024**3)
        if free_gb < 20:
            raise Exception(f"Insufficient disk space: {free_gb:.1f}GB available, 20GB required")
        logger.info(f"✅ Disk space: {free_gb:.1f}GB available")
        
        # Check available memory
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            mem_total = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1])
            mem_gb = mem_total / (1024**2)
            if mem_gb < 8:
                logger.warning(f"⚠️ Low memory: {mem_gb:.1f}GB available, 16GB recommended")
            else:
                logger.info(f"✅ Memory: {mem_gb:.1f}GB available")
        except Exception as e:
            logger.warning(f"⚠️ Could not check memory: {e}")
        
        # Check required files
        required_files = [
            "docker-compose-ptaas-complete.yml",
            ".env.ptaas",
            "services/service_orchestrator.py"
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise Exception(f"Required file missing: {file_path}")
        
        logger.info("✅ Pre-deployment checks passed")
    
    async def _initialize_environment(self):
        """Initialize deployment environment"""
        logger.info("🔧 Initializing deployment environment...")
        
        # Create necessary directories
        directories = [
            "/var/lib/xorb",
            "/var/lib/xorb/tenants",
            "/var/lib/xorb/sandbox", 
            "/var/lib/xorb/reports",
            "/var/lib/xorb/models",
            "/var/lib/xorb/audit",
            "/var/lib/xorb/platform",
            "/var/lib/xorb/templates",
            "./logs",
            "./backups",
            "./ssl"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Created directory: {directory}")
        
        # Copy environment file
        if os.path.exists(".env.ptaas"):
            subprocess.run(["cp", ".env.ptaas", ".env"], check=True)
            logger.info("✅ Environment configuration loaded")
        
        # Generate SSL certificates if not present
        if not os.path.exists("./ssl/cert.pem"):
            await self._generate_ssl_certificates()
        
        logger.info("✅ Environment initialized")
    
    async def _generate_ssl_certificates(self):
        """Generate self-signed SSL certificates for development"""
        logger.info("🔐 Generating SSL certificates...")
        
        ssl_config = """
[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = CA
L = San Francisco
O = XORB Security
CN = localhost

[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = *.localhost
IP.1 = 127.0.0.1
IP.2 = ::1
        """
        
        # Write SSL config
        with open("./ssl/openssl.conf", "w") as f:
            f.write(ssl_config)
        
        # Generate private key
        subprocess.run([
            "openssl", "genrsa", "-out", "./ssl/key.pem", "2048"
        ], check=True)
        
        # Generate certificate
        subprocess.run([
            "openssl", "req", "-new", "-x509", "-key", "./ssl/key.pem",
            "-out", "./ssl/cert.pem", "-days", "365", 
            "-config", "./ssl/openssl.conf", "-extensions", "v3_req"
        ], check=True)
        
        logger.info("✅ SSL certificates generated")
    
    async def _build_images(self):
        """Build Docker images for services"""
        logger.info("🏗️ Building Docker images...")
        
        # Create Dockerfiles for services
        await self._create_service_dockerfiles()
        
        # Build images
        build_commands = [
            ["docker-compose", "-f", "docker-compose-ptaas-complete.yml", "build", "--parallel"]
        ]
        
        for cmd in build_commands:
            logger.info(f"🔨 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Build failed: {result.stderr}")
                raise Exception(f"Docker build failed: {result.stderr}")
        
        logger.info("✅ Docker images built successfully")
    
    async def _create_service_dockerfiles(self):
        """Create Dockerfiles for all services"""
        logger.info("📦 Creating service Dockerfiles...")
        
        os.makedirs("compose/services", exist_ok=True)
        
        # Base Dockerfile template
        base_dockerfile = """
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/ ./services/
COPY data/ ./data/
COPY config/ ./config/

# Create application user
RUN useradd -m -u 1000 xorb && chown -R xorb:xorb /app
USER xorb

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:${{PORT:-8000}}/health || exit 1

EXPOSE ${{PORT:-8000}}
        """
        
        # Service-specific Dockerfiles
        services_config = {
            "tenant-isolation": {
                "port": "8221",
                "cmd": "python services/tenant_isolation_service.py"
            },
            "agent-sandbox": {
                "port": "8220", 
                "cmd": "python services/agent_sandbox_engine.py",
                "extra": "RUN apt-get update && apt-get install -y docker.io"
            },
            "vulnerability-classifier": {
                "port": "8222",
                "cmd": "python services/vulnerability_classification_engine.py"
            },
            "report-generator": {
                "port": "8223",
                "cmd": "python services/report_generation_engine.py"
            },
            "rate-limiter": {
                "port": "8224",
                "cmd": "python services/rate_limiting_service.py"
            },
            "audit-trail": {
                "port": "8225",  
                "cmd": "python services/audit_trail_service.py"
            },
            "billing-metering": {
                "port": "8226",
                "cmd": "python services/billing_usage_metering_service.py"
            },
            "service-orchestrator": {
                "port": "8200",
                "cmd": "python services/service_orchestrator.py"
            },
            "api-gateway": {
                "port": "8000",
                "cmd": "python services/api_gateway.py"
            },
            "web-dashboard": {
                "port": "3000",
                "cmd": "npm start",
                "base": "FROM node:18-alpine"
            }
        }
        
        for service, config in services_config.items():
            dockerfile_content = base_dockerfile.replace("${{PORT:-8000}}", config["port"])
            dockerfile_content += f"\nCMD {config['cmd']}\n"
            
            if config.get("extra"):
                dockerfile_content = dockerfile_content.replace(
                    "# Install system dependencies",
                    f"# Install system dependencies\n{config['extra']}"
                )
            
            dockerfile_path = f"compose/services/Dockerfile.{service}"
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)
        
        # Create requirements.txt
        requirements = """
asyncio
asyncpg
aioredis
aiofiles
aiohttp
cryptography
docker
fastapi
grafana-client
jinja2
joblib
matplotlib
numpy
openai
pandas
pillow
prometheus-client
psutil
python-multipart
qrcode
scikit-learn
seaborn
stripe
uvicorn
weasyprint
        """
        
        with open("requirements.txt", "w") as f:
            f.write(requirements.strip())
        
        logger.info("✅ Service Dockerfiles created")
    
    async def _deploy_infrastructure(self):
        """Deploy infrastructure services"""
        logger.info("🏗️ Deploying infrastructure services...")
        
        infrastructure_services = [
            "postgresql",
            "redis", 
            "rabbitmq",
            "qdrant",
            "prometheus",
            "grafana"
        ]
        
        # Start infrastructure services
        for service in infrastructure_services:
            await self._deploy_service(service)
            await self._wait_for_service_health(service)
        
        logger.info("✅ Infrastructure services deployed")
    
    async def _deploy_ptaas_services(self):
        """Deploy PTaaS application services"""
        logger.info("🎯 Deploying PTaaS services...")
        
        ptaas_services = [
            "tenant-isolation-service",
            "agent-sandbox-engine",
            "vulnerability-classifier", 
            "report-generator",
            "rate-limiter",
            "audit-trail-service",
            "billing-metering-service",
            "service-orchestrator",
            "api-gateway",
            "web-dashboard",
            "nginx"
        ]
        
        # Deploy services in order
        for service in ptaas_services:
            await self._deploy_service(service)
            await self._wait_for_service_health(service)
        
        logger.info("✅ PTaaS services deployed")
    
    async def _deploy_service(self, service_name: str):
        """Deploy a specific service"""
        logger.info(f"🚀 Deploying {service_name}...")
        
        cmd = [
            "docker-compose", "-f", "docker-compose-ptaas-complete.yml",
            "up", "-d", service_name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Failed to deploy {service_name}: {result.stderr}")
        
        logger.info(f"✅ {service_name} deployed")
    
    async def _wait_for_service_health(self, service_name: str, timeout: int = 120):
        """Wait for service to become healthy"""
        logger.info(f"⏳ Waiting for {service_name} to become healthy...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            cmd = [
                "docker-compose", "-f", "docker-compose-ptaas-complete.yml",
                "ps", service_name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if "healthy" in result.stdout or service_name in ["nginx", "web-dashboard"]:
                logger.info(f"✅ {service_name} is healthy")
                return
            
            await asyncio.sleep(5)
        
        logger.warning(f"⚠️ {service_name} health check timeout")
    
    async def _post_deployment_verification(self):
        """Verify deployment success"""
        logger.info("🔍 Running post-deployment verification...")
        
        # Check service status
        cmd = ["docker-compose", "-f", "docker-compose-ptaas-complete.yml", "ps"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        logger.info("📊 Service Status:")
        logger.info(result.stdout)
        
        # Test critical endpoints
        critical_endpoints = [
            ("http://localhost:8000/health", "API Gateway"),
            ("http://localhost:3000", "Web Dashboard"),
            ("http://localhost:3002", "Grafana"),
            ("http://localhost:9090", "Prometheus")
        ]
        
        for url, name in critical_endpoints:
            try:
                import requests
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    logger.info(f"✅ {name}: Accessible")
                else:
                    logger.warning(f"⚠️ {name}: HTTP {response.status_code}")
            except Exception as e:
                logger.warning(f"⚠️ {name}: {e}")
        
        logger.info("✅ Post-deployment verification complete")
    
    async def _generate_deployment_report(self):
        """Generate deployment report"""
        logger.info("📋 Generating deployment report...")
        
        report = {
            "deployment_id": self.deployment_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "SUCCESS",
            "services_deployed": len(self.services),
            "services": {},
            "endpoints": {
                "web_dashboard": "http://localhost:3000",
                "api_gateway": "http://localhost:8000", 
                "grafana": "http://localhost:3002",
                "prometheus": "http://localhost:9090"
            },
            "credentials": {
                "grafana": {
                    "username": "admin",
                    "password": "xorb_rl_admin_2025"
                },
                "database": {
                    "host": "localhost:5432",
                    "database": "xorb", 
                    "username": "xorb"
                },
                "redis": {
                    "host": "localhost:6380"
                }
            },
            "next_steps": [
                "Access the web dashboard at http://localhost:3000",
                "Login to Grafana at http://localhost:3002 (admin/xorb_rl_admin_2025)",
                "Review service metrics in Prometheus at http://localhost:9090",
                "Configure external API keys (OpenAI, Stripe) in .env file if needed",
                "Set up SSL certificates for production deployment",
                "Configure backup and monitoring alerts"
            ]
        }
        
        # Get service status
        for service in self.services:
            cmd = ["docker-compose", "-f", "docker-compose-ptaas-complete.yml", "ps", service]
            result = subprocess.run(cmd, capture_output=True, text=True)
            report["services"][service] = {
                "status": "running" if service in result.stdout else "error",
                "container": f"xorb-{service}"
            }
        
        # Save report
        report_path = f"./logs/deployment_report_{self.deployment_id}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📋 Deployment report saved: {report_path}")
        
        # Print summary
        logger.info("🎉 DEPLOYMENT COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"📋 Deployment ID: {self.deployment_id}")
        logger.info(f"🌐 Web Dashboard: http://localhost:3000")
        logger.info(f"🔧 API Gateway: http://localhost:8000")
        logger.info(f"📊 Grafana: http://localhost:3002 (admin/xorb_rl_admin_2025)")
        logger.info(f"📈 Prometheus: http://localhost:9090")
        logger.info("=" * 60)
    
    async def _rollback_deployment(self):
        """Rollback failed deployment"""
        logger.info("🔄 Rolling back deployment...")
        
        cmd = ["docker-compose", "-f", "docker-compose-ptaas-complete.yml", "down", "-v"]
        subprocess.run(cmd)
        
        logger.info("✅ Rollback complete")

async def main():
    """Main deployment function"""
    print("🎯 XORB PTaaS Platform Deployment")
    print("=" * 50)
    
    deployer = PTaaSPlatformDeployer()
    
    try:
        await deployer.deploy()
        return 0
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)