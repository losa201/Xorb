#!/usr/bin/env python3

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMPTaaSDeploymentManager:
    """Comprehensive deployment manager for LLM PTaaS platform"""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.deployment_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.deployment_id = f"LLM_PTAAS_DEPLOY_{self.deployment_timestamp}"

        # Service configurations
        self.services = {
            'core_services': [
                'advanced-penetration-testing-engine',
                'ai-driven-vulnerability-assessment',
                'vulnerability-logging-metrics',
                'automated-red-team-simulation'
            ],
            'llm_services': [
                'llm-powered-pentest-agents',
                'multi-step-exploit-orchestrator',
                'llm-vulnerability-integration'
            ],
            'infrastructure': [
                'redis-llm-cache',
                'prometheus-llm',
                'grafana-llm'
            ],
            'api_gateway': [
                'llm-ptaas-gateway'
            ]
        }

        # Deployment configuration
        self.deployment_config = {
            'docker_compose_file': 'infra/docker-compose-llm-ptaas.yml',
            'environment_files': [
                'infra/config/.env.production.secure',
                'infra/config/.env.development'
            ],
            'data_directories': [
                'data',
                'data/agent_memory',
                'data/swarm_state',
                'data/vector_embeddings',
                'logs'
            ],
            'backup_directories': [
                'backups/llm_ptaas_pre_deploy',
                f'backups/llm_ptaas_{self.deployment_timestamp}'
            ]
        }

        self.deployment_status = {
            'phase': 'initialization',
            'services_deployed': [],
            'services_failed': [],
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'success': False,
            'errors': []
        }

    def log_deployment_step(self, step: str, status: str, details: str = ""):
        """Log deployment step with status"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        logger.info(f"[{timestamp}] {step}: {status} {details}")

        # Also write to deployment log file
        log_file = self.base_dir / 'logs' / f'llm_ptaas_deployment_{self.deployment_timestamp}.log'
        log_file.parent.mkdir(exist_ok=True)

        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {step}: {status} {details}\n")

    def create_backup(self) -> bool:
        """Create backup of existing deployment"""
        self.log_deployment_step("BACKUP", "STARTING", "Creating pre-deployment backup")

        try:
            backup_dir = self.base_dir / self.deployment_config['backup_directories'][1]
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Backup existing services
            if (self.base_dir / 'services').exists():
                subprocess.run([
                    'tar', '-czf',
                    str(backup_dir / 'services_backup.tar.gz'),
                    '-C', str(self.base_dir),
                    'services'
                ], check=True)

            # Backup existing data
            if (self.base_dir / 'data').exists():
                subprocess.run([
                    'tar', '-czf',
                    str(backup_dir / 'data_backup.tar.gz'),
                    '-C', str(self.base_dir),
                    'data'
                ], check=True)

            # Backup existing config
            if (self.base_dir / 'config').exists():
                subprocess.run([
                    'tar', '-czf',
                    str(backup_dir / 'config_backup.tar.gz'),
                    '-C', str(self.base_dir),
                    'config'
                ], check=True)

            self.log_deployment_step("BACKUP", "SUCCESS", f"Backup created at {backup_dir}")
            return True

        except Exception as e:
            self.log_deployment_step("BACKUP", "FAILED", f"Error: {e}")
            self.deployment_status['errors'].append(f"Backup failed: {e}")
            return False

    def prepare_environment(self) -> bool:
        """Prepare deployment environment"""
        self.log_deployment_step("ENVIRONMENT", "PREPARING", "Setting up directories and permissions")

        try:
            # Create necessary directories
            for directory in self.deployment_config['data_directories']:
                dir_path = self.base_dir / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                self.log_deployment_step("DIRECTORY", "CREATED", f"{directory}")

            # Set permissions
            os.chmod(self.base_dir / 'data', 0o755)
            os.chmod(self.base_dir / 'logs', 0o755)

            # Create environment file if not exists
            env_file = self.base_dir / 'infra/config/.env.llm.ptaas'
            if not env_file.exists():
                with open(env_file, 'w') as f:
                    f.write(f"""# LLM PTaaS Environment Configuration
# Generated: {datetime.now().isoformat()}

# API Keys (Configure these for production)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Service Configuration
DEPLOYMENT_ID={self.deployment_id}
DEPLOYMENT_TIMESTAMP={self.deployment_timestamp}

# Redis Configuration
REDIS_URL=redis://redis-llm-cache:6379

# Monitoring Configuration
PROMETHEUS_RETENTION=30d
GRAFANA_ADMIN_PASSWORD=xorb_llm_admin_2025

# Security Configuration
JWT_SECRET_KEY=your_jwt_secret_key_here
API_RATE_LIMIT=1000

# Database Configuration
SQLITE_DATA_PATH=/app/data
""")

            self.log_deployment_step("ENVIRONMENT", "SUCCESS", "Environment prepared")
            return True

        except Exception as e:
            self.log_deployment_step("ENVIRONMENT", "FAILED", f"Error: {e}")
            self.deployment_status['errors'].append(f"Environment preparation failed: {e}")
            return False

    def validate_docker_environment(self) -> bool:
        """Validate Docker environment"""
        self.log_deployment_step("DOCKER", "VALIDATING", "Checking Docker installation")

        try:
            # Check Docker
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception("Docker not installed or not accessible")

            self.log_deployment_step("DOCKER", "VALIDATED", f"Docker version: {result.stdout.strip()}")

            # Check Docker Compose
            result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception("Docker Compose not installed or not accessible")

            self.log_deployment_step("DOCKER_COMPOSE", "VALIDATED", f"Docker Compose version: {result.stdout.strip()}")

            # Check Docker daemon
            result = subprocess.run(['docker', 'info'], capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception("Docker daemon not running")

            self.log_deployment_step("DOCKER_DAEMON", "VALIDATED", "Docker daemon is running")
            return True

        except Exception as e:
            self.log_deployment_step("DOCKER", "FAILED", f"Error: {e}")
            self.deployment_status['errors'].append(f"Docker validation failed: {e}")
            return False

    def stop_existing_services(self) -> bool:
        """Stop existing services gracefully"""
        self.log_deployment_step("SERVICES", "STOPPING", "Stopping existing services")

        try:
            # Stop existing docker-compose services
            compose_file = self.base_dir / self.deployment_config['docker_compose_file']
            if compose_file.exists():
                subprocess.run([
                    'docker-compose', '-f', str(compose_file), 'down'
                ], cwd=self.base_dir)

            # Also try to stop any other XORB containers
            result = subprocess.run([
                'docker', 'ps', '-a', '--filter', 'name=xorb', '--format', '{{.Names}}'
            ], capture_output=True, text=True)

            if result.stdout.strip():
                container_names = result.stdout.strip().split('\n')
                for container in container_names:
                    subprocess.run(['docker', 'stop', container], capture_output=True)
                    subprocess.run(['docker', 'rm', container], capture_output=True)
                    self.log_deployment_step("CONTAINER", "STOPPED", f"{container}")

            self.log_deployment_step("SERVICES", "STOPPED", "All existing services stopped")
            return True

        except Exception as e:
            self.log_deployment_step("SERVICES", "STOP_FAILED", f"Error: {e}")
            # Continue with deployment even if stop fails
            return True

    def build_and_deploy_services(self) -> bool:
        """Build and deploy all services"""
        self.log_deployment_step("DEPLOYMENT", "STARTING", "Building and deploying services")

        try:
            compose_file = self.base_dir / self.deployment_config['docker_compose_file']

            # Build all services
            self.log_deployment_step("BUILD", "STARTING", "Building Docker images")
            build_result = subprocess.run([
                'docker-compose', '-f', str(compose_file), 'build', '--no-cache'
            ], cwd=self.base_dir, capture_output=True, text=True)

            if build_result.returncode != 0:
                raise Exception(f"Build failed: {build_result.stderr}")

            self.log_deployment_step("BUILD", "SUCCESS", "All images built successfully")

            # Start infrastructure services first
            self.log_deployment_step("INFRASTRUCTURE", "STARTING", "Starting infrastructure services")
            infra_services = ' '.join(self.services['infrastructure'])
            subprocess.run([
                'docker-compose', '-f', str(compose_file), 'up', '-d'
            ] + self.services['infrastructure'], cwd=self.base_dir, check=True)

            # Wait for infrastructure to be ready
            self.log_deployment_step("INFRASTRUCTURE", "WAITING", "Waiting for infrastructure services")
            time.sleep(30)

            # Start core services
            self.log_deployment_step("CORE_SERVICES", "STARTING", "Starting core services")
            subprocess.run([
                'docker-compose', '-f', str(compose_file), 'up', '-d'
            ] + self.services['core_services'], cwd=self.base_dir, check=True)

            # Wait for core services
            time.sleep(20)

            # Start LLM services
            self.log_deployment_step("LLM_SERVICES", "STARTING", "Starting LLM services")
            subprocess.run([
                'docker-compose', '-f', str(compose_file), 'up', '-d'
            ] + self.services['llm_services'], cwd=self.base_dir, check=True)

            # Wait for LLM services
            time.sleep(20)

            # Start API Gateway
            self.log_deployment_step("API_GATEWAY", "STARTING", "Starting API Gateway")
            subprocess.run([
                'docker-compose', '-f', str(compose_file), 'up', '-d'
            ] + self.services['api_gateway'], cwd=self.base_dir, check=True)

            self.log_deployment_step("DEPLOYMENT", "SUCCESS", "All services deployed")
            return True

        except Exception as e:
            self.log_deployment_step("DEPLOYMENT", "FAILED", f"Error: {e}")
            self.deployment_status['errors'].append(f"Deployment failed: {e}")
            return False

    def verify_services_health(self) -> bool:
        """Verify that all services are healthy"""
        self.log_deployment_step("HEALTH_CHECK", "STARTING", "Verifying service health")

        service_endpoints = {
            'advanced-penetration-testing-engine': 'http://localhost:8008/health',
            'ai-driven-vulnerability-assessment': 'http://localhost:8009/health',
            'vulnerability-logging-metrics': 'http://localhost:8010/health',
            'automated-red-team-simulation': 'http://localhost:8011/health',
            'llm-powered-pentest-agents': 'http://localhost:8012/health',
            'multi-step-exploit-orchestrator': 'http://localhost:8013/health',
            'llm-vulnerability-integration': 'http://localhost:8014/health',
            'llm-ptaas-gateway': 'http://localhost:8000/health'
        }

        healthy_services = []
        failed_services = []

        # Wait for services to start
        self.log_deployment_step("HEALTH_CHECK", "WAITING", "Waiting for services to initialize")
        time.sleep(45)

        for service_name, endpoint in service_endpoints.items():
            try:
                import requests
                response = requests.get(endpoint, timeout=10)
                if response.status_code == 200:
                    healthy_services.append(service_name)
                    self.log_deployment_step("HEALTH_CHECK", "HEALTHY", f"{service_name}")
                else:
                    failed_services.append(service_name)
                    self.log_deployment_step("HEALTH_CHECK", "UNHEALTHY", f"{service_name} - HTTP {response.status_code}")
            except Exception as e:
                failed_services.append(service_name)
                self.log_deployment_step("HEALTH_CHECK", "FAILED", f"{service_name} - {e}")

        self.deployment_status['services_deployed'] = healthy_services
        self.deployment_status['services_failed'] = failed_services

        success_rate = len(healthy_services) / len(service_endpoints)
        self.log_deployment_step("HEALTH_CHECK", "COMPLETED", f"Success rate: {success_rate:.1%} ({len(healthy_services)}/{len(service_endpoints)})")

        return success_rate >= 0.8  # 80% success rate threshold

    def create_deployment_report(self) -> None:
        """Create deployment report"""
        self.deployment_status['end_time'] = datetime.now().isoformat()

        report = {
            'deployment_id': self.deployment_id,
            'deployment_timestamp': self.deployment_timestamp,
            'status': self.deployment_status,
            'services_deployed': len(self.deployment_status['services_deployed']),
            'services_failed': len(self.deployment_status['services_failed']),
            'success_rate': len(self.deployment_status['services_deployed']) / (len(self.deployment_status['services_deployed']) + len(self.deployment_status['services_failed'])) if (self.deployment_status['services_deployed'] or self.deployment_status['services_failed']) else 0,
            'endpoints': {
                'api_gateway': 'http://localhost:8000',
                'grafana_dashboard': 'http://localhost:3001',
                'prometheus_metrics': 'http://localhost:9090',
                'swagger_docs': 'http://localhost:8000/docs'
            },
            'credentials': {
                'grafana_admin': 'admin / xorb_llm_admin_2025'
            }
        }

        # Save report
        report_file = self.base_dir / 'logs' / f'llm_ptaas_deployment_report_{self.deployment_timestamp}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.log_deployment_step("REPORT", "CREATED", f"Deployment report saved to {report_file}")

        # Print summary
        logger.info("=" * 80)
        logger.info("LLM PTaaS DEPLOYMENT SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Deployment ID: {self.deployment_id}")
        logger.info(f"Success Rate: {report['success_rate']:.1%}")
        logger.info(f"Services Deployed: {report['services_deployed']}")
        logger.info(f"Services Failed: {report['services_failed']}")
        logger.info("")
        logger.info("ENDPOINTS:")
        for name, url in report['endpoints'].items():
            logger.info(f"  {name}: {url}")
        logger.info("")
        logger.info("CREDENTIALS:")
        for service, cred in report['credentials'].items():
            logger.info(f"  {service}: {cred}")
        logger.info("=" * 80)

    def deploy(self) -> bool:
        """Execute full deployment"""
        logger.info(f"Starting LLM PTaaS Platform Deployment: {self.deployment_id}")

        # Phase 1: Preparation
        self.deployment_status['phase'] = 'preparation'
        if not self.create_backup():
            return False

        if not self.prepare_environment():
            return False

        if not self.validate_docker_environment():
            return False

        # Phase 2: Service Deployment
        self.deployment_status['phase'] = 'deployment'
        if not self.stop_existing_services():
            return False

        if not self.build_and_deploy_services():
            return False

        # Phase 3: Verification
        self.deployment_status['phase'] = 'verification'
        if not self.verify_services_health():
            self.log_deployment_step("DEPLOYMENT", "PARTIAL_SUCCESS", "Some services failed health checks")
            self.deployment_status['success'] = False
        else:
            self.log_deployment_step("DEPLOYMENT", "SUCCESS", "All services healthy")
            self.deployment_status['success'] = True

        # Phase 4: Reporting
        self.deployment_status['phase'] = 'completed'
        self.create_deployment_report()

        return self.deployment_status['success']

def main():
    """Main deployment function"""

    # Check if running as root/admin (recommended for Docker operations)
    if os.geteuid() != 0:
        logger.warning("Not running as root. You may encounter permission issues with Docker.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Initialize deployment manager
    deployment_manager = LLMPTaaSDeploymentManager()

    # Execute deployment
    success = deployment_manager.deploy()

    if success:
        logger.info("🎉 LLM PTaaS Platform deployment completed successfully!")
        logger.info("🚀 Platform is ready for advanced penetration testing with LLM agents")
        sys.exit(0)
    else:
        logger.error("❌ LLM PTaaS Platform deployment failed or partially failed")
        logger.error("📋 Check the deployment logs for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
