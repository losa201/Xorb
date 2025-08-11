#!/usr/bin/env python3
"""
XORB Production Deployment Script
Automated deployment and validation for production environments
"""

import os
import sys
import time
import json
import subprocess
import argparse
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'deployment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ProductionDeployer:
    """
    Production deployment orchestrator for XORB platform
    """
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.deployment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.project_root = Path(__file__).parent.parent
        self.deployment_config = self._load_deployment_config()
        self.health_checks = []
        
        logger.info(f"Initializing deployment {self.deployment_id} for {environment}")
    
    def _load_deployment_config(self) -> Dict:
        """Load deployment configuration"""
        config_file = self.project_root / f"config/{self.environment}.yaml"
        
        if config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "database": {
                "host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", "5432")),
                "name": os.getenv("DB_NAME", "xorb"),
                "user": os.getenv("DB_USER", "xorb"),
            },
            "redis": {
                "host": os.getenv("REDIS_HOST", "localhost"),
                "port": int(os.getenv("REDIS_PORT", "6379")),
            },
            "services": {
                "api": {"port": 8000, "replicas": 3},
                "orchestrator": {"port": 8001, "replicas": 2},
                "worker": {"replicas": 5}
            },
            "monitoring": {
                "prometheus_port": 9090,
                "grafana_port": 3000
            }
        }
    
    def run_command(self, command: List[str], cwd: Path = None) -> Tuple[bool, str, str]:
        """Execute shell command and return result"""
        try:
            logger.info(f"Executing: {' '.join(command)}")
            result = subprocess.run(
                command,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info(f"Command succeeded: {' '.join(command)}")
                return True, result.stdout, result.stderr
            else:
                logger.error(f"Command failed: {' '.join(command)}")
                logger.error(f"Exit code: {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                return False, result.stdout, result.stderr
                
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(command)}")
            return False, "", "Command timed out"
        except Exception as e:
            logger.error(f"Command error: {e}")
            return False, "", str(e)
    
    def check_prerequisites(self) -> bool:
        """Validate deployment prerequisites"""
        logger.info("🔍 Checking deployment prerequisites...")
        
        prerequisites = [
            ("docker", ["docker", "--version"]),
            ("docker-compose", ["docker-compose", "--version"]),
            ("python", ["python3", "--version"]),
            ("git", ["git", "--version"]),
        ]
        
        for name, command in prerequisites:
            success, stdout, stderr = self.run_command(command)
            if not success:
                logger.error(f"❌ {name} not available")
                return False
            logger.info(f"✅ {name}: {stdout.strip()}")
        
        # Check Docker daemon
        success, _, _ = self.run_command(["docker", "info"])
        if not success:
            logger.error("❌ Docker daemon not running")
            return False
        
        logger.info("✅ All prerequisites satisfied")
        return True
    
    def setup_environment(self) -> bool:
        """Setup deployment environment"""
        logger.info("🛠️ Setting up deployment environment...")
        
        try:
            # Create required directories
            directories = [
                "logs", "data", "backups", "monitoring/prometheus",
                "monitoring/grafana", "ssl", "uploads"
            ]
            
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created directory: {directory}")
            
            # Set proper permissions
            data_dir = self.project_root / "data"
            os.chmod(data_dir, 0o755)
            
            # Copy environment configuration
            env_template = self.project_root / ".env.template"
            env_production = self.project_root / ".env.production"
            
            if env_template.exists() and not env_production.exists():
                import shutil
                shutil.copy(env_template, env_production)
                logger.info("✅ Created production environment file")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment setup failed: {e}")
            return False
    
    def backup_existing_deployment(self) -> bool:
        """Backup existing deployment if present"""
        logger.info("💾 Creating deployment backup...")
        
        try:
            backup_dir = self.project_root / f"backups/backup_{self.deployment_id}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup database
            db_config = self.deployment_config["database"]
            backup_file = backup_dir / "database_backup.sql"
            
            pg_dump_cmd = [
                "pg_dump",
                f"postgresql://{db_config['user']}@{db_config['host']}:{db_config['port']}/{db_config['name']}",
                "-f", str(backup_file),
                "--format=custom",
                "--no-password"
            ]
            
            success, _, stderr = self.run_command(pg_dump_cmd)
            if success:
                logger.info(f"✅ Database backup created: {backup_file}")
            else:
                logger.warning(f"⚠️ Database backup failed: {stderr}")
            
            # Backup current configuration
            config_backup = backup_dir / "config"
            config_backup.mkdir(exist_ok=True)
            
            import shutil
            for config_file in self.project_root.glob("*.env*"):
                shutil.copy(config_file, config_backup)
            
            logger.info(f"✅ Configuration backup created: {config_backup}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            return False
    
    def build_container_images(self) -> bool:
        """Build Docker container images"""
        logger.info("🐳 Building container images...")
        
        images = [
            ("xorb-api", "src/api/Dockerfile", "src/api"),
            ("xorb-orchestrator", "src/orchestrator/Dockerfile", "src/orchestrator"),
            ("xorb-worker", "src/services/worker/Dockerfile", "src/services/worker"),
        ]
        
        for image_name, dockerfile, context in images:
            logger.info(f"Building {image_name}...")
            
            build_cmd = [
                "docker", "build",
                "-t", f"{image_name}:latest",
                "-t", f"{image_name}:{self.deployment_id}",
                "-f", dockerfile,
                context
            ]
            
            success, stdout, stderr = self.run_command(build_cmd)
            if not success:
                logger.error(f"❌ Failed to build {image_name}: {stderr}")
                return False
            
            logger.info(f"✅ Built {image_name}")
        
        return True
    
    def run_security_scan(self) -> bool:
        """Run security scan on built images"""
        logger.info("🔒 Running security scans...")
        
        try:
            # Install Trivy if not present
            trivy_check = self.run_command(["trivy", "--version"])
            if not trivy_check[0]:
                logger.info("Installing Trivy...")
                install_cmd = [
                    "curl", "-sfL",
                    "https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh",
                    "|", "sh", "-s", "--", "-b", "/usr/local/bin"
                ]
                self.run_command(install_cmd)
            
            # Scan each image
            images = ["xorb-api:latest", "xorb-orchestrator:latest", "xorb-worker:latest"]
            
            for image in images:
                logger.info(f"Scanning {image}...")
                scan_cmd = [
                    "trivy", "image",
                    "--format", "json",
                    "--output", f"security_scan_{image.replace(':', '_')}.json",
                    image
                ]
                
                success, stdout, stderr = self.run_command(scan_cmd)
                if not success:
                    logger.warning(f"⚠️ Security scan failed for {image}: {stderr}")
                else:
                    logger.info(f"✅ Security scan completed for {image}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Security scan failed: {e}")
            return False
    
    def deploy_infrastructure(self) -> bool:
        """Deploy infrastructure components"""
        logger.info("🏗️ Deploying infrastructure...")
        
        try:
            # Start core infrastructure
            infrastructure_services = [
                "docker-compose.production.yml"
            ]
            
            for compose_file in infrastructure_services:
                if (self.project_root / compose_file).exists():
                    logger.info(f"Starting services from {compose_file}")
                    
                    deploy_cmd = [
                        "docker-compose",
                        "-f", compose_file,
                        "up", "-d",
                        "--remove-orphans"
                    ]
                    
                    success, stdout, stderr = self.run_command(deploy_cmd)
                    if not success:
                        logger.error(f"❌ Failed to deploy {compose_file}: {stderr}")
                        return False
                    
                    logger.info(f"✅ Deployed {compose_file}")
            
            # Wait for services to start
            logger.info("⏳ Waiting for services to initialize...")
            time.sleep(30)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Infrastructure deployment failed: {e}")
            return False
    
    def run_database_migrations(self) -> bool:
        """Run database migrations"""
        logger.info("📊 Running database migrations...")
        
        try:
            # Check if alembic is available
            migration_dir = self.project_root / "src/api/migrations"
            if not migration_dir.exists():
                logger.warning("⚠️ No migrations directory found")
                return True
            
            # Run migrations
            migrate_cmd = [
                "docker-compose", "-f", "docker-compose.production.yml",
                "exec", "-T", "api",
                "alembic", "upgrade", "head"
            ]
            
            success, stdout, stderr = self.run_command(migrate_cmd)
            if not success:
                logger.error(f"❌ Database migration failed: {stderr}")
                return False
            
            logger.info("✅ Database migrations completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration error: {e}")
            return False
    
    def validate_deployment(self) -> bool:
        """Validate deployment health"""
        logger.info("🩺 Validating deployment...")
        
        # Health check endpoints
        health_checks = [
            ("API Health", "http://localhost:8000/api/v1/health"),
            ("API Readiness", "http://localhost:8000/api/v1/readiness"),
            ("API Info", "http://localhost:8000/api/v1/info"),
        ]
        
        import requests
        
        for check_name, url in health_checks:
            try:
                logger.info(f"Checking {check_name}...")
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    logger.info(f"✅ {check_name}: OK")
                    self.health_checks.append({
                        "name": check_name,
                        "status": "healthy",
                        "response_time": response.elapsed.total_seconds(),
                        "details": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text[:100]
                    })
                else:
                    logger.error(f"❌ {check_name}: HTTP {response.status_code}")
                    self.health_checks.append({
                        "name": check_name,
                        "status": "unhealthy",
                        "error": f"HTTP {response.status_code}"
                    })
                    
            except Exception as e:
                logger.error(f"❌ {check_name}: {e}")
                self.health_checks.append({
                    "name": check_name,
                    "status": "error",
                    "error": str(e)
                })
        
        # Check if all critical services are healthy
        critical_checks = ["API Health", "API Readiness"]
        healthy_criticals = [
            check for check in self.health_checks 
            if check["name"] in critical_checks and check["status"] == "healthy"
        ]
        
        if len(healthy_criticals) == len(critical_checks):
            logger.info("✅ All critical services are healthy")
            return True
        else:
            logger.error("❌ Critical services are not healthy")
            return False
    
    def setup_monitoring(self) -> bool:
        """Setup monitoring and observability"""
        logger.info("📊 Setting up monitoring...")
        
        try:
            # Start monitoring stack
            monitoring_compose = self.project_root / "docker-compose.monitoring.yml"
            if monitoring_compose.exists():
                monitor_cmd = [
                    "docker-compose",
                    "-f", "docker-compose.monitoring.yml",
                    "up", "-d"
                ]
                
                success, stdout, stderr = self.run_command(monitor_cmd)
                if not success:
                    logger.warning(f"⚠️ Monitoring setup failed: {stderr}")
                    return False
                
                logger.info("✅ Monitoring stack deployed")
                
                # Wait for Prometheus and Grafana to start
                time.sleep(20)
                
                # Validate monitoring endpoints
                monitoring_checks = [
                    ("Prometheus", f"http://localhost:{self.deployment_config['monitoring']['prometheus_port']}/-/healthy"),
                    ("Grafana", f"http://localhost:{self.deployment_config['monitoring']['grafana_port']}/api/health"),
                ]
                
                import requests
                
                for service, url in monitoring_checks:
                    try:
                        response = requests.get(url, timeout=10)
                        if response.status_code == 200:
                            logger.info(f"✅ {service} monitoring: OK")
                        else:
                            logger.warning(f"⚠️ {service} monitoring: HTTP {response.status_code}")
                    except Exception as e:
                        logger.warning(f"⚠️ {service} monitoring check failed: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Monitoring setup failed: {e}")
            return False
    
    def generate_deployment_report(self) -> bool:
        """Generate deployment report"""
        logger.info("📄 Generating deployment report...")
        
        try:
            report = {
                "deployment_id": self.deployment_id,
                "environment": self.environment,
                "timestamp": datetime.now().isoformat(),
                "configuration": self.deployment_config,
                "health_checks": self.health_checks,
                "status": "success" if all(check.get("status") == "healthy" for check in self.health_checks if check["name"] in ["API Health", "API Readiness"]) else "failed"
            }
            
            # Save report
            report_file = self.project_root / f"deployment_report_{self.deployment_id}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Generate summary
            summary_file = self.project_root / f"deployment_summary_{self.deployment_id}.txt"
            with open(summary_file, 'w') as f:
                f.write(f"XORB Production Deployment Report\n")
                f.write(f"=================================\n\n")
                f.write(f"Deployment ID: {self.deployment_id}\n")
                f.write(f"Environment: {self.environment}\n")
                f.write(f"Status: {report['status'].upper()}\n")
                f.write(f"Timestamp: {report['timestamp']}\n\n")
                
                f.write("Health Checks:\n")
                for check in self.health_checks:
                    status_icon = "✅" if check["status"] == "healthy" else "❌"
                    f.write(f"  {status_icon} {check['name']}: {check['status']}\n")
                
                f.write(f"\nFull report: {report_file}\n")
            
            logger.info(f"✅ Deployment report generated: {report_file}")
            logger.info(f"✅ Deployment summary: {summary_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            return False
    
    def rollback_deployment(self) -> bool:
        """Rollback deployment if needed"""
        logger.info("🔄 Rolling back deployment...")
        
        try:
            # Stop current services
            rollback_cmd = [
                "docker-compose",
                "-f", "docker-compose.production.yml",
                "down"
            ]
            
            success, stdout, stderr = self.run_command(rollback_cmd)
            if not success:
                logger.error(f"❌ Rollback failed: {stderr}")
                return False
            
            logger.info("✅ Deployment rolled back")
            return True
            
        except Exception as e:
            logger.error(f"❌ Rollback error: {e}")
            return False
    
    def deploy(self) -> bool:
        """Execute complete deployment process"""
        logger.info(f"🚀 Starting XORB production deployment {self.deployment_id}")
        
        deployment_steps = [
            ("Prerequisites Check", self.check_prerequisites),
            ("Environment Setup", self.setup_environment),
            ("Backup Creation", self.backup_existing_deployment),
            ("Container Build", self.build_container_images),
            ("Security Scan", self.run_security_scan),
            ("Infrastructure Deploy", self.deploy_infrastructure),
            ("Database Migration", self.run_database_migrations),
            ("Deployment Validation", self.validate_deployment),
            ("Monitoring Setup", self.setup_monitoring),
            ("Report Generation", self.generate_deployment_report),
        ]
        
        for step_name, step_func in deployment_steps:
            logger.info(f"🔄 Executing: {step_name}")
            
            try:
                success = step_func()
                if not success:
                    logger.error(f"❌ {step_name} failed")
                    
                    # Ask for rollback on critical failures
                    if step_name in ["Infrastructure Deploy", "Database Migration"]:
                        logger.info("🔄 Initiating rollback...")
                        self.rollback_deployment()
                    
                    return False
                
                logger.info(f"✅ {step_name} completed")
                
            except Exception as e:
                logger.error(f"❌ {step_name} error: {e}")
                return False
        
        logger.info("🎉 XORB production deployment completed successfully!")
        return True


def main():
    """Main deployment entry point"""
    parser = argparse.ArgumentParser(description="XORB Production Deployment")
    parser.add_argument(
        "--environment",
        choices=["staging", "production"],
        default="production",
        help="Deployment environment"
    )
    parser.add_argument(
        "--skip-security-scan",
        action="store_true",
        help="Skip security scanning (not recommended for production)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform validation checks only"
    )
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = ProductionDeployer(args.environment)
    
    if args.dry_run:
        logger.info("🧪 Performing dry run validation...")
        success = (
            deployer.check_prerequisites() and
            deployer.setup_environment()
        )
        if success:
            logger.info("✅ Dry run completed successfully")
        else:
            logger.error("❌ Dry run failed")
        return success
    
    # Execute full deployment
    success = deployer.deploy()
    
    if success:
        logger.info("🎉 Deployment completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Deployment failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()