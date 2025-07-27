#!/usr/bin/env python3
"""
XORB 2.0 Quick Deployment Script

Simple deployment script for XORB ecosystem with all advanced features.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

class QuickDeploy:
    def __init__(self):
        self.project_root = Path(__file__).parent
        os.chdir(self.project_root)
    
    def run_command(self, command, description):
        """Run a shell command with error handling."""
        print(f"🔄 {description}...")
        try:
            if isinstance(command, str):
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
            else:
                result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {description} - Complete")
                return True
            else:
                print(f"❌ {description} - Failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ {description} - Error: {e}")
            return False
    
    def deploy(self):
        """Deploy XORB ecosystem."""
        print("🚀 XORB 2.0 Quick Deployment")
        print("=" * 40)
        
        steps = [
            # Environment setup
            ("python3 -m venv venv", "Creating virtual environment"),
            ("venv/bin/pip install -r requirements.txt", "Installing core dependencies"),
            ("venv/bin/pip install prometheus-client structlog cryptography psutil aiohttp", "Installing advanced dependencies"),
            
            # Configuration
            (self.create_config, "Generating configuration"),
            
            # Start services
            ("docker-compose --env-file config/local/.xorb.env -f docker-compose.local.yml up -d postgres redis", "Starting databases"),
            (lambda: time.sleep(20), "Waiting for databases"),
            ("docker-compose --env-file config/local/.xorb.env -f docker-compose.local.yml up -d", "Starting all services"),
            (lambda: time.sleep(30), "Waiting for services"),
            
            # Validation
            (self.test_deployment, "Testing deployment")
        ]
        
        failed = False
        for command, description in steps:
            if callable(command):
                try:
                    command()
                    print(f"✅ {description} - Complete")
                except Exception as e:
                    print(f"❌ {description} - Failed: {e}")
                    failed = True
            else:
                if not self.run_command(command, description):
                    failed = True
                    # Don't stop for non-critical failures
                    if "test" not in description.lower():
                        continue
        
        if not failed:
            self.print_success()
        else:
            self.print_failure()
    
    def create_config(self):
        """Create basic configuration."""
        config = """# XORB Configuration
XORB_ENV=development
XORB_ADVANCED_FEATURES=true
XORB_LOG_LEVEL=INFO

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=xorb
POSTGRES_USER=xorb
POSTGRES_PASSWORD=xorb_password

REDIS_HOST=localhost
REDIS_PORT=6379

# Performance
XORB_MAX_CONCURRENT_AGENTS=8
XORB_WORKER_PROCESSES=4

# Service Configuration (for Docker Compose)
CACHE_SIZE_MB=1024
LOG_LEVEL=INFO
ORCHESTRATOR_WORKERS=4
MAX_CONCURRENT_AGENTS=8

# Resource Limits
MEMORY_LIMIT_API=2g
MEMORY_LIMIT_WORKER=1g
MEMORY_LIMIT_ORCHESTRATOR=1g
CPU_LIMIT_API=4.0
CPU_LIMIT_WORKER=2.0
CPU_LIMIT_ORCHESTRATOR=2.0
"""
        with open("config/local/.xorb.env", "w") as f:
            f.write(config)
    
    def test_deployment(self):
        """Test if deployment is working."""
        # Test advanced features
        test_commands = [
            "venv/bin/python -c 'from xorb_core.vulnerabilities import vulnerability_manager; print(\"Vulnerability management OK\")'",
            "venv/bin/python -c 'from xorb_core.intelligence.threat_intelligence_engine import threat_intel_engine; print(\"Threat intelligence OK\")'",
            "venv/bin/python -c 'from xorb_core.hunting import ai_threat_hunter; print(\"AI threat hunting OK\")'",
            "venv/bin/python -c 'from xorb_core.orchestration import distributed_coordinator; print(\"Distributed coordination OK\")'"
        ]
        
        for cmd in test_commands:
            self.run_command(cmd, "Testing advanced features")
    
    def print_success(self):
        """Print success message."""
        print("\n🎉 XORB 2.0 Deployment Complete!")
        print("=" * 35)
        print("🔗 Access Points:")
        print("   • API: http://localhost:8000")
        print("   • API Docs: http://localhost:8000/docs")
        print("   • Orchestrator: http://localhost:8080")
        print("   • Grafana: http://localhost:3000")
        print("   • Prometheus: http://localhost:9090")
        
        print("\n🧪 Test Commands:")
        print("   • Basic Tests: make -f Makefile.advanced advanced-tests")
        print("   • Vulnerability Demo: make -f Makefile.advanced vulnerability-demo")
        print("   • AI Hunting Demo: make -f Makefile.advanced ai-hunting-demo")
        print("   • Full Demo: make -f Makefile.advanced advanced-demo")
        
        print("\n📊 Management:")
        print("   • Status: make -f Makefile.advanced status-report")
        print("   • Stop: docker-compose -f docker-compose.local.yml down")
        print("   • Logs: docker-compose -f docker-compose.local.yml logs -f")
    
    def print_failure(self):
        """Print failure message."""
        print("\n❌ Deployment completed with some issues")
        print("🔧 Troubleshooting:")
        print("   1. Check Docker: docker --version")
        print("   2. Check services: docker-compose ps")
        print("   3. Check logs: docker-compose logs")
        print("   4. Manual start: docker-compose -f docker-compose.local.yml up -d")

if __name__ == "__main__":
    deployer = QuickDeploy()
    deployer.deploy()