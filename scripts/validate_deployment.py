#!/usr/bin/env python3
"""
XORB Deployment Validation Script

Validates that the XORB ecosystem is properly deployed and functioning.
"""

import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import httpx
import psutil
import redis
from sqlalchemy import create_engine, text

# Color output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"

def info(msg: str) -> None:
    print(f"{Colors.BLUE}[INFO]{Colors.ENDC} {msg}")

def success(msg: str) -> None:
    print(f"{Colors.GREEN}[SUCCESS]{Colors.ENDC} {msg}")

def warning(msg: str) -> None:
    print(f"{Colors.YELLOW}[WARNING]{Colors.ENDC} {msg}")

def error(msg: str) -> None:
    print(f"{Colors.RED}[ERROR]{Colors.ENDC} {msg}")

class XORBDeploymentValidator:
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.results = {
            "environment": environment,
            "timestamp": time.time(),
            "checks": {},
            "overall_status": "unknown"
        }
        
        # Load environment-specific configuration
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration based on environment."""
        env_file = f"config/environments/{self.environment}.env"
        config = {
            "api_url": "http://localhost:8000",
            "worker_url": "http://localhost:9000",
            "orchestrator_url": "http://localhost:8001",
            "postgres_dsn": "postgresql://temporal:temporal@localhost:5432/temporal",
            "redis_url": "redis://localhost:6379/0",
            "prometheus_url": "http://localhost:9090",
            "grafana_url": "http://localhost:3000"
        }
        
        # Override with environment variables if available
        config.update({
            "api_url": os.getenv("API_URL", config["api_url"]),
            "postgres_dsn": os.getenv("POSTGRES_DSN", config["postgres_dsn"]),
            "redis_url": os.getenv("REDIS_URL", config["redis_url"])
        })
        
        return config
    
    async def validate_api_service(self) -> Tuple[bool, str]:
        """Validate API service is running and responding."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Health check
                response = await client.get(f"{self.config['api_url']}/health")
                if response.status_code != 200:
                    return False, f"API health check failed: {response.status_code}"
                
                # Check metrics endpoint
                response = await client.get(f"{self.config['api_url']}/metrics")
                if response.status_code != 200:
                    return False, f"API metrics endpoint failed: {response.status_code}"
                
                return True, "API service is healthy"
        except Exception as e:
            return False, f"API service check failed: {str(e)}"
    
    async def validate_worker_service(self) -> Tuple[bool, str]:
        """Validate Worker service is running."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.config['worker_url']}/health")
                if response.status_code != 200:
                    return False, f"Worker health check failed: {response.status_code}"
                
                return True, "Worker service is healthy"
        except Exception as e:
            return False, f"Worker service check failed: {str(e)}"
    
    def validate_database(self) -> Tuple[bool, str]:
        """Validate PostgreSQL database connectivity."""
        try:
            engine = create_engine(self.config["postgres_dsn"])
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                return True, f"Database connected: {version}"
        except Exception as e:
            return False, f"Database check failed: {str(e)}"
    
    def validate_redis(self) -> Tuple[bool, str]:
        """Validate Redis connectivity."""
        try:
            r = redis.from_url(self.config["redis_url"])
            r.ping()
            info = r.info()
            return True, f"Redis connected: version {info.get('redis_version', 'unknown')}"
        except Exception as e:
            return False, f"Redis check failed: {str(e)}"
    
    async def validate_monitoring(self) -> Tuple[bool, str]:
        """Validate monitoring stack."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check Prometheus
                response = await client.get(f"{self.config['prometheus_url']}/api/v1/query?query=up")
                if response.status_code != 200:
                    return False, f"Prometheus check failed: {response.status_code}"
                
                return True, "Monitoring stack is healthy"
        except Exception as e:
            return False, f"Monitoring check failed: {str(e)}"
    
    def validate_system_resources(self) -> Tuple[bool, str]:
        """Validate system resources."""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Check memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Warning thresholds
            if cpu_percent > 80:
                return False, f"High CPU usage: {cpu_percent}%"
            if memory_percent > 85:
                return False, f"High memory usage: {memory_percent}%"
            if disk_percent > 90:
                return False, f"High disk usage: {disk_percent}%"
            
            return True, f"System resources OK (CPU: {cpu_percent}%, RAM: {memory_percent}%, Disk: {disk_percent}%)"
        except Exception as e:
            return False, f"System resource check failed: {str(e)}"
    
    async def validate_agent_discovery(self) -> Tuple[bool, str]:
        """Validate agent discovery system."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.config['api_url']}/agents/discovery")
                if response.status_code != 200:
                    return False, f"Agent discovery check failed: {response.status_code}"
                
                agents = response.json()
                if not agents or len(agents) == 0:
                    return False, "No agents discovered"
                
                return True, f"Agent discovery OK: {len(agents)} agents found"
        except Exception as e:
            return False, f"Agent discovery check failed: {str(e)}"
    
    async def validate_orchestration(self) -> Tuple[bool, str]:
        """Validate orchestration system."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.config['orchestrator_url']}/health")
                if response.status_code != 200:
                    return False, f"Orchestrator health check failed: {response.status_code}"
                
                return True, "Orchestration system is healthy"
        except Exception as e:
            return False, f"Orchestration check failed: {str(e)}"
    
    async def run_all_checks(self) -> Dict:
        """Run all validation checks."""
        info(f"🚀 Starting XORB deployment validation for {self.environment} environment...")
        
        checks = [
            ("api_service", self.validate_api_service()),
            ("worker_service", self.validate_worker_service()),
            ("database", self.validate_database()),
            ("redis", self.validate_redis()),
            ("system_resources", self.validate_system_resources()),
            ("monitoring", self.validate_monitoring()),
            ("agent_discovery", self.validate_agent_discovery()),
            ("orchestration", self.validate_orchestration())
        ]
        
        all_passed = True
        
        for check_name, check_coro in checks:
            if asyncio.iscoroutine(check_coro):
                passed, message = await check_coro
            else:
                passed, message = check_coro
            
            self.results["checks"][check_name] = {
                "passed": passed,
                "message": message,
                "timestamp": time.time()
            }
            
            if passed:
                success(f"✅ {check_name}: {message}")
            else:
                error(f"❌ {check_name}: {message}")
                all_passed = False
        
        self.results["overall_status"] = "passed" if all_passed else "failed"
        
        # Summary
        print("\n" + "="*60)
        if all_passed:
            success(f"🎉 All validation checks passed for {self.environment} environment!")
        else:
            error(f"❌ Some validation checks failed for {self.environment} environment")
        
        return self.results
    
    def save_results(self, filename: Optional[str] = None) -> None:
        """Save validation results to file."""
        if not filename:
            filename = f"deployment_validation_{self.environment}_{int(time.time())}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        info(f"Validation results saved to: {filename}")

async def main():
    """Main function."""
    environment = os.getenv("XORB_ENV", "development")
    
    if len(sys.argv) > 1:
        environment = sys.argv[1]
    
    validator = XORBDeploymentValidator(environment)
    results = await validator.run_all_checks()
    validator.save_results()
    
    # Exit with appropriate code
    exit_code = 0 if results["overall_status"] == "passed" else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    asyncio.run(main())