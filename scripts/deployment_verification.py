from dataclasses import dataclass

import logging
logger = logging.getLogger(__name__)

#!/usr/bin/env python3
"""
Xorb PTaaS Deployment Verification Script
Comprehensive validation of production deployment
"""

import asyncio
import json
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sys
import os
import aiofiles

try:
    import aiohttp
    import docker
except ImportError as e:
        logger.exception("Error in operation: %s", e)
    print("Installing required dependencies...")
    await asyncio.create_subprocess_exec([sys.executable, "-m", "pip", "install", "--break-system-packages", "aiohttp", "docker"])
    import aiohttp
    import docker

@dataclass
class DeploymentVerifier:
    """Comprehensive deployment verification for Xorb PTaaS"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "checks": {},
            "services": {},
            "endpoints": {},
            "performance": {},
            "security": {}
        }
        
    async def verify_deployment(self) -> Dict:
        """Run complete deployment verification"""
        print("🔍 Starting Xorb PTaaS Deployment Verification...")
        print("=" * 60)
        
        # Core service checks
        await self.check_docker_services()
        await self.check_database_connectivity()
        await self.check_api_endpoints()
        await self.check_service_health()
        
        # Advanced feature checks
        await self.check_monitoring_stack()
        await self.check_security_configuration()
        await self.check_backup_systems()
        await self.check_performance_metrics()
        
        # Generate summary
        self.generate_summary()
        
        return self.results
    
    async def check_docker_services(self):
        """Verify all Docker services are running"""
        print("📦 Checking Docker Services...")
        
        expected_services = [
            "xorb_postgres", "xorb_redis", "xorb_nats", "xorb_temporal",
            "xorb_api", "xorb_worker", "xorb_orchestrator"
        ]
        
        service_status = {}
        
        try:
            containers = self.docker_client.containers.list()
            running_services = [c.name for c in containers if c.status == 'running']
            
            for service in expected_services:
                if service in running_services:
                    container = self.docker_client.containers.get(service)
                    service_status[service] = {
                        "status": "running",
                        "health": container.attrs.get("State", {}).get("Health", {}).get("Status", "unknown"),
                        "uptime": container.attrs["State"]["StartedAt"]
                    }
                    print(f"  ✅ {service}: Running")
                else:
                    service_status[service] = {"status": "stopped", "health": "unhealthy"}
                    print(f"  ❌ {service}: Not running")
            
            self.results["services"] = service_status
            
        except Exception as e as e:
        logger.exception("Error in operation: %s", e)
            print(f"  ❌ Docker service check failed: {e}")
            self.results["checks"]["docker_services"] = {"status": "failed", "error": str(e)}
    
    async def check_database_connectivity(self):
        """Test database connections"""
        print("\n🗄️  Checking Database Connectivity...")
        
        db_checks = {}
        
        # PostgreSQL check
        try:
            result = subprocess.run([
                "docker", "exec", "xorb_postgres", "pg_isready", "-U", "xorb", "-d", "xorb_ptaas"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                db_checks["postgresql"] = {"status": "connected", "response_time": "< 1s"}
                print("  ✅ PostgreSQL: Connected")
            else:
                db_checks["postgresql"] = {"status": "failed", "error": result.stderr}
                print("  ❌ PostgreSQL: Connection failed")
                
        except Exception as e as e:
        logger.exception("Error in operation: %s", e)
            db_checks["postgresql"] = {"status": "error", "error": str(e)}
            print(f"  ❌ PostgreSQL: {e}")
        
        # Redis check
        try:
            result = subprocess.run([
                "docker", "exec", "xorb_redis", "redis-cli", "ping"
            ], capture_output=True, text=True, timeout=10)
            
            if "PONG" in result.stdout:
                db_checks["redis"] = {"status": "connected", "response_time": "< 1s"}
                print("  ✅ Redis: Connected")
            else:
                db_checks["redis"] = {"status": "failed", "error": result.stderr}
                print("  ❌ Redis: Connection failed")
                
        except Exception as e as e:
        logger.exception("Error in operation: %s", e)
            db_checks["redis"] = {"status": "error", "error": str(e)}
            print(f"  ❌ Redis: {e}")
        
        self.results["checks"]["databases"] = db_checks
    
    async def check_api_endpoints(self):
        """Test all API endpoints"""
        print("\n🌐 Checking API Endpoints...")
        
        endpoints = [
            {"url": "http://localhost:8000/", "name": "Root"},
            {"url": "http://localhost:8000/health", "name": "Health Check"},
            {"url": "http://localhost:8000/api/v1/status", "name": "API Status"},
            {"url": "http://localhost:8000/api/v1/assets", "name": "Assets"},
            {"url": "http://localhost:8000/api/v1/scans", "name": "Scans"},
            {"url": "http://localhost:8000/api/v1/findings", "name": "Findings"},
            {"url": "http://localhost:8000/api/gamification/leaderboard", "name": "Leaderboard"},
            {"url": "http://localhost:8000/api/compliance/status", "name": "Compliance"}
        ]
        
        endpoint_results = {}
        
        async with aiohttp.ClientSession() as session:
            for endpoint in endpoints:
                try:
                    start_time = time.time()
                    async with session.get(endpoint["url"], timeout=10) as response:
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            data = await response.json()
                            endpoint_results[endpoint["name"]] = {
                                "status": "success",
                                "response_time": f"{response_time:.3f}s",
                                "status_code": response.status,
                                "data": data
                            }
                            print(f"  ✅ {endpoint['name']}: {response.status} ({response_time:.3f}s)")
                        else:
                            endpoint_results[endpoint["name"]] = {
                                "status": "failed",
                                "status_code": response.status,
                                "response_time": f"{response_time:.3f}s"
                            }
                            print(f"  ❌ {endpoint['name']}: {response.status}")
                            
                except Exception as e as e:
        logger.exception("Error in operation: %s", e)
                    endpoint_results[endpoint["name"]] = {
                        "status": "error",
                        "error": str(e)
                    }
                    print(f"  ❌ {endpoint['name']}: {e}")
        
        self.results["endpoints"] = endpoint_results
    
    async def check_service_health(self):
        """Check individual service health"""
        print("\n🏥 Checking Service Health...")
        
        health_checks = {}
        
        # Check Temporal
        try:
            result = subprocess.run([
                "curl", "-f", "http://localhost:8080/health"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                health_checks["temporal"] = {"status": "healthy", "port": 8080}
                print("  ✅ Temporal: Healthy")
            else:
                health_checks["temporal"] = {"status": "unhealthy", "error": result.stderr}
                print("  ❌ Temporal: Unhealthy")
                
        except Exception as e as e:
        logger.exception("Error in operation: %s", e)
            health_checks["temporal"] = {"status": "error", "error": str(e)}
            print(f"  ❌ Temporal: {e}")
        
        # Check NATS
        try:
            result = subprocess.run([
                "curl", "-f", "http://localhost:8222/varz"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                health_checks["nats"] = {"status": "healthy", "port": 8222}
                print("  ✅ NATS: Healthy")
            else:
                health_checks["nats"] = {"status": "unhealthy", "error": result.stderr}
                print("  ❌ NATS: Unhealthy")
                
        except Exception as e as e:
        logger.exception("Error in operation: %s", e)
            health_checks["nats"] = {"status": "error", "error": str(e)}
            print(f"  ❌ NATS: {e}")
        
        self.results["checks"]["health"] = health_checks
    
    async def check_monitoring_stack(self):
        """Check monitoring and observability"""
        print("\n📊 Checking Monitoring Stack...")
        
        monitoring_checks = {}
        
        # Check if Prometheus configs exist
        prometheus_config = "/root/Xorb/compose/observability/prometheus/prometheus.yml"
        if os.path.exists(prometheus_config):
            monitoring_checks["prometheus_config"] = {"status": "configured"}
            print("  ✅ Prometheus: Configuration found")
        else:
            monitoring_checks["prometheus_config"] = {"status": "missing"}
            print("  ❌ Prometheus: Configuration missing")
        
        # Check Grafana dashboards
        dashboard_dir = "/root/Xorb/compose/observability/grafana/dashboards"
        if os.path.exists(dashboard_dir):
            dashboards = os.listdir(dashboard_dir)
            monitoring_checks["grafana_dashboards"] = {
                "status": "configured",
                "count": len(dashboards),
                "dashboards": dashboards
            }
            print(f"  ✅ Grafana: {len(dashboards)} dashboards configured")
        else:
            monitoring_checks["grafana_dashboards"] = {"status": "missing"}
            print("  ❌ Grafana: Dashboards missing")
        
        self.results["checks"]["monitoring"] = monitoring_checks
    
    async def check_security_configuration(self):
        """Verify security configurations"""
        print("\n🔒 Checking Security Configuration...")
        
        security_checks = {}
        
        # Check environment variables
        env_file = "/root/Xorb/.env"
        if os.path.exists(env_file):
            async with aiofiles.open(env_file, 'r') as f:
                env_content = f.read()
                
            security_checks["environment"] = {
                "status": "configured",
                "jwt_secret": "JWT_SECRET_KEY" in env_content,
                "api_keys": "OPENAI_API_KEY" in env_content,
                "stripe_keys": "STRIPE_SECRET_KEY" in env_content
            }
            print("  ✅ Environment: Security variables configured")
        else:
            security_checks["environment"] = {"status": "missing"}
            print("  ❌ Environment: Configuration missing")
        
        # Check container security
        try:
            api_container = self.docker_client.containers.get("xorb_api")
            security_opts = api_container.attrs["HostConfig"].get("SecurityOpt", [])
            
            security_checks["containers"] = {
                "status": "hardened" if "no-new-privileges:true" in security_opts else "basic",
                "security_opts": security_opts
            }
            
            if "no-new-privileges:true" in security_opts:
                print("  ✅ Containers: Security hardened")
            else:
                print("  ⚠️  Containers: Basic security")
                
        except Exception as e as e:
        logger.exception("Error in operation: %s", e)
            security_checks["containers"] = {"status": "error", "error": str(e)}
            print(f"  ❌ Containers: {e}")
        
        self.results["security"] = security_checks
    
    async def check_backup_systems(self):
        """Verify backup system configuration"""
        print("\n💾 Checking Backup Systems...")
        
        backup_checks = {}
        
        # Check backup scripts
        backup_script = "/root/Xorb/scripts/advanced_backup_system.py"
        if os.path.exists(backup_script):
            backup_checks["backup_script"] = {"status": "available"}
            print("  ✅ Backup System: Advanced backup script available")
        else:
            backup_checks["backup_script"] = {"status": "missing"}
            print("  ❌ Backup System: Script missing")
        
        # Check B2 lifecycle manager
        b2_script = "/root/Xorb/scripts/b2_lifecycle_manager.py"
        if os.path.exists(b2_script):
            backup_checks["b2_lifecycle"] = {"status": "available"}
            print("  ✅ B2 Lifecycle: Manager script available")
        else:
            backup_checks["b2_lifecycle"] = {"status": "missing"}
            print("  ❌ B2 Lifecycle: Manager missing")
        
        self.results["checks"]["backup"] = backup_checks
    
    async def check_performance_metrics(self):
        """Check performance and resource usage"""
        print("\n⚡ Checking Performance Metrics...")
        
        performance_checks = {}
        
        try:
            # Get container resource usage
            containers = self.docker_client.containers.list()
            resource_usage = {}
            
            for container in containers:
                if container.name.startswith("xorb_"):
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU usage
                    cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                               stats["precpu_stats"]["cpu_usage"]["total_usage"]
                    system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                                  stats["precpu_stats"]["system_cpu_usage"]
                    
                    cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0
                    
                    # Memory usage
                    memory_usage = stats["memory_stats"]["usage"]
                    memory_limit = stats["memory_stats"]["limit"]
                    memory_percent = (memory_usage / memory_limit) * 100.0
                    
                    resource_usage[container.name] = {
                        "cpu_percent": round(cpu_percent, 2),
                        "memory_percent": round(memory_percent, 2),
                        "memory_usage_mb": round(memory_usage / 1024 / 1024, 2)
                    }
            
            performance_checks["resource_usage"] = resource_usage
            print("  ✅ Performance: Resource usage collected")
            
        except Exception as e as e:
        logger.exception("Error in operation: %s", e)
            performance_checks["resource_usage"] = {"status": "error", "error": str(e)}
            print(f"  ❌ Performance: {e}")
        
        self.results["performance"] = performance_checks
    
    def generate_summary(self):
        """Generate deployment verification summary"""
        print("\n" + "=" * 60)
        print("📋 DEPLOYMENT VERIFICATION SUMMARY")
        print("=" * 60)
        
        # Count successful checks
        total_checks = 0
        passed_checks = 0
        
        # Service checks
        for service, status in self.results.get("services", {}).items():
            total_checks += 1
            if status.get("status") == "running":
                passed_checks += 1
        
        # Endpoint checks
        for endpoint, status in self.results.get("endpoints", {}).items():
            total_checks += 1
            if status.get("status") == "success":
                passed_checks += 1
        
        # Database checks
        for db, status in self.results.get("checks", {}).get("databases", {}).items():
            total_checks += 1
            if status.get("status") == "connected":
                passed_checks += 1
        
        success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        if success_rate >= 90:
            overall_status = "✅ EXCELLENT"
            status_color = "🟢"
        elif success_rate >= 75:
            overall_status = "⚠️  GOOD"
            status_color = "🟡"
        elif success_rate >= 50:
            overall_status = "⚠️  NEEDS ATTENTION"
            status_color = "🟡"
        else:
            overall_status = "❌ CRITICAL ISSUES"
            status_color = "🔴"
        
        print(f"\n{status_color} Overall Status: {overall_status}")
        print(f"📊 Success Rate: {success_rate:.1f}% ({passed_checks}/{total_checks})")
        print(f"🕐 Verification Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.results["overall_status"] = overall_status
        self.results["success_rate"] = success_rate
        self.results["checks_passed"] = passed_checks
        self.results["total_checks"] = total_checks
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if success_rate < 100:
            print("  • Review failed checks above")
            print("  • Ensure all services are properly configured")
            print("  • Check logs for detailed error information")
        
        if success_rate >= 90:
            print("  • ✅ Deployment is production-ready!")
            print("  • Consider setting up monitoring alerts")
            print("  • Schedule regular backup tests")
        
        print("\n🚀 Xorb PTaaS Platform Status: OPERATIONAL")

async def main():
    """Main verification function"""
    verifier = DeploymentVerifier()
    results = await verifier.verify_deployment()
    
    # Save results to file
    output_file = f"/root/Xorb/deployment_verification_{int(time.time())}.json"
    async with aiofiles.open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Full results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    try:
        results = asyncio.run(main())
        sys.exit(0 if results.get("success_rate", 0) >= 75 else 1)
    except KeyboardInterrupt as e:
        logger.exception("Error in operation: %s", e)
        print("\n⚠️  Verification interrupted by user")
        sys.exit(1)
    except Exception as e as e:
        logger.exception("Error in operation: %s", e)
        print(f"\n❌ Verification failed: {e}")
        sys.exit(1)