#!/usr/bin/env python3
"""
Xorb Production Deployment Verification Script
Validates security hardening and operational readiness
"""

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import httpx


class DeploymentVerifier:
    """Verifies hardened deployment configuration"""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("API_URL", "http://localhost:8000")
        self.results: dict[str, dict] = {}

    async def run_all_checks(self) -> bool:
        """Run all verification checks"""

        print("🔍 Xorb Production Deployment Verification")
        print("=" * 50)

        checks = [
            ("API Health", self.check_api_health),
            ("Service Status", self.check_service_status),
            ("Security Configuration", self.check_security_config),
            ("Container Hardening", self.check_container_hardening),
            ("Database Connectivity", self.check_database_connectivity),
            ("Monitoring Stack", self.check_monitoring_stack),
            ("Resource Limits", self.check_resource_limits),
            ("Network Security", self.check_network_security),
            ("Backup Configuration", self.check_backup_config)
        ]

        passed = 0
        total = len(checks)

        for check_name, check_func in checks:
            print(f"\n🧪 Running: {check_name}")
            try:
                result = await check_func()
                if result.get("status") == "pass":
                    print(f"✅ {check_name}: PASSED")
                    passed += 1
                else:
                    print(f"❌ {check_name}: FAILED")
                    if result.get("details"):
                        print(f"   Details: {result['details']}")

                self.results[check_name] = result

            except Exception as e:
                print(f"💥 {check_name}: ERROR - {str(e)}")
                self.results[check_name] = {
                    "status": "error",
                    "details": str(e)
                }

        print(f"\n📊 Verification Summary: {passed}/{total} checks passed")

        if passed == total:
            print("🎉 Deployment hardened successfully!")
            return True
        else:
            print("⚠️  Some hardening checks failed. Review results above.")
            return False

    async def check_api_health(self) -> dict:
        """Check API service health"""

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/health")

                if response.status_code == 200:
                    health_data = response.json()

                    # Check dependencies
                    dependencies = health_data.get("dependencies", {})
                    unhealthy = [k for k, v in dependencies.items() if v != "healthy"]

                    if not unhealthy:
                        return {"status": "pass", "details": "All dependencies healthy"}
                    else:
                        return {
                            "status": "fail",
                            "details": f"Unhealthy dependencies: {unhealthy}"
                        }
                else:
                    return {
                        "status": "fail",
                        "details": f"HTTP {response.status_code}"
                    }

        except Exception as e:
            return {"status": "fail", "details": str(e)}

    async def check_service_status(self) -> dict:
        """Check Docker Compose service status"""

        try:
            result = subprocess.run(
                ["docker", "compose", "ps", "--format", "json"],
                capture_output=True,
                text=True,
                check=True
            )

            services = json.loads(result.stdout) if result.stdout.strip() else []

            if not services:
                return {"status": "fail", "details": "No services found"}

            unhealthy_services = []
            for service in services:
                if service.get("State") != "running":
                    unhealthy_services.append(service.get("Name", "unknown"))

            if not unhealthy_services:
                return {
                    "status": "pass",
                    "details": f"{len(services)} services running"
                }
            else:
                return {
                    "status": "fail",
                    "details": f"Non-running services: {unhealthy_services}"
                }

        except subprocess.CalledProcessError as e:
            return {"status": "fail", "details": f"Docker command failed: {e}"}
        except Exception as e:
            return {"status": "fail", "details": str(e)}

    async def check_security_config(self) -> dict:
        """Check security configuration"""

        security_issues = []

        # Check secrets
        secrets_dir = Path(".secrets")
        if not secrets_dir.exists():
            security_issues.append("Secrets directory missing")
        else:
            required_secrets = ["nvidia_api_key", "postgres_password"]
            for secret in required_secrets:
                secret_file = secrets_dir / secret
                if not secret_file.exists():
                    security_issues.append(f"Missing secret: {secret}")
                elif secret_file.stat().st_mode & 0o077:
                    security_issues.append(f"Secret {secret} has loose permissions")

        # Check environment file
        env_file = Path(".env")
        if not env_file.exists():
            security_issues.append("Environment file missing")

        # Check compose file permissions
        compose_file = Path("docker-compose.yml")
        if compose_file.exists():
            if compose_file.stat().st_mode & 0o022:
                security_issues.append("Compose file has loose permissions")

        if not security_issues:
            return {"status": "pass", "details": "Security configuration valid"}
        else:
            return {"status": "fail", "details": "; ".join(security_issues)}

    async def check_container_hardening(self) -> dict:
        """Check container security hardening"""

        try:
            # Check for non-root users
            result = subprocess.run(
                ["docker", "compose", "config"],
                capture_output=True,
                text=True,
                check=True
            )

            config = result.stdout
            hardening_issues = []

            # Check for user specifications
            xorb_services = ["api", "worker", "embedding"]
            for service in xorb_services:
                if "user: \"101" not in config:
                    hardening_issues.append(f"Service {service} may be running as root")

            # Check for read-only filesystem
            if "read_only: true" not in config:
                hardening_issues.append("Read-only filesystem not configured")

            # Check for security options
            if "no-new-privileges:true" not in config:
                hardening_issues.append("no-new-privileges not set")

            if not hardening_issues:
                return {"status": "pass", "details": "Container hardening applied"}
            else:
                return {"status": "fail", "details": "; ".join(hardening_issues)}

        except subprocess.CalledProcessError as e:
            return {"status": "fail", "details": f"Docker config check failed: {e}"}

    async def check_database_connectivity(self) -> dict:
        """Check database connectivity and configuration"""

        try:
            # Check PostgreSQL
            postgres_result = subprocess.run(
                ["docker", "compose", "exec", "-T", "postgres",
                 "pg_isready", "-U", "xorb", "-d", "xorb"],
                capture_output=True,
                text=True
            )

            if postgres_result.returncode != 0:
                return {"status": "fail", "details": "PostgreSQL not ready"}

            # Check Redis
            redis_result = subprocess.run(
                ["docker", "compose", "exec", "-T", "redis",
                 "redis-cli", "ping"],
                capture_output=True,
                text=True
            )

            if redis_result.returncode != 0:
                return {"status": "fail", "details": "Redis not responding"}

            return {"status": "pass", "details": "All databases accessible"}

        except subprocess.CalledProcessError as e:
            return {"status": "fail", "details": f"Database check failed: {e}"}

    async def check_monitoring_stack(self) -> dict:
        """Check monitoring and observability stack"""

        monitoring_endpoints = [
            ("Prometheus", "http://localhost:9090/-/healthy"),
            ("Grafana", "http://localhost:3000/api/health"),
            ("Tempo", "http://localhost:3200/ready")
        ]

        failed_services = []

        for service_name, endpoint in monitoring_endpoints:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(endpoint)
                    if response.status_code not in [200, 204]:
                        failed_services.append(service_name)
            except Exception:
                failed_services.append(service_name)

        if not failed_services:
            return {"status": "pass", "details": "All monitoring services healthy"}
        else:
            return {
                "status": "fail",
                "details": f"Failed services: {', '.join(failed_services)}"
            }

    async def check_resource_limits(self) -> dict:
        """Check resource limits and constraints"""

        try:
            # Check container resource usage
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format",
                 "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"],
                capture_output=True,
                text=True,
                check=True
            )

            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            high_usage_containers = []

            for line in lines:
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        container = parts[0]
                        cpu_usage = parts[1].replace('%', '')
                        try:
                            if float(cpu_usage) > 90:
                                high_usage_containers.append(container)
                        except ValueError:
                            pass

            if not high_usage_containers:
                return {"status": "pass", "details": "Resource usage within limits"}
            else:
                return {
                    "status": "fail",
                    "details": f"High CPU usage: {', '.join(high_usage_containers)}"
                }

        except subprocess.CalledProcessError as e:
            return {"status": "fail", "details": f"Resource check failed: {e}"}

    async def check_network_security(self) -> dict:
        """Check network security configuration"""

        try:
            # Check for exposed services
            result = subprocess.run(
                ["docker", "compose", "ps", "--services"],
                capture_output=True,
                text=True,
                check=True
            )

            services = result.stdout.strip().split('\n')

            # Check port configuration
            port_result = subprocess.run(
                ["docker", "compose", "port", "api", "8000"],
                capture_output=True,
                text=True
            )

            if "0.0.0.0:" in port_result.stdout:
                return {
                    "status": "warning",
                    "details": "API exposed on all interfaces - ensure firewall is configured"
                }

            return {"status": "pass", "details": "Network configuration secure"}

        except subprocess.CalledProcessError as e:
            return {"status": "fail", "details": f"Network check failed: {e}"}

    async def check_backup_config(self) -> dict:
        """Check backup configuration"""

        backup_script = Path("scripts/backup_data.sh")
        if not backup_script.exists():
            return {"status": "fail", "details": "Backup script not found"}

        if not backup_script.stat().st_mode & 0o100:
            return {"status": "fail", "details": "Backup script not executable"}

        # Check if backup directory exists
        backup_dir = Path("backups")
        if not backup_dir.exists():
            backup_dir.mkdir(exist_ok=True)

        return {"status": "pass", "details": "Backup configuration ready"}

    def save_results(self) -> None:
        """Save verification results to file"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"deployment_verification_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": self.results
            }, f, indent=2)

        print(f"📄 Results saved to: {results_file}")


async def main():
    """Main verification function"""

    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    else:
        api_url = os.getenv("API_URL", "http://localhost:8000")

    verifier = DeploymentVerifier(api_url)

    try:
        success = await verifier.run_all_checks()
        verifier.save_results()

        if success:
            print("\n🎯 Deployment verification completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️  Deployment verification found issues!")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Verification failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
