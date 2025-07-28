#!/usr/bin/env python3
"""
Xorb PTaaS Automated Health Monitor
Continuous health monitoring with alerting and self-healing
"""

import asyncio
import logging
import os
import smtplib
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from email.mime.text import MIMEText

try:
    import aiohttp

    import docker
except ImportError:
    print("Installing required dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages", "aiohttp", "docker"])
    import aiohttp

    import docker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class HealthStatus:
    """Health status for a service"""
    service: str
    status: str  # healthy, unhealthy, unknown
    response_time: float
    last_check: datetime
    error_message: str | None = None
    consecutive_failures: int = 0

@dataclass
class Alert:
    """Alert information"""
    service: str
    severity: str  # critical, warning, info
    message: str
    timestamp: datetime
    resolved: bool = False

class HealthMonitor:
    """Automated health monitoring system"""

    def __init__(self):
        self.docker_client = docker.from_env()
        self.service_status: dict[str, HealthStatus] = {}
        self.alerts: list[Alert] = []
        self.monitoring = False

        # Configuration
        self.check_interval = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))  # seconds
        self.failure_threshold = int(os.getenv("FAILURE_THRESHOLD", "3"))  # consecutive failures
        self.enable_self_healing = os.getenv("ENABLE_SELF_HEALING", "true").lower() == "true"
        self.enable_email_alerts = os.getenv("ENABLE_EMAIL_ALERTS", "false").lower() == "true"

        # Services to monitor
        self.services = {
            "api": {"url": "http://localhost:8000/health", "port": 8000, "container": "xorb_api"},
            "orchestrator": {"url": "http://localhost:8001/health", "port": 8001, "container": "xorb_orchestrator"},
            "postgres": {"container": "xorb_postgres", "health_cmd": ["pg_isready", "-U", "xorb"]},
            "redis": {"container": "xorb_redis", "health_cmd": ["redis-cli", "ping"]},
            "temporal": {"url": "http://localhost:8080/health", "port": 8080, "container": "xorb_temporal"},
            "nats": {"url": "http://localhost:8222/varz", "port": 8222, "container": "xorb_nats"}
        }

    async def start_monitoring(self):
        """Start the health monitoring loop"""
        logger.info("🩺 Starting Xorb PTaaS Health Monitor")
        logger.info(f"📊 Check interval: {self.check_interval}s")
        logger.info(f"🔄 Self-healing: {'Enabled' if self.enable_self_healing else 'Disabled'}")
        logger.info(f"📧 Email alerts: {'Enabled' if self.enable_email_alerts else 'Disabled'}")

        self.monitoring = True

        try:
            while self.monitoring:
                await self.run_health_checks()
                await self.process_alerts()
                await self.generate_status_report()

                await asyncio.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("🛑 Health monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Health monitoring failed: {e}")
        finally:
            self.monitoring = False

    async def run_health_checks(self):
        """Execute health checks for all services"""
        logger.debug("🔍 Running health checks...")

        # Check HTTP services
        http_services = ["api", "orchestrator", "temporal", "nats"]
        for service_name in http_services:
            await self.check_http_service(service_name)

        # Check database services
        await self.check_database_service("postgres")
        await self.check_database_service("redis")

        # Check container health
        await self.check_container_health()

    async def check_http_service(self, service_name: str):
        """Check HTTP service health"""
        service_config = self.services[service_name]
        start_time = time.time()

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(service_config["url"]) as response:
                    response_time = time.time() - start_time

                    if response.status == 200:
                        await self.update_service_status(
                            service_name, "healthy", response_time
                        )
                    else:
                        await self.update_service_status(
                            service_name, "unhealthy", response_time,
                            f"HTTP {response.status}"
                        )

        except Exception as e:
            response_time = time.time() - start_time
            await self.update_service_status(
                service_name, "unhealthy", response_time, str(e)
            )

    async def check_database_service(self, service_name: str):
        """Check database service health"""
        service_config = self.services[service_name]
        container_name = service_config["container"]
        health_cmd = service_config["health_cmd"]

        start_time = time.time()

        try:
            # Execute health check command in container
            result = subprocess.run([
                "docker", "exec", container_name
            ] + health_cmd, capture_output=True, text=True, timeout=10)

            response_time = time.time() - start_time

            if result.returncode == 0:
                await self.update_service_status(
                    service_name, "healthy", response_time
                )
            else:
                await self.update_service_status(
                    service_name, "unhealthy", response_time,
                    f"Command failed: {result.stderr}"
                )

        except Exception as e:
            response_time = time.time() - start_time
            await self.update_service_status(
                service_name, "unhealthy", response_time, str(e)
            )

    async def check_container_health(self):
        """Check Docker container status"""
        try:
            containers = self.docker_client.containers.list()
            running_containers = {c.name: c for c in containers if c.status == 'running'}

            for service_name, config in self.services.items():
                container_name = config["container"]

                if container_name in running_containers:
                    container = running_containers[container_name]

                    # Check if container is healthy
                    health_status = container.attrs.get("State", {}).get("Health", {})
                    if health_status:
                        status = health_status.get("Status", "unknown")
                        if status == "healthy":
                            continue  # HTTP check will handle this
                        elif status == "unhealthy":
                            logger.warning(f"⚠️  Container {container_name} is unhealthy")
                            await self.handle_unhealthy_container(service_name, container)
                else:
                    # Container is not running
                    logger.error(f"❌ Container {container_name} is not running")
                    await self.handle_stopped_container(service_name, container_name)

        except Exception as e:
            logger.error(f"Failed to check container health: {e}")

    async def update_service_status(self, service: str, status: str, response_time: float, error_message: str = None):
        """Update service health status"""
        now = datetime.now()

        if service in self.service_status:
            current_status = self.service_status[service]

            if status == "unhealthy":
                current_status.consecutive_failures += 1
            else:
                current_status.consecutive_failures = 0

            current_status.status = status
            current_status.response_time = response_time
            current_status.last_check = now
            current_status.error_message = error_message
        else:
            self.service_status[service] = HealthStatus(
                service=service,
                status=status,
                response_time=response_time,
                last_check=now,
                error_message=error_message,
                consecutive_failures=1 if status == "unhealthy" else 0
            )

        # Check if we need to create an alert
        await self.check_alert_conditions(service)

    async def check_alert_conditions(self, service: str):
        """Check if alert conditions are met"""
        status = self.service_status[service]

        # Create critical alert for repeated failures
        if status.consecutive_failures >= self.failure_threshold:
            await self.create_alert(
                service, "critical",
                f"Service {service} has failed {status.consecutive_failures} consecutive health checks"
            )

            # Trigger self-healing if enabled
            if self.enable_self_healing:
                await self.attempt_self_healing(service)

        # Create warning for slow response
        elif status.status == "healthy" and status.response_time > 5.0:
            await self.create_alert(
                service, "warning",
                f"Service {service} is responding slowly ({status.response_time:.2f}s)"
            )

        # Resolve alerts if service is healthy
        elif status.status == "healthy" and status.consecutive_failures == 0:
            await self.resolve_alerts(service)

    async def create_alert(self, service: str, severity: str, message: str):
        """Create a new alert"""
        # Check if similar alert already exists
        existing_alert = next((a for a in self.alerts
                             if a.service == service and a.severity == severity and not a.resolved), None)

        if not existing_alert:
            alert = Alert(
                service=service,
                severity=severity,
                message=message,
                timestamp=datetime.now()
            )

            self.alerts.append(alert)
            logger.warning(f"🚨 ALERT [{severity.upper()}] {service}: {message}")

            # Send email notification if enabled
            if self.enable_email_alerts:
                await self.send_email_alert(alert)

    async def resolve_alerts(self, service: str):
        """Resolve alerts for a service"""
        resolved_count = 0

        for alert in self.alerts:
            if alert.service == service and not alert.resolved:
                alert.resolved = True
                resolved_count += 1

        if resolved_count > 0:
            logger.info(f"✅ Resolved {resolved_count} alerts for {service}")

    async def attempt_self_healing(self, service: str):
        """Attempt to heal unhealthy service"""
        if not self.enable_self_healing:
            return

        logger.info(f"🔧 Attempting self-healing for {service}")

        try:
            service_config = self.services[service]
            container_name = service_config["container"]

            # Try restarting the container
            container = self.docker_client.containers.get(container_name)
            container.restart()

            logger.info(f"🔄 Restarted container {container_name}")

            # Wait a bit for service to start
            await asyncio.sleep(30)

            # Create info alert about self-healing action
            await self.create_alert(
                service, "info",
                f"Attempted self-healing by restarting {container_name}"
            )

        except Exception as e:
            logger.error(f"❌ Self-healing failed for {service}: {e}")
            await self.create_alert(
                service, "critical",
                f"Self-healing failed: {e}"
            )

    async def handle_unhealthy_container(self, service: str, container):
        """Handle unhealthy container"""
        logger.warning(f"⚠️  Handling unhealthy container for {service}")

        # Log container health details
        health = container.attrs.get("State", {}).get("Health", {})
        if health:
            logger.info(f"Health status: {health.get('Status')}")
            last_output = health.get("Log", [])[-1] if health.get("Log") else {}
            if last_output:
                logger.info(f"Last health check: {last_output.get('Output', 'No output')}")

    async def handle_stopped_container(self, service: str, container_name: str):
        """Handle stopped container"""
        logger.error(f"❌ Handling stopped container: {container_name}")

        if self.enable_self_healing:
            try:
                # Try to start the container
                container = self.docker_client.containers.get(container_name)
                container.start()

                logger.info(f"🔄 Started stopped container {container_name}")

                await self.create_alert(
                    service, "info",
                    f"Started stopped container {container_name}"
                )

            except Exception as e:
                logger.error(f"❌ Failed to start container {container_name}: {e}")

    async def process_alerts(self):
        """Process and manage alerts"""
        # Clean up old resolved alerts (keep for 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [a for a in self.alerts if not a.resolved or a.timestamp > cutoff_time]

    async def generate_status_report(self):
        """Generate periodic status report"""
        now = datetime.now()

        # Generate report every 5 minutes
        if not hasattr(self, '_last_report') or (now - self._last_report).seconds >= 300:
            healthy_services = sum(1 for s in self.service_status.values() if s.status == "healthy")
            total_services = len(self.service_status)
            active_alerts = sum(1 for a in self.alerts if not a.resolved)

            logger.info(f"📊 Health Report: {healthy_services}/{total_services} services healthy, {active_alerts} active alerts")

            self._last_report = now

    async def send_email_alert(self, alert: Alert):
        """Send email alert notification"""
        try:
            smtp_server = os.getenv("SMTP_SERVER", "localhost")
            smtp_port = int(os.getenv("SMTP_PORT", "587"))
            smtp_user = os.getenv("SMTP_USER")
            smtp_password = os.getenv("SMTP_PASSWORD")
            alert_email = os.getenv("ALERT_EMAIL")

            if not all([smtp_user, smtp_password, alert_email]):
                logger.warning("Email configuration incomplete, skipping email alert")
                return

            subject = f"Xorb PTaaS Alert [{alert.severity.upper()}] - {alert.service}"
            body = f"""
Xorb PTaaS Health Monitor Alert

Service: {alert.service}
Severity: {alert.severity.upper()}
Message: {alert.message}
Timestamp: {alert.timestamp.isoformat()}

This is an automated alert from the Xorb PTaaS health monitoring system.
"""

            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = smtp_user
            msg['To'] = alert_email

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)

            logger.info(f"📧 Email alert sent for {alert.service}")

        except Exception as e:
            logger.error(f"❌ Failed to send email alert: {e}")

    def get_status_summary(self) -> dict:
        """Get current status summary"""
        return {
            "timestamp": datetime.now().isoformat(),
            "monitoring": self.monitoring,
            "services": {name: asdict(status) for name, status in self.service_status.items()},
            "active_alerts": [asdict(a) for a in self.alerts if not a.resolved],
            "total_alerts": len(self.alerts),
            "configuration": {
                "check_interval": self.check_interval,
                "failure_threshold": self.failure_threshold,
                "self_healing_enabled": self.enable_self_healing,
                "email_alerts_enabled": self.enable_email_alerts
            }
        }

    async def stop_monitoring(self):
        """Stop health monitoring"""
        logger.info("🛑 Stopping health monitor...")
        self.monitoring = False

async def main():
    """Main function"""
    monitor = HealthMonitor()

    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        await monitor.stop_monitoring()
    except Exception as e:
        logger.error(f"❌ Monitor failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    import sys

    print("🩺 Xorb PTaaS Automated Health Monitor")
    print("=" * 50)

    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Health monitoring interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Failed to start health monitor: {e}")
        sys.exit(1)
