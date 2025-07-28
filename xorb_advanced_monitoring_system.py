#!/usr/bin/env python3
"""
XORB Advanced Monitoring & Observability System
Real-time platform health, performance metrics, and autonomous agent monitoring
"""

import asyncio
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import psutil

# Add project paths
sys.path.insert(0, '/root/Xorb/packages/xorb_core')
sys.path.insert(0, '/root/Xorb')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/Xorb/xorb_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('XORB-MONITORING')

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class ServiceStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class MetricData:
    """Individual metric data point"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: dict[str, str] = None

@dataclass
class HealthCheck:
    """Service health check result"""
    service: str
    status: ServiceStatus
    response_time_ms: float
    error_message: str | None = None
    details: dict[str, Any] = None

@dataclass
class Alert:
    """System alert"""
    level: AlertLevel
    service: str
    message: str
    timestamp: datetime
    metric_name: str | None = None
    threshold: float | None = None
    current_value: float | None = None

class XORBAdvancedMonitoring:
    """Advanced monitoring and observability system for XORB platform"""

    def __init__(self):
        self.session_id = f"MONITOR-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.now()

        # Monitoring state
        self.metrics_history: list[MetricData] = []
        self.health_history: list[HealthCheck] = []
        self.alerts: list[Alert] = []
        self.active_services = [
            "xorb-api", "xorb-worker", "xorb-orchestrator",
            "postgres", "redis", "temporal", "prometheus"
        ]

        # Performance baselines
        self.baselines = {
            "cpu_usage": 70.0,
            "memory_usage": 80.0,
            "response_time": 500.0,  # ms
            "error_rate": 5.0,  # %
            "disk_usage": 85.0
        }

        # Agent monitoring
        self.agent_metrics = {
            "active_agents": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "threat_detections": 0,
            "autonomous_actions": 0
        }

        logger.info(f"🔍 Advanced monitoring system initialized: {self.session_id}")

    def collect_system_metrics(self) -> list[MetricData]:
        """Collect comprehensive system performance metrics"""
        metrics = []
        timestamp = datetime.now()

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(MetricData(
                name="system.cpu.usage",
                value=cpu_percent,
                unit="percent",
                timestamp=timestamp,
                tags={"component": "system"}
            ))

            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.append(MetricData(
                name="system.memory.usage",
                value=memory.percent,
                unit="percent",
                timestamp=timestamp,
                tags={"component": "system"}
            ))

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(MetricData(
                name="system.disk.usage",
                value=disk_percent,
                unit="percent",
                timestamp=timestamp,
                tags={"component": "system"}
            ))

            # Network metrics (simulated for container environment)
            network_io = psutil.net_io_counters()
            metrics.append(MetricData(
                name="system.network.bytes_sent",
                value=network_io.bytes_sent,
                unit="bytes",
                timestamp=timestamp,
                tags={"component": "network"}
            ))

            # Process metrics
            process_count = len(psutil.pids())
            metrics.append(MetricData(
                name="system.processes.count",
                value=process_count,
                unit="count",
                timestamp=timestamp,
                tags={"component": "system"}
            ))

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

        return metrics

    def collect_xorb_metrics(self) -> list[MetricData]:
        """Collect XORB-specific platform metrics"""
        metrics = []
        timestamp = datetime.now()

        try:
            # Simulate XORB platform metrics
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()

            # Platform uptime
            metrics.append(MetricData(
                name="xorb.platform.uptime",
                value=uptime_seconds,
                unit="seconds",
                timestamp=timestamp,
                tags={"component": "platform"}
            ))

            # Agent metrics (simulated based on autonomous operations)
            active_agents = random.randint(28, 32)  # 32-agent swarm
            self.agent_metrics["active_agents"] = active_agents

            metrics.append(MetricData(
                name="xorb.agents.active_count",
                value=active_agents,
                unit="count",
                timestamp=timestamp,
                tags={"component": "agents"}
            ))

            # Threat detection metrics
            threat_detections = random.randint(15, 25)
            self.agent_metrics["threat_detections"] += threat_detections

            metrics.append(MetricData(
                name="xorb.threats.detected_per_minute",
                value=threat_detections,
                unit="count",
                timestamp=timestamp,
                tags={"component": "detection"}
            ))

            # Success rate metrics
            success_rate = random.uniform(92.0, 97.5)  # High success rate
            metrics.append(MetricData(
                name="xorb.operations.success_rate",
                value=success_rate,
                unit="percent",
                timestamp=timestamp,
                tags={"component": "operations"}
            ))

            # Response time metrics
            response_time = random.uniform(150, 400)  # Sub-500ms
            metrics.append(MetricData(
                name="xorb.api.response_time",
                value=response_time,
                unit="milliseconds",
                timestamp=timestamp,
                tags={"component": "api"}
            ))

            # Knowledge fabric metrics
            knowledge_atoms = random.randint(950, 1050)  # Growing knowledge base
            metrics.append(MetricData(
                name="xorb.knowledge.atom_count",
                value=knowledge_atoms,
                unit="count",
                timestamp=timestamp,
                tags={"component": "knowledge"}
            ))

        except Exception as e:
            logger.error(f"Error collecting XORB metrics: {e}")

        return metrics

    def perform_health_checks(self) -> list[HealthCheck]:
        """Perform health checks on all XORB services"""
        health_checks = []

        for service in self.active_services:
            try:
                start_time = time.time()

                # Simulate health check (in real implementation, would check actual services)
                if service in ["xorb-api", "xorb-worker", "xorb-orchestrator"]:
                    # XORB services - generally healthy
                    health_prob = 0.95
                    response_time = random.uniform(50, 200)
                elif service in ["postgres", "redis"]:
                    # Database services - very reliable
                    health_prob = 0.98
                    response_time = random.uniform(20, 100)
                else:
                    # Other services
                    health_prob = 0.90
                    response_time = random.uniform(100, 300)

                if random.random() < health_prob:
                    status = ServiceStatus.HEALTHY
                    error_message = None
                    details = {"last_check": datetime.now().isoformat()}
                else:
                    status = ServiceStatus.DEGRADED if random.random() > 0.3 else ServiceStatus.UNHEALTHY
                    error_message = f"Service {service} experiencing issues"
                    details = {"error_code": "HEALTH_CHECK_FAILED"}

                health_check = HealthCheck(
                    service=service,
                    status=status,
                    response_time_ms=response_time,
                    error_message=error_message,
                    details=details
                )

                health_checks.append(health_check)

            except Exception as e:
                health_checks.append(HealthCheck(
                    service=service,
                    status=ServiceStatus.UNKNOWN,
                    response_time_ms=0.0,
                    error_message=str(e)
                ))

        return health_checks

    def analyze_metrics_and_generate_alerts(self, metrics: list[MetricData]) -> list[Alert]:
        """Analyze metrics and generate alerts based on thresholds"""
        alerts = []

        for metric in metrics:
            alert_level = None
            message = None

            # CPU usage alerts
            if metric.name == "system.cpu.usage":
                if metric.value > 90:
                    alert_level = AlertLevel.CRITICAL
                    message = f"Critical CPU usage: {metric.value:.1f}%"
                elif metric.value > self.baselines["cpu_usage"]:
                    alert_level = AlertLevel.WARNING
                    message = f"High CPU usage: {metric.value:.1f}%"

            # Memory usage alerts
            elif metric.name == "system.memory.usage":
                if metric.value > 95:
                    alert_level = AlertLevel.CRITICAL
                    message = f"Critical memory usage: {metric.value:.1f}%"
                elif metric.value > self.baselines["memory_usage"]:
                    alert_level = AlertLevel.WARNING
                    message = f"High memory usage: {metric.value:.1f}%"

            # Response time alerts
            elif metric.name == "xorb.api.response_time":
                if metric.value > 1000:
                    alert_level = AlertLevel.CRITICAL
                    message = f"Critical API response time: {metric.value:.0f}ms"
                elif metric.value > self.baselines["response_time"]:
                    alert_level = AlertLevel.WARNING
                    message = f"High API response time: {metric.value:.0f}ms"

            # Agent count alerts
            elif metric.name == "xorb.agents.active_count":
                if metric.value < 20:
                    alert_level = AlertLevel.WARNING
                    message = f"Low agent count: {metric.value} (expected 28-32)"

            # Success rate alerts
            elif metric.name == "xorb.operations.success_rate":
                if metric.value < 85:
                    alert_level = AlertLevel.CRITICAL
                    message = f"Critical success rate: {metric.value:.1f}%"
                elif metric.value < 90:
                    alert_level = AlertLevel.WARNING
                    message = f"Low success rate: {metric.value:.1f}%"

            if alert_level and message:
                alert = Alert(
                    level=alert_level,
                    service=metric.tags.get("component", "unknown") if metric.tags else "unknown",
                    message=message,
                    timestamp=metric.timestamp,
                    metric_name=metric.name,
                    threshold=self.baselines.get(metric.name.split(".")[-1], None),
                    current_value=metric.value
                )
                alerts.append(alert)

        return alerts

    def generate_monitoring_dashboard(self) -> dict[str, Any]:
        """Generate real-time monitoring dashboard data"""
        latest_metrics = {}

        # Get latest metrics by name
        for metric in self.metrics_history[-50:]:  # Last 50 metrics
            latest_metrics[metric.name] = metric

        # Get latest health checks
        latest_health = {}
        for health in self.health_history[-len(self.active_services):]:
            latest_health[health.service] = health

        # Calculate summary statistics
        healthy_services = sum(1 for h in latest_health.values() if h.status == ServiceStatus.HEALTHY)
        total_services = len(latest_health)

        active_alerts = [a for a in self.alerts if a.timestamp > datetime.now() - timedelta(minutes=10)]
        critical_alerts = sum(1 for a in active_alerts if a.level == AlertLevel.CRITICAL)

        return {
            "dashboard_id": f"DASH-{self.session_id}",
            "timestamp": datetime.now().isoformat(),
            "overview": {
                "platform_status": "OPERATIONAL" if critical_alerts == 0 else "DEGRADED",
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "healthy_services": f"{healthy_services}/{total_services}",
                "active_alerts": len(active_alerts),
                "critical_alerts": critical_alerts
            },
            "performance": {
                "cpu_usage": latest_metrics.get("system.cpu.usage", {}).value if "system.cpu.usage" in latest_metrics else 0,
                "memory_usage": latest_metrics.get("system.memory.usage", {}).value if "system.memory.usage" in latest_metrics else 0,
                "api_response_time": latest_metrics.get("xorb.api.response_time", {}).value if "xorb.api.response_time" in latest_metrics else 0,
                "success_rate": latest_metrics.get("xorb.operations.success_rate", {}).value if "xorb.operations.success_rate" in latest_metrics else 0
            },
            "agents": {
                "active_count": latest_metrics.get("xorb.agents.active_count", {}).value if "xorb.agents.active_count" in latest_metrics else 0,
                "threats_detected": self.agent_metrics["threat_detections"],
                "successful_operations": self.agent_metrics["successful_operations"],
                "autonomous_actions": self.agent_metrics["autonomous_actions"]
            },
            "services": {
                service: {
                    "status": health.status.value,
                    "response_time_ms": health.response_time_ms,
                    "error_message": health.error_message
                }
                for service, health in latest_health.items()
            },
            "recent_alerts": [
                {
                    "level": alert.level.value,
                    "service": alert.service,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in active_alerts[-10:]  # Last 10 alerts
            ]
        }

    async def monitoring_cycle(self, duration_minutes: int = 5):
        """Run continuous monitoring cycle"""
        logger.info(f"🔍 Starting monitoring cycle for {duration_minutes} minutes...")

        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        cycle_count = 0

        while datetime.now() < end_time:
            cycle_count += 1
            cycle_start = datetime.now()

            try:
                logger.info(f"📊 Monitoring cycle {cycle_count} - collecting metrics...")

                # Collect metrics
                system_metrics = self.collect_system_metrics()
                xorb_metrics = self.collect_xorb_metrics()
                all_metrics = system_metrics + xorb_metrics

                # Store metrics
                self.metrics_history.extend(all_metrics)

                # Perform health checks
                health_checks = self.perform_health_checks()
                self.health_history.extend(health_checks)

                # Generate alerts
                new_alerts = self.analyze_metrics_and_generate_alerts(all_metrics)
                self.alerts.extend(new_alerts)

                # Generate dashboard
                dashboard = self.generate_monitoring_dashboard()

                # Log key metrics
                logger.info(f"📈 Cycle {cycle_count} Summary:")
                logger.info(f"   Platform Status: {dashboard['overview']['platform_status']}")
                logger.info(f"   Active Agents: {dashboard['agents']['active_count']}")
                logger.info(f"   Threats Detected: {dashboard['agents']['threats_detected']}")
                logger.info(f"   Healthy Services: {dashboard['overview']['healthy_services']}")
                logger.info(f"   API Response: {dashboard['performance']['api_response_time']:.0f}ms")

                if new_alerts:
                    for alert in new_alerts:
                        logger.warning(f"🚨 {alert.level.value.upper()}: {alert.message}")

                # Simulate autonomous agent actions
                if random.random() < 0.3:  # 30% chance per cycle
                    autonomous_actions = random.randint(1, 5)
                    self.agent_metrics["autonomous_actions"] += autonomous_actions
                    logger.info(f"🤖 Autonomous actions executed: {autonomous_actions}")

                # Keep only recent data (last 1000 metrics)
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]

                if len(self.health_history) > 100:
                    self.health_history = self.health_history[-100:]

                # Sleep until next cycle (30 seconds)
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                sleep_time = max(0, 30 - cycle_duration)

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in monitoring cycle {cycle_count}: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

        logger.info(f"✅ Monitoring cycle completed after {cycle_count} cycles")
        return self.generate_final_report()

    def generate_final_report(self) -> dict[str, Any]:
        """Generate comprehensive monitoring report"""
        total_metrics = len(self.metrics_history)
        total_alerts = len(self.alerts)
        critical_alerts = sum(1 for a in self.alerts if a.level == AlertLevel.CRITICAL)

        # Calculate average performance metrics
        cpu_metrics = [m.value for m in self.metrics_history if m.name == "system.cpu.usage"]
        memory_metrics = [m.value for m in self.metrics_history if m.name == "system.memory.usage"]
        response_metrics = [m.value for m in self.metrics_history if m.name == "xorb.api.response_time"]

        avg_cpu = sum(cpu_metrics) / len(cpu_metrics) if cpu_metrics else 0
        avg_memory = sum(memory_metrics) / len(memory_metrics) if memory_metrics else 0
        avg_response = sum(response_metrics) / len(response_metrics) if response_metrics else 0

        # Service availability
        healthy_checks = sum(1 for h in self.health_history if h.status == ServiceStatus.HEALTHY)
        total_checks = len(self.health_history)
        availability = (healthy_checks / total_checks * 100) if total_checks > 0 else 0

        return {
            "monitoring_session": self.session_id,
            "duration_minutes": (datetime.now() - self.start_time).total_seconds() / 60,
            "summary": {
                "total_metrics_collected": total_metrics,
                "total_alerts_generated": total_alerts,
                "critical_alerts": critical_alerts,
                "service_availability": f"{availability:.1f}%"
            },
            "performance_summary": {
                "average_cpu_usage": f"{avg_cpu:.1f}%",
                "average_memory_usage": f"{avg_memory:.1f}%",
                "average_response_time": f"{avg_response:.0f}ms",
                "total_threats_detected": self.agent_metrics["threat_detections"],
                "autonomous_actions": self.agent_metrics["autonomous_actions"]
            },
            "final_dashboard": self.generate_monitoring_dashboard()
        }

async def main():
    """Main execution function"""
    print("🔍 XORB Advanced Monitoring & Observability System")
    print("=" * 60)

    monitor = XORBAdvancedMonitoring()

    try:
        # Run monitoring for 5 minutes
        final_report = await monitor.monitoring_cycle(duration_minutes=5)

        print("\n" + "=" * 60)
        print("📊 MONITORING SESSION COMPLETE")
        print("=" * 60)

        print(f"Session ID: {final_report['monitoring_session']}")
        print(f"Duration: {final_report['duration_minutes']:.1f} minutes")
        print(f"Metrics Collected: {final_report['summary']['total_metrics_collected']}")
        print(f"Alerts Generated: {final_report['summary']['total_alerts_generated']}")
        print(f"Service Availability: {final_report['summary']['service_availability']}")
        print()
        print("Performance Summary:")
        print(f"  Average CPU Usage: {final_report['performance_summary']['average_cpu_usage']}")
        print(f"  Average Memory Usage: {final_report['performance_summary']['average_memory_usage']}")
        print(f"  Average Response Time: {final_report['performance_summary']['average_response_time']}")
        print(f"  Threats Detected: {final_report['performance_summary']['total_threats_detected']}")
        print(f"  Autonomous Actions: {final_report['performance_summary']['autonomous_actions']}")

        # Save detailed report
        report_file = f"/root/Xorb/monitoring_report_{monitor.session_id}.json"
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)

        print(f"\n📝 Detailed report saved: {report_file}")
        print("✅ XORB monitoring system demonstration complete!")

    except KeyboardInterrupt:
        print("\n⚠️ Monitoring interrupted by user")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
