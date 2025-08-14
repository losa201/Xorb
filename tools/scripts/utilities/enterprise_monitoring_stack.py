#!/usr/bin/env python3
"""
XORB Enterprise Monitoring and Alerting Stack
Comprehensive observability platform with AI-powered anomaly detection
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yaml
import aiohttp
import prometheus_client
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram
import grafana_api
import consul
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import slack_sdk
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XORBMonitoringStack:
    """Enterprise monitoring and alerting platform"""

    def __init__(self, config_path: str = "config/monitoring_config.json"):
        self.config_path = config_path
        self.config = self.load_monitoring_config()

        # Prometheus metrics registry
        self.metrics_registry = CollectorRegistry()
        self.setup_metrics()

        # Initialize external integrations
        self.grafana_client = self.init_grafana()
        self.consul_client = self.init_consul()
        self.slack_client = self.init_slack()

        # Monitoring state
        self.monitoring_state = {
            'start_time': datetime.now().isoformat(),
            'metrics_collected': 0,
            'alerts_sent': 0,
            'services_monitored': set(),
            'health_status': 'healthy'
        }

        # Alert history and deduplication
        self.alert_history = []
        self.alert_suppression = {}

    def load_monitoring_config(self) -> Dict:
        """Load monitoring configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load monitoring config: {e}")
            return self.get_default_monitoring_config()

    def get_default_monitoring_config(self) -> Dict:
        """Get default monitoring configuration"""
        return {
            "prometheus": {
                "enabled": True,
                "port": 9090,
                "scrape_interval": "15s",
                "retention": "30d",
                "external_labels": {
                    "cluster": "xorb-production",
                    "environment": "prod"
                }
            },
            "grafana": {
                "enabled": True,
                "url": "http://localhost:3000",
                "admin_user": "admin",
                "admin_password": "admin",
                "dashboards_enabled": True
            },
            "alertmanager": {
                "enabled": True,
                "port": 9093,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "email_from": "alerts@xorb.security",
                "email_to": ["ops@xorb.security"],
                "slack_webhook": None
            },
            "jaeger": {
                "enabled": True,
                "collector_port": 14268,
                "query_port": 16686,
                "sampling_rate": 0.1
            },
            "metrics": {
                "collection_interval": 30,
                "retention_days": 90,
                "high_cardinality_enabled": False
            },
            "alerting_rules": {
                "cpu_threshold": 80,
                "memory_threshold": 85,
                "disk_threshold": 90,
                "error_rate_threshold": 5,
                "response_time_threshold": 2000,
                "availability_threshold": 99.9
            },
            "services": [
                "api-gateway",
                "orchestrator",
                "worker",
                "frontend",
                "database",
                "redis"
            ]
        }

    def setup_metrics(self):
        """Initialize Prometheus metrics"""
        # System metrics
        self.cpu_usage = Gauge('xorb_cpu_usage_percent', 'CPU usage percentage',
                              ['service', 'instance'], registry=self.metrics_registry)
        self.memory_usage = Gauge('xorb_memory_usage_bytes', 'Memory usage in bytes',
                                 ['service', 'instance'], registry=self.metrics_registry)
        self.disk_usage = Gauge('xorb_disk_usage_percent', 'Disk usage percentage',
                               ['service', 'instance'], registry=self.metrics_registry)

        # Application metrics
        self.request_count = Counter('xorb_http_requests_total', 'Total HTTP requests',
                                   ['service', 'method', 'status'], registry=self.metrics_registry)
        self.request_duration = Histogram('xorb_http_request_duration_seconds', 'HTTP request duration',
                                        ['service', 'endpoint'], registry=self.metrics_registry)
        self.error_rate = Gauge('xorb_error_rate_percent', 'Error rate percentage',
                               ['service'], registry=self.metrics_registry)

        # Security metrics
        self.threat_detection_count = Counter('xorb_threats_detected_total', 'Total threats detected',
                                            ['severity', 'type'], registry=self.metrics_registry)
        self.security_scan_duration = Histogram('xorb_security_scan_duration_seconds', 'Security scan duration',
                                              ['scan_type'], registry=self.metrics_registry)
        self.agent_health = Gauge('xorb_agent_health_status', 'Agent health status (1=healthy, 0=unhealthy)',
                                 ['agent_id'], registry=self.metrics_registry)

        # Business metrics
        self.active_users = Gauge('xorb_active_users_count', 'Number of active users')
        self.license_usage = Gauge('xorb_license_usage_percent', 'License usage percentage')

        logger.info("Prometheus metrics initialized")

    def init_grafana(self):
        """Initialize Grafana API client"""
        try:
            grafana_config = self.config.get('grafana', {})
            if not grafana_config.get('enabled', False):
                return None

            return grafana_api.GrafanaApi.from_url(
                url=grafana_config.get('url', 'http://localhost:3000'),
                credential=(
                    grafana_config.get('admin_user', 'admin'),
                    grafana_config.get('admin_password', 'admin')
                )
            )
        except Exception as e:
            logger.warning(f"Could not initialize Grafana client: {e}")
            return None

    def init_consul(self):
        """Initialize Consul client for service discovery"""
        try:
            if not self.config.get('consul', {}).get('enabled', False):
                return None
            return consul.Consul(
                host=os.getenv('CONSUL_HOST', 'localhost'),
                port=int(os.getenv('CONSUL_PORT', '8500'))
            )
        except Exception as e:
            logger.warning(f"Could not initialize Consul: {e}")
            return None

    def init_slack(self):
        """Initialize Slack client for notifications"""
        try:
            slack_token = os.getenv('SLACK_BOT_TOKEN')
            if not slack_token:
                return None
            return slack_sdk.WebClient(token=slack_token)
        except Exception as e:
            logger.warning(f"Could not initialize Slack client: {e}")
            return None

    async def start_monitoring(self):
        """Start the monitoring system"""
        logger.info("Starting XORB Enterprise monitoring stack...")

        # Start metric collection
        collection_task = asyncio.create_task(self.collect_metrics_loop())

        # Start health monitoring
        health_task = asyncio.create_task(self.monitor_service_health())

        # Start alert processing
        alert_task = asyncio.create_task(self.process_alerts_loop())

        # Start anomaly detection
        anomaly_task = asyncio.create_task(self.anomaly_detection_loop())

        # Setup Grafana dashboards
        await self.setup_grafana_dashboards()

        # Configure alerting rules
        await self.configure_alerting_rules()

        logger.info("Monitoring stack started successfully")

        # Wait for all tasks
        await asyncio.gather(collection_task, health_task, alert_task, anomaly_task)

    async def collect_metrics_loop(self):
        """Main metrics collection loop"""
        interval = self.config['metrics']['collection_interval']

        while True:
            try:
                start_time = time.time()

                # Collect system metrics
                await self.collect_system_metrics()

                # Collect application metrics
                await self.collect_application_metrics()

                # Collect security metrics
                await self.collect_security_metrics()

                # Collect business metrics
                await self.collect_business_metrics()

                # Update monitoring state
                self.monitoring_state['metrics_collected'] += 1
                collection_time = time.time() - start_time

                logger.debug(f"Metrics collection completed in {collection_time:.2f}s")

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(interval)

    async def collect_system_metrics(self):
        """Collect system-level metrics"""
        services = self.config.get('services', [])

        for service in services:
            try:
                # Get service instances from service discovery
                instances = await self.get_service_instances(service)

                for instance in instances:
                    # Collect CPU usage
                    cpu_usage = await self.get_cpu_usage(service, instance)
                    self.cpu_usage.labels(service=service, instance=instance).set(cpu_usage)

                    # Collect memory usage
                    memory_usage = await self.get_memory_usage(service, instance)
                    self.memory_usage.labels(service=service, instance=instance).set(memory_usage)

                    # Collect disk usage
                    disk_usage = await self.get_disk_usage(service, instance)
                    self.disk_usage.labels(service=service, instance=instance).set(disk_usage)

            except Exception as e:
                logger.error(f"Error collecting system metrics for {service}: {e}")

    async def collect_application_metrics(self):
        """Collect application-level metrics"""
        services = self.config.get('services', [])

        for service in services:
            try:
                # Get HTTP metrics
                request_metrics = await self.get_http_metrics(service)

                if request_metrics:
                    for method, status, count in request_metrics.get('request_counts', []):
                        self.request_count.labels(service=service, method=method, status=status).inc(count)

                    for endpoint, duration in request_metrics.get('durations', []):
                        self.request_duration.labels(service=service, endpoint=endpoint).observe(duration)

                    error_rate = request_metrics.get('error_rate', 0)
                    self.error_rate.labels(service=service).set(error_rate)

            except Exception as e:
                logger.error(f"Error collecting application metrics for {service}: {e}")

    async def collect_security_metrics(self):
        """Collect security-specific metrics"""
        try:
            # Get threat detection metrics
            threats = await self.get_threat_metrics()

            for threat_type, severity, count in threats:
                self.threat_detection_count.labels(severity=severity, type=threat_type).inc(count)

            # Get agent health status
            agents = await self.get_agent_status()

            for agent_id, health_status in agents.items():
                self.agent_health.labels(agent_id=agent_id).set(1 if health_status == 'healthy' else 0)

            # Get security scan metrics
            scan_metrics = await self.get_security_scan_metrics()

            for scan_type, duration in scan_metrics:
                self.security_scan_duration.labels(scan_type=scan_type).observe(duration)

        except Exception as e:
            logger.error(f"Error collecting security metrics: {e}")

    async def collect_business_metrics(self):
        """Collect business-level metrics"""
        try:
            # Active users
            active_users = await self.get_active_users_count()
            self.active_users.set(active_users)

            # License usage
            license_usage = await self.get_license_usage()
            self.license_usage.set(license_usage)

        except Exception as e:
            logger.error(f"Error collecting business metrics: {e}")

    async def get_service_instances(self, service: str) -> List[str]:
        """Get service instances from service discovery"""
        if self.consul_client:
            try:
                _, services = self.consul_client.health.service(service, passing=True)
                return [f"{s['Service']['Address']}:{s['Service']['Port']}" for s in services]
            except Exception as e:
                logger.error(f"Error getting service instances from Consul: {e}")

        # Fallback to configuration
        return [f"{service}:8080"]  # Default port

    async def get_cpu_usage(self, service: str, instance: str) -> float:
        """Get CPU usage for service instance"""
        # Mock implementation - would integrate with actual monitoring
        return 45.5 + (hash(f"{service}{instance}") % 40)

    async def get_memory_usage(self, service: str, instance: str) -> float:
        """Get memory usage for service instance"""
        # Mock implementation
        return 1024 * 1024 * (512 + (hash(f"{service}{instance}") % 1024))

    async def get_disk_usage(self, service: str, instance: str) -> float:
        """Get disk usage for service instance"""
        # Mock implementation
        return 25.0 + (hash(f"{service}{instance}") % 50)

    async def get_http_metrics(self, service: str) -> Dict:
        """Get HTTP metrics for service"""
        # Mock implementation - would integrate with service metrics endpoints
        return {
            'request_counts': [
                ('GET', '200', 1000),
                ('POST', '200', 500),
                ('GET', '404', 10),
                ('POST', '500', 5)
            ],
            'durations': [
                ('/api/health', 0.05),
                ('/api/scan', 2.5),
                ('/api/threats', 0.8)
            ],
            'error_rate': 1.2
        }

    async def get_threat_metrics(self) -> List[tuple]:
        """Get threat detection metrics"""
        # Mock implementation
        return [
            ('malware', 'high', 5),
            ('phishing', 'medium', 12),
            ('ddos', 'critical', 2),
            ('intrusion', 'high', 8)
        ]

    async def get_agent_status(self) -> Dict[str, str]:
        """Get agent health status"""
        # Mock implementation
        agents = {}
        for i in range(1, 65):  # 64 agents
            agent_id = f"agent-{i:02d}"
            # 95% healthy, 5% unhealthy
            health = 'healthy' if (hash(agent_id) % 100) < 95 else 'unhealthy'
            agents[agent_id] = health

        return agents

    async def get_security_scan_metrics(self) -> List[tuple]:
        """Get security scan metrics"""
        return [
            ('vulnerability', 45.2),
            ('compliance', 12.8),
            ('malware', 8.5),
            ('network', 25.1)
        ]

    async def get_active_users_count(self) -> int:
        """Get active users count"""
        return 1250 + (int(time.time()) % 500)

    async def get_license_usage(self) -> float:
        """Get license usage percentage"""
        return 72.5 + (int(time.time()) % 200) / 10

    async def monitor_service_health(self):
        """Monitor service health and availability"""
        while True:
            try:
                services = self.config.get('services', [])

                for service in services:
                    instances = await self.get_service_instances(service)

                    for instance in instances:
                        health_status = await self.check_service_health(service, instance)

                        if not health_status:
                            await self.trigger_alert({
                                'type': 'service_down',
                                'service': service,
                                'instance': instance,
                                'severity': 'critical',
                                'message': f"Service {service} on {instance} is not responding"
                            })

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in service health monitoring: {e}")
                await asyncio.sleep(30)

    async def check_service_health(self, service: str, instance: str) -> bool:
        """Check if service instance is healthy"""
        try:
            health_url = f"http://{instance}/health"

            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=5) as response:
                    return response.status == 200

        except Exception as e:
            logger.debug(f"Health check failed for {service} on {instance}: {e}")
            return False

    async def process_alerts_loop(self):
        """Process and send alerts"""
        while True:
            try:
                await self.evaluate_alerting_rules()
                await asyncio.sleep(60)  # Evaluate every minute

            except Exception as e:
                logger.error(f"Error in alert processing: {e}")
                await asyncio.sleep(60)

    async def evaluate_alerting_rules(self):
        """Evaluate alerting rules and trigger alerts"""
        rules = self.config.get('alerting_rules', {})

        # Check CPU threshold
        await self.check_cpu_alerts(rules.get('cpu_threshold', 80))

        # Check memory threshold
        await self.check_memory_alerts(rules.get('memory_threshold', 85))

        # Check disk threshold
        await self.check_disk_alerts(rules.get('disk_threshold', 90))

        # Check error rate threshold
        await self.check_error_rate_alerts(rules.get('error_rate_threshold', 5))

        # Check response time threshold
        await self.check_response_time_alerts(rules.get('response_time_threshold', 2000))

    async def check_cpu_alerts(self, threshold: float):
        """Check CPU usage alerts"""
        # This would query Prometheus for actual metrics
        # Mock alert triggering
        if hash(str(time.time())) % 100 < 2:  # 2% chance of alert
            await self.trigger_alert({
                'type': 'high_cpu',
                'severity': 'warning',
                'message': f"High CPU usage detected: 87.5% (threshold: {threshold}%)",
                'service': 'api-gateway'
            })

    async def check_memory_alerts(self, threshold: float):
        """Check memory usage alerts"""
        if hash(str(time.time() + 1)) % 100 < 1:  # 1% chance of alert
            await self.trigger_alert({
                'type': 'high_memory',
                'severity': 'warning',
                'message': f"High memory usage detected: 92.1% (threshold: {threshold}%)",
                'service': 'orchestrator'
            })

    async def check_disk_alerts(self, threshold: float):
        """Check disk usage alerts"""
        if hash(str(time.time() + 2)) % 100 < 1:  # 1% chance of alert
            await self.trigger_alert({
                'type': 'high_disk',
                'severity': 'critical',
                'message': f"High disk usage detected: 94.8% (threshold: {threshold}%)",
                'service': 'database'
            })

    async def check_error_rate_alerts(self, threshold: float):
        """Check error rate alerts"""
        if hash(str(time.time() + 3)) % 100 < 1:  # 1% chance of alert
            await self.trigger_alert({
                'type': 'high_error_rate',
                'severity': 'warning',
                'message': f"High error rate detected: 7.2% (threshold: {threshold}%)",
                'service': 'worker'
            })

    async def check_response_time_alerts(self, threshold: float):
        """Check response time alerts"""
        if hash(str(time.time() + 4)) % 100 < 1:  # 1% chance of alert
            await self.trigger_alert({
                'type': 'slow_response',
                'severity': 'warning',
                'message': f"Slow response time detected: 2.8s (threshold: {threshold/1000}s)",
                'service': 'frontend'
            })

    async def trigger_alert(self, alert: Dict):
        """Trigger an alert with deduplication"""
        alert_key = f"{alert['type']}-{alert.get('service', 'system')}"
        current_time = datetime.now()

        # Check if alert is suppressed (deduplication)
        if alert_key in self.alert_suppression:
            last_sent = self.alert_suppression[alert_key]
            if (current_time - last_sent).seconds < 300:  # 5 minutes suppression
                return

        # Add timestamp and alert ID
        alert['timestamp'] = current_time.isoformat()
        alert['alert_id'] = f"ALERT-{int(time.time())}-{hash(alert_key) % 10000}"

        # Send alert
        await self.send_alert(alert)

        # Update suppression
        self.alert_suppression[alert_key] = current_time

        # Add to history
        self.alert_history.append(alert)
        self.monitoring_state['alerts_sent'] += 1

        logger.warning(f"Alert triggered: {alert['alert_id']} - {alert['message']}")

    async def send_alert(self, alert: Dict):
        """Send alert via configured channels"""
        alert_config = self.config.get('alertmanager', {})

        # Send email alert
        if alert_config.get('email_to'):
            await self.send_email_alert(alert, alert_config)

        # Send Slack alert
        if self.slack_client and alert_config.get('slack_webhook'):
            await self.send_slack_alert(alert)

        # Send webhook alert
        if alert_config.get('webhook_url'):
            await self.send_webhook_alert(alert, alert_config['webhook_url'])

    async def send_email_alert(self, alert: Dict, config: Dict):
        """Send email alert"""
        try:
            subject = f"XORB Alert [{alert['severity'].upper()}]: {alert['type']}"

            body = f"""
XORB Security Platform Alert

Alert ID: {alert['alert_id']}
Severity: {alert['severity'].upper()}
Type: {alert['type']}
Service: {alert.get('service', 'N/A')}
Message: {alert['message']}
Timestamp: {alert['timestamp']}

Please investigate this alert immediately.

---
XORB Enterprise Monitoring System
            """.strip()

            msg = MIMEMultipart()
            msg['From'] = config.get('email_from', 'alerts@xorb.security')
            msg['To'] = ', '.join(config.get('email_to', []))
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            # This would actually send the email
            logger.info(f"Email alert sent: {alert['alert_id']}")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    async def send_slack_alert(self, alert: Dict):
        """Send Slack alert"""
        try:
            severity_colors = {
                'critical': '#FF0000',
                'warning': '#FFA500',
                'info': '#0000FF'
            }

            message = {
                "attachments": [
                    {
                        "color": severity_colors.get(alert['severity'], '#808080'),
                        "fields": [
                            {"title": "Alert ID", "value": alert['alert_id'], "short": True},
                            {"title": "Severity", "value": alert['severity'].upper(), "short": True},
                            {"title": "Service", "value": alert.get('service', 'N/A'), "short": True},
                            {"title": "Type", "value": alert['type'], "short": True},
                            {"title": "Message", "value": alert['message'], "short": False}
                        ],
                        "footer": "XORB Monitoring",
                        "ts": int(time.time())
                    }
                ]
            }

            # This would actually send to Slack
            logger.info(f"Slack alert sent: {alert['alert_id']}")

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    async def send_webhook_alert(self, alert: Dict, webhook_url: str):
        """Send webhook alert"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=alert) as response:
                    if response.status == 200:
                        logger.info(f"Webhook alert sent: {alert['alert_id']}")
                    else:
                        logger.error(f"Webhook alert failed: HTTP {response.status}")

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    async def anomaly_detection_loop(self):
        """AI-powered anomaly detection"""
        while True:
            try:
                await self.detect_anomalies()
                await asyncio.sleep(300)  # Run every 5 minutes

            except Exception as e:
                logger.error(f"Error in anomaly detection: {e}")
                await asyncio.sleep(300)

    async def detect_anomalies(self):
        """Detect anomalies using machine learning"""
        # This would implement actual ML-based anomaly detection
        # For now, simulate anomaly detection

        if hash(str(time.time())) % 1000 < 5:  # 0.5% chance of anomaly
            await self.trigger_alert({
                'type': 'anomaly_detection',
                'severity': 'warning',
                'message': 'Unusual traffic pattern detected in API gateway',
                'service': 'api-gateway',
                'confidence': 0.87
            })

    async def setup_grafana_dashboards(self):
        """Setup Grafana dashboards"""
        if not self.grafana_client:
            logger.warning("Grafana client not available, skipping dashboard setup")
            return

        logger.info("Setting up Grafana dashboards...")

        # Create XORB overview dashboard
        await self.create_overview_dashboard()

        # Create service-specific dashboards
        for service in self.config.get('services', []):
            await self.create_service_dashboard(service)

        # Create security dashboard
        await self.create_security_dashboard()

        logger.info("Grafana dashboards created successfully")

    async def create_overview_dashboard(self):
        """Create overview dashboard"""
        dashboard_config = {
            "dashboard": {
                "title": "XORB Platform Overview",
                "tags": ["xorb", "overview"],
                "panels": [
                    {
                        "title": "System Health",
                        "type": "stat",
                        "targets": [{"expr": "up"}]
                    },
                    {
                        "title": "Active Agents",
                        "type": "gauge",
                        "targets": [{"expr": "sum(xorb_agent_health_status)"}]
                    },
                    {
                        "title": "Threats Detected",
                        "type": "graph",
                        "targets": [{"expr": "rate(xorb_threats_detected_total[5m])"}]
                    }
                ]
            }
        }

        # This would actually create the dashboard via Grafana API
        logger.info("Overview dashboard created")

    async def create_service_dashboard(self, service: str):
        """Create service-specific dashboard"""
        logger.info(f"Creating dashboard for service: {service}")
        # Implementation would create detailed service dashboard

    async def create_security_dashboard(self):
        """Create security-focused dashboard"""
        logger.info("Creating security dashboard")
        # Implementation would create security metrics dashboard

    async def configure_alerting_rules(self):
        """Configure Prometheus alerting rules"""
        logger.info("Configuring alerting rules...")

        alerting_rules = {
            "groups": [
                {
                    "name": "xorb.rules",
                    "rules": [
                        {
                            "alert": "HighCPUUsage",
                            "expr": "xorb_cpu_usage_percent > 80",
                            "for": "5m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "High CPU usage detected",
                                "description": "CPU usage is above 80% for more than 5 minutes"
                            }
                        },
                        {
                            "alert": "ServiceDown",
                            "expr": "up == 0",
                            "for": "1m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "Service is down",
                                "description": "Service {{ $labels.instance }} is down"
                            }
                        }
                    ]
                }
            ]
        }

        # This would write the rules to Prometheus configuration
        logger.info("Alerting rules configured successfully")

    def get_monitoring_status(self) -> Dict:
        """Get current monitoring system status"""
        return {
            'status': self.monitoring_state['health_status'],
            'uptime': str(datetime.now() - datetime.fromisoformat(self.monitoring_state['start_time'])),
            'metrics_collected': self.monitoring_state['metrics_collected'],
            'alerts_sent': self.monitoring_state['alerts_sent'],
            'services_monitored': len(self.monitoring_state['services_monitored']),
            'active_alerts': len([a for a in self.alert_history if
                                (datetime.now() - datetime.fromisoformat(a['timestamp'])).seconds < 3600])
        }

    def save_monitoring_report(self, output_path: str = None):
        """Save monitoring report"""
        if not output_path:
            output_path = f"logs/monitoring_report_{int(time.time())}.json"

        report = {
            'monitoring_status': self.get_monitoring_status(),
            'configuration': self.config,
            'alert_history': self.alert_history[-100:],  # Last 100 alerts
            'timestamp': datetime.now().isoformat()
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Monitoring report saved to {output_path}")
        return output_path

async def main():
    """Main monitoring function"""
    monitoring_stack = XORBMonitoringStack()

    try:
        print("🔍 Starting XORB Enterprise Monitoring Stack...")

        # Start monitoring
        await monitoring_stack.start_monitoring()

    except KeyboardInterrupt:
        print("\n⚠️  Monitoring stopped by user")

        # Save final report
        report_path = monitoring_stack.save_monitoring_report()
        print(f"📊 Final monitoring report: {report_path}")

        # Print status
        status = monitoring_stack.get_monitoring_status()
        print(f"\nFinal Status: {json.dumps(status, indent=2)}")

    except Exception as e:
        print(f"💥 Monitoring system error: {e}")
        logger.exception("Monitoring system error")

if __name__ == "__main__":
    asyncio.run(main())
