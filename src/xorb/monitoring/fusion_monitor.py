#!/usr/bin/env python3
"""
XORB Post-Fusion Monitoring System
Advanced monitoring and validation of fused architecture performance
"""

import asyncio
import logging
import time
import psutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class MetricType(Enum):
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"
    RESOURCE_USAGE = "resource_usage"
    FUSION_INTEGRITY = "fusion_integrity"
    SECURITY = "security"

@dataclass
class HealthMetric:
    """Health metric for monitoring fused services."""
    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def status(self) -> HealthStatus:
        if self.value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.value >= self.threshold_warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY

@dataclass
class ServiceHealth:
    """Health assessment for a fused service."""
    service_name: str
    overall_status: HealthStatus
    metrics: List[HealthMetric]
    fusion_validation: Dict[str, bool]
    performance_baseline: Dict[str, float]
    last_check: datetime
    alerts: List[str] = field(default_factory=list)

class XORBFusionMonitor:
    """Advanced monitoring system for post-fusion architecture validation."""

    def __init__(self):
        self.service_health: Dict[str, ServiceHealth] = {}
        self.fusion_baseline: Dict[str, Dict[str, float]] = {}
        self.alert_history: List[Dict[str, Any]] = []

        # Monitoring configuration
        self.check_interval = 30  # seconds
        self.retention_period = timedelta(days=7)

        # Define fused services to monitor
        self.fused_services = [
            "pristine-intelligence-engine",  # Absorbed legacy-analytics-engine
            "legacy-ai-engine",             # Refactored
            "pristine-core-platform",       # Absorbed legacy-api-gateway
            "legacy-api-gateway"            # Final absorption target
        ]

    async def initialize(self):
        """Initialize monitoring system with baseline metrics."""
        logger.info("🔍 Initializing XORB Fusion Monitoring System")

        # Establish performance baselines for each fused service
        for service in self.fused_services:
            baseline = await self._establish_baseline(service)
            self.fusion_baseline[service] = baseline

        logger.info(f"Established baselines for {len(self.fused_services)} fused services")

    async def _establish_baseline(self, service_name: str) -> Dict[str, float]:
        """Establish performance baseline for a service."""
        logger.info(f"Establishing baseline for {service_name}")

        # Simulate baseline measurement
        baseline = {
            "response_time_ms": 150.0 + (hash(service_name) % 100),
            "cpu_usage_percent": 25.0 + (hash(service_name) % 20),
            "memory_usage_mb": 512.0 + (hash(service_name) % 256),
            "throughput_rps": 1000.0 + (hash(service_name) % 500),
            "error_rate_percent": 0.1 + (hash(service_name) % 5) / 10,
            "availability_percent": 99.95
        }

        return baseline

    async def start_monitoring(self):
        """Start continuous monitoring of fused services."""
        logger.info("🚀 Starting continuous fusion monitoring")

        while True:
            try:
                # Monitor all fused services
                for service in self.fused_services:
                    health = await self._check_service_health(service)
                    self.service_health[service] = health

                    # Generate alerts if needed
                    await self._process_alerts(service, health)

                # Generate monitoring report
                await self._generate_monitoring_report()

                # Wait for next check interval
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)  # Brief recovery pause

    async def _check_service_health(self, service_name: str) -> ServiceHealth:
        """Perform comprehensive health check for a fused service."""

        metrics = []
        fusion_validation = {}
        baseline = self.fusion_baseline.get(service_name, {})

        # Performance metrics
        perf_metrics = await self._collect_performance_metrics(service_name)
        for metric_name, value in perf_metrics.items():
            threshold_base = baseline.get(metric_name, value)

            metric = HealthMetric(
                name=f"{service_name}_{metric_name}",
                value=value,
                threshold_warning=threshold_base * 1.2,  # 20% degradation warning
                threshold_critical=threshold_base * 1.5,  # 50% degradation critical
                unit=self._get_metric_unit(metric_name)
            )
            metrics.append(metric)

        # Fusion integrity validation
        fusion_validation = await self._validate_fusion_integrity(service_name)

        # Resource usage metrics
        resource_metrics = await self._collect_resource_metrics(service_name)
        for metric_name, value in resource_metrics.items():
            metric = HealthMetric(
                name=f"{service_name}_resource_{metric_name}",
                value=value,
                threshold_warning=80.0,   # 80% resource usage warning
                threshold_critical=95.0,  # 95% resource usage critical
                unit="percent"
            )
            metrics.append(metric)

        # Determine overall health status
        overall_status = self._calculate_overall_health(metrics, fusion_validation)

        return ServiceHealth(
            service_name=service_name,
            overall_status=overall_status,
            metrics=metrics,
            fusion_validation=fusion_validation,
            performance_baseline=baseline,
            last_check=datetime.utcnow()
        )

    async def _collect_performance_metrics(self, service_name: str) -> Dict[str, float]:
        """Collect performance metrics for a service."""

        # Simulate realistic performance metrics
        base_response = 150.0
        variation = (hash(service_name + str(time.time())) % 100) / 10.0

        return {
            "response_time_ms": base_response + variation,
            "throughput_rps": 1000.0 + variation * 10,
            "error_rate_percent": max(0, 0.1 + variation / 100),
            "availability_percent": min(100, 99.9 + variation / 100)
        }

    async def _collect_resource_metrics(self, service_name: str) -> Dict[str, float]:
        """Collect resource usage metrics for a service."""

        # Use actual system metrics as proxy
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        # Simulate service-specific resource usage
        service_factor = (hash(service_name) % 50) / 100.0

        return {
            "cpu_usage": min(100, cpu_percent + service_factor * 20),
            "memory_usage": min(100, memory.percent + service_factor * 15),
            "disk_usage": min(100, psutil.disk_usage('/').percent + service_factor * 10)
        }

    async def _validate_fusion_integrity(self, service_name: str) -> Dict[str, bool]:
        """Validate that fusion was successful and services are properly integrated."""

        validation_results = {}

        # Simulate fusion integrity checks
        validation_results["service_discovery"] = True  # Service is discoverable
        validation_results["api_compatibility"] = True  # APIs remain compatible
        validation_results["data_consistency"] = True   # Data migration successful
        validation_results["dependency_resolution"] = True  # Dependencies properly resolved

        # Add service-specific validation
        if "intelligence-engine" in service_name:
            validation_results["ai_models_loaded"] = True
            validation_results["analytics_pipeline"] = True
        elif "core-platform" in service_name:
            validation_results["gateway_routing"] = True
            validation_results["auth_services"] = True
        elif "ai-engine" in service_name:
            validation_results["refactor_integrity"] = True
            validation_results["performance_optimization"] = True

        return validation_results

    def _calculate_overall_health(self, metrics: List[HealthMetric],
                                fusion_validation: Dict[str, bool]) -> HealthStatus:
        """Calculate overall health status from metrics and validation."""

        # Check for critical metrics
        critical_metrics = [m for m in metrics if m.status == HealthStatus.CRITICAL]
        if critical_metrics:
            return HealthStatus.CRITICAL

        # Check fusion validation failures
        validation_failures = [k for k, v in fusion_validation.items() if not v]
        if validation_failures:
            return HealthStatus.CRITICAL

        # Check for warning metrics
        warning_metrics = [m for m in metrics if m.status == HealthStatus.WARNING]
        if len(warning_metrics) > 2:  # Multiple warnings indicate degradation
            return HealthStatus.WARNING

        return HealthStatus.HEALTHY

    async def _process_alerts(self, service_name: str, health: ServiceHealth):
        """Process and generate alerts based on service health."""

        if health.overall_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
            alert = {
                "timestamp": datetime.utcnow().isoformat(),
                "service": service_name,
                "severity": health.overall_status.value,
                "metrics": [
                    {"name": m.name, "value": m.value, "status": m.status.value}
                    for m in health.metrics
                    if m.status != HealthStatus.HEALTHY
                ],
                "validation_failures": [
                    k for k, v in health.fusion_validation.items() if not v
                ]
            }

            self.alert_history.append(alert)
            logger.warning(f"🚨 {health.overall_status.value.upper()} alert for {service_name}")

    async def _generate_monitoring_report(self):
        """Generate comprehensive monitoring report."""

        healthy_services = sum(1 for h in self.service_health.values() if h.overall_status == HealthStatus.HEALTHY)
        total_services = len(self.service_health)

        if total_services > 0:
            health_percentage = (healthy_services / total_services) * 100

            logger.info(f"📊 Fusion Health Report: {healthy_services}/{total_services} services healthy ({health_percentage:.1f}%)")

            # Log any issues
            for service_name, health in self.service_health.items():
                if health.overall_status != HealthStatus.HEALTHY:
                    logger.warning(f"⚠️ {service_name}: {health.overall_status.value}")

    def _get_metric_unit(self, metric_name: str) -> str:
        """Get appropriate unit for metric type."""
        unit_map = {
            "response_time_ms": "ms",
            "throughput_rps": "rps",
            "error_rate_percent": "%",
            "availability_percent": "%",
            "cpu_usage": "%",
            "memory_usage": "%",
            "disk_usage": "%"
        }
        return unit_map.get(metric_name, "units")

    async def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""

        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_services": len(self.fused_services),
            "service_health": {},
            "overall_status": "healthy",
            "fusion_integrity": True,
            "performance_summary": {},
            "alerts_last_hour": 0
        }

        # Aggregate service health
        healthy_count = 0
        for service_name, health in self.service_health.items():
            summary["service_health"][service_name] = {
                "status": health.overall_status.value,
                "last_check": health.last_check.isoformat(),
                "metric_count": len(health.metrics),
                "validation_passed": all(health.fusion_validation.values())
            }

            if health.overall_status == HealthStatus.HEALTHY:
                healthy_count += 1
            elif health.overall_status == HealthStatus.CRITICAL:
                summary["overall_status"] = "critical"
                summary["fusion_integrity"] = False
            elif health.overall_status == HealthStatus.WARNING and summary["overall_status"] == "healthy":
                summary["overall_status"] = "warning"

        # Calculate health percentage
        if self.service_health:
            summary["health_percentage"] = (healthy_count / len(self.service_health)) * 100
        else:
            summary["health_percentage"] = 0

        # Count recent alerts
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert["timestamp"]) > one_hour_ago
        ]
        summary["alerts_last_hour"] = len(recent_alerts)

        return summary

# Global monitoring instance
fusion_monitor: Optional[XORBFusionMonitor] = None

async def initialize_fusion_monitor() -> XORBFusionMonitor:
    """Initialize the global fusion monitor."""
    global fusion_monitor
    fusion_monitor = XORBFusionMonitor()
    await fusion_monitor.initialize()
    return fusion_monitor

async def get_fusion_monitor() -> Optional[XORBFusionMonitor]:
    """Get the global fusion monitor."""
    return fusion_monitor
