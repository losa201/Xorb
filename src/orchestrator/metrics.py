"""
Prometheus Metrics for PTaaS Performance Monitoring on AMD EPYC 7002
Custom metrics for latency, throughput, fairness, and resource utilization
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import threading

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Summary,
        Info,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Mock classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

    class Histogram:
        def __init__(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

    class Gauge:
        def __init__(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def dec(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

    class Summary:
        def __init__(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

    class Info:
        def __init__(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass


logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance metrics snapshot"""

    timestamp: float
    active_jobs: int
    jobs_per_second: float
    p95_latency_ms: float
    error_rate: float
    fairness_index: float
    memory_usage_mb: float
    cpu_utilization: float


class PTaaSMetrics:
    """Comprehensive PTaaS performance metrics for EPYC monitoring"""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._lock = threading.Lock()

        # Historical data for trend analysis
        self.performance_history: deque = deque(maxlen=1000)
        self.tenant_stats = defaultdict(
            lambda: {"jobs_completed": 0, "total_latency_ms": 0.0, "last_job_time": 0.0}
        )

        # Initialize Prometheus metrics
        self._init_prometheus_metrics()

        logger.info(
            f"PTaaS metrics initialized (Prometheus available: {PROMETHEUS_AVAILABLE})"
        )

    def _init_prometheus_metrics(self) -> None:
        """Initialize all Prometheus metrics"""

        # Job execution metrics
        self.ptaas_job_total = Counter(
            "ptaas_job_total",
            "Total number of PTaaS jobs processed",
            ["tenant_id", "job_type", "scan_profile", "status"],
            registry=self.registry,
        )

        self.ptaas_job_latency_ms = Histogram(
            "ptaas_job_latency_ms",
            "End-to-end job latency in milliseconds",
            ["tenant_id", "job_type", "scan_profile"],
            buckets=[10, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 30000],
            registry=self.registry,
        )

        self.ptaas_queue_time_ms = Histogram(
            "ptaas_queue_time_ms",
            "Time jobs spend in queue before execution",
            ["tenant_id", "priority"],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000],
            registry=self.registry,
        )

        self.ptaas_module_cpu_ms = Histogram(
            "ptaas_module_cpu_ms",
            "CPU time spent in PTaaS modules",
            ["module_name", "scan_profile"],
            buckets=[10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
            registry=self.registry,
        )

        # Fairness and resource allocation
        self.ptaas_fairness_index = Gauge(
            "ptaas_fairness_index",
            "Jain fairness index across tenants (0-1, higher is better)",
            registry=self.registry,
        )

        self.ptaas_tenant_throughput = Gauge(
            "ptaas_tenant_throughput",
            "Jobs per second per tenant",
            ["tenant_id"],
            registry=self.registry,
        )

        self.ptaas_tenant_queue_depth = Gauge(
            "ptaas_tenant_queue_depth",
            "Number of queued jobs per tenant",
            ["tenant_id"],
            registry=self.registry,
        )

        # System resource utilization
        self.ptaas_active_jobs = Gauge(
            "ptaas_active_jobs",
            "Number of currently executing jobs",
            registry=self.registry,
        )

        self.ptaas_worker_utilization = Gauge(
            "ptaas_worker_utilization",
            "Worker pool utilization percentage",
            [
                "worker_type"
            ],  # fast_scanners, medium_scanners, slow_scanners, orchestration
            registry=self.registry,
        )

        self.ptaas_memory_usage_mb = Gauge(
            "ptaas_memory_usage_mb",
            "Memory usage in megabytes",
            ["component"],  # orchestrator, task_runner, nats_client
            registry=self.registry,
        )

        # NATS JetStream metrics
        self.nats_consumer_lag = Gauge(
            "nats_consumer_lag",
            "Number of unprocessed messages in NATS consumers",
            ["consumer_name", "subject"],
            registry=self.registry,
        )

        self.nats_redeliveries = Counter(
            "nats_redeliveries_total",
            "Total number of message redeliveries",
            ["consumer_name", "reason"],
            registry=self.registry,
        )

        self.nats_ack_duration_ms = Histogram(
            "nats_ack_duration_ms",
            "Time to acknowledge NATS messages",
            ["consumer_name"],
            buckets=[1, 2, 5, 10, 25, 50, 100, 250, 500],
            registry=self.registry,
        )

        # Error tracking
        self.ptaas_errors_total = Counter(
            "ptaas_errors_total",
            "Total number of PTaaS errors",
            ["tenant_id", "error_type", "component"],
            registry=self.registry,
        )

        # Performance targets tracking
        self.ptaas_sla_violations = Counter(
            "ptaas_sla_violations_total",
            "SLA violations (P95 > 2s, error rate > 0.5%, etc.)",
            ["sla_type", "tenant_id"],
            registry=self.registry,
        )

        # EPYC-specific system metrics
        self.epyc_numa_memory_usage = Gauge(
            "epyc_numa_memory_usage_mb",
            "NUMA node memory usage on EPYC",
            ["numa_node"],
            registry=self.registry,
        )

        self.epyc_cpu_utilization = Gauge(
            "epyc_cpu_utilization_percent",
            "CPU utilization per core group on EPYC",
            ["core_group"],  # ccx_0, ccx_1, etc.
            registry=self.registry,
        )

        # System information
        self.ptaas_info = Info(
            "ptaas_info", "PTaaS system information", registry=self.registry
        )

        # Set static system info
        self.ptaas_info.info(
            {
                "version": "2025.08-rc2",
                "cpu_architecture": "amd_epyc_7002",
                "optimization_target": "high_throughput",
                "max_workers": "28",
                "numa_aware": "true",
            }
        )

    def record_job_completion(
        self,
        tenant_id: str,
        job_type: str,
        scan_profile: str,
        latency_ms: float,
        queue_time_ms: float,
        success: bool,
        error_type: Optional[str] = None,
    ) -> None:
        """Record job completion metrics"""

        status = "success" if success else "failure"

        # Job counters
        self.ptaas_job_total.labels(
            tenant_id=tenant_id,
            job_type=job_type,
            scan_profile=scan_profile,
            status=status,
        ).inc()

        # Latency tracking
        if success:
            self.ptaas_job_latency_ms.labels(
                tenant_id=tenant_id, job_type=job_type, scan_profile=scan_profile
            ).observe(latency_ms)

            self.ptaas_queue_time_ms.labels(
                tenant_id=tenant_id,
                priority="medium",  # Default, should be passed in
            ).observe(queue_time_ms)

        # Error tracking
        if not success and error_type:
            self.ptaas_errors_total.labels(
                tenant_id=tenant_id, error_type=error_type, component="orchestrator"
            ).inc()

        # SLA violation tracking
        if latency_ms > 2000:  # P95 target
            self.ptaas_sla_violations.labels(
                sla_type="latency_p95", tenant_id=tenant_id
            ).inc()

        # Update tenant statistics
        with self._lock:
            stats = self.tenant_stats[tenant_id]
            stats["jobs_completed"] += 1
            stats["total_latency_ms"] += latency_ms
            stats["last_job_time"] = time.time()

    def record_module_execution(
        self, module_name: str, scan_profile: str, cpu_time_ms: float
    ) -> None:
        """Record module-level execution metrics"""

        self.ptaas_module_cpu_ms.labels(
            module_name=module_name, scan_profile=scan_profile
        ).observe(cpu_time_ms)

    def update_system_metrics(
        self,
        active_jobs: int,
        worker_utilization: Dict[str, float],
        memory_usage: Dict[str, float],
        fairness_index: float,
    ) -> None:
        """Update system-level metrics"""

        # Active jobs
        self.ptaas_active_jobs.set(active_jobs)

        # Worker utilization
        for worker_type, utilization in worker_utilization.items():
            self.ptaas_worker_utilization.labels(worker_type=worker_type).set(
                utilization * 100
            )  # Convert to percentage

        # Memory usage
        for component, usage_mb in memory_usage.items():
            self.ptaas_memory_usage_mb.labels(component=component).set(usage_mb)

        # Fairness index
        self.ptaas_fairness_index.set(fairness_index)

        # Check SLA violations
        if fairness_index < 0.7:
            self.ptaas_sla_violations.labels(
                sla_type="fairness_index", tenant_id="all"
            ).inc()

    def update_tenant_metrics(self) -> None:
        """Update per-tenant throughput metrics"""

        current_time = time.time()

        with self._lock:
            for tenant_id, stats in self.tenant_stats.items():
                # Calculate throughput (jobs per second over last minute)
                time_window = 60.0  # 1 minute
                if current_time - stats["last_job_time"] < time_window:
                    # Simplified throughput calculation
                    throughput = stats["jobs_completed"] / time_window
                    self.ptaas_tenant_throughput.labels(tenant_id=tenant_id).set(
                        throughput
                    )

    def record_nats_metrics(
        self,
        consumer_name: str,
        subject: str,
        lag: int,
        ack_duration_ms: float,
        redelivery_reason: Optional[str] = None,
    ) -> None:
        """Record NATS JetStream metrics"""

        # Consumer lag
        self.nats_consumer_lag.labels(consumer_name=consumer_name, subject=subject).set(
            lag
        )

        # ACK duration
        self.nats_ack_duration_ms.labels(consumer_name=consumer_name).observe(
            ack_duration_ms
        )

        # Redeliveries
        if redelivery_reason:
            self.nats_redeliveries.labels(
                consumer_name=consumer_name, reason=redelivery_reason
            ).inc()

    def record_epyc_system_metrics(
        self, numa_memory_usage: Dict[int, float], cpu_utilization: Dict[str, float]
    ) -> None:
        """Record EPYC-specific system metrics"""

        # NUMA memory usage
        for numa_node, usage_mb in numa_memory_usage.items():
            self.epyc_numa_memory_usage.labels(numa_node=str(numa_node)).set(usage_mb)

        # CPU utilization by core group
        for core_group, utilization in cpu_utilization.items():
            self.epyc_cpu_utilization.labels(core_group=core_group).set(
                utilization * 100
            )

    def get_performance_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot"""

        current_time = time.time()

        # Calculate current metrics (simplified)
        active_jobs = 0  # Would get from actual system
        jobs_per_second = 0.0
        p95_latency_ms = 0.0
        error_rate = 0.0
        fairness_index = 0.8  # Would calculate from actual data
        memory_usage_mb = 0.0
        cpu_utilization = 0.0

        snapshot = PerformanceSnapshot(
            timestamp=current_time,
            active_jobs=active_jobs,
            jobs_per_second=jobs_per_second,
            p95_latency_ms=p95_latency_ms,
            error_rate=error_rate,
            fairness_index=fairness_index,
            memory_usage_mb=memory_usage_mb,
            cpu_utilization=cpu_utilization,
        )

        # Store in history
        self.performance_history.append(snapshot)

        return snapshot

    def get_tenant_fairness_metrics(self) -> Dict[str, Any]:
        """Calculate detailed fairness metrics across tenants"""

        tenant_throughputs = []
        tenant_latencies = []

        with self._lock:
            for tenant_id, stats in self.tenant_stats.items():
                if stats["jobs_completed"] > 0:
                    avg_latency = stats["total_latency_ms"] / stats["jobs_completed"]
                    tenant_latencies.append(avg_latency)
                    # Simplified throughput calculation
                    tenant_throughputs.append(stats["jobs_completed"])

        # Calculate Jain's fairness index
        if tenant_throughputs:
            n = len(tenant_throughputs)
            sum_xi = sum(tenant_throughputs)
            sum_xi_squared = sum(x * x for x in tenant_throughputs)

            if sum_xi_squared > 0:
                fairness_index = (sum_xi * sum_xi) / (n * sum_xi_squared)
            else:
                fairness_index = 1.0
        else:
            fairness_index = 1.0

        return {
            "fairness_index": fairness_index,
            "tenant_count": len(self.tenant_stats),
            "tenant_throughputs": tenant_throughputs,
            "tenant_latencies": tenant_latencies,
            "coefficient_of_variation": self._calculate_cv(tenant_throughputs),
        }

    def _calculate_cv(self, values: List[float]) -> float:
        """Calculate coefficient of variation"""
        if not values or len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0

        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance**0.5

        return std_dev / mean

    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry).decode("utf-8")
        else:
            return "# Prometheus client not available\n"

    def get_metrics_content_type(self) -> str:
        """Get content type for metrics endpoint"""
        return CONTENT_TYPE_LATEST if PROMETHEUS_AVAILABLE else "text/plain"


# Global metrics instance
_metrics_instance: Optional[PTaaSMetrics] = None


def get_metrics() -> PTaaSMetrics:
    """Get global metrics instance"""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = PTaaSMetrics()
    return _metrics_instance


def reset_metrics() -> None:
    """Reset global metrics instance (useful for testing)"""
    global _metrics_instance
    _metrics_instance = None
