"""
PTaaS Performance Configuration for AMD EPYC 7002 Series
Optimized for high-core-count server workloads with NUMA awareness
"""

import os
import multiprocessing
from dataclasses import dataclass
from typing import Dict


@dataclass
class PerformanceConfig:
    """Performance configuration optimized for AMD EPYC 7002 series"""

    # AMD EPYC 7002 Optimized Defaults (64+ cores, NUMA topology)
    # Assumes 32-64 core EPYC with SMT enabled
    ptaas_workers: int = int(os.getenv("PTAAS_WORKERS", "28"))  # 80% of cores for PTaaS
    cpu_pool_size: int = int(os.getenv("PTAAS_CPU_POOL", "16"))  # CPU-bound tasks
    io_concurrency: int = int(
        os.getenv("PTAAS_IO_CONCURRENCY", "128")
    )  # High I/O capacity

    # NUMA-aware settings for EPYC
    numa_aware: bool = os.getenv("PTAAS_NUMA_AWARE", "true").lower() == "true"
    memory_pool_mb: int = int(
        os.getenv("PTAAS_MEMORY_POOL_MB", "8192")
    )  # 8GB memory pool

    # NATS JetStream tuning for high throughput
    nats_max_ack_pending: int = int(os.getenv("NATS_MAX_ACK_PENDING", "10000"))
    nats_ack_wait_ms: int = int(os.getenv("NATS_ACK_WAIT_MS", "30000"))  # 30s
    nats_max_deliver: int = int(os.getenv("NATS_MAX_DELIVER", "5"))
    nats_max_inflight: int = int(
        os.getenv("NATS_MAX_INFLIGHT", "256")
    )  # High throughput

    # G8 WFQ (Weighted Fair Queueing) settings
    tenant_max_concurrent: int = int(os.getenv("G8_TENANT_MAX_CONCURRENT", "8"))
    fairness_window_sec: int = int(os.getenv("G8_FAIRNESS_WINDOW_SEC", "60"))
    priority_weight_high: float = float(os.getenv("G8_WEIGHT_HIGH", "3.0"))
    priority_weight_medium: float = float(os.getenv("G8_WEIGHT_MEDIUM", "2.0"))
    priority_weight_low: float = float(os.getenv("G8_WEIGHT_LOW", "1.0"))

    # Performance targets for EPYC
    target_p95_latency_ms: int = int(os.getenv("PTAAS_TARGET_P95_MS", "2000"))  # 2s P95
    target_error_rate: float = float(
        os.getenv("PTAAS_TARGET_ERROR_RATE", "0.005")
    )  # 0.5%
    target_fairness_index: float = float(os.getenv("PTAAS_TARGET_FAIRNESS", "0.7"))

    # Database configuration toggle
    database_backend: str = os.getenv("PTAAS_DB", "sqlite")  # sqlite|postgres
    postgres_pool_size: int = int(os.getenv("POSTGRES_POOL_SIZE", "20"))
    postgres_max_overflow: int = int(os.getenv("POSTGRES_MAX_OVERFLOW", "30"))

    # Module execution timeouts (by complexity)
    timeout_fast_sec: int = int(os.getenv("PTAAS_TIMEOUT_FAST", "30"))  # 70% of modules
    timeout_medium_sec: int = int(
        os.getenv("PTAAS_TIMEOUT_MEDIUM", "120")
    )  # 25% of modules
    timeout_slow_sec: int = int(os.getenv("PTAAS_TIMEOUT_SLOW", "300"))  # 5% of modules

    def __post_init__(self):
        """Validate and auto-adjust configuration for EPYC architecture"""
        # Auto-detect core count if not explicitly set
        cpu_count = multiprocessing.cpu_count()

        # EPYC-specific adjustments
        if "PTAAS_WORKERS" not in os.environ:
            # For EPYC: use 70-80% of cores, accounting for NUMA topology
            self.ptaas_workers = max(min(int(cpu_count * 0.75), 32), 4)

        if "PTAAS_CPU_POOL" not in os.environ:
            # CPU pool should be ~50% of total workers for mixed workloads
            self.cpu_pool_size = max(self.ptaas_workers // 2, 2)

        if "PTAAS_IO_CONCURRENCY" not in os.environ:
            # High I/O concurrency for network scanning workloads
            self.io_concurrency = min(cpu_count * 4, 256)

        # Validate constraints
        assert self.ptaas_workers > 0, "PTAAS_WORKERS must be positive"
        assert self.cpu_pool_size > 0, "CPU pool size must be positive"
        assert self.io_concurrency > 0, "I/O concurrency must be positive"
        assert 0 < self.target_error_rate < 1, "Error rate must be between 0 and 1"
        assert (
            0 < self.target_fairness_index <= 1
        ), "Fairness index must be between 0 and 1"

    def get_cpu_profiles(self) -> Dict[str, Dict[str, int]]:
        """Get CPU configuration profiles for different server sizes"""
        return {
            "epyc_7002_8core": {
                "ptaas_workers": 6,
                "cpu_pool_size": 3,
                "io_concurrency": 32,
                "memory_pool_mb": 2048,
            },
            "epyc_7002_16core": {
                "ptaas_workers": 12,
                "cpu_pool_size": 6,
                "io_concurrency": 64,
                "memory_pool_mb": 4096,
            },
            "epyc_7002_32core": {
                "ptaas_workers": 24,
                "cpu_pool_size": 12,
                "io_concurrency": 128,
                "memory_pool_mb": 8192,
            },
            "epyc_7002_64core": {
                "ptaas_workers": 48,
                "cpu_pool_size": 24,
                "io_concurrency": 256,
                "memory_pool_mb": 16384,
            },
        }

    def apply_profile(self, profile_name: str) -> None:
        """Apply a predefined CPU profile"""
        profiles = self.get_cpu_profiles()
        if profile_name in profiles:
            profile = profiles[profile_name]
            self.ptaas_workers = profile["ptaas_workers"]
            self.cpu_pool_size = profile["cpu_pool_size"]
            self.io_concurrency = profile["io_concurrency"]
            self.memory_pool_mb = profile["memory_pool_mb"]

    def to_env_dict(self) -> Dict[str, str]:
        """Export configuration as environment variables"""
        return {
            "PTAAS_WORKERS": str(self.ptaas_workers),
            "PTAAS_CPU_POOL": str(self.cpu_pool_size),
            "PTAAS_IO_CONCURRENCY": str(self.io_concurrency),
            "PTAAS_NUMA_AWARE": str(self.numa_aware).lower(),
            "PTAAS_MEMORY_POOL_MB": str(self.memory_pool_mb),
            "NATS_MAX_ACK_PENDING": str(self.nats_max_ack_pending),
            "NATS_ACK_WAIT_MS": str(self.nats_ack_wait_ms),
            "NATS_MAX_DELIVER": str(self.nats_max_deliver),
            "NATS_MAX_INFLIGHT": str(self.nats_max_inflight),
            "G8_TENANT_MAX_CONCURRENT": str(self.tenant_max_concurrent),
            "PTAAS_DB": self.database_backend,
            "POSTGRES_POOL_SIZE": str(self.postgres_pool_size),
        }


# Global performance configuration instance
perf_config = PerformanceConfig()

# AMD EPYC 7002 specific optimizations
EPYC_OPTIMIZATIONS = {
    # NUMA topology awareness
    "numa_policy": "interleave",  # Distribute memory across NUMA nodes
    "cpu_affinity": "auto",  # Let scheduler handle core assignment
    # Memory optimizations
    "transparent_hugepages": "madvise",  # Use huge pages for large allocations
    "vm_swappiness": 10,  # Minimize swap usage
    # Network optimizations
    "tcp_congestion_control": "bbr",  # Better for high-bandwidth workloads
    "net_core_rmem_max": 16777216,  # 16MB receive buffer
    "net_core_wmem_max": 16777216,  # 16MB send buffer
    # Scheduler optimizations for EPYC
    "sched_migration_cost_ns": 500000,  # Reduce migration cost
    "sched_nr_migrate": 32,  # Batch migrations
}


def get_optimal_worker_distribution(total_workers: int) -> Dict[str, int]:
    """Calculate optimal worker distribution for mixed PTaaS workloads"""
    return {
        "fast_scanners": int(total_workers * 0.4),  # 40% for quick scans (nmap, ping)
        "medium_scanners": int(
            total_workers * 0.35
        ),  # 35% for medium scans (nuclei, nikto)
        "slow_scanners": int(
            total_workers * 0.15
        ),  # 15% for slow scans (comprehensive)
        "orchestration": int(total_workers * 0.1),  # 10% for workflow management
    }


def calculate_fairness_index(tenant_throughputs: list) -> float:
    """Calculate Jain's fairness index for tenant resource allocation"""
    if not tenant_throughputs:
        return 0.0

    n = len(tenant_throughputs)
    sum_xi = sum(tenant_throughputs)
    sum_xi_squared = sum(x * x for x in tenant_throughputs)

    if sum_xi_squared == 0:
        return 1.0  # Perfect fairness when all are zero

    return (sum_xi * sum_xi) / (n * sum_xi_squared)
