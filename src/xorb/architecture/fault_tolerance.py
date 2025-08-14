#!/usr/bin/env python3
"""
XORB Fault-Tolerant Distributed Systems
Advanced resilience patterns with circuit breakers, bulkheads, and EPYC optimization
"""

import asyncio
import logging
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, TypeVar, Generic
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import uuid
import weakref

# Skip aioredis for now due to compatibility issue
# import aioredis
aioredis = None

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available for fault tolerance metrics")

    # Functional mock implementations for fault tolerance
    class MockMetric:
        def __init__(self, name, description, labelnames=None, registry=None):
            self.name = name
            self._value = 0
            self.labelnames = labelnames or []

        def inc(self, amount=1):
            self._value += amount
            logger.debug(f"Fault tolerance metric {self.name}: {self._value}")

        def observe(self, value):
            logger.debug(f"Fault tolerance metric {self.name}: observed {value}")

        def set(self, value):
            self._value = value
            logger.debug(f"Fault tolerance metric {self.name}: set to {value}")

        def labels(self, **kwargs):
            return self

        def time(self):
            return MockTimer()

    class MockTimer:
        def __init__(self):
            self.start_time = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, *args):
            if self.start_time:
                duration = time.time() - self.start_time
                logger.debug(f"Operation completed in {duration:.3f}s")

    Counter = MockMetric
    Histogram = MockMetric
    Gauge = MockMetric
    Summary = MockMetric

logger = logging.getLogger(__name__)

T = TypeVar('T')

class FaultTolerancePattern(Enum):
    CIRCUIT_BREAKER = "circuit_breaker"
    BULKHEAD = "bulkhead"
    TIMEOUT = "timeout"
    RETRY = "retry"
    RATE_LIMITER = "rate_limiter"
    FALLBACK = "fallback"
    CACHE_ASIDE = "cache_aside"
    LEADER_ELECTION = "leader_election"

class IsolationLevel(Enum):
    THREAD = "thread"           # Thread-level isolation
    PROCESS = "process"         # Process-level isolation
    NUMA_NODE = "numa_node"     # NUMA node isolation
    CCX = "ccx"                # Core Complex isolation
    SERVICE = "service"         # Service-level isolation

class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    IO = "io"
    NETWORK = "network"
    STORAGE = "storage"

@dataclass
class ResourceQuota:
    """Resource quota definition for bulkhead isolation."""
    resource_type: ResourceType
    limit: Union[int, float]
    current_usage: Union[int, float] = 0
    peak_usage: Union[int, float] = 0
    unit: str = "units"

    @property
    def utilization_percentage(self) -> float:
        """Calculate current utilization percentage."""
        if self.limit == 0:
            return 0.0
        return (self.current_usage / self.limit) * 100

@dataclass
class BulkheadConfig:
    """Bulkhead isolation configuration."""
    name: str
    isolation_level: IsolationLevel
    resource_quotas: Dict[ResourceType, ResourceQuota]
    max_concurrent_requests: int = 100
    queue_size: int = 1000
    timeout_seconds: float = 30.0
    epyc_numa_binding: Optional[int] = None
    epyc_ccx_affinity: Optional[int] = None
    thread_pool_size: Optional[int] = None
    process_pool_size: Optional[int] = None

class BulkheadViolationError(Exception):
    """Raised when bulkhead resource limits are exceeded."""
    pass

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass

class EPYCBulkhead:
    """EPYC-optimized bulkhead implementation with resource isolation."""

    def __init__(self, config: BulkheadConfig):
        self.config = config
        self.active_requests = 0
        self.queued_requests = 0
        self.request_queue = asyncio.Queue(maxsize=config.queue_size)
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        # EPYC-specific optimizations
        self.numa_node = config.epyc_numa_binding
        self.ccx_affinity = config.epyc_ccx_affinity

        # Thread/Process pools for isolation
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None

        if config.isolation_level == IsolationLevel.THREAD and config.thread_pool_size:
            self.thread_pool = ThreadPoolExecutor(
                max_workers=config.thread_pool_size,
                thread_name_prefix=f"bulkhead-{config.name}"
            )
        elif config.isolation_level == IsolationLevel.PROCESS and config.process_pool_size:
            self.process_pool = ProcessPoolExecutor(
                max_workers=config.process_pool_size
            )

        # Metrics
        self.bulkhead_requests = Counter(
            'bulkhead_requests_total',
            'Total bulkhead requests',
            ['bulkhead', 'status']
        )
        self.bulkhead_queue_size = Gauge(
            'bulkhead_queue_size',
            'Current bulkhead queue size',
            ['bulkhead']
        )
        self.bulkhead_active_requests = Gauge(
            'bulkhead_active_requests',
            'Current active requests in bulkhead',
            ['bulkhead']
        )
        self.resource_utilization = Gauge(
            'bulkhead_resource_utilization_percent',
            'Resource utilization percentage',
            ['bulkhead', 'resource_type']
        )

        # Initialize resource monitoring
        self._start_resource_monitoring()

    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function within bulkhead constraints."""
        # Check resource quotas
        self._check_resource_quotas()

        # Queue management
        if self.queued_requests >= self.config.queue_size:
            self.bulkhead_requests.labels(bulkhead=self.config.name, status="rejected").inc()
            raise BulkheadViolationError(f"Bulkhead {self.config.name} queue is full")

        self.queued_requests += 1
        self.bulkhead_queue_size.labels(bulkhead=self.config.name).set(self.queued_requests)

        try:
            # Acquire semaphore (rate limiting)
            async with self.semaphore:
                self.queued_requests -= 1
                self.active_requests += 1
                self.bulkhead_active_requests.labels(bulkhead=self.config.name).set(self.active_requests)

                try:
                    # Execute with timeout
                    result = await asyncio.wait_for(
                        self._execute_with_isolation(func, *args, **kwargs),
                        timeout=self.config.timeout_seconds
                    )

                    self.bulkhead_requests.labels(bulkhead=self.config.name, status="success").inc()
                    return result

                except asyncio.TimeoutError:
                    self.bulkhead_requests.labels(bulkhead=self.config.name, status="timeout").inc()
                    raise
                except Exception as e:
                    self.bulkhead_requests.labels(bulkhead=self.config.name, status="error").inc()
                    raise
                finally:
                    self.active_requests -= 1
                    self.bulkhead_active_requests.labels(bulkhead=self.config.name).set(self.active_requests)
        finally:
            if self.queued_requests > 0:
                self.queued_requests -= 1
            self.bulkhead_queue_size.labels(bulkhead=self.config.name).set(self.queued_requests)

    async def _execute_with_isolation(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with appropriate isolation level."""
        if self.config.isolation_level == IsolationLevel.THREAD and self.thread_pool:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
        elif self.config.isolation_level == IsolationLevel.PROCESS and self.process_pool:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.process_pool, func, *args, **kwargs)
        elif self.config.isolation_level == IsolationLevel.NUMA_NODE:
            return await self._execute_with_numa_binding(func, *args, **kwargs)
        elif self.config.isolation_level == IsolationLevel.CCX:
            return await self._execute_with_ccx_affinity(func, *args, **kwargs)
        else:
            # Default async execution
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

    async def _execute_with_numa_binding(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute with NUMA node binding (Linux-specific)."""
        if self.numa_node is None:
            return await self._execute_with_isolation(func, *args, **kwargs)

        try:
            import os
            import psutil

            # Set NUMA memory policy for current thread
            if hasattr(os, 'sched_setaffinity'):
                # Get CPU cores for NUMA node
                numa_cores = []
                for cpu_id, numa_node in enumerate(psutil.cpu_count(logical=True)):
                    # This is simplified - real implementation would use libnuma
                    if cpu_id % 2 == self.numa_node:  # Simplified NUMA detection
                        numa_cores.append(cpu_id)

                if numa_cores:
                    original_affinity = os.sched_getaffinity(0)
                    os.sched_setaffinity(0, numa_cores)
                    try:
                        if asyncio.iscoroutinefunction(func):
                            return await func(*args, **kwargs)
                        else:
                            return func(*args, **kwargs)
                    finally:
                        os.sched_setaffinity(0, original_affinity)

            # Fallback to normal execution
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        except ImportError:
            logger.warning("NUMA binding not available, falling back to normal execution")
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

    async def _execute_with_ccx_affinity(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute with Core Complex (CCX) affinity."""
        # CCX affinity is more complex and would require lower-level CPU management
        # For now, fall back to NUMA binding
        return await self._execute_with_numa_binding(func, *args, **kwargs)

    def _check_resource_quotas(self):
        """Check if resource quotas allow new requests."""
        for resource_type, quota in self.config.resource_quotas.items():
            if quota.current_usage >= quota.limit:
                raise BulkheadViolationError(
                    f"Resource quota exceeded for {resource_type.value}: "
                    f"{quota.current_usage}/{quota.limit} {quota.unit}"
                )

    def _start_resource_monitoring(self):
        """Start background resource monitoring."""
        async def monitor_resources():
            while True:
                try:
                    for resource_type, quota in self.config.resource_quotas.items():
                        utilization = quota.utilization_percentage
                        self.resource_utilization.labels(
                            bulkhead=self.config.name,
                            resource_type=resource_type.value
                        ).set(utilization)

                    await asyncio.sleep(5)  # Monitor every 5 seconds
                except Exception as e:
                    logger.error(f"Resource monitoring error: {e}")
                    await asyncio.sleep(10)

        # Start monitoring task
        asyncio.create_task(monitor_resources())

    def close(self):
        """Clean shutdown of bulkhead resources."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)

class AdvancedCircuitBreaker:
    """Advanced circuit breaker with multiple failure modes and EPYC optimization."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

        # Circuit breaker state
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None

        # Configuration
        self.failure_threshold = config.get('failure_threshold', 5)
        self.success_threshold = config.get('success_threshold', 3)
        self.timeout_duration = config.get('timeout_duration', 60)  # seconds
        self.half_open_timeout = config.get('half_open_timeout', 30)  # seconds

        # Advanced features
        self.failure_rate_threshold = config.get('failure_rate_threshold', 0.5)  # 50%
        self.minimum_throughput = config.get('minimum_throughput', 10)  # requests
        self.sliding_window_size = config.get('sliding_window_size', 100)  # requests

        # Sliding window for failure rate calculation
        self.request_history: List[bool] = []  # True = success, False = failure
        self.request_times: List[datetime] = []

        # Metrics
        self.circuit_breaker_state = Gauge(
            'circuit_breaker_state',
            'Circuit breaker state (0=CLOSED, 1=OPEN, 2=HALF_OPEN)',
            ['circuit_breaker']
        )
        self.circuit_breaker_failures = Counter(
            'circuit_breaker_failures_total',
            'Total circuit breaker failures',
            ['circuit_breaker']
        )
        self.circuit_breaker_successes = Counter(
            'circuit_breaker_successes_total',
            'Total circuit breaker successes',
            ['circuit_breaker']
        )
        self.circuit_breaker_calls = Counter(
            'circuit_breaker_calls_total',
            'Total circuit breaker calls',
            ['circuit_breaker', 'state']
        )

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        current_state = self._get_current_state()

        self.circuit_breaker_calls.labels(
            circuit_breaker=self.name,
            state=current_state
        ).inc()

        if current_state == "OPEN":
            raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")

        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            self._record_success(time.time() - start_time)
            return result

        except Exception as e:
            self._record_failure(time.time() - start_time)
            raise e

    def _get_current_state(self) -> str:
        """Get current circuit breaker state with state transitions."""
        now = datetime.utcnow()

        if self.state == "OPEN":
            if self._should_attempt_reset(now):
                self.state = "HALF_OPEN"
                logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                self.circuit_breaker_state.labels(circuit_breaker=self.name).set(2)

        elif self.state == "HALF_OPEN":
            if self._should_close_from_half_open():
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info(f"Circuit breaker {self.name} transitioning to CLOSED")
                self.circuit_breaker_state.labels(circuit_breaker=self.name).set(0)
            elif self._should_open_from_half_open(now):
                self.state = "OPEN"
                logger.warning(f"Circuit breaker {self.name} transitioning to OPEN from HALF_OPEN")
                self.circuit_breaker_state.labels(circuit_breaker=self.name).set(1)

        return self.state

    def _record_success(self, response_time: float):
        """Record successful call."""
        now = datetime.utcnow()
        self.last_success_time = now
        self.success_count += 1

        # Update sliding window
        self.request_history.append(True)
        self.request_times.append(now)
        self._trim_sliding_window()

        # Reset failure count on success
        if self.state == "CLOSED":
            self.failure_count = max(0, self.failure_count - 1)

        self.circuit_breaker_successes.labels(circuit_breaker=self.name).inc()

    def _record_failure(self, response_time: float):
        """Record failed call."""
        now = datetime.utcnow()
        self.last_failure_time = now
        self.failure_count += 1

        # Update sliding window
        self.request_history.append(False)
        self.request_times.append(now)
        self._trim_sliding_window()

        self.circuit_breaker_failures.labels(circuit_breaker=self.name).inc()

        # Check if should transition to OPEN
        if self.state == "CLOSED" and self._should_open():
            self.state = "OPEN"
            logger.warning(f"Circuit breaker {self.name} transitioning to OPEN")
            self.circuit_breaker_state.labels(circuit_breaker=self.name).set(1)
        elif self.state == "HALF_OPEN":
            self.state = "OPEN"
            logger.warning(f"Circuit breaker {self.name} transitioning to OPEN from HALF_OPEN")
            self.circuit_breaker_state.labels(circuit_breaker=self.name).set(1)

    def _should_open(self) -> bool:
        """Check if circuit breaker should open."""
        # Simple failure count threshold
        if self.failure_count >= self.failure_threshold:
            return True

        # Failure rate threshold with minimum throughput
        if len(self.request_history) >= self.minimum_throughput:
            failure_rate = 1 - (sum(self.request_history) / len(self.request_history))
            if failure_rate >= self.failure_rate_threshold:
                return True

        return False

    def _should_attempt_reset(self, now: datetime) -> bool:
        """Check if should attempt reset from OPEN state."""
        if not self.last_failure_time:
            return True

        elapsed = (now - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout_duration

    def _should_close_from_half_open(self) -> bool:
        """Check if should close from HALF_OPEN state."""
        return self.success_count >= self.success_threshold

    def _should_open_from_half_open(self, now: datetime) -> bool:
        """Check if should open from HALF_OPEN state."""
        if not self.last_failure_time:
            return False

        elapsed = (now - self.last_failure_time).total_seconds()
        return elapsed < self.half_open_timeout and self.failure_count > 0

    def _trim_sliding_window(self):
        """Trim sliding window to configured size."""
        if len(self.request_history) > self.sliding_window_size:
            excess = len(self.request_history) - self.sliding_window_size
            self.request_history = self.request_history[excess:]
            self.request_times = self.request_times[excess:]

class FaultTolerantExecutor:
    """High-level fault-tolerant execution coordinator."""

    def __init__(self, redis_url: str = "redis://redis:6379"):
        self.bulkheads: Dict[str, EPYCBulkhead] = {}
        self.circuit_breakers: Dict[str, AdvancedCircuitBreaker] = {}
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None

        # Global metrics
        self.fault_tolerance_operations = Counter(
            'fault_tolerance_operations_total',
            'Total fault tolerance operations',
            ['operation_type', 'pattern', 'status']
        )
        self.operation_duration = Histogram(
            'fault_tolerance_operation_duration_seconds',
            'Fault tolerance operation duration',
            ['operation_type', 'pattern']
        )

    async def initialize(self):
        """Initialize fault tolerance system."""
        self.redis = aioredis.from_url(self.redis_url)

        # Load configurations from Redis or default
        await self._load_configurations()

        logger.info("Fault-tolerant executor initialized")

    async def close(self):
        """Clean shutdown of fault tolerance system."""
        for bulkhead in self.bulkheads.values():
            bulkhead.close()

        if self.redis:
            await self.redis.close()

    def register_bulkhead(self, config: BulkheadConfig):
        """Register a new bulkhead."""
        bulkhead = EPYCBulkhead(config)
        self.bulkheads[config.name] = bulkhead
        logger.info(f"Registered bulkhead: {config.name}")

    def register_circuit_breaker(self, name: str, config: Dict[str, Any]):
        """Register a new circuit breaker."""
        circuit_breaker = AdvancedCircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker
        logger.info(f"Registered circuit breaker: {name}")

    async def execute_with_bulkhead(
        self,
        bulkhead_name: str,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """Execute function within specified bulkhead."""
        if bulkhead_name not in self.bulkheads:
            raise ValueError(f"Bulkhead {bulkhead_name} not found")

        start_time = time.time()
        try:
            result = await self.bulkheads[bulkhead_name].execute(func, *args, **kwargs)

            self.fault_tolerance_operations.labels(
                operation_type="execute",
                pattern="bulkhead",
                status="success"
            ).inc()

            return result

        except Exception as e:
            self.fault_tolerance_operations.labels(
                operation_type="execute",
                pattern="bulkhead",
                status="error"
            ).inc()
            raise e
        finally:
            duration = time.time() - start_time
            self.operation_duration.labels(
                operation_type="execute",
                pattern="bulkhead"
            ).observe(duration)

    async def execute_with_circuit_breaker(
        self,
        circuit_breaker_name: str,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """Execute function with circuit breaker protection."""
        if circuit_breaker_name not in self.circuit_breakers:
            raise ValueError(f"Circuit breaker {circuit_breaker_name} not found")

        start_time = time.time()
        try:
            result = await self.circuit_breakers[circuit_breaker_name].call(func, *args, **kwargs)

            self.fault_tolerance_operations.labels(
                operation_type="execute",
                pattern="circuit_breaker",
                status="success"
            ).inc()

            return result

        except CircuitBreakerOpenError:
            self.fault_tolerance_operations.labels(
                operation_type="execute",
                pattern="circuit_breaker",
                status="open"
            ).inc()
            raise
        except Exception as e:
            self.fault_tolerance_operations.labels(
                operation_type="execute",
                pattern="circuit_breaker",
                status="error"
            ).inc()
            raise e
        finally:
            duration = time.time() - start_time
            self.operation_duration.labels(
                operation_type="execute",
                pattern="circuit_breaker"
            ).observe(duration)

    async def execute_with_full_protection(
        self,
        bulkhead_name: str,
        circuit_breaker_name: str,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """Execute function with both bulkhead and circuit breaker protection."""
        async def protected_execution():
            return await self.execute_with_circuit_breaker(
                circuit_breaker_name, func, *args, **kwargs
            )

        return await self.execute_with_bulkhead(
            bulkhead_name, protected_execution
        )

    async def _load_configurations(self):
        """Load fault tolerance configurations."""
        try:
            # Load bulkhead configurations
            bulkhead_configs_data = await self.redis.get("fault_tolerance:bulkheads")
            if bulkhead_configs_data:
                bulkhead_configs = json.loads(bulkhead_configs_data)
                for config_data in bulkhead_configs:
                    config = self._parse_bulkhead_config(config_data)
                    self.register_bulkhead(config)

            # Load circuit breaker configurations
            cb_configs_data = await self.redis.get("fault_tolerance:circuit_breakers")
            if cb_configs_data:
                cb_configs = json.loads(cb_configs_data)
                for name, config in cb_configs.items():
                    self.register_circuit_breaker(name, config)

        except Exception as e:
            logger.warning(f"Failed to load configurations from Redis: {e}")
            # Load default configurations
            self._load_default_configurations()

    def _parse_bulkhead_config(self, config_data: Dict[str, Any]) -> BulkheadConfig:
        """Parse bulkhead configuration from JSON data."""
        resource_quotas = {}
        for resource_name, quota_data in config_data.get('resource_quotas', {}).items():
            resource_type = ResourceType(resource_name)
            quota = ResourceQuota(
                resource_type=resource_type,
                limit=quota_data['limit'],
                unit=quota_data.get('unit', 'units')
            )
            resource_quotas[resource_type] = quota

        return BulkheadConfig(
            name=config_data['name'],
            isolation_level=IsolationLevel(config_data.get('isolation_level', 'thread')),
            resource_quotas=resource_quotas,
            max_concurrent_requests=config_data.get('max_concurrent_requests', 100),
            queue_size=config_data.get('queue_size', 1000),
            timeout_seconds=config_data.get('timeout_seconds', 30.0),
            epyc_numa_binding=config_data.get('epyc_numa_binding'),
            epyc_ccx_affinity=config_data.get('epyc_ccx_affinity'),
            thread_pool_size=config_data.get('thread_pool_size'),
            process_pool_size=config_data.get('process_pool_size')
        )

    def _load_default_configurations(self):
        """Load default fault tolerance configurations."""
        # Default bulkheads for XORB services
        default_bulkheads = [
            BulkheadConfig(
                name="ai_inference",
                isolation_level=IsolationLevel.THREAD,
                resource_quotas={
                    ResourceType.CPU: ResourceQuota(ResourceType.CPU, 4.0, unit="cores"),
                    ResourceType.MEMORY: ResourceQuota(ResourceType.MEMORY, 8192, unit="MB")
                },
                max_concurrent_requests=10,
                queue_size=50,
                timeout_seconds=120.0,
                epyc_numa_binding=0,
                epyc_ccx_affinity=0,
                thread_pool_size=4
            ),
            BulkheadConfig(
                name="vulnerability_scanning",
                isolation_level=IsolationLevel.PROCESS,
                resource_quotas={
                    ResourceType.CPU: ResourceQuota(ResourceType.CPU, 8.0, unit="cores"),
                    ResourceType.MEMORY: ResourceQuota(ResourceType.MEMORY, 4096, unit="MB"),
                    ResourceType.NETWORK: ResourceQuota(ResourceType.NETWORK, 1000, unit="connections")
                },
                max_concurrent_requests=50,
                queue_size=200,
                timeout_seconds=300.0,
                epyc_numa_binding=1,
                process_pool_size=4
            ),
            BulkheadConfig(
                name="general_processing",
                isolation_level=IsolationLevel.THREAD,
                resource_quotas={
                    ResourceType.CPU: ResourceQuota(ResourceType.CPU, 2.0, unit="cores"),
                    ResourceType.MEMORY: ResourceQuota(ResourceType.MEMORY, 2048, unit="MB")
                },
                max_concurrent_requests=100,
                queue_size=500,
                timeout_seconds=30.0,
                thread_pool_size=8
            )
        ]

        for config in default_bulkheads:
            self.register_bulkhead(config)

        # Default circuit breakers
        default_circuit_breakers = {
            "ai_gateway": {
                "failure_threshold": 5,
                "success_threshold": 3,
                "timeout_duration": 60,
                "failure_rate_threshold": 0.5,
                "minimum_throughput": 10
            },
            "vulnerability_scanner": {
                "failure_threshold": 3,
                "success_threshold": 2,
                "timeout_duration": 30,
                "failure_rate_threshold": 0.3,
                "minimum_throughput": 5
            },
            "database": {
                "failure_threshold": 5,
                "success_threshold": 3,
                "timeout_duration": 10,
                "failure_rate_threshold": 0.2,
                "minimum_throughput": 20
            }
        }

        for name, config in default_circuit_breakers.items():
            self.register_circuit_breaker(name, config)

# Global fault-tolerant executor
fault_tolerant_executor: Optional[FaultTolerantExecutor] = None

async def initialize_fault_tolerance(redis_url: str = "redis://redis:6379") -> FaultTolerantExecutor:
    """Initialize global fault tolerance system."""
    global fault_tolerant_executor
    fault_tolerant_executor = FaultTolerantExecutor(redis_url)
    await fault_tolerant_executor.initialize()
    return fault_tolerant_executor

async def get_fault_tolerance() -> Optional[FaultTolerantExecutor]:
    """Get global fault tolerance system."""
    return fault_tolerant_executor
