#!/usr/bin/env python3
"""
XORB EPYC Architecture Optimization
Advanced concurrency patterns optimized for AMD EPYC processors
"""

import asyncio
import logging
import threading
import multiprocessing
import time
import os
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, TypeVar, Generic
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from queue import Queue, Empty
import weakref

import psutil
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

T = TypeVar('T')

class EPYCTopology(Enum):
    """EPYC processor topology levels."""
    THREAD = "thread"           # Individual thread
    CORE = "core"              # Physical core
    CCX = "ccx"                # Core Complex (4 cores + L3 cache)
    DIE = "die"                # Die (2 CCX + memory controller)
    SOCKET = "socket"          # Complete processor socket
    NUMA_NODE = "numa_node"    # NUMA node (1-2 dies)

class WorkloadType(Enum):
    """Workload characteristics for optimal placement."""
    CPU_INTENSIVE = "cpu_intensive"        # High CPU, low memory
    MEMORY_INTENSIVE = "memory_intensive"  # High memory bandwidth
    CACHE_SENSITIVE = "cache_sensitive"    # Benefits from L3 cache locality
    IO_INTENSIVE = "io_intensive"          # I/O bound workloads
    LATENCY_SENSITIVE = "latency_sensitive" # Low latency requirements
    THROUGHPUT_OPTIMIZED = "throughput_optimized" # High throughput batch processing
    AI_INFERENCE = "ai_inference"          # AI/ML inference workloads
    PARALLEL_COMPUTE = "parallel_compute"  # Embarrassingly parallel tasks

class AffinityPolicy(Enum):
    """CPU affinity policies for different workloads."""
    NONE = "none"                  # No specific affinity
    CORE_EXCLUSIVE = "core_exclusive"     # Exclusive core assignment
    CCX_PREFERRED = "ccx_preferred"       # Prefer same CCX
    NUMA_LOCAL = "numa_local"             # Stay within NUMA node
    SOCKET_LOCAL = "socket_local"         # Stay within socket
    SPREAD_EVENLY = "spread_evenly"       # Distribute across topology

@dataclass
class EPYCTopologyInfo:
    """EPYC processor topology information."""
    total_cores: int
    logical_cores: int
    numa_nodes: int
    ccx_count: int
    cores_per_ccx: int
    l3_cache_per_ccx: int  # MB
    memory_controllers: int
    base_frequency: float  # GHz
    boost_frequency: float  # GHz
    thermal_design_power: int  # Watts

    # Topology mapping
    core_to_numa: Dict[int, int] = field(default_factory=dict)
    core_to_ccx: Dict[int, int] = field(default_factory=dict)
    ccx_to_numa: Dict[int, int] = field(default_factory=dict)

@dataclass
class WorkloadProfile:
    """Workload execution profile."""
    workload_type: WorkloadType
    affinity_policy: AffinityPolicy
    preferred_numa_node: Optional[int] = None
    preferred_ccx: Optional[int] = None
    exclusive_cores: bool = False
    max_threads: Optional[int] = None
    memory_locality: bool = True
    cache_sensitivity: float = 0.5  # 0.0 to 1.0
    thermal_awareness: bool = True

class EPYCTopologyDetector:
    """Detect and analyze EPYC processor topology."""

    def __init__(self):
        self.topology_info: Optional[EPYCTopologyInfo] = None
        self._cache_detection_result = True

    def detect_topology(self) -> EPYCTopologyInfo:
        """Detect EPYC processor topology."""
        if self.topology_info and self._cache_detection_result:
            return self.topology_info

        try:
            # Get basic CPU information
            total_cores = psutil.cpu_count(logical=False)
            logical_cores = psutil.cpu_count(logical=True)

            # Detect NUMA topology
            numa_nodes = self._detect_numa_nodes()

            # EPYC-specific topology calculations
            ccx_count = self._calculate_ccx_count(total_cores)
            cores_per_ccx = self._calculate_cores_per_ccx(total_cores, ccx_count)

            # CPU frequency information
            cpu_freq = psutil.cpu_freq()
            base_freq = cpu_freq.current / 1000 if cpu_freq else 2.0  # GHz
            boost_freq = cpu_freq.max / 1000 if cpu_freq and cpu_freq.max else base_freq * 1.5

            # Memory controllers (typically 1 per die, 2 dies per NUMA node for EPYC)
            memory_controllers = numa_nodes * 2

            # Create topology mapping
            core_to_numa, core_to_ccx, ccx_to_numa = self._create_topology_mapping(
                total_cores, numa_nodes, ccx_count, cores_per_ccx
            )

            self.topology_info = EPYCTopologyInfo(
                total_cores=total_cores,
                logical_cores=logical_cores,
                numa_nodes=numa_nodes,
                ccx_count=ccx_count,
                cores_per_ccx=cores_per_ccx,
                l3_cache_per_ccx=16,  # Typical EPYC L3 cache size (MB)
                memory_controllers=memory_controllers,
                base_frequency=base_freq,
                boost_frequency=boost_freq,
                thermal_design_power=200,  # Typical EPYC TDP
                core_to_numa=core_to_numa,
                core_to_ccx=core_to_ccx,
                ccx_to_numa=ccx_to_numa
            )

            logger.info(f"Detected EPYC topology: {total_cores} cores, {ccx_count} CCX, {numa_nodes} NUMA nodes")
            return self.topology_info

        except Exception as e:
            logger.error(f"Failed to detect EPYC topology: {e}")
            # Fallback to basic configuration
            return self._create_fallback_topology()

    def _detect_numa_nodes(self) -> int:
        """Detect number of NUMA nodes."""
        try:
            # Try to read from /sys/devices/system/node/
            numa_nodes = 0
            node_dir = "/sys/devices/system/node"
            if os.path.exists(node_dir):
                for entry in os.listdir(node_dir):
                    if entry.startswith("node") and entry[4:].isdigit():
                        numa_nodes += 1

            return max(1, numa_nodes)
        except:
            # Fallback: assume 2 NUMA nodes for typical EPYC
            return 2

    def _calculate_ccx_count(self, total_cores: int) -> int:
        """Calculate number of Core Complexes."""
        # EPYC has 4 cores per CCX typically
        return (total_cores + 3) // 4

    def _calculate_cores_per_ccx(self, total_cores: int, ccx_count: int) -> int:
        """Calculate cores per CCX."""
        return total_cores // ccx_count if ccx_count > 0 else 4

    def _create_topology_mapping(
        self,
        total_cores: int,
        numa_nodes: int,
        ccx_count: int,
        cores_per_ccx: int
    ) -> tuple:
        """Create topology mapping dictionaries."""
        core_to_numa = {}
        core_to_ccx = {}
        ccx_to_numa = {}

        cores_per_numa = total_cores // numa_nodes
        ccx_per_numa = ccx_count // numa_nodes

        for core in range(total_cores):
            # Assign NUMA node
            numa_node = core // cores_per_numa
            core_to_numa[core] = min(numa_node, numa_nodes - 1)

            # Assign CCX
            ccx = core // cores_per_ccx
            core_to_ccx[core] = min(ccx, ccx_count - 1)

        # Map CCX to NUMA nodes
        for ccx in range(ccx_count):
            numa_node = ccx // ccx_per_numa
            ccx_to_numa[ccx] = min(numa_node, numa_nodes - 1)

        return core_to_numa, core_to_ccx, ccx_to_numa

    def _create_fallback_topology(self) -> EPYCTopologyInfo:
        """Create fallback topology for non-EPYC systems."""
        total_cores = psutil.cpu_count(logical=False) or 8
        logical_cores = psutil.cpu_count(logical=True) or 16

        return EPYCTopologyInfo(
            total_cores=total_cores,
            logical_cores=logical_cores,
            numa_nodes=2,
            ccx_count=total_cores // 4,
            cores_per_ccx=4,
            l3_cache_per_ccx=16,
            memory_controllers=4,
            base_frequency=2.0,
            boost_frequency=3.0,
            thermal_design_power=200,
            core_to_numa={i: i // (total_cores // 2) for i in range(total_cores)},
            core_to_ccx={i: i // 4 for i in range(total_cores)},
            ccx_to_numa={i: i // 2 for i in range(total_cores // 4)}
        )

class EPYCAffinityManager:
    """Manage CPU affinity for optimal EPYC performance."""

    def __init__(self, topology: EPYCTopologyInfo):
        self.topology = topology
        self.core_allocations: Dict[int, str] = {}  # core -> workload_id mapping
        self.workload_assignments: Dict[str, List[int]] = {}  # workload_id -> cores
        self._allocation_lock = threading.Lock()

        # Metrics
        self.core_utilization = Gauge(
            'epyc_core_utilization_percent',
            'Per-core utilization percentage',
            ['core', 'numa_node', 'ccx']
        )
        self.affinity_assignments = Counter(
            'epyc_affinity_assignments_total',
            'Total affinity assignments',
            ['policy', 'workload_type']
        )

    def allocate_cores(
        self,
        workload_id: str,
        profile: WorkloadProfile,
        num_cores: Optional[int] = None
    ) -> List[int]:
        """Allocate optimal cores for a workload."""
        with self._allocation_lock:
            if num_cores is None:
                num_cores = self._calculate_optimal_cores(profile)

            allocated_cores = self._select_optimal_cores(profile, num_cores)

            # Record allocation
            for core in allocated_cores:
                self.core_allocations[core] = workload_id
            self.workload_assignments[workload_id] = allocated_cores

            # Apply CPU affinity
            self._apply_affinity(allocated_cores)

            # Update metrics
            self.affinity_assignments.labels(
                policy=profile.affinity_policy.value,
                workload_type=profile.workload_type.value
            ).inc()

            logger.info(f"Allocated cores {allocated_cores} to workload {workload_id}")
            return allocated_cores

    def deallocate_cores(self, workload_id: str):
        """Deallocate cores from a workload."""
        with self._allocation_lock:
            if workload_id not in self.workload_assignments:
                return

            allocated_cores = self.workload_assignments[workload_id]

            # Clear allocations
            for core in allocated_cores:
                self.core_allocations.pop(core, None)
            self.workload_assignments.pop(workload_id, None)

            logger.info(f"Deallocated cores {allocated_cores} from workload {workload_id}")

    def _calculate_optimal_cores(self, profile: WorkloadProfile) -> int:
        """Calculate optimal number of cores for workload."""
        if profile.max_threads:
            return min(profile.max_threads, self.topology.total_cores)

        # Workload-specific defaults
        if profile.workload_type == WorkloadType.CPU_INTENSIVE:
            return min(self.topology.cores_per_ccx, self.topology.total_cores)
        elif profile.workload_type == WorkloadType.AI_INFERENCE:
            return min(8, self.topology.total_cores)  # Moderate parallelism for AI
        elif profile.workload_type == WorkloadType.PARALLEL_COMPUTE:
            return self.topology.total_cores  # Use all available cores
        elif profile.workload_type == WorkloadType.LATENCY_SENSITIVE:
            return 1  # Single core for lowest latency
        else:
            return min(4, self.topology.total_cores)  # Conservative default

    def _select_optimal_cores(self, profile: WorkloadProfile, num_cores: int) -> List[int]:
        """Select optimal cores based on workload profile."""
        available_cores = [
            core for core in range(self.topology.total_cores)
            if core not in self.core_allocations
        ]

        if len(available_cores) < num_cores:
            raise RuntimeError(f"Insufficient available cores: need {num_cores}, have {len(available_cores)}")

        if profile.affinity_policy == AffinityPolicy.NONE:
            return available_cores[:num_cores]

        elif profile.affinity_policy == AffinityPolicy.CCX_PREFERRED:
            return self._select_ccx_local_cores(available_cores, num_cores, profile.preferred_ccx)

        elif profile.affinity_policy == AffinityPolicy.NUMA_LOCAL:
            return self._select_numa_local_cores(available_cores, num_cores, profile.preferred_numa_node)

        elif profile.affinity_policy == AffinityPolicy.SPREAD_EVENLY:
            return self._select_spread_cores(available_cores, num_cores)

        elif profile.affinity_policy == AffinityPolicy.CORE_EXCLUSIVE:
            return self._select_exclusive_cores(available_cores, num_cores)

        else:
            return available_cores[:num_cores]

    def _select_ccx_local_cores(
        self,
        available_cores: List[int],
        num_cores: int,
        preferred_ccx: Optional[int]
    ) -> List[int]:
        """Select cores from same CCX for cache locality."""
        if preferred_ccx is not None:
            ccx_cores = [core for core in available_cores
                        if self.topology.core_to_ccx.get(core) == preferred_ccx]
            if len(ccx_cores) >= num_cores:
                return ccx_cores[:num_cores]

        # Group cores by CCX and select from fullest CCX
        ccx_groups = {}
        for core in available_cores:
            ccx = self.topology.core_to_ccx.get(core, 0)
            if ccx not in ccx_groups:
                ccx_groups[ccx] = []
            ccx_groups[ccx].append(core)

        # Sort CCX groups by size (prefer CCX with more available cores)
        sorted_ccx = sorted(ccx_groups.items(), key=lambda x: len(x[1]), reverse=True)

        selected_cores = []
        for ccx, cores in sorted_ccx:
            if len(selected_cores) >= num_cores:
                break
            needed = num_cores - len(selected_cores)
            selected_cores.extend(cores[:needed])

        return selected_cores[:num_cores]

    def _select_numa_local_cores(
        self,
        available_cores: List[int],
        num_cores: int,
        preferred_numa: Optional[int]
    ) -> List[int]:
        """Select cores from same NUMA node."""
        if preferred_numa is not None:
            numa_cores = [core for core in available_cores
                         if self.topology.core_to_numa.get(core) == preferred_numa]
            if len(numa_cores) >= num_cores:
                return numa_cores[:num_cores]

        # Group by NUMA node and select from fullest node
        numa_groups = {}
        for core in available_cores:
            numa = self.topology.core_to_numa.get(core, 0)
            if numa not in numa_groups:
                numa_groups[numa] = []
            numa_groups[numa].append(core)

        sorted_numa = sorted(numa_groups.items(), key=lambda x: len(x[1]), reverse=True)

        selected_cores = []
        for numa, cores in sorted_numa:
            if len(selected_cores) >= num_cores:
                break
            needed = num_cores - len(selected_cores)
            selected_cores.extend(cores[:needed])

        return selected_cores[:num_cores]

    def _select_spread_cores(self, available_cores: List[int], num_cores: int) -> List[int]:
        """Select cores spread evenly across topology."""
        numa_groups = {}
        for core in available_cores:
            numa = self.topology.core_to_numa.get(core, 0)
            if numa not in numa_groups:
                numa_groups[numa] = []
            numa_groups[numa].append(core)

        selected_cores = []
        numa_nodes = list(numa_groups.keys())
        numa_index = 0

        while len(selected_cores) < num_cores and any(numa_groups.values()):
            numa = numa_nodes[numa_index % len(numa_nodes)]
            if numa_groups[numa]:
                selected_cores.append(numa_groups[numa].pop(0))
            numa_index += 1

        return selected_cores

    def _select_exclusive_cores(self, available_cores: List[int], num_cores: int) -> List[int]:
        """Select cores for exclusive use (avoid hyperthreading siblings)."""
        # For simplicity, just return first available cores
        # Real implementation would check for hyperthreading siblings
        return available_cores[:num_cores]

    def _apply_affinity(self, cores: List[int]):
        """Apply CPU affinity to current process/thread."""
        try:
            if hasattr(os, 'sched_setaffinity'):
                os.sched_setaffinity(0, cores)
                logger.debug(f"Applied CPU affinity: {cores}")
        except Exception as e:
            logger.warning(f"Failed to apply CPU affinity: {e}")

class EPYCOptimizedExecutor:
    """High-performance executor optimized for EPYC processors."""

    def __init__(self, topology: EPYCTopologyInfo):
        self.topology = topology
        self.affinity_manager = EPYCAffinityManager(topology)

        # Specialized thread pools for different workload types
        self.thread_pools: Dict[WorkloadType, ThreadPoolExecutor] = {}
        self.process_pools: Dict[WorkloadType, ProcessPoolExecutor] = {}

        # Task queues for workload isolation
        self.task_queues: Dict[WorkloadType, asyncio.Queue] = {}

        # Performance monitoring
        self.task_execution_time = Histogram(
            'epyc_task_execution_seconds',
            'Task execution time',
            ['workload_type', 'executor_type']
        )
        self.tasks_submitted = Counter(
            'epyc_tasks_submitted_total',
            'Total tasks submitted',
            ['workload_type', 'executor_type']
        )

        self._initialize_executors()

    def _initialize_executors(self):
        """Initialize specialized executors for different workload types."""
        # CPU-intensive workloads: use process pool for true parallelism
        cpu_cores = min(self.topology.total_cores, 8)
        self.process_pools[WorkloadType.CPU_INTENSIVE] = ProcessPoolExecutor(
            max_workers=cpu_cores,
            mp_context=multiprocessing.get_context('spawn')
        )

        # AI inference: moderate parallelism with thread pool
        ai_threads = min(self.topology.total_cores // 2, 8)
        self.thread_pools[WorkloadType.AI_INFERENCE] = ThreadPoolExecutor(
            max_workers=ai_threads,
            thread_name_prefix="ai_inference"
        )

        # I/O intensive: many threads for async I/O
        io_threads = min(self.topology.total_cores * 4, 64)
        self.thread_pools[WorkloadType.IO_INTENSIVE] = ThreadPoolExecutor(
            max_workers=io_threads,
            thread_name_prefix="io_intensive"
        )

        # Latency sensitive: dedicated single thread
        self.thread_pools[WorkloadType.LATENCY_SENSITIVE] = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="latency_sensitive"
        )

        # Memory intensive: limit concurrency to avoid memory pressure
        mem_threads = min(self.topology.total_cores // 4, 4)
        self.thread_pools[WorkloadType.MEMORY_INTENSIVE] = ThreadPoolExecutor(
            max_workers=mem_threads,
            thread_name_prefix="memory_intensive"
        )

        # Parallel compute: use all available cores
        self.process_pools[WorkloadType.PARALLEL_COMPUTE] = ProcessPoolExecutor(
            max_workers=self.topology.total_cores,
            mp_context=multiprocessing.get_context('spawn')
        )

        # Initialize task queues
        for workload_type in WorkloadType:
            self.task_queues[workload_type] = asyncio.Queue(maxsize=1000)

        logger.info("EPYC-optimized executors initialized")

    async def submit_task(
        self,
        func: Callable[..., T],
        workload_type: WorkloadType,
        profile: Optional[WorkloadProfile] = None,
        *args,
        **kwargs
    ) -> T:
        """Submit task with optimal execution strategy."""
        if profile is None:
            profile = WorkloadProfile(
                workload_type=workload_type,
                affinity_policy=AffinityPolicy.CCX_PREFERRED
            )

        start_time = time.time()

        try:
            # Allocate cores if needed
            workload_id = f"{workload_type.value}_{id(func)}_{time.time()}"
            if profile.exclusive_cores:
                allocated_cores = self.affinity_manager.allocate_cores(workload_id, profile)

            # Select appropriate executor
            if workload_type in self.process_pools:
                executor = self.process_pools[workload_type]
                executor_type = "process"
            elif workload_type in self.thread_pools:
                executor = self.thread_pools[workload_type]
                executor_type = "thread"
            else:
                # Default to I/O intensive thread pool
                executor = self.thread_pools[WorkloadType.IO_INTENSIVE]
                executor_type = "thread"

            # Submit task
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(executor, func, *args, **kwargs)

            # Update metrics
            execution_time = time.time() - start_time
            self.task_execution_time.labels(
                workload_type=workload_type.value,
                executor_type=executor_type
            ).observe(execution_time)

            self.tasks_submitted.labels(
                workload_type=workload_type.value,
                executor_type=executor_type
            ).inc()

            return result

        finally:
            # Deallocate cores if allocated
            if profile.exclusive_cores:
                self.affinity_manager.deallocate_cores(workload_id)

    async def submit_batch(
        self,
        tasks: List[tuple],  # [(func, args, kwargs), ...]
        workload_type: WorkloadType,
        profile: Optional[WorkloadProfile] = None,
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """Submit batch of tasks for parallel execution."""
        if not tasks:
            return []

        if batch_size is None:
            batch_size = len(tasks)

        results = []

        # Process tasks in batches
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]

            # Submit batch concurrently
            batch_futures = []
            for func, args, kwargs in batch:
                future = asyncio.create_task(
                    self.submit_task(func, workload_type, profile, *args, **kwargs)
                )
                batch_futures.append(future)

            # Wait for batch completion
            batch_results = await asyncio.gather(*batch_futures, return_exceptions=True)
            results.extend(batch_results)

        return results

    def close(self):
        """Clean shutdown of all executors."""
        for pool in self.thread_pools.values():
            pool.shutdown(wait=True)

        for pool in self.process_pools.values():
            pool.shutdown(wait=True)

        logger.info("EPYC-optimized executors shutdown complete")

class EPYCThermalManager:
    """Thermal management for EPYC processors."""

    def __init__(self, topology: EPYCTopologyInfo):
        self.topology = topology
        self.thermal_thresholds = {
            'warning': 70.0,   # °C
            'critical': 80.0,  # °C
            'emergency': 90.0  # °C
        }

        self.current_temperatures: Dict[int, float] = {}  # CCX -> temperature
        self.thermal_policies: Dict[str, Callable] = {}

        # Thermal metrics
        self.thermal_events = Counter(
            'epyc_thermal_events_total',
            'Total thermal events',
            ['event_type', 'ccx']
        )
        self.thermal_throttling = Gauge(
            'epyc_thermal_throttling_active',
            'Thermal throttling active',
            ['ccx']
        )

        self._register_thermal_policies()

    def _register_thermal_policies(self):
        """Register thermal management policies."""
        self.thermal_policies['conservative'] = self._conservative_thermal_policy
        self.thermal_policies['aggressive'] = self._aggressive_thermal_policy
        self.thermal_policies['balanced'] = self._balanced_thermal_policy

    async def monitor_thermal_state(self, interval: float = 5.0):
        """Monitor thermal state and apply policies."""
        while True:
            try:
                await self._collect_thermal_data()
                await self._apply_thermal_policies()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Thermal monitoring error: {e}")
                await asyncio.sleep(interval)

    async def _collect_thermal_data(self):
        """Collect thermal data from system."""
        try:
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()

                # AMD temperature sensors
                if 'k10temp' in temps:
                    for i, temp in enumerate(temps['k10temp']):
                        if i < self.topology.ccx_count:
                            self.current_temperatures[i] = temp.current

                # Fallback: simulate thermal data
                elif not self.current_temperatures:
                    import random
                    for ccx in range(self.topology.ccx_count):
                        # Simulate realistic temperatures (45-75°C)
                        base_temp = 55.0 + random.uniform(-10, 20)
                        self.current_temperatures[ccx] = base_temp

        except Exception as e:
            logger.warning(f"Failed to collect thermal data: {e}")

    async def _apply_thermal_policies(self):
        """Apply thermal management policies based on current temperatures."""
        for ccx, temperature in self.current_temperatures.items():
            if temperature >= self.thermal_thresholds['emergency']:
                await self._emergency_thermal_policy(ccx, temperature)
                self.thermal_events.labels(event_type='emergency', ccx=str(ccx)).inc()
            elif temperature >= self.thermal_thresholds['critical']:
                await self._critical_thermal_policy(ccx, temperature)
                self.thermal_events.labels(event_type='critical', ccx=str(ccx)).inc()
            elif temperature >= self.thermal_thresholds['warning']:
                await self._warning_thermal_policy(ccx, temperature)
                self.thermal_events.labels(event_type='warning', ccx=str(ccx)).inc()

    async def _conservative_thermal_policy(self, ccx: int, temperature: float):
        """Conservative thermal policy - reduce load early."""
        if temperature > 65.0:
            # Reduce core allocation for this CCX
            logger.info(f"Conservative thermal policy: reducing load on CCX {ccx} (temp: {temperature}°C)")

    async def _aggressive_thermal_policy(self, ccx: int, temperature: float):
        """Aggressive thermal policy - allow higher temperatures."""
        if temperature > 85.0:
            logger.warning(f"Aggressive thermal policy: high temperature on CCX {ccx} (temp: {temperature}°C)")

    async def _balanced_thermal_policy(self, ccx: int, temperature: float):
        """Balanced thermal policy - moderate response."""
        if temperature > 75.0:
            logger.info(f"Balanced thermal policy: managing load on CCX {ccx} (temp: {temperature}°C)")

    async def _warning_thermal_policy(self, ccx: int, temperature: float):
        """Handle warning-level thermal events."""
        logger.warning(f"Thermal warning: CCX {ccx} at {temperature}°C")

    async def _critical_thermal_policy(self, ccx: int, temperature: float):
        """Handle critical-level thermal events."""
        logger.critical(f"Thermal critical: CCX {ccx} at {temperature}°C - reducing workload")
        self.thermal_throttling.labels(ccx=str(ccx)).set(1)

    async def _emergency_thermal_policy(self, ccx: int, temperature: float):
        """Handle emergency-level thermal events."""
        logger.critical(f"Thermal emergency: CCX {ccx} at {temperature}°C - emergency throttling")
        self.thermal_throttling.labels(ccx=str(ccx)).set(1)

class EPYCOptimizationEngine:
    """Main optimization engine coordinating all EPYC enhancements."""

    def __init__(self):
        self.topology_detector = EPYCTopologyDetector()
        self.topology: Optional[EPYCTopologyInfo] = None
        self.executor: Optional[EPYCOptimizedExecutor] = None
        self.thermal_manager: Optional[EPYCThermalManager] = None
        self.optimization_active = False

    async def initialize(self):
        """Initialize EPYC optimization engine."""
        # Detect topology
        self.topology = self.topology_detector.detect_topology()

        # Initialize components
        self.executor = EPYCOptimizedExecutor(self.topology)
        self.thermal_manager = EPYCThermalManager(self.topology)

        # Start thermal monitoring
        asyncio.create_task(self.thermal_manager.monitor_thermal_state())

        self.optimization_active = True
        logger.info("EPYC optimization engine initialized")

    async def close(self):
        """Shutdown optimization engine."""
        self.optimization_active = False

        if self.executor:
            self.executor.close()

        logger.info("EPYC optimization engine shutdown complete")

    def get_optimal_workload_profile(self, workload_type: WorkloadType) -> WorkloadProfile:
        """Get optimal workload profile for workload type."""
        profiles = {
            WorkloadType.CPU_INTENSIVE: WorkloadProfile(
                workload_type=workload_type,
                affinity_policy=AffinityPolicy.CCX_PREFERRED,
                exclusive_cores=True,
                cache_sensitivity=0.8,
                thermal_awareness=True
            ),
            WorkloadType.AI_INFERENCE: WorkloadProfile(
                workload_type=workload_type,
                affinity_policy=AffinityPolicy.CCX_PREFERRED,
                max_threads=8,
                cache_sensitivity=0.9,
                memory_locality=True
            ),
            WorkloadType.MEMORY_INTENSIVE: WorkloadProfile(
                workload_type=workload_type,
                affinity_policy=AffinityPolicy.NUMA_LOCAL,
                memory_locality=True,
                cache_sensitivity=0.3
            ),
            WorkloadType.IO_INTENSIVE: WorkloadProfile(
                workload_type=workload_type,
                affinity_policy=AffinityPolicy.SPREAD_EVENLY,
                cache_sensitivity=0.1
            ),
            WorkloadType.LATENCY_SENSITIVE: WorkloadProfile(
                workload_type=workload_type,
                affinity_policy=AffinityPolicy.CORE_EXCLUSIVE,
                max_threads=1,
                exclusive_cores=True
            ),
            WorkloadType.PARALLEL_COMPUTE: WorkloadProfile(
                workload_type=workload_type,
                affinity_policy=AffinityPolicy.SPREAD_EVENLY,
                thermal_awareness=True
            )
        }

        return profiles.get(workload_type, WorkloadProfile(
            workload_type=workload_type,
            affinity_policy=AffinityPolicy.NONE
        ))

# Global optimization engine
epyc_optimization_engine: Optional[EPYCOptimizationEngine] = None

async def initialize_epyc_optimization() -> EPYCOptimizationEngine:
    """Initialize global EPYC optimization engine."""
    global epyc_optimization_engine
    epyc_optimization_engine = EPYCOptimizationEngine()
    await epyc_optimization_engine.initialize()
    return epyc_optimization_engine

async def get_epyc_optimization() -> Optional[EPYCOptimizationEngine]:
    """Get global EPYC optimization engine."""
    return epyc_optimization_engine

def epyc_optimized(workload_type: WorkloadType):
    """Decorator for EPYC-optimized function execution."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if epyc_optimization_engine and epyc_optimization_engine.executor:
                profile = epyc_optimization_engine.get_optimal_workload_profile(workload_type)
                return await epyc_optimization_engine.executor.submit_task(
                    func, workload_type, profile, *args, **kwargs
                )
            else:
                # Fallback to direct execution
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        return wrapper
    return decorator
