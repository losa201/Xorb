"""
High-Performance Task Runner for PTaaS on AMD EPYC 7002
Implements async task graphs with process pools for CPU-bound operations
"""

import asyncio
import time
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Callable, Union, Awaitable
from enum import Enum
import psutil
import numpy as np
from collections import defaultdict, deque

from ..config import (
    perf_config,
    get_optimal_worker_distribution,
    calculate_fairness_index,
)

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task types based on resource requirements"""

    FAST = "fast"  # Network discovery, ping sweeps (CPU light, I/O heavy)
    MEDIUM = "medium"  # Vulnerability scans, web crawling (CPU medium, I/O heavy)
    SLOW = "slow"  # Deep analysis, compliance checks (CPU heavy)
    ORCHESTRATION = "orchestration"  # Workflow management (CPU light, coordination)


class TaskPriority(Enum):
    """Task execution priority"""

    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class TaskContext:
    """Execution context for a PTaaS task"""

    task_id: str
    tenant_id: str
    task_type: TaskType
    priority: TaskPriority
    timeout_sec: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def queue_time_ms(self) -> Optional[float]:
        """Time spent in queue before execution"""
        if self.started_at:
            return (self.started_at - self.created_at) * 1000
        return None

    @property
    def execution_time_ms(self) -> Optional[float]:
        """Task execution time"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return None


@dataclass
class TaskResult:
    """Result of task execution"""

    context: TaskContext
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class WeightedFairQueue:
    """Weighted Fair Queueing implementation for tenant isolation"""

    def __init__(self):
        self.tenant_queues: Dict[str, Dict[TaskPriority, deque]] = defaultdict(
            lambda: {priority: deque() for priority in TaskPriority}
        )
        self.tenant_weights = defaultdict(lambda: 1.0)
        self.tenant_virtual_time = defaultdict(float)
        self.global_virtual_time = 0.0
        self.tenant_stats = defaultdict(
            lambda: {"executed": 0, "queue_time_total": 0.0}
        )

    def enqueue(self, task: TaskContext) -> None:
        """Add task to tenant-specific priority queue"""
        queue = self.tenant_queues[task.tenant_id][task.priority]
        queue.append(task)

    def dequeue(self) -> Optional[TaskContext]:
        """Dequeue next task using weighted fair queueing"""
        eligible_tenants = []

        # Find tenants with tasks and calculate virtual finish times
        for tenant_id, queues in self.tenant_queues.items():
            for priority in TaskPriority:
                if queues[priority]:
                    weight = self.tenant_weights[tenant_id] * (
                        4 - priority.value
                    )  # Higher priority = higher weight
                    virtual_finish_time = max(
                        self.tenant_virtual_time[tenant_id], self.global_virtual_time
                    ) + (1.0 / weight)
                    eligible_tenants.append((virtual_finish_time, tenant_id, priority))

        if not eligible_tenants:
            return None

        # Select tenant with earliest virtual finish time
        eligible_tenants.sort()
        _, selected_tenant, selected_priority = eligible_tenants[0]

        # Dequeue task and update virtual times
        task = self.tenant_queues[selected_tenant][selected_priority].popleft()
        weight = self.tenant_weights[selected_tenant] * (4 - selected_priority.value)
        service_time = 1.0 / weight

        self.tenant_virtual_time[selected_tenant] += service_time
        self.global_virtual_time = max(
            self.global_virtual_time, self.tenant_virtual_time[selected_tenant]
        )

        # Update stats
        self.tenant_stats[selected_tenant]["executed"] += 1
        if task.queue_time_ms:
            self.tenant_stats[selected_tenant]["queue_time_total"] += task.queue_time_ms

        return task

    def get_fairness_metrics(self) -> Dict[str, Any]:
        """Calculate fairness metrics across tenants"""
        tenant_throughputs = []
        total_executed = 0

        for tenant_id, stats in self.tenant_stats.items():
            executed = stats["executed"]
            tenant_throughputs.append(executed)
            total_executed += executed

        fairness_index = calculate_fairness_index(tenant_throughputs)

        return {
            "fairness_index": fairness_index,
            "total_executed": total_executed,
            "tenant_count": len(self.tenant_stats),
            "tenant_throughputs": dict(
                zip(self.tenant_stats.keys(), tenant_throughputs)
            ),
        }


class HighPerformanceTaskRunner:
    """High-performance task runner optimized for AMD EPYC architecture"""

    def __init__(self):
        self.config = perf_config
        self.wfq = WeightedFairQueue()

        # Initialize executor pools based on EPYC configuration
        self.process_pool = ProcessPoolExecutor(
            max_workers=self.config.cpu_pool_size,
            mp_context=mp.get_context("spawn"),  # Better for NUMA systems
        )
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.io_concurrency, thread_name_prefix="ptaas-io"
        )

        # Worker distribution for different task types
        self.worker_distribution = get_optimal_worker_distribution(
            self.config.ptaas_workers
        )

        # Semaphores for concurrency control
        self.semaphores = {
            TaskType.FAST: asyncio.Semaphore(self.worker_distribution["fast_scanners"]),
            TaskType.MEDIUM: asyncio.Semaphore(
                self.worker_distribution["medium_scanners"]
            ),
            TaskType.SLOW: asyncio.Semaphore(self.worker_distribution["slow_scanners"]),
            TaskType.ORCHESTRATION: asyncio.Semaphore(
                self.worker_distribution["orchestration"]
            ),
        }

        # Performance metrics
        self.metrics = {
            "tasks_executed": 0,
            "tasks_failed": 0,
            "total_queue_time_ms": 0.0,
            "total_execution_time_ms": 0.0,
            "latency_p95_ms": deque(maxlen=1000),
            "tenant_fairness_history": deque(maxlen=100),
        }

        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown = False

    async def start(self) -> None:
        """Start the task runner and monitoring"""
        logger.info(
            f"Starting high-performance task runner on EPYC with {self.config.ptaas_workers} workers"
        )
        logger.info(f"Worker distribution: {self.worker_distribution}")

        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def stop(self) -> None:
        """Gracefully stop the task runner"""
        self._shutdown = True

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        # Shutdown executor pools
        self.process_pool.shutdown(wait=True)
        self.thread_pool.shutdown(wait=True)

    async def submit_task(
        self,
        task_func: Union[Callable, Awaitable],
        context: TaskContext,
        *args,
        **kwargs,
    ) -> TaskResult:
        """Submit a task for execution with WFQ scheduling"""

        # Add to weighted fair queue
        self.wfq.enqueue(context)

        # Wait for scheduling
        scheduled_context = await self._wait_for_scheduling(context)
        if not scheduled_context:
            return TaskResult(
                context=context,
                success=False,
                error=Exception("Task cancelled during scheduling"),
            )

        return await self._execute_task(task_func, scheduled_context, *args, **kwargs)

    async def _wait_for_scheduling(self, context: TaskContext) -> Optional[TaskContext]:
        """Wait for task to be scheduled via WFQ"""
        while not self._shutdown:
            # Try to dequeue this specific task (simplified for demo)
            scheduled_task = self.wfq.dequeue()
            if scheduled_task and scheduled_task.task_id == context.task_id:
                scheduled_task.started_at = time.time()
                return scheduled_task
            elif scheduled_task:
                # Re-queue other tasks (in real implementation, this would be handled differently)
                self.wfq.enqueue(scheduled_task)

            await asyncio.sleep(0.01)  # Small delay to prevent busy waiting

        return None

    async def _execute_task(
        self,
        task_func: Union[Callable, Awaitable],
        context: TaskContext,
        *args,
        **kwargs,
    ) -> TaskResult:
        """Execute task with appropriate executor based on type"""

        semaphore = self.semaphores[context.task_type]

        async with semaphore:
            try:
                if context.task_type in [TaskType.SLOW]:
                    # CPU-bound tasks use process pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.process_pool,
                        self._run_cpu_task,
                        task_func,
                        context,
                        args,
                        kwargs,
                    )
                elif context.task_type in [TaskType.FAST, TaskType.MEDIUM]:
                    # I/O-bound tasks use thread pool or async execution
                    if asyncio.iscoroutinefunction(task_func):
                        result = await asyncio.wait_for(
                            task_func(*args, **kwargs), timeout=context.timeout_sec
                        )
                    else:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            self.thread_pool, task_func, *args, **kwargs
                        )
                else:
                    # Orchestration tasks run directly in async context
                    result = await task_func(*args, **kwargs)

                context.completed_at = time.time()

                # Update metrics
                self._update_metrics(context, True)

                return TaskResult(
                    context=context,
                    success=True,
                    result=result,
                    metrics={
                        "queue_time_ms": context.queue_time_ms,
                        "execution_time_ms": context.execution_time_ms,
                        "memory_peak_mb": self._get_memory_usage(),
                    },
                )

            except Exception as e:
                context.completed_at = time.time()
                self._update_metrics(context, False)

                return TaskResult(
                    context=context,
                    success=False,
                    error=e,
                    metrics={
                        "queue_time_ms": context.queue_time_ms,
                        "execution_time_ms": context.execution_time_ms,
                    },
                )

    def _run_cpu_task(
        self, task_func: Callable, context: TaskContext, args: tuple, kwargs: dict
    ) -> Any:
        """Run CPU-bound task in process pool (must be picklable)"""
        try:
            # Set CPU affinity for NUMA optimization if available
            if self.config.numa_aware and hasattr(psutil.Process(), "cpu_affinity"):
                # Let OS handle NUMA placement, but pin to specific cores if needed
                pass

            return task_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"CPU task {context.task_id} failed: {e}")
            raise

    def _update_metrics(self, context: TaskContext, success: bool) -> None:
        """Update performance metrics"""
        if success:
            self.metrics["tasks_executed"] += 1
        else:
            self.metrics["tasks_failed"] += 1

        if context.queue_time_ms:
            self.metrics["total_queue_time_ms"] += context.queue_time_ms

        if context.execution_time_ms:
            self.metrics["total_execution_time_ms"] += context.execution_time_ms
            # Track P95 latency
            total_latency = (context.queue_time_ms or 0) + context.execution_time_ms
            self.metrics["latency_p95_ms"].append(total_latency)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    async def _monitoring_loop(self) -> None:
        """Background monitoring and metrics collection"""
        while not self._shutdown:
            try:
                # Update fairness metrics
                fairness_metrics = self.wfq.get_fairness_metrics()
                self.metrics["tenant_fairness_history"].append(
                    fairness_metrics["fairness_index"]
                )

                # Log performance summary every 60 seconds
                if self.metrics["tasks_executed"] > 0:
                    avg_queue_time = (
                        self.metrics["total_queue_time_ms"]
                        / self.metrics["tasks_executed"]
                    )
                    avg_exec_time = (
                        self.metrics["total_execution_time_ms"]
                        / self.metrics["tasks_executed"]
                    )

                    if self.metrics["latency_p95_ms"]:
                        p95_latency = np.percentile(
                            list(self.metrics["latency_p95_ms"]), 95
                        )
                    else:
                        p95_latency = 0

                    logger.info(
                        f"PTaaS Performance: "
                        f"Tasks={self.metrics['tasks_executed']}, "
                        f"Failed={self.metrics['tasks_failed']}, "
                        f"AvgQueue={avg_queue_time:.1f}ms, "
                        f"AvgExec={avg_exec_time:.1f}ms, "
                        f"P95={p95_latency:.1f}ms, "
                        f"Fairness={fairness_metrics['fairness_index']:.3f}"
                    )

                await asyncio.sleep(60)  # Monitor every minute

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        fairness_metrics = self.wfq.get_fairness_metrics()

        # Calculate P95 latency
        p95_latency = 0
        if self.metrics["latency_p95_ms"]:
            p95_latency = np.percentile(list(self.metrics["latency_p95_ms"]), 95)

        return {
            "tasks_executed": self.metrics["tasks_executed"],
            "tasks_failed": self.metrics["tasks_failed"],
            "error_rate": self.metrics["tasks_failed"]
            / max(self.metrics["tasks_executed"], 1),
            "p95_latency_ms": p95_latency,
            "fairness_index": fairness_metrics["fairness_index"],
            "memory_usage_mb": self._get_memory_usage(),
            "worker_distribution": self.worker_distribution,
            "tenant_metrics": fairness_metrics,
        }


# Global task runner instance
task_runner: Optional[HighPerformanceTaskRunner] = None


async def get_task_runner() -> HighPerformanceTaskRunner:
    """Get or create the global task runner instance"""
    global task_runner
    if not task_runner:
        task_runner = HighPerformanceTaskRunner()
        await task_runner.start()
    return task_runner


async def shutdown_task_runner() -> None:
    """Shutdown the global task runner"""
    global task_runner
    if task_runner:
        await task_runner.stop()
        task_runner = None
