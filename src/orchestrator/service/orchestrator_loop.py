"""
PTaaS Orchestrator Loop with NATS Consumer Backpressure and WFQ Integration
Optimized for AMD EPYC 7002 high-throughput workloads
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import nats
from nats.aio.client import Client as NATSClient
from nats.js import JetStreamContext
from nats.js.api import ConsumerConfig, AckPolicy, ReplayPolicy
import uuid

from ..config import perf_config
from ..executors.runner import (
    get_task_runner,
    TaskContext,
    TaskType,
    TaskPriority,
    HighPerformanceTaskRunner,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkflowJob:
    """PTaaS workflow job definition"""

    job_id: str
    tenant_id: str
    job_type: str  # "discovery", "vulnerability_scan", "compliance_check"
    targets: List[str]
    scan_profile: str  # "quick", "comprehensive", "stealth", "web-focused"
    priority: TaskPriority
    metadata: Dict[str, Any]
    created_at: float
    timeout_sec: int = 300


class PTaaSOrchestrator:
    """High-performance PTaaS orchestrator with NATS JetStream backpressure"""

    def __init__(self):
        self.config = perf_config
        self.nats_client: Optional[NATSClient] = None
        self.js: Optional[JetStreamContext] = None
        self.task_runner: Optional[HighPerformanceTaskRunner] = None

        # Consumer management
        self.consumers: Dict[str, Any] = {}
        self.active_jobs = 0
        self.max_concurrent_jobs = self.config.ptaas_workers

        # Performance tracking
        self.metrics = {
            "jobs_received": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "redeliveries": 0,
            "consumer_lag": 0,
            "backpressure_events": 0,
            "last_fairness_index": 0.0,
        }

        # Workflow modules registry
        self.workflow_modules = {
            "discovery": self._execute_discovery_scan,
            "vulnerability_scan": self._execute_vulnerability_scan,
            "compliance_check": self._execute_compliance_check,
            "threat_simulation": self._execute_threat_simulation,
        }

        self._shutdown = False

    async def start(self) -> None:
        """Start the orchestrator with NATS connectivity"""
        logger.info("Starting PTaaS Orchestrator with EPYC optimization")

        # Initialize task runner
        self.task_runner = await get_task_runner()

        # Connect to NATS
        await self._connect_nats()

        # Setup JetStream consumers with backpressure
        await self._setup_jetstream_consumers()

        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())

        logger.info(
            f"PTaaS Orchestrator started with {self.max_concurrent_jobs} concurrent job limit"
        )

    async def stop(self) -> None:
        """Gracefully stop the orchestrator"""
        self._shutdown = True

        # Stop consumers
        for consumer in self.consumers.values():
            if hasattr(consumer, "stop"):
                await consumer.stop()

        # Close NATS connection
        if self.nats_client:
            await self.nats_client.close()

        logger.info("PTaaS Orchestrator stopped")

    async def _connect_nats(self) -> None:
        """Connect to NATS with optimized settings for EPYC"""
        try:
            self.nats_client = await nats.connect(
                servers=["nats://localhost:4222"],
                max_reconnect_attempts=10,
                reconnect_time_wait=2.0,
                max_outstanding_pings=3,
                ping_interval=60,
                # High-performance settings for EPYC
                max_payload=1024 * 1024,  # 1MB max payload
                pedantic=False,  # Disable strict parsing for performance
                verbose=False,  # Reduce chattiness
            )

            self.js = self.nats_client.jetstream()
            logger.info("Connected to NATS JetStream")

        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}")
            raise

    async def _setup_jetstream_consumers(self) -> None:
        """Setup JetStream consumers with backpressure control"""

        # Consumer configuration optimized for EPYC throughput
        consumer_config = ConsumerConfig(
            ack_policy=AckPolicy.EXPLICIT,
            replay_policy=ReplayPolicy.INSTANT,
            max_ack_pending=self.config.nats_max_ack_pending,
            ack_wait=self.config.nats_ack_wait_ms * 1_000_000,  # Convert to nanoseconds
            max_deliver=self.config.nats_max_deliver,
            max_waiting=self.config.nats_max_inflight,
        )

        # Setup consumers for different job types with tenant isolation
        job_types = [
            "discovery",
            "vulnerability_scan",
            "compliance_check",
            "threat_simulation",
        ]

        for job_type in job_types:
            try:
                # Consumer per job type for parallel processing
                consumer = await self.js.subscribe(
                    subject=f"ptaas.jobs.{job_type}",
                    queue="ptaas-workers",
                    config=consumer_config,
                    cb=self._create_job_handler(job_type),
                )

                self.consumers[job_type] = consumer
                logger.info(
                    f"Setup consumer for {job_type} with max_ack_pending={self.config.nats_max_ack_pending}"
                )

            except Exception as e:
                logger.error(f"Failed to setup consumer for {job_type}: {e}")

    def _create_job_handler(self, job_type: str) -> Callable:
        """Create message handler with backpressure control"""

        async def handle_job_message(msg):
            """Handle incoming job message with backpressure"""

            # Backpressure control: check if we're at capacity
            if self.active_jobs >= self.max_concurrent_jobs:
                self.metrics["backpressure_events"] += 1
                logger.warning(
                    f"Backpressure triggered: {self.active_jobs}/{self.max_concurrent_jobs} jobs active"
                )

                # NAK with delay to implement backpressure
                await msg.nak(delay=5.0)  # 5 second delay
                return

            try:
                # Parse job message
                job_data = json.loads(msg.data.decode())
                job = WorkflowJob(
                    job_id=job_data.get("job_id", str(uuid.uuid4())),
                    tenant_id=job_data.get("tenant_id", "default"),
                    job_type=job_type,
                    targets=job_data.get("targets", []),
                    scan_profile=job_data.get("scan_profile", "quick"),
                    priority=TaskPriority(job_data.get("priority", 2)),
                    metadata=job_data.get("metadata", {}),
                    created_at=time.time(),
                    timeout_sec=job_data.get(
                        "timeout_sec",
                        self._get_timeout_for_profile(
                            job_data.get("scan_profile", "quick")
                        ),
                    ),
                )

                self.metrics["jobs_received"] += 1
                self.active_jobs += 1

                # Execute job asynchronously
                asyncio.create_task(self._process_job(job, msg))

            except Exception as e:
                logger.error(f"Error parsing job message: {e}")
                await msg.ack()  # ACK to prevent redelivery of malformed messages

        return handle_job_message

    async def _process_job(self, job: WorkflowJob, msg) -> None:
        """Process a PTaaS job with performance tracking"""
        start_time = time.time()

        try:
            logger.info(
                f"Processing job {job.job_id} ({job.job_type}) for tenant {job.tenant_id}"
            )

            # Determine task type based on scan profile
            task_type = self._get_task_type_for_profile(job.scan_profile)

            # Create task context
            context = TaskContext(
                task_id=job.job_id,
                tenant_id=job.tenant_id,
                task_type=task_type,
                priority=job.priority,
                timeout_sec=job.timeout_sec,
                metadata=job.metadata,
            )

            # Execute workflow module
            workflow_func = self.workflow_modules.get(job.job_type)
            if not workflow_func:
                raise ValueError(f"Unknown job type: {job.job_type}")

            # Submit to high-performance task runner
            result = await self.task_runner.submit_task(workflow_func, context, job)

            if result.success:
                self.metrics["jobs_completed"] += 1
                logger.info(
                    f"Job {job.job_id} completed in {result.context.execution_time_ms:.1f}ms"
                )

                # Publish results to tenant-specific subject
                await self._publish_job_result(job, result)

            else:
                self.metrics["jobs_failed"] += 1
                logger.error(f"Job {job.job_id} failed: {result.error}")

                # Publish error to tenant
                await self._publish_job_error(job, result.error)

            # ACK the message after successful processing
            await msg.ack()

        except Exception as e:
            self.metrics["jobs_failed"] += 1
            logger.error(f"Error processing job {job.job_id}: {e}")

            # NAK for redelivery on processing errors
            if (
                hasattr(msg, "metadata")
                and msg.metadata.num_delivered < self.config.nats_max_deliver
            ):
                self.metrics["redeliveries"] += 1
                await msg.nak()
            else:
                # Max deliveries reached, ACK to prevent infinite redelivery
                await msg.ack()

        finally:
            self.active_jobs -= 1
            execution_time = (time.time() - start_time) * 1000
            logger.debug(
                f"Job {job.job_id} processing completed in {execution_time:.1f}ms"
            )

    def _get_task_type_for_profile(self, scan_profile: str) -> TaskType:
        """Map scan profile to task type for resource allocation"""
        profile_mapping = {
            "quick": TaskType.FAST,  # Light network scans
            "comprehensive": TaskType.SLOW,  # Deep analysis
            "stealth": TaskType.MEDIUM,  # Balanced approach
            "web-focused": TaskType.MEDIUM,  # Web application scanning
        }
        return profile_mapping.get(scan_profile, TaskType.MEDIUM)

    def _get_timeout_for_profile(self, scan_profile: str) -> int:
        """Get timeout based on scan profile"""
        timeouts = {
            "quick": self.config.timeout_fast_sec,
            "comprehensive": self.config.timeout_slow_sec,
            "stealth": self.config.timeout_medium_sec,
            "web-focused": self.config.timeout_medium_sec,
        }
        return timeouts.get(scan_profile, self.config.timeout_medium_sec)

    async def _execute_discovery_scan(self, job: WorkflowJob) -> Dict[str, Any]:
        """Execute network discovery scan"""
        results = {"job_id": job.job_id, "scan_type": "discovery", "targets": []}

        for target in job.targets:
            # Simulate discovery scan (replace with actual implementation)
            await asyncio.sleep(0.1)  # Simulate network I/O

            target_result = {
                "target": target,
                "status": "completed",
                "discovered_hosts": f"10.0.1.{hash(target) % 254 + 1}",  # Mock discovery
                "scan_duration_ms": 100,
            }
            results["targets"].append(target_result)

        return results

    async def _execute_vulnerability_scan(self, job: WorkflowJob) -> Dict[str, Any]:
        """Execute vulnerability scanning"""
        results = {"job_id": job.job_id, "scan_type": "vulnerability", "findings": []}

        # Simulate CPU-intensive vulnerability analysis
        for target in job.targets:
            await asyncio.sleep(0.5)  # Simulate scan time

            # Mock vulnerability findings
            finding = {
                "target": target,
                "severity": "medium",
                "cve_id": f"CVE-2024-{hash(target) % 9999:04d}",
                "description": f"Mock vulnerability on {target}",
            }
            results["findings"].append(finding)

        return results

    async def _execute_compliance_check(self, job: WorkflowJob) -> Dict[str, Any]:
        """Execute compliance checking"""
        results = {"job_id": job.job_id, "scan_type": "compliance", "checks": []}

        compliance_frameworks = job.metadata.get("frameworks", ["PCI-DSS"])

        for framework in compliance_frameworks:
            await asyncio.sleep(0.2)  # Simulate check time

            check_result = {
                "framework": framework,
                "status": "compliant",
                "score": 85 + (hash(job.job_id) % 15),  # Mock score 85-100
                "recommendations": [
                    "Enable additional logging",
                    "Update security policies",
                ],
            }
            results["checks"].append(check_result)

        return results

    async def _execute_threat_simulation(self, job: WorkflowJob) -> Dict[str, Any]:
        """Execute threat simulation"""
        return {
            "job_id": job.job_id,
            "simulation_type": job.metadata.get("simulation_type", "apt_simulation"),
            "attack_vectors": job.metadata.get("attack_vectors", ["spear_phishing"]),
            "status": "completed",
            "recommendations": ["Implement additional email filtering"],
        }

    async def _publish_job_result(self, job: WorkflowJob, result) -> None:
        """Publish job results to tenant-specific subject"""
        try:
            result_data = {
                "job_id": job.job_id,
                "tenant_id": job.tenant_id,
                "status": "completed",
                "result": result.result,
                "metrics": result.metrics,
                "completed_at": time.time(),
            }

            await self.js.publish(
                subject=f"ptaas.results.{job.tenant_id}",
                payload=json.dumps(result_data).encode(),
            )

        except Exception as e:
            logger.error(f"Failed to publish result for job {job.job_id}: {e}")

    async def _publish_job_error(self, job: WorkflowJob, error: Exception) -> None:
        """Publish job error to tenant-specific subject"""
        try:
            error_data = {
                "job_id": job.job_id,
                "tenant_id": job.tenant_id,
                "status": "failed",
                "error": str(error),
                "failed_at": time.time(),
            }

            await self.js.publish(
                subject=f"ptaas.errors.{job.tenant_id}",
                payload=json.dumps(error_data).encode(),
            )

        except Exception as e:
            logger.error(f"Failed to publish error for job {job.job_id}: {e}")

    async def _monitoring_loop(self) -> None:
        """Background monitoring and metrics collection"""
        while not self._shutdown:
            try:
                # Get task runner metrics
                if self.task_runner:
                    runner_metrics = self.task_runner.get_performance_metrics()
                    self.metrics["last_fairness_index"] = runner_metrics[
                        "fairness_index"
                    ]

                # Calculate consumer lag (simplified)
                total_lag = 0
                for consumer in self.consumers.values():
                    if hasattr(consumer, "_info") and consumer._info:
                        total_lag += consumer._info.num_pending

                self.metrics["consumer_lag"] = total_lag

                # Log performance summary
                logger.info(
                    f"PTaaS Orchestrator Metrics: "
                    f"Active={self.active_jobs}/{self.max_concurrent_jobs}, "
                    f"Received={self.metrics['jobs_received']}, "
                    f"Completed={self.metrics['jobs_completed']}, "
                    f"Failed={self.metrics['jobs_failed']}, "
                    f"ConsumerLag={self.metrics['consumer_lag']}, "
                    f"Backpressure={self.metrics['backpressure_events']}, "
                    f"Fairness={self.metrics['last_fairness_index']:.3f}"
                )

                await asyncio.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)

    def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Get current orchestrator metrics"""
        error_rate = self.metrics["jobs_failed"] / max(self.metrics["jobs_received"], 1)

        return {
            "active_jobs": self.active_jobs,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "jobs_received": self.metrics["jobs_received"],
            "jobs_completed": self.metrics["jobs_completed"],
            "jobs_failed": self.metrics["jobs_failed"],
            "error_rate": error_rate,
            "redeliveries": self.metrics["redeliveries"],
            "consumer_lag": self.metrics["consumer_lag"],
            "backpressure_events": self.metrics["backpressure_events"],
            "fairness_index": self.metrics["last_fairness_index"],
            "utilization": self.active_jobs / self.max_concurrent_jobs,
        }


# Global orchestrator instance
orchestrator: Optional[PTaaSOrchestrator] = None


async def get_orchestrator() -> PTaaSOrchestrator:
    """Get or create the global orchestrator instance"""
    global orchestrator
    if not orchestrator:
        orchestrator = PTaaSOrchestrator()
        await orchestrator.start()
    return orchestrator


async def shutdown_orchestrator() -> None:
    """Shutdown the global orchestrator"""
    global orchestrator
    if orchestrator:
        await orchestrator.stop()
        orchestrator = None
