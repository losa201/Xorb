"""Job worker implementation."""
import asyncio
import logging
import signal
import socket
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from uuid import uuid4

import redis.asyncio as redis

from .models import JobDefinition, JobExecution, JobResult, JobType, JobStatus
from .queue import JobQueue


logger = logging.getLogger(__name__)


class JobWorker:
    """Async job worker with graceful shutdown and health monitoring."""
    
    def __init__(
        self,
        redis_client: redis.Redis,
        worker_id: Optional[str] = None,
        queues: List[str] = None,
        concurrency: int = 1,
        queue_prefix: str = "job_queue"
    ):
        self.redis = redis_client
        self.worker_id = worker_id or f"worker-{uuid4().hex[:8]}"
        self.queues = queues or ["default"]
        self.concurrency = concurrency
        self.queue = JobQueue(redis_client, queue_prefix)
        
        # Worker state
        self.is_running = False
        self.shutdown_requested = False
        self.current_jobs: Set[str] = set()
        self.handlers: Dict[JobType, callable] = {}
        
        # Stats
        self.jobs_completed = 0
        self.jobs_failed = 0
        self.started_at = datetime.utcnow()
        
        # Health monitoring
        self.heartbeat_interval = 30  # seconds
        self.last_heartbeat = datetime.utcnow()
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def register_handler(self, job_type: JobType, handler: callable) -> None:
        """Register a job handler function."""
        self.handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")
    
    async def start(self) -> None:
        """Start the worker."""
        logger.info(f"Starting worker {self.worker_id}")
        self.is_running = True
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Start worker tasks
        worker_tasks = []
        for i in range(self.concurrency):
            task = asyncio.create_task(self._worker_loop(f"{self.worker_id}-{i}"))
            worker_tasks.append(task)
        
        try:
            # Wait for shutdown signal
            await asyncio.gather(heartbeat_task, *worker_tasks)
        except asyncio.CancelledError:
            logger.info("Worker tasks cancelled")
        finally:
            await self._cleanup()
    
    async def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info(f"Stopping worker {self.worker_id}")
        self.shutdown_requested = True
        
        # Wait for current jobs to complete (with timeout)
        timeout = 60  # 1 minute
        start_time = datetime.utcnow()
        
        while self.current_jobs and (datetime.utcnow() - start_time).seconds < timeout:
            logger.info(f"Waiting for {len(self.current_jobs)} jobs to complete...")
            await asyncio.sleep(1)
        
        if self.current_jobs:
            logger.warning(f"Force stopping with {len(self.current_jobs)} jobs still running")
        
        self.is_running = False
    
    async def _worker_loop(self, worker_instance_id: str) -> None:
        """Main worker loop."""
        logger.info(f"Worker instance {worker_instance_id} started")
        
        while self.is_running and not self.shutdown_requested:
            try:
                # Dequeue job
                job_data = await self.queue.dequeue_job(
                    queues=self.queues,
                    worker_id=worker_instance_id,
                    timeout=5  # 5 second timeout
                )
                
                if job_data is None:
                    continue  # Timeout, try again
                
                job_def, execution = job_data
                
                # Track current job
                self.current_jobs.add(str(job_def.id))
                
                try:
                    # Process job
                    await self._process_job(job_def, execution)
                finally:
                    # Remove from current jobs
                    self.current_jobs.discard(str(job_def.id))
                
            except Exception as e:
                logger.error(f"Worker loop error: {e}", exc_info=True)
                await asyncio.sleep(1)  # Brief pause before retrying
        
        logger.info(f"Worker instance {worker_instance_id} stopped")
    
    async def _process_job(self, job_def: JobDefinition, execution: JobExecution) -> None:
        """Process a single job."""
        logger.info(f"Processing job {job_def.id} of type {job_def.job_type}")
        
        start_time = datetime.utcnow()
        success = False
        result = None
        error = None
        
        try:
            # Check for cancellation
            if await self._is_job_cancelled(job_def.id):
                logger.info(f"Job {job_def.id} was cancelled")
                return
            
            # Get handler
            handler = self.handlers.get(job_def.job_type)
            if not handler:
                raise ValueError(f"No handler registered for job type: {job_def.job_type}")
            
            # Execute job with timeout
            job_result = await asyncio.wait_for(
                self._execute_job_handler(handler, job_def),
                timeout=job_def.execution_timeout
            )
            
            if job_result.success:
                success = True
                result = job_result.result
                self.jobs_completed += 1
                logger.info(f"Job {job_def.id} completed successfully")
            else:
                error = job_result.error
                self.jobs_failed += 1
                logger.error(f"Job {job_def.id} failed: {error}")
        
        except asyncio.TimeoutError:
            error = f"Job timed out after {job_def.execution_timeout} seconds"
            self.jobs_failed += 1
            logger.error(f"Job {job_def.id} timed out")
        
        except Exception as e:
            error = str(e)
            self.jobs_failed += 1
            logger.error(f"Job {job_def.id} failed with exception: {e}", exc_info=True)
        
        finally:
            # Record completion
            try:
                await self.queue.complete_job(
                    job_id=job_def.id,
                    execution_id=execution.id,
                    success=success,
                    result=result,
                    error=error
                )
            except Exception as e:
                logger.error(f"Failed to record job completion: {e}")
    
    async def _execute_job_handler(self, handler: callable, job_def: JobDefinition) -> JobResult:
        """Execute job handler with proper error handling."""
        try:
            # Call handler (may be sync or async)
            if asyncio.iscoroutinefunction(handler):
                result = await handler(job_def)
            else:
                # Run sync handler in thread pool
                result = await asyncio.get_event_loop().run_in_executor(
                    None, handler, job_def
                )
            
            # Ensure result is JobResult
            if isinstance(result, JobResult):
                return result
            elif isinstance(result, dict):
                return JobResult(success=True, result=result)
            else:
                return JobResult(success=True, result={"data": result})
        
        except Exception as e:
            logger.error(f"Job handler error: {e}")
            return JobResult(
                success=False,
                error=str(e)
            )
    
    async def _is_job_cancelled(self, job_id: str) -> bool:
        """Check if job has been cancelled."""
        cancel_key = f"{self.queue.queue_prefix}:cancel:{job_id}"
        return await self.redis.exists(cancel_key) > 0
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat."""
        while self.is_running and not self.shutdown_requested:
            try:
                await self._send_heartbeat()
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)  # Shorter retry interval
    
    async def _send_heartbeat(self) -> None:
        """Send worker heartbeat."""
        self.last_heartbeat = datetime.utcnow()
        
        worker_data = {
            "worker_id": self.worker_id,
            "hostname": socket.gethostname(),
            "status": "active" if not self.shutdown_requested else "stopping",
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "current_jobs": len(self.current_jobs),
            "queues": self.queues,
            "started_at": self.started_at.isoformat()
        }
        
        # Store worker data with TTL
        await self.redis.setex(
            f"{self.queue.worker_key}:{self.worker_id}",
            self.heartbeat_interval * 3,  # TTL is 3x heartbeat interval
            str(worker_data)
        )
    
    async def _cleanup(self) -> None:
        """Cleanup worker resources."""
        logger.info(f"Cleaning up worker {self.worker_id}")
        
        # Remove worker from registry
        await self.redis.delete(f"{self.queue.worker_key}:{self.worker_id}")
        
        # Log final stats
        uptime = datetime.utcnow() - self.started_at
        logger.info(
            f"Worker {self.worker_id} shutting down. "
            f"Uptime: {uptime}, "
            f"Jobs completed: {self.jobs_completed}, "
            f"Jobs failed: {self.jobs_failed}"
        )
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            asyncio.create_task(self.stop())
        
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        except ValueError:
            # Signals not available (e.g., in tests)
            pass


# Example job handlers
async def evidence_processing_handler(job_def: JobDefinition) -> JobResult:
    """Example handler for evidence processing jobs."""
    logger.info(f"Processing evidence: {job_def.payload}")
    
    # Simulate processing
    await asyncio.sleep(2)
    
    return JobResult(
        success=True,
        result={
            "processed_at": datetime.utcnow().isoformat(),
            "evidence_id": job_def.payload.get("evidence_id"),
            "status": "processed"
        }
    )


async def malware_scan_handler(job_def: JobDefinition) -> JobResult:
    """Example handler for malware scanning jobs."""
    logger.info(f"Scanning for malware: {job_def.payload}")
    
    # Simulate scan
    await asyncio.sleep(5)
    
    return JobResult(
        success=True,
        result={
            "scan_result": "clean",
            "scan_time": datetime.utcnow().isoformat(),
            "file_hash": job_def.payload.get("file_hash")
        }
    )


def threat_analysis_handler(job_def: JobDefinition) -> JobResult:
    """Example synchronous handler for threat analysis."""
    logger.info(f"Analyzing threat: {job_def.payload}")
    
    # Simulate analysis
    import time
    time.sleep(3)
    
    return JobResult(
        success=True,
        result={
            "threat_level": "medium",
            "analysis_time": datetime.utcnow().isoformat(),
            "indicators": job_def.payload.get("indicators", [])
        }
    )


# Default handler registry
DEFAULT_HANDLERS = {
    JobType.EVIDENCE_PROCESSING: evidence_processing_handler,
    JobType.MALWARE_SCAN: malware_scan_handler,
    JobType.THREAT_ANALYSIS: threat_analysis_handler,
}