"""Redis-based job queue implementation."""
import asyncio
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import redis.asyncio as redis
from redis.asyncio import Redis

from .models import (
    JobDefinition, JobExecution, JobStatus, JobPriority,
    JobStatusUpdate, QueueStats, RetryPolicy
)


logger = logging.getLogger(__name__)


class JobQueue:
    """Redis-based job queue with priority support."""
    
    def __init__(self, redis_client: Redis, queue_prefix: str = "job_queue"):
        self.redis = redis_client
        self.queue_prefix = queue_prefix
        
        # Queue name patterns
        self.pending_queue = f"{queue_prefix}:pending"
        self.running_queue = f"{queue_prefix}:running"
        self.completed_queue = f"{queue_prefix}:completed"
        self.failed_queue = f"{queue_prefix}:failed"
        self.dead_letter_queue = f"{queue_prefix}:dlq"
        
        # Job data storage
        self.job_data_key = f"{queue_prefix}:jobs"
        self.execution_data_key = f"{queue_prefix}:executions"
        
        # Idempotency tracking
        self.idempotency_key = f"{queue_prefix}:idempotency"
        
        # Worker tracking
        self.worker_key = f"{queue_prefix}:workers"
        self.heartbeat_key = f"{queue_prefix}:heartbeat"
        
        # Metrics
        self.metrics_key = f"{queue_prefix}:metrics"
    
    async def enqueue_job(
        self, 
        job_def: JobDefinition,
        execution_time: Optional[datetime] = None
    ) -> JobExecution:
        """Enqueue a job for processing."""
        
        # Check idempotency
        if job_def.idempotency_key:
            existing_job_id = await self.redis.get(
                f"{self.idempotency_key}:{job_def.idempotency_key}"
            )
            if existing_job_id:
                logger.info(f"Job with idempotency key {job_def.idempotency_key} already exists")
                # Return existing job execution
                return await self.get_job_execution(UUID(existing_job_id.decode()))
        
        # Create job execution
        execution = JobExecution(
            job_id=job_def.id,
            status=JobStatus.PENDING,
            created_at=datetime.utcnow()
        )
        
        # Store job data
        await self.redis.hset(
            self.job_data_key,
            str(job_def.id),
            job_def.model_dump_json()
        )
        
        # Store execution data
        await self.redis.hset(
            self.execution_data_key,
            str(execution.id),
            execution.model_dump_json()
        )
        
        # Set idempotency key if provided
        if job_def.idempotency_key:
            await self.redis.setex(
                f"{self.idempotency_key}:{job_def.idempotency_key}",
                3600 * 24,  # 24 hours
                str(job_def.id)
            )
        
        # Calculate execution time
        if execution_time is None:
            if job_def.scheduled_at:
                execution_time = job_def.scheduled_at
            elif job_def.delay_seconds > 0:
                execution_time = datetime.utcnow() + timedelta(seconds=job_def.delay_seconds)
            else:
                execution_time = datetime.utcnow()
        
        # Add to appropriate queue
        queue_name = self._get_priority_queue_name(job_def.queue_name, job_def.priority)
        
        if execution_time <= datetime.utcnow():
            # Add to pending queue immediately
            await self.redis.lpush(queue_name, str(job_def.id))
        else:
            # Schedule for later execution
            score = execution_time.timestamp()
            await self.redis.zadd(
                f"{self.queue_prefix}:scheduled",
                {str(job_def.id): score}
            )
        
        # Update metrics
        await self._update_metrics("jobs_enqueued", 1)
        
        logger.info(f"Enqueued job {job_def.id} of type {job_def.job_type}")
        return execution
    
    async def dequeue_job(
        self, 
        queues: List[str],
        worker_id: str,
        timeout: int = 10
    ) -> Optional[Tuple[JobDefinition, JobExecution]]:
        """Dequeue a job for processing."""
        
        # Move scheduled jobs to pending if their time has come
        await self._process_scheduled_jobs()
        
        # Build priority-ordered queue list
        priority_queues = []
        for queue in queues:
            for priority in [JobPriority.CRITICAL, JobPriority.HIGH, JobPriority.NORMAL, JobPriority.LOW]:
                priority_queues.append(self._get_priority_queue_name(queue, priority))
        
        # Blocking pop from highest priority queue
        result = await self.redis.brpop(priority_queues, timeout=timeout)
        
        if not result:
            return None
        
        queue_name, job_id_bytes = result
        job_id = UUID(job_id_bytes.decode())
        
        # Get job definition
        job_data = await self.redis.hget(self.job_data_key, str(job_id))
        if not job_data:
            logger.error(f"Job data not found for {job_id}")
            return None
        
        job_def = JobDefinition.model_validate_json(job_data)
        
        # Create new execution for retry or get existing
        execution_data = await self.redis.hget(self.execution_data_key, str(job_id))
        if execution_data:
            execution = JobExecution.model_validate_json(execution_data)
            execution.attempt += 1
        else:
            execution = JobExecution(
                job_id=job_id,
                attempt=1
            )
        
        # Update execution status
        execution.status = JobStatus.RUNNING
        execution.started_at = datetime.utcnow()
        execution.worker_id = worker_id
        
        # Store updated execution
        await self.redis.hset(
            self.execution_data_key,
            str(execution.id),
            execution.model_dump_json()
        )
        
        # Move to running queue for tracking
        await self.redis.lpush(self.running_queue, str(job_id))
        
        # Set execution timeout
        await self.redis.setex(
            f"{self.queue_prefix}:timeout:{job_id}",
            job_def.execution_timeout,
            worker_id
        )
        
        logger.info(f"Dequeued job {job_id} for worker {worker_id}")
        return job_def, execution
    
    async def complete_job(
        self, 
        job_id: UUID, 
        execution_id: UUID,
        success: bool,
        result: Optional[Dict] = None,
        error: Optional[str] = None
    ) -> None:
        """Mark job as completed."""
        
        # Get execution
        execution_data = await self.redis.hget(self.execution_data_key, str(execution_id))
        if not execution_data:
            logger.error(f"Execution data not found for {execution_id}")
            return
        
        execution = JobExecution.model_validate_json(execution_data)
        
        # Update execution
        execution.completed_at = datetime.utcnow()
        execution.duration_seconds = (
            execution.completed_at - execution.started_at
        ).total_seconds() if execution.started_at else None
        
        if success:
            execution.status = JobStatus.COMPLETED
            execution.result = result
            
            # Move to completed queue
            await self.redis.lrem(self.running_queue, 0, str(job_id))
            await self.redis.lpush(self.completed_queue, str(job_id))
            
            # Update metrics
            await self._update_metrics("jobs_completed", 1)
            
        else:
            execution.status = JobStatus.FAILED
            execution.error = error
            
            # Get job definition for retry policy
            job_data = await self.redis.hget(self.job_data_key, str(job_id))
            if job_data:
                job_def = JobDefinition.model_validate_json(job_data)
                
                # Check if we should retry
                if execution.attempt < job_def.retry_policy.max_attempts:
                    await self._schedule_retry(job_def, execution)
                else:
                    # Move to dead letter queue
                    await self._move_to_dead_letter_queue(job_id)
            
            # Remove from running queue
            await self.redis.lrem(self.running_queue, 0, str(job_id))
            
            # Update metrics
            await self._update_metrics("jobs_failed", 1)
        
        # Store updated execution
        await self.redis.hset(
            self.execution_data_key,
            str(execution.id),
            execution.model_dump_json()
        )
        
        # Clean up timeout
        await self.redis.delete(f"{self.queue_prefix}:timeout:{job_id}")
        
        logger.info(f"Completed job {job_id} with status {execution.status}")
    
    async def cancel_job(self, job_id: UUID, reason: Optional[str] = None) -> bool:
        """Cancel a pending or running job."""
        
        # Remove from all pending queues
        removed = False
        for queue in await self.redis.keys(f"{self.queue_prefix}:*:pending"):
            result = await self.redis.lrem(queue, 0, str(job_id))
            if result > 0:
                removed = True
        
        # Remove from scheduled queue
        result = await self.redis.zrem(f"{self.queue_prefix}:scheduled", str(job_id))
        if result > 0:
            removed = True
        
        # For running jobs, we can only mark them as cancelled
        # The worker should check for cancellation
        if await self.redis.lrem(self.running_queue, 0, str(job_id)) > 0:
            removed = True
            # Set cancellation flag
            await self.redis.setex(
                f"{self.queue_prefix}:cancel:{job_id}",
                3600,  # 1 hour
                reason or "Cancelled by user"
            )
        
        if removed:
            # Update execution status
            execution_data = await self.redis.hget(self.execution_data_key, str(job_id))
            if execution_data:
                execution = JobExecution.model_validate_json(execution_data)
                execution.status = JobStatus.CANCELLED
                execution.error = reason or "Cancelled"
                execution.completed_at = datetime.utcnow()
                
                await self.redis.hset(
                    self.execution_data_key,
                    str(execution.id),
                    execution.model_dump_json()
                )
            
            logger.info(f"Cancelled job {job_id}")
        
        return removed
    
    async def get_job_status(self, job_id: UUID) -> Optional[JobExecution]:
        """Get current job execution status."""
        execution_data = await self.redis.hget(self.execution_data_key, str(job_id))
        if execution_data:
            return JobExecution.model_validate_json(execution_data)
        return None
    
    async def get_queue_stats(self, queue_name: str) -> QueueStats:
        """Get queue statistics."""
        
        # Count jobs in different states
        pending_count = 0
        for priority in JobPriority:
            pq_name = self._get_priority_queue_name(queue_name, priority)
            pending_count += await self.redis.llen(pq_name)
        
        running_count = await self.redis.llen(self.running_queue)
        
        # Get 24h metrics (simplified - would use time series in production)
        completed_24h = await self._get_metric("jobs_completed") or 0
        failed_24h = await self._get_metric("jobs_failed") or 0
        
        return QueueStats(
            queue_name=queue_name,
            pending_jobs=pending_count,
            running_jobs=running_count,
            completed_jobs_24h=completed_24h,
            failed_jobs_24h=failed_24h
        )
    
    async def cleanup_old_jobs(self, older_than_days: int = 7) -> int:
        """Clean up old completed/failed jobs."""
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        cutoff_timestamp = cutoff.timestamp()
        
        # This is a simplified cleanup - would be more sophisticated in production
        cleaned = 0
        
        # Remove from completed queue (keeping recent ones)
        # In production, would check timestamps properly
        completed_jobs = await self.redis.lrange(self.completed_queue, 100, -1)
        if completed_jobs:
            await self.redis.ltrim(self.completed_queue, 0, 99)
            cleaned += len(completed_jobs)
        
        logger.info(f"Cleaned up {cleaned} old jobs")
        return cleaned
    
    async def _process_scheduled_jobs(self) -> None:
        """Move scheduled jobs to pending queue when their time comes."""
        now = datetime.utcnow().timestamp()
        
        # Get jobs that should be executed now
        scheduled_jobs = await self.redis.zrangebyscore(
            f"{self.queue_prefix}:scheduled",
            0, now
        )
        
        for job_id_bytes in scheduled_jobs:
            job_id = job_id_bytes.decode()
            
            # Get job definition to determine queue and priority
            job_data = await self.redis.hget(self.job_data_key, job_id)
            if job_data:
                job_def = JobDefinition.model_validate_json(job_data)
                queue_name = self._get_priority_queue_name(job_def.queue_name, job_def.priority)
                
                # Move to pending queue
                await self.redis.lpush(queue_name, job_id)
                
                # Remove from scheduled queue
                await self.redis.zrem(f"{self.queue_prefix}:scheduled", job_id)
    
    async def _schedule_retry(self, job_def: JobDefinition, execution: JobExecution) -> None:
        """Schedule job for retry with exponential backoff."""
        policy = job_def.retry_policy
        
        # Calculate delay with exponential backoff
        delay = min(
            policy.initial_delay * (policy.exponential_base ** (execution.attempt - 1)),
            policy.max_delay
        )
        
        # Add jitter if enabled
        if policy.jitter:
            jitter = random.uniform(0, delay * 0.1)  # Up to 10% jitter
            delay += jitter
        
        retry_time = datetime.utcnow() + timedelta(seconds=delay)
        
        # Schedule retry
        await self.redis.zadd(
            f"{self.queue_prefix}:scheduled",
            {str(job_def.id): retry_time.timestamp()}
        )
        
        # Update execution status
        execution.status = JobStatus.RETRY
        await self.redis.hset(
            self.execution_data_key,
            str(execution.id),
            execution.model_dump_json()
        )
        
        logger.info(f"Scheduled retry for job {job_def.id} in {delay:.2f} seconds")
    
    async def _move_to_dead_letter_queue(self, job_id: UUID) -> None:
        """Move job to dead letter queue."""
        await self.redis.lpush(self.dead_letter_queue, str(job_id))
        
        # Update execution status
        execution_data = await self.redis.hget(self.execution_data_key, str(job_id))
        if execution_data:
            execution = JobExecution.model_validate_json(execution_data)
            execution.status = JobStatus.DEAD_LETTER
            execution.completed_at = datetime.utcnow()
            
            await self.redis.hset(
                self.execution_data_key,
                str(execution.id),
                execution.model_dump_json()
            )
        
        logger.warning(f"Moved job {job_id} to dead letter queue")
    
    def _get_priority_queue_name(self, queue_name: str, priority: JobPriority) -> str:
        """Get priority-specific queue name."""
        return f"{self.queue_prefix}:{queue_name}:{priority.value}:pending"
    
    async def _update_metrics(self, metric_name: str, value: int) -> None:
        """Update metric counter."""
        await self.redis.hincrby(self.metrics_key, metric_name, value)
    
    async def _get_metric(self, metric_name: str) -> Optional[int]:
        """Get metric value."""
        value = await self.redis.hget(self.metrics_key, metric_name)
        return int(value) if value else None
    
    async def get_job_execution(self, job_id: UUID) -> Optional[JobExecution]:
        """Get job execution by job ID."""
        execution_data = await self.redis.hget(self.execution_data_key, str(job_id))
        if execution_data:
            return JobExecution.model_validate_json(execution_data)
        return None