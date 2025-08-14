"""Job orchestration service."""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID

import redis.asyncio as redis

from .models import (
    JobDefinition, JobExecution, JobScheduleRequest, BulkJobRequest,
    JobCancelRequest, JobFilter, JobType, JobStatus, JobPriority,
    QueueStats, WorkerStats
)
from .queue import JobQueue


logger = logging.getLogger(__name__)


class JobService:
    """Service for job scheduling and management."""

    def __init__(self, redis_client: redis.Redis, queue_prefix: str = "job_queue"):
        self.redis = redis_client
        self.queue = JobQueue(redis_client, queue_prefix)

    async def schedule_job(self, request: JobScheduleRequest) -> Dict:
        """Schedule a single job."""

        # Create job definition
        job_def = JobDefinition(
            job_type=request.job_type,
            priority=request.priority,
            payload=request.payload,
            queue_name=request.queue_name,
            scheduled_at=request.scheduled_at,
            delay_seconds=request.delay_seconds,
            idempotency_key=request.idempotency_key,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            tags=request.tags,
            custom_metadata=request.custom_metadata
        )

        # Apply configuration overrides
        if request.retry_policy:
            job_def.retry_policy = request.retry_policy
        if request.execution_timeout:
            job_def.execution_timeout = request.execution_timeout

        # Enqueue job
        execution = await self.queue.enqueue_job(job_def)

        logger.info(f"Scheduled job {job_def.id} of type {job_def.job_type}")

        return {
            "job_id": str(job_def.id),
            "execution_id": str(execution.id),
            "status": execution.status.value,
            "created_at": execution.created_at.isoformat(),
            "queue_name": job_def.queue_name,
            "priority": job_def.priority.value
        }

    async def schedule_bulk_jobs(self, request: BulkJobRequest) -> Dict:
        """Schedule multiple jobs as a batch."""

        results = []
        failed = []

        for job_request in request.jobs:
            try:
                result = await self.schedule_job(job_request)
                results.append(result)
            except Exception as e:
                failed.append({
                    "job_type": job_request.job_type.value,
                    "error": str(e)
                })
                logger.error(f"Failed to schedule job in batch: {e}")

        return {
            "batch_id": request.batch_id,
            "total_jobs": len(request.jobs),
            "scheduled": len(results),
            "failed": len(failed),
            "results": results,
            "errors": failed
        }

    async def cancel_jobs(self, request: JobCancelRequest) -> Dict:
        """Cancel one or more jobs."""

        results = []

        for job_id in request.job_ids:
            try:
                cancelled = await self.queue.cancel_job(job_id, request.reason)
                results.append({
                    "job_id": str(job_id),
                    "cancelled": cancelled
                })
                if cancelled:
                    logger.info(f"Cancelled job {job_id}")
                else:
                    logger.warning(f"Could not cancel job {job_id} (may not exist or already completed)")
            except Exception as e:
                results.append({
                    "job_id": str(job_id),
                    "cancelled": False,
                    "error": str(e)
                })
                logger.error(f"Error cancelling job {job_id}: {e}")

        return {
            "total_jobs": len(request.job_ids),
            "results": results
        }

    async def get_job_status(self, job_id: UUID) -> Optional[Dict]:
        """Get job status and execution details."""

        execution = await self.queue.get_job_status(job_id)
        if not execution:
            return None

        # Get job definition
        job_data = await self.redis.hget(self.queue.job_data_key, str(job_id))
        if not job_data:
            return None

        job_def = JobDefinition.model_validate_json(job_data)

        return {
            "job_id": str(job_def.id),
            "job_type": job_def.job_type.value,
            "status": execution.status.value,
            "priority": job_def.priority.value,
            "queue_name": job_def.queue_name,
            "attempt": execution.attempt,
            "created_at": execution.created_at.isoformat(),
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "duration_seconds": execution.duration_seconds,
            "worker_id": execution.worker_id,
            "progress": execution.progress,
            "progress_message": execution.progress_message,
            "result": execution.result,
            "error": execution.error,
            "tenant_id": str(job_def.tenant_id) if job_def.tenant_id else None,
            "user_id": job_def.user_id,
            "tags": job_def.tags,
            "idempotency_key": job_def.idempotency_key
        }

    async def list_jobs(
        self,
        job_filter: Optional[JobFilter] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict:
        """List jobs with filtering."""

        # This is a simplified implementation
        # In production, would use proper indexing and filtering

        all_job_ids = []

        # Get job IDs from different sources based on status filter
        if not job_filter or not job_filter.statuses:
            # Get all jobs
            all_job_ids.extend(await self.redis.lrange(self.queue.pending_queue, 0, -1))
            all_job_ids.extend(await self.redis.lrange(self.queue.running_queue, 0, -1))
            all_job_ids.extend(await self.redis.lrange(self.queue.completed_queue, 0, -1))
            all_job_ids.extend(await self.redis.lrange(self.queue.failed_queue, 0, -1))
        else:
            # Filter by status
            for status in job_filter.statuses:
                if status == JobStatus.PENDING:
                    all_job_ids.extend(await self.redis.lrange(self.queue.pending_queue, 0, -1))
                elif status == JobStatus.RUNNING:
                    all_job_ids.extend(await self.redis.lrange(self.queue.running_queue, 0, -1))
                elif status == JobStatus.COMPLETED:
                    all_job_ids.extend(await self.redis.lrange(self.queue.completed_queue, 0, -1))
                elif status == JobStatus.FAILED:
                    all_job_ids.extend(await self.redis.lrange(self.queue.failed_queue, 0, -1))

        # Convert to strings and remove duplicates
        job_ids = list(set(job_id.decode() if isinstance(job_id, bytes) else str(job_id) for job_id in all_job_ids))

        # Apply pagination
        paginated_ids = job_ids[offset:offset + limit]

        # Get job details
        jobs = []
        for job_id in paginated_ids:
            job_status = await self.get_job_status(UUID(job_id))
            if job_status:
                # Apply additional filters
                if self._job_matches_filter(job_status, job_filter):
                    jobs.append(job_status)

        return {
            "total": len(job_ids),
            "limit": limit,
            "offset": offset,
            "jobs": jobs
        }

    async def get_queue_stats(self, queue_name: str = "default") -> Dict:
        """Get queue statistics."""

        stats = await self.queue.get_queue_stats(queue_name)

        return {
            "queue_name": stats.queue_name,
            "pending_jobs": stats.pending_jobs,
            "running_jobs": stats.running_jobs,
            "completed_jobs_24h": stats.completed_jobs_24h,
            "failed_jobs_24h": stats.failed_jobs_24h,
            "avg_execution_time": stats.avg_execution_time,
            "oldest_pending_job": stats.oldest_pending_job.isoformat() if stats.oldest_pending_job else None
        }

    async def get_worker_stats(self) -> List[Dict]:
        """Get statistics for all active workers."""

        workers = []

        # Get all worker keys
        worker_keys = await self.redis.keys(f"{self.queue.worker_key}:*")

        for key in worker_keys:
            worker_data = await self.redis.get(key)
            if worker_data:
                try:
                    # Parse worker data (stored as string representation of dict)
                    import ast
                    worker_info = ast.literal_eval(worker_data.decode())
                    workers.append(worker_info)
                except Exception as e:
                    logger.error(f"Error parsing worker data: {e}")

        return workers

    async def retry_failed_jobs(
        self,
        job_ids: Optional[List[UUID]] = None,
        max_age_hours: int = 24
    ) -> Dict:
        """Retry failed jobs."""

        if job_ids:
            # Retry specific jobs
            retry_count = 0
            for job_id in job_ids:
                execution = await self.queue.get_job_status(job_id)
                if execution and execution.status == JobStatus.FAILED:
                    # Get job definition and re-enqueue
                    job_data = await self.redis.hget(self.queue.job_data_key, str(job_id))
                    if job_data:
                        job_def = JobDefinition.model_validate_json(job_data)
                        await self.queue.enqueue_job(job_def)
                        retry_count += 1

            return {"retried": retry_count}

        else:
            # Retry all recent failed jobs
            cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)

            # Get failed job IDs
            failed_job_ids = await self.redis.lrange(self.queue.failed_queue, 0, -1)

            retry_count = 0
            for job_id_bytes in failed_job_ids:
                job_id = UUID(job_id_bytes.decode())
                execution = await self.queue.get_job_status(job_id)

                if (execution and
                    execution.completed_at and
                    execution.completed_at > cutoff):

                    # Re-enqueue job
                    job_data = await self.redis.hget(self.queue.job_data_key, str(job_id))
                    if job_data:
                        job_def = JobDefinition.model_validate_json(job_data)
                        await self.queue.enqueue_job(job_def)
                        retry_count += 1

            return {"retried": retry_count}

    async def cleanup_old_jobs(self, older_than_days: int = 7) -> Dict:
        """Clean up old completed and failed jobs."""

        cleaned_count = await self.queue.cleanup_old_jobs(older_than_days)

        return {
            "cleaned_jobs": cleaned_count,
            "older_than_days": older_than_days
        }

    def _job_matches_filter(self, job_data: Dict, job_filter: Optional[JobFilter]) -> bool:
        """Check if job matches the filter criteria."""

        if not job_filter:
            return True

        # Filter by job types
        if job_filter.job_types:
            if job_data["job_type"] not in [jt.value for jt in job_filter.job_types]:
                return False

        # Filter by priorities
        if job_filter.priorities:
            if job_data["priority"] not in [p.value for p in job_filter.priorities]:
                return False

        # Filter by queue names
        if job_filter.queue_names:
            if job_data["queue_name"] not in job_filter.queue_names:
                return False

        # Filter by tenant
        if job_filter.tenant_id:
            if job_data.get("tenant_id") != str(job_filter.tenant_id):
                return False

        # Filter by user
        if job_filter.user_id:
            if job_data.get("user_id") != job_filter.user_id:
                return False

        # Filter by creation time
        created_at = datetime.fromisoformat(job_data["created_at"])

        if job_filter.created_after and created_at < job_filter.created_after:
            return False

        if job_filter.created_before and created_at > job_filter.created_before:
            return False

        # Filter by tags
        if job_filter.tags:
            job_tags = job_data.get("tags", [])
            if not any(tag in job_tags for tag in job_filter.tags):
                return False

        return True
