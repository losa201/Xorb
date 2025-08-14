"""Job orchestration API routes."""
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..auth.dependencies import require_auth, require_permissions
from ..auth.models import Permission, UserClaims
from ..middleware.tenant_context import require_tenant_context
from ..jobs.service import JobService
from ..jobs.models import (
    JobScheduleRequest, BulkJobRequest, JobCancelRequest,
    JobFilter, JobType, JobStatus, JobPriority
)
from ..security.input_validation import validate_pagination
from ..infrastructure.observability import add_trace_context
import redis.asyncio as redis
import os
import structlog

logger = structlog.get_logger("jobs_api")
router = APIRouter(prefix="/api/jobs", tags=["Jobs"])

# Initialize job service (would be dependency injected in production)
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(redis_url)
job_service = JobService(redis_client)


@router.post("/schedule", response_model=Dict)
async def schedule_job(
    request: Request,
    job_request: JobScheduleRequest,
    current_user: UserClaims = Depends(require_permissions(Permission.JOBS_WRITE))
):
    """Schedule a new job for execution."""
    tenant_id = require_tenant_context(request)

    # Set tenant context in job request
    job_request.tenant_id = tenant_id
    job_request.user_id = current_user.sub

    try:
        result = await job_service.schedule_job(job_request)

        logger.info(
            "Job scheduled",
            job_id=result["job_id"],
            job_type=job_request.job_type,
            tenant_id=str(tenant_id),
            user_id=current_user.sub
        )

        add_trace_context(
            operation="schedule_job",
            job_type=job_request.job_type.value,
            tenant_id=str(tenant_id)
        )

        return result

    except Exception as e:
        logger.error(
            "Failed to schedule job",
            job_type=job_request.job_type,
            tenant_id=str(tenant_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to schedule job"
        )


@router.post("/schedule-bulk", response_model=Dict)
async def schedule_bulk_jobs(
    request: Request,
    bulk_request: BulkJobRequest,
    current_user: UserClaims = Depends(require_permissions(Permission.JOBS_WRITE))
):
    """Schedule multiple jobs as a batch."""
    tenant_id = require_tenant_context(request)

    # Set tenant context for all jobs
    for job_req in bulk_request.jobs:
        job_req.tenant_id = tenant_id
        job_req.user_id = current_user.sub

    try:
        result = await job_service.schedule_bulk_jobs(bulk_request)

        logger.info(
            "Bulk jobs scheduled",
            batch_id=bulk_request.batch_id,
            total_jobs=result["total_jobs"],
            scheduled=result["scheduled"],
            failed=result["failed"],
            tenant_id=str(tenant_id),
            user_id=current_user.sub
        )

        return result

    except Exception as e:
        logger.error(
            "Failed to schedule bulk jobs",
            batch_id=bulk_request.batch_id,
            tenant_id=str(tenant_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to schedule bulk jobs"
        )


@router.post("/cancel", response_model=Dict)
async def cancel_jobs(
    request: Request,
    cancel_request: JobCancelRequest,
    current_user: UserClaims = Depends(require_permissions(Permission.JOBS_CANCEL))
):
    """Cancel one or more jobs."""
    tenant_id = require_tenant_context(request)

    try:
        result = await job_service.cancel_jobs(cancel_request)

        logger.info(
            "Jobs cancellation requested",
            job_count=len(cancel_request.job_ids),
            reason=cancel_request.reason,
            tenant_id=str(tenant_id),
            user_id=current_user.sub
        )

        return result

    except Exception as e:
        logger.error(
            "Failed to cancel jobs",
            job_ids=[str(jid) for jid in cancel_request.job_ids],
            tenant_id=str(tenant_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel jobs"
        )


@router.get("/status/{job_id}", response_model=Dict)
async def get_job_status(
    request: Request,
    job_id: UUID,
    current_user: UserClaims = Depends(require_permissions(Permission.JOBS_READ))
):
    """Get job status and execution details."""
    tenant_id = require_tenant_context(request)

    try:
        job_status = await job_service.get_job_status(job_id)

        if not job_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )

        # Verify tenant access
        if job_status.get("tenant_id") != str(tenant_id) and not current_user.is_super_admin():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )

        return job_status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get job status",
            job_id=str(job_id),
            tenant_id=str(tenant_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get job status"
        )


@router.get("/list", response_model=Dict)
async def list_jobs(
    request: Request,
    limit: int = 20,
    offset: int = 0,
    job_types: Optional[List[JobType]] = None,
    statuses: Optional[List[JobStatus]] = None,
    priorities: Optional[List[JobPriority]] = None,
    current_user: UserClaims = Depends(require_permissions(Permission.JOBS_READ))
):
    """List jobs with filtering and pagination."""
    tenant_id = require_tenant_context(request)

    # Validate pagination
    limit, offset = validate_pagination(limit, offset, max_limit=100)

    # Create filter
    job_filter = JobFilter(
        job_types=job_types,
        statuses=statuses,
        priorities=priorities,
        tenant_id=tenant_id if not current_user.is_super_admin() else None
    )

    try:
        result = await job_service.list_jobs(
            job_filter=job_filter,
            limit=limit,
            offset=offset
        )

        logger.info(
            "Listed jobs",
            tenant_id=str(tenant_id),
            count=len(result["jobs"]),
            total=result["total"],
            user_id=current_user.sub
        )

        return result

    except Exception as e:
        logger.error(
            "Failed to list jobs",
            tenant_id=str(tenant_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list jobs"
        )


@router.get("/queue-stats/{queue_name}", response_model=Dict)
async def get_queue_stats(
    queue_name: str = "default",
    current_user: UserClaims = Depends(require_permissions(Permission.JOBS_READ))
):
    """Get queue statistics."""
    try:
        stats = await job_service.get_queue_stats(queue_name)

        logger.info(
            "Retrieved queue stats",
            queue_name=queue_name,
            pending_jobs=stats["pending_jobs"],
            running_jobs=stats["running_jobs"],
            user_id=current_user.sub
        )

        return stats

    except Exception as e:
        logger.error(
            "Failed to get queue stats",
            queue_name=queue_name,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get queue statistics"
        )


@router.get("/worker-stats", response_model=List[Dict])
async def get_worker_stats(
    current_user: UserClaims = Depends(require_permissions(Permission.JOBS_READ))
):
    """Get statistics for all active workers."""
    try:
        stats = await job_service.get_worker_stats()

        logger.info(
            "Retrieved worker stats",
            worker_count=len(stats),
            user_id=current_user.sub
        )

        return stats

    except Exception as e:
        logger.error(
            "Failed to get worker stats",
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get worker statistics"
        )


@router.post("/retry-failed", response_model=Dict)
async def retry_failed_jobs(
    request: Request,
    job_ids: Optional[List[UUID]] = None,
    max_age_hours: int = 24,
    current_user: UserClaims = Depends(require_permissions(Permission.JOBS_WRITE))
):
    """Retry failed jobs."""
    try:
        result = await job_service.retry_failed_jobs(
            job_ids=job_ids,
            max_age_hours=max_age_hours
        )

        logger.info(
            "Retried failed jobs",
            retried_count=result["retried"],
            job_ids=job_ids,
            max_age_hours=max_age_hours,
            user_id=current_user.sub
        )

        return result

    except Exception as e:
        logger.error(
            "Failed to retry jobs",
            job_ids=job_ids,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retry failed jobs"
        )


@router.post("/cleanup", response_model=Dict)
async def cleanup_old_jobs(
    older_than_days: int = 7,
    current_user: UserClaims = Depends(require_permissions(Permission.JOBS_WRITE))
):
    """Clean up old completed and failed jobs."""
    if not current_user.is_super_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin access required for cleanup operations"
        )

    try:
        result = await job_service.cleanup_old_jobs(older_than_days)

        logger.info(
            "Cleaned up old jobs",
            cleaned_count=result["cleaned_jobs"],
            older_than_days=older_than_days,
            user_id=current_user.sub
        )

        return result

    except Exception as e:
        logger.error(
            "Failed to cleanup jobs",
            older_than_days=older_than_days,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cleanup old jobs"
        )


# Predefined job scheduling endpoints for common operations
@router.post("/evidence-processing/{evidence_id}")
async def schedule_evidence_processing(
    request: Request,
    evidence_id: UUID,
    priority: JobPriority = JobPriority.NORMAL,
    current_user: UserClaims = Depends(require_permissions(Permission.EVIDENCE_WRITE))
):
    """Schedule evidence processing job."""
    tenant_id = require_tenant_context(request)

    job_request = JobScheduleRequest(
        job_type=JobType.EVIDENCE_PROCESSING,
        payload={"evidence_id": str(evidence_id)},
        priority=priority,
        tenant_id=tenant_id,
        user_id=current_user.sub,
        tags=["evidence", "processing"]
    )

    return await schedule_job(request, job_request, current_user)


@router.post("/malware-scan/{evidence_id}")
async def schedule_malware_scan(
    request: Request,
    evidence_id: UUID,
    priority: JobPriority = JobPriority.HIGH,
    current_user: UserClaims = Depends(require_permissions(Permission.EVIDENCE_WRITE))
):
    """Schedule malware scanning job."""
    tenant_id = require_tenant_context(request)

    job_request = JobScheduleRequest(
        job_type=JobType.MALWARE_SCAN,
        payload={"evidence_id": str(evidence_id)},
        priority=priority,
        tenant_id=tenant_id,
        user_id=current_user.sub,
        tags=["malware", "security"]
    )

    return await schedule_job(request, job_request, current_user)
