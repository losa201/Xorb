"""Tests for job orchestration system."""
import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.jobs.models import (
    JobDefinition, JobExecution, JobScheduleRequest, JobType,
    JobPriority, JobStatus, RetryPolicy, JobResult
)
from app.jobs.queue import JobQueue
from app.jobs.worker import JobWorker
from app.jobs.service import JobService


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_mock = Mock()

    # Mock basic Redis operations
    redis_mock.hset = AsyncMock(return_value=1)
    redis_mock.hget = AsyncMock(return_value=None)
    redis_mock.lpush = AsyncMock(return_value=1)
    redis_mock.brpop = AsyncMock(return_value=None)
    redis_mock.lrem = AsyncMock(return_value=0)
    redis_mock.llen = AsyncMock(return_value=0)
    redis_mock.lrange = AsyncMock(return_value=[])
    redis_mock.setex = AsyncMock(return_value=True)
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.zadd = AsyncMock(return_value=1)
    redis_mock.zrem = AsyncMock(return_value=1)
    redis_mock.zrangebyscore = AsyncMock(return_value=[])
    redis_mock.exists = AsyncMock(return_value=0)
    redis_mock.keys = AsyncMock(return_value=[])
    redis_mock.hincrby = AsyncMock(return_value=1)

    return redis_mock


@pytest.fixture
def job_queue(mock_redis):
    """Job queue instance for testing."""
    return JobQueue(mock_redis)


@pytest.fixture
def job_service(mock_redis):
    """Job service instance for testing."""
    return JobService(mock_redis)


@pytest.fixture
def sample_job_definition():
    """Sample job definition for testing."""
    return JobDefinition(
        id=uuid4(),
        job_type=JobType.EVIDENCE_PROCESSING,
        priority=JobPriority.NORMAL,
        payload={"evidence_id": str(uuid4())},
        queue_name="default",
        tenant_id=uuid4(),
        user_id="user123"
    )


@pytest.fixture
def sample_job_request():
    """Sample job schedule request."""
    return JobScheduleRequest(
        job_type=JobType.MALWARE_SCAN,
        payload={"file_hash": "abc123"},
        priority=JobPriority.HIGH,
        tenant_id=uuid4(),
        user_id="user123",
        tags=["security", "scan"]
    )


class TestJobQueue:
    """Test job queue functionality."""

    @pytest.mark.asyncio
    async def test_enqueue_job_success(self, job_queue, sample_job_definition):
        """Test successful job enqueuing."""
        execution = await job_queue.enqueue_job(sample_job_definition)

        assert execution.job_id == sample_job_definition.id
        assert execution.status == JobStatus.PENDING
        assert execution.created_at is not None

        # Verify Redis operations were called
        job_queue.redis.hset.assert_called()
        job_queue.redis.lpush.assert_called()

    @pytest.mark.asyncio
    async def test_enqueue_job_with_idempotency(self, job_queue, sample_job_definition, mock_redis):
        """Test job enqueuing with idempotency key."""
        idempotency_key = "unique_key_123"
        sample_job_definition.idempotency_key = idempotency_key

        # First enqueue should succeed
        execution1 = await job_queue.enqueue_job(sample_job_definition)
        assert execution1.job_id == sample_job_definition.id

        # Mock idempotency key exists
        mock_redis.get.return_value = str(sample_job_definition.id).encode()

        # Mock existing execution
        mock_redis.hget.return_value = execution1.model_dump_json().encode()

        # Second enqueue should return existing execution
        execution2 = await job_queue.enqueue_job(sample_job_definition)
        assert execution2.job_id == execution1.job_id

    @pytest.mark.asyncio
    async def test_dequeue_job_success(self, job_queue, sample_job_definition, mock_redis):
        """Test successful job dequeuing."""
        # Mock queue pop result
        job_id_bytes = str(sample_job_definition.id).encode()
        mock_redis.brpop.return_value = ("queue_name", job_id_bytes)

        # Mock job data
        mock_redis.hget.return_value = sample_job_definition.model_dump_json().encode()

        result = await job_queue.dequeue_job(["default"], "worker-1")

        assert result is not None
        job_def, execution = result
        assert job_def.id == sample_job_definition.id
        assert execution.status == JobStatus.RUNNING
        assert execution.worker_id == "worker-1"

    @pytest.mark.asyncio
    async def test_dequeue_job_timeout(self, job_queue, mock_redis):
        """Test job dequeue timeout."""
        # Mock timeout (no job available)
        mock_redis.brpop.return_value = None

        result = await job_queue.dequeue_job(["default"], "worker-1", timeout=1)
        assert result is None

    @pytest.mark.asyncio
    async def test_complete_job_success(self, job_queue, mock_redis):
        """Test successful job completion."""
        job_id = uuid4()
        execution_id = uuid4()

        # Mock execution data
        execution = JobExecution(
            id=execution_id,
            job_id=job_id,
            status=JobStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        mock_redis.hget.return_value = execution.model_dump_json().encode()

        await job_queue.complete_job(
            job_id=job_id,
            execution_id=execution_id,
            success=True,
            result={"status": "completed"}
        )

        # Verify completion operations
        mock_redis.hset.assert_called()  # Update execution
        mock_redis.lrem.assert_called()  # Remove from running
        mock_redis.lpush.assert_called()  # Add to completed

    @pytest.mark.asyncio
    async def test_complete_job_with_retry(self, job_queue, sample_job_definition, mock_redis):
        """Test job completion that triggers retry."""
        job_id = sample_job_definition.id
        execution_id = uuid4()

        # Mock execution data (first attempt)
        execution = JobExecution(
            id=execution_id,
            job_id=job_id,
            status=JobStatus.RUNNING,
            attempt=1,
            started_at=datetime.utcnow()
        )
        mock_redis.hget.side_effect = [
            execution.model_dump_json().encode(),  # First call for execution
            sample_job_definition.model_dump_json().encode()  # Second call for job def
        ]

        await job_queue.complete_job(
            job_id=job_id,
            execution_id=execution_id,
            success=False,
            error="Processing failed"
        )

        # Should schedule retry (zadd for scheduled queue)
        mock_redis.zadd.assert_called()

    @pytest.mark.asyncio
    async def test_cancel_job_pending(self, job_queue, mock_redis):
        """Test cancelling a pending job."""
        job_id = uuid4()

        # Mock successful removal from pending queue
        mock_redis.lrem.return_value = 1

        result = await job_queue.cancel_job(job_id)
        assert result is True

        mock_redis.lrem.assert_called()

    @pytest.mark.asyncio
    async def test_cancel_job_running(self, job_queue, mock_redis):
        """Test cancelling a running job."""
        job_id = uuid4()

        # Mock removal from running queue
        mock_redis.lrem.side_effect = [0, 1]  # Not in pending, but in running

        result = await job_queue.cancel_job(job_id)
        assert result is True

        # Should set cancellation flag
        mock_redis.setex.assert_called()


class TestJobWorker:
    """Test job worker functionality."""

    @pytest.fixture
    def job_worker(self, mock_redis):
        """Job worker instance for testing."""
        return JobWorker(mock_redis, worker_id="test-worker", queues=["default"])

    def test_register_handler(self, job_worker):
        """Test handler registration."""
        def test_handler(job_def):
            return JobResult(success=True)

        job_worker.register_handler(JobType.EVIDENCE_PROCESSING, test_handler)

        assert JobType.EVIDENCE_PROCESSING in job_worker.handlers
        assert job_worker.handlers[JobType.EVIDENCE_PROCESSING] == test_handler

    @pytest.mark.asyncio
    async def test_process_job_success(self, job_worker, sample_job_definition):
        """Test successful job processing."""
        # Register handler
        async def test_handler(job_def):
            return JobResult(success=True, result={"processed": True})

        job_worker.register_handler(JobType.EVIDENCE_PROCESSING, test_handler)

        # Mock job execution
        execution = JobExecution(
            job_id=sample_job_definition.id,
            status=JobStatus.RUNNING
        )

        # Mock queue methods
        job_worker.queue.complete_job = AsyncMock()

        with patch.object(job_worker, '_is_job_cancelled', return_value=False):
            await job_worker._process_job(sample_job_definition, execution)

        # Verify job was completed successfully
        job_worker.queue.complete_job.assert_called_once()
        args = job_worker.queue.complete_job.call_args[1]
        assert args['success'] is True
        assert args['result'] == {"processed": True}

    @pytest.mark.asyncio
    async def test_process_job_failure(self, job_worker, sample_job_definition):
        """Test job processing failure."""
        # Register handler that fails
        async def failing_handler(job_def):
            raise ValueError("Processing failed")

        job_worker.register_handler(JobType.EVIDENCE_PROCESSING, failing_handler)

        execution = JobExecution(
            job_id=sample_job_definition.id,
            status=JobStatus.RUNNING
        )

        job_worker.queue.complete_job = AsyncMock()

        with patch.object(job_worker, '_is_job_cancelled', return_value=False):
            await job_worker._process_job(sample_job_definition, execution)

        # Verify job was marked as failed
        job_worker.queue.complete_job.assert_called_once()
        args = job_worker.queue.complete_job.call_args[1]
        assert args['success'] is False
        assert "Processing failed" in args['error']

    @pytest.mark.asyncio
    async def test_process_job_timeout(self, job_worker, sample_job_definition):
        """Test job processing timeout."""
        # Register handler that takes too long
        async def slow_handler(job_def):
            await asyncio.sleep(10)  # Longer than timeout
            return JobResult(success=True)

        job_worker.register_handler(JobType.EVIDENCE_PROCESSING, slow_handler)

        # Set short timeout
        sample_job_definition.execution_timeout = 1

        execution = JobExecution(
            job_id=sample_job_definition.id,
            status=JobStatus.RUNNING
        )

        job_worker.queue.complete_job = AsyncMock()

        with patch.object(job_worker, '_is_job_cancelled', return_value=False):
            await job_worker._process_job(sample_job_definition, execution)

        # Verify job was marked as failed due to timeout
        job_worker.queue.complete_job.assert_called_once()
        args = job_worker.queue.complete_job.call_args[1]
        assert args['success'] is False
        assert "timed out" in args['error']

    @pytest.mark.asyncio
    async def test_process_job_cancelled(self, job_worker, sample_job_definition):
        """Test processing of cancelled job."""
        execution = JobExecution(
            job_id=sample_job_definition.id,
            status=JobStatus.RUNNING
        )

        job_worker.queue.complete_job = AsyncMock()

        # Mock job as cancelled
        with patch.object(job_worker, '_is_job_cancelled', return_value=True):
            await job_worker._process_job(sample_job_definition, execution)

        # Should not complete the job
        job_worker.queue.complete_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_heartbeat(self, job_worker, mock_redis):
        """Test worker heartbeat."""
        await job_worker._send_heartbeat()

        # Should store worker data
        mock_redis.setex.assert_called()

        # Verify worker data structure
        call_args = mock_redis.setex.call_args[0]
        key, ttl, data = call_args
        assert job_worker.worker_id in key
        assert ttl > 0


class TestJobService:
    """Test job service functionality."""

    @pytest.mark.asyncio
    async def test_schedule_job(self, job_service, sample_job_request):
        """Test job scheduling."""
        # Mock queue enqueue
        mock_execution = JobExecution(
            job_id=uuid4(),
            status=JobStatus.PENDING,
            created_at=datetime.utcnow()
        )

        with patch.object(job_service.queue, 'enqueue_job', return_value=mock_execution) as mock_enqueue:
            result = await job_service.schedule_job(sample_job_request)

            assert "job_id" in result
            assert result["status"] == JobStatus.PENDING.value
            assert result["priority"] == sample_job_request.priority.value

            mock_enqueue.assert_called_once()

    @pytest.mark.asyncio
    async def test_schedule_bulk_jobs(self, job_service):
        """Test bulk job scheduling."""
        from app.jobs.models import BulkJobRequest

        job_requests = [
            JobScheduleRequest(
                job_type=JobType.MALWARE_SCAN,
                payload={"file": f"file_{i}"}
            )
            for i in range(3)
        ]

        bulk_request = BulkJobRequest(
            jobs=job_requests,
            batch_id="batch_123"
        )

        # Mock successful scheduling
        with patch.object(job_service, 'schedule_job') as mock_schedule:
            mock_schedule.return_value = {
                "job_id": str(uuid4()),
                "status": "pending"
            }

            result = await job_service.schedule_bulk_jobs(bulk_request)

            assert result["total_jobs"] == 3
            assert result["scheduled"] == 3
            assert result["failed"] == 0
            assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_cancel_jobs(self, job_service):
        """Test job cancellation."""
        from app.jobs.models import JobCancelRequest

        job_ids = [uuid4(), uuid4()]
        cancel_request = JobCancelRequest(
            job_ids=job_ids,
            reason="User requested"
        )

        # Mock queue cancel
        with patch.object(job_service.queue, 'cancel_job', return_value=True) as mock_cancel:
            result = await job_service.cancel_jobs(cancel_request)

            assert result["total_jobs"] == 2
            assert len(result["results"]) == 2
            assert all(r["cancelled"] for r in result["results"])

            assert mock_cancel.call_count == 2

    @pytest.mark.asyncio
    async def test_get_job_status(self, job_service, mock_redis):
        """Test getting job status."""
        job_id = uuid4()

        # Mock execution data
        execution = JobExecution(
            id=uuid4(),
            job_id=job_id,
            status=JobStatus.COMPLETED,
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            result={"success": True}
        )

        # Mock job definition
        job_def = JobDefinition(
            id=job_id,
            job_type=JobType.EVIDENCE_PROCESSING,
            priority=JobPriority.NORMAL
        )

        mock_redis.hget.side_effect = [
            execution.model_dump_json().encode(),  # execution data
            job_def.model_dump_json().encode()      # job data
        ]

        with patch.object(job_service.queue, 'get_job_status', return_value=execution):
            result = await job_service.get_job_status(job_id)

            assert result is not None
            assert result["job_id"] == str(job_id)
            assert result["status"] == JobStatus.COMPLETED.value
            assert result["result"] == {"success": True}

    @pytest.mark.asyncio
    async def test_retry_failed_jobs(self, job_service, mock_redis):
        """Test retrying failed jobs."""
        job_id = uuid4()

        # Mock failed execution
        execution = JobExecution(
            id=uuid4(),
            job_id=job_id,
            status=JobStatus.FAILED,
            completed_at=datetime.utcnow()
        )

        # Mock job definition
        job_def = JobDefinition(
            id=job_id,
            job_type=JobType.EVIDENCE_PROCESSING
        )

        mock_redis.hget.side_effect = [
            execution.model_dump_json().encode(),
            job_def.model_dump_json().encode()
        ]

        with patch.object(job_service.queue, 'get_job_status', return_value=execution), \
             patch.object(job_service.queue, 'enqueue_job') as mock_enqueue:

            result = await job_service.retry_failed_jobs(job_ids=[job_id])

            assert result["retried"] == 1
            mock_enqueue.assert_called_once()


class TestJobIntegration:
    """Integration tests for job orchestration."""

    @pytest.mark.asyncio
    async def test_end_to_end_job_processing(self):
        """Test complete job lifecycle."""
        # This would be a full integration test:
        # 1. Schedule job
        # 2. Worker picks up job
        # 3. Job executes successfully
        # 4. Status updates correctly
        # 5. Metrics are updated
        pass

    @pytest.mark.asyncio
    async def test_job_retry_mechanism(self):
        """Test job retry with exponential backoff."""
        # This would test:
        # 1. Job fails
        # 2. Gets scheduled for retry
        # 3. Retry delay calculation
        # 4. Eventually succeeds or goes to DLQ
        pass

    @pytest.mark.asyncio
    async def test_worker_failure_recovery(self):
        """Test worker failure and job recovery."""
        # This would test:
        # 1. Worker picks up job
        # 2. Worker crashes/dies
        # 3. Job timeout detection
        # 4. Job gets requeued
        pass

    @pytest.mark.asyncio
    async def test_idempotency_enforcement(self):
        """Test that idempotency keys prevent duplicate processing."""
        # This would test:
        # 1. Submit job with idempotency key
        # 2. Submit same job again
        # 3. Verify only one execution
        pass

    @pytest.mark.asyncio
    async def test_priority_queue_ordering(self):
        """Test that high priority jobs are processed first."""
        # This would test:
        # 1. Submit jobs with different priorities
        # 2. Verify processing order
        pass

    @pytest.mark.asyncio
    async def test_tenant_isolation_in_jobs(self):
        """Test that jobs respect tenant isolation."""
        # This would test:
        # 1. Submit jobs for different tenants
        # 2. Verify tenant context in processing
        # 3. Verify isolation of results
        pass
