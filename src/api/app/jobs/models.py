"""Job orchestration models and types."""
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"
    DEAD_LETTER = "dead_letter"


class JobPriority(str, Enum):
    """Job priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class JobType(str, Enum):
    """Predefined job types."""
    EVIDENCE_PROCESSING = "evidence_processing"
    MALWARE_SCAN = "malware_scan"
    THREAT_ANALYSIS = "threat_analysis"
    REPORT_GENERATION = "report_generation"
    DATA_EXPORT = "data_export"
    SYSTEM_MAINTENANCE = "system_maintenance"
    NOTIFICATION = "notification"
    CUSTOM = "custom"


class RetryPolicy(BaseModel):
    """Retry policy configuration."""
    max_attempts: int = 3
    initial_delay: int = 60  # seconds
    max_delay: int = 3600  # 1 hour
    exponential_base: float = 2.0
    jitter: bool = True


class JobDefinition(BaseModel):
    """Job definition and metadata."""
    id: UUID = Field(default_factory=uuid4)
    job_type: JobType
    priority: JobPriority = JobPriority.NORMAL
    
    # Execution parameters
    payload: Dict[str, Any] = Field(default_factory=dict)
    queue_name: str = "default"
    
    # Scheduling
    scheduled_at: Optional[datetime] = None
    delay_seconds: int = 0
    
    # Retry configuration
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    
    # Idempotency
    idempotency_key: Optional[str] = None
    
    # Tenant context
    tenant_id: Optional[UUID] = None
    user_id: Optional[str] = None
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    custom_metadata: Dict[str, str] = Field(default_factory=dict)
    
    # Timeouts
    execution_timeout: int = 300  # 5 minutes
    queue_timeout: int = 3600  # 1 hour


class JobExecution(BaseModel):
    """Job execution instance."""
    id: UUID = Field(default_factory=uuid4)
    job_id: UUID
    attempt: int = 1
    status: JobStatus = JobStatus.PENDING
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Worker information
    worker_id: Optional[str] = None
    worker_hostname: Optional[str] = None
    
    # Results
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    
    # Progress tracking
    progress: int = 0  # 0-100
    progress_message: Optional[str] = None
    
    # Metrics
    duration_seconds: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None


class JobResult(BaseModel):
    """Job execution result."""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Progress and metrics
    progress: int = 100
    progress_message: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class JobStatusUpdate(BaseModel):
    """Job status update event."""
    job_id: UUID
    execution_id: UUID
    status: JobStatus
    progress: Optional[int] = None
    progress_message: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QueueStats(BaseModel):
    """Queue statistics."""
    queue_name: str
    pending_jobs: int
    running_jobs: int
    completed_jobs_24h: int
    failed_jobs_24h: int
    avg_execution_time: Optional[float] = None
    oldest_pending_job: Optional[datetime] = None


class WorkerStats(BaseModel):
    """Worker statistics."""
    worker_id: str
    hostname: str
    status: str  # active, idle, dead
    last_heartbeat: datetime
    jobs_completed: int
    jobs_failed: int
    avg_execution_time: Optional[float] = None
    current_job: Optional[UUID] = None


class JobFilter(BaseModel):
    """Job filtering criteria."""
    job_types: Optional[List[JobType]] = None
    statuses: Optional[List[JobStatus]] = None
    priorities: Optional[List[JobPriority]] = None
    queue_names: Optional[List[str]] = None
    tenant_id: Optional[UUID] = None
    user_id: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    tags: Optional[List[str]] = None


class JobScheduleRequest(BaseModel):
    """Request to schedule a new job."""
    job_type: JobType
    payload: Dict[str, Any] = Field(default_factory=dict)
    priority: JobPriority = JobPriority.NORMAL
    queue_name: str = "default"
    
    # Scheduling options
    delay_seconds: int = 0
    scheduled_at: Optional[datetime] = None
    
    # Idempotency
    idempotency_key: Optional[str] = None
    
    # Context
    tenant_id: Optional[UUID] = None
    user_id: Optional[str] = None
    
    # Configuration overrides
    retry_policy: Optional[RetryPolicy] = None
    execution_timeout: Optional[int] = None
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    custom_metadata: Dict[str, str] = Field(default_factory=dict)


class BulkJobRequest(BaseModel):
    """Request to schedule multiple jobs."""
    jobs: List[JobScheduleRequest]
    batch_id: Optional[str] = None


class JobCancelRequest(BaseModel):
    """Request to cancel job(s)."""
    job_ids: List[UUID]
    reason: Optional[str] = None


# Job handler registry type
JobHandler = Callable[[JobDefinition], JobResult]