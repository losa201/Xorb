"""
XORB Task Orchestration API Endpoints
Provides intelligent task coordination and workflow management
"""
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ..security import (
    SecurityContext,
    get_security_context,
    require_orchestrator,
    require_permission,
    Permission
)


class TaskType(str, Enum):
    """Types of orchestrated tasks"""
    VULNERABILITY_SCAN = "vulnerability_scan"
    THREAT_HUNT = "threat_hunt"
    SECURITY_ASSESSMENT = "security_assessment"
    COMPLIANCE_CHECK = "compliance_check"
    INCIDENT_RESPONSE = "incident_response"
    FORENSIC_ANALYSIS = "forensic_analysis"
    NETWORK_MONITORING = "network_monitoring"
    LOG_ANALYSIS = "log_analysis"
    INTELLIGENCE_GATHERING = "intelligence_gathering"
    CUSTOM_WORKFLOW = "custom_workflow"


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskPriority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class OrchestrationStrategy(str, Enum):
    """Task orchestration strategies"""
    FIFO = "fifo"  # First In, First Out
    PRIORITY = "priority"  # Priority-based scheduling
    LOAD_BALANCED = "load_balanced"  # Load-aware distribution
    CAPABILITY_MATCH = "capability_match"  # Best capability matching
    AI_OPTIMIZED = "ai_optimized"  # AI-driven optimization


# Pydantic Models
class TaskParameters(BaseModel):
    """Task execution parameters"""
    target: Optional[str] = None
    scope: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    timeout_minutes: Optional[int] = Field(None, ge=1, le=1440)


class TaskResult(BaseModel):
    """Task execution result"""
    status: str
    data: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)
    metrics: Dict[str, Union[int, float, str]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class CreateTaskRequest(BaseModel):
    """Request to create a new task"""
    name: str = Field(..., min_length=1, max_length=200)
    task_type: TaskType
    priority: TaskPriority = TaskPriority.MEDIUM
    parameters: TaskParameters
    description: Optional[str] = Field(None, max_length=1000)
    preferred_agent_id: Optional[str] = None
    orchestration_strategy: OrchestrationStrategy = OrchestrationStrategy.AI_OPTIMIZED
    dependencies: List[str] = Field(default_factory=list)
    schedule_at: Optional[datetime] = None
    tags: Dict[str, str] = Field(default_factory=dict)


class UpdateTaskRequest(BaseModel):
    """Request to update task"""
    priority: Optional[TaskPriority] = None
    parameters: Optional[TaskParameters] = None
    description: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


class Task(BaseModel):
    """Task information"""
    id: str
    name: str
    task_type: TaskType
    status: TaskStatus
    priority: TaskPriority
    parameters: TaskParameters
    description: Optional[str] = None

    # Assignment and execution
    assigned_agent_id: Optional[str] = None
    orchestration_strategy: OrchestrationStrategy
    dependencies: List[str] = Field(default_factory=list)

    # Timing
    created_at: datetime
    updated_at: datetime
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results and tracking
    result: Optional[TaskResult] = None
    progress_percentage: float = 0.0
    execution_log: List[Dict[str, Any]] = Field(default_factory=list)

    # Metadata
    created_by: str
    tags: Dict[str, str] = Field(default_factory=dict)
    parent_task_id: Optional[str] = None
    subtasks: List[str] = Field(default_factory=list)


class TasksListResponse(BaseModel):
    """Response for listing tasks"""
    tasks: List[Task]
    total: int
    page: int
    per_page: int
    has_next: bool


class OrchestrationMetrics(BaseModel):
    """Orchestration system metrics"""
    total_tasks: int
    pending_tasks: int
    running_tasks: int
    completed_tasks: int
    failed_tasks: int
    average_execution_time_minutes: float
    queue_depth: int
    active_agents: int
    system_load: float


class IntelligentSchedulingRequest(BaseModel):
    """Request for AI-optimized task scheduling"""
    tasks: List[str]  # Task IDs to schedule
    optimization_criteria: List[str] = Field(default_factory=lambda: ["efficiency", "priority", "resource_usage"])
    constraints: Dict[str, Any] = Field(default_factory=dict)


class SchedulingRecommendation(BaseModel):
    """AI scheduling recommendation"""
    task_id: str
    recommended_agent_id: str
    estimated_start_time: datetime
    estimated_duration_minutes: int
    confidence_score: float
    reasoning: List[str]


class OrchestrationDecision(BaseModel):
    """Orchestration decision from AI brain"""
    strategy: OrchestrationStrategy
    agent_assignments: Dict[str, str]  # task_id -> agent_id
    priority_adjustments: Dict[str, TaskPriority]
    schedule_recommendations: List[SchedulingRecommendation]
    confidence: float
    reasoning: List[str]


# In-memory task storage (replace with database in production)
tasks_store: Dict[str, Task] = {}
orchestration_queue: List[str] = []  # Task IDs in processing order


router = APIRouter(prefix="/orchestration", tags=["Task Orchestration"])


@router.get("/tasks", response_model=TasksListResponse)
async def list_tasks(
    context: SecurityContext = Depends(get_security_context),
    status: Optional[TaskStatus] = Query(None, description="Filter by status"),
    task_type: Optional[TaskType] = Query(None, description="Filter by task type"),
    priority: Optional[TaskPriority] = Query(None, description="Filter by priority"),
    assigned_agent_id: Optional[str] = Query(None, description="Filter by assigned agent"),
    created_by: Optional[str] = Query(None, description="Filter by creator"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(100, ge=1, le=1000, description="Items per page"),
) -> TasksListResponse:
    """List tasks with filtering and pagination"""

    if Permission.TASK_READ not in context.permissions:
        raise HTTPException(status_code=403, detail="Permission required: task:read")

    # Filter tasks
    filtered_tasks = []
    for task in tasks_store.values():
        # Apply filters
        if status and task.status != status:
            continue
        if task_type and task.task_type != task_type:
            continue
        if priority and task.priority != priority:
            continue
        if assigned_agent_id and task.assigned_agent_id != assigned_agent_id:
            continue
        if created_by and task.created_by != created_by:
            continue

        filtered_tasks.append(task)

    # Sort by priority and creation time
    filtered_tasks.sort(key=lambda t: (
        _priority_value(t.priority),
        t.created_at
    ), reverse=True)

    # Apply pagination
    total = len(filtered_tasks)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    page_tasks = filtered_tasks[start_idx:end_idx]

    return TasksListResponse(
        tasks=page_tasks,
        total=total,
        page=page,
        per_page=per_page,
        has_next=end_idx < total
    )


@router.post("/tasks", response_model=Task, status_code=201)
async def create_task(
    request: CreateTaskRequest,
    background_tasks: BackgroundTasks,
    context: SecurityContext = Depends(require_orchestrator)
) -> Task:
    """Create and orchestrate a new task"""

    task_id = str(uuid.uuid4())
    current_time = datetime.utcnow()

    # Create task
    task = Task(
        id=task_id,
        name=request.name,
        task_type=request.task_type,
        status=TaskStatus.PENDING,
        priority=request.priority,
        parameters=request.parameters,
        description=request.description,
        orchestration_strategy=request.orchestration_strategy,
        dependencies=request.dependencies,
        created_at=current_time,
        updated_at=current_time,
        scheduled_at=request.schedule_at or current_time,
        created_by=context.user_id,
        tags=request.tags
    )

    # Store task
    tasks_store[task_id] = task

    # Add to orchestration queue
    orchestration_queue.append(task_id)

    # Log task creation
    _add_execution_log(task, "Task created", {"created_by": context.user_id})

    # Start orchestration process
    background_tasks.add_task(_orchestrate_task, task_id)

    return task


@router.get("/tasks/{task_id}", response_model=Task)
async def get_task(
    task_id: str,
    context: SecurityContext = Depends(get_security_context)
) -> Task:
    """Get detailed task information"""

    if Permission.TASK_READ not in context.permissions:
        raise HTTPException(status_code=403, detail="Permission required: task:read")

    task = tasks_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return task


@router.put("/tasks/{task_id}", response_model=Task)
async def update_task(
    task_id: str,
    request: UpdateTaskRequest,
    context: SecurityContext = Depends(require_permission(Permission.TASK_PRIORITY))
) -> Task:
    """Update task configuration"""

    task = tasks_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Check if task can be updated
    if task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]:
        raise HTTPException(status_code=409, detail="Cannot update completed or cancelled task")

    # Update fields
    updated = False
    if request.priority and request.priority != task.priority:
        old_priority = task.priority
        task.priority = request.priority
        _add_execution_log(task, "Priority updated", {
            "old_priority": old_priority.value,
            "new_priority": request.priority.value,
            "updated_by": context.user_id
        })
        updated = True

    if request.parameters:
        task.parameters = request.parameters
        _add_execution_log(task, "Parameters updated", {"updated_by": context.user_id})
        updated = True

    if request.description is not None:
        task.description = request.description
        updated = True

    if request.tags is not None:
        task.tags.update(request.tags)
        updated = True

    if updated:
        task.updated_at = datetime.utcnow()

    return task


@router.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    context: SecurityContext = Depends(require_permission(Permission.TASK_CANCEL))
) -> Dict[str, str]:
    """Cancel a pending or running task"""

    task = tasks_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Check if task can be cancelled
    if task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]:
        raise HTTPException(status_code=409, detail="Task already completed or cancelled")

    # Cancel task
    task.status = TaskStatus.CANCELLED
    task.updated_at = datetime.utcnow()

    _add_execution_log(task, "Task cancelled", {"cancelled_by": context.user_id})

    # Remove from queue if pending
    if task_id in orchestration_queue:
        orchestration_queue.remove(task_id)

    return {"message": "Task cancelled successfully", "task_id": task_id}


@router.get("/metrics", response_model=OrchestrationMetrics)
async def get_orchestration_metrics(
    context: SecurityContext = Depends(get_security_context)
) -> OrchestrationMetrics:
    """Get orchestration system metrics"""

    if Permission.TELEMETRY_READ not in context.permissions:
        raise HTTPException(status_code=403, detail="Permission required: telemetry:read")

    # Calculate metrics
    total_tasks = len(tasks_store)
    pending_tasks = sum(1 for t in tasks_store.values() if t.status == TaskStatus.PENDING)
    running_tasks = sum(1 for t in tasks_store.values() if t.status == TaskStatus.RUNNING)
    completed_tasks = sum(1 for t in tasks_store.values() if t.status == TaskStatus.COMPLETED)
    failed_tasks = sum(1 for t in tasks_store.values() if t.status == TaskStatus.FAILED)

    # Calculate average execution time
    completed_with_times = [
        t for t in tasks_store.values()
        if t.status == TaskStatus.COMPLETED and t.started_at and t.completed_at
    ]

    if completed_with_times:
        total_execution_time = sum(
            (t.completed_at - t.started_at).total_seconds() / 60
            for t in completed_with_times
        )
        avg_execution_time = total_execution_time / len(completed_with_times)
    else:
        avg_execution_time = 0.0

    # Simulate other metrics
    queue_depth = len(orchestration_queue)
    active_agents = 3  # This would come from agent registry
    system_load = min(queue_depth / 10, 1.0)  # Simplified load calculation

    return OrchestrationMetrics(
        total_tasks=total_tasks,
        pending_tasks=pending_tasks,
        running_tasks=running_tasks,
        completed_tasks=completed_tasks,
        failed_tasks=failed_tasks,
        average_execution_time_minutes=avg_execution_time,
        queue_depth=queue_depth,
        active_agents=active_agents,
        system_load=system_load
    )


@router.post("/optimize", response_model=OrchestrationDecision)
async def optimize_orchestration(
    request: IntelligentSchedulingRequest,
    context: SecurityContext = Depends(require_orchestrator)
) -> OrchestrationDecision:
    """Get AI-optimized orchestration decisions"""

    # Simulate AI brain decision making
    await asyncio.sleep(1)  # Simulate processing time

    # Get tasks to optimize
    tasks_to_optimize = [tasks_store[tid] for tid in request.tasks if tid in tasks_store]

    if not tasks_to_optimize:
        raise HTTPException(status_code=400, detail="No valid tasks provided for optimization")

    # Simulate AI decision
    decision = OrchestrationDecision(
        strategy=OrchestrationStrategy.AI_OPTIMIZED,
        agent_assignments={},
        priority_adjustments={},
        schedule_recommendations=[],
        confidence=0.85,
        reasoning=[
            "Analyzed task dependencies and agent capabilities",
            "Optimized for resource utilization and priority",
            "Considered historical performance data"
        ]
    )

    # Generate recommendations for each task
    for i, task in enumerate(tasks_to_optimize):
        # Simulate agent assignment logic
        agent_id = f"agent_{i % 3 + 1}"  # Round-robin for demo
        decision.agent_assignments[task.id] = agent_id

        # Generate scheduling recommendation
        start_time = datetime.utcnow() + timedelta(minutes=i * 5)
        duration = _estimate_task_duration(task)

        recommendation = SchedulingRecommendation(
            task_id=task.id,
            recommended_agent_id=agent_id,
            estimated_start_time=start_time,
            estimated_duration_minutes=duration,
            confidence_score=0.8 + (i * 0.05),
            reasoning=[
                f"Agent {agent_id} has optimal capabilities for {task.task_type.value}",
                f"Estimated completion in {duration} minutes",
                "No resource conflicts detected"
            ]
        )

        decision.schedule_recommendations.append(recommendation)

    return decision


@router.post("/tasks/{task_id}/pause")
async def pause_task(
    task_id: str,
    context: SecurityContext = Depends(require_orchestrator)
) -> Dict[str, str]:
    """Pause a running task"""

    task = tasks_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != TaskStatus.RUNNING:
        raise HTTPException(status_code=409, detail="Task is not running")

    task.status = TaskStatus.PAUSED
    task.updated_at = datetime.utcnow()

    _add_execution_log(task, "Task paused", {"paused_by": context.user_id})

    return {"message": "Task paused successfully", "task_id": task_id}


@router.post("/tasks/{task_id}/resume")
async def resume_task(
    task_id: str,
    background_tasks: BackgroundTasks,
    context: SecurityContext = Depends(require_orchestrator)
) -> Dict[str, str]:
    """Resume a paused task"""

    task = tasks_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != TaskStatus.PAUSED:
        raise HTTPException(status_code=409, detail="Task is not paused")

    task.status = TaskStatus.RUNNING
    task.updated_at = datetime.utcnow()

    _add_execution_log(task, "Task resumed", {"resumed_by": context.user_id})

    # Continue task execution
    background_tasks.add_task(_continue_task_execution, task_id)

    return {"message": "Task resumed successfully", "task_id": task_id}


# Helper functions
def _priority_value(priority: TaskPriority) -> int:
    """Convert priority to numeric value for sorting"""
    priority_values = {
        TaskPriority.EMERGENCY: 5,
        TaskPriority.CRITICAL: 4,
        TaskPriority.HIGH: 3,
        TaskPriority.MEDIUM: 2,
        TaskPriority.LOW: 1
    }
    return priority_values.get(priority, 1)


def _add_execution_log(task: Task, event: str, details: Dict[str, Any] = None):
    """Add entry to task execution log"""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event,
        "details": details or {}
    }
    task.execution_log.append(log_entry)


def _estimate_task_duration(task: Task) -> int:
    """Estimate task duration in minutes"""
    base_durations = {
        TaskType.VULNERABILITY_SCAN: 30,
        TaskType.THREAT_HUNT: 60,
        TaskType.SECURITY_ASSESSMENT: 120,
        TaskType.COMPLIANCE_CHECK: 45,
        TaskType.INCIDENT_RESPONSE: 90,
        TaskType.FORENSIC_ANALYSIS: 180,
        TaskType.NETWORK_MONITORING: 15,
        TaskType.LOG_ANALYSIS: 25,
        TaskType.INTELLIGENCE_GATHERING: 40,
        TaskType.CUSTOM_WORKFLOW: 60
    }

    base = base_durations.get(task.task_type, 30)

    # Adjust for priority
    if task.priority in [TaskPriority.CRITICAL, TaskPriority.EMERGENCY]:
        base = int(base * 0.8)  # Faster processing for critical tasks
    elif task.priority == TaskPriority.LOW:
        base = int(base * 1.2)  # Slower processing for low priority

    return base


async def _orchestrate_task(task_id: str):
    """Orchestrate task execution"""
    task = tasks_store.get(task_id)
    if not task:
        return

    try:
        # Wait for dependencies
        await _wait_for_dependencies(task)

        # Find suitable agent
        agent_id = await _assign_agent(task)
        if not agent_id:
            task.status = TaskStatus.FAILED
            _add_execution_log(task, "No suitable agent available")
            return

        # Update task status
        task.status = TaskStatus.ASSIGNED
        task.assigned_agent_id = agent_id
        task.updated_at = datetime.utcnow()

        _add_execution_log(task, "Agent assigned", {"agent_id": agent_id})

        # Start execution
        await _execute_task(task)

    except Exception as e:
        task.status = TaskStatus.FAILED
        _add_execution_log(task, "Orchestration failed", {"error": str(e)})


async def _wait_for_dependencies(task: Task):
    """Wait for task dependencies to complete"""
    if not task.dependencies:
        return

    task.status = TaskStatus.QUEUED
    _add_execution_log(task, "Waiting for dependencies")

    while task.dependencies:
        # Check if dependencies are completed
        completed_deps = []
        for dep_id in task.dependencies:
            dep_task = tasks_store.get(dep_id)
            if dep_task and dep_task.status == TaskStatus.COMPLETED:
                completed_deps.append(dep_id)
            elif dep_task and dep_task.status == TaskStatus.FAILED:
                raise Exception(f"Dependency task {dep_id} failed")

        # Remove completed dependencies
        for dep_id in completed_deps:
            task.dependencies.remove(dep_id)

        if task.dependencies:
            await asyncio.sleep(5)  # Check every 5 seconds


async def _assign_agent(task: Task) -> Optional[str]:
    """Assign suitable agent to task"""
    # This would integrate with the agent management system
    # For now, simulate agent assignment

    if task.orchestration_strategy == OrchestrationStrategy.AI_OPTIMIZED:
        # Simulate AI-based agent selection
        await asyncio.sleep(0.5)
        return f"ai_selected_agent_{hash(task.id) % 3 + 1}"
    else:
        # Simple assignment for other strategies
        return f"agent_{hash(task.id) % 3 + 1}"


async def _execute_task(task: Task):
    """Execute task on assigned agent"""
    task.status = TaskStatus.RUNNING
    task.started_at = datetime.utcnow()
    task.updated_at = datetime.utcnow()

    _add_execution_log(task, "Task execution started")

    try:
        # Simulate task execution
        duration = _estimate_task_duration(task)

        # Update progress periodically
        for progress in [25, 50, 75, 100]:
            await asyncio.sleep(duration * 0.6 / 4)  # 60% of estimated time
            task.progress_percentage = progress
            task.updated_at = datetime.utcnow()

            _add_execution_log(task, f"Progress update: {progress}%")

        # Simulate task completion
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.utcnow()
        task.progress_percentage = 100.0

        # Generate mock results
        task.result = TaskResult(
            status="success",
            data={
                "task_type": task.task_type.value,
                "execution_time_minutes": duration,
                "findings": ["Sample finding 1", "Sample finding 2"],
                "confidence": 0.9
            },
            artifacts=[f"artifact_{task.id}.json", f"report_{task.id}.pdf"],
            metrics={
                "execution_time_seconds": duration * 60,
                "cpu_usage_percent": 25.0,
                "memory_usage_mb": 128.0
            }
        )

        _add_execution_log(task, "Task completed successfully")

        # Remove from orchestration queue
        if task.id in orchestration_queue:
            orchestration_queue.remove(task.id)

    except Exception as e:
        task.status = TaskStatus.FAILED
        task.completed_at = datetime.utcnow()

        task.result = TaskResult(
            status="failed",
            errors=[str(e)]
        )

        _add_execution_log(task, "Task execution failed", {"error": str(e)})

        # Remove from orchestration queue
        if task.id in orchestration_queue:
            orchestration_queue.remove(task.id)


async def _continue_task_execution(task_id: str):
    """Continue execution of resumed task"""
    task = tasks_store.get(task_id)
    if not task:
        return

    await _execute_task(task)
