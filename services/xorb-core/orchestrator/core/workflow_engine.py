from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TaskType(Enum):
    VULNERABILITY_SCAN = "vulnerability_scan"
    COMPLIANCE_CHECK = "compliance_check"
    THREAT_ANALYSIS = "threat_analysis"
    REPORT_GENERATION = "report_generation"
    NOTIFICATION = "notification"
    DATA_COLLECTION = "data_collection"
    REMEDIATION = "remediation"
    APPROVAL = "approval"
    INTEGRATION = "integration"

class TriggerType(Enum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    API_TRIGGER = "api_trigger"
    WEBHOOK = "webhook"

@dataclass
class WorkflowTask:
    id: str
    name: str
    task_type: TaskType
    description: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    timeout_minutes: int
    retry_count: int
    retry_delay_seconds: int
    condition: Optional[str] = None
    on_success: Optional[List[str]] = None
    on_failure: Optional[List[str]] = None
    parallel_execution: bool = False

@dataclass
class WorkflowDefinition:
    id: str
    name: str
    description: str
    version: str
    tasks: List[WorkflowTask]
    triggers: List[Dict[str, Any]]
    variables: Dict[str, Any]
    notifications: Dict[str, List[str]]
    sla_minutes: Optional[int] = None
    tags: List[str] = None
    enabled: bool = True

@dataclass
class WorkflowExecution:
    id: str
    workflow_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime]
    triggered_by: str
    trigger_data: Dict[str, Any]
    task_results: Dict[str, Any]
    error_message: Optional[str] = None
    variables: Dict[str, Any] = None

class TaskExecutor(ABC):
    """Abstract base class for task executors"""

    @abstractmethod
    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow task"""
        pass

    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate task parameters"""
        pass

class WorkflowOrchestrator(ABC):
    """Main workflow orchestration interface"""

    @abstractmethod
    async def initialize(self):
        """Initialize the orchestrator"""
        pass

    @abstractmethod
    async def create_workflow(self, workflow_def: WorkflowDefinition) -> str:
        """Create a new workflow definition"""
        pass

    @abstractmethod
    async def execute_workflow(self, workflow_id: str, trigger_data: Dict[str, Any] = None,
                             triggered_by: str = "manual") -> str:
        """Execute a workflow"""
        pass

    @abstractmethod
    async def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution status"""
        pass

    @abstractmethod
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel workflow execution"""
        pass

    @abstractmethod
    async def shutdown(self):
        """Shutdown the orchestrator"""
        pass

# Factory function
def create_workflow_orchestrator(config: Dict[str, Any]) -> WorkflowOrchestrator:
    """Create and configure workflow orchestrator"""
    from .advanced_workflow_orchestrator import AdvancedWorkflowOrchestrator
    return AdvancedWorkflowOrchestrator(config)

# Event types for pub/sub
class WorkflowEventType(Enum):
    WORKFLOW_CREATED = "workflow.created"
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_CANCELLED = "workflow.cancelled"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_RETRYING = "task.retrying"

# Event data classes
@dataclass
class WorkflowEvent:
    event_type: WorkflowEventType
    timestamp: datetime
    data: Dict[str, Any]
