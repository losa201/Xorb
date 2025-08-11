"""
Unified XORB Orchestrator
Consolidates service orchestration, workflow management, and fusion orchestration
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import UUID, uuid4
from abc import ABC, abstractmethod

import aioredis
from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker

logger = logging.getLogger(__name__)


# Unified Enums
class ServiceStatus(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ServiceType(Enum):
    CORE = "core"
    ANALYTICS = "analytics"
    SECURITY = "security"
    INTELLIGENCE = "intelligence"
    INTEGRATION = "integration"
    MONITORING = "monitoring"
    FRONTEND = "frontend"


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
    SERVICE_FUSION = "service_fusion"


class TriggerType(Enum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    API_TRIGGER = "api_trigger"
    WEBHOOK = "webhook"


# Unified Data Models
@dataclass
class ServiceDefinition:
    """Definition of a service in the platform"""
    service_id: str
    name: str
    service_type: ServiceType
    module_path: str
    class_name: str
    dependencies: List[str]
    config: Dict[str, Any]
    health_check_url: Optional[str] = None
    startup_timeout: int = 30
    shutdown_timeout: int = 10
    restart_policy: str = "on-failure"
    max_restarts: int = 3
    resource_limits: Dict[str, Any] = None
    tags: List[str] = None


@dataclass
class ServiceInstance:
    """Runtime instance of a service"""
    service_id: str
    instance_id: str
    definition: ServiceDefinition
    status: ServiceStatus
    instance_object: Optional[Any] = None
    start_time: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    restart_count: int = 0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None


@dataclass
class WorkflowTask:
    """Task within a workflow"""
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
    """Definition of a workflow"""
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
    """Execution instance of a workflow"""
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


@dataclass
class OrchestrationMetrics:
    """Orchestration metrics and statistics"""
    total_services: int = 0
    running_services: int = 0
    failed_services: int = 0
    total_workflows: int = 0
    active_workflows: int = 0
    completed_workflows: int = 0
    failed_workflows: int = 0
    average_task_duration: float = 0.0
    system_load: float = 0.0
    last_updated: datetime = None


# Task Executor Interface
class TaskExecutor(ABC):
    """Abstract base class for task executors"""
    
    @abstractmethod
    async def execute(self, task: WorkflowTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow task"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if executor is healthy"""
        pass


# Unified Orchestrator
class UnifiedOrchestrator:
    """
    Unified XORB Orchestrator
    Consolidates service orchestration, workflow management, and fusion orchestration
    """
    
    def __init__(
        self,
        redis_client: aioredis.Redis,
        temporal_client: Optional[Client] = None
    ):
        self.redis = redis_client
        self.temporal_client = temporal_client
        
        # Service management
        self.services: Dict[str, ServiceInstance] = {}
        self.service_registry: Dict[str, ServiceDefinition] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        
        # Workflow management
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.workflow_executions: Dict[str, WorkflowExecution] = {}
        self.task_executors: Dict[TaskType, TaskExecutor] = {}
        
        # Orchestrator state
        self.running = False
        self.health_check_interval = 30
        self.health_check_task: Optional[asyncio.Task] = None
        self.workflow_monitor_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.metrics = OrchestrationMetrics()
        
        logger.info("Unified Orchestrator initialized")
    
    # Service Management Methods
    async def initialize(self):
        """Initialize the unified orchestrator"""
        logger.info("Initializing Unified Orchestrator...")
        
        try:
            # Initialize built-in services
            await self._initialize_built_in_services()
            
            # Build dependency graph
            self._build_dependency_graph()
            
            # Load workflows from storage
            await self._load_workflows()
            
            # Start monitoring tasks
            self.health_check_task = asyncio.create_task(self._health_check_monitor())
            self.workflow_monitor_task = asyncio.create_task(self._workflow_monitor())
            
            self.running = True
            logger.info(f"Unified Orchestrator initialized with {len(self.service_registry)} services and {len(self.workflows)} workflows")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the orchestrator"""
        logger.info("Shutting down Unified Orchestrator...")
        
        self.running = False
        
        # Cancel monitoring tasks
        if self.health_check_task:
            self.health_check_task.cancel()
        if self.workflow_monitor_task:
            self.workflow_monitor_task.cancel()
        
        # Stop all services
        await self._stop_all_services()
        
        # Close connections
        if self.temporal_client:
            await self.temporal_client.close()
        
        logger.info("Unified Orchestrator shutdown complete")
    
    def register_service(self, definition: ServiceDefinition):
        """Register a new service definition"""
        self.service_registry[definition.service_id] = definition
        logger.info(f"Registered service: {definition.name} ({definition.service_id})")
    
    async def start_service(self, service_id: str) -> bool:
        """Start a specific service"""
        if service_id not in self.service_registry:
            logger.error(f"Service not found: {service_id}")
            return False
        
        definition = self.service_registry[service_id]
        
        # Check dependencies
        if not await self._check_dependencies(definition.dependencies):
            logger.error(f"Dependencies not met for service: {service_id}")
            return False
        
        try:
            # Create service instance
            instance_id = str(uuid4())
            instance = ServiceInstance(
                service_id=service_id,
                instance_id=instance_id,
                definition=definition,
                status=ServiceStatus.INITIALIZING
            )
            
            # Load and instantiate service class
            module = __import__(definition.module_path, fromlist=[definition.class_name])
            service_class = getattr(module, definition.class_name)
            service_object = service_class(**definition.config)
            
            # Initialize service
            if hasattr(service_object, 'initialize'):
                await service_object.initialize()
            
            instance.instance_object = service_object
            instance.status = ServiceStatus.RUNNING
            instance.start_time = datetime.utcnow()
            
            self.services[service_id] = instance
            
            logger.info(f"Started service: {definition.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start service {service_id}: {e}")
            return False
    
    async def stop_service(self, service_id: str) -> bool:
        """Stop a specific service"""
        if service_id not in self.services:
            logger.warning(f"Service not running: {service_id}")
            return False
        
        instance = self.services[service_id]
        instance.status = ServiceStatus.STOPPING
        
        try:
            # Shutdown service
            if hasattr(instance.instance_object, 'shutdown'):
                await instance.instance_object.shutdown()
            
            instance.status = ServiceStatus.STOPPED
            del self.services[service_id]
            
            logger.info(f"Stopped service: {instance.definition.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop service {service_id}: {e}")
            instance.status = ServiceStatus.ERROR
            instance.error_message = str(e)
            return False
    
    # Workflow Management Methods
    def register_workflow(self, workflow_def: WorkflowDefinition):
        """Register a new workflow definition"""
        self.workflows[workflow_def.id] = workflow_def
        logger.info(f"Registered workflow: {workflow_def.name} ({workflow_def.id})")
    
    def register_task_executor(self, task_type: TaskType, executor: TaskExecutor):
        """Register a task executor for a specific task type"""
        self.task_executors[task_type] = executor
        logger.info(f"Registered task executor for: {task_type.value}")
    
    async def execute_workflow(
        self,
        workflow_id: str,
        trigger_data: Dict[str, Any],
        triggered_by: str = "manual"
    ) -> str:
        """Execute a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow_def = self.workflows[workflow_id]
        if not workflow_def.enabled:
            raise ValueError(f"Workflow disabled: {workflow_id}")
        
        # Create execution instance
        execution_id = str(uuid4())
        execution = WorkflowExecution(
            id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            started_at=datetime.utcnow(),
            triggered_by=triggered_by,
            trigger_data=trigger_data,
            task_results={},
            variables=workflow_def.variables.copy()
        )
        
        self.workflow_executions[execution_id] = execution
        
        # Start execution
        asyncio.create_task(self._execute_workflow_tasks(execution))
        
        logger.info(f"Started workflow execution: {execution_id}")
        return execution_id
    
    async def _execute_workflow_tasks(self, execution: WorkflowExecution):
        """Execute workflow tasks"""
        try:
            execution.status = WorkflowStatus.RUNNING
            workflow_def = self.workflows[execution.workflow_id]
            
            # Build task dependency graph
            task_graph = self._build_task_graph(workflow_def.tasks)
            
            # Execute tasks in dependency order
            completed_tasks = set()
            
            while len(completed_tasks) < len(workflow_def.tasks):
                # Find tasks ready to execute
                ready_tasks = []
                for task in workflow_def.tasks:
                    if (task.id not in completed_tasks and 
                        all(dep in completed_tasks for dep in task.dependencies)):
                        ready_tasks.append(task)
                
                if not ready_tasks:
                    raise RuntimeError("Circular dependency detected in workflow tasks")
                
                # Execute ready tasks (parallel if specified)
                if any(task.parallel_execution for task in ready_tasks):
                    await self._execute_tasks_parallel(ready_tasks, execution)
                else:
                    for task in ready_tasks:
                        await self._execute_single_task(task, execution)
                
                completed_tasks.update(task.id for task in ready_tasks)
            
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
    
    async def _execute_single_task(self, task: WorkflowTask, execution: WorkflowExecution):
        """Execute a single workflow task"""
        if task.task_type not in self.task_executors:
            raise ValueError(f"No executor registered for task type: {task.task_type}")
        
        executor = self.task_executors[task.task_type]
        
        try:
            # Prepare task context
            context = {
                'execution_id': execution.id,
                'workflow_id': execution.workflow_id,
                'trigger_data': execution.trigger_data,
                'variables': execution.variables,
                'previous_results': execution.task_results
            }
            
            # Execute task with timeout and retries
            result = await self._execute_with_retries(
                executor.execute(task, context),
                task.retry_count,
                task.retry_delay_seconds,
                task.timeout_minutes * 60
            )
            
            execution.task_results[task.id] = result
            
        except Exception as e:
            logger.error(f"Task execution failed: {task.id} - {e}")
            execution.task_results[task.id] = {"error": str(e)}
            
            if task.on_failure:
                # Handle failure actions
                pass
    
    async def _execute_tasks_parallel(self, tasks: List[WorkflowTask], execution: WorkflowExecution):
        """Execute multiple tasks in parallel"""
        task_coroutines = []
        for task in tasks:
            if task.parallel_execution:
                task_coroutines.append(self._execute_single_task(task, execution))
        
        if task_coroutines:
            await asyncio.gather(*task_coroutines, return_exceptions=True)
    
    # Monitoring and Health Checks
    async def _health_check_monitor(self):
        """Monitor service health"""
        while self.running:
            try:
                for service_id, instance in self.services.items():
                    await self._check_service_health(instance)
                
                await self._update_metrics()
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _workflow_monitor(self):
        """Monitor workflow executions"""
        while self.running:
            try:
                # Clean up completed workflows
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                to_remove = []
                for exec_id, execution in self.workflow_executions.items():
                    if (execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED] and
                        execution.completed_at and execution.completed_at < cutoff_time):
                        to_remove.append(exec_id)
                
                for exec_id in to_remove:
                    del self.workflow_executions[exec_id]
                
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Workflow monitor error: {e}")
                await asyncio.sleep(300)
    
    async def _check_service_health(self, instance: ServiceInstance):
        """Check health of a service instance"""
        try:
            if hasattr(instance.instance_object, 'health_check'):
                is_healthy = await instance.instance_object.health_check()
                if not is_healthy and instance.status == ServiceStatus.RUNNING:
                    instance.status = ServiceStatus.ERROR
                    logger.warning(f"Service health check failed: {instance.service_id}")
            
            instance.last_health_check = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Health check error for {instance.service_id}: {e}")
            instance.status = ServiceStatus.ERROR
            instance.error_message = str(e)
    
    # Utility Methods
    def _build_dependency_graph(self):
        """Build service dependency graph"""
        for service_id, definition in self.service_registry.items():
            self.dependency_graph[service_id] = definition.dependencies
    
    def _build_task_graph(self, tasks: List[WorkflowTask]) -> Dict[str, List[str]]:
        """Build task dependency graph"""
        return {task.id: task.dependencies for task in tasks}
    
    async def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if service dependencies are met"""
        for dep in dependencies:
            if dep not in self.services or self.services[dep].status != ServiceStatus.RUNNING:
                return False
        return True
    
    async def _execute_with_retries(
        self,
        coro,
        retry_count: int,
        retry_delay: int,
        timeout: int
    ):
        """Execute coroutine with retries and timeout"""
        for attempt in range(retry_count + 1):
            try:
                return await asyncio.wait_for(coro, timeout=timeout)
            except Exception as e:
                if attempt == retry_count:
                    raise
                await asyncio.sleep(retry_delay)
    
    async def _update_metrics(self):
        """Update orchestration metrics"""
        self.metrics.total_services = len(self.service_registry)
        self.metrics.running_services = sum(
            1 for s in self.services.values() 
            if s.status == ServiceStatus.RUNNING
        )
        self.metrics.failed_services = sum(
            1 for s in self.services.values() 
            if s.status == ServiceStatus.ERROR
        )
        self.metrics.total_workflows = len(self.workflows)
        self.metrics.active_workflows = sum(
            1 for e in self.workflow_executions.values() 
            if e.status == WorkflowStatus.RUNNING
        )
        self.metrics.completed_workflows = sum(
            1 for e in self.workflow_executions.values() 
            if e.status == WorkflowStatus.COMPLETED
        )
        self.metrics.failed_workflows = sum(
            1 for e in self.workflow_executions.values() 
            if e.status == WorkflowStatus.FAILED
        )
        self.metrics.last_updated = datetime.utcnow()
    
    async def _initialize_built_in_services(self):
        """Initialize built-in service definitions"""
        # This would be populated with actual service definitions
        pass
    
    async def _load_workflows(self):
        """Load workflow definitions from storage"""
        # This would load workflows from Redis or database
        pass
    
    async def _stop_all_services(self):
        """Stop all running services"""
        for service_id in list(self.services.keys()):
            await self.stop_service(service_id)
    
    # Public API Methods
    def get_service_status(self, service_id: str) -> Optional[ServiceStatus]:
        """Get status of a service"""
        if service_id in self.services:
            return self.services[service_id].status
        return None
    
    def get_workflow_status(self, execution_id: str) -> Optional[WorkflowStatus]:
        """Get status of a workflow execution"""
        if execution_id in self.workflow_executions:
            return self.workflow_executions[execution_id].status
        return None
    
    def get_metrics(self) -> OrchestrationMetrics:
        """Get orchestration metrics"""
        return self.metrics
    
    def list_services(self) -> List[Dict[str, Any]]:
        """List all services and their status"""
        return [
            {
                'service_id': instance.service_id,
                'name': instance.definition.name,
                'status': instance.status.value,
                'start_time': instance.start_time.isoformat() if instance.start_time else None,
                'restart_count': instance.restart_count
            }
            for instance in self.services.values()
        ]
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflow executions"""
        return [
            {
                'execution_id': execution.id,
                'workflow_id': execution.workflow_id,
                'status': execution.status.value,
                'started_at': execution.started_at.isoformat(),
                'triggered_by': execution.triggered_by
            }
            for execution in self.workflow_executions.values()
        ]