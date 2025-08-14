"""
Production Orchestration Engine
Advanced workflow automation and orchestration for security operations
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import yaml
from pathlib import Path
import croniter
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import inspect

from .interfaces import WorkflowOrchestrationService

logger = logging.getLogger(__name__)

class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

class TriggerType(str, Enum):
    """Workflow trigger types"""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    WEBHOOK = "webhook"
    THRESHOLD = "threshold"

@dataclass
class WorkflowTask:
    """Individual task within a workflow"""
    task_id: str
    name: str
    task_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 300  # 5 minutes default
    retry_count: int = 3
    retry_delay: int = 10  # seconds
    condition: Optional[str] = None
    on_failure: str = "fail"  # fail, continue, retry
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    workflow_id: str
    name: str
    description: str
    version: str
    tasks: List[WorkflowTask]
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 3600  # 1 hour default
    max_parallel_tasks: int = 5
    on_failure_action: str = "stop"
    notifications: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class WorkflowExecution:
    """Workflow execution instance"""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    task_executions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    trigger_type: str = "manual"
    trigger_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskExecution:
    """Task execution details"""
    task_id: str
    execution_id: str
    status: TaskStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    logs: List[str] = field(default_factory=list)

class ProductionOrchestrationEngine(WorkflowOrchestrationService):
    """Production-ready workflow orchestration engine"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.task_handlers: Dict[str, Callable] = {}
        self.scheduled_workflows: Dict[str, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = False

        # Initialize built-in task handlers
        self._register_builtin_handlers()

        # Start the orchestration engine
        asyncio.create_task(self._start_orchestration_loop())

    async def create_workflow(
        self,
        workflow_definition: Dict[str, Any],
        tenant_id: Optional[str] = None
    ) -> str:
        """Create a new workflow definition"""

        try:
            # Validate workflow definition
            await self._validate_workflow_definition(workflow_definition)

            # Create workflow object
            tasks = [
                WorkflowTask(**task)
                for task in workflow_definition.get("tasks", [])
            ]

            workflow = WorkflowDefinition(
                workflow_id=workflow_definition.get("workflow_id", str(uuid.uuid4())),
                name=workflow_definition["name"],
                description=workflow_definition.get("description", ""),
                version=workflow_definition.get("version", "1.0"),
                tasks=tasks,
                triggers=workflow_definition.get("triggers", []),
                variables=workflow_definition.get("variables", {}),
                timeout=workflow_definition.get("timeout", 3600),
                max_parallel_tasks=workflow_definition.get("max_parallel_tasks", 5),
                on_failure_action=workflow_definition.get("on_failure_action", "stop"),
                notifications=workflow_definition.get("notifications", [])
            )

            # Store workflow
            self.workflows[workflow.workflow_id] = workflow

            # Schedule if needed
            await self._schedule_workflow_triggers(workflow)

            logger.info(f"Created workflow {workflow.workflow_id}: {workflow.name}")
            return workflow.workflow_id

        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            raise

    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Optional[Dict[str, Any]] = None,
        trigger_type: str = "manual"
    ) -> str:
        """Execute a workflow"""

        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")

            workflow = self.workflows[workflow_id]

            # Create execution instance
            execution = WorkflowExecution(
                execution_id=str(uuid.uuid4()),
                workflow_id=workflow_id,
                status=WorkflowStatus.PENDING,
                started_at=datetime.utcnow(),
                context=input_data or {},
                trigger_type=trigger_type
            )

            self.executions[execution.execution_id] = execution

            # Start execution in background
            asyncio.create_task(self._execute_workflow_async(execution))

            logger.info(f"Started workflow execution {execution.execution_id}")
            return execution.execution_id

        except Exception as e:
            logger.error(f"Failed to execute workflow {workflow_id}: {e}")
            raise

    async def get_workflow_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow execution status"""

        if execution_id not in self.executions:
            return None

        execution = self.executions[execution_id]

        return {
            "execution_id": execution.execution_id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "started_at": execution.started_at.isoformat(),
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "context": execution.context,
            "task_executions": execution.task_executions,
            "error_message": execution.error_message,
            "trigger_type": execution.trigger_type
        }

    async def schedule_recurring_scan(
        self,
        scan_config: Dict[str, Any],
        schedule: str
    ) -> str:
        """Schedule recurring security scan"""

        # Create workflow for recurring scan
        workflow_definition = {
            "name": f"Recurring Scan - {scan_config.get('name', 'Unnamed')}",
            "description": "Automated recurring security scan",
            "tasks": [
                {
                    "task_id": "validate_targets",
                    "name": "Validate Scan Targets",
                    "task_type": "target_validation",
                    "parameters": {
                        "targets": scan_config.get("targets", [])
                    }
                },
                {
                    "task_id": "execute_scan",
                    "name": "Execute Security Scan",
                    "task_type": "security_scan",
                    "parameters": scan_config,
                    "dependencies": ["validate_targets"]
                },
                {
                    "task_id": "analyze_results",
                    "name": "Analyze Scan Results",
                    "task_type": "result_analysis",
                    "dependencies": ["execute_scan"]
                },
                {
                    "task_id": "generate_report",
                    "name": "Generate Security Report",
                    "task_type": "report_generation",
                    "dependencies": ["analyze_results"]
                },
                {
                    "task_id": "send_notifications",
                    "name": "Send Notifications",
                    "task_type": "notification",
                    "dependencies": ["generate_report"],
                    "parameters": {
                        "recipients": scan_config.get("notification_recipients", [])
                    }
                }
            ],
            "triggers": [
                {
                    "type": "scheduled",
                    "schedule": schedule,
                    "enabled": True
                }
            ]
        }

        return await self.create_workflow(workflow_definition)

    async def _execute_workflow_async(self, execution: WorkflowExecution):
        """Execute workflow asynchronously"""

        try:
            execution.status = WorkflowStatus.RUNNING
            workflow = self.workflows[execution.workflow_id]

            # Build task dependency graph
            task_graph = self._build_task_dependency_graph(workflow.tasks)

            # Execute tasks based on dependencies
            await self._execute_task_graph(execution, task_graph, workflow)

            # Mark as completed
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.utcnow()

            # Send completion notifications
            await self._send_workflow_notifications(execution, "completed")

            logger.info(f"Workflow execution {execution.execution_id} completed successfully")

        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()

            await self._send_workflow_notifications(execution, "failed")
            logger.error(f"Workflow execution {execution.execution_id} failed: {e}")

    async def _execute_task_graph(
        self,
        execution: WorkflowExecution,
        task_graph: Dict[str, List[str]],
        workflow: WorkflowDefinition
    ):
        """Execute tasks based on dependency graph"""

        completed_tasks = set()
        failed_tasks = set()

        while len(completed_tasks) + len(failed_tasks) < len(workflow.tasks):

            # Find tasks ready to execute
            ready_tasks = []
            for task in workflow.tasks:
                if (task.task_id not in completed_tasks and
                    task.task_id not in failed_tasks and
                    all(dep in completed_tasks for dep in task.dependencies)):
                    ready_tasks.append(task)

            if not ready_tasks:
                # No tasks ready - check for circular dependencies
                remaining_tasks = [
                    task.task_id for task in workflow.tasks
                    if task.task_id not in completed_tasks and task.task_id not in failed_tasks
                ]
                if remaining_tasks:
                    raise Exception(f"Circular dependency or unresolvable dependencies: {remaining_tasks}")
                break

            # Execute ready tasks (up to max parallel limit)
            parallel_limit = min(len(ready_tasks), workflow.max_parallel_tasks)

            # Create task execution objects
            task_executions = []
            for task in ready_tasks[:parallel_limit]:
                task_exec = TaskExecution(
                    task_id=task.task_id,
                    execution_id=execution.execution_id,
                    status=TaskStatus.PENDING
                )
                task_executions.append((task, task_exec))
                execution.task_executions[task.task_id] = asdict(task_exec)

            # Execute tasks in parallel
            results = await asyncio.gather(
                *[self._execute_single_task(task, task_exec, execution) for task, task_exec in task_executions],
                return_exceptions=True
            )

            # Process results
            for i, result in enumerate(results):
                task, task_exec = task_executions[i]

                if isinstance(result, Exception):
                    task_exec.status = TaskStatus.FAILED
                    task_exec.error_message = str(result)
                    failed_tasks.add(task.task_id)

                    if workflow.on_failure_action == "stop":
                        raise Exception(f"Task {task.task_id} failed: {result}")
                else:
                    task_exec.status = TaskStatus.COMPLETED
                    task_exec.result = result
                    completed_tasks.add(task.task_id)

                task_exec.completed_at = datetime.utcnow()
                execution.task_executions[task.task_id] = asdict(task_exec)

    async def _execute_single_task(
        self,
        task: WorkflowTask,
        task_execution: TaskExecution,
        workflow_execution: WorkflowExecution
    ) -> Any:
        """Execute a single task"""

        try:
            task_execution.status = TaskStatus.RUNNING
            task_execution.started_at = datetime.utcnow()

            # Get task handler
            if task.task_type not in self.task_handlers:
                raise Exception(f"No handler found for task type: {task.task_type}")

            handler = self.task_handlers[task.task_type]

            # Prepare task context
            task_context = {
                "task": task,
                "execution": workflow_execution,
                "task_execution": task_execution
            }

            # Execute with timeout
            result = await asyncio.wait_for(
                self._call_task_handler(handler, task, task_context),
                timeout=task.timeout
            )

            return result

        except asyncio.TimeoutError:
            raise Exception(f"Task {task.task_id} timed out after {task.timeout} seconds")
        except Exception as e:
            # Handle retries
            if task_execution.retry_count < task.retry_count:
                task_execution.retry_count += 1
                task_execution.status = TaskStatus.RETRYING
                await asyncio.sleep(task.retry_delay)
                return await self._execute_single_task(task, task_execution, workflow_execution)
            else:
                raise e

    async def _call_task_handler(
        self,
        handler: Callable,
        task: WorkflowTask,
        context: Dict[str, Any]
    ) -> Any:
        """Call task handler (sync or async)"""

        if inspect.iscoroutinefunction(handler):
            return await handler(task.parameters, context)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                handler,
                task.parameters,
                context
            )

    def _build_task_dependency_graph(self, tasks: List[WorkflowTask]) -> Dict[str, List[str]]:
        """Build task dependency graph"""

        graph = {}
        for task in tasks:
            graph[task.task_id] = task.dependencies

        return graph

    def _register_builtin_handlers(self):
        """Register built-in task handlers"""

        self.task_handlers.update({
            "target_validation": self._handle_target_validation,
            "security_scan": self._handle_security_scan,
            "result_analysis": self._handle_result_analysis,
            "report_generation": self._handle_report_generation,
            "notification": self._handle_notification,
            "delay": self._handle_delay,
            "conditional": self._handle_conditional,
            "data_transformation": self._handle_data_transformation,
            "api_call": self._handle_api_call,
            "file_operation": self._handle_file_operation
        })

    # Built-in task handlers

    async def _handle_target_validation(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate scan targets"""

        targets = parameters.get("targets", [])
        validated_targets = []
        invalid_targets = []

        for target in targets:
            # Simple validation logic
            if isinstance(target, str) and len(target) > 0:
                validated_targets.append(target)
            else:
                invalid_targets.append(target)

        return {
            "validated_targets": validated_targets,
            "invalid_targets": invalid_targets,
            "validation_passed": len(invalid_targets) == 0
        }

    async def _handle_security_scan(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute security scan"""

        # Mock security scan execution
        await asyncio.sleep(2)  # Simulate scan time

        return {
            "scan_id": str(uuid.uuid4()),
            "targets_scanned": len(parameters.get("targets", [])),
            "vulnerabilities_found": 3,
            "scan_duration": 120,
            "status": "completed"
        }

    async def _handle_result_analysis(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze scan results"""

        # Mock result analysis
        return {
            "analysis_id": str(uuid.uuid4()),
            "risk_score": 7.5,
            "critical_vulnerabilities": 1,
            "high_vulnerabilities": 2,
            "recommendations": [
                "Patch critical vulnerabilities immediately",
                "Update SSL configuration",
                "Implement additional access controls"
            ]
        }

    async def _handle_report_generation(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate security report"""

        report_id = str(uuid.uuid4())

        return {
            "report_id": report_id,
            "report_url": f"/reports/{report_id}",
            "format": parameters.get("format", "pdf"),
            "generated_at": datetime.utcnow().isoformat()
        }

    async def _handle_notification(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send notifications"""

        recipients = parameters.get("recipients", [])
        message = parameters.get("message", "Workflow completed")

        # Mock notification sending
        for recipient in recipients:
            logger.info(f"Sending notification to {recipient}: {message}")

        return {
            "notifications_sent": len(recipients),
            "status": "success"
        }

    async def _handle_delay(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Delay execution"""

        delay_seconds = parameters.get("delay_seconds", 1)
        await asyncio.sleep(delay_seconds)

        return {"delayed_for": delay_seconds}

    async def _handle_conditional(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Conditional execution"""

        condition = parameters.get("condition", "true")

        # Simple condition evaluation
        if condition == "true":
            return {"condition_met": True}
        else:
            return {"condition_met": False}

    async def _handle_data_transformation(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Transform data"""

        input_data = parameters.get("input_data", {})
        transformation_type = parameters.get("transformation_type", "passthrough")

        if transformation_type == "passthrough":
            return input_data
        elif transformation_type == "json_to_dict":
            if isinstance(input_data, str):
                return json.loads(input_data)
            return input_data
        else:
            return {"transformed_data": input_data}

    async def _handle_api_call(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make API call"""

        url = parameters.get("url")
        method = parameters.get("method", "GET")
        headers = parameters.get("headers", {})
        payload = parameters.get("payload")

        # Mock API call
        return {
            "status_code": 200,
            "response": {"message": "API call successful"},
            "url": url,
            "method": method
        }

    async def _handle_file_operation(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """File operations"""

        operation = parameters.get("operation", "read")
        file_path = parameters.get("file_path")

        if operation == "read":
            # Mock file read
            return {"content": "file content", "file_path": file_path}
        elif operation == "write":
            # Mock file write
            return {"bytes_written": 100, "file_path": file_path}
        else:
            return {"operation": operation, "file_path": file_path}

    async def _start_orchestration_loop(self):
        """Start the main orchestration loop"""

        self.running = True

        while self.running:
            try:
                # Check for scheduled workflows
                await self._check_scheduled_workflows()

                # Cleanup completed executions (keep last 100)
                await self._cleanup_old_executions()

                # Wait before next check
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Orchestration loop error: {e}")
                await asyncio.sleep(60)

    async def _check_scheduled_workflows(self):
        """Check and trigger scheduled workflows"""

        current_time = datetime.utcnow()

        for workflow_id, workflow in self.workflows.items():
            for trigger in workflow.triggers:
                if trigger.get("type") == "scheduled" and trigger.get("enabled", True):
                    schedule = trigger.get("schedule")
                    last_run = trigger.get("last_run")

                    if schedule and self._should_trigger_scheduled_workflow(schedule, last_run, current_time):
                        # Trigger workflow
                        await self.execute_workflow(workflow_id, trigger_type="scheduled")
                        trigger["last_run"] = current_time.isoformat()

    def _should_trigger_scheduled_workflow(
        self,
        schedule: str,
        last_run: Optional[str],
        current_time: datetime
    ) -> bool:
        """Check if scheduled workflow should be triggered"""

        try:
            # Parse cron schedule
            cron = croniter.croniter(schedule, current_time)
            next_run = cron.get_prev(datetime)

            if last_run:
                last_run_dt = datetime.fromisoformat(last_run)
                return next_run > last_run_dt
            else:
                # First run
                return True

        except Exception as e:
            logger.error(f"Failed to parse schedule {schedule}: {e}")
            return False

    async def _cleanup_old_executions(self):
        """Cleanup old workflow executions"""

        if len(self.executions) > 100:
            # Sort by completion time and keep the latest 100
            sorted_executions = sorted(
                self.executions.items(),
                key=lambda x: x[1].completed_at or x[1].started_at,
                reverse=True
            )

            # Keep only the latest 100
            executions_to_keep = dict(sorted_executions[:100])
            self.executions = executions_to_keep

    async def _validate_workflow_definition(self, workflow_def: Dict[str, Any]):
        """Validate workflow definition"""

        required_fields = ["name", "tasks"]
        for field in required_fields:
            if field not in workflow_def:
                raise ValueError(f"Missing required field: {field}")

        # Validate tasks
        tasks = workflow_def.get("tasks", [])
        if not tasks:
            raise ValueError("Workflow must have at least one task")

        task_ids = set()
        for task in tasks:
            if "task_id" not in task:
                raise ValueError("Task missing task_id")

            if task["task_id"] in task_ids:
                raise ValueError(f"Duplicate task_id: {task['task_id']}")

            task_ids.add(task["task_id"])

            # Validate dependencies
            for dep in task.get("dependencies", []):
                if dep not in task_ids and dep not in [t["task_id"] for t in tasks]:
                    raise ValueError(f"Invalid dependency: {dep}")

    async def _send_workflow_notifications(self, execution: WorkflowExecution, event_type: str):
        """Send workflow notifications"""

        try:
            workflow = self.workflows.get(execution.workflow_id)
            if not workflow or not workflow.notifications:
                return

            for notification in workflow.notifications:
                if event_type in notification.get("events", []):
                    # Send notification (mock implementation)
                    logger.info(f"Sending {event_type} notification for workflow {execution.workflow_id}")

        except Exception as e:
            logger.error(f"Failed to send workflow notifications: {e}")

    async def _schedule_workflow_triggers(self, workflow: WorkflowDefinition):
        """Schedule workflow triggers"""

        for trigger in workflow.triggers:
            if trigger.get("type") == "scheduled":
                self.scheduled_workflows[workflow.workflow_id] = {
                    "workflow": workflow,
                    "trigger": trigger
                }

    def register_task_handler(self, task_type: str, handler: Callable):
        """Register custom task handler"""
        self.task_handlers[task_type] = handler
        logger.info(f"Registered task handler for type: {task_type}")

    def stop(self):
        """Stop the orchestration engine"""
        self.running = False
