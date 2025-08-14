from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable, Awaitable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from src.orchestrator.core.workflow_engine import (
    WorkflowStatus, WorkflowTask, WorkflowDefinition, WorkflowExecution, TaskExecutor
)

@dataclass
class OrchestratorConfig:
    """Configuration for workflow orchestrator"""
    redis_url: str = "redis://localhost:6379"
    temporal_host: str = "localhost:7233"
    max_concurrent_executions: int = 100
    default_task_timeout: int = 60  # minutes
    default_retry_count: int = 3
    default_retry_delay: int = 60  # seconds
    enable_distributed_execution: bool = False
    service_urls: Dict[str, str] = None
    workflows_dir: str = "./workflows"

    def __post_init__(self):
        if self.service_urls is None:
            self.service_urls = {
                'scanner': 'http://localhost:8001',
                'compliance': 'http://localhost:8002',
                'notifications': 'http://localhost:8003'
            }

class BaseOrchestrator(ABC):
    """Abstract base class for workflow orchestrators"""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.executors: Dict[str, TaskExecutor] = {}
        self.redis_client = None
        self.temporal_client = None
        self.running = False
        self._initialization_complete = False

    @abstractmethod
    async def initialize(self):
        """Initialize the orchestrator"""
        pass

    @abstractmethod
    async def shutdown(self):
        """Shutdown the orchestrator"""
        pass

    async def create_workflow(self, workflow_def: WorkflowDefinition) -> str:
        """Create a new workflow definition"""
        try:
            # Validate workflow
            await self._validate_workflow(workflow_def)

            # Store workflow
            self.workflows[workflow_def.id] = workflow_def

            # Persist to storage
            await self._persist_workflow(workflow_def)

            logger.info(f"Created workflow: {workflow_def.id}")
            return workflow_def.id

        except Exception as e:
            logger.error(f"Failed to create workflow {workflow_def.id}: {e}")
            raise

    async def execute_workflow(self, workflow_id: str, trigger_data: Dict[str, Any] = None,
                             triggered_by: str = "manual") -> str:
        """Execute a workflow"""
        try:
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")

            workflow_def = self.workflows[workflow_id]

            if not workflow_def.enabled:
                raise ValueError(f"Workflow {workflow_id} is disabled")

            execution_id = self._generate_execution_id()

            # Create execution record
            execution = WorkflowExecution(
                id=execution_id,
                workflow_id=workflow_id,
                status=WorkflowStatus.PENDING,
                started_at=datetime.now(),
                completed_at=None,
                triggered_by=triggered_by,
                trigger_data=trigger_data or {},
                task_results={},
                variables=workflow_def.variables.copy()
            )

            self.executions[execution_id] = execution

            # Start execution asynchronously
            asyncio.create_task(self._execute_workflow_tasks(execution, workflow_def))

            logger.info(f"Started workflow execution: {execution_id}")
            return execution_id

        except Exception as e:
            logger.error(f"Failed to execute workflow {workflow_id}: {e}")
            raise

    async def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution status"""
        return self.executions.get(execution_id)

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel workflow execution"""
        if execution_id in self.executions:
            execution = self.executions[execution_id]
            if execution.status == WorkflowStatus.RUNNING:
                execution.status = WorkflowStatus.CANCELLED
                execution.completed_at = datetime.now()
                return True
        return False

    async def _validate_workflow(self, workflow_def: WorkflowDefinition):
        """Validate workflow definition"""
        # Check for circular dependencies
        task_ids = {task.id for task in workflow_def.tasks}

        for task in workflow_def.tasks:
            for dep in task.dependencies:
                if dep not in task_ids:
                    raise ValueError(f"Task {task.id} depends on non-existent task {dep}")

        # Check for cycles (simplified check)
        visited = set()
        for task in workflow_def.tasks:
            if await self._has_circular_dependency(task, workflow_def.tasks, visited):
                raise ValueError(f"Circular dependency detected involving task {task.id}")

    async def _has_circular_dependency(self, task: WorkflowTask, all_tasks: List[WorkflowTask],
                                     visited: set, path: set = None) -> bool:
        """Check for circular dependencies"""
        if path is None:
            path = set()

        if task.id in path:
            return True

        if task.id in visited:
            return False

        visited.add(task.id)
        path.add(task.id)

        task_map = {t.id: t for t in all_tasks}

        for dep_id in task.dependencies:
            if dep_id in task_map:
                dep_task = task_map[dep_id]
                if await self._has_circular_dependency(dep_task, all_tasks, visited, path):
                    return True

        path.remove(task.id)
        return False

    async def _execute_workflow_tasks(self, execution: WorkflowExecution,
                                    workflow_def: WorkflowDefinition):
        """Execute workflow tasks"""
        try:
            execution.status = WorkflowStatus.RUNNING

            # Build task dependency graph
            task_graph = self._build_task_graph(workflow_def.tasks)

            # Execute tasks in topological order
            completed_tasks = set()

            while len(completed_tasks) < len(workflow_def.tasks):
                # Find tasks ready to execute
                ready_tasks = []
                for task in workflow_def.tasks:
                    if (task.id not in completed_tasks and
                        all(dep in completed_tasks for dep in task.dependencies)):
                        ready_tasks.append(task)

                if not ready_tasks:
                    raise Exception("Circular dependency detected in workflow tasks")

                # Execute ready tasks (parallel if allowed)
                parallel_tasks = [t for t in ready_tasks if t.parallel_execution]
                serial_tasks = [t for t in ready_tasks if not t.parallel_execution]

                # Execute parallel tasks
                if parallel_tasks:
                    await asyncio.gather(*[
                        self._execute_task(task, execution)
                        for task in parallel_tasks
                    ])
                    completed_tasks.update(task.id for task in parallel_tasks)

                # Execute serial tasks
                for task in serial_tasks:
                    await self._execute_task(task, execution)
                    completed_tasks.add(task.id)

            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now()

            # Send success notifications
            await self._send_workflow_notifications(execution, workflow_def, 'success')

            logger.info(f"Workflow execution completed: {execution.id}")

        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()

            # Send failure notifications
            await self._send_workflow_notifications(execution, workflow_def, 'failure')

            logger.error(f"Workflow execution failed: {execution.id} - {e}")

    async def _execute_task(self, task: WorkflowTask, execution: WorkflowExecution):
        """Execute a single task"""
        task_start_time = datetime.now()

        try:
            logger.info(f"Executing task: {task.id} in workflow {execution.id}")

            # Check condition if specified
            if task.condition and not self._evaluate_condition(task.condition, execution):
                logger.info(f"Task {task.id} skipped due to condition: {task.condition}")
                execution.task_results[task.id] = {
                    'status': 'skipped',
                    'reason': 'condition_not_met',
                    'condition': task.condition
                }
                return

            # Get executor for task type
            if task.task_type not in self.executors:
                raise Exception(f"No executor found for task type: {task.task_type}")

            executor = self.executors[task.task_type]

            # Validate parameters
            if not executor.validate_parameters(task.parameters):
                raise Exception(f"Invalid parameters for task {task.id}")

            # Execute with retries
            retry_count = 0
            while retry_count <= task.retry_count:
                try:
                    # Create execution context
                    context = {
                        'execution_id': execution.id,
                        'workflow_id': execution.workflow_id,
                        'task_id': task.id,
                        'variables': execution.variables,
                        'trigger_data': execution.trigger_data,
                        'previous_results': execution.task_results
                    }

                    # Execute with timeout
                    result = await asyncio.wait_for(
                        executor.execute(task, context),
                        timeout=task.timeout_minutes * 60
                    )

                    # Store successful result
                    execution.task_results[task.id] = {
                        'status': 'completed',
                        'result': result,
                        'duration_seconds': (datetime.now() - task_start_time).total_seconds(),
                        'retry_count': retry_count
                    }

                    # Update execution variables with task results
                    if isinstance(result, dict):
                        for key, value in result.items():
                            execution.variables[f"{task.id}_{key}"] = value

                    logger.info(f"Task {task.id} completed successfully")
                    return

                except asyncio.TimeoutError:
                    error_msg = f"Task {task.id} timed out after {task.timeout_minutes} minutes"
                    logger.warning(f"{error_msg} (attempt {retry_count + 1})")

                    if retry_count >= task.retry_count:
                        raise Exception(error_msg)

                except Exception as e:
                    error_msg = f"Task {task.id} failed: {str(e)}"
                    logger.warning(f"{error_msg} (attempt {retry_count + 1})")

                    if retry_count >= task.retry_count:
                        raise Exception(error_msg)

                retry_count += 1
                if retry_count <= task.retry_count:
                    await asyncio.sleep(task.retry_delay_seconds)

        except Exception as e:
            # Store failed result
            execution.task_results[task.id] = {
                'status': 'failed',
                'error': str(e),
                'duration_seconds': (datetime.now() - task_start_time).total_seconds(),
                'retry_count': retry_count
            }

            # Execute failure actions if specified
            if task.on_failure:
                for action in task.on_failure:
                    await self._execute_failure_action(action, execution, task)

            raise

    def _build_task_graph(self, tasks: List[WorkflowTask]) -> Dict[str, List[str]]:
        """Build task dependency graph"""
        graph = {}
        for task in tasks:
            graph[task.id] = task.dependencies
        return graph

    def _evaluate_condition(self, condition: str, execution: WorkflowExecution) -> bool:
        """Evaluate task condition"""
        try:
            # Simple condition evaluation - in production, use a proper expression evaluator
            # For now, support basic comparisons
            context = {
                'variables': execution.variables,
                'trigger_data': execution.trigger_data,
                'task_results': execution.task_results
            }

            # Very basic evaluation - extend as needed
            if 'threat_severity' in condition:
                severity = execution.trigger_data.get('severity', 'low')
                if '>=' in condition:
                    target_severity = condition.split('>=')[1].strip().strip("'\"")
                    severity_levels = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
                    return severity_levels.get(severity, 1) >= severity_levels.get(target_severity, 1)

            return True  # Default to true for unknown conditions

        except Exception:
            return True  # Default to true on evaluation error

    async def _execute_failure_action(self, action: str, execution: WorkflowExecution,
                                    failed_task: WorkflowTask):
        """Execute failure action"""
        try:
            # Simple failure actions - extend as needed
            if action == 'notify_admin':
                # Send admin notification
                pass
            elif action == 'rollback':
                # Execute rollback procedures
                pass
        except Exception as e:
            logger.error(f"Failed to execute failure action {action}: {e}")

    async def _send_workflow_notifications(self, execution: WorkflowExecution,
                                         workflow_def: WorkflowDefinition,
                                         event_type: str):
        """Send workflow notifications"""
        try:
            notification_config = workflow_def.notifications.get(f'on_{event_type}', [])

            if notification_config and self.executors.get('notification'):
                notification_task = WorkflowTask(
                    id=f"notification_{event_type}",
                    name=f"Workflow {workflow_def.name} - {event_type.title()}",
                    task_type='notification',
                    description=f"Send {event_type} notification",
                    parameters={
                        'recipients': notification_config,
                        'subject': f"Workflow {workflow_def.name} has {event_type}ed. Execution ID: {execution.id}",
                        'template': f"Workflow {workflow_def.name} has {event_type}ed. Execution ID: {execution.id}",
                        'channels': ['email']
                    },
                    dependencies=[],
                    timeout_minutes=5,
                    retry_count=2,
                    retry_delay_seconds=30
                )

                context = {
                    'execution_id': execution.id,
                    'workflow_id': execution.workflow_id,
                    'variables': execution.variables
                }

                await self.executors['notification'].execute(notification_task, context)

        except Exception as e:
            logger.error(f"Failed to send workflow notifications: {e}")

    def _generate_execution_id(self) -> str:
        """Generate a unique execution ID"""
        return str(uuid.uuid4())

    async def _persist_workflow(self, workflow_def: WorkflowDefinition):
        """Persist workflow definition to storage"""
        # Implementation depends on specific storage backend
        pass

    async def _load_workflows(self):
        """Load workflow definitions from storage"""
        # Implementation depends on specific storage backend
        pass

    async def _scheduler_loop(self):
        """Main scheduler loop for handling scheduled workflows"""
        while self.running:
            try:
                current_time = datetime.now()

                # Check for scheduled workflows
                for workflow_id, workflow_def in self.workflows.items():
                    if not workflow_def.enabled:
                        continue

                    for trigger in workflow_def.triggers:
                        if trigger.get('type') == 'scheduled':
                            if await self._should_trigger_scheduled_workflow(workflow_def, trigger, current_time):
                                await self.execute_workflow(workflow_id, {}, 'scheduler')

            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")

            await asyncio.sleep(60)  # Check every minute

    async def _should_trigger_scheduled_workflow(self, workflow_def: WorkflowDefinition,
                                               trigger: Dict[str, Any],
                                               current_time: datetime) -> bool:
        """Check if scheduled workflow should be triggered"""
        try:
            # Simple cron-like scheduling - in production, use proper cron library
            schedule = trigger.get('schedule', '')

            # For now, just handle daily schedules
            if schedule.startswith('0 6 * * *'):  # Daily at 6 AM
                last_run_key = f"last_run:{workflow_def.id}"
                last_run = await self.redis_client.get(last_run_key)

                if last_run:
                    last_run_date = datetime.fromisoformat(last_run.decode())
                    if (current_time - last_run_date).days < 1:
                        return False

                # Check if it's around 6 AM
                if current_time.hour == 6 and current_time.minute < 5:
                    await self.redis_client.set(last_run_key, current_time.isoformat(), ex=86400)
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking scheduled trigger: {e}")
            return False
