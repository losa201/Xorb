from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.orchestrator.core.workflow_engine import WorkflowTask, WorkflowDefinition, WorkflowExecution, TaskExecutor

class ExecutionState:
    """Base class for execution state management"""

    def __init__(self):
        self._state = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the state"""
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the state"""
        self._state[key] = value

    def update(self, state: Dict[str, Any]) -> None:
        """Update the state with a dictionary"""
        self._state.update(state)

    def clear(self) -> None:
        """Clear the state"""
        self._state.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary"""
        return self._state.copy()

@dataclass
class TaskContext:
    """Execution context for task execution"""
    execution_id: str
    workflow_id: str
    task_id: str
    variables: Dict[str, Any]
    trigger_data: Dict[str, Any]
    previous_results: Dict[str, Any]
    state: ExecutionState

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context"""
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the context"""
        self.state.set(key, value)

class TaskExecutionEngine(ABC):
    """Abstract base class for task execution engines"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._executors = {}

    @abstractmethod
    async def initialize(self):
        """Initialize the execution engine"""
        pass

    @abstractmethod
    async def execute_task(self, task: WorkflowTask, context: TaskContext) -> Dict[str, Any]:
        """Execute a single task"""
        pass

    @abstractmethod
    async def execute_workflow(self, workflow: WorkflowDefinition, trigger_data: Dict[str, Any]) -> str:
        """Execute a complete workflow"""
        pass

    def register_executor(self, task_type: str, executor: TaskExecutor) -> None:
        """Register a task executor"""
        self._executors[task_type] = executor

    def get_executor(self, task_type: str) -> Optional[TaskExecutor]:
        """Get a task executor by type"""
        return self._executors.get(task_type)

    def validate_task(self, task: WorkflowTask) -> bool:
        """Validate a task before execution"""
        if task.task_type not in self._executors:
            return False

        executor = self._executors[task.task_type]
        return executor.validate_parameters(task.parameters)

    async def _execute_with_retries(self, task: WorkflowTask, context: TaskContext) -> Dict[str, Any]:
        """Execute a task with retries"""
        retry_count = 0
        last_error = None

        while retry_count <= task.retry_count:
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute_task(task, context),
                    timeout=task.timeout_minutes * 60
                )

                # Store successful result
                return {
                    'status': 'completed',
                    'result': result,
                    'duration_seconds': (datetime.now() - task_start_time).total_seconds(),
                    'retry_count': retry_count
                }

            except asyncio.TimeoutError:
                error_msg = f"Task {task.id} timed out after {task.timeout_minutes} minutes"

                if retry_count >= task.retry_count:
                    raise Exception(error_msg)

            except Exception as e:
                error_msg = f"Task {task.id} failed: {str(e)}"

                if retry_count >= task.retry_count:
                    raise Exception(error_msg)

            retry_count += 1
            if retry_count <= task.retry_count:
                await asyncio.sleep(task.retry_delay_seconds)

        raise Exception(f"Task {task.id} failed after {retry_count} attempts")

    async def _execute_task(self, task: WorkflowTask, context: TaskContext) -> Dict[str, Any]:
        """Execute a single task with all pre/post processing"""
        task_start_time = datetime.now()

        try:
            logger.info(f"Executing task: {task.id} in workflow {context.workflow_id}")

            # Check condition if specified
            if task.condition and not self._evaluate_condition(task.condition, context):
                logger.info(f"Task {task.id} skipped due to condition: {task.condition}")
                return {
                    'status': 'skipped',
                    'reason': 'condition_not_met',
                    'condition': task.condition
                }

            # Get executor for task type
            executor = self._executors.get(task.task_type)
            if not executor:
                raise Exception(f"No executor found for task type: {task.task_type}")

            # Validate parameters
            if not executor.validate_parameters(task.parameters):
                raise Exception(f"Invalid parameters for task {task.id}")

            # Execute the task
            result = await executor.execute(task, context)

            # Process result
            processed_result = self._process_result(result)

            return {
                'status': 'completed',
                'result': processed_result,
                'duration_seconds': (datetime.now() - task_start_time).total_seconds()
            }

        except Exception as e:
            # Log error
            logger.error(f"Task {task.id} failed: {str(e)}")

            # Execute failure actions if specified
            if task.on_failure:
                await self._execute_failure_actions(task.on_failure, context, task)

            raise

    def _process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process task result and update context"""
        # Default implementation - can be overridden
        return result

    def _evaluate_condition(self, condition: str, context: TaskContext) -> bool:
        """Evaluate task condition"""
        try:
            # Simple condition evaluation - in production, use a proper expression evaluator
            # For now, support basic comparisons
            context_dict = {
                'variables': context.variables,
                'trigger_data': context.trigger_data,
                'task_results': context.previous_results
            }

            # Very basic evaluation - extend as needed
            if 'threat_severity' in condition:
                severity = context.trigger_data.get('severity', 'low')
                if '>=' in condition:
                    target_severity = condition.split('>=')[1].strip().strip("'\"")
                    severity_levels = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
                    return severity_levels.get(severity, 1) >= severity_levels.get(target_severity, 1)

            return True  # Default to true for unknown conditions

        except Exception:
            return True  # Default to true on evaluation error

    async def _execute_failure_actions(self, actions: List[str], context: TaskContext,
                                    failed_task: WorkflowTask):
        """Execute failure actions"""
        try:
            # Simple failure actions - extend as needed
            for action in actions:
                if action == 'notify_admin':
                    # Send admin notification
                    await self._execute_notification_action(context, failed_task)
                elif action == 'rollback':
                    # Execute rollback procedures
                    await self._execute_rollback_action(context, failed_task)
                elif action.startswith('custom:'):  # Custom actions
                    await self._execute_custom_action(action[7:], context, failed_task)

        except Exception as e:
            logger.error(f"Failed to execute failure action {action}: {e}")

    async def _execute_notification_action(self, context: TaskContext, task: WorkflowTask):
        """Execute notification action for failures"""
        # Implementation for notification action
        pass

    async def _execute_rollback_action(self, context: TaskContext, task: WorkflowTask):
        """Execute rollback action for failures"""
        # Implementation for rollback action
        pass

    async def _execute_custom_action(self, action: str, context: TaskContext, task: WorkflowTask):
        """Execute custom action"""
        # Implementation for custom actions
        pass

class WorkflowExecutionEngine(TaskExecutionEngine):
    """Workflow execution engine that manages complete workflows"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._workflows = {}

    async def initialize(self):
        """Initialize the workflow engine"""
        await super().initialize()
        await self._load_workflows()

    async def _load_workflows(self):
        """Load workflow definitions from configuration"""
        # Implementation to load workflows
        pass

    async def execute_workflow(self, workflow_id: str, trigger_data: Dict[str, Any] = None,
                             triggered_by: str = "manual") -> str:
        """Execute a workflow"""
        if workflow_id not in self._workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow_def = self._workflows[workflow_id]

        if not workflow_def.enabled:
            raise ValueError(f"Workflow {workflow_id} is disabled")

        execution_id = str(uuid.uuid4())

        # Create execution context
        context = TaskContext(
            execution_id=execution_id,
            workflow_id=workflow_id,
            task_id="",
            variables=workflow_def.variables.copy(),
            trigger_data=trigger_data or {},
            previous_results={},
            state=ExecutionState()
        )

        # Start execution asynchronously
        asyncio.create_task(self._execute_workflow_tasks(workflow_def, context))

        logger.info(f"Started workflow execution: {execution_id}")
        return execution_id

    async def _execute_workflow_tasks(self, workflow_def: WorkflowDefinition, context: TaskContext):
        """Execute workflow tasks in correct order"""
        try:
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
                        self._execute_task(task, context)
                        for task in parallel_tasks
                    ])
                    completed_tasks.update(task.id for task in parallel_tasks)

                # Execute serial tasks
                for task in serial_tasks:
                    await self._execute_task(task, context)
                    completed_tasks.add(task.id)

            logger.info(f"Workflow execution completed: {context.execution_id}")

        except Exception as e:
            logger.error(f"Workflow execution failed: {context.execution_id} - {e}")

    def _build_task_graph(self, tasks: List[WorkflowTask]) -> Dict[str, List[str]]:
        """Build task dependency graph"""
        graph = {}
        for task in tasks:
            graph[task.id] = task.dependencies
        return graph

    def _validate_workflow(self, workflow_def: WorkflowDefinition):
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
            if self._has_circular_dependency(task, workflow_def.tasks, visited):
                raise ValueError(f"Circular dependency detected involving task {task.id}")

    def _has_circular_dependency(self, task: WorkflowTask, all_tasks: List[WorkflowTask],
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
                if self._has_circular_dependency(dep_task, all_tasks, visited, path):
                    return True

        path.remove(task.id)
        return False

class OrchestratorFactory:
    """Factory class for creating orchestrator instances"""

    @staticmethod
    def create_orchestrator(config: Dict[str, Any]) -> WorkflowExecutionEngine:
        """Create and configure a workflow orchestrator"""
        # Merge with default config
        default_config = {
            'redis_url': 'redis://localhost:6379',
            'temporal_host': 'localhost:7233',
            'service_urls': {
                'scanner': 'http://localhost:8001',
                'compliance': 'http://localhost:8002',
                'notifications': 'http://localhost:8003'
            },
            'workflows_dir': './workflows'
        }

        final_config = {**default_config, **config}

        # Create execution engine
        orchestrator = WorkflowExecutionEngine(final_config)

        return orchestrator
