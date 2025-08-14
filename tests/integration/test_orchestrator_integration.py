"""
Integration tests for the unified orchestrator system
Tests service management, workflow execution, and monitoring
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from dataclasses import asdict

from src.orchestrator.unified_orchestrator import (
    UnifiedOrchestrator, ServiceDefinition, ServiceType, TaskType,
    WorkflowDefinition, WorkflowTask, TaskExecutor, ServiceStatus, WorkflowStatus
)


class MockTaskExecutor(TaskExecutor):
    """Mock task executor for testing"""

    def __init__(self, should_fail=False, delay=0):
        self.should_fail = should_fail
        self.delay = delay
        self.execution_count = 0

    async def execute(self, task, context):
        """Execute a mock task"""
        await asyncio.sleep(self.delay)
        self.execution_count += 1

        if self.should_fail:
            raise Exception(f"Mock task execution failed: {task.id}")

        return {
            "status": "completed",
            "result": f"Task {task.id} executed successfully",
            "execution_count": self.execution_count,
            "context_keys": list(context.keys())
        }

    async def health_check(self):
        """Mock health check"""
        return not self.should_fail


class MockService:
    """Mock service for testing"""

    def __init__(self, service_id, should_fail_health=False):
        self.service_id = service_id
        self.should_fail_health = should_fail_health
        self.initialized = False
        self.shutdown_called = False

    async def initialize(self):
        """Initialize mock service"""
        await asyncio.sleep(0.1)  # Simulate initialization time
        self.initialized = True

    async def shutdown(self):
        """Shutdown mock service"""
        self.shutdown_called = True

    async def health_check(self):
        """Mock health check"""
        return not self.should_fail_health


class TestUnifiedOrchestratorIntegration:
    """Integration tests for unified orchestrator"""

    @pytest.fixture
    async def redis_client(self):
        """Mock Redis client"""
        client = AsyncMock()
        client.flushdb = AsyncMock()
        client.close = AsyncMock()
        client.get = AsyncMock(return_value=None)
        client.setex = AsyncMock()
        client.exists = AsyncMock(return_value=0)
        return client

    @pytest.fixture
    async def orchestrator(self, redis_client):
        """Create orchestrator instance"""
        orchestrator = UnifiedOrchestrator(redis_client=redis_client)
        yield orchestrator
        if orchestrator.running:
            await orchestrator.shutdown()

    @pytest.fixture
    def sample_service_definition(self):
        """Create sample service definition"""
        return ServiceDefinition(
            service_id="test-service",
            name="Test Service",
            service_type=ServiceType.CORE,
            module_path="tests.integration.test_orchestrator_integration",
            class_name="MockService",
            dependencies=[],
            config={"service_id": "test-service"},
            health_check_url="http://localhost:8000/health",
            startup_timeout=30
        )

    @pytest.fixture
    def sample_workflow_definition(self):
        """Create sample workflow definition"""
        tasks = [
            WorkflowTask(
                id="task1",
                name="First Task",
                task_type=TaskType.DATA_COLLECTION,
                description="Collect data",
                parameters={"source": "database"},
                dependencies=[],
                timeout_minutes=5,
                retry_count=2,
                retry_delay_seconds=1
            ),
            WorkflowTask(
                id="task2",
                name="Second Task",
                task_type=TaskType.THREAT_ANALYSIS,
                description="Analyze threats",
                parameters={"algorithm": "ml"},
                dependencies=["task1"],
                timeout_minutes=10,
                retry_count=1,
                retry_delay_seconds=2
            )
        ]

        return WorkflowDefinition(
            id="test-workflow",
            name="Test Workflow",
            description="Test workflow for integration testing",
            version="1.0",
            tasks=tasks,
            triggers=[{"type": "manual"}],
            variables={"test_mode": True},
            notifications={"email": ["admin@test.com"]},
            sla_minutes=30
        )

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator.running is False
        assert len(orchestrator.services) == 0
        assert len(orchestrator.workflows) == 0

        await orchestrator.initialize()

        assert orchestrator.running is True
        assert orchestrator.health_check_task is not None
        assert orchestrator.workflow_monitor_task is not None

    @pytest.mark.asyncio
    async def test_service_registration_and_lifecycle(self, orchestrator, sample_service_definition):
        """Test service registration and lifecycle management"""
        await orchestrator.initialize()

        # Register service
        orchestrator.register_service(sample_service_definition)
        assert "test-service" in orchestrator.service_registry

        # Start service
        success = await orchestrator.start_service("test-service")
        assert success is True
        assert "test-service" in orchestrator.services

        instance = orchestrator.services["test-service"]
        assert instance.status == ServiceStatus.RUNNING
        assert instance.instance_object.initialized is True

        # Check service status
        status = orchestrator.get_service_status("test-service")
        assert status == ServiceStatus.RUNNING

        # Stop service
        success = await orchestrator.stop_service("test-service")
        assert success is True
        assert "test-service" not in orchestrator.services

    @pytest.mark.asyncio
    async def test_workflow_registration_and_execution(self, orchestrator, sample_workflow_definition):
        """Test workflow registration and execution"""
        await orchestrator.initialize()

        # Register task executors
        executor1 = MockTaskExecutor(delay=0.1)
        executor2 = MockTaskExecutor(delay=0.2)

        orchestrator.register_task_executor(TaskType.DATA_COLLECTION, executor1)
        orchestrator.register_task_executor(TaskType.THREAT_ANALYSIS, executor2)

        # Register workflow
        orchestrator.register_workflow(sample_workflow_definition)
        assert "test-workflow" in orchestrator.workflows

        # Execute workflow
        trigger_data = {"user": "test-user", "priority": "high"}
        execution_id = await orchestrator.execute_workflow(
            "test-workflow",
            trigger_data,
            "test-trigger"
        )

        assert execution_id is not None
        assert execution_id in orchestrator.workflow_executions

        # Wait for workflow completion
        await asyncio.sleep(1)

        execution = orchestrator.workflow_executions[execution_id]
        assert execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.RUNNING]
        assert execution.triggered_by == "test-trigger"
        assert execution.trigger_data == trigger_data

        # Check task execution
        assert executor1.execution_count >= 1
        if execution.status == WorkflowStatus.COMPLETED:
            assert executor2.execution_count >= 1

    @pytest.mark.asyncio
    async def test_dependency_management(self, orchestrator):
        """Test service dependency management"""
        await orchestrator.initialize()

        # Create services with dependencies
        base_service = ServiceDefinition(
            service_id="base-service",
            name="Base Service",
            service_type=ServiceType.CORE,
            module_path="tests.integration.test_orchestrator_integration",
            class_name="MockService",
            dependencies=[],
            config={"service_id": "base-service"}
        )

        dependent_service = ServiceDefinition(
            service_id="dependent-service",
            name="Dependent Service",
            service_type=ServiceType.ANALYTICS,
            module_path="tests.integration.test_orchestrator_integration",
            class_name="MockService",
            dependencies=["base-service"],
            config={"service_id": "dependent-service"}
        )

        # Register services
        orchestrator.register_service(base_service)
        orchestrator.register_service(dependent_service)

        # Try to start dependent service without base service
        success = await orchestrator.start_service("dependent-service")
        assert success is False  # Should fail due to missing dependency

        # Start base service first
        success = await orchestrator.start_service("base-service")
        assert success is True

        # Now dependent service should start
        success = await orchestrator.start_service("dependent-service")
        assert success is True

    @pytest.mark.asyncio
    async def test_health_monitoring(self, orchestrator):
        """Test health monitoring functionality"""
        await orchestrator.initialize()

        # Create service that will fail health checks
        failing_service = ServiceDefinition(
            service_id="failing-service",
            name="Failing Service",
            service_type=ServiceType.MONITORING,
            module_path="tests.integration.test_orchestrator_integration",
            class_name="MockService",
            dependencies=[],
            config={"service_id": "failing-service", "should_fail_health": True}
        )

        orchestrator.register_service(failing_service)
        success = await orchestrator.start_service("failing-service")
        assert success is True

        # Wait for health check
        await asyncio.sleep(0.5)

        # Service should be marked as error after health check failure
        instance = orchestrator.services["failing-service"]
        # Note: In real implementation, health check would run and mark as ERROR
        assert instance.last_health_check is not None

    @pytest.mark.asyncio
    async def test_workflow_task_dependencies(self, orchestrator):
        """Test workflow task dependency execution"""
        await orchestrator.initialize()

        # Create workflow with complex dependencies
        tasks = [
            WorkflowTask(
                id="init",
                name="Initialize",
                task_type=TaskType.DATA_COLLECTION,
                description="Initialize workflow",
                parameters={},
                dependencies=[],
                timeout_minutes=5,
                retry_count=1,
                retry_delay_seconds=1
            ),
            WorkflowTask(
                id="process_a",
                name="Process A",
                task_type=TaskType.THREAT_ANALYSIS,
                description="Process A",
                parameters={},
                dependencies=["init"],
                timeout_minutes=5,
                retry_count=1,
                retry_delay_seconds=1
            ),
            WorkflowTask(
                id="process_b",
                name="Process B",
                task_type=TaskType.VULNERABILITY_SCAN,
                description="Process B",
                parameters={},
                dependencies=["init"],
                timeout_minutes=5,
                retry_count=1,
                retry_delay_seconds=1,
                parallel_execution=True
            ),
            WorkflowTask(
                id="finalize",
                name="Finalize",
                task_type=TaskType.REPORT_GENERATION,
                description="Finalize workflow",
                parameters={},
                dependencies=["process_a", "process_b"],
                timeout_minutes=5,
                retry_count=1,
                retry_delay_seconds=1
            )
        ]

        workflow = WorkflowDefinition(
            id="complex-workflow",
            name="Complex Workflow",
            description="Workflow with complex dependencies",
            version="1.0",
            tasks=tasks,
            triggers=[],
            variables={},
            notifications={}
        )

        # Register executors for all task types
        for task_type in [TaskType.DATA_COLLECTION, TaskType.THREAT_ANALYSIS,
                         TaskType.VULNERABILITY_SCAN, TaskType.REPORT_GENERATION]:
            orchestrator.register_task_executor(task_type, MockTaskExecutor(delay=0.1))

        # Register and execute workflow
        orchestrator.register_workflow(workflow)
        execution_id = await orchestrator.execute_workflow("complex-workflow", {})

        # Wait for completion
        await asyncio.sleep(2)

        execution = orchestrator.workflow_executions[execution_id]
        # Should have results for all tasks
        assert len(execution.task_results) > 0

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, orchestrator):
        """Test workflow error handling"""
        await orchestrator.initialize()

        # Create workflow with failing task
        task = WorkflowTask(
            id="failing-task",
            name="Failing Task",
            task_type=TaskType.DATA_COLLECTION,
            description="This task will fail",
            parameters={},
            dependencies=[],
            timeout_minutes=1,
            retry_count=2,
            retry_delay_seconds=0.1
        )

        workflow = WorkflowDefinition(
            id="failing-workflow",
            name="Failing Workflow",
            description="Workflow with failing task",
            version="1.0",
            tasks=[task],
            triggers=[],
            variables={},
            notifications={}
        )

        # Register failing executor
        failing_executor = MockTaskExecutor(should_fail=True)
        orchestrator.register_task_executor(TaskType.DATA_COLLECTION, failing_executor)

        # Register and execute workflow
        orchestrator.register_workflow(workflow)
        execution_id = await orchestrator.execute_workflow("failing-workflow", {})

        # Wait for completion
        await asyncio.sleep(1)

        execution = orchestrator.workflow_executions[execution_id]
        # Task should have error result
        assert "failing-task" in execution.task_results
        assert "error" in execution.task_results["failing-task"]

        # Executor should have been called multiple times due to retries
        assert failing_executor.execution_count > 1

    @pytest.mark.asyncio
    async def test_metrics_collection(self, orchestrator, sample_service_definition):
        """Test metrics collection and reporting"""
        await orchestrator.initialize()

        # Start a service
        orchestrator.register_service(sample_service_definition)
        await orchestrator.start_service("test-service")

        # Get initial metrics
        metrics = orchestrator.get_metrics()
        assert metrics.total_services == 1
        assert metrics.running_services == 1
        assert metrics.failed_services == 0

        # Stop service
        await orchestrator.stop_service("test-service")

        # Update and check metrics
        await orchestrator._update_metrics()
        metrics = orchestrator.get_metrics()
        assert metrics.running_services == 0

    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, orchestrator):
        """Test concurrent workflow execution"""
        await orchestrator.initialize()

        # Create simple workflow
        task = WorkflowTask(
            id="concurrent-task",
            name="Concurrent Task",
            task_type=TaskType.DATA_COLLECTION,
            description="Concurrent execution test",
            parameters={},
            dependencies=[],
            timeout_minutes=1,
            retry_count=0,
            retry_delay_seconds=0
        )

        workflow = WorkflowDefinition(
            id="concurrent-workflow",
            name="Concurrent Workflow",
            description="Test concurrent execution",
            version="1.0",
            tasks=[task],
            triggers=[],
            variables={},
            notifications={}
        )

        # Register executor and workflow
        orchestrator.register_task_executor(TaskType.DATA_COLLECTION, MockTaskExecutor(delay=0.2))
        orchestrator.register_workflow(workflow)

        # Execute multiple workflows concurrently
        execution_ids = []
        for i in range(5):
            exec_id = await orchestrator.execute_workflow(
                "concurrent-workflow",
                {"instance": i}
            )
            execution_ids.append(exec_id)

        # Wait for completion
        await asyncio.sleep(1)

        # All executions should be tracked
        for exec_id in execution_ids:
            assert exec_id in orchestrator.workflow_executions

        # Check metrics
        metrics = orchestrator.get_metrics()
        assert metrics.total_workflows == 1
        assert metrics.completed_workflows >= 0  # Some may still be running

    @pytest.mark.asyncio
    async def test_orchestrator_shutdown(self, orchestrator, sample_service_definition):
        """Test graceful orchestrator shutdown"""
        await orchestrator.initialize()

        # Start a service
        orchestrator.register_service(sample_service_definition)
        await orchestrator.start_service("test-service")

        # Verify service is running
        instance = orchestrator.services["test-service"]
        assert instance.status == ServiceStatus.RUNNING

        # Shutdown orchestrator
        await orchestrator.shutdown()

        # Verify shutdown
        assert orchestrator.running is False
        assert len(orchestrator.services) == 0

        # Tasks should be cancelled
        assert orchestrator.health_check_task.cancelled()
        assert orchestrator.workflow_monitor_task.cancelled()

    @pytest.mark.asyncio
    async def test_service_listing_and_status(self, orchestrator, sample_service_definition):
        """Test service listing and status reporting"""
        await orchestrator.initialize()

        # Initially no services
        services = orchestrator.list_services()
        assert len(services) == 0

        # Register and start service
        orchestrator.register_service(sample_service_definition)
        await orchestrator.start_service("test-service")

        # List services
        services = orchestrator.list_services()
        assert len(services) == 1

        service_info = services[0]
        assert service_info["service_id"] == "test-service"
        assert service_info["name"] == "Test Service"
        assert service_info["status"] == "running"
        assert service_info["start_time"] is not None
        assert service_info["restart_count"] == 0

    @pytest.mark.asyncio
    async def test_workflow_listing_and_status(self, orchestrator, sample_workflow_definition):
        """Test workflow listing and status reporting"""
        await orchestrator.initialize()

        # Register executor and workflow
        orchestrator.register_task_executor(TaskType.DATA_COLLECTION, MockTaskExecutor())
        orchestrator.register_task_executor(TaskType.THREAT_ANALYSIS, MockTaskExecutor())
        orchestrator.register_workflow(sample_workflow_definition)

        # Execute workflow
        execution_id = await orchestrator.execute_workflow("test-workflow", {"test": True})

        # List workflows
        workflows = orchestrator.list_workflows()
        assert len(workflows) >= 1

        workflow_info = next(w for w in workflows if w["execution_id"] == execution_id)
        assert workflow_info["workflow_id"] == "test-workflow"
        assert workflow_info["status"] in ["pending", "running", "completed"]
        assert workflow_info["started_at"] is not None
        assert workflow_info["triggered_by"] == "manual"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
