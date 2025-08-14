from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import redis.asyncio as redis
import json
from dataclasses import asdict
from src.orchestrator.core.workflow_engine import WorkflowDefinition, WorkflowExecution, WorkflowStatus


class RedisStorage:
    """Redis-based storage implementation for workflow definitions and executions"""

    def __init__(self, redis_url: str, key_prefix: str = "xorb:orchestrator:", default_ttl: int = 86400):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self.redis = None

    async def initialize(self):
        """Initialize the Redis connection"""
        self.redis = redis.from_url(self.redis_url)

    async def store_workflow(self, workflow: WorkflowDefinition) -> bool:
        """Store a workflow definition"""
        try:
            key = f"{self.key_prefix}workflow:{workflow.id}"
            value = json.dumps(asdict(workflow))

            # Store workflow definition
            await self.redis.set(key, value, ex=self.default_ttl)

            # Store version history
            version_key = f"{key}:versions"
            await self.redis.hset(version_key, workflow.version, value)

            return True
        except Exception as e:
            print(f"Error storing workflow: {e}")
            return False

    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get a workflow definition by ID"""
        try:
            key = f"{self.key_prefix}workflow:{workflow_id}"
            data = await self.redis.get(key)

            if data:
                return WorkflowDefinition(**json.loads(data))
            return None
        except Exception as e:
            print(f"Error getting workflow: {e}")
            return None

    async def store_execution(self, execution: WorkflowExecution) -> bool:
        """Store a workflow execution record"""
        try:
            # Store execution metadata
            execution_key = f"{self.key_prefix}execution:{execution.id}"
            execution_value = json.dumps(asdict(execution))
            await self.redis.set(execution_key, execution_value, ex=self.default_ttl * 2)  # Longer TTL for executions

            # Store execution in workflow index
            workflow_executions_key = f"{self.key_prefix}workflow:{execution.workflow_id}:executions"
            await self.redis.zadd(workflow_executions_key, {execution.id: execution.started_at.timestamp()})

            # Store execution status index
            status_key = f"{self.key_prefix}execution:status:{execution.status.value}"
            await self.redis.sadd(status_key, execution.id)

            return True
        except Exception as e:
            print(f"Error storing execution: {e}")
            return False

    async def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get a workflow execution by ID"""
        try:
            key = f"{self.key_prefix}execution:{execution_id}"
            data = await self.redis.get(key)

            if data:
                return WorkflowExecution(**json.loads(data))
            return None
        except Exception as e:
            print(f"Error getting execution: {e}")
            return None

    async def update_execution_status(self, execution_id: str, status: WorkflowStatus, error_message: Optional[str] = None) -> bool:
        """Update the status of a workflow execution"""
        try:
            key = f"{self.key_prefix}execution:{execution_id}"

            # Get current execution data
            data = await self.redis.get(key)
            if not data:
                return False

            execution_data = json.loads(data)

            # Update status and error message
            execution_data['status'] = status.value
            if error_message:
                execution_data['error_message'] = error_message

            # Update completed_at if execution is completing
            if status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
                execution_data['completed_at'] = datetime.now().isoformat()

            # Store updated execution
            await self.redis.set(key, json.dumps(execution_data), ex=self.default_ttl * 2)

            # Update status index
            old_status = execution_data.get('status', 'pending')
            status_key_old = f"{self.key_prefix}execution:status:{old_status}"
            status_key_new = f"{self.key_prefix}execution:status:{status.value}"
            await self.redis.srem(status_key_old, execution_id)
            await self.redis.sadd(status_key_new, execution_id)

            return True
        except Exception as e:
            print(f"Error updating execution status: {e}")
            return False

    async def store_task_result(self, execution_id: str, task_id: str, result: Dict[str, Any]) -> bool:
        """Store the result of a task execution"""
        try:
            key = f"{self.key_prefix}execution:{execution_id}:task:{task_id}"
            value = json.dumps(result)

            await self.redis.set(key, value, ex=self.default_ttl)

            # Update execution record
            execution = await self.get_execution(execution_id)
            if execution:
                execution.task_results[task_id] = result
                await self.store_execution(execution)

            return True
        except Exception as e:
            print(f"Error storing task result: {e}")
            return False

    async def get_task_result(self, execution_id: str, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the result of a task execution"""
        try:
            key = f"{self.key_prefix}execution:{execution_id}:task:{task_id}"
            data = await self.redis.get(key)

            if data:
                return json.loads(data)
            return None
        except Exception as e:
            print(f"Error getting task result: {e}")
            return None

    async def get_workflow_executions(self, workflow_id: str, limit: int = 100, offset: int = 0) -> List[WorkflowExecution]:
        """Get executions for a specific workflow"""
        try:
            key = f"{self.key_prefix}workflow:{workflow_id}:executions"
            execution_ids = await self.redis.zrange(key, offset, offset + limit - 1)

            executions = []
            for execution_id in execution_ids:
                execution = await self.get_execution(execution_id.decode())
                if execution:
                    executions.append(execution)

            return executions
        except Exception as e:
            print(f"Error getting workflow executions: {e}")
            return []

    async def get_executions_by_status(self, status: WorkflowStatus, limit: int = 100, offset: int = 0) -> List[WorkflowExecution]:
        """Get executions by status"""
        try:
            key = f"{self.key_prefix}execution:status:{status.value}"
            execution_ids = await self.redis.sscan(key, cursor=offset, count=limit)[1]

            executions = []
            for execution_id in execution_ids:
                execution = await self.get_execution(execution_id.decode())
                if execution:
                    executions.append(execution)

            return executions
        except Exception as e:
            print(f"Error getting executions by status: {e}")
            return []

    async def get_active_executions(self, limit: int = 100) -> List[WorkflowExecution]:
        """Get active (running/pending) executions"""
        try:
            # Get all active status
            active_statuses = [WorkflowStatus.PENDING.value, WorkflowStatus.RUNNING.value]

            # Get executions for each status
            executions = []
            for status in active_statuses:
                key = f"{self.key_prefix}execution:status:{status}"
                execution_ids = await self.redis.sscan(key, cursor=0, count=limit)[1]

                for execution_id in execution_ids:
                    execution = await self.get_execution(execution_id.decode())
                    if execution:
                        executions.append(execution)

            return executions
        except Exception as e:
            print(f"Error getting active executions: {e}")
            return []

    async def store_workflow_metrics(self, workflow_id: str, metrics: Dict[str, Any]) -> bool:
        """Store workflow execution metrics"""
        try:
            key = f"{self.key_prefix}workflow:{workflow_id}:metrics"
            value = json.dumps(metrics)

            await self.redis.set(key, value, ex=self.default_ttl)
            return True
        except Exception as e:
            print(f"Error storing workflow metrics: {e}")
            return False

    async def get_workflow_metrics(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow execution metrics"""
        try:
            key = f"{self.key_prefix}workflow:{workflow_id}:metrics"
            data = await self.redis.get(key)

            if data:
                return json.loads(data)
            return None
        except Exception as e:
            print(f"Error getting workflow metrics: {e}")
            return None

    async def store_variable(self, execution_id: str, key: str, value: Any) -> bool:
        """Store a variable for an execution"""
        try:
            variable_key = f"{self.key_prefix}execution:{execution_id}:variable:{key}"
            variable_value = json.dumps(value)

            await self.redis.set(variable_key, variable_value, ex=self.default_ttl * 2)

            # Update execution record
            execution = await self.get_execution(execution_id)
            if execution:
                execution.variables[key] = value
                await self.store_execution(execution)

            return True
        except Exception as e:
            print(f"Error storing variable: {e}")
            return False

    async def get_variable(self, execution_id: str, key: str) -> Optional[Any]:
        """Get a variable for an execution"""
        try:
            variable_key = f"{self.key_prefix}execution:{execution_id}:variable:{key}"
            data = await self.redis.get(variable_key)

            if data:
                return json.loads(data)
            return None
        except Exception as e:
            print(f"Error getting variable: {e}")
            return None

    async def create_lock(self, key: str, ttl: int = 60) -> bool:
        """Create a distributed lock"""
        try:
            lock_key = f"{self.key_prefix}lock:{key}"

            # Use Redis SETNX to create a lock
            result = await self.redis.setnx(lock_key, "locked")
            if result == 1:
                await self.redis.expire(lock_key, ttl)
                return True
            return False
        except Exception as e:
            print(f"Error creating lock: {e}")
            return False

    async def release_lock(self, key: str) -> bool:
        """Release a distributed lock"""
        try:
            lock_key = f"{self.key_prefix}lock:{key}"

            # Use Redis DEL to release the lock
            result = await self.redis.delete(lock_key)
            return result == 1
        except Exception as e:
            print(f"Error releasing lock: {e}")
            return False

    async def shutdown(self):
        """Shutdown the storage layer"""
        if self.redis:
            await self.redis.close()
