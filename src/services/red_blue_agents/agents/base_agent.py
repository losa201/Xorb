"""
Base Agent Class for Red/Blue Team Framework

Provides common functionality for all specialized agents including telemetry collection,
capability validation, and execution framework.
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Agent classification types"""
    RED_TEAM = "red_team"
    BLUE_TEAM = "blue_team"


class TaskStatus(Enum):
    """Task execution statuses"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskResult:
    """Result of a task execution"""
    task_id: str
    technique_id: str
    status: TaskStatus
    output: Dict[str, Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: Optional[datetime] = None
    telemetry: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.output is None:
            self.output = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.telemetry is None:
            self.telemetry = {}


@dataclass
class AgentConfiguration:
    """Configuration for an agent instance"""
    agent_id: str
    agent_type: str
    mission_id: str
    environment: str
    sandbox_id: Optional[str] = None
    capabilities: List[str] = None
    constraints: Dict[str, Any] = None
    telemetry_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.constraints is None:
            self.constraints = {}
        if self.telemetry_config is None:
            self.telemetry_config = {"enabled": True, "level": "info"}


class BaseAgent(ABC):
    """
    Abstract base class for all red and blue team agents.
    
    Provides common functionality including:
    - Task execution framework
    - Telemetry collection
    - Capability validation
    - Error handling and logging
    - Communication protocols
    """
    
    def __init__(self, config: AgentConfiguration):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{config.agent_id}")
        
        # Execution state
        self.is_running = False
        self.current_task: Optional[str] = None
        self.task_history: List[TaskResult] = []
        
        # Capability registry and telemetry hooks
        self.capability_registry = None
        self.telemetry_collector = None
        
        # Custom technique implementations
        self.technique_handlers: Dict[str, Callable] = {}
        
    async def initialize(self, capability_registry=None, telemetry_collector=None):
        """Initialize the agent with external dependencies"""
        self.capability_registry = capability_registry
        self.telemetry_collector = telemetry_collector
        
        # Register technique handlers
        await self._register_technique_handlers()
        
        self.logger.info(f"Agent {self.config.agent_id} initialized with {len(self.technique_handlers)} techniques")
        
    @abstractmethod
    async def _register_technique_handlers(self):
        """Register technique handlers specific to this agent type"""
        pass
        
    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return the agent classification type"""
        pass
        
    @property
    @abstractmethod
    def supported_categories(self) -> List[str]:
        """Return list of MITRE ATT&CK categories this agent supports"""
        pass
        
    async def execute_task(self, task_id: str, technique_id: str, parameters: Dict[str, Any]) -> TaskResult:
        """Execute a single task with the given technique and parameters"""
        start_time = datetime.utcnow()
        self.current_task = task_id
        
        self.logger.info(f"Executing task {task_id}: {technique_id}")
        
        try:
            # Validate technique is supported
            if technique_id not in self.technique_handlers:
                raise ValueError(f"Technique {technique_id} not supported by {self.__class__.__name__}")
                
            # Validate parameters if capability registry available
            if self.capability_registry:
                validated_params = await self.capability_registry.validate_technique_parameters(
                    technique_id, parameters
                )
            else:
                validated_params = parameters
                
            # Collect pre-execution telemetry
            if self.telemetry_collector:
                await self.telemetry_collector.collect_agent_event(
                    agent_id=self.config.agent_id,
                    event_type="task_started",
                    data={
                        "task_id": task_id,
                        "technique_id": technique_id,
                        "parameters": validated_params
                    }
                )
                
            # Execute the technique
            handler = self.technique_handlers[technique_id]
            output = await handler(validated_params)
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create successful result
            result = TaskResult(
                task_id=task_id,
                technique_id=technique_id,
                status=TaskStatus.COMPLETED,
                output=output,
                execution_time=execution_time,
                timestamp=start_time
            )
            
            # Collect success telemetry
            if self.telemetry_collector:
                await self.telemetry_collector.collect_agent_event(
                    agent_id=self.config.agent_id,
                    event_type="task_completed",
                    data={
                        "task_id": task_id,
                        "technique_id": technique_id,
                        "execution_time": execution_time,
                        "output_size": len(json.dumps(output))
                    }
                )
                
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create failed result
            result = TaskResult(
                task_id=task_id,
                technique_id=technique_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                timestamp=start_time
            )
            
            # Collect failure telemetry
            if self.telemetry_collector:
                await self.telemetry_collector.collect_agent_event(
                    agent_id=self.config.agent_id,
                    event_type="task_failed",
                    data={
                        "task_id": task_id,
                        "technique_id": technique_id,
                        "error": str(e),
                        "execution_time": execution_time
                    }
                )
                
            self.logger.error(f"Task {task_id} failed: {e}")
            
        finally:
            self.current_task = None
            self.task_history.append(result)
            
        return result
        
    async def execute_task_sequence(self, tasks: List[Dict[str, Any]]) -> List[TaskResult]:
        """Execute a sequence of tasks in order"""
        results = []
        
        for task_config in tasks:
            task_id = task_config.get("task_id", str(uuid.uuid4()))
            technique_id = task_config["technique_id"]
            parameters = task_config.get("parameters", {})
            
            result = await self.execute_task(task_id, technique_id, parameters)
            results.append(result)
            
            # Stop on failure if configured
            if result.status == TaskStatus.FAILED and task_config.get("stop_on_failure", False):
                break
                
        return results
        
    async def get_capabilities(self) -> List[str]:
        """Get list of techniques this agent can execute"""
        return list(self.technique_handlers.keys())
        
    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.config.agent_id,
            "agent_type": self.config.agent_type,
            "is_running": self.is_running,
            "current_task": self.current_task,
            "total_tasks_executed": len(self.task_history),
            "successful_tasks": len([r for r in self.task_history if r.status == TaskStatus.COMPLETED]),
            "failed_tasks": len([r for r in self.task_history if r.status == TaskStatus.FAILED]),
            "supported_techniques": len(self.technique_handlers),
            "last_activity": self.task_history[-1].timestamp.isoformat() if self.task_history else None
        }
        
    async def get_task_history(self, limit: Optional[int] = None) -> List[TaskResult]:
        """Get agent task execution history"""
        history = self.task_history
        if limit:
            history = history[-limit:]
        return history
        
    # Common utility methods for technique implementations
    
    async def _execute_command(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute a system command safely"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            
            return {
                "returncode": process.returncode,
                "stdout": stdout.decode('utf-8', errors='ignore'),
                "stderr": stderr.decode('utf-8', errors='ignore'),
                "command": command
            }
            
        except asyncio.TimeoutError:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "command": command
            }
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "command": command
            }
            
    async def _make_http_request(self, url: str, method: str = "GET", 
                               headers: Optional[Dict[str, str]] = None,
                               data: Optional[Dict[str, Any]] = None,
                               timeout: int = 30) -> Dict[str, Any]:
        """Make HTTP requests safely"""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data
                ) as response:
                    content = await response.text()
                    
                    return {
                        "status": response.status,
                        "headers": dict(response.headers),
                        "content": content,
                        "url": str(response.url),
                        "method": method
                    }
                    
        except Exception as e:
            return {
                "status": -1,
                "headers": {},
                "content": "",
                "error": str(e),
                "url": url,
                "method": method
            }
            
    async def _scan_network_port(self, host: str, port: int, timeout: int = 5) -> bool:
        """Check if a network port is open"""
        try:
            future = asyncio.open_connection(host, port)
            reader, writer = await asyncio.wait_for(future, timeout=timeout)
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False
            
    async def _log_technique_execution(self, technique_id: str, parameters: Dict[str, Any], 
                                     result: Dict[str, Any]):
        """Log technique execution for audit purposes"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": self.config.agent_id,
            "technique_id": technique_id,
            "parameters": parameters,
            "result_summary": {
                "success": result.get("success", False),
                "output_length": len(str(result.get("output", "")))
            }
        }
        
        self.logger.info(f"Technique executed: {json.dumps(log_entry)}")
        
        # Send to telemetry collector if available
        if self.telemetry_collector:
            await self.telemetry_collector.collect_technique_execution(
                agent_id=self.config.agent_id,
                technique_id=technique_id,
                parameters=parameters,
                result=result
            )
            
    async def shutdown(self):
        """Shutdown the agent gracefully"""
        self.is_running = False
        if self.current_task:
            self.logger.warning(f"Shutting down with active task: {self.current_task}")
            
        self.logger.info(f"Agent {self.config.agent_id} shutdown complete")
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(agent_id={self.config.agent_id}, mission_id={self.config.mission_id})"