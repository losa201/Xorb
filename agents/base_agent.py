#!/usr/bin/env python3

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running" 
    PAUSED = "paused"
    ERROR = "error"
    TERMINATED = "terminated"


class AgentType(str, Enum):
    RECONNAISSANCE = "reconnaissance"
    WEB_CRAWLER = "web_crawler"
    VULNERABILITY_SCANNER = "vulnerability_scanner"
    EXPLOIT = "exploit"
    POST_EXPLOIT = "post_exploit"
    INTELLIGENCE_GATHERER = "intelligence_gatherer"


@dataclass
class AgentCapability:
    name: str
    description: str
    required_tools: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    enabled: bool = True


@dataclass
class AgentTask:
    task_id: str
    task_type: str
    target: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class AgentResult:
    task_id: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    def __init__(self, agent_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.config = config or {}
        
        self.status = AgentStatus.IDLE
        self.capabilities: List[AgentCapability] = []
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.results: Dict[str, AgentResult] = {}
        
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.total_tasks_completed = 0
        self.total_errors = 0
        
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{self.agent_id[:8]}]")
        
        self._initialize_capabilities()

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        pass

    @abstractmethod
    async def _execute_task(self, task: AgentTask) -> AgentResult:
        pass

    @abstractmethod
    def _initialize_capabilities(self):
        pass

    async def start(self):
        if self._running:
            self.logger.warning("Agent is already running")
            return
        
        self._running = True
        self.status = AgentStatus.IDLE
        self._worker_task = asyncio.create_task(self._worker_loop())
        
        await self._on_start()
        self.logger.info(f"Agent {self.agent_id} started")

    async def stop(self):
        if not self._running:
            return
        
        self._running = False
        self.status = AgentStatus.TERMINATED
        self._shutdown_event.set()
        
        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=30.0)
            except asyncio.TimeoutError:
                self._worker_task.cancel()
                self.logger.warning("Worker task cancelled due to timeout")
        
        await self._on_stop()
        self.logger.info(f"Agent {self.agent_id} stopped")

    async def pause(self):
        if self.status == AgentStatus.RUNNING:
            self.status = AgentStatus.PAUSED
            self.logger.info(f"Agent {self.agent_id} paused")

    async def resume(self):
        if self.status == AgentStatus.PAUSED:
            self.status = AgentStatus.IDLE
            self.logger.info(f"Agent {self.agent_id} resumed")

    async def add_task(self, task: AgentTask) -> bool:
        try:
            await self.task_queue.put(task)
            self.logger.debug(f"Added task {task.task_id} to queue")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add task {task.task_id}: {e}")
            return False

    async def get_result(self, task_id: str, timeout: float = 30.0) -> Optional[AgentResult]:
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            if task_id in self.results:
                return self.results[task_id]
            await asyncio.sleep(0.1)
        
        return None

    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "status": self.status.value,
            "queue_size": self.task_queue.qsize(),
            "completed_tasks": self.total_tasks_completed,
            "total_errors": self.total_errors,
            "uptime_seconds": (datetime.utcnow() - self.created_at).total_seconds(),
            "last_activity": self.last_activity.isoformat(),
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "success_rate": cap.success_rate,
                    "avg_execution_time": cap.avg_execution_time,
                    "enabled": cap.enabled
                } for cap in self.capabilities
            ]
        }

    def has_capability(self, capability_name: str) -> bool:
        return any(cap.name == capability_name and cap.enabled for cap in self.capabilities)

    async def update_capability_stats(self, capability_name: str, success: bool, execution_time: float):
        for cap in self.capabilities:
            if cap.name == capability_name:
                if cap.success_rate == 0:
                    cap.success_rate = 1.0 if success else 0.0
                else:
                    total_attempts = max(1, self.total_tasks_completed)
                    cap.success_rate = ((cap.success_rate * (total_attempts - 1)) + (1 if success else 0)) / total_attempts
                
                if cap.avg_execution_time == 0:
                    cap.avg_execution_time = execution_time
                else:
                    cap.avg_execution_time = (cap.avg_execution_time * 0.8) + (execution_time * 0.2)
                break

    async def _worker_loop(self):
        self.logger.debug("Worker loop started")
        
        while self._running:
            try:
                if self.status == AgentStatus.PAUSED:
                    await asyncio.sleep(1)
                    continue
                
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                    await self._process_task(task)
                except asyncio.TimeoutError:
                    continue
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in worker loop: {e}")
                self.status = AgentStatus.ERROR
                await asyncio.sleep(5)
                if self.status == AgentStatus.ERROR:
                    self.status = AgentStatus.IDLE

    async def _process_task(self, task: AgentTask):
        self.status = AgentStatus.RUNNING
        self.last_activity = datetime.utcnow()
        start_time = datetime.utcnow()
        
        self.logger.info(f"Processing task {task.task_id}: {task.task_type}")
        
        try:
            result = await self._execute_task(task)
            result.execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            self.results[task.task_id] = result
            
            if result.success:
                self.total_tasks_completed += 1
                self.logger.debug(f"Task {task.task_id} completed successfully")
            else:
                self.total_errors += 1
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    await self.add_task(task)
                    self.logger.info(f"Task {task.task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
                else:
                    self.logger.error(f"Task {task.task_id} failed permanently after {task.max_retries} retries")
            
            # Update capability stats
            capability_used = self._get_capability_for_task(task.task_type)
            if capability_used:
                await self.update_capability_stats(capability_used.name, result.success, result.execution_time)
            
            await self._on_task_completed(task, result)
            
        except Exception as e:
            self.logger.error(f"Error processing task {task.task_id}: {e}")
            
            error_result = AgentResult(
                task_id=task.task_id,
                success=False,
                errors=[str(e)],
                execution_time=(datetime.utcnow() - start_time).total_seconds()
            )
            
            self.results[task.task_id] = error_result
            self.total_errors += 1
            
        finally:
            self.status = AgentStatus.IDLE
            self.task_queue.task_done()

    def _get_capability_for_task(self, task_type: str) -> Optional[AgentCapability]:
        for cap in self.capabilities:
            if cap.name == task_type or task_type in cap.name:
                return cap
        return None

    async def _validate_task(self, task: AgentTask) -> bool:
        if not task.target:
            self.logger.error(f"Task {task.task_id} has no target")
            return False
        
        if task.deadline and task.deadline < datetime.utcnow():
            self.logger.error(f"Task {task.task_id} has expired deadline")
            return False
        
        required_capability = self._get_capability_for_task(task.task_type)
        if not required_capability or not required_capability.enabled:
            self.logger.error(f"Task {task.task_id} requires unavailable capability: {task.task_type}")
            return False
        
        return True

    async def _on_start(self):
        """Override in subclasses for startup initialization"""
        pass

    async def _on_stop(self):
        """Override in subclasses for cleanup"""
        pass

    async def _on_task_completed(self, task: AgentTask, result: AgentResult):
        """Override in subclasses for post-task processing"""
        pass

    def cleanup_old_results(self, max_age_hours: int = 24):
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        old_results = []
        for task_id, result in self.results.items():
            # Assuming results have a timestamp - we'd need to add this to AgentResult
            old_results.append(task_id)
        
        for task_id in old_results:
            del self.results[task_id]
        
        if old_results:
            self.logger.debug(f"Cleaned up {len(old_results)} old results")

    async def health_check(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "healthy": self.status not in [AgentStatus.ERROR, AgentStatus.TERMINATED],
            "uptime": (datetime.utcnow() - self.created_at).total_seconds(),
            "queue_size": self.task_queue.qsize(),
            "success_rate": (self.total_tasks_completed / max(1, self.total_tasks_completed + self.total_errors)),
            "last_activity_age": (datetime.utcnow() - self.last_activity).total_seconds()
        }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id[:8]}, status={self.status.value})"

    def __repr__(self) -> str:
        return self.__str__()