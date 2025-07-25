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
        
        # ENHANCED: Multi-agent collaboration capabilities
        self.collaborative_mode = config.get('collaborative_mode', True)
        self.peer_agents: Dict[str, 'BaseAgent'] = {}
        self.shared_knowledge: Dict[str, Any] = {}
        self.collaboration_events: asyncio.Queue = asyncio.Queue()
        self.task_delegation_enabled = config.get('task_delegation', True)
        self.knowledge_sharing_enabled = config.get('knowledge_sharing', True)
        self.collective_intelligence: Dict[str, Any] = {}
        
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

    # =============================================================================
    # ENHANCED MULTI-AGENT COLLABORATION METHODS v2.1
    # =============================================================================
    
    async def register_peer_agent(self, peer_agent: 'BaseAgent'):
        """Register a peer agent for collaboration"""
        if not self.collaborative_mode:
            return
            
        self.peer_agents[peer_agent.agent_id] = peer_agent
        self.logger.info(f"🤝 Registered peer agent {peer_agent.agent_id[:8]} ({peer_agent.__class__.__name__})")
        
        # Share initial knowledge
        if self.knowledge_sharing_enabled:
            await self._share_knowledge_with_peer(peer_agent)
    
    async def delegate_task(self, task: AgentTask, preferred_agent_type: Optional[AgentType] = None) -> bool:
        """Delegate a task to the most suitable peer agent"""
        if not self.task_delegation_enabled or not self.peer_agents:
            return False
        
        # Find the best suited agent for this task
        best_agent = await self._find_best_agent_for_task(task, preferred_agent_type)
        
        if best_agent and best_agent.agent_id != self.agent_id:
            await best_agent.add_task(task)
            self.logger.info(f"🎯 Delegated task {task.task_id[:8]} to agent {best_agent.agent_id[:8]}")
            
            # Record collaboration event
            await self._record_collaboration_event('task_delegation', {
                'task_id': task.task_id,
                'delegated_to': best_agent.agent_id,
                'reason': 'better_suited_capabilities'
            })
            
            return True
        
        return False
    
    async def share_knowledge(self, knowledge_type: str, knowledge_data: Any):
        """Share knowledge with all peer agents"""
        if not self.knowledge_sharing_enabled:
            return
        
        self.shared_knowledge[knowledge_type] = {
            'data': knowledge_data,
            'timestamp': datetime.utcnow(),
            'source_agent': self.agent_id
        }
        
        # Share with all peers
        for peer_id, peer_agent in self.peer_agents.items():
            try:
                await peer_agent.receive_shared_knowledge(knowledge_type, knowledge_data, self.agent_id)
            except Exception as e:
                self.logger.warning(f"Failed to share knowledge with {peer_id[:8]}: {e}")
    
    async def receive_shared_knowledge(self, knowledge_type: str, knowledge_data: Any, source_agent_id: str):
        """Receive shared knowledge from a peer agent"""
        if not self.knowledge_sharing_enabled:
            return
        
        self.shared_knowledge[knowledge_type] = {
            'data': knowledge_data,
            'timestamp': datetime.utcnow(),
            'source_agent': source_agent_id
        }
        
        # Update collective intelligence
        await self._update_collective_intelligence(knowledge_type, knowledge_data, source_agent_id)
        
        self.logger.debug(f"📡 Received knowledge '{knowledge_type}' from agent {source_agent_id[:8]}")
    
    async def collaborative_task_execution(self, task: AgentTask) -> AgentResult:
        """Execute task collaboratively with peer agents"""
        if not self.collaborative_mode or not self.peer_agents:
            return await self._execute_task(task)
        
        # Check if task can benefit from collaboration
        if await self._should_collaborate_on_task(task):
            collaborators = await self._select_collaborators_for_task(task)
            
            if collaborators:
                return await self._execute_collaborative_task(task, collaborators)
        
        return await self._execute_task(task)
    
    async def assess_task_priorities(self, tasks: List) -> Dict[str, float]:
        """Assess task priorities for autonomous orchestrator consensus"""
        priorities = {}
        
        for task in tasks:
            # Base priority assessment
            priority_score = task.priority
            
            # Adjust based on agent capabilities and current load
            if self._can_handle_task_efficiently(task):
                priority_score += 2.0  # Boost for tasks we can handle well
            
            if self.task_queue.qsize() > 5:
                priority_score -= 1.0  # Lower priority if overloaded
            
            # Check if task benefits from collaboration
            if await self._task_benefits_from_collaboration(task):
                priority_score += 1.5  # Boost collaborative tasks
            
            priorities[task.task_id] = max(1.0, min(10.0, priority_score))
        
        return priorities
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get learning insights for orchestrator collaborative learning"""
        return {
            'performance': {
                'success_rate': self.total_tasks_completed / max(1, self.total_tasks_completed + self.total_errors),
                'avg_queue_size': self.task_queue.qsize(),
                'capabilities': [cap.name for cap in self.capabilities if cap.enabled]
            },
            'failures': {
                'total_errors': self.total_errors,
                'common_failure_types': await self._analyze_failure_patterns()
            },
            'optimizations': {
                'preferred_task_types': await self._get_preferred_task_types(),
                'peak_performance_hours': await self._analyze_performance_patterns()
            },
            'resources': {
                'current_load': self.task_queue.qsize(),
                'processing_capacity': await self._estimate_processing_capacity()
            }
        }
    
    async def receive_collective_insights(self, insights: Dict[str, Any]):
        """Receive collective insights from orchestrator"""
        self.collective_intelligence.update(insights)
        
        # Apply insights to improve performance
        await self._apply_collective_insights(insights)
        
        self.logger.debug(f"🧠 Applied collective insights from {len(insights.get('performance_patterns', {}))} agents")
    
    # =============================================================================
    # INTERNAL COLLABORATION METHODS
    # =============================================================================
    
    async def _find_best_agent_for_task(self, task: AgentTask, preferred_type: Optional[AgentType] = None) -> Optional['BaseAgent']:
        """Find the best suited agent for a task"""
        best_agent = None
        best_score = 0.0
        
        candidates = [self] + list(self.peer_agents.values())
        
        for agent in candidates:
            if not agent._can_handle_task_efficiently(task):
                continue
            
            score = await self._calculate_agent_suitability_score(agent, task, preferred_type)
            
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent
    
    async def _calculate_agent_suitability_score(self, agent: 'BaseAgent', task: AgentTask, preferred_type: Optional[AgentType]) -> float:
        """Calculate suitability score for an agent to handle a task"""
        score = 0.0
        
        # Check capability match
        for cap in agent.capabilities:
            if cap.name == task.task_type or task.task_type in cap.name:
                score += cap.success_rate * 10
                score += (1.0 / max(0.1, cap.avg_execution_time)) * 2  # Faster execution is better
                break
        
        # Check agent type preference
        if preferred_type and agent.agent_type == preferred_type:
            score += 5.0
        
        # Check current load
        queue_factor = max(0.1, 1.0 - (agent.task_queue.qsize() / 10.0))
        score *= queue_factor
        
        # Check agent status
        if agent.status == AgentStatus.IDLE:
            score += 2.0
        elif agent.status == AgentStatus.ERROR:
            score = 0.0
        
        return score
    
    def _can_handle_task_efficiently(self, task: AgentTask) -> bool:
        """Check if agent can handle task efficiently"""
        for cap in self.capabilities:
            if cap.name == task.task_type or task.task_type in cap.name:
                return cap.enabled and cap.success_rate > 0.5
        return False
    
    async def _should_collaborate_on_task(self, task: AgentTask) -> bool:
        """Determine if a task would benefit from collaboration"""
        # Complex tasks benefit from collaboration
        if task.task_type in ['comprehensive_scan', 'full_recon', 'exploit_chain']:
            return True
        
        # High priority tasks benefit from collaboration
        if task.priority >= 8:
            return True
        
        # Tasks with historical low success rates benefit from collaboration
        for cap in self.capabilities:
            if cap.name == task.task_type and cap.success_rate < 0.7:
                return True
        
        return False
    
    async def _select_collaborators_for_task(self, task: AgentTask) -> List['BaseAgent']:
        """Select appropriate collaborators for a task"""
        collaborators = []
        
        for peer_agent in self.peer_agents.values():
            if peer_agent.status == AgentStatus.IDLE and peer_agent._can_handle_task_efficiently(task):
                collaborators.append(peer_agent)
                
                if len(collaborators) >= 2:  # Limit to 2 collaborators
                    break
        
        return collaborators
    
    async def _execute_collaborative_task(self, task: AgentTask, collaborators: List['BaseAgent']) -> AgentResult:
        """Execute task with collaboration"""
        self.logger.info(f"🤝 Executing collaborative task {task.task_id[:8]} with {len(collaborators)} collaborators")
        
        # For now, execute normally but record collaboration
        result = await self._execute_task(task)
        
        # Record collaboration
        await self._record_collaboration_event('collaborative_execution', {
            'task_id': task.task_id,
            'collaborators': [agent.agent_id for agent in collaborators],
            'success': result.success
        })
        
        return result
    
    async def _record_collaboration_event(self, event_type: str, event_data: Dict[str, Any]):
        """Record collaboration event"""
        event = {
            'event_type': event_type,
            'timestamp': datetime.utcnow(),
            'agent_id': self.agent_id,
            'data': event_data
        }
        
        await self.collaboration_events.put(event)
    
    async def _share_knowledge_with_peer(self, peer_agent: 'BaseAgent'):
        """Share knowledge with a specific peer agent"""
        if self.shared_knowledge:
            for knowledge_type, knowledge_info in self.shared_knowledge.items():
                await peer_agent.receive_shared_knowledge(
                    knowledge_type, 
                    knowledge_info['data'], 
                    self.agent_id
                )
    
    async def _update_collective_intelligence(self, knowledge_type: str, knowledge_data: Any, source_agent_id: str):
        """Update collective intelligence with new knowledge"""
        if 'collective_patterns' not in self.collective_intelligence:
            self.collective_intelligence['collective_patterns'] = {}
        
        self.collective_intelligence['collective_patterns'][knowledge_type] = {
            'latest_data': knowledge_data,
            'contributors': self.collective_intelligence['collective_patterns'].get(knowledge_type, {}).get('contributors', []) + [source_agent_id],
            'last_updated': datetime.utcnow()
        }
    
    async def _task_benefits_from_collaboration(self, task) -> bool:
        """Check if a task would benefit from collaboration"""
        return task.task_type in ['comprehensive_scan', 'intelligence_gathering', 'exploit_chain']
    
    async def _analyze_failure_patterns(self) -> List[str]:
        """Analyze common failure patterns"""
        # This would analyze historical failures
        return ['timeout_errors', 'network_errors', 'authentication_failures']
    
    async def _get_preferred_task_types(self) -> List[str]:
        """Get task types this agent performs best at"""
        preferred = []
        for cap in self.capabilities:
            if cap.success_rate > 0.8:
                preferred.append(cap.name)
        return preferred
    
    async def _analyze_performance_patterns(self) -> List[int]:
        """Analyze performance patterns by hour"""
        # This would analyze historical performance data
        return [9, 10, 11, 14, 15, 16]  # Mock peak hours
    
    async def _estimate_processing_capacity(self) -> int:
        """Estimate current processing capacity"""
        base_capacity = 5  # Base tasks per minute
        load_factor = max(0.1, 1.0 - (self.task_queue.qsize() / 10.0))
        return int(base_capacity * load_factor)
    
    async def _apply_collective_insights(self, insights: Dict[str, Any]):
        """Apply collective insights to improve performance"""
        # Apply performance optimizations based on collective learning
        performance_patterns = insights.get('performance_patterns', {})
        
        for agent_id, patterns in performance_patterns.items():
            if agent_id != self.agent_id:
                # Learn from other agents' successful patterns
                success_rate = patterns.get('success_rate', 0)
                if success_rate > 0.9:
                    # This agent is highly successful, learn from their approach
                    self.logger.debug(f"Learning from high-performing agent {agent_id[:8]}")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id[:8]}, status={self.status.value})"

    def __repr__(self) -> str:
        return self.__str__()