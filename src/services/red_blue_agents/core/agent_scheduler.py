"""
Agent Scheduler for Red/Blue Agent Framework

Orchestrates the execution of specialized red and blue team agents across missions.
Manages agent lifecycles, capability validation, and mission coordination.
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import redis.asyncio as redis

from .capability_registry import CapabilityRegistry, Environment, TechniqueCategory
from .sandbox_orchestrator import SandboxOrchestrator, SandboxConfig, SandboxType, ResourceConstraints

logger = logging.getLogger(__name__)


class MissionStatus(Enum):
    """Mission execution statuses"""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStatus(Enum):
    """Agent execution statuses"""
    IDLE = "idle"
    ASSIGNED = "assigned"
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class AgentTask:
    """Individual task for an agent to execute"""
    task_id: str
    technique_id: str
    parameters: Dict[str, Any]
    dependencies: List[str] = None  # Other task IDs this depends on
    priority: int = 5  # 1-10, higher is more important
    timeout_seconds: int = 3600
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class AgentAssignment:
    """Assignment of an agent to a mission"""
    assignment_id: str
    mission_id: str
    agent_id: str
    agent_type: str  # red_recon, red_exploit, blue_detect, etc.
    sandbox_id: Optional[str] = None
    status: AgentStatus = AgentStatus.IDLE
    assigned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tasks: List[AgentTask] = None
    results: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = []
        if self.results is None:
            self.results = {}


@dataclass
class Mission:
    """Red/Blue team mission definition"""
    mission_id: str
    name: str
    description: str
    environment: Environment
    objectives: List[str]
    targets: List[Dict[str, Any]]
    constraints: Dict[str, Any] = None
    status: MissionStatus = MissionStatus.PENDING
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_seconds: int = 86400  # 24 hours default
    red_team_config: Dict[str, Any] = None
    blue_team_config: Dict[str, Any] = None
    agents: List[AgentAssignment] = None
    telemetry: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.red_team_config is None:
            self.red_team_config = {}
        if self.blue_team_config is None:
            self.blue_team_config = {}
        if self.agents is None:
            self.agents = []
        if self.telemetry is None:
            self.telemetry = {}


class AgentScheduler:
    """
    Central scheduler for coordinating red and blue team agents across missions.
    
    Features:
    - Mission planning and execution coordination
    - Agent capability validation and assignment
    - Task dependency resolution and scheduling
    - Sandbox resource management
    - Real-time mission monitoring and telemetry
    """
    
    def __init__(self, capability_registry: CapabilityRegistry,
                 sandbox_orchestrator: SandboxOrchestrator,
                 redis_client: Optional[redis.Redis] = None):
        self.capability_registry = capability_registry
        self.sandbox_orchestrator = sandbox_orchestrator
        self.redis_client = redis_client
        
        # Active missions and agents
        self.active_missions: Dict[str, Mission] = {}
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.mission_queues: Dict[str, asyncio.Queue] = {}  # mission_id -> task queue
        
        # Execution state
        self._execution_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()
        
    async def initialize(self):
        """Initialize the agent scheduler"""
        logger.info("Initializing Agent Scheduler...")
        
        # Load available agent types
        await self._discover_agent_types()
        
        # Start background monitoring
        asyncio.create_task(self._monitor_missions())
        
        logger.info(f"Agent Scheduler initialized with {len(self.agent_registry)} agent types")
        
    async def _discover_agent_types(self):
        """Discover available agent types and their capabilities"""
        # Red team agents
        self.agent_registry.update({
            "red_recon": {
                "name": "Reconnaissance Agent",
                "categories": [TechniqueCategory.RECONNAISSANCE, TechniqueCategory.DISCOVERY],
                "image": "xorb/red-recon:latest",
                "resource_requirements": ResourceConstraints(cpu_cores=1.0, memory_mb=512)
            },
            "red_exploit": {
                "name": "Exploitation Agent", 
                "categories": [TechniqueCategory.INITIAL_ACCESS, TechniqueCategory.PRIVILEGE_ESCALATION],
                "image": "xorb/red-exploit:latest",
                "resource_requirements": ResourceConstraints(cpu_cores=2.0, memory_mb=1024)
            },
            "red_persistence": {
                "name": "Persistence Agent",
                "categories": [TechniqueCategory.PERSISTENCE],
                "image": "xorb/red-persistence:latest", 
                "resource_requirements": ResourceConstraints(cpu_cores=1.0, memory_mb=512)
            },
            "red_evasion": {
                "name": "Evasion Agent",
                "categories": [TechniqueCategory.DEFENSE_EVASION],
                "image": "xorb/red-evasion:latest",
                "resource_requirements": ResourceConstraints(cpu_cores=1.5, memory_mb=768)
            },
            "red_collection": {
                "name": "Collection Agent",
                "categories": [TechniqueCategory.COLLECTION, TechniqueCategory.CREDENTIAL_ACCESS],
                "image": "xorb/red-collection:latest",
                "resource_requirements": ResourceConstraints(cpu_cores=1.0, memory_mb=512)
            },
            # Blue team agents
            "blue_detect": {
                "name": "Detection Agent",
                "categories": [TechniqueCategory.DETECTION],
                "image": "xorb/blue-detect:latest",
                "resource_requirements": ResourceConstraints(cpu_cores=2.0, memory_mb=2048)
            },
            "blue_analyze": {
                "name": "Analysis Agent", 
                "categories": [TechniqueCategory.ANALYSIS],
                "image": "xorb/blue-analyze:latest",
                "resource_requirements": ResourceConstraints(cpu_cores=4.0, memory_mb=4096)
            },
            "blue_hunt": {
                "name": "Threat Hunting Agent",
                "categories": [TechniqueCategory.THREAT_HUNTING],
                "image": "xorb/blue-hunt:latest",
                "resource_requirements": ResourceConstraints(cpu_cores=2.0, memory_mb=2048)
            },
            "blue_respond": {
                "name": "Incident Response Agent",
                "categories": [TechniqueCategory.MITIGATION, TechniqueCategory.RECOVERY],
                "image": "xorb/blue-respond:latest",
                "resource_requirements": ResourceConstraints(cpu_cores=1.0, memory_mb=1024)
            }
        })
        
    async def create_mission(self, mission_config: Dict[str, Any]) -> str:
        """Create a new red/blue team mission"""
        mission_id = str(uuid.uuid4())
        
        # Create mission object
        mission = Mission(
            mission_id=mission_id,
            name=mission_config["name"],
            description=mission_config["description"],
            environment=Environment(mission_config["environment"]),
            objectives=mission_config["objectives"],
            targets=mission_config["targets"],
            constraints=mission_config.get("constraints", {}),
            timeout_seconds=mission_config.get("timeout_seconds", 86400),
            red_team_config=mission_config.get("red_team_config", {}),
            blue_team_config=mission_config.get("blue_team_config", {})
        )
        
        # Validate mission against environment policies
        await self._validate_mission(mission)
        
        # Plan agent assignments
        await self._plan_mission_agents(mission)
        
        # Store mission
        self.active_missions[mission_id] = mission
        self.mission_queues[mission_id] = asyncio.Queue()
        
        # Cache in Redis
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"mission:{mission_id}",
                    mission.timeout_seconds,
                    json.dumps(asdict(mission), default=str)
                )
            except Exception as e:
                logger.warning(f"Failed to cache mission in Redis: {e}")
                
        logger.info(f"Created mission {mission_id}: {mission.name}")
        return mission_id
        
    async def _validate_mission(self, mission: Mission):
        """Validate mission objectives against environment capabilities"""
        # Check that required techniques are available in the environment
        for objective in mission.objectives:
            # This would parse objectives and check technique availability
            # For now, just validate environment exists
            if mission.environment not in [env for env in Environment]:
                raise ValueError(f"Invalid environment: {mission.environment}")
                
    async def _plan_mission_agents(self, mission: Mission):
        """Plan agent assignments for a mission based on objectives"""
        # Determine required agent types based on objectives and techniques
        required_agents = set()
        
        # Parse objectives to determine needed capabilities
        for objective in mission.objectives:
            if any(keyword in objective.lower() for keyword in ["recon", "scan", "discover"]):
                required_agents.add("red_recon")
            if any(keyword in objective.lower() for keyword in ["exploit", "attack", "penetrate"]):
                required_agents.add("red_exploit")
            if any(keyword in objective.lower() for keyword in ["persist", "maintain"]):
                required_agents.add("red_persistence")
            if any(keyword in objective.lower() for keyword in ["evade", "hide", "stealth"]):
                required_agents.add("red_evasion")
            if any(keyword in objective.lower() for keyword in ["detect", "monitor"]):
                required_agents.add("blue_detect")
            if any(keyword in objective.lower() for keyword in ["hunt", "investigate"]):
                required_agents.add("blue_hunt")
            if any(keyword in objective.lower() for keyword in ["respond", "mitigate"]):
                required_agents.add("blue_respond")
                
        # Create agent assignments
        for agent_type in required_agents:
            assignment = AgentAssignment(
                assignment_id=str(uuid.uuid4()),
                mission_id=mission.mission_id,
                agent_id=f"{agent_type}_{mission.mission_id[:8]}",
                agent_type=agent_type
            )
            mission.agents.append(assignment)
            
        # Add default blue team monitoring if not present
        if not any(a.agent_type.startswith("blue_") for a in mission.agents):
            assignment = AgentAssignment(
                assignment_id=str(uuid.uuid4()),
                mission_id=mission.mission_id,
                agent_id=f"blue_detect_{mission.mission_id[:8]}",
                agent_type="blue_detect"
            )
            mission.agents.append(assignment)
            
        logger.info(f"Planned {len(mission.agents)} agents for mission {mission.mission_id}")
        
    async def start_mission(self, mission_id: str) -> bool:
        """Start executing a mission"""
        mission = self.active_missions.get(mission_id)
        if not mission:
            logger.error(f"Mission {mission_id} not found")
            return False
            
        if mission.status != MissionStatus.PENDING:
            logger.error(f"Mission {mission_id} is not in pending status")
            return False
            
        try:
            mission.status = MissionStatus.PLANNING
            mission.started_at = datetime.utcnow()
            
            # Create sandbox environments for each agent
            for assignment in mission.agents:
                await self._create_agent_sandbox(mission, assignment)
                
            # Generate tasks for each agent
            for assignment in mission.agents:
                await self._generate_agent_tasks(mission, assignment)
                
            mission.status = MissionStatus.EXECUTING
            
            # Start execution task
            self._execution_tasks[mission_id] = asyncio.create_task(
                self._execute_mission(mission)
            )
            
            logger.info(f"Started mission {mission_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start mission {mission_id}: {e}")
            mission.status = MissionStatus.FAILED
            return False
            
    async def _create_agent_sandbox(self, mission: Mission, assignment: AgentAssignment):
        """Create a sandbox environment for an agent"""
        agent_info = self.agent_registry.get(assignment.agent_type)
        if not agent_info:
            raise ValueError(f"Unknown agent type: {assignment.agent_type}")
            
        # Create sandbox configuration
        sandbox_config = SandboxConfig(
            sandbox_id=f"sandbox_{assignment.assignment_id}",
            sandbox_type=SandboxType.DOCKER_SIDECAR,
            mission_id=mission.mission_id,
            agent_type=assignment.agent_type,
            environment=mission.environment.value,
            image=agent_info["image"],
            resource_constraints=agent_info["resource_requirements"],
            ttl_seconds=mission.timeout_seconds
        )
        
        # Create and start sandbox
        sandbox_id = await self.sandbox_orchestrator.create_sandbox(sandbox_config)
        assignment.sandbox_id = sandbox_id
        
        await self.sandbox_orchestrator.start_sandbox(sandbox_id)
        
        logger.info(f"Created sandbox {sandbox_id} for agent {assignment.agent_type}")
        
    async def _generate_agent_tasks(self, mission: Mission, assignment: AgentAssignment):
        """Generate tasks for an agent based on mission objectives"""
        agent_info = self.agent_registry.get(assignment.agent_type)
        if not agent_info:
            return
            
        # Get available techniques for this agent type
        available_techniques = []
        for category in agent_info["categories"]:
            techniques = await self.capability_registry.get_allowed_techniques(
                mission.environment, category
            )
            available_techniques.extend(techniques)
            
        # Generate tasks based on agent type and mission objectives
        tasks = []
        
        if assignment.agent_type == "red_recon":
            # Generate reconnaissance tasks
            for target in mission.targets:
                if "host" in target:
                    tasks.append(AgentTask(
                        task_id=str(uuid.uuid4()),
                        technique_id="recon.port_scan",
                        parameters={
                            "target": target["host"],
                            "ports": target.get("ports", "1-1000")
                        },
                        priority=8
                    ))
                    
        elif assignment.agent_type == "red_exploit":
            # Generate exploitation tasks (dependent on recon)
            recon_task_ids = [t.task_id for a in mission.agents if a.agent_type == "red_recon" for t in a.tasks]
            
            for target in mission.targets:
                if "web_url" in target:
                    tasks.append(AgentTask(
                        task_id=str(uuid.uuid4()),
                        technique_id="exploit.web_sqli",
                        parameters={
                            "url": target["web_url"],
                            "parameter": target.get("parameter", "id")
                        },
                        dependencies=recon_task_ids,
                        priority=9
                    ))
                    
        elif assignment.agent_type == "blue_detect":
            # Generate detection tasks
            tasks.append(AgentTask(
                task_id=str(uuid.uuid4()),
                technique_id="detect.network_anomaly",
                parameters={
                    "interface": "eth0",
                    "threshold": 0.8
                },
                priority=7
            ))
            
            tasks.append(AgentTask(
                task_id=str(uuid.uuid4()),
                technique_id="detect.process_monitoring",
                parameters={
                    "processes": ["*"],
                    "events": ["process_creation", "network_connection"]
                },
                priority=7
            ))
            
        assignment.tasks = tasks
        logger.info(f"Generated {len(tasks)} tasks for agent {assignment.agent_type}")
        
    async def _execute_mission(self, mission: Mission):
        """Execute a mission by coordinating agent tasks"""
        try:
            # Create task execution coroutines for each agent
            agent_tasks = []
            for assignment in mission.agents:
                agent_tasks.append(self._execute_agent_tasks(mission, assignment))
                
            # Wait for all agents to complete or timeout
            timeout = mission.timeout_seconds
            await asyncio.wait_for(
                asyncio.gather(*agent_tasks, return_exceptions=True),
                timeout=timeout
            )
            
            mission.status = MissionStatus.COMPLETED
            mission.completed_at = datetime.utcnow()
            
            logger.info(f"Mission {mission.mission_id} completed successfully")
            
        except asyncio.TimeoutError:
            logger.warning(f"Mission {mission.mission_id} timed out")
            mission.status = MissionStatus.FAILED
            
        except Exception as e:
            logger.error(f"Mission {mission.mission_id} failed: {e}")
            mission.status = MissionStatus.FAILED
            
        finally:
            # Cleanup agent sandboxes
            for assignment in mission.agents:
                if assignment.sandbox_id:
                    try:
                        await self.sandbox_orchestrator.stop_sandbox(assignment.sandbox_id)
                        await self.sandbox_orchestrator.remove_sandbox(assignment.sandbox_id)
                    except Exception as e:
                        logger.error(f"Failed to cleanup sandbox {assignment.sandbox_id}: {e}")
                        
    async def _execute_agent_tasks(self, mission: Mission, assignment: AgentAssignment):
        """Execute tasks for a specific agent"""
        assignment.status = AgentStatus.RUNNING
        assignment.started_at = datetime.utcnow()
        
        try:
            # Sort tasks by priority and dependencies
            sorted_tasks = await self._resolve_task_dependencies(assignment.tasks)
            
            for task in sorted_tasks:
                await self._execute_task(mission, assignment, task)
                
            assignment.status = AgentStatus.COMPLETED
            assignment.completed_at = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Agent {assignment.agent_type} failed: {e}")
            assignment.status = AgentStatus.FAILED
            
    async def _resolve_task_dependencies(self, tasks: List[AgentTask]) -> List[AgentTask]:
        """Resolve task dependencies and return execution order"""
        # Simple topological sort for task dependencies
        remaining_tasks = tasks.copy()
        sorted_tasks = []
        
        while remaining_tasks:
            # Find tasks with no unresolved dependencies
            ready_tasks = []
            for task in remaining_tasks:
                dependencies_met = all(
                    any(t.task_id == dep_id for t in sorted_tasks)
                    for dep_id in task.dependencies
                )
                if dependencies_met:
                    ready_tasks.append(task)
                    
            if not ready_tasks:
                # Circular dependency or external dependency
                logger.warning("Circular or external dependencies detected, executing remaining tasks anyway")
                ready_tasks = remaining_tasks
                
            # Sort ready tasks by priority
            ready_tasks.sort(key=lambda t: t.priority, reverse=True)
            
            # Add to sorted list and remove from remaining
            for task in ready_tasks:
                sorted_tasks.append(task)
                remaining_tasks.remove(task)
                
        return sorted_tasks
        
    async def _execute_task(self, mission: Mission, assignment: AgentAssignment, task: AgentTask):
        """Execute a single agent task"""
        logger.info(f"Executing task {task.task_id} ({task.technique_id}) for agent {assignment.agent_type}")
        
        try:
            # Validate technique parameters
            validated_params = await self.capability_registry.validate_technique_parameters(
                task.technique_id, task.parameters
            )
            
            # This would send the task to the agent in the sandbox
            # For now, simulate execution
            await asyncio.sleep(5)  # Simulate task execution
            
            # Store results
            assignment.results[task.task_id] = {
                "technique_id": task.technique_id,
                "status": "completed",
                "output": f"Simulated execution of {task.technique_id}",
                "executed_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            assignment.results[task.task_id] = {
                "technique_id": task.technique_id,
                "status": "failed",
                "error": str(e),
                "executed_at": datetime.utcnow().isoformat()
            }
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
                await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                await self._execute_task(mission, assignment, task)
                
    async def stop_mission(self, mission_id: str, force: bool = False) -> bool:
        """Stop a running mission"""
        mission = self.active_missions.get(mission_id)
        if not mission:
            logger.error(f"Mission {mission_id} not found")
            return False
            
        mission.status = MissionStatus.CANCELLED
        
        # Cancel execution task
        if mission_id in self._execution_tasks:
            self._execution_tasks[mission_id].cancel()
            
        # Stop agent sandboxes
        for assignment in mission.agents:
            if assignment.sandbox_id:
                await self.sandbox_orchestrator.stop_sandbox(assignment.sandbox_id, force)
                
        logger.info(f"Stopped mission {mission_id}")
        return True
        
    async def get_mission_status(self, mission_id: str) -> Optional[Mission]:
        """Get the current status of a mission"""
        return self.active_missions.get(mission_id)
        
    async def list_missions(self) -> List[Mission]:
        """List all active missions"""
        return list(self.active_missions.values())
        
    async def _monitor_missions(self):
        """Background task to monitor mission health and progress"""
        while not self._shutdown_event.is_set():
            try:
                # Check for timed-out missions
                now = datetime.utcnow()
                for mission in list(self.active_missions.values()):
                    if (mission.status == MissionStatus.EXECUTING and 
                        mission.started_at and
                        now > mission.started_at + timedelta(seconds=mission.timeout_seconds)):
                        logger.warning(f"Mission {mission.mission_id} timed out")
                        await self.stop_mission(mission.mission_id, force=True)
                        
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in mission monitoring: {e}")
                await asyncio.sleep(30)
                
    async def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        stats = {
            "total_missions": len(self.active_missions),
            "missions_by_status": {},
            "missions_by_environment": {},
            "total_agents": 0,
            "agents_by_type": {},
            "agents_by_status": {}
        }
        
        for mission in self.active_missions.values():
            # Count by status
            status = mission.status.value
            stats["missions_by_status"][status] = stats["missions_by_status"].get(status, 0) + 1
            
            # Count by environment
            env = mission.environment.value
            stats["missions_by_environment"][env] = stats["missions_by_environment"].get(env, 0) + 1
            
            # Count agents
            stats["total_agents"] += len(mission.agents)
            
            for assignment in mission.agents:
                # Count by type
                agent_type = assignment.agent_type
                stats["agents_by_type"][agent_type] = stats["agents_by_type"].get(agent_type, 0) + 1
                
                # Count by status
                agent_status = assignment.status.value
                stats["agents_by_status"][agent_status] = stats["agents_by_status"].get(agent_status, 0) + 1
                
        return stats
        
    async def shutdown(self):
        """Shutdown the agent scheduler"""
        logger.info("Shutting down Agent Scheduler...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Stop all missions
        for mission_id in list(self.active_missions.keys()):
            await self.stop_mission(mission_id, force=True)
            
        # Cancel execution tasks
        for task in self._execution_tasks.values():
            task.cancel()
            
        logger.info("Agent Scheduler shutdown complete")