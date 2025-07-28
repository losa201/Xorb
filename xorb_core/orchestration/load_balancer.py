#!/usr/bin/env python3
"""
Advanced Load Balancer for XORB Multi-Agent Coordination
Implements intelligent workload distribution and EPYC optimization
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
import logging
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    PERFORMANCE_BASED = "performance_based"
    CAPABILITY_AWARE = "capability_aware"
    SWARM_INTELLIGENT = "swarm_intelligent"


class TaskPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


@dataclass
class LoadMetrics:
    """Real-time load metrics for agents"""
    agent_id: str
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_tasks: int = 0
    queue_length: int = 0
    response_time: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TaskRequest:
    """Enhanced task request with load balancing metadata"""
    task_id: str
    task_type: str
    priority: TaskPriority
    required_capabilities: Set[str]
    estimated_duration: float
    resource_requirements: Dict[str, float]
    deadline: Optional[datetime] = None
    affinity_rules: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


class EPYCOptimizedLoadBalancer:
    """EPYC-optimized load balancer with advanced multi-agent coordination"""
    
    def __init__(self, max_agents: int = 64, epyc_cores: int = 64):
        self.max_agents = max_agents
        self.epyc_cores = epyc_cores
        self.agent_metrics: Dict[str, LoadMetrics] = {}
        self.task_queue: Dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }
        
        # EPYC-specific optimizations
        self.numa_topology = self._initialize_numa_topology()
        self.core_affinity_map: Dict[str, Set[int]] = {}
        self.memory_channels = 8  # EPYC typical memory channels
        
        # Load balancing configuration
        self.strategy = LoadBalancingStrategy.SWARM_INTELLIGENT
        self.rebalancing_interval = 10.0  # seconds
        self.health_check_interval = 5.0
        self.performance_window = 300  # 5 minutes
        
        # Performance tracking
        self.performance_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self.load_distribution_stats = {
            "total_tasks": 0,
            "balanced_tasks": 0,
            "failed_assignments": 0,
            "rebalancing_events": 0
        }
        
        # Circuit breaker configuration
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.circuit_breaker_threshold = 0.5  # 50% error rate
        self.circuit_breaker_timeout = 60.0  # seconds
        
    def _initialize_numa_topology(self) -> Dict[int, List[int]]:
        """Initialize NUMA topology for EPYC processors"""
        # EPYC 7702: 4 NUMA nodes, 16 cores per node
        numa_nodes = {}
        cores_per_node = self.epyc_cores // 4
        
        for node in range(4):
            start_core = node * cores_per_node
            end_core = start_core + cores_per_node
            numa_nodes[node] = list(range(start_core, end_core))
            
        return numa_nodes
        
    async def register_agent(self, agent_id: str, capabilities: Set[str],
                           initial_metrics: Optional[LoadMetrics] = None) -> bool:
        """Register agent with NUMA-aware core affinity"""
        
        if len(self.agent_metrics) >= self.max_agents:
            logger.warning(f"Maximum agents ({self.max_agents}) reached")
            return False
            
        # Initialize metrics
        if initial_metrics:
            self.agent_metrics[agent_id] = initial_metrics
        else:
            self.agent_metrics[agent_id] = LoadMetrics(agent_id=agent_id)
            
        # Assign NUMA-aware core affinity
        await self._assign_core_affinity(agent_id)
        
        # Initialize circuit breaker
        self.circuit_breakers[agent_id] = {
            "state": "closed",  # closed, open, half_open
            "failure_count": 0,
            "last_failure": None,
            "success_count": 0
        }
        
        logger.info(f"Agent {agent_id} registered with core affinity: {self.core_affinity_map.get(agent_id, 'auto')}")
        return True
        
    async def deregister_agent(self, agent_id: str) -> bool:
        """Deregister agent and redistribute its tasks"""
        if agent_id not in self.agent_metrics:
            return False
            
        # Redistribute pending tasks
        await self._redistribute_agent_tasks(agent_id)
        
        # Cleanup
        del self.agent_metrics[agent_id]
        if agent_id in self.core_affinity_map:
            del self.core_affinity_map[agent_id]
        if agent_id in self.circuit_breakers:
            del self.circuit_breakers[agent_id]
        if agent_id in self.performance_history:
            del self.performance_history[agent_id]
            
        logger.info(f"Agent {agent_id} deregistered")
        return True
        
    async def assign_task(self, task: TaskRequest) -> Optional[str]:
        """Intelligently assign task to optimal agent"""
        
        # Get available agents
        available_agents = await self._get_available_agents(task)
        
        if not available_agents:
            # Queue task for later assignment
            await self._queue_task(task)
            return None
            
        # Select optimal agent based on strategy
        selected_agent = await self._select_optimal_agent(task, available_agents)
        
        if selected_agent:
            # Update agent metrics
            await self._update_agent_load(selected_agent, task)
            
            # Record assignment
            self.load_distribution_stats["total_tasks"] += 1
            self.load_distribution_stats["balanced_tasks"] += 1
            
            logger.debug(f"Task {task.task_id} assigned to agent {selected_agent}")
            return selected_agent
        else:
            # Assignment failed
            self.load_distribution_stats["failed_assignments"] += 1
            await self._queue_task(task)
            return None
            
    async def update_agent_metrics(self, agent_id: str, metrics: LoadMetrics):
        """Update real-time agent metrics"""
        if agent_id not in self.agent_metrics:
            return
            
        self.agent_metrics[agent_id] = metrics
        
        # Update performance history
        performance_score = self._calculate_performance_score(metrics)
        self.performance_history[agent_id].append({
            "timestamp": datetime.utcnow(),
            "performance": performance_score,
            "cpu": metrics.cpu_usage,
            "memory": metrics.memory_usage,
            "throughput": metrics.throughput
        })
        
        # Update circuit breaker
        await self._update_circuit_breaker(agent_id, metrics)
        
    async def rebalance_workload(self) -> Dict[str, Any]:
        """Proactively rebalance workload across agents"""
        
        if not self.agent_metrics:
            return {"status": "no_agents"}
            
        # Analyze current load distribution
        load_analysis = self._analyze_load_distribution()
        
        # Identify rebalancing opportunities
        rebalancing_actions = await self._identify_rebalancing_actions(load_analysis)
        
        # Execute rebalancing
        executed_actions = []
        for action in rebalancing_actions:
            if await self._execute_rebalancing_action(action):
                executed_actions.append(action)
                
        if executed_actions:
            self.load_distribution_stats["rebalancing_events"] += 1
            logger.info(f"Executed {len(executed_actions)} rebalancing actions")
            
        return {
            "status": "completed",
            "actions_executed": len(executed_actions),
            "load_analysis": load_analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    async def _assign_core_affinity(self, agent_id: str):
        """Assign NUMA-aware core affinity for agent"""
        
        # Find least loaded NUMA node
        numa_loads = {}
        for node, cores in self.numa_topology.items():
            assigned_agents = sum(
                1 for affinity in self.core_affinity_map.values()
                if any(core in cores for core in affinity)
            )
            numa_loads[node] = assigned_agents
            
        # Select least loaded NUMA node
        optimal_node = min(numa_loads, key=numa_loads.get)
        
        # Assign cores from optimal node
        node_cores = self.numa_topology[optimal_node]
        cores_per_agent = max(1, len(node_cores) // (self.max_agents // 4))
        
        # Find available cores in the node
        used_cores = set()
        for affinity in self.core_affinity_map.values():
            used_cores.update(affinity.intersection(set(node_cores)))
            
        available_cores = set(node_cores) - used_cores
        
        if available_cores:
            assigned_cores = set(list(available_cores)[:cores_per_agent])
            self.core_affinity_map[agent_id] = assigned_cores
        else:
            # Fallback to any available cores
            self.core_affinity_map[agent_id] = set([len(self.core_affinity_map) % self.epyc_cores])
            
    async def _get_available_agents(self, task: TaskRequest) -> List[str]:
        """Get agents available for task assignment"""
        available = []
        
        for agent_id, metrics in self.agent_metrics.items():
            # Check circuit breaker
            if self.circuit_breakers[agent_id]["state"] == "open":
                continue
                
            # Check capability match
            # Note: This would need integration with agent capability system
            
            # Check resource availability
            if (metrics.cpu_usage < 0.8 and 
                metrics.memory_usage < 0.8 and
                metrics.queue_length < 10):
                available.append(agent_id)
                
        return available
        
    async def _select_optimal_agent(self, task: TaskRequest, 
                                  available_agents: List[str]) -> Optional[str]:
        """Select optimal agent based on current strategy"""
        
        if not available_agents:
            return None
            
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_agents)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(available_agents)
        elif self.strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
            return self._performance_based_selection(available_agents)
        elif self.strategy == LoadBalancingStrategy.SWARM_INTELLIGENT:
            return await self._swarm_intelligent_selection(task, available_agents)
        else:
            return available_agents[0]  # Fallback
            
    def _round_robin_selection(self, available_agents: List[str]) -> str:
        """Simple round-robin selection"""
        index = self.load_distribution_stats["total_tasks"] % len(available_agents)
        return available_agents[index]
        
    def _least_connections_selection(self, available_agents: List[str]) -> str:
        """Select agent with least active connections"""
        return min(available_agents, 
                  key=lambda agent: self.agent_metrics[agent].active_tasks)
                  
    def _performance_based_selection(self, available_agents: List[str]) -> str:
        """Select agent based on recent performance"""
        best_agent = available_agents[0]
        best_score = 0.0
        
        for agent_id in available_agents:
            score = self._calculate_agent_score(agent_id)
            if score > best_score:
                best_score = score
                best_agent = agent_id
                
        return best_agent
        
    async def _swarm_intelligent_selection(self, task: TaskRequest,
                                         available_agents: List[str]) -> str:
        """Advanced swarm-based agent selection"""
        
        # Multi-criteria scoring
        scores = {}
        
        for agent_id in available_agents:
            metrics = self.agent_metrics[agent_id]
            
            # Performance score (40%)
            performance = self._calculate_performance_score(metrics)
            
            # Load score (30%)
            load_factor = 1.0 - (metrics.cpu_usage + metrics.memory_usage) / 2.0
            
            # NUMA affinity bonus (20%)
            numa_bonus = self._calculate_numa_bonus(agent_id, task)
            
            # Priority alignment (10%)
            priority_bonus = self._calculate_priority_bonus(agent_id, task.priority)
            
            # Combined score
            scores[agent_id] = (
                performance * 0.4 +
                load_factor * 0.3 +
                numa_bonus * 0.2 +
                priority_bonus * 0.1
            )
            
        # Select highest scoring agent
        return max(scores, key=scores.get)
        
    def _calculate_performance_score(self, metrics: LoadMetrics) -> float:
        """Calculate normalized performance score"""
        if metrics.response_time <= 0:
            response_score = 1.0
        else:
            response_score = max(0.1, 1.0 / metrics.response_time)
            
        error_score = max(0.1, 1.0 - metrics.error_rate)
        throughput_score = min(1.0, metrics.throughput / 100.0)  # Normalize to 100 ops/sec
        
        return (response_score + error_score + throughput_score) / 3.0
        
    def _calculate_agent_score(self, agent_id: str) -> float:
        """Calculate overall agent score"""
        if agent_id not in self.performance_history:
            return 0.5
            
        recent_performance = list(self.performance_history[agent_id])[-10:]  # Last 10 entries
        if not recent_performance:
            return 0.5
            
        return np.mean([entry["performance"] for entry in recent_performance])
        
    def _calculate_numa_bonus(self, agent_id: str, task: TaskRequest) -> float:
        """Calculate NUMA affinity bonus"""
        if agent_id not in self.core_affinity_map:
            return 0.5
            
        # Simple bonus for dedicated cores
        assigned_cores = len(self.core_affinity_map[agent_id])
        return min(1.0, assigned_cores / 4.0)  # Bonus for having more dedicated cores
        
    def _calculate_priority_bonus(self, agent_id: str, priority: TaskPriority) -> float:
        """Calculate priority-based assignment bonus"""
        metrics = self.agent_metrics[agent_id]
        
        # High-priority tasks prefer low-loaded agents
        if priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
            return 1.0 - (metrics.cpu_usage + metrics.memory_usage) / 2.0
        else:
            return 0.5  # Neutral for normal priority
            
    async def _queue_task(self, task: TaskRequest):
        """Queue task for later assignment"""
        self.task_queue[task.priority].append(task)
        logger.debug(f"Task {task.task_id} queued with priority {task.priority}")
        
    async def _update_agent_load(self, agent_id: str, task: TaskRequest):
        """Update agent load after task assignment"""
        if agent_id in self.agent_metrics:
            metrics = self.agent_metrics[agent_id]
            metrics.active_tasks += 1
            metrics.queue_length += 1
            
            # Estimate resource impact
            cpu_impact = task.resource_requirements.get("cpu", 0.1)
            memory_impact = task.resource_requirements.get("memory", 0.1)
            
            metrics.cpu_usage = min(1.0, metrics.cpu_usage + cpu_impact)
            metrics.memory_usage = min(1.0, metrics.memory_usage + memory_impact)
            
    async def _update_circuit_breaker(self, agent_id: str, metrics: LoadMetrics):
        """Update circuit breaker state based on agent metrics"""
        breaker = self.circuit_breakers[agent_id]
        
        # Check if agent is failing
        if metrics.error_rate > self.circuit_breaker_threshold:
            breaker["failure_count"] += 1
            breaker["last_failure"] = datetime.utcnow()
            
            # Open circuit if too many failures
            if breaker["failure_count"] >= 5 and breaker["state"] == "closed":
                breaker["state"] = "open"
                logger.warning(f"Circuit breaker opened for agent {agent_id}")
        else:
            # Reset on success
            if breaker["state"] == "closed":
                breaker["failure_count"] = max(0, breaker["failure_count"] - 1)
            elif breaker["state"] == "half_open":
                breaker["success_count"] += 1
                if breaker["success_count"] >= 3:
                    breaker["state"] = "closed"
                    breaker["failure_count"] = 0
                    logger.info(f"Circuit breaker closed for agent {agent_id}")
                    
        # Transition from open to half-open after timeout
        if (breaker["state"] == "open" and 
            breaker["last_failure"] and
            (datetime.utcnow() - breaker["last_failure"]).total_seconds() > self.circuit_breaker_timeout):
            breaker["state"] = "half_open"
            breaker["success_count"] = 0
            logger.info(f"Circuit breaker half-open for agent {agent_id}")
            
    def _analyze_load_distribution(self) -> Dict[str, Any]:
        """Analyze current load distribution across agents"""
        if not self.agent_metrics:
            return {"status": "no_agents"}
            
        cpu_loads = [m.cpu_usage for m in self.agent_metrics.values()]
        memory_loads = [m.memory_usage for m in self.agent_metrics.values()]
        task_counts = [m.active_tasks for m in self.agent_metrics.values()]
        
        return {
            "agent_count": len(self.agent_metrics),
            "cpu_stats": {
                "mean": np.mean(cpu_loads),
                "std": np.std(cpu_loads),
                "max": np.max(cpu_loads),
                "min": np.min(cpu_loads)
            },
            "memory_stats": {
                "mean": np.mean(memory_loads),
                "std": np.std(memory_loads),
                "max": np.max(memory_loads),
                "min": np.min(memory_loads)
            },
            "task_distribution": {
                "mean": np.mean(task_counts),
                "std": np.std(task_counts),
                "max": np.max(task_counts),
                "min": np.min(task_counts)
            },
            "load_balance_score": self._calculate_load_balance_score()
        }
        
    def _calculate_load_balance_score(self) -> float:
        """Calculate overall load balance score (0-1, higher is better)"""
        if not self.agent_metrics:
            return 1.0
            
        cpu_loads = [m.cpu_usage for m in self.agent_metrics.values()]
        memory_loads = [m.memory_usage for m in self.agent_metrics.values()]
        
        # Lower standard deviation = better balance
        cpu_balance = max(0.0, 1.0 - np.std(cpu_loads))
        memory_balance = max(0.0, 1.0 - np.std(memory_loads))
        
        return (cpu_balance + memory_balance) / 2.0
        
    async def _identify_rebalancing_actions(self, load_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify opportunities for load rebalancing"""
        actions = []
        
        # Check if rebalancing is needed
        balance_score = load_analysis.get("load_balance_score", 1.0)
        if balance_score > 0.8:  # Already well balanced
            return actions
            
        # Find overloaded and underloaded agents
        overloaded = []
        underloaded = []
        
        for agent_id, metrics in self.agent_metrics.items():
            load_factor = (metrics.cpu_usage + metrics.memory_usage) / 2.0
            
            if load_factor > 0.8:
                overloaded.append((agent_id, load_factor))
            elif load_factor < 0.3:
                underloaded.append((agent_id, load_factor))
                
        # Generate rebalancing actions
        for overloaded_agent, _ in overloaded:
            for underloaded_agent, _ in underloaded:
                actions.append({
                    "type": "migrate_tasks",
                    "source_agent": overloaded_agent,
                    "target_agent": underloaded_agent,
                    "task_count": 2  # Migrate 2 tasks
                })
                
        return actions[:5]  # Limit to 5 actions per rebalancing cycle
        
    async def _execute_rebalancing_action(self, action: Dict[str, Any]) -> bool:
        """Execute a rebalancing action"""
        if action["type"] == "migrate_tasks":
            source = action["source_agent"]
            target = action["target_agent"]
            
            # Simulate task migration (in real implementation, would coordinate with orchestrator)
            if source in self.agent_metrics and target in self.agent_metrics:
                source_metrics = self.agent_metrics[source]
                target_metrics = self.agent_metrics[target]
                
                # Transfer load
                tasks_to_migrate = min(action["task_count"], source_metrics.active_tasks)
                
                source_metrics.active_tasks -= tasks_to_migrate
                target_metrics.active_tasks += tasks_to_migrate
                
                # Adjust resource usage
                cpu_transfer = min(0.2, source_metrics.cpu_usage * 0.3)
                memory_transfer = min(0.2, source_metrics.memory_usage * 0.3)
                
                source_metrics.cpu_usage -= cpu_transfer
                source_metrics.memory_usage -= memory_transfer
                target_metrics.cpu_usage += cpu_transfer
                target_metrics.memory_usage += memory_transfer
                
                logger.info(f"Migrated {tasks_to_migrate} tasks from {source} to {target}")
                return True
                
        return False
        
    async def _redistribute_agent_tasks(self, failed_agent_id: str):
        """Redistribute tasks from failed agent"""
        if failed_agent_id not in self.agent_metrics:
            return
            
        failed_metrics = self.agent_metrics[failed_agent_id]
        tasks_to_redistribute = failed_metrics.active_tasks
        
        if tasks_to_redistribute == 0:
            return
            
        # Find available agents for redistribution
        available_agents = [
            agent_id for agent_id, metrics in self.agent_metrics.items()
            if (agent_id != failed_agent_id and 
                metrics.cpu_usage < 0.7 and 
                self.circuit_breakers[agent_id]["state"] != "open")
        ]
        
        if not available_agents:
            logger.warning(f"No available agents for task redistribution from {failed_agent_id}")
            return
            
        # Distribute tasks evenly
        tasks_per_agent = tasks_to_redistribute // len(available_agents)
        remaining_tasks = tasks_to_redistribute % len(available_agents)
        
        for i, agent_id in enumerate(available_agents):
            tasks_assigned = tasks_per_agent + (1 if i < remaining_tasks else 0)
            
            if tasks_assigned > 0:
                self.agent_metrics[agent_id].active_tasks += tasks_assigned
                logger.info(f"Redistributed {tasks_assigned} tasks to agent {agent_id}")
                
    async def get_load_statistics(self) -> Dict[str, Any]:
        """Get comprehensive load balancing statistics"""
        return {
            "total_agents": len(self.agent_metrics),
            "active_agents": len([
                a for a in self.agent_metrics.values() 
                if a.active_tasks > 0
            ]),
            "circuit_breakers_open": len([
                b for b in self.circuit_breakers.values() 
                if b["state"] == "open"
            ]),
            "load_distribution": self._analyze_load_distribution(),
            "performance_stats": self.load_distribution_stats.copy(),
            "epyc_optimization": {
                "numa_nodes": len(self.numa_topology),
                "cores_assigned": sum(len(cores) for cores in self.core_affinity_map.values()),
                "memory_channels": self.memory_channels
            }
        }