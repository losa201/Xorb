#!/usr/bin/env python3
"""
Advanced Swarm Intelligence for XORB Agent Coordination
Implements collective decision-making, role switching, and redundancy handling
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class SwarmDecisionType(str, Enum):
    ROLE_ASSIGNMENT = "role_assignment"
    TASK_PRIORITIZATION = "task_prioritization"
    RESOURCE_ALLOCATION = "resource_allocation"
    FAILURE_RECOVERY = "failure_recovery"
    LOAD_BALANCING = "load_balancing"


class AgentRole(str, Enum):
    SCOUT = "scout"           # Reconnaissance and discovery
    HUNTER = "hunter"         # Active exploitation
    GUARDIAN = "guardian"     # Defense and monitoring
    ANALYST = "analyst"       # Data analysis and intelligence
    COORDINATOR = "coordinator"  # Swarm coordination
    HEALER = "healer"        # System recovery and repair


@dataclass
class SwarmAgent:
    """Enhanced agent with swarm intelligence capabilities"""
    agent_id: str
    name: str
    current_role: AgentRole
    capabilities: Set[str]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    workload: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    health_score: float = 1.0
    specialization_score: Dict[AgentRole, float] = field(default_factory=dict)
    adaptive_parameters: Dict[str, Any] = field(default_factory=dict)
    failure_count: int = 0
    recovery_attempts: int = 0


@dataclass
class SwarmDecision:
    """Collective decision made by the swarm"""
    decision_id: str
    decision_type: SwarmDecisionType
    participants: List[str]
    consensus_score: float
    decision_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    execution_priority: int = 5


class SwarmIntelligence:
    """Advanced swarm intelligence coordinator for XORB agents"""
    
    def __init__(self, max_agents: int = 64, convergence_threshold: float = 0.8):
        self.agents: Dict[str, SwarmAgent] = {}
        self.max_agents = max_agents
        self.convergence_threshold = convergence_threshold
        self.decision_history: List[SwarmDecision] = []
        self.collective_memory: Dict[str, Any] = {}
        self.performance_matrix = np.zeros((len(AgentRole), len(AgentRole)))
        
        # EPYC-optimized parameters
        self.coordination_interval = 5.0  # seconds
        self.health_check_interval = 10.0
        self.role_switching_cooldown = 30.0
        
        # Decision-making weights
        self.decision_weights = {
            "performance": 0.4,
            "availability": 0.3,
            "specialization": 0.2,
            "load_balance": 0.1
        }
        
    async def register_agent(self, agent: SwarmAgent) -> bool:
        """Register a new agent in the swarm"""
        if len(self.agents) >= self.max_agents:
            logger.warning(f"Swarm at maximum capacity ({self.max_agents})")
            return False
            
        # Initialize specialization scores
        for role in AgentRole:
            agent.specialization_score[role] = self._calculate_initial_specialization(
                agent.capabilities, role
            )
            
        self.agents[agent.agent_id] = agent
        logger.info(f"Agent {agent.agent_id} registered with role {agent.current_role}")
        
        # Trigger role optimization
        await self._optimize_role_distribution()
        return True
        
    async def deregister_agent(self, agent_id: str) -> bool:
        """Remove agent from swarm and redistribute its tasks"""
        if agent_id not in self.agents:
            return False
            
        agent = self.agents[agent_id]
        
        # Trigger recovery decision for orphaned tasks
        await self._handle_agent_failure(agent_id)
        
        del self.agents[agent_id]
        logger.info(f"Agent {agent_id} deregistered from swarm")
        
        # Rebalance remaining agents
        await self._rebalance_workload()
        return True
        
    async def make_collective_decision(self, decision_type: SwarmDecisionType, 
                                     context: Dict[str, Any]) -> SwarmDecision:
        """Make a collective decision using swarm intelligence"""
        
        # Select participating agents based on decision type
        participants = self._select_decision_participants(decision_type)
        
        if not participants:
            # Fallback to single-agent decision
            return await self._fallback_decision(decision_type, context)
            
        # Gather input from participating agents
        agent_inputs = await self._gather_agent_inputs(participants, decision_type, context)
        
        # Apply consensus algorithm
        consensus_result = await self._achieve_consensus(agent_inputs, decision_type)
        
        # Create decision record
        decision = SwarmDecision(
            decision_id=f"swarm_decision_{int(time.time())}",
            decision_type=decision_type,
            participants=participants,
            consensus_score=consensus_result["consensus_score"],
            decision_data=consensus_result["decision_data"]
        )
        
        self.decision_history.append(decision)
        
        # Update collective memory
        await self._update_collective_memory(decision)
        
        logger.info(f"Collective decision made: {decision_type} with {decision.consensus_score:.2f} consensus")
        return decision
        
    async def autonomous_role_switching(self) -> Dict[str, AgentRole]:
        """Autonomously switch agent roles based on performance and needs"""
        role_changes = {}
        
        # Analyze current role distribution efficiency
        role_efficiency = self._analyze_role_efficiency()
        
        # Identify optimization opportunities
        for agent_id, agent in self.agents.items():
            optimal_role = await self._calculate_optimal_role(agent, role_efficiency)
            
            if optimal_role != agent.current_role:
                # Check if role switch is beneficial
                switch_benefit = self._calculate_switch_benefit(agent, optimal_role)
                
                if switch_benefit > 0.2:  # Threshold for beneficial switch
                    role_changes[agent_id] = optimal_role
                    agent.current_role = optimal_role
                    logger.info(f"Agent {agent_id} switching to role {optimal_role}")
                    
        # Update performance tracking
        await self._update_performance_matrix(role_changes)
        
        return role_changes
        
    async def handle_redundancy(self, failed_agent_id: str) -> List[str]:
        """Handle agent failure with intelligent redundancy management"""
        if failed_agent_id not in self.agents:
            return []
            
        failed_agent = self.agents[failed_agent_id]
        
        # Find suitable backup agents
        backup_candidates = self._find_backup_candidates(failed_agent)
        
        # Distribute workload among backups
        selected_backups = await self._distribute_workload(
            failed_agent.workload, 
            backup_candidates
        )
        
        # Attempt self-healing if possible
        if await self._attempt_self_healing(failed_agent_id):
            logger.info(f"Agent {failed_agent_id} successfully self-healed")
            return []
            
        # Mark agent for recovery
        failed_agent.failure_count += 1
        failed_agent.health_score *= 0.8  # Degrade health score
        
        logger.warning(f"Agent {failed_agent_id} failed, workload redistributed to {selected_backups}")
        return selected_backups
        
    async def optimize_swarm_performance(self) -> Dict[str, Any]:
        """Continuously optimize swarm performance using collective intelligence"""
        
        # Performance analysis
        performance_metrics = self._calculate_swarm_metrics()
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks()
        
        # Generate optimization recommendations
        optimizations = await self._generate_optimizations(bottlenecks)
        
        # Apply approved optimizations
        applied_optimizations = []
        for optimization in optimizations:
            if optimization["confidence"] > 0.7:
                await self._apply_optimization(optimization)
                applied_optimizations.append(optimization)
                
        return {
            "performance_metrics": performance_metrics,
            "bottlenecks": bottlenecks,
            "optimizations_applied": applied_optimizations,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    def _calculate_initial_specialization(self, capabilities: Set[str], 
                                        role: AgentRole) -> float:
        """Calculate initial specialization score for a role"""
        role_capability_map = {
            AgentRole.SCOUT: {"reconnaissance", "discovery", "scanning"},
            AgentRole.HUNTER: {"exploitation", "payload", "attack"},
            AgentRole.GUARDIAN: {"defense", "monitoring", "protection"},
            AgentRole.ANALYST: {"analysis", "intelligence", "reporting"},
            AgentRole.COORDINATOR: {"orchestration", "coordination", "planning"},
            AgentRole.HEALER: {"recovery", "repair", "healing"}
        }
        
        role_capabilities = role_capability_map.get(role, set())
        if not role_capabilities:
            return 0.5
            
        overlap = len(capabilities.intersection(role_capabilities))
        return min(1.0, overlap / len(role_capabilities) + 0.2)
        
    def _select_decision_participants(self, decision_type: SwarmDecisionType) -> List[str]:
        """Select agents to participate in decision making"""
        participants = []
        
        # Role-based participation logic
        if decision_type == SwarmDecisionType.ROLE_ASSIGNMENT:
            # Include coordinators and high-performing agents
            participants = [
                agent_id for agent_id, agent in self.agents.items()
                if agent.current_role == AgentRole.COORDINATOR or 
                agent.performance_metrics.get("overall", 0) > 0.8
            ]
        elif decision_type == SwarmDecisionType.FAILURE_RECOVERY:
            # Include healers and available agents
            participants = [
                agent_id for agent_id, agent in self.agents.items()
                if agent.current_role in [AgentRole.HEALER, AgentRole.COORDINATOR] or
                agent.workload < 0.7
            ]
        else:
            # General participation based on availability and health
            participants = [
                agent_id for agent_id, agent in self.agents.items()
                if agent.health_score > 0.6 and agent.workload < 0.8
            ]
            
        return participants[:min(8, len(participants))]  # Limit for efficiency
        
    async def _gather_agent_inputs(self, participants: List[str], 
                                 decision_type: SwarmDecisionType,
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather input from participating agents"""
        inputs = {}
        
        for agent_id in participants:
            agent = self.agents[agent_id]
            
            # Simulate agent decision input based on role and capabilities
            agent_input = {
                "weight": agent.health_score * (1 + agent.specialization_score.get(agent.current_role, 0)),
                "preference": self._generate_agent_preference(agent, decision_type, context),
                "confidence": min(1.0, agent.performance_metrics.get("confidence", 0.5) + 0.2)
            }
            
            inputs[agent_id] = agent_input
            
        return inputs
        
    async def _achieve_consensus(self, agent_inputs: Dict[str, Any], 
                               decision_type: SwarmDecisionType) -> Dict[str, Any]:
        """Achieve consensus using weighted voting and convergence"""
        
        if not agent_inputs:
            return {"consensus_score": 0.0, "decision_data": {}}
            
        # Weight votes by agent reliability and expertise
        weighted_preferences = {}
        total_weight = 0
        
        for agent_id, input_data in agent_inputs.items():
            weight = input_data["weight"] * input_data["confidence"]
            preference = input_data["preference"]
            
            for key, value in preference.items():
                if key not in weighted_preferences:
                    weighted_preferences[key] = 0
                weighted_preferences[key] += value * weight
                
            total_weight += weight
            
        # Normalize preferences
        if total_weight > 0:
            for key in weighted_preferences:
                weighted_preferences[key] /= total_weight
                
        # Calculate consensus score
        consensus_score = self._calculate_consensus_score(agent_inputs)
        
        return {
            "consensus_score": consensus_score,
            "decision_data": weighted_preferences
        }
        
    def _generate_agent_preference(self, agent: SwarmAgent, 
                                 decision_type: SwarmDecisionType,
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate agent's preference for the decision"""
        
        # Role-based preference generation
        if decision_type == SwarmDecisionType.ROLE_ASSIGNMENT:
            return self._generate_role_preference(agent, context)
        elif decision_type == SwarmDecisionType.TASK_PRIORITIZATION:
            return self._generate_task_preference(agent, context)
        elif decision_type == SwarmDecisionType.LOAD_BALANCING:
            return self._generate_load_preference(agent, context)
        else:
            return {"default": 0.5}
            
    def _generate_role_preference(self, agent: SwarmAgent, 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate role assignment preferences"""
        preferences = {}
        
        for role in AgentRole:
            # Base preference on specialization and current performance
            base_preference = agent.specialization_score.get(role, 0.5)
            
            # Adjust based on current workload
            workload_factor = max(0.1, 1.0 - agent.workload)
            
            # Consider swarm needs
            role_demand = context.get("role_demand", {}).get(role.value, 0.5)
            
            preferences[role.value] = base_preference * workload_factor * (1 + role_demand)
            
        return preferences
        
    def _calculate_consensus_score(self, agent_inputs: Dict[str, Any]) -> float:
        """Calculate consensus score based on agent agreement"""
        if len(agent_inputs) < 2:
            return 1.0
            
        # Calculate variance in preferences
        preferences = [input_data["preference"] for input_data in agent_inputs.values()]
        
        # Simple consensus metric based on preference alignment
        consensus_values = []
        for key in preferences[0].keys():
            values = [pref.get(key, 0) for pref in preferences]
            variance = np.var(values) if len(values) > 1 else 0
            consensus_values.append(1.0 - min(1.0, variance))
            
        return np.mean(consensus_values) if consensus_values else 0.5
        
    async def _fallback_decision(self, decision_type: SwarmDecisionType,
                               context: Dict[str, Any]) -> SwarmDecision:
        """Make fallback decision when no agents available"""
        
        return SwarmDecision(
            decision_id=f"fallback_decision_{int(time.time())}",
            decision_type=decision_type,
            participants=[],
            consensus_score=0.5,
            decision_data={"fallback": True, "context": context}
        )
        
    async def _update_collective_memory(self, decision: SwarmDecision):
        """Update collective memory with decision outcomes"""
        memory_key = f"{decision.decision_type}_{len(self.decision_history)}"
        
        self.collective_memory[memory_key] = {
            "decision_id": decision.decision_id,
            "consensus_score": decision.consensus_score,
            "participant_count": len(decision.participants),
            "timestamp": decision.timestamp.isoformat()
        }
        
        # Maintain memory size limit
        if len(self.collective_memory) > 1000:
            oldest_key = min(self.collective_memory.keys())
            del self.collective_memory[oldest_key]
            
    def _analyze_role_efficiency(self) -> Dict[AgentRole, float]:
        """Analyze current role distribution efficiency"""
        role_counts = defaultdict(int)
        role_performance = defaultdict(list)
        
        for agent in self.agents.values():
            role_counts[agent.current_role] += 1
            performance = agent.performance_metrics.get("overall", 0.5)
            role_performance[agent.current_role].append(performance)
            
        efficiency = {}
        for role in AgentRole:
            count = role_counts[role]
            if count > 0:
                avg_performance = np.mean(role_performance[role])
                # Efficiency considers both performance and distribution
                efficiency[role] = avg_performance * (1.0 / (1.0 + abs(count - 2)))
            else:
                efficiency[role] = 0.0
                
        return efficiency
        
    async def _calculate_optimal_role(self, agent: SwarmAgent, 
                                    role_efficiency: Dict[AgentRole, float]) -> AgentRole:
        """Calculate optimal role for an agent"""
        best_role = agent.current_role
        best_score = 0.0
        
        for role in AgentRole:
            # Combined score: specialization + efficiency + performance potential
            specialization = agent.specialization_score.get(role, 0.0)
            efficiency = role_efficiency.get(role, 0.0)
            
            # Penalty for frequent role changes
            change_penalty = 0.1 if role != agent.current_role else 0.0
            
            score = (specialization * 0.6 + efficiency * 0.4) - change_penalty
            
            if score > best_score:
                best_score = score
                best_role = role
                
        return best_role
        
    def _calculate_switch_benefit(self, agent: SwarmAgent, new_role: AgentRole) -> float:
        """Calculate benefit of switching to new role"""
        current_performance = agent.specialization_score.get(agent.current_role, 0.5)
        new_performance = agent.specialization_score.get(new_role, 0.5)
        
        # Consider workload impact
        workload_factor = max(0.5, 1.0 - agent.workload)
        
        return (new_performance - current_performance) * workload_factor
        
    async def _update_performance_matrix(self, role_changes: Dict[str, AgentRole]):
        """Update performance tracking matrix"""
        for agent_id, new_role in role_changes.items():
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                old_role_idx = list(AgentRole).index(agent.current_role)
                new_role_idx = list(AgentRole).index(new_role)
                
                # Update transition probability
                self.performance_matrix[old_role_idx][new_role_idx] += 0.1
                
    def _find_backup_candidates(self, failed_agent: SwarmAgent) -> List[SwarmAgent]:
        """Find suitable backup agents for failed agent"""
        candidates = []
        
        for agent in self.agents.values():
            if (agent.agent_id != failed_agent.agent_id and
                agent.health_score > 0.7 and
                agent.workload < 0.6):
                
                # Check capability overlap
                overlap = len(agent.capabilities.intersection(failed_agent.capabilities))
                if overlap > 0:
                    candidates.append(agent)
                    
        # Sort by suitability
        candidates.sort(key=lambda a: (
            len(a.capabilities.intersection(failed_agent.capabilities)),
            -a.workload,
            a.health_score
        ), reverse=True)
        
        return candidates[:3]  # Top 3 candidates
        
    async def _distribute_workload(self, failed_workload: float, 
                                 backup_agents: List[SwarmAgent]) -> List[str]:
        """Distribute failed agent workload among backups"""
        if not backup_agents:
            return []
            
        # Calculate distribution based on capacity
        total_capacity = sum(1.0 - agent.workload for agent in backup_agents)
        
        if total_capacity <= 0:
            return []
            
        selected_backups = []
        remaining_workload = failed_workload
        
        for agent in backup_agents:
            if remaining_workload <= 0:
                break
                
            capacity = 1.0 - agent.workload
            allocation = min(remaining_workload, capacity)
            
            if allocation > 0.1:  # Minimum meaningful allocation
                agent.workload += allocation
                remaining_workload -= allocation
                selected_backups.append(agent.agent_id)
                
        return selected_backups
        
    async def _attempt_self_healing(self, failed_agent_id: str) -> bool:
        """Attempt to self-heal failed agent"""
        if failed_agent_id not in self.agents:
            return False
            
        agent = self.agents[failed_agent_id]
        
        # Simple self-healing logic
        if agent.failure_count < 3 and agent.recovery_attempts < 2:
            agent.recovery_attempts += 1
            agent.health_score = min(1.0, agent.health_score + 0.3)
            
            # Reset some parameters
            agent.workload *= 0.8
            agent.last_heartbeat = datetime.utcnow()
            
            return True
            
        return False
        
    def _calculate_swarm_metrics(self) -> Dict[str, float]:
        """Calculate overall swarm performance metrics"""
        if not self.agents:
            return {}
            
        metrics = {
            "agent_count": len(self.agents),
            "avg_health_score": np.mean([a.health_score for a in self.agents.values()]),
            "avg_workload": np.mean([a.workload for a in self.agents.values()]),
            "role_diversity": len(set(a.current_role for a in self.agents.values())),
            "total_capabilities": len(set().union(*[a.capabilities for a in self.agents.values()]))
        }
        
        return metrics
        
    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks in the swarm"""
        bottlenecks = []
        
        # Check for overloaded agents
        overloaded = [a for a in self.agents.values() if a.workload > 0.9]
        if overloaded:
            bottlenecks.append({
                "type": "agent_overload",
                "count": len(overloaded),
                "severity": "high" if len(overloaded) > len(self.agents) * 0.3 else "medium"
            })
            
        # Check for role imbalance
        role_counts = defaultdict(int)
        for agent in self.agents.values():
            role_counts[agent.current_role] += 1
            
        if max(role_counts.values()) > len(self.agents) * 0.6:
            bottlenecks.append({
                "type": "role_imbalance",
                "dominant_role": max(role_counts, key=role_counts.get).value,
                "severity": "medium"
            })
            
        return bottlenecks
        
    async def _generate_optimizations(self, bottlenecks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations"""
        optimizations = []
        
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "agent_overload":
                optimizations.append({
                    "type": "workload_redistribution",
                    "action": "redistribute_workload",
                    "confidence": 0.8,
                    "priority": "high"
                })
            elif bottleneck["type"] == "role_imbalance":
                optimizations.append({
                    "type": "role_rebalancing",
                    "action": "trigger_role_switching",
                    "confidence": 0.7,
                    "priority": "medium"
                })
                
        return optimizations
        
    async def _apply_optimization(self, optimization: Dict[str, Any]):
        """Apply an optimization to the swarm"""
        if optimization["action"] == "redistribute_workload":
            await self._rebalance_workload()
        elif optimization["action"] == "trigger_role_switching":
            await self.autonomous_role_switching()
            
        logger.info(f"Applied optimization: {optimization['type']}")
        
    async def _rebalance_workload(self):
        """Rebalance workload across available agents"""
        if not self.agents:
            return
            
        # Calculate target workload
        total_workload = sum(agent.workload for agent in self.agents.values())
        target_workload = total_workload / len(self.agents)
        
        # Redistribute from overloaded to underloaded agents
        overloaded = [a for a in self.agents.values() if a.workload > target_workload * 1.2]
        underloaded = [a for a in self.agents.values() if a.workload < target_workload * 0.8]
        
        for overloaded_agent in overloaded:
            excess = overloaded_agent.workload - target_workload
            
            for underloaded_agent in underloaded:
                if excess <= 0:
                    break
                    
                capacity = target_workload - underloaded_agent.workload
                transfer = min(excess, capacity)
                
                if transfer > 0.05:  # Minimum meaningful transfer
                    overloaded_agent.workload -= transfer
                    underloaded_agent.workload += transfer
                    excess -= transfer