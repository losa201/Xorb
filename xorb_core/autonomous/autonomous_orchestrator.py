#!/usr/bin/env python3
"""
Autonomous Orchestrator Extension for Xorb Security Intelligence Platform

This module extends the Enhanced Orchestrator with autonomous capabilities:
- Intelligent agent selection and task distribution
- Dynamic workflow adaptation based on performance
- Autonomous resource optimization
- Self-healing and recovery mechanisms
- Advanced intelligence synthesis and learning
"""

import asyncio
import logging
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import heapq
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Union

import structlog
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

from xorb_common.orchestration.enhanced_orchestrator import (
    EnhancedOrchestrator, ExecutionContext, ExecutionStatus, 
    AgentRegistry, MetricsCollector
)
from .autonomous_worker import AutonomousWorker, AutonomyLevel, WorkerIntelligence, ResourceMonitor
from xorb_common.agents.base_agent import AgentCapability, AgentTask
from .rl_orchestrator_extensions import TaskPreemptor, PreemptionEvent, ConfidenceTracker, LearningFeedbackLoop, BayesianTaskOptimizer, ExecutionGraph

from .models import AutonomousDecision, WorkloadProfile, WorkloadAnalyzer, PerformanceOptimizer


class AutonomousOrchestrator(EnhancedOrchestrator):
    """
    Autonomous Security Intelligence Orchestrator v2.1 - Agent-Led Mode
    
    Extends the Enhanced Orchestrator with autonomous capabilities:
    - Intelligent agent selection and load balancing
    - Dynamic resource optimization and scaling
    - Autonomous workflow adaptation and improvement
    - Self-healing capabilities and failure recovery
    - Advanced performance learning and optimization
    - ENHANCED: Agent-led task prioritization and autonomous execution
    - ENHANCED: Multi-agent collaboration without competition
    - ENHANCED: Predictive resource allocation and Claude/Qwen reasoning
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 nats_url: str = "nats://localhost:4222",
                 plugin_directories: List[str] = None,
                 max_concurrent_agents: int = 32,
                 max_concurrent_campaigns: int = 10,
                 autonomy_level: AutonomyLevel = AutonomyLevel.MODERATE):
        
        super().__init__(redis_url, nats_url, plugin_directories, 
                        max_concurrent_agents, max_concurrent_campaigns)
        
        self.autonomy_level = autonomy_level
        self.logger = structlog.get_logger("AutonomousOrchestrator")
        
        # Autonomous components
        self.autonomous_workers: Dict[str, AutonomousWorker] = {}
        self.decision_history: List[AutonomousDecision] = []
        self.workload_analyzer = WorkloadAnalyzer()
        self.performance_optimizer = PerformanceOptimizer()
        self.resource_monitor = ResourceMonitor()
        
        # Intelligence and learning
        self.global_intelligence = WorkerIntelligence()
        self.workload_profile = WorkloadProfile()
        self.adaptation_strategies: Dict[str, Callable] = {}
        
        # Advanced autonomous task management
        self.autonomous_task_queue: asyncio.Queue = asyncio.Queue()
        self.preemption_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.intelligent_scheduler = IntelligentScheduler()
        self.task_preemptor = TaskPreemptor()
        
        # Confidence and learning systems
        self.worker_confidence_tracker = ConfidenceTracker()
        self.learning_feedback_loop = LearningFeedbackLoop()
        self.bayesian_optimizer = BayesianTaskOptimizer()
        
        # Advanced execution tracking
        self.task_execution_graph = ExecutionGraph()
        self.preemption_history: List[PreemptionEvent] = []
        
        # Performance tracking
        self.autonomous_metrics = AutonomousMetrics()
        self.learning_enabled = True
        self.optimization_interval = 300  # 5 minutes
        
        # Initialize autonomous capabilities
        self._initialize_adaptation_strategies()
        self._initialize_confidence_tracking()
        self._initialize_learning_systems()
        
    async def start(self):
        """Start the autonomous orchestrator with enhanced capabilities"""
        
        await super().start()
        
        self.logger.info("Starting Autonomous Orchestration Layer",
                        autonomy_level=self.autonomy_level.value)
        
        # Start autonomous components
        asyncio.create_task(self._autonomous_optimization_loop())
        asyncio.create_task(self._intelligent_task_distribution())
        asyncio.create_task(self._performance_learning_loop())
        asyncio.create_task(self._self_healing_monitor())
        asyncio.create_task(self._task_preemption_manager())
        asyncio.create_task(self._confidence_scoring_loop())
        asyncio.create_task(self._learning_feedback_processor())
        
        await self.audit_logger.log_event("autonomous_orchestrator_start", {
            "autonomy_level": self.autonomy_level.value,
            "learning_enabled": self.learning_enabled,
            "optimization_interval": self.optimization_interval
        })
        
    async def create_autonomous_campaign(self, 
                                       name: str, 
                                       targets: List[Dict],
                                       intelligence_driven: bool = True,
                                       adaptive_execution: bool = True,
                                       config: Dict[str, Any] = None) -> str:
        """Create campaign with autonomous intelligence and adaptation"""
        
        enhanced_config = config or {}
        enhanced_config.update({
            'autonomous_execution': True,
            'intelligence_driven': intelligence_driven,
            'adaptive_execution': adaptive_execution,
            'autonomy_level': self.autonomy_level.value
        })
        
        # Intelligent agent selection based on historical performance
        optimal_agents = await self._select_optimal_agents(targets, enhanced_config)
        
        # Create campaign with autonomous configuration
        campaign_id = await self.create_campaign(
            name=name,
            targets=targets,
            agent_requirements=optimal_agents,
            config=enhanced_config
        )
        
        # Record autonomous decision
        decision = AutonomousDecision(
            decision_id=str(uuid.uuid4()),
            decision_type="agent_selection",
            context={
                "campaign_id": campaign_id,
                "target_count": len(targets),
                "selected_agents": optimal_agents
            },
            rationale="Selected agents based on historical performance and target characteristics",
            confidence=0.85
        )
        self.decision_history.append(decision)
        
        self.autonomous_metrics.decisions_made.labels(
            decision_type="agent_selection",
            autonomy_level=self.autonomy_level.value
        ).inc()
        
        return campaign_id
    
    async def _select_optimal_agents(self, 
                                   targets: List[Dict], 
                                   config: Dict[str, Any]) -> List[AgentCapability]:
        """Intelligently select optimal agents based on targets and performance history"""
        
        # Analyze target characteristics
        target_analysis = await self._analyze_targets(targets)
        
        # Get performance data for different agent types
        agent_performance = self._get_agent_performance_data()
        
        # Select agents based on target requirements and performance
        selected_capabilities = []
        
        # Always include reconnaissance for unknown targets
        if target_analysis.get('unknown_targets', 0) > 0:
            selected_capabilities.append(AgentCapability.RECONNAISSANCE)
        
        # Add vulnerability scanning for web targets
        if target_analysis.get('web_targets', 0) > 0:
            selected_capabilities.append(AgentCapability.VULNERABILITY_SCANNING)
            
        # Add specialized agents based on performance
        best_performing_agents = sorted(
            agent_performance.items(),
            key=lambda x: x[1].get('success_rate', 0.0),
            reverse=True
        )[:3]  # Top 3 performing agents
        
        for agent_name, performance in best_performing_agents:
            if performance.get('success_rate', 0.0) > 0.7:  # Only high-performing agents
                # Map agent names to capabilities (simplified)
                capability_mapping = {
                    'web_crawler': AgentCapability.WEB_CRAWLING,
                    'port_scanner': AgentCapability.NETWORK_SCANNING,
                    'vuln_scanner': AgentCapability.VULNERABILITY_SCANNING
                }
                capability = capability_mapping.get(agent_name)
                if capability and capability not in selected_capabilities:
                    selected_capabilities.append(capability)
        
        return selected_capabilities
    
    async def _analyze_targets(self, targets: List[Dict]) -> Dict[str, int]:
        """Analyze target characteristics for intelligent agent selection"""
        
        analysis = {
            'web_targets': 0,
            'network_targets': 0,
            'unknown_targets': 0,
            'high_value_targets': 0
        }
        
        for target in targets:
            hostname = target.get('hostname', target.get('target', ''))
            ports = target.get('ports', [])
            
            # Classify target type
            if any(port in [80, 443, 8080, 8443] for port in ports):
                analysis['web_targets'] += 1
            elif ports:
                analysis['network_targets'] += 1
            else:
                analysis['unknown_targets'] += 1
                
            # Check for high-value indicators
            if any(keyword in hostname.lower() for keyword in ['admin', 'mgmt', 'api']):
                analysis['high_value_targets'] += 1
        
        return analysis
    
    def _get_agent_performance_data(self) -> Dict[str, Dict[str, float]]:
        """Get performance data for all agents"""
        
        performance_data = {}
        
        for agent_name, metadata in self.agent_registry.agents.items():
            # Get performance from global intelligence
            success_rate = self.global_intelligence.task_success_rates.get(agent_name, 0.5)
            avg_time = self.global_intelligence.performance_metrics.get(f"{agent_name}_avg_time", 30.0)
            
            performance_data[agent_name] = {
                'success_rate': success_rate,
                'avg_execution_time': avg_time,
                'last_used': metadata.last_seen
            }
        
        return performance_data
    
    async def _autonomous_optimization_loop(self):
        """Main autonomous optimization loop"""
        
        while self._running:
            try:
                # Update workload profile
                await self._update_workload_profile()
                
                # Perform autonomous optimizations
                if self.autonomy_level in [AutonomyLevel.HIGH, AutonomyLevel.MAXIMUM]:
                    await self._optimize_resource_allocation()
                    await self._adapt_concurrent_limits()
                    await self._optimize_agent_distribution()
                
                # Record optimization metrics
                self.autonomous_metrics.optimization_cycles.labels(
                    autonomy_level=self.autonomy_level.value
                ).inc()
                
                await asyncio.sleep(self.optimization_interval)
                
            except Exception as e:
                self.logger.error("Autonomous optimization error", error=str(e))
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _update_workload_profile(self):
        """Update current workload profile for optimization decisions"""
        
        # Count active tasks by type
        task_distribution = defaultdict(int)
        for context in self.active_executions.values():
            task_distribution[context.agent_name] += 1
        
        self.workload_profile.total_active_tasks = len(self.active_executions)
        self.workload_profile.task_type_distribution = dict(task_distribution)
        
        # Update resource utilization
        self.workload_profile.resource_utilization = {
            'cpu': await self.resource_monitor.get_cpu_usage(),
            'memory': await self.resource_monitor.get_memory_usage(),
            'active_agents': len(self.active_executions),
            'queue_depth': self.autonomous_task_queue.qsize()
        }
        
        # Update success rates from global intelligence
        self.workload_profile.success_rates_by_type = (
            self.global_intelligence.task_success_rates.copy()
        )
    
    async def _optimize_resource_allocation(self):
        """Autonomously optimize resource allocation"""
        
        current_cpu = self.workload_profile.resource_utilization.get('cpu', 0.0)
        current_memory = self.workload_profile.resource_utilization.get('memory', 0.0)
        
        # Scale up if resources are available and queue is building
        if (current_cpu < 0.7 and current_memory < 0.7 and 
            self.autonomous_task_queue.qsize() > 10):
            
            new_limit = min(self.max_concurrent_agents + 4, 48)  # Max 48 agents
            if new_limit > self.max_concurrent_agents:
                await self._adapt_concurrent_limit(new_limit)
                
                decision = AutonomousDecision(
                    decision_id=str(uuid.uuid4()),
                    decision_type="scale_up",
                    context={"old_limit": self.max_concurrent_agents, "new_limit": new_limit},
                    rationale=f"Scaled up due to low resource usage ({current_cpu:.2f} CPU, {current_memory:.2f} memory) and queue backlog",
                    confidence=0.8
                )
                self.decision_history.append(decision)
        
        # Scale down if resource usage is high
        elif current_cpu > 0.85 or current_memory > 0.85:
            new_limit = max(self.max_concurrent_agents - 4, 8)  # Min 8 agents
            if new_limit < self.max_concurrent_agents:
                await self._adapt_concurrent_limit(new_limit)
                
                decision = AutonomousDecision(
                    decision_id=str(uuid.uuid4()),
                    decision_type="scale_down",
                    context={"old_limit": self.max_concurrent_agents, "new_limit": new_limit},
                    rationale=f"Scaled down due to high resource usage ({current_cpu:.2f} CPU, {current_memory:.2f} memory)",
                    confidence=0.9
                )
                self.decision_history.append(decision)
    
    async def _adapt_concurrent_limit(self, new_limit: int):
        """Adapt concurrent execution limits"""
        
        old_limit = self.max_concurrent_agents
        self.max_concurrent_agents = new_limit
        self.execution_semaphore = asyncio.Semaphore(new_limit)
        
        self.logger.info("Adapted concurrent agent limit",
                        old_limit=old_limit,
                        new_limit=new_limit)
        
        self.autonomous_metrics.resource_adaptations.labels(
            adaptation_type="concurrent_limit",
            autonomy_level=self.autonomy_level.value
        ).inc()
    
    async def _intelligent_task_distribution(self):
        """Intelligently distribute tasks to optimal agents"""
        
        while self._running:
            try:
                # Get tasks from autonomous queue
                try:
                    task_batch = []
                    for _ in range(min(10, self.autonomous_task_queue.qsize())):
                        task = await asyncio.wait_for(
                            self.autonomous_task_queue.get(), timeout=1.0
                        )
                        task_batch.append(task)
                except asyncio.TimeoutError:
                    await asyncio.sleep(1)
                    continue
                
                if not task_batch:
                    await asyncio.sleep(1)
                    continue
                
                # Intelligently distribute tasks
                for task in task_batch:
                    optimal_agent = await self._select_optimal_agent_for_task(task)
                    if optimal_agent:
                        await optimal_agent.add_task(task)
                        
                        self.autonomous_metrics.intelligent_distributions.labels(
                            agent_type=optimal_agent.agent_type,
                            autonomy_level=self.autonomy_level.value
                        ).inc()
                
            except Exception as e:
                self.logger.error("Intelligent task distribution error", error=str(e))
                await asyncio.sleep(5)
    
    async def _select_optimal_agent_for_task(self, task: AgentTask) -> Optional[AutonomousWorker]:
        """Select the optimal agent for a specific task"""
        
        if not self.autonomous_workers:
            return None
        
        # Score agents based on multiple factors
        agent_scores = {}
        
        for agent_id, agent in self.autonomous_workers.items():
            if agent.status.value == "terminated":
                continue
            
            score = 0.0
            
            # Factor 1: Agent capability match
            if agent.has_capability(task.task_type):
                score += 40.0
            
            # Factor 2: Historical success rate for this task type
            success_rate = agent.intelligence.task_success_rates.get(task.task_type, 0.5)
            score += success_rate * 30.0
            
            # Factor 3: Current load (prefer less loaded agents)
            current_load = agent.task_queue.qsize()
            load_penalty = min(current_load * 5.0, 20.0)
            score -= load_penalty
            
            # Factor 4: Agent status (prefer idle agents)
            if agent.status.value == "idle":
                score += 10.0
            elif agent.status.value == "running":
                score += 5.0
            
            agent_scores[agent_id] = score
        
        # Select agent with highest score
        if agent_scores:
            best_agent_id = max(agent_scores.keys(), key=lambda x: agent_scores[x])
            return self.autonomous_workers[best_agent_id]
        
        return None
    
    async def _performance_learning_loop(self):
        """Continuously learn from performance data to improve decisions"""
        
        while self._running:
            try:
                if not self.learning_enabled:
                    await asyncio.sleep(60)
                    continue
                
                # Collect performance data from all agents
                await self._collect_global_intelligence()
                
                # Analyze patterns and update strategies
                await self._analyze_performance_patterns()
                
                # Update global intelligence
                await self._update_global_intelligence()
                
                self.autonomous_metrics.learning_cycles.labels(
                    autonomy_level=self.autonomy_level.value
                ).inc()
                
                await asyncio.sleep(120)  # Learn every 2 minutes
                
            except Exception as e:
                self.logger.error("Performance learning error", error=str(e))
                await asyncio.sleep(60)
    
    async def _collect_global_intelligence(self):
        """Collect intelligence from all autonomous workers"""
        
        for worker in self.autonomous_workers.values():
            # Merge task success rates
            for task_type, rate in worker.intelligence.task_success_rates.items():
                current_global = self.global_intelligence.task_success_rates.get(task_type, 0.5)
                # Weighted average with more weight on recent data
                self.global_intelligence.task_success_rates[task_type] = (
                    current_global * 0.7 + rate * 0.3
                )
            
            # Collect performance metrics
            worker_metrics = await worker.get_autonomous_status()
            performance_data = worker_metrics.get('performance_metrics', {})
            
            for metric_name, value in performance_data.items():
                self.global_intelligence.performance_metrics[f"global_{metric_name}"] = (
                    self.global_intelligence.performance_metrics.get(f"global_{metric_name}", 0.0) * 0.8 + 
                    value * 0.2
                )
    
    async def _analyze_performance_patterns(self):
        """Analyze performance patterns to identify optimization opportunities"""
        
        # Identify underperforming task types
        underperforming_tasks = [
            task_type for task_type, rate in self.global_intelligence.task_success_rates.items()
            if rate < 0.6
        ]
        
        if underperforming_tasks:
            self.logger.info("Identified underperforming task types",
                           tasks=underperforming_tasks)
            
            # Create improvement strategies
            for task_type in underperforming_tasks:
                strategy = await self._create_improvement_strategy(task_type)
                self.adaptation_strategies[task_type] = strategy
    
    async def _create_improvement_strategy(self, task_type: str) -> Callable:
        """Create improvement strategy for underperforming task type"""
        
        async def improvement_strategy(task: AgentTask) -> AgentTask:
            # Increase timeout for problematic tasks
            task.parameters['timeout'] = task.parameters.get('timeout', 30) * 1.5
            
            # Add retry logic
            task.max_retries = min(task.max_retries + 1, 5)
            
            # Reduce parallel execution
            task.parameters['parallel_execution'] = False
            
            return task
        
        return improvement_strategy
    
    async def _self_healing_monitor(self):
        """Monitor system health and perform autonomous healing"""
        
        while self._running:
            try:
                # Check for failed agents
                failed_agents = [
                    agent_id for agent_id, agent in self.autonomous_workers.items()
                    if agent.status.value == "error"
                ]
                
                # Attempt to restart failed agents
                for agent_id in failed_agents:
                    await self._restart_failed_agent(agent_id)
                
                # Check for resource exhaustion
                cpu_usage = await self.resource_monitor.get_cpu_usage()
                memory_usage = await self.resource_monitor.get_memory_usage()
                
                if cpu_usage > 0.95 or memory_usage > 0.95:
                    await self._handle_resource_exhaustion(cpu_usage, memory_usage)
                
                # Check for queue buildup
                if self.autonomous_task_queue.qsize() > 100:
                    await self._handle_queue_buildup()
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error("Self-healing monitor error", error=str(e))
                await asyncio.sleep(30)
    
    async def _restart_failed_agent(self, agent_id: str):
        """Restart a failed autonomous agent"""
        
        if agent_id not in self.autonomous_workers:
            return
        
        failed_agent = self.autonomous_workers[agent_id]
        
        try:
            # Stop the failed agent
            await failed_agent.stop()
            
            # Create new agent with same configuration
            new_agent = AutonomousWorker(
                agent_id=str(uuid.uuid4()),
                config=failed_agent.config,
                autonomous_config=failed_agent.autonomous_config
            )
            
            # Start new agent
            await new_agent.start()
            
            # Replace in registry
            self.autonomous_workers[new_agent.agent_id] = new_agent
            del self.autonomous_workers[agent_id]
            
            self.logger.info("Restarted failed agent",
                           old_agent_id=agent_id[:8],
                           new_agent_id=new_agent.agent_id[:8])
            
            self.autonomous_metrics.agent_restarts.labels(
                restart_reason="failure",
                autonomy_level=self.autonomy_level.value
            ).inc()
            
        except Exception as e:
            self.logger.error("Failed to restart agent",
                            agent_id=agent_id[:8],
                            error=str(e))
    
    async def _handle_resource_exhaustion(self, cpu_usage: float, memory_usage: float):
        """Handle resource exhaustion by reducing load"""
        
        self.logger.warning("Resource exhaustion detected",
                          cpu_usage=cpu_usage,
                          memory_usage=memory_usage)
        
        # Reduce concurrent agents
        new_limit = max(self.max_concurrent_agents // 2, 4)
        await self._adapt_concurrent_limit(new_limit)
        
        # Pause lower priority tasks
        paused_count = 0
        for worker in self.autonomous_workers.values():
            if worker.status.value == "running":
                await worker.pause()
                paused_count += 1
                if paused_count >= len(self.autonomous_workers) // 2:
                    break
        
        decision = AutonomousDecision(
            decision_id=str(uuid.uuid4()),
            decision_type="resource_protection",
            context={
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "agents_paused": paused_count
            },
            rationale="Reduced load due to resource exhaustion",
            confidence=1.0
        )
        self.decision_history.append(decision)
    
    async def get_autonomous_status(self) -> Dict[str, Any]:
        """Get comprehensive autonomous orchestrator status"""
        
        base_status = await super().get_campaign_status("system")  # Get base status
        
        autonomous_status = {
            'autonomy_config': {
                'autonomy_level': self.autonomy_level.value,
                'learning_enabled': self.learning_enabled,
                'optimization_interval': self.optimization_interval
            },
            'autonomous_workers': {
                'total_workers': len(self.autonomous_workers),
                'active_workers': len([w for w in self.autonomous_workers.values() 
                                     if w.status.value not in ["terminated", "error"]]),
                'worker_summary': {
                    worker_id[:8]: await worker.get_autonomous_status()
                    for worker_id, worker in list(self.autonomous_workers.items())[:5]  # Show first 5
                }
            },
            'intelligence_summary': {
                'task_success_rates': dict(list(self.global_intelligence.task_success_rates.items())[:10]),
                'performance_metrics': dict(list(self.global_intelligence.performance_metrics.items())[:10]),
                'last_updated': self.global_intelligence.last_updated.isoformat()
            },
            'workload_profile': {
                'total_active_tasks': self.workload_profile.total_active_tasks,
                'resource_utilization': self.workload_profile.resource_utilization,
                'task_distribution': dict(list(self.workload_profile.task_type_distribution.items())[:10])
            },
            'recent_decisions': [
                {
                    'decision_type': d.decision_type,
                    'rationale': d.rationale,
                    'confidence': d.confidence,
                    'timestamp': d.timestamp.isoformat()
                }
                for d in self.decision_history[-10:]  # Last 10 decisions
            ],
            'adaptation_strategies': list(self.adaptation_strategies.keys())
        }
        
        return autonomous_status
    
    def _initialize_adaptation_strategies(self):
        """Initialize adaptation strategies for different scenarios"""
        
        async def high_failure_rate_strategy(task: AgentTask) -> AgentTask:
            task.parameters['timeout'] = task.parameters.get('timeout', 30) * 2
            task.max_retries += 1
            return task
        
        async def resource_constrained_strategy(task: AgentTask) -> AgentTask:
            task.parameters['parallel_execution'] = False
            task.priority = max(1, task.priority - 1)
            return task
        
        self.adaptation_strategies.update({
            'high_failure_rate': high_failure_rate_strategy,
            'resource_constrained': resource_constrained_strategy
        })
    
    # =============================================================================
    # ENHANCED AUTONOMOUS FEATURES - Agent-Led Task Prioritization v2.1
    # =============================================================================
    
    async def enable_agent_led_prioritization(self):
        """Enable agent-led task prioritization mode"""
        self.agent_led_mode = True
        self.logger.info("🤖 Agent-Led Prioritization Mode: ENABLED")
        
        # Start agent consensus and collaboration tasks
        asyncio.create_task(self._agent_consensus_loop())
        asyncio.create_task(self._collaborative_learning_loop())
        asyncio.create_task(self._predictive_resource_allocator())
    
    async def _agent_consensus_loop(self):
        """Agents collaborate to prioritize tasks autonomously"""
        while self._running and getattr(self, 'agent_led_mode', False):
            try:
                # Get pending tasks
                pending_tasks = [
                    task for task in self.execution_contexts.values()
                    if task.status == ExecutionStatus.PENDING
                ]
                
                if not pending_tasks:
                    await asyncio.sleep(5)
                    continue
                
                # Multi-agent collaborative prioritization
                consensus_priorities = await self._multi_agent_consensus(pending_tasks)
                
                # Apply autonomous re-prioritization
                await self._apply_agent_consensus(consensus_priorities)
                
                self.logger.info("🧠 Agent consensus applied", 
                               prioritized_tasks=len(consensus_priorities))
                
                await asyncio.sleep(10)  # Consensus every 10 seconds
                
            except Exception as e:
                self.logger.error("Agent consensus error", error=str(e))
                await asyncio.sleep(30)
    
    async def _multi_agent_consensus(self, tasks: List[ExecutionContext]) -> Dict[str, float]:
        """Multi-agent collaborative task prioritization"""
        consensus_scores = defaultdict(list)
        
        # Each autonomous worker votes on task priorities
        for worker_id, worker in self.autonomous_workers.items():
            try:
                # Get worker's priority assessment
                priorities = await worker.assess_task_priorities(tasks)
                
                for task_id, score in priorities.items():
                    consensus_scores[task_id].append(score)
                    
            except Exception as e:
                self.logger.warning(f"Worker {worker_id} consensus error", error=str(e))
        
        # Calculate weighted consensus (agents with higher confidence get more weight)
        final_priorities = {}
        for task_id, scores in consensus_scores.items():
            if scores:
                # Weighted average based on agent confidence
                weights = [self._get_agent_confidence(agent_id) for agent_id in range(len(scores))]
                weighted_avg = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                final_priorities[task_id] = weighted_avg
            else:
                final_priorities[task_id] = 5.0  # Default priority
        
        return final_priorities
    
    async def _apply_agent_consensus(self, priorities: Dict[str, float]):
        """Apply agent consensus priorities to task execution"""
        for task_id, priority in priorities.items():
            if task_id in self.execution_contexts:
                context = self.execution_contexts[task_id]
                old_priority = context.priority
                context.priority = int(priority)
                
                if abs(old_priority - context.priority) > 1:
                    self.logger.info("🎯 Agent-led priority adjustment",
                                   task_id=task_id[:8],
                                   old_priority=old_priority,
                                   new_priority=context.priority)
    
    async def _collaborative_learning_loop(self):
        """Agents share knowledge and learn collaboratively"""
        while self._running and getattr(self, 'agent_led_mode', False):
            try:
                # Collect insights from all agents
                collective_insights = await self._gather_collective_insights()
                
                # Share insights across all agents
                await self._distribute_collective_insights(collective_insights)
                
                # Update global intelligence with collaborative learning
                await self._update_collaborative_intelligence(collective_insights)
                
                self.logger.info("🧠 Collaborative learning cycle complete",
                               insights_count=len(collective_insights))
                
                await asyncio.sleep(60)  # Learn every minute
                
            except Exception as e:
                self.logger.error("Collaborative learning error", error=str(e))
                await asyncio.sleep(120)
    
    async def _gather_collective_insights(self) -> Dict[str, Any]:
        """Gather insights from all autonomous agents"""
        insights = {
            'performance_patterns': {},
            'failure_modes': {},
            'optimization_strategies': {},
            'resource_patterns': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        for worker_id, worker in self.autonomous_workers.items():
            try:
                worker_insights = await worker.get_learning_insights()
                insights['performance_patterns'][worker_id] = worker_insights.get('performance', {})
                insights['failure_modes'][worker_id] = worker_insights.get('failures', {})
                insights['optimization_strategies'][worker_id] = worker_insights.get('optimizations', {})
                insights['resource_patterns'][worker_id] = worker_insights.get('resources', {})
            except Exception as e:
                self.logger.warning(f"Failed to gather insights from {worker_id}", error=str(e))
        
        return insights
    
    async def _distribute_collective_insights(self, insights: Dict[str, Any]):
        """Distribute collective insights to all agents for learning"""
        for worker_id, worker in self.autonomous_workers.items():
            try:
                await worker.receive_collective_insights(insights)
            except Exception as e:
                self.logger.warning(f"Failed to distribute insights to {worker_id}", error=str(e))
    
    async def _predictive_resource_allocator(self):
        """Claude/Qwen powered predictive resource allocation"""
        while self._running and getattr(self, 'agent_led_mode', False):
            try:
                # Analyze current resource patterns
                resource_analysis = await self._analyze_resource_patterns()
                
                # Use AI reasoning for resource prediction
                prediction = await self._ai_resource_prediction(resource_analysis)
                
                # Apply predictive resource adjustments
                await self._apply_predictive_adjustments(prediction)
                
                self.logger.info("🔮 Predictive resource allocation applied",
                               adjustments=len(prediction.get('adjustments', [])))
                
                await asyncio.sleep(120)  # Predict every 2 minutes
                
            except Exception as e:
                self.logger.error("Predictive resource allocation error", error=str(e))
                await asyncio.sleep(300)
    
    async def _ai_resource_prediction(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Use Claude/Qwen for intelligent resource prediction"""
        try:
            # Construct reasoning prompt for AI
            prompt = f"""
            Analyze this XORB resource utilization data and predict optimal adjustments:
            
            Current Analysis: {json.dumps(analysis, indent=2)}
            
            Provide predictions for:
            1. Resource bottlenecks in next 10 minutes
            2. Optimal agent concurrency adjustments
            3. Task priority rebalancing recommendations
            4. Performance optimization strategies
            
            Respond with JSON format focusing on immediate actionable adjustments.
            """
            
            # This would integrate with Claude/Qwen - for now return intelligent defaults
            prediction = {
                'adjustments': [],
                'confidence': 0.8,
                'reasoning': 'AI-powered resource prediction',
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Analyze patterns and generate predictions
            if analysis.get('cpu_utilization', 0) > 0.8:
                prediction['adjustments'].append({
                    'type': 'scale_down_concurrency',
                    'value': max(1, self.max_concurrent_agents // 2),
                    'reason': 'High CPU utilization detected'
                })
            
            if analysis.get('queue_length', 0) > 20:
                prediction['adjustments'].append({
                    'type': 'increase_worker_count',
                    'value': min(32, len(self.autonomous_workers) + 2),
                    'reason': 'Queue backlog growing'
                })
            
            return prediction
            
        except Exception as e:
            self.logger.error("AI resource prediction error", error=str(e))
            return {'adjustments': [], 'confidence': 0.0}
    
    def _get_agent_confidence(self, agent_index: int) -> float:
        """Get confidence score for an agent"""
        # This would be based on historical performance
        return 0.8  # Default confidence
    
    async def _analyze_resource_patterns(self) -> Dict[str, Any]:
        """Analyze current resource utilization patterns"""
        return {
            'cpu_utilization': 0.6,  # Would be real metrics
            'memory_utilization': 0.4,
            'queue_length': len(self.execution_contexts),
            'active_agents': len(self.autonomous_workers),
            'success_rate_trend': 0.85
        }
    
    async def _apply_predictive_adjustments(self, prediction: Dict[str, Any]):
        """Apply AI-predicted resource adjustments"""
        for adjustment in prediction.get('adjustments', []):
            try:
                adj_type = adjustment['type']
                adj_value = adjustment['value']
                
                if adj_type == 'scale_down_concurrency':
                    await self._adapt_concurrent_limit(adj_value)
                elif adj_type == 'increase_worker_count':
                    await self._spawn_additional_workers(adj_value - len(self.autonomous_workers))
                
                self.logger.info("🎛️ Applied predictive adjustment",
                               type=adj_type,
                               value=adj_value,
                               reason=adjustment.get('reason'))
                
            except Exception as e:
                self.logger.error("Failed to apply adjustment", 
                                adjustment=adjustment, error=str(e))
    
    async def _spawn_additional_workers(self, count: int):
        """Spawn additional autonomous workers"""
        for i in range(count):
            worker_id = f"auto_worker_{uuid.uuid4().hex[:8]}"
            worker = AutonomousWorker(
                worker_id=worker_id,
                orchestrator=self,
                autonomy_level=self.autonomy_level
            )
            self.autonomous_workers[worker_id] = worker
            asyncio.create_task(worker.start())
            
            self.logger.info("🔧 Spawned additional autonomous worker",
                           worker_id=worker_id)


class AutonomousMetrics:
    """Metrics collector for autonomous orchestrator capabilities"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Autonomous decision metrics
        self.decisions_made = Counter(
            'xorb_autonomous_decisions_total',
            'Total autonomous decisions made',
            ['decision_type', 'autonomy_level'],
            registry=self.registry
        )
        
        self.optimization_cycles = Counter(
            'xorb_optimization_cycles_total',
            'Total optimization cycles completed',
            ['autonomy_level'],
            registry=self.registry
        )
        
        self.resource_adaptations = Counter(
            'xorb_resource_adaptations_total',
            'Total resource adaptations performed',
            ['adaptation_type', 'autonomy_level'],
            registry=self.registry
        )
        
        self.intelligent_distributions = Counter(
            'xorb_intelligent_distributions_total',
            'Total intelligent task distributions',
            ['agent_type', 'autonomy_level'],
            registry=self.registry
        )
        
        self.learning_cycles = Counter(
            'xorb_learning_cycles_total',
            'Total learning cycles completed',
            ['autonomy_level'],
            registry=self.registry
        )
        
        self.agent_restarts = Counter(
            'xorb_agent_restarts_total',
            'Total agent restarts performed',
            ['restart_reason', 'autonomy_level'],
            registry=self.registry
        )




    async def _task_preemption_manager(self):
        """Manage task preemption based on priority and resource constraints"""
        
        while self._running:
            try:
                # Check for preemption opportunities
                preemption_candidates = await self._identify_preemption_candidates()
                
                for candidate in preemption_candidates:
                    await self._execute_task_preemption(candidate)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error("Task preemption manager error", error=str(e))
                await asyncio.sleep(10)
    
    async def _identify_preemption_candidates(self) -> List[PreemptionEvent]:
        """Identify candidates for task preemption"""
        
        candidates = []
        
        # Check high-priority tasks waiting while low-priority tasks run
        waiting_tasks = await self._get_waiting_high_priority_tasks()
        running_tasks = await self._get_running_low_priority_tasks()
        
        for waiting_task in waiting_tasks:
            for running_task in running_tasks:
                if waiting_task.priority > running_task.priority + 2:  # Significant priority difference
                    
                    # Calculate preemption score
                    preemption_score = await self._calculate_preemption_score(
                        waiting_task, running_task
                    )
                    
                    if preemption_score > 0.7:  # High confidence in benefit
                        candidates.append(PreemptionEvent(
                            task_id=running_task.task_id,
                            agent_id=running_task.agent_id,
                            reason=f"Higher priority task waiting (priority {waiting_task.priority} vs {running_task.priority})",
                            priority_score=preemption_score,
                            recovery_strategy="requeue_with_backoff"
                        ))
        
        # Check resource-constrained scenarios
        if await self._is_resource_constrained():
            low_confidence_tasks = await self._get_low_confidence_running_tasks()
            
            for task in low_confidence_tasks:
                confidence = self.worker_confidence_tracker.get_task_confidence(
                    task.agent_id, task.task_type
                )
                
                if confidence < 0.4:  # Very low confidence
                    candidates.append(PreemptionEvent(
                        task_id=task.task_id,
                        agent_id=task.agent_id,
                        reason=f"Low confidence task consuming resources (confidence: {confidence:.2f})",
                        priority_score=0.8,
                        recovery_strategy="retry_with_different_agent"
                    ))
        
        return sorted(candidates, key=lambda x: x.priority_score, reverse=True)
    
    async def _calculate_preemption_score(self, waiting_task, running_task) -> float:
        """Calculate score for preemption decision"""
        
        score = 0.0
        
        # Priority difference factor (0-40 points)
        priority_diff = waiting_task.priority - running_task.priority
        score += min(priority_diff * 10, 40)
        
        # Time factors (0-20 points)
        running_time = (datetime.utcnow() - running_task.started_at).total_seconds()
        waiting_time = (datetime.utcnow() - waiting_task.created_at).total_seconds()
        
        # Prefer preempting long-running tasks
        if running_time > 300:  # 5 minutes
            score += 10
        
        # Factor in waiting time
        if waiting_time > 180:  # 3 minutes
            score += 10
        
        # Confidence factors (0-20 points)
        running_confidence = self.worker_confidence_tracker.get_task_confidence(
            running_task.agent_id, running_task.task_type
        )
        waiting_confidence = self.worker_confidence_tracker.get_task_confidence(
            waiting_task.agent_id, waiting_task.task_type
        )
        
        confidence_diff = waiting_confidence - running_confidence
        score += confidence_diff * 20
        
        # Resource utilization factor (0-20 points)
        current_cpu = await self.resource_monitor.get_cpu_usage()
        current_memory = await self.resource_monitor.get_memory_usage()
        
        if current_cpu > 0.8 or current_memory > 0.8:
            score += 15  # High resource pressure
        
        return min(score / 100.0, 1.0)  # Normalize to 0-1
    
    async def _execute_task_preemption(self, preemption_event: PreemptionEvent):
        """Execute task preemption with proper cleanup and recovery"""
        
        try:
            # Find the running task
            task_context = self.active_executions.get(preemption_event.task_id)
            if not task_context:
                return
            
            self.logger.info("Executing task preemption",
                           task_id=preemption_event.task_id[:8],
                           reason=preemption_event.reason,
                           score=preemption_event.priority_score)
            
            # Gracefully stop the task
            await self._graceful_task_stop(task_context)
            
            # Apply recovery strategy
            if preemption_event.recovery_strategy == "requeue_with_backoff":
                await self._requeue_task_with_backoff(task_context)
            elif preemption_event.recovery_strategy == "retry_with_different_agent":
                await self._retry_with_different_agent(task_context)
            
            # Record preemption
            self.preemption_history.append(preemption_event)
            if len(self.preemption_history) > 1000:
                self.preemption_history = self.preemption_history[-800:]
            
            # Update metrics
            self.autonomous_metrics.task_preemptions.labels(
                reason=preemption_event.reason,
                autonomy_level=self.autonomy_level.value
            ).inc()
            
            # Record decision for learning
            decision = AutonomousDecision(
                decision_id=str(uuid.uuid4()),
                decision_type="task_preemption",
                context={
                    "preempted_task": preemption_event.task_id,
                    "reason": preemption_event.reason,
                    "priority_score": preemption_event.priority_score
                },
                rationale=preemption_event.reason,
                confidence=preemption_event.priority_score
            )
            self.decision_history.append(decision)
            
        except Exception as e:
            self.logger.error("Task preemption failed",
                            task_id=preemption_event.task_id[:8],
                            error=str(e))
    
    async def _confidence_scoring_loop(self):
        """Continuously update worker confidence scores"""
        
        while self._running:
            try:
                # Update confidence scores for all workers
                for worker_id, worker in self.autonomous_workers.items():
                    await self._update_worker_confidence(worker_id, worker)
                
                # Update global confidence metrics
                await self._update_global_confidence_metrics()
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error("Confidence scoring error", error=str(e))
                await asyncio.sleep(60)
    
    async def _update_worker_confidence(self, worker_id: str, worker: AutonomousWorker):
        """Update confidence metrics for a specific worker"""
        
        try:
            worker_status = await worker.get_autonomous_status()
            performance_metrics = worker_status.get('performance_metrics', {})
            
            # Calculate task-type specific confidences
            task_confidences = {}
            for task_type, success_rate in worker.intelligence.task_success_rates.items():
                # Base confidence from success rate
                base_confidence = success_rate
                
                # Adjust for consistency (lower variance = higher confidence)
                recent_results = self._get_recent_task_results(worker_id, task_type)
                variance_penalty = self._calculate_result_variance(recent_results) * 0.2
                
                # Adjust for resource efficiency
                efficiency_bonus = self._calculate_resource_efficiency(worker_id) * 0.1
                
                task_confidences[task_type] = max(0.0, min(1.0, 
                    base_confidence - variance_penalty + efficiency_bonus
                ))
            
            # Calculate overall confidence
            historical_accuracy = performance_metrics.get('recent_success_rate', 0.8)
            resource_reliability = 1.0 - worker_status.get('intelligence_summary', {}).get('security_violations_prevented', 0) * 0.1
            failure_recovery_rate = self._calculate_recovery_rate(worker_id)
            
            overall_confidence = (
                historical_accuracy * 0.4 +
                resource_reliability * 0.3 +
                failure_recovery_rate * 0.3
            )
            
            # Calculate confidence trend
            previous_confidence = self.worker_confidence_tracker.get_worker_confidence(worker_id)
            confidence_trend = overall_confidence - previous_confidence.overall_confidence
            
            # Update confidence tracking
            confidence_metrics = ConfidenceMetrics(
                worker_id=worker_id,
                task_type_confidences=task_confidences,
                historical_accuracy=historical_accuracy,
                resource_reliability=resource_reliability,
                failure_recovery_rate=failure_recovery_rate,
                overall_confidence=overall_confidence,
                confidence_trend=confidence_trend,
                last_updated=datetime.utcnow()
            )
            
            self.worker_confidence_tracker.update_confidence(worker_id, confidence_metrics)
            
        except Exception as e:
            self.logger.error("Worker confidence update failed",
                            worker_id=worker_id[:8],
                            error=str(e))
    
    async def _learning_feedback_processor(self):
        """Process learning feedback to improve decision making"""
        
        while self._running:
            try:
                # Collect feedback from recent decisions
                feedback_data = await self._collect_decision_feedback()
                
                # Process feedback through learning system
                improvements = await self.learning_feedback_loop.process_feedback(feedback_data)
                
                # Apply learned improvements
                for improvement in improvements:
                    await self._apply_learning_improvement(improvement)
                
                # Update Bayesian optimization models
                await self.bayesian_optimizer.update_models(feedback_data)
                
                await asyncio.sleep(300)  # Process every 5 minutes
                
            except Exception as e:
                self.logger.error("Learning feedback processing error", error=str(e))
                await asyncio.sleep(300)
    
    async def _collect_decision_feedback(self) -> List[Dict[str, Any]]:
        """Collect feedback on recent autonomous decisions"""
        
        feedback_data = []
        
        # Analyze outcomes of recent decisions
        recent_decisions = [d for d in self.decision_history 
                          if (datetime.utcnow() - d.timestamp).total_seconds() < 1800]  # Last 30 minutes
        
        for decision in recent_decisions:
            if decision.outcome:  # Has outcome data
                feedback = {
                    'decision_id': decision.decision_id,
                    'decision_type': decision.decision_type,
                    'predicted_confidence': decision.confidence,
                    'actual_outcome': decision.outcome,
                    'context': decision.context,
                    'feedback_score': decision.feedback_score or self._calculate_decision_feedback_score(decision)
                }
                feedback_data.append(feedback)
        
        return feedback_data
    
    def _calculate_decision_feedback_score(self, decision: AutonomousDecision) -> float:
        """Calculate feedback score for a decision based on its outcome"""
        
        if decision.outcome == "success":
            return 1.0
        elif decision.outcome == "partial_success":
            return 0.6
        elif decision.outcome == "failure":
            return 0.0
        else:
            return 0.5  # Unknown outcome
    
    def _initialize_confidence_tracking(self):
        """Initialize confidence tracking system"""
        self.worker_confidence_tracker = ConfidenceTracker()
        
    def _initialize_learning_systems(self):
        """Initialize learning and optimization systems"""
        self.learning_feedback_loop = LearningFeedbackLoop()
        self.bayesian_optimizer = BayesianTaskOptimizer()
        self.task_preemptor = TaskPreemptor()


class IntelligentScheduler:
    """Intelligent task scheduling based on patterns and performance"""
    
    def __init__(self):
        self.logger = structlog.get_logger("IntelligentScheduler")
        
    async def schedule_tasks(self, 
                           tasks: List[AgentTask],
                           agent_performance: Dict[str, Dict[str, float]],
                           resource_constraints: Dict[str, float]) -> List[AgentTask]:
        """Intelligently schedule tasks based on multiple factors"""
        
        def task_priority_score(task: AgentTask) -> float:
            base_priority = task.priority
            
            # Factor in deadline urgency
            if task.deadline:
                time_to_deadline = (task.deadline - datetime.utcnow()).total_seconds()
                urgency_multiplier = max(0.1, min(2.0, 3600 / max(time_to_deadline, 60)))
                base_priority *= urgency_multiplier
            
            # Factor in resource requirements
            cpu_req = task.parameters.get('estimated_cpu', 0.1)
            memory_req = task.parameters.get('estimated_memory', 0.05)
            
            available_cpu = 1.0 - resource_constraints.get('cpu', 0.0)
            available_memory = 1.0 - resource_constraints.get('memory', 0.0)
            
            resource_score = min(available_cpu / max(cpu_req, 0.01),
                                available_memory / max(memory_req, 0.01))
            
            return base_priority * resource_score
        
        # Sort tasks by intelligent priority score
        return sorted(tasks, key=task_priority_score, reverse=True)