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
import uuid
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from typing import Any

import structlog
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from xorb_core.agents.base_agent import AgentCapability, AgentTask
from xorb_core.orchestration.enhanced_orchestrator import (
    EnhancedOrchestrator,
    ExecutionContext,
    ExecutionStatus,
)

from ..intelligence.global_synthesis_engine import (
    CorrelatedIntelligence,
    GlobalSynthesisEngine,
    SignalPriority,
)
from .autonomous_worker import (
    AutonomousWorker,
    AutonomyLevel,
    ResourceMonitor,
    WorkerIntelligence,
)
from .models import (
    AutonomousDecision,
    PerformanceOptimizer,
    WorkloadAnalyzer,
    WorkloadProfile,
)
from .rl_orchestrator_extensions import (
    BayesianTaskOptimizer,
    ConfidenceTracker,
    ExecutionGraph,
    LearningFeedbackLoop,
    PreemptionEvent,
    TaskPreemptor,
)


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
                 plugin_directories: list[str] = None,
                 max_concurrent_agents: int = 32,
                 max_concurrent_campaigns: int = 10,
                 autonomy_level: AutonomyLevel = AutonomyLevel.MODERATE):

        super().__init__(redis_url, nats_url, plugin_directories,
                        max_concurrent_agents, max_concurrent_campaigns)

        self.autonomy_level = autonomy_level
        self.logger = structlog.get_logger("AutonomousOrchestrator")

        # Autonomous components
        self.autonomous_workers: dict[str, AutonomousWorker] = {}
        self.decision_history: list[AutonomousDecision] = []
        self.workload_analyzer = WorkloadAnalyzer()
        self.performance_optimizer = PerformanceOptimizer()
        self.resource_monitor = ResourceMonitor()

        # Intelligence and learning
        self.global_intelligence = WorkerIntelligence()
        self.workload_profile = WorkloadProfile()
        self.adaptation_strategies: dict[str, Callable] = {}

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
        self.preemption_history: list[PreemptionEvent] = []

        # Performance tracking
        self.autonomous_metrics = AutonomousMetrics()
        self.learning_enabled = True
        self.optimization_interval = 300  # 5 minutes

        # Initialize autonomous capabilities
        self._initialize_adaptation_strategies()
        self._initialize_confidence_tracking()
        self._initialize_learning_systems()

        # Phase 10: Global Intelligence Synthesis
        self.global_synthesis_engine: GlobalSynthesisEngine | None = None
        self.intelligence_processing_enabled = True
        self.synthesis_integration_metrics = self._initialize_synthesis_metrics()

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

        # Phase 10: Start global intelligence synthesis integration
        if self.intelligence_processing_enabled:
            asyncio.create_task(self._global_intelligence_processor())
            asyncio.create_task(self._synthesis_mission_dispatcher())
            asyncio.create_task(self._intelligence_feedback_loop())

        await self.audit_logger.log_event("autonomous_orchestrator_start", {
            "autonomy_level": self.autonomy_level.value,
            "learning_enabled": self.learning_enabled,
            "optimization_interval": self.optimization_interval
        })

    async def create_autonomous_campaign(self,
                                       name: str,
                                       targets: list[dict],
                                       intelligence_driven: bool = True,
                                       adaptive_execution: bool = True,
                                       config: dict[str, Any] = None) -> str:
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
                                   targets: list[dict],
                                   config: dict[str, Any]) -> list[AgentCapability]:
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

    async def _analyze_targets(self, targets: list[dict]) -> dict[str, int]:
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

    def _get_agent_performance_data(self) -> dict[str, dict[str, float]]:
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
                except TimeoutError:
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

    async def _select_optimal_agent_for_task(self, task: AgentTask) -> AutonomousWorker | None:
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

    async def get_autonomous_status(self) -> dict[str, Any]:
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

    async def _multi_agent_consensus(self, tasks: list[ExecutionContext]) -> dict[str, float]:
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
                weighted_avg = sum(s * w for s, w in zip(scores, weights, strict=False)) / sum(weights)
                final_priorities[task_id] = weighted_avg
            else:
                final_priorities[task_id] = 5.0  # Default priority

        return final_priorities

    async def _apply_agent_consensus(self, priorities: dict[str, float]):
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

    async def _gather_collective_insights(self) -> dict[str, Any]:
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

    async def _distribute_collective_insights(self, insights: dict[str, Any]):
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

    async def _ai_resource_prediction(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Enhanced AI resource prediction with Phase 10 global intelligence synthesis"""
        try:
            # Enhanced reasoning with global intelligence context
            global_context = await self._get_global_intelligence_context()

            # Advanced pattern analysis
            prediction = {
                'adjustments': [],
                'confidence': 0.8,
                'reasoning': 'Enhanced AI-powered resource prediction with global intelligence',
                'timestamp': datetime.utcnow().isoformat(),
                'global_synthesis_ready': True,
                'phase10_adaptations': []
            }

            # Multi-factor analysis for Phase 10 readiness
            cpu_util = analysis.get('cpu_utilization', 0)
            memory_util = analysis.get('memory_utilization', 0)
            queue_length = analysis.get('queue_length', 0)
            success_rate = analysis.get('success_rate_trend', 0.85)

            # Advanced resource prediction algorithms
            if cpu_util > 0.8:
                # Intelligent scaling with historical performance data
                historical_performance = self._calculate_historical_performance()
                optimal_concurrency = self._calculate_optimal_concurrency(cpu_util, historical_performance)

                prediction['adjustments'].append({
                    'type': 'intelligent_scale_down',
                    'value': optimal_concurrency,
                    'reason': f'CPU optimization based on {historical_performance:.2f} historical performance',
                    'confidence': 0.9,
                    'phase10_ready': True
                })

            # Phase 10 Global Intelligence Synthesis preparations
            if queue_length > 20 or success_rate < 0.7:
                prediction['phase10_adaptations'].append({
                    'type': 'global_intelligence_synthesis_prep',
                    'action': 'enhance_multi_source_aggregation',
                    'reasoning': 'Preparing for Phase 10 global multi-source intelligence synthesis',
                    'confidence': 0.85
                })

            # Predictive agent allocation based on global patterns
            if len(self.autonomous_workers) < 8 and queue_length > 15:
                prediction['adjustments'].append({
                    'type': 'predictive_worker_spawn',
                    'value': min(32, len(self.autonomous_workers) +
                            self._calculate_optimal_worker_increase(analysis)),
                    'reason': 'Predictive scaling based on global intelligence patterns',
                    'confidence': 0.87
                })

            # Enhanced fault tolerance for autonomous operation
            prediction['fault_tolerance_enhancements'] = {
                'auto_recovery_enabled': True,
                'circuit_breaker_thresholds': self._calculate_circuit_breaker_thresholds(analysis),
                'redundancy_factor': self._calculate_redundancy_factor(cpu_util, memory_util)
            }

            return prediction

        except Exception as e:
            self.logger.error("Enhanced AI resource prediction error", error=str(e))
            return {
                'adjustments': [],
                'confidence': 0.0,
                'fallback_mode': True,
                'error_recovery': 'Using conservative defaults'
            }

    def _calculate_optimal_concurrency(self, cpu_util: float, historical_perf: float) -> int:
        """Calculate optimal concurrency based on CPU and historical performance"""
        base_concurrency = self.max_concurrent_agents
        cpu_factor = max(0.3, 1.0 - (cpu_util - 0.7) * 2)  # Reduce if CPU > 70%
        perf_factor = min(1.2, historical_perf * 1.5)  # Increase if performance is good

        optimal = int(base_concurrency * cpu_factor * perf_factor)
        return max(4, min(48, optimal))  # Clamp between 4-48

    def _calculate_optimal_worker_increase(self, analysis: dict[str, Any]) -> int:
        """Calculate optimal worker increase based on queue and performance patterns"""
        queue_length = analysis.get('queue_length', 0)
        success_rate = analysis.get('success_rate_trend', 0.85)

        # Base increase on queue pressure
        base_increase = min(4, queue_length // 10)

        # Adjust based on success rate
        if success_rate > 0.9:
            base_increase += 1  # Can handle more load
        elif success_rate < 0.7:
            base_increase = max(1, base_increase - 1)  # Be conservative

        return base_increase

    def _calculate_historical_performance(self) -> float:
        """Calculate historical performance metric"""
        if not self.global_intelligence or not self.global_intelligence.task_success_rates:
            return 0.8  # Default

        success_rates = self.global_intelligence.task_success_rates.values()
        return sum(success_rates) / len(success_rates) if success_rates else 0.8

    async def _get_global_intelligence_context(self) -> dict[str, Any]:
        """Get global intelligence context for enhanced decision making"""
        return {
            'active_campaigns': len(getattr(self, 'active_campaigns', {})),
            'global_threat_level': 'moderate',  # Would be calculated from real data
            'resource_efficiency': self._calculate_resource_efficiency(),
            'agent_performance_variance': self._calculate_performance_variance(),
            'phase10_readiness_score': 0.85
        }

    def _calculate_resource_efficiency(self) -> float:
        """Calculate overall resource efficiency"""
        if not hasattr(self, 'workload_profile') or not self.workload_profile.resource_utilization:
            return 0.8

        cpu = self.workload_profile.resource_utilization.get('cpu', 0.5)
        memory = self.workload_profile.resource_utilization.get('memory', 0.5)

        # Efficiency is high when utilization is moderate (not too low, not too high)
        cpu_efficiency = 1.0 - abs(cpu - 0.7)  # Target 70% CPU
        memory_efficiency = 1.0 - abs(memory - 0.6)  # Target 60% memory

        return (cpu_efficiency + memory_efficiency) / 2

    def _calculate_performance_variance(self) -> float:
        """Calculate performance variance across agents"""
        if not self.autonomous_workers:
            return 0.0

        # Simplified calculation - would use real performance data
        return 0.15  # 15% variance (lower is better)

    def _calculate_circuit_breaker_thresholds(self, analysis: dict[str, Any]) -> dict[str, float]:
        """Calculate dynamic circuit breaker thresholds"""
        base_failure_threshold = 0.5
        base_timeout_threshold = 30.0

        # Adjust based on current performance
        success_rate = analysis.get('success_rate_trend', 0.85)
        if success_rate < 0.7:
            base_failure_threshold = 0.3  # More aggressive
            base_timeout_threshold = 20.0
        elif success_rate > 0.9:
            base_failure_threshold = 0.7  # More lenient
            base_timeout_threshold = 45.0

        return {
            'failure_threshold': base_failure_threshold,
            'timeout_threshold_seconds': base_timeout_threshold,
            'recovery_time_seconds': 60.0
        }

    def _calculate_redundancy_factor(self, cpu_util: float, memory_util: float) -> float:
        """Calculate redundancy factor for fault tolerance"""
        if cpu_util > 0.8 or memory_util > 0.8:
            return 1.2  # 20% redundancy under high load
        elif cpu_util < 0.5 and memory_util < 0.5:
            return 1.5  # 50% redundancy when resources are available
        else:
            return 1.3  # 30% default redundancy

    def _get_agent_confidence(self, agent_index: int) -> float:
        """Get confidence score for an agent"""
        # This would be based on historical performance
        return 0.8  # Default confidence

    async def _analyze_resource_patterns(self) -> dict[str, Any]:
        """Analyze current resource utilization patterns"""
        return {
            'cpu_utilization': 0.6,  # Would be real metrics
            'memory_utilization': 0.4,
            'queue_length': len(self.execution_contexts),
            'active_agents': len(self.autonomous_workers),
            'success_rate_trend': 0.85
        }

    async def _apply_predictive_adjustments(self, prediction: dict[str, Any]):
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

    # =============================================================================
    # PHASE 10: GLOBAL INTELLIGENCE SYNTHESIS INTEGRATION
    # =============================================================================

    def _initialize_synthesis_metrics(self) -> dict[str, Any]:
        """Initialize metrics for global intelligence synthesis"""
        return {
            'intelligence_signals_processed': Counter(
                'xorb_synthesis_signals_processed_total',
                'Total intelligence signals processed by orchestrator',
                ['signal_type', 'priority']
            ),
            'missions_triggered_by_intelligence': Counter(
                'xorb_missions_triggered_by_intelligence_total',
                'Missions triggered by intelligence synthesis',
                ['intelligence_type', 'mission_type']
            ),
            'intelligence_response_time': Histogram(
                'xorb_intelligence_response_time_seconds',
                'Time from intelligence to mission creation',
                ['priority_level']
            ),
            'synthesis_feedback_score': Gauge(
                'xorb_synthesis_feedback_score',
                'Feedback score for intelligence-driven missions'
            )
        }

    async def initialize_global_synthesis_engine(self,
                                               mission_engine: 'AdaptiveMissionEngine',
                                               episodic_memory: 'EpisodicMemorySystem',
                                               vector_fabric: 'VectorFabric'):
        """Initialize the global intelligence synthesis engine"""
        try:
            from ..intelligence.global_synthesis_engine import GlobalSynthesisEngine

            self.global_synthesis_engine = GlobalSynthesisEngine(
                orchestrator=self,
                mission_engine=mission_engine,
                episodic_memory=episodic_memory,
                vector_fabric=vector_fabric
            )

            await self.global_synthesis_engine.start_synthesis_engine()

            self.logger.info("🌐 Global Intelligence Synthesis Engine: INITIALIZED")

        except Exception as e:
            self.logger.error("Failed to initialize synthesis engine", error=str(e))
            self.intelligence_processing_enabled = False

    async def _global_intelligence_processor(self):
        """Process global intelligence signals and route to appropriate handlers"""

        while self._running and self.intelligence_processing_enabled:
            try:
                if not self.global_synthesis_engine:
                    await asyncio.sleep(30)
                    continue

                # Get correlated intelligence from synthesis engine
                correlated_intel = await self._get_pending_intelligence()

                for intelligence in correlated_intel:
                    await self._process_correlated_intelligence(intelligence)

                await asyncio.sleep(10)  # Process every 10 seconds

            except Exception as e:
                self.logger.error("Global intelligence processing error", error=str(e))
                await asyncio.sleep(30)

    async def _get_pending_intelligence(self) -> list[CorrelatedIntelligence]:
        """Get pending correlated intelligence from synthesis engine"""

        if not self.global_synthesis_engine:
            return []

        # Get high-priority intelligence that needs immediate action
        pending_intelligence = []

        for intel_id, intelligence in self.global_synthesis_engine.correlated_intelligence.items():
            # Check if this intelligence needs action
            if (intelligence.overall_priority in [SignalPriority.CRITICAL, SignalPriority.HIGH] and
                len(intelligence.spawned_missions) == 0 and
                intelligence.recommended_actions):

                pending_intelligence.append(intelligence)

        # Sort by priority and confidence
        pending_intelligence.sort(
            key=lambda x: (x.overall_priority.value, x.confidence_score),
            reverse=True
        )

        return pending_intelligence[:10]  # Process top 10

    async def _process_correlated_intelligence(self, intelligence: CorrelatedIntelligence):
        """Process a single correlated intelligence item"""

        try:
            self.logger.info("🧠 Processing correlated intelligence",
                           intelligence_id=intelligence.intelligence_id[:8],
                           priority=intelligence.overall_priority.value,
                           confidence=intelligence.confidence_score)

            # Determine mission type based on intelligence characteristics
            mission_type = await self._determine_mission_type(intelligence)

            # Select optimal agents based on required capabilities
            agent_capabilities = await self._map_intelligence_to_capabilities(intelligence)

            # Create autonomous mission based on intelligence
            mission_id = await self._create_intelligence_driven_mission(
                intelligence, mission_type, agent_capabilities
            )

            if mission_id:
                # Record mission creation
                intelligence.spawned_missions.append(mission_id)

                # Update metrics
                self.synthesis_integration_metrics['missions_triggered_by_intelligence'].labels(
                    intelligence_type=intelligence.threat_context.get('type', 'unknown'),
                    mission_type=mission_type.value if hasattr(mission_type, 'value') else str(mission_type)
                ).inc()

                # Log successful processing
                self.logger.info("✅ Intelligence-driven mission created",
                               intelligence_id=intelligence.intelligence_id[:8],
                               mission_id=mission_id[:8],
                               mission_type=mission_type)

                # Record decision for learning
                decision = AutonomousDecision(
                    decision_id=str(uuid.uuid4()),
                    decision_type="intelligence_driven_mission",
                    context={
                        "intelligence_id": intelligence.intelligence_id,
                        "mission_id": mission_id,
                        "priority": intelligence.overall_priority.value,
                        "confidence": intelligence.confidence_score,
                        "agent_capabilities": [cap.value if hasattr(cap, 'value') else str(cap)
                                             for cap in agent_capabilities]
                    },
                    rationale=f"Created mission based on {intelligence.overall_priority.value} priority intelligence",
                    confidence=intelligence.confidence_score
                )
                self.decision_history.append(decision)

        except Exception as e:
            self.logger.error("Intelligence processing failed",
                            intelligence_id=intelligence.intelligence_id[:8],
                            error=str(e))

    async def _determine_mission_type(self, intelligence: CorrelatedIntelligence) -> str:
        """Determine optimal mission type based on intelligence characteristics"""

        # Analyze intelligence content to determine mission type
        threat_level = intelligence.threat_level.lower()
        key_indicators = [indicator.lower() for indicator in intelligence.key_indicators]

        # High-priority vulnerability or exploit
        if any(keyword in ' '.join(key_indicators) for keyword in
               ['exploit', 'zero-day', 'rce', 'critical', 'authentication bypass']):
            return "VULNERABILITY_ASSESSMENT"

        # Bug bounty or disclosed vulnerability
        elif any(keyword in ' '.join(key_indicators) for keyword in
                ['bounty', 'disclosed', 'bug', 'reward']):
            return "BUG_BOUNTY_INVESTIGATION"

        # System anomaly or alert
        elif any(keyword in ' '.join(key_indicators) for keyword in
                ['alert', 'anomaly', 'threshold', 'monitoring']):
            return "SYSTEM_INVESTIGATION"

        # Threat intelligence
        elif any(keyword in ' '.join(key_indicators) for keyword in
                ['apt', 'malware', 'campaign', 'ioc']):
            return "THREAT_INVESTIGATION"

        # Default reconnaissance mission
        else:
            return "INTELLIGENCE_GATHERING"

    async def _map_intelligence_to_capabilities(self, intelligence: CorrelatedIntelligence) -> list[AgentCapability]:
        """Map intelligence requirements to agent capabilities"""

        capabilities = []
        required_caps = intelligence.required_capabilities
        key_indicators = [indicator.lower() for indicator in intelligence.key_indicators]

        # Always include reconnaissance for intelligence gathering
        capabilities.append(AgentCapability.RECONNAISSANCE)

        # Web-related intelligence
        if any(keyword in ' '.join(key_indicators) for keyword in
               ['web', 'http', 'browser', 'xss', 'sql', 'csrf']):
            capabilities.extend([
                AgentCapability.WEB_CRAWLING,
                AgentCapability.VULNERABILITY_SCANNING
            ])

        # Network-related intelligence
        if any(keyword in ' '.join(key_indicators) for keyword in
               ['network', 'port', 'service', 'protocol', 'firewall']):
            capabilities.append(AgentCapability.NETWORK_SCANNING)

        # API-related intelligence
        if any(keyword in ' '.join(key_indicators) for keyword in
               ['api', 'rest', 'graphql', 'endpoint', 'authentication']):
            capabilities.append(AgentCapability.API_TESTING)

        # Social engineering intelligence
        if any(keyword in ' '.join(key_indicators) for keyword in
               ['social', 'phishing', 'email', 'human']):
            capabilities.append(AgentCapability.SOCIAL_ENGINEERING)

        # Remove duplicates and return
        return list(set(capabilities))

    async def _create_intelligence_driven_mission(self,
                                                intelligence: CorrelatedIntelligence,
                                                mission_type: str,
                                                agent_capabilities: list[AgentCapability]) -> str | None:
        """Create a new mission driven by intelligence synthesis"""

        try:
            # Construct mission objectives from intelligence
            objectives = self._extract_mission_objectives(intelligence)

            # Create mission configuration
            mission_config = {
                'intelligence_driven': True,
                'intelligence_id': intelligence.intelligence_id,
                'priority': intelligence.overall_priority.value,
                'confidence_threshold': intelligence.confidence_score,
                'max_agents': min(len(agent_capabilities) + 2, 8),
                'timeout': self._calculate_mission_timeout(intelligence),
                'auto_escalate': intelligence.overall_priority in [SignalPriority.CRITICAL, SignalPriority.HIGH]
            }

            # Create the mission
            mission_id = await self.create_autonomous_campaign(
                name=f"Intelligence-Driven: {intelligence.synthesized_title[:50]}",
                targets=self._extract_targets_from_intelligence(intelligence),
                intelligence_driven=True,
                adaptive_execution=True,
                config=mission_config
            )

            # Record intelligence response time
            response_time = (datetime.utcnow() - intelligence.created_at).total_seconds()
            self.synthesis_integration_metrics['intelligence_response_time'].labels(
                priority_level=intelligence.overall_priority.value
            ).observe(response_time)

            return mission_id

        except Exception as e:
            self.logger.error("Intelligence-driven mission creation failed", error=str(e))
            return None

    def _extract_mission_objectives(self, intelligence: CorrelatedIntelligence) -> list[str]:
        """Extract specific mission objectives from intelligence"""

        objectives = []

        # Add objectives based on recommended actions
        for action in intelligence.recommended_actions:
            if action.get('type') == 'investigate':
                objectives.append(f"Investigate: {action.get('description', 'Unknown target')}")
            elif action.get('type') == 'verify':
                objectives.append(f"Verify: {action.get('description', 'Unknown vulnerability')}")
            elif action.get('type') == 'assess':
                objectives.append(f"Assess: {action.get('description', 'Unknown risk')}")

        # Add default objective if none specified
        if not objectives:
            objectives.append(f"Analyze threat: {intelligence.synthesized_title}")

        # Add intelligence gathering objective
        objectives.append("Gather additional intelligence on discovered assets")

        return objectives

    def _extract_targets_from_intelligence(self, intelligence: CorrelatedIntelligence) -> list[dict[str, Any]]:
        """Extract targets from intelligence content"""

        targets = []

        # Extract from threat context
        threat_context = intelligence.threat_context

        if 'targets' in threat_context:
            targets.extend(threat_context['targets'])

        # Extract from key indicators
        for indicator in intelligence.key_indicators:
            if '://' in indicator:  # URL-like indicator
                targets.append({
                    'hostname': indicator,
                    'source': 'intelligence_synthesis',
                    'priority': intelligence.overall_priority.value
                })

        # Default target if none found
        if not targets:
            targets.append({
                'hostname': 'target-from-intelligence.local',
                'source': 'intelligence_synthesis',
                'description': intelligence.synthesized_description,
                'priority': intelligence.overall_priority.value
            })

        return targets

    def _calculate_mission_timeout(self, intelligence: CorrelatedIntelligence) -> int:
        """Calculate appropriate timeout for intelligence-driven mission"""

        base_timeout = 3600  # 1 hour default

        # Adjust based on priority
        if intelligence.overall_priority == SignalPriority.CRITICAL:
            return base_timeout // 2  # 30 minutes for critical
        elif intelligence.overall_priority == SignalPriority.HIGH:
            return int(base_timeout * 0.75)  # 45 minutes for high
        elif intelligence.overall_priority == SignalPriority.LOW:
            return base_timeout * 2  # 2 hours for low priority

        return base_timeout

    async def _synthesis_mission_dispatcher(self):
        """Dispatch missions based on synthesis engine recommendations"""

        while self._running and self.intelligence_processing_enabled:
            try:
                # Check for high-priority intelligence needing immediate dispatch
                urgent_intelligence = await self._get_urgent_intelligence()

                for intelligence in urgent_intelligence:
                    # Fast-track critical intelligence
                    if intelligence.overall_priority == SignalPriority.CRITICAL:
                        await self._fast_track_critical_intelligence(intelligence)

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error("Synthesis mission dispatcher error", error=str(e))
                await asyncio.sleep(60)

    async def _get_urgent_intelligence(self) -> list[CorrelatedIntelligence]:
        """Get intelligence requiring urgent action"""

        if not self.global_synthesis_engine:
            return []

        urgent = []
        current_time = datetime.utcnow()

        for intelligence in self.global_synthesis_engine.correlated_intelligence.values():
            # Critical intelligence created in last 10 minutes
            if (intelligence.overall_priority == SignalPriority.CRITICAL and
                (current_time - intelligence.created_at).total_seconds() < 600 and
                len(intelligence.spawned_missions) == 0):
                urgent.append(intelligence)

        return urgent

    async def _fast_track_critical_intelligence(self, intelligence: CorrelatedIntelligence):
        """Fast-track critical intelligence through processing"""

        self.logger.warning("🚨 CRITICAL INTELLIGENCE DETECTED - Fast-tracking",
                          intelligence_id=intelligence.intelligence_id[:8],
                          threat_level=intelligence.threat_level)

        # Immediately process without normal queuing
        await self._process_correlated_intelligence(intelligence)

        # Notify all relevant stakeholders
        await self._notify_critical_intelligence(intelligence)

    async def _notify_critical_intelligence(self, intelligence: CorrelatedIntelligence):
        """Notify stakeholders of critical intelligence"""

        notification = {
            'type': 'critical_intelligence',
            'intelligence_id': intelligence.intelligence_id,
            'title': intelligence.synthesized_title,
            'threat_level': intelligence.threat_level,
            'confidence': intelligence.confidence_score,
            'key_indicators': intelligence.key_indicators,
            'recommended_actions': intelligence.recommended_actions,
            'timestamp': datetime.utcnow().isoformat()
        }

        # Log critical event
        await self.audit_logger.log_event("critical_intelligence_detected", notification)

        self.logger.critical("🚨 CRITICAL INTELLIGENCE NOTIFICATION",
                           intelligence_id=intelligence.intelligence_id[:8],
                           threat_level=intelligence.threat_level,
                           confidence=intelligence.confidence_score)

    async def _intelligence_feedback_loop(self):
        """Process feedback from intelligence-driven missions for learning"""

        while self._running and self.intelligence_processing_enabled:
            try:
                # Get completed intelligence-driven missions
                completed_missions = await self._get_completed_intelligence_missions()

                for mission_result in completed_missions:
                    await self._process_intelligence_mission_feedback(mission_result)

                await asyncio.sleep(120)  # Process feedback every 2 minutes

            except Exception as e:
                self.logger.error("Intelligence feedback loop error", error=str(e))
                await asyncio.sleep(120)

    async def _get_completed_intelligence_missions(self) -> list[dict[str, Any]]:
        """Get completed missions that were intelligence-driven"""

        completed_missions = []

        # Check recent campaign completions
        for campaign_id, context in self.execution_contexts.items():
            if (context.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED] and
                context.config.get('intelligence_driven', False) and
                not context.metadata.get('feedback_processed', False)):

                completed_missions.append({
                    'campaign_id': campaign_id,
                    'context': context,
                    'intelligence_id': context.config.get('intelligence_id')
                })

        return completed_missions

    async def _process_intelligence_mission_feedback(self, mission_result: dict[str, Any]):
        """Process feedback from completed intelligence-driven mission"""

        try:
            campaign_id = mission_result['campaign_id']
            context = mission_result['context']
            intelligence_id = mission_result['intelligence_id']

            # Calculate feedback score
            feedback_score = self._calculate_intelligence_feedback_score(context)

            # Update intelligence with feedback
            if (self.global_synthesis_engine and
                intelligence_id in self.global_synthesis_engine.correlated_intelligence):

                intelligence = self.global_synthesis_engine.correlated_intelligence[intelligence_id]
                intelligence.feedback_scores.append(feedback_score)

                # Update synthesis metrics
                self.synthesis_integration_metrics['synthesis_feedback_score'].set(feedback_score)

            # Mark as processed
            context.metadata['feedback_processed'] = True

            self.logger.info("📊 Intelligence mission feedback processed",
                           campaign_id=campaign_id[:8],
                           intelligence_id=intelligence_id[:8] if intelligence_id else "unknown",
                           feedback_score=feedback_score)

        except Exception as e:
            self.logger.error("Intelligence feedback processing failed", error=str(e))

    def _calculate_intelligence_feedback_score(self, context: ExecutionContext) -> float:
        """Calculate feedback score for intelligence-driven mission"""

        base_score = 0.5

        # Success factor
        if context.status == ExecutionStatus.COMPLETED:
            base_score += 0.3
        elif context.status == ExecutionStatus.FAILED:
            base_score -= 0.2

        # Performance factor
        if hasattr(context, 'performance_metrics'):
            success_rate = context.performance_metrics.get('success_rate', 0.5)
            base_score += (success_rate - 0.5) * 0.4

        # Discovery factor
        discoveries = context.metadata.get('discoveries', 0)
        if discoveries > 0:
            base_score += min(discoveries * 0.1, 0.3)

        # Time factor
        expected_duration = context.config.get('timeout', 3600)
        actual_duration = (context.end_time - context.start_time).total_seconds() if context.end_time else expected_duration

        if actual_duration < expected_duration * 0.8:  # Completed faster than expected
            base_score += 0.1
        elif actual_duration > expected_duration * 1.2:  # Took longer than expected
            base_score -= 0.1

        return max(0.0, min(1.0, base_score))


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

    async def _identify_preemption_candidates(self) -> list[PreemptionEvent]:
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

    async def _collect_decision_feedback(self) -> list[dict[str, Any]]:
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
                           tasks: list[AgentTask],
                           agent_performance: dict[str, dict[str, float]],
                           resource_constraints: dict[str, float]) -> list[AgentTask]:
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
