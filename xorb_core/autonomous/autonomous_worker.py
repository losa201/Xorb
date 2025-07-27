#!/usr/bin/env python3
"""
Autonomous Worker Framework for Xorb Security Intelligence Platform

This module provides secure autonomous worker capabilities that:
- Operate within defensive security boundaries
- Adapt workflows based on intelligence and metrics
- Manage resources dynamically within security constraints
- Provide comprehensive monitoring and auditability
- Maintain strict compliance with Rules of Engagement (RoE)
"""

import asyncio
import logging
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import structlog
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from tenacity import retry, stop_after_attempt, wait_exponential

from ..plugins.bases import BaseAgent
from ..domain.models import AgentCapability
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

class AgentStatus(str, Enum):
    """Status of an agent"""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"

@dataclass
class AgentTask:
    """Represents a task for an agent to execute"""
    task_id: str
    task_type: str
    target: str
    priority: int = 5
    parameters: Dict[str, Any] = field(default_factory=dict)
    max_retries: int = 3
    created_at: Optional[datetime] = None

@dataclass
class AgentResult:
    """Represents the result of an agent's task execution"""
    task_id: str
    success: bool
    errors: List[str] = field(default_factory=list)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)



from xorb_core.orchestration.enhanced_orchestrator import EnhancedOrchestrator, ExecutionContext, ExecutionStatus


class AutonomyLevel(str, Enum):
    """Levels of worker autonomy within security boundaries"""
    MINIMAL = "minimal"       # Basic task execution only
    MODERATE = "moderate"     # Dynamic task selection and resource adjustment
    HIGH = "high"            # Autonomous workflow adaptation and learning
    MAXIMUM = "maximum"      # Full autonomy within security constraints


class WorkerCapability(str, Enum):
    """Enhanced worker capabilities for autonomous operation"""
    DYNAMIC_TASK_SELECTION = "dynamic_task_selection"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    WORKFLOW_ADAPTATION = "workflow_adaptation"
    INTELLIGENCE_SYNTHESIS = "intelligence_synthesis"
    FAILURE_RECOVERY = "failure_recovery"
    PERFORMANCE_LEARNING = "performance_learning"


@dataclass
class AutonomousConfig:
    """Configuration for autonomous worker operations"""
    autonomy_level: AutonomyLevel = AutonomyLevel.MODERATE
    max_concurrent_tasks: int = 8
    resource_adaptation_threshold: float = 0.8
    intelligence_update_interval: int = 300  # 5 minutes
    failure_recovery_attempts: int = 3
    performance_learning_enabled: bool = True
    security_validation_required: bool = True
    roe_compliance_strict: bool = True


@dataclass 
class WorkerIntelligence:
    """Intelligence data for autonomous decision making"""
    task_success_rates: Dict[str, float] = field(default_factory=dict)
    resource_utilization_history: List[Dict[str, float]] = field(default_factory=list)
    threat_landscape_updates: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    vulnerability_patterns: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SecurityConstraint:
    """Security constraints for autonomous operations"""
    constraint_id: str
    constraint_type: str
    description: str
    validation_function: str
    severity: str = "high"
    active: bool = True


class AutonomousWorker(BaseAgent):
    """
    Autonomous Security Intelligence Worker
    
    Provides secure autonomous capabilities while maintaining defensive boundaries:
    - Dynamic task selection based on intelligence
    - Adaptive resource management
    - Autonomous workflow optimization
    - Comprehensive security validation
    - Performance learning and improvement
    """
    
    def __init__(self, 
                 agent_id: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 autonomous_config: Optional[AutonomousConfig] = None):
        
        super().__init__(agent_id, config)
        
        self.autonomous_config = autonomous_config or AutonomousConfig()
        self.logger = structlog.get_logger(f"AutonomousWorker[{self.agent_id[:8]}]")
        
        # Autonomous state management
        self.intelligence = WorkerIntelligence()
        self.security_constraints: List[SecurityConstraint] = []
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.resource_monitor = ResourceMonitor()
        self.decision_engine = DecisionEngine()
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.adaptation_count = 0
        self.security_violations_prevented = 0
        
        # Autonomous task management
        self.task_prioritizer = TaskPrioritizer()
        self.workflow_optimizer = WorkflowOptimizer()
        
        # Initialize security constraints
        self._initialize_security_constraints()
        
    @property
    def agent_type(self):
        return "autonomous_security_worker"
        
    def _initialize_capabilities(self):
        """Initialize autonomous worker capabilities"""
        self.capabilities = [
            AgentCapability(
                name=WorkerCapability.DYNAMIC_TASK_SELECTION.value,
                description="Autonomously select and prioritize security tasks",
                required_tools=["intelligence_engine", "task_analyzer"],
                success_rate=0.85,
                enabled=True
            ),
            AgentCapability(
                name=WorkerCapability.RESOURCE_OPTIMIZATION.value,
                description="Dynamically optimize resource allocation",
                required_tools=["resource_monitor", "performance_analyzer"],
                success_rate=0.90,
                enabled=True
            ),
            AgentCapability(
                name=WorkerCapability.WORKFLOW_ADAPTATION.value,
                description="Adapt workflows based on performance and intelligence",
                required_tools=["workflow_engine", "adaptation_engine"],
                success_rate=0.78,
                enabled=True
            ),
            AgentCapability(
                name=WorkerCapability.INTELLIGENCE_SYNTHESIS.value,
                description="Synthesize and analyze security intelligence",
                required_tools=["intelligence_processor", "pattern_analyzer"],
                success_rate=0.82,
                enabled=True
            ),
            AgentCapability(
                name=WorkerCapability.FAILURE_RECOVERY.value,
                description="Autonomous failure detection and recovery",
                required_tools=["health_monitor", "recovery_engine"],
                success_rate=0.88,
                enabled=True
            )
        ]
        
    def _initialize_security_constraints(self):
        """Initialize security constraints for autonomous operations"""
        self.security_constraints = [
            SecurityConstraint(
                constraint_id="roe_validation",
                constraint_type="compliance",
                description="Ensure all activities comply with Rules of Engagement",
                validation_function="validate_roe_compliance",
                severity="critical"
            ),
            SecurityConstraint(
                constraint_id="resource_limits",
                constraint_type="resource",
                description="Prevent resource exhaustion attacks",
                validation_function="validate_resource_limits",
                severity="high"
            ),
            SecurityConstraint(
                constraint_id="data_protection",
                constraint_type="privacy",
                description="Protect sensitive data during operations",
                validation_function="validate_data_protection",
                severity="critical"
            ),
            SecurityConstraint(
                constraint_id="network_boundaries",  
                constraint_type="network",
                description="Ensure operations stay within authorized network boundaries",
                validation_function="validate_network_boundaries",
                severity="high"
            )
        ]
        
    async def _execute_task(self, task: AgentTask) -> AgentResult:
        """Execute task with autonomous capabilities and security validation"""
        
        start_time = time.time()
        
        try:
            # Security validation
            if not await self._validate_security_constraints(task):
                return AgentResult(
                    task_id=task.task_id,
                    success=False,
                    errors=["Task failed security constraint validation"],
                    confidence=0.0
                )
            
            # Autonomous task analysis and adaptation
            adapted_task = await self._adapt_task_execution(task)
            
            # Execute with autonomous monitoring
            result = await self._execute_with_monitoring(adapted_task)
            
            # Learn from execution
            await self._learn_from_execution(adapted_task, result)
            
            # Update intelligence
            await self._update_intelligence(adapted_task, result)
            
            return result
            
        except Exception as e:
            self.logger.error("Autonomous task execution failed", 
                            task_id=task.task_id, error=str(e))
            
            # Autonomous failure recovery
            recovery_result = await self._attempt_failure_recovery(task, str(e))
            if recovery_result:
                return recovery_result
                
            return AgentResult(
                task_id=task.task_id,
                success=False,
                errors=[f"Execution failed: {str(e)}"],
                execution_time=time.time() - start_time
            )
    
    async def _validate_security_constraints(self, task: AgentTask) -> bool:
        """Validate task against all security constraints"""
        
        for constraint in self.security_constraints:
            if not constraint.active:
                continue
                
            try:
                # Get validation function
                validator = getattr(self, constraint.validation_function, None)
                if not validator:
                    self.logger.warning("Missing validator", 
                                      function=constraint.validation_function)
                    continue
                
                # Execute validation
                if not await validator(task):
                    self.logger.warning("Security constraint violation prevented",
                                      constraint_id=constraint.constraint_id,
                                      task_id=task.task_id)
                    self.security_violations_prevented += 1
                    return False
                    
            except Exception as e:
                self.logger.error("Security validation error",
                                constraint_id=constraint.constraint_id,
                                error=str(e))
                return False
                
        return True
    
    async def validate_roe_compliance(self, task: AgentTask) -> bool:
        """Validate Rules of Engagement compliance"""
        
        # Check if target is authorized
        target = task.parameters.get('target', task.target)
        if not target:
            return False
            
        # Validate against authorized target patterns
        authorized_patterns = self.config.get('authorized_targets', [])
        if not any(pattern in target for pattern in authorized_patterns):
            return False
            
        # Check for prohibited activities
        prohibited_actions = self.config.get('prohibited_actions', [])
        if task.task_type in prohibited_actions:
            return False
            
        return True
    
    async def validate_resource_limits(self, task: AgentTask) -> bool:
        """Validate resource consumption limits"""
        
        current_cpu = await self.resource_monitor.get_cpu_usage()
        current_memory = await self.resource_monitor.get_memory_usage()
        
        # Check if adding this task would exceed limits
        estimated_cpu = task.parameters.get('estimated_cpu', 0.1)
        estimated_memory = task.parameters.get('estimated_memory', 0.05)
        
        if current_cpu + estimated_cpu > 0.8:  # 80% CPU limit
            return False
            
        if current_memory + estimated_memory > 0.8:  # 80% memory limit
            return False
            
        return True
    
    async def validate_data_protection(self, task: AgentTask) -> bool:
        """Validate data protection requirements"""
        
        # Check for sensitive data patterns
        sensitive_patterns = ['password', 'key', 'token', 'secret', 'credential']
        task_str = json.dumps(task.parameters, default=str).lower()
        
        for pattern in sensitive_patterns:
            if pattern in task_str:
                # Ensure proper protection is configured
                if not task.parameters.get('data_protection_enabled', False):
                    return False
                    
        return True
    
    async def validate_network_boundaries(self, task: AgentTask) -> bool:
        """Validate network boundary constraints"""
        
        target = task.parameters.get('target', task.target)
        if not target:
            return True
            
        # Check against authorized network ranges
        authorized_networks = self.config.get('authorized_networks', [])
        if not authorized_networks:
            return True  # No restrictions configured
            
        # Simple validation - could be enhanced with proper IP validation
        return any(network in target for network in authorized_networks)
    
    async def _adapt_task_execution(self, task: AgentTask) -> AgentTask:
        """Autonomously adapt task execution based on intelligence"""
        
        if self.autonomous_config.autonomy_level == AutonomyLevel.MINIMAL:
            return task
            
        # Analyze task performance history
        task_type_history = self.intelligence.task_success_rates.get(task.task_type, 0.5)
        
        # Adapt timeout based on historical performance
        if task_type_history < 0.5:  # Low success rate
            task.parameters['timeout'] = task.parameters.get('timeout', 30) * 1.5
            task.max_retries = min(task.max_retries + 1, 5)
            
        # Adapt concurrency based on resource availability
        current_load = await self.resource_monitor.get_current_load()
        if current_load > self.autonomous_config.resource_adaptation_threshold:
            task.parameters['parallel_execution'] = False
            task.priority = max(1, task.priority - 1)  # Lower priority under high load
            
        # Add intelligence-based parameters
        task.parameters['autonomous_adaptations'] = {
            'timeout_adjusted': task.parameters.get('timeout') != 30,
            'concurrency_adapted': current_load > 0.8,
            'priority_adjusted': True,
            'adaptation_timestamp': datetime.utcnow().isoformat()
        }
        
        self.adaptation_count += 1
        
        return task
    
    async def _execute_with_monitoring(self, task: AgentTask) -> AgentResult:
        """Execute task with autonomous monitoring and adjustment"""
        
        start_time = time.time()
        
        # Start resource monitoring
        monitor_task = asyncio.create_task(
            self._monitor_execution_resources(task.task_id)
        )
        
        try:
            # Simulate actual task execution
            # In reality, this would call specific agent implementations
            await asyncio.sleep(0.1)  # Simulate work
            
            # Generate realistic result based on task type
            result = await self._generate_task_result(task)
            
            # Stop monitoring
            monitor_task.cancel()
            
            return result
            
        except Exception as e:
            monitor_task.cancel()
            raise e
    
    async def _generate_task_result(self, task: AgentTask) -> AgentResult:
        """Generate realistic task result for demonstration"""
        
        # Simulate different outcomes based on task type and history
        success_rate = self.intelligence.task_success_rates.get(task.task_type, 0.8)
        success = time.time() % 1.0 < success_rate  # Pseudo-random based on time
        
        findings = []
        if success and task.task_type in ['reconnaissance', 'vulnerability_scan']:
            findings = [
                {
                    'type': 'open_port',
                    'details': {'port': 80, 'service': 'http'},
                    'severity': 'info',
                    'confidence': 0.95
                },
                {
                    'type': 'service_banner',
                    'details': {'service': 'Apache/2.4.41', 'version': '2.4.41'},
                    'severity': 'info',
                    'confidence': 0.88
                }
            ]
        
        return AgentResult(
            task_id=task.task_id,
            success=success,
            findings=findings,
            data={'autonomous_execution': True, 'adaptations_applied': self.adaptation_count},
            confidence=0.85 if success else 0.3,
            metadata={
                'autonomy_level': self.autonomous_config.autonomy_level.value,
                'security_validated': True,
                'resource_optimized': True
            }
        )
    
    async def _monitor_execution_resources(self, task_id: str):
        """Monitor resource usage during task execution"""
        
        try:
            while True:
                cpu_usage = await self.resource_monitor.get_cpu_usage()
                memory_usage = await self.resource_monitor.get_memory_usage()
                
                # Log resource metrics
                self.logger.debug("Resource monitoring",
                                task_id=task_id,
                                cpu_usage=cpu_usage,
                                memory_usage=memory_usage)
                
                # Autonomous resource adjustment
                if cpu_usage > 0.9:  # Emergency CPU limit
                    self.logger.warning("High CPU usage detected, triggering adaptation",
                                      cpu_usage=cpu_usage)
                    # Could implement CPU throttling here
                    
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            pass
    
    async def _learn_from_execution(self, task: AgentTask, result: AgentResult):
        """Learn from task execution to improve future performance"""
        
        if not self.autonomous_config.performance_learning_enabled:
            return
            
        # Update task success rates
        current_rate = self.intelligence.task_success_rates.get(task.task_type, 0.5)
        new_sample = 1.0 if result.success else 0.0
        
        # Exponential moving average
        self.intelligence.task_success_rates[task.task_type] = (
            current_rate * 0.8 + new_sample * 0.2
        )
        
        # Record performance metrics
        self.performance_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'task_type': task.task_type,
            'success': result.success,
            'execution_time': result.execution_time,
            'confidence': result.confidence,
            'adaptations_applied': self.adaptation_count
        })
        
        # Limit history size
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-800:]
            
        self.logger.debug("Performance learning update",
                        task_type=task.task_type,
                        success_rate=self.intelligence.task_success_rates[task.task_type])
    
    async def _update_intelligence(self, task: AgentTask, result: AgentResult):
        """Update intelligence database with execution results"""
        
        # Update resource utilization history
        current_resources = {
            'timestamp': datetime.utcnow().isoformat(),
            'cpu_usage': await self.resource_monitor.get_cpu_usage(),
            'memory_usage': await self.resource_monitor.get_memory_usage(),
            'task_type': task.task_type,
            'success': result.success
        }
        
        self.intelligence.resource_utilization_history.append(current_resources)
        
        # Limit history size
        if len(self.intelligence.resource_utilization_history) > 500:
            self.intelligence.resource_utilization_history = (
                self.intelligence.resource_utilization_history[-400:]
            )
        
        # Extract and store vulnerability patterns if found
        if result.findings:
            for finding in result.findings:
                pattern = {
                    'pattern_type': finding.get('type'),
                    'target_context': task.target,
                    'confidence': finding.get('confidence', 0.5),
                    'timestamp': datetime.utcnow().isoformat()
                }
                self.intelligence.vulnerability_patterns.append(pattern)
        
        self.intelligence.last_updated = datetime.utcnow()
    
    async def _attempt_failure_recovery(self, task: AgentTask, error: str) -> Optional[AgentResult]:
        """Attempt autonomous failure recovery"""
        
        if self.autonomous_config.autonomy_level == AutonomyLevel.MINIMAL:
            return None
            
        recovery_strategies = [
            self._retry_with_reduced_parameters,
            self._fallback_to_alternative_method,
            self._request_manual_intervention
        ]
        
        for strategy in recovery_strategies:
            try:
                result = await strategy(task, error)
                if result and result.success:
                    self.logger.info("Autonomous recovery successful",
                                   task_id=task.task_id,
                                   strategy=strategy.__name__)
                    return result
            except Exception as e:
                self.logger.debug("Recovery strategy failed",
                                strategy=strategy.__name__,
                                error=str(e))
                continue
                
        return None
    
    async def _retry_with_reduced_parameters(self, task: AgentTask, error: str) -> AgentResult:
        """Retry task with reduced parameters"""
        
        # Reduce timeout and complexity
        task.parameters['timeout'] = max(10, task.parameters.get('timeout', 30) * 0.5)
        task.parameters['max_depth'] = max(1, task.parameters.get('max_depth', 3) - 1)
        
        return await self._execute_with_monitoring(task)
    
    async def _fallback_to_alternative_method(self, task: AgentTask, error: str) -> AgentResult:
        """Try alternative execution method"""
        
        # Switch to alternative tool or method
        current_tool = task.parameters.get('tool', 'default')
        alternative_tools = {'nmap': 'masscan', 'requests': 'urllib', 'selenium': 'playwright'}
        
        alternative = alternative_tools.get(current_tool)
        if alternative:
            task.parameters['tool'] = alternative
            task.parameters['fallback_mode'] = True
            return await self._execute_with_monitoring(task)
            
        return AgentResult(task_id=task.task_id, success=False, errors=["No alternative method available"])
    
    async def _request_manual_intervention(self, task: AgentTask, error: str) -> AgentResult:
        """Request manual intervention for complex failures"""
        
        return AgentResult(
            task_id=task.task_id,
            success=False,
            errors=[f"Manual intervention required: {error}"],
            metadata={'intervention_requested': True, 'timestamp': datetime.utcnow().isoformat()}
        )
    
    async def get_autonomous_status(self) -> Dict[str, Any]:
        """Get detailed autonomous worker status"""
        
        base_status = self.get_status()
        
        autonomous_status = {
            **base_status,
            'autonomous_config': {
                'autonomy_level': self.autonomous_config.autonomy_level.value,
                'max_concurrent_tasks': self.autonomous_config.max_concurrent_tasks,
                'performance_learning_enabled': self.autonomous_config.performance_learning_enabled
            },
            'intelligence_summary': {
                'task_success_rates': dict(list(self.intelligence.task_success_rates.items())[:10]),
                'adaptation_count': self.adaptation_count,
                'security_violations_prevented': self.security_violations_prevented,
                'last_intelligence_update': self.intelligence.last_updated.isoformat()
            },
            'performance_metrics': {
                'total_adaptations': self.adaptation_count,
                'recent_success_rate': self._calculate_recent_success_rate(),
                'average_execution_time': self._calculate_average_execution_time()
            },
            'security_status': {
                'active_constraints': len([c for c in self.security_constraints if c.active]),
                'violations_prevented': self.security_violations_prevented,
                'compliance_rate': 1.0 - (self.security_violations_prevented / max(1, self.total_tasks_completed))
            }
        }
        
        return autonomous_status
    
    def _calculate_recent_success_rate(self) -> float:
        """Calculate success rate for recent executions"""
        
        if len(self.performance_history) < 10:
            return 0.0
            
        recent_history = self.performance_history[-50:]  # Last 50 executions
        successful = sum(1 for h in recent_history if h['success'])
        
        return successful / len(recent_history)
    
    def _calculate_average_execution_time(self) -> float:
        """Calculate average execution time"""
        
        if not self.performance_history:
            return 0.0
            
        times = [h.get('execution_time', 0) for h in self.performance_history[-100:]]
        return sum(times) / len(times) if times else 0.0


class ResourceMonitor:
    """Monitor system resources for autonomous optimization"""
    
    def __init__(self):
        self.logger = structlog.get_logger("ResourceMonitor")
        
    async def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        # Simplified implementation - would use psutil in production
        return min(0.9, time.time() % 1.0)
        
    async def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        # Simplified implementation - would use psutil in production
        return min(0.8, (time.time() * 1.3) % 1.0)
        
    async def get_current_load(self) -> float:
        """Get current system load"""
        cpu = await self.get_cpu_usage()
        memory = await self.get_memory_usage()
        return max(cpu, memory)


class DecisionEngine:
    """Autonomous decision making engine"""
    
    def __init__(self):
        self.logger = structlog.get_logger("DecisionEngine")
        
    async def should_adapt_workflow(self, performance_data: Dict[str, Any]) -> bool:
        """Decide if workflow adaptation is needed"""
        
        success_rate = performance_data.get('success_rate', 1.0)
        avg_time = performance_data.get('avg_execution_time', 0)
        
        # Adapt if success rate is low or execution time is high
        return success_rate < 0.7 or avg_time > 60.0


class TaskPrioritizer:
    """Autonomous task prioritization"""
    
    def __init__(self):
        self.logger = structlog.get_logger("TaskPrioritizer")
        
    async def prioritize_tasks(self, tasks: List[AgentTask], 
                             intelligence: WorkerIntelligence) -> List[AgentTask]:
        """Autonomously prioritize tasks based on intelligence"""
        
        def priority_score(task: AgentTask) -> float:
            base_priority = task.priority
            success_rate = intelligence.task_success_rates.get(task.task_type, 0.5)
            
            # Higher priority for tasks with higher success rates
            return base_priority * (1 + success_rate)
            
        return sorted(tasks, key=priority_score, reverse=True)


class WorkflowOptimizer:
    """Autonomous workflow optimization"""
    
    def __init__(self):
        self.logger = structlog.get_logger("WorkflowOptimizer")
        
    async def optimize_workflow(self, workflow: Dict[str, Any], 
                              performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize workflow based on performance data"""
        
        optimized = workflow.copy()
        
        # Adjust parallelism based on success rates
        if performance_data.get('success_rate', 1.0) < 0.6:
            optimized['parallel_execution'] = False
            optimized['retry_delays'] = [2, 4, 8]  # Exponential backoff
            
        return optimized