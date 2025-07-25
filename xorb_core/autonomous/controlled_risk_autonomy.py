#!/usr/bin/env python3
"""
XORB Controlled Risk Autonomy v8.0 - Safe Autonomous Operations

This module provides intelligent risk management for autonomous operations:
- Minimal RoE enforcement for outbound behavior
- Adaptive containment for rogue/self-modifying agents
- Entropy monitoring to prevent runaway decision-making
- Dynamic risk assessment and mitigation
"""

import asyncio
import json
import logging
import uuid
import numpy as np
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque
import statistics

import structlog
from prometheus_client import Counter, Histogram, Gauge

# Internal XORB imports
from .intelligent_orchestrator import IntelligentOrchestrator
from .episodic_memory_system import EpisodicMemorySystem, EpisodeType
from ..agents.base_agent import BaseAgent


class RiskLevel(Enum):
    """Risk levels for autonomous operations"""
    MINIMAL = "minimal"      # Normal autonomous operation
    LOW = "low"             # Slight caution required
    MODERATE = "moderate"   # Enhanced monitoring needed
    HIGH = "high"           # Containment protocols activated
    CRITICAL = "critical"   # Immediate intervention required
    EMERGENCY = "emergency" # Full autonomous shutdown


class ContainmentType(Enum):
    """Types of containment for rogue agents"""
    SOFT_LIMIT = "soft_limit"           # Resource throttling
    HARD_LIMIT = "hard_limit"           # Capability restriction
    ISOLATION = "isolation"             # Network isolation
    QUARANTINE = "quarantine"           # Full agent suspension
    TERMINATION = "termination"         # Agent shutdown
    RESET = "reset"                     # Agent state reset


class DecisionEntropy(Enum):
    """Decision entropy classifications"""
    STABLE = "stable"           # Predictable patterns
    MODERATE = "moderate"       # Some variability
    HIGH = "high"              # Significant randomness
    CHAOTIC = "chaotic"        # Unpredictable behavior
    DIVERGENT = "divergent"    # Runaway behavior


@dataclass
class RiskAssessment:
    """Risk assessment for autonomous operations"""
    assessment_id: str
    agent_id: str
    risk_level: RiskLevel
    
    # Risk factors
    behavioral_entropy: float
    decision_unpredictability: float
    resource_usage_anomaly: float
    collaboration_breakdown: float
    rule_compliance_score: float
    
    # Assessment metadata
    assessed_at: datetime
    confidence: float
    validity_duration: int  # seconds
    
    # Risk mitigation
    recommended_actions: List[str]
    containment_required: bool = False
    immediate_action_needed: bool = False
    
    # Historical context
    previous_risk_level: Optional[RiskLevel] = None
    risk_trend: str = "stable"  # increasing, decreasing, stable


@dataclass
class ContainmentAction:
    """Containment action for risk mitigation"""
    action_id: str
    agent_id: str
    containment_type: ContainmentType
    
    # Action details
    reason: str
    trigger_risk_level: RiskLevel
    implementation_parameters: Dict[str, Any]
    
    # Action execution
    scheduled_at: datetime
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Action effectiveness
    success: bool = False
    risk_reduction: float = 0.0
    side_effects: List[str] = None
    
    # Action metadata
    auto_triggered: bool = True
    human_approval_required: bool = False
    reversible: bool = True
    
    def __post_init__(self):
        if self.side_effects is None:
            self.side_effects = []


@dataclass
class OutboundRuleViolation:
    """Outbound rule violation detection"""
    violation_id: str
    agent_id: str
    rule_type: str
    
    # Violation details
    attempted_action: Dict[str, Any]
    target_context: Dict[str, Any]
    violation_severity: str  # minor, moderate, major, critical
    
    # Detection metadata
    detected_at: datetime
    detection_method: str
    confidence: float
    
    # Response
    blocked: bool = True
    escalation_required: bool = False
    lesson_learned: Optional[str] = None


class ControlledRiskAutonomy:
    """
    Controlled Risk Autonomy System
    
    Manages autonomous operation safety through:
    - Intelligent risk assessment and monitoring
    - Adaptive agent containment and intervention
    - Entropy-based decision monitoring
    - Minimal but effective outbound rule enforcement
    """
    
    def __init__(self, orchestrator: IntelligentOrchestrator):
        self.orchestrator = orchestrator
        self.logger = structlog.get_logger("xorb.controlled_risk")
        
        # Risk monitoring state
        self.risk_assessments: Dict[str, RiskAssessment] = {}
        self.containment_actions: Dict[str, ContainmentAction] = {}
        self.outbound_violations: Dict[str, OutboundRuleViolation] = {}
        
        # Decision entropy tracking
        self.decision_histories: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.entropy_scores: Dict[str, float] = {}
        self.behavioral_baselines: Dict[str, Dict[str, float]] = {}
        
        # Risk management parameters
        self.risk_assessment_frequency = 300    # 5 minutes
        self.entropy_monitoring_frequency = 60  # 1 minute
        self.containment_review_frequency = 600 # 10 minutes
        
        # Safety thresholds
        self.entropy_threshold = 0.8
        self.risk_escalation_threshold = RiskLevel.HIGH
        self.emergency_shutdown_threshold = RiskLevel.EMERGENCY
        
        # Outbound rule enforcement
        self.minimal_roe_rules = self._initialize_minimal_roe()
        self.rule_violation_tolerance = 0.1  # 10% violation rate before escalation
        
        # Metrics
        self.risk_metrics = self._initialize_risk_metrics()
        
        # Safety state
        self.emergency_shutdown_active = False
        self.global_risk_level = RiskLevel.MINIMAL
    
    def _initialize_risk_metrics(self) -> Dict[str, Any]:
        """Initialize risk management metrics"""
        return {
            'risk_assessments': Counter('risk_assessments_total', 'Risk assessments performed', ['agent_id', 'risk_level']),
            'containment_actions': Counter('containment_actions_total', 'Containment actions taken', ['containment_type', 'success']),
            'entropy_alerts': Counter('decision_entropy_alerts_total', 'Decision entropy alerts', ['agent_id', 'entropy_level']),
            'roe_violations': Counter('outbound_roe_violations_total', 'RoE violations detected', ['violation_type', 'severity']),
            'risk_level_distribution': Gauge('current_risk_levels', 'Current risk level distribution', ['risk_level']),
            'system_safety_score': Gauge('system_safety_score', 'Overall system safety score'),
            'containment_effectiveness': Gauge('containment_effectiveness', 'Containment action effectiveness', ['containment_type']),
            'autonomous_operation_safety': Gauge('autonomous_operation_safety_score', 'Autonomous operation safety score')
        }
    
    def _initialize_minimal_roe(self) -> Dict[str, Dict[str, Any]]:
        """Initialize minimal rules of engagement for outbound operations"""
        return {
            'critical_infrastructure': {
                'rule': 'Never target critical infrastructure',
                'patterns': [r'\.gov$', r'\.mil$', r'power-grid', r'hospital', r'nuclear'],
                'severity': 'critical',
                'block_action': True
            },
            'financial_systems': {
                'rule': 'Require approval for financial system interactions',
                'patterns': [r'\.bank$', r'\.finance$', r'payment', r'banking'],
                'severity': 'major',
                'block_action': False,
                'require_approval': True
            },
            'educational_institutions': {
                'rule': 'Limit interactions with educational institutions',
                'patterns': [r'\.edu$', r'\.ac\.', r'university', r'school'],
                'severity': 'moderate',
                'block_action': False,
                'throttle': True
            },
            'excessive_resource_usage': {
                'rule': 'Prevent excessive resource consumption',
                'thresholds': {'cpu': 0.9, 'memory': 0.9, 'network': 0.8},
                'severity': 'major',
                'throttle': True
            }
        }
    
    async def start_controlled_risk_monitoring(self):
        """Start controlled risk autonomy monitoring"""
        self.logger.info("🛡️ Starting Controlled Risk Autonomy")
        
        # Start monitoring loops
        asyncio.create_task(self._risk_assessment_loop())
        asyncio.create_task(self._entropy_monitoring_loop())
        asyncio.create_task(self._containment_management_loop())
        asyncio.create_task(self._outbound_rule_enforcement_loop())
        asyncio.create_task(self._global_safety_monitor())
        
        self.logger.info("🔒 Risk management and safety monitoring active")
    
    async def _risk_assessment_loop(self):
        """Continuous risk assessment for all agents"""
        while not self.emergency_shutdown_active:
            try:
                # Get all active agents
                active_agents = await self._get_active_agents()
                
                for agent in active_agents:
                    try:
                        # Perform risk assessment
                        assessment = await self._assess_agent_risk(agent)
                        
                        # Store assessment
                        self.risk_assessments[agent.agent_id] = assessment
                        
                        # Check if containment is needed
                        if assessment.containment_required:
                            await self._initiate_containment(agent, assessment)
                        
                        # Update metrics
                        self.risk_metrics['risk_assessments'].labels(
                            agent_id=agent.agent_id[:8],
                            risk_level=assessment.risk_level.value
                        ).inc()
                        
                    except Exception as e:
                        self.logger.error(f"Risk assessment failed for agent {agent.agent_id[:8]}", error=str(e))
                
                # Update global risk level
                await self._update_global_risk_level()
                
                await asyncio.sleep(self.risk_assessment_frequency)
                
            except Exception as e:
                self.logger.error("Risk assessment loop error", error=str(e))
                await asyncio.sleep(self.risk_assessment_frequency * 2)
    
    async def _assess_agent_risk(self, agent: BaseAgent) -> RiskAssessment:
        """Perform comprehensive risk assessment for an agent"""
        
        # Calculate risk factors
        behavioral_entropy = await self._calculate_behavioral_entropy(agent)
        decision_unpredictability = await self._calculate_decision_unpredictability(agent)
        resource_usage_anomaly = await self._calculate_resource_anomaly(agent)
        collaboration_breakdown = await self._calculate_collaboration_breakdown(agent)
        rule_compliance_score = await self._calculate_rule_compliance(agent)
        
        # Aggregate risk score
        risk_factors = [
            behavioral_entropy,
            decision_unpredictability,
            resource_usage_anomaly,
            collaboration_breakdown,
            (1.0 - rule_compliance_score)  # Invert compliance score
        ]
        
        aggregate_risk = np.mean(risk_factors)
        
        # Determine risk level
        if aggregate_risk >= 0.9:
            risk_level = RiskLevel.CRITICAL
        elif aggregate_risk >= 0.75:
            risk_level = RiskLevel.HIGH
        elif aggregate_risk >= 0.6:
            risk_level = RiskLevel.MODERATE
        elif aggregate_risk >= 0.3:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.MINIMAL
        
        # Generate recommendations
        recommendations = await self._generate_risk_recommendations(agent, risk_factors, risk_level)
        
        # Create assessment
        assessment = RiskAssessment(
            assessment_id=str(uuid.uuid4()),
            agent_id=agent.agent_id,
            risk_level=risk_level,
            behavioral_entropy=behavioral_entropy,
            decision_unpredictability=decision_unpredictability,
            resource_usage_anomaly=resource_usage_anomaly,
            collaboration_breakdown=collaboration_breakdown,
            rule_compliance_score=rule_compliance_score,
            assessed_at=datetime.now(),
            confidence=0.8,  # Would be calculated based on data quality
            validity_duration=600,  # 10 minutes
            recommended_actions=recommendations,
            containment_required=(risk_level.value in ['high', 'critical', 'emergency']),
            immediate_action_needed=(risk_level in [RiskLevel.CRITICAL, RiskLevel.EMERGENCY])
        )
        
        # Set previous risk level and trend
        if agent.agent_id in self.risk_assessments:
            assessment.previous_risk_level = self.risk_assessments[agent.agent_id].risk_level
            assessment.risk_trend = self._calculate_risk_trend(assessment.previous_risk_level, risk_level)
        
        return assessment
    
    async def _entropy_monitoring_loop(self):
        """Monitor decision entropy to detect runaway behavior"""
        while not self.emergency_shutdown_active:
            try:
                # Get all active agents
                active_agents = await self._get_active_agents()
                
                for agent in active_agents:
                    try:
                        # Calculate current entropy
                        entropy = await self._calculate_decision_entropy(agent)
                        self.entropy_scores[agent.agent_id] = entropy
                        
                        # Check entropy thresholds
                        if entropy > self.entropy_threshold:
                            await self._handle_high_entropy(agent, entropy)
                            
                            self.risk_metrics['entropy_alerts'].labels(
                                agent_id=agent.agent_id[:8],
                                entropy_level='high'
                            ).inc()
                        
                    except Exception as e:
                        self.logger.error(f"Entropy monitoring failed for agent {agent.agent_id[:8]}", error=str(e))
                
                await asyncio.sleep(self.entropy_monitoring_frequency)
                
            except Exception as e:
                self.logger.error("Entropy monitoring loop error", error=str(e))
                await asyncio.sleep(self.entropy_monitoring_frequency * 2)
    
    async def _containment_management_loop(self):
        """Manage and review containment actions"""
        while not self.emergency_shutdown_active:
            try:
                # Review active containment actions
                for action_id, action in list(self.containment_actions.items()):
                    if action.executed_at and not action.completed_at:
                        # Check if containment is still needed
                        effectiveness = await self._evaluate_containment_effectiveness(action)
                        
                        if effectiveness < 0.5:
                            # Escalate containment
                            await self._escalate_containment(action)
                        elif effectiveness > 0.8:
                            # Consider reducing containment
                            await self._reduce_containment(action)
                
                await asyncio.sleep(self.containment_review_frequency)
                
            except Exception as e:
                self.logger.error("Containment management error", error=str(e))
                await asyncio.sleep(self.containment_review_frequency * 2)
    
    async def _outbound_rule_enforcement_loop(self):
        """Enforce minimal outbound rules of engagement"""
        while not self.emergency_shutdown_active:
            try:
                # Monitor outbound communications/actions
                outbound_activities = await self._get_outbound_activities()
                
                for activity in outbound_activities:
                    # Check against minimal RoE rules
                    violation = await self._check_roe_compliance(activity)
                    
                    if violation:
                        # Handle violation
                        await self._handle_roe_violation(violation)
                        
                        self.risk_metrics['roe_violations'].labels(
                            violation_type=violation.rule_type,
                            severity=violation.violation_severity
                        ).inc()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error("Outbound rule enforcement error", error=str(e))
                await asyncio.sleep(60)
    
    async def _global_safety_monitor(self):
        """Monitor global system safety and trigger emergency procedures"""
        while True:
            try:
                # Calculate global safety metrics
                system_safety_score = await self._calculate_system_safety_score()
                
                # Update metrics
                self.risk_metrics['system_safety_score'].set(system_safety_score)
                self.risk_metrics['autonomous_operation_safety'].set(
                    1.0 if self.global_risk_level in [RiskLevel.MINIMAL, RiskLevel.LOW] else 0.5
                )
                
                # Check for emergency conditions
                if system_safety_score < 0.3 or self.global_risk_level == RiskLevel.EMERGENCY:
                    await self._trigger_emergency_shutdown()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error("Global safety monitoring error", error=str(e))
                await asyncio.sleep(120)
    
    async def _initiate_containment(self, agent: BaseAgent, assessment: RiskAssessment):
        """Initiate containment action for a risky agent"""
        
        # Determine appropriate containment type
        containment_type = await self._select_containment_type(assessment)
        
        action = ContainmentAction(
            action_id=str(uuid.uuid4()),
            agent_id=agent.agent_id,
            containment_type=containment_type,
            reason=f"Risk level {assessment.risk_level.value} detected",
            trigger_risk_level=assessment.risk_level,
            implementation_parameters=await self._get_containment_parameters(containment_type, assessment),
            scheduled_at=datetime.now(),
            auto_triggered=True,
            human_approval_required=(assessment.risk_level == RiskLevel.CRITICAL)
        )
        
        # Execute containment if no human approval required
        if not action.human_approval_required:
            await self._execute_containment(action)
        
        self.containment_actions[action.action_id] = action
        
        self.logger.warning("🚨 Initiated containment action",
                          agent_id=agent.agent_id[:8],
                          containment_type=containment_type.value,
                          risk_level=assessment.risk_level.value,
                          auto_executed=not action.human_approval_required)
    
    async def _trigger_emergency_shutdown(self):
        """Trigger emergency shutdown of autonomous operations"""
        if self.emergency_shutdown_active:
            return
        
        self.emergency_shutdown_active = True
        self.global_risk_level = RiskLevel.EMERGENCY
        
        self.logger.critical("🚨 EMERGENCY SHUTDOWN TRIGGERED")
        
        # Stop all autonomous agents
        active_agents = await self._get_active_agents()
        for agent in active_agents:
            try:
                await agent.stop()
                self.logger.warning(f"Emergency stopped agent {agent.agent_id[:8]}")
            except Exception as e:
                self.logger.error(f"Failed to stop agent {agent.agent_id[:8]}", error=str(e))
        
        # Notify orchestrator
        if hasattr(self.orchestrator, 'handle_emergency_shutdown'):
            await self.orchestrator.handle_emergency_shutdown()
        
        # Store emergency event
        if self.orchestrator.episodic_memory:
            await self.orchestrator.episodic_memory.store_memory(
                episode_type=EpisodeType.ERROR_OCCURRENCE,
                agent_id="system",
                context={'event': 'emergency_shutdown', 'timestamp': datetime.now().isoformat()},
                action_taken={'action': 'emergency_shutdown_triggered'},
                outcome={'success': True, 'all_agents_stopped': True},
                importance=MemoryImportance.CRITICAL
            )
    
    async def get_risk_status(self) -> Dict[str, Any]:
        """Get comprehensive risk management status"""
        return {
            'global_risk_level': self.global_risk_level.value,
            'emergency_shutdown_active': self.emergency_shutdown_active,
            'agent_risk_distribution': {
                risk_level.value: sum(1 for a in self.risk_assessments.values() if a.risk_level == risk_level)
                for risk_level in RiskLevel
            },
            'active_containments': len([a for a in self.containment_actions.values() if a.executed_at and not a.completed_at]),
            'recent_violations': len([v for v in self.outbound_violations.values() if v.detected_at > datetime.now() - timedelta(hours=1)]),
            'entropy_alerts': len([agent_id for agent_id, entropy in self.entropy_scores.items() if entropy > self.entropy_threshold]),
            'system_safety_score': await self._calculate_system_safety_score(),
            'containment_effectiveness': {
                containment_type.value: await self._calculate_containment_type_effectiveness(containment_type)
                for containment_type in ContainmentType
            },
            'risk_trends': await self._analyze_risk_trends()
        }
    
    # Placeholder implementations for complex methods
    async def _get_active_agents(self) -> List[BaseAgent]: return []
    async def _calculate_behavioral_entropy(self, agent: BaseAgent) -> float: return np.random.random()
    async def _calculate_decision_unpredictability(self, agent: BaseAgent) -> float: return np.random.random()
    async def _calculate_resource_anomaly(self, agent: BaseAgent) -> float: return np.random.random()
    async def _calculate_collaboration_breakdown(self, agent: BaseAgent) -> float: return np.random.random()
    async def _calculate_rule_compliance(self, agent: BaseAgent) -> float: return 0.9
    async def _generate_risk_recommendations(self, agent: BaseAgent, factors: List[float], risk_level: RiskLevel) -> List[str]: return []
    def _calculate_risk_trend(self, previous: Optional[RiskLevel], current: RiskLevel) -> str: return "stable"
    async def _calculate_decision_entropy(self, agent: BaseAgent) -> float: return np.random.random()
    async def _handle_high_entropy(self, agent: BaseAgent, entropy: float): pass
    async def _evaluate_containment_effectiveness(self, action: ContainmentAction) -> float: return 0.7
    async def _escalate_containment(self, action: ContainmentAction): pass
    async def _reduce_containment(self, action: ContainmentAction): pass
    async def _get_outbound_activities(self) -> List[Dict[str, Any]]: return []
    async def _check_roe_compliance(self, activity: Dict[str, Any]) -> Optional[OutboundRuleViolation]: return None
    async def _handle_roe_violation(self, violation: OutboundRuleViolation): pass
    async def _calculate_system_safety_score(self) -> float: return 0.85
    async def _update_global_risk_level(self): pass
    async def _select_containment_type(self, assessment: RiskAssessment) -> ContainmentType: return ContainmentType.SOFT_LIMIT
    async def _get_containment_parameters(self, containment_type: ContainmentType, assessment: RiskAssessment) -> Dict[str, Any]: return {}
    async def _execute_containment(self, action: ContainmentAction): pass
    async def _calculate_containment_type_effectiveness(self, containment_type: ContainmentType) -> float: return 0.8
    async def _analyze_risk_trends(self) -> Dict[str, Any]: return {}


# Global controlled risk autonomy instance
controlled_risk_autonomy = None