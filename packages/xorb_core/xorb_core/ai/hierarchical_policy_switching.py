#!/usr/bin/env python3
"""
Hierarchical Policy Switching System for Xorb 2.0

This module implements a hierarchical policy architecture that dynamically switches
between macro-level strategic policies and micro-level tactical policies based on
context, performance, and environmental conditions.
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import json
import abc
from pathlib import Path


class PolicyLevel(Enum):
    """Hierarchical policy levels"""
    STRATEGIC = "strategic"         # Long-term campaign strategy (hours/days)
    OPERATIONAL = "operational"     # Medium-term operations (minutes/hours)  
    TACTICAL = "tactical"           # Short-term tactics (seconds/minutes)
    REACTIVE = "reactive"           # Immediate reactions (milliseconds/seconds)


class PolicySwitchTrigger(Enum):
    """Triggers for policy switching"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CONTEXT_CHANGE = "context_change"
    TIME_BASED = "time_based"
    RESOURCE_CONSTRAINT = "resource_constraint"
    UNCERTAINTY_INCREASE = "uncertainty_increase"
    MANUAL_OVERRIDE = "manual_override"
    HIERARCHICAL_SIGNAL = "hierarchical_signal"


@dataclass
class PolicyContext:
    """Context information for policy decision making"""
    # Environmental context
    target_environment: Dict[str, Any]
    system_resources: Dict[str, float]
    time_constraints: Dict[str, float]
    
    # Performance context
    recent_performance: Dict[str, float]
    success_rate: float
    resource_efficiency: float
    
    # Strategic context
    campaign_objectives: List[str]
    current_phase: str
    risk_tolerance: float
    
    # Temporal context
    timestamp: datetime
    time_since_last_switch: float
    policy_history: List[str]
    
    # Meta context
    uncertainty_level: float
    novelty_score: float
    complexity_estimate: float


@dataclass
class PolicySwitchEvent:
    """Event representing a policy switch"""
    event_id: str
    timestamp: datetime
    
    # Switch details
    from_policy: str
    to_policy: str
    from_level: PolicyLevel
    to_level: PolicyLevel
    
    # Switch trigger and reasoning
    trigger: PolicySwitchTrigger
    trigger_confidence: float
    switch_rationale: str
    
    # Context at switch
    context_snapshot: PolicyContext
    
    # Performance tracking
    expected_improvement: float
    actual_improvement: Optional[float] = None
    switch_overhead: float = 0.0
    
    # Hierarchical information
    parent_policy_signal: Optional[str] = None
    child_policies_affected: List[str] = field(default_factory=list)


class BasePolicy(abc.ABC):
    """Abstract base class for hierarchical policies"""
    
    def __init__(self, policy_id: str, policy_level: PolicyLevel):
        self.policy_id = policy_id
        self.policy_level = policy_level
        self.activation_count = 0
        self.total_execution_time = 0.0
        self.performance_history = deque(maxlen=100)
        self.logger = logging.getLogger(__name__)
    
    @abc.abstractmethod
    async def evaluate_action(self, context: PolicyContext) -> Dict[str, Any]:
        """Evaluate and return action based on current context"""
        pass
    
    @abc.abstractmethod  
    async def assess_performance(self, 
                               context: PolicyContext,
                               action_result: Dict[str, Any]) -> float:
        """Assess performance of the policy given context and results"""
        pass
    
    @abc.abstractmethod
    async def should_switch_policy(self, context: PolicyContext) -> Tuple[bool, str]:
        """Determine if policy should be switched and provide reason"""
        pass
    
    async def activate(self, context: PolicyContext):
        """Called when policy becomes active"""
        self.activation_count += 1
        self.logger.debug(f"Policy {self.policy_id} activated")
    
    async def deactivate(self, context: PolicyContext):
        """Called when policy becomes inactive"""
        self.logger.debug(f"Policy {self.policy_id} deactivated")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for this policy"""
        if not self.performance_history:
            return {'average_performance': 0.0, 'performance_variance': 0.0}
        
        performance_values = list(self.performance_history)
        return {
            'average_performance': np.mean(performance_values),
            'performance_variance': np.var(performance_values),
            'performance_trend': self._calculate_trend(performance_values),
            'activation_count': self.activation_count,
            'average_execution_time': self.total_execution_time / max(self.activation_count, 1)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate performance trend (-1 to 1)"""
        if len(values) < 3:
            return 0.0
        
        x = np.arange(len(values))
        correlation = np.corrcoef(x, values)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0


class StrategicPolicy(BasePolicy):
    """Strategic-level policy for long-term planning"""
    
    def __init__(self, policy_id: str, strategy_type: str = "balanced"):
        super().__init__(policy_id, PolicyLevel.STRATEGIC)
        self.strategy_type = strategy_type
        self.planning_horizon_hours = 24
        self.objective_weights = {
            'stealth': 0.3,
            'speed': 0.3,
            'thoroughness': 0.2,
            'resource_efficiency': 0.2
        }
    
    async def evaluate_action(self, context: PolicyContext) -> Dict[str, Any]:
        """Evaluate strategic action based on campaign objectives"""
        
        # Analyze current campaign phase and objectives
        phase_strategy = await self._determine_phase_strategy(context)
        resource_allocation = await self._optimize_resource_allocation(context)
        risk_assessment = await self._assess_strategic_risks(context)
        
        return {
            'action_type': 'strategic_directive',
            'phase_strategy': phase_strategy,
            'resource_allocation': resource_allocation,
            'risk_assessment': risk_assessment,
            'time_horizon_hours': self.planning_horizon_hours,
            'success_criteria': await self._define_success_criteria(context),
            'contingency_plans': await self._generate_contingencies(context)
        }
    
    async def assess_performance(self, 
                               context: PolicyContext,
                               action_result: Dict[str, Any]) -> float:
        """Assess strategic performance based on long-term objectives"""
        
        performance_score = 0.0
        
        # Objective achievement
        for objective in context.campaign_objectives:
            achievement_score = action_result.get(f'{objective}_achievement', 0.0)
            weight = self.objective_weights.get(objective, 0.25)
            performance_score += weight * achievement_score
        
        # Resource efficiency
        resource_efficiency = context.resource_efficiency
        performance_score += 0.2 * resource_efficiency
        
        # Risk management
        risk_mitigation = action_result.get('risk_mitigation_effectiveness', 0.5)
        performance_score += 0.1 * risk_mitigation
        
        # Timeline adherence
        timeline_adherence = action_result.get('timeline_adherence', 0.5)
        performance_score += 0.1 * timeline_adherence
        
        self.performance_history.append(performance_score)
        return performance_score
    
    async def should_switch_policy(self, context: PolicyContext) -> Tuple[bool, str]:
        """Determine if strategic policy should be switched"""
        
        # Strategic policies change infrequently
        if context.time_since_last_switch < 3600:  # Less than 1 hour
            return False, "Strategic policy requires time to show effectiveness"
        
        # Check for major context changes
        if context.novelty_score > 0.8:
            return True, "Significant environmental changes detected"
        
        # Check performance degradation
        if len(self.performance_history) >= 5:
            recent_performance = np.mean(list(self.performance_history)[-5:])
            if recent_performance < 0.3:
                return True, "Strategic performance below acceptable threshold"
        
        # Check objective changes
        if len(context.campaign_objectives) != len(self.objective_weights):
            return True, "Campaign objectives have changed"
        
        return False, "Strategic policy performing adequately"
    
    async def _determine_phase_strategy(self, context: PolicyContext) -> Dict[str, Any]:
        """Determine strategy for current campaign phase"""
        
        phase = context.current_phase
        
        if phase == "reconnaissance":
            return {
                'primary_approach': 'stealth_first',
                'information_gathering_priority': 'high',
                'risk_tolerance': 'low',
                'speed_priority': 'medium'
            }
        elif phase == "exploitation":
            return {
                'primary_approach': 'targeted_strikes',
                'exploitation_depth': 'thorough',
                'risk_tolerance': 'medium',
                'speed_priority': 'high'
            }
        elif phase == "persistence":
            return {
                'primary_approach': 'establish_foothold',
                'persistence_methods': 'multiple',
                'risk_tolerance': 'low',
                'stealth_priority': 'high'
            }
        else:
            return {
                'primary_approach': 'adaptive',
                'risk_tolerance': 'medium',
                'speed_priority': 'medium'
            }
    
    async def _optimize_resource_allocation(self, context: PolicyContext) -> Dict[str, float]:
        """Optimize resource allocation across operational areas"""
        
        available_resources = context.system_resources
        total_cpu = available_resources.get('cpu_available', 1.0)
        total_memory = available_resources.get('memory_available', 1.0)
        
        # Allocate based on strategy type and phase
        if self.strategy_type == "aggressive":
            allocation = {
                'reconnaissance': 0.2,
                'exploitation': 0.5,
                'persistence': 0.2,
                'monitoring': 0.1
            }
        elif self.strategy_type == "stealth":
            allocation = {
                'reconnaissance': 0.4,
                'exploitation': 0.3,
                'persistence': 0.2,
                'monitoring': 0.1
            }
        else:  # balanced
            allocation = {
                'reconnaissance': 0.3,
                'exploitation': 0.3,
                'persistence': 0.3,
                'monitoring': 0.1
            }
        
        # Adjust for current phase
        current_phase = context.current_phase
        if current_phase in allocation:
            # Boost current phase allocation
            boost = 0.2
            allocation[current_phase] = min(0.8, allocation[current_phase] + boost)
            
            # Reduce others proportionally
            remaining = 1.0 - allocation[current_phase]
            other_phases = [p for p in allocation if p != current_phase]
            for phase in other_phases:
                allocation[phase] *= remaining / sum(allocation[p] for p in other_phases)
        
        return allocation


class OperationalPolicy(BasePolicy):
    """Operational-level policy for medium-term operations"""
    
    def __init__(self, policy_id: str, operation_type: str = "standard"):
        super().__init__(policy_id, PolicyLevel.OPERATIONAL)
        self.operation_type = operation_type
        self.planning_horizon_minutes = 60
        self.tactical_policies = []
    
    async def evaluate_action(self, context: PolicyContext) -> Dict[str, Any]:
        """Evaluate operational action based on strategic directives"""
        
        # Translate strategic directives into operational plans
        operational_plan = await self._create_operational_plan(context)
        resource_requirements = await self._calculate_resource_requirements(context)
        timeline = await self._create_execution_timeline(context)
        
        return {
            'action_type': 'operational_plan',
            'operational_plan': operational_plan,
            'resource_requirements': resource_requirements,
            'execution_timeline': timeline,
            'success_metrics': await self._define_operational_metrics(context),
            'tactical_policies_needed': await self._select_tactical_policies(context)
        }
    
    async def assess_performance(self, 
                               context: PolicyContext,
                               action_result: Dict[str, Any]) -> float:
        """Assess operational performance"""
        
        performance_score = 0.0
        
        # Plan execution success
        plan_success = action_result.get('plan_execution_success', 0.0)
        performance_score += 0.4 * plan_success
        
        # Resource utilization efficiency
        resource_efficiency = action_result.get('resource_utilization_efficiency', 0.0)
        performance_score += 0.2 * resource_efficiency
        
        # Timeline adherence
        timeline_adherence = action_result.get('timeline_adherence', 0.0)
        performance_score += 0.2 * timeline_adherence
        
        # Tactical coordination
        tactical_coordination = action_result.get('tactical_coordination_effectiveness', 0.0)
        performance_score += 0.2 * tactical_coordination
        
        self.performance_history.append(performance_score)
        return performance_score
    
    async def should_switch_policy(self, context: PolicyContext) -> Tuple[bool, str]:
        """Determine if operational policy should be switched"""
        
        # Check for resource constraints
        if context.system_resources.get('cpu_available', 1.0) < 0.3:
            return True, "Insufficient CPU resources for current operational policy"
        
        # Check for tactical feedback
        recent_tactical_performance = context.recent_performance.get('tactical_average', 0.5)
        if recent_tactical_performance < 0.4:
            return True, "Tactical policies underperforming, need operational adjustment"
        
        # Check for context changes
        if context.complexity_estimate > 0.8 and self.operation_type == "simple":
            return True, "Context complexity requires more sophisticated operational approach"
        
        # Check performance trend
        if len(self.performance_history) >= 3:
            recent_trend = self._calculate_trend(list(self.performance_history)[-3:])
            if recent_trend < -0.5:
                return True, "Operational performance declining, switch needed"
        
        return False, "Operational policy performing within acceptable range"


class TacticalPolicy(BasePolicy):
    """Tactical-level policy for short-term actions"""
    
    def __init__(self, policy_id: str, tactic_type: str = "adaptive"):
        super().__init__(policy_id, PolicyLevel.TACTICAL)
        self.tactic_type = tactic_type
        self.action_horizon_seconds = 300  # 5 minutes
        self.action_history = deque(maxlen=50)
    
    async def evaluate_action(self, context: PolicyContext) -> Dict[str, Any]:
        """Evaluate tactical action based on immediate context"""
        
        # Analyze immediate environment
        immediate_actions = await self._identify_immediate_actions(context)
        action_priorities = await self._prioritize_actions(context, immediate_actions)
        execution_plan = await self._create_execution_plan(context, action_priorities)
        
        return {
            'action_type': 'tactical_execution',
            'immediate_actions': immediate_actions,
            'action_priorities': action_priorities,
            'execution_plan': execution_plan,
            'expected_duration_seconds': self.action_horizon_seconds,
            'fallback_options': await self._generate_fallbacks(context)
        }
    
    async def assess_performance(self, 
                               context: PolicyContext,
                               action_result: Dict[str, Any]) -> float:
        """Assess tactical performance"""
        
        performance_score = 0.0
        
        # Action success rate
        action_success = action_result.get('action_success_rate', 0.0)
        performance_score += 0.5 * action_success
        
        # Execution speed
        execution_speed = action_result.get('execution_speed_score', 0.0)
        performance_score += 0.2 * execution_speed
        
        # Resource efficiency
        resource_efficiency = action_result.get('resource_efficiency', 0.0)
        performance_score += 0.2 * resource_efficiency
        
        # Adaptability
        adaptability_score = action_result.get('adaptability_score', 0.0)
        performance_score += 0.1 * adaptability_score
        
        self.performance_history.append(performance_score)
        self.action_history.append({
            'timestamp': datetime.utcnow(),
            'performance': performance_score,
            'context_hash': hash(str(context.target_environment))
        })
        
        return performance_score
    
    async def should_switch_policy(self, context: PolicyContext) -> Tuple[bool, str]:
        """Determine if tactical policy should be switched"""
        
        # Tactical policies switch more frequently
        if context.time_since_last_switch < 60:  # Less than 1 minute
            return False, "Recent switch, allow time for tactical policy to stabilize"
        
        # Check for immediate performance issues
        if context.recent_performance.get('success_rate', 0.5) < 0.3:
            return True, "Immediate performance degradation detected"
        
        # Check for environmental changes
        if context.novelty_score > 0.6:
            return True, "Environmental changes require tactical adaptation"
        
        # Check resource constraints
        if context.system_resources.get('memory_available', 1.0) < 0.2:
            return True, "Memory constraints require different tactical approach"
        
        # Check uncertainty levels
        if context.uncertainty_level > 0.7:
            return True, "High uncertainty requires more robust tactical policy"
        
        return False, "Tactical policy performing adequately"


class ReactivePolicy(BasePolicy):
    """Reactive-level policy for immediate responses"""
    
    def __init__(self, policy_id: str, reaction_type: str = "defensive"):
        super().__init__(policy_id, PolicyLevel.REACTIVE)
        self.reaction_type = reaction_type
        self.response_time_ms = 100
        self.reaction_patterns = {}
    
    async def evaluate_action(self, context: PolicyContext) -> Dict[str, Any]:
        """Evaluate immediate reactive response"""
        
        # Detect immediate threats or opportunities
        threats = await self._detect_immediate_threats(context)
        opportunities = await self._detect_immediate_opportunities(context)
        
        # Generate immediate response
        immediate_response = await self._generate_immediate_response(context, threats, opportunities)
        
        return {
            'action_type': 'immediate_reaction',
            'immediate_response': immediate_response,
            'response_time_ms': self.response_time_ms,
            'threat_level': len(threats),
            'opportunity_level': len(opportunities),
            'automated_actions': await self._get_automated_actions(context)
        }
    
    async def assess_performance(self, 
                               context: PolicyContext,
                               action_result: Dict[str, Any]) -> float:
        """Assess reactive performance"""
        
        performance_score = 0.0
        
        # Response time
        actual_response_time = action_result.get('actual_response_time_ms', 1000)
        response_score = max(0.0, 1.0 - (actual_response_time / 1000.0))  # Normalize to seconds
        performance_score += 0.4 * response_score
        
        # Threat mitigation
        threat_mitigation = action_result.get('threat_mitigation_effectiveness', 0.0)
        performance_score += 0.3 * threat_mitigation
        
        # Opportunity capture
        opportunity_capture = action_result.get('opportunity_capture_rate', 0.0)
        performance_score += 0.2 * opportunity_capture
        
        # System stability
        stability_maintained = action_result.get('system_stability_maintained', 0.0)
        performance_score += 0.1 * stability_maintained
        
        self.performance_history.append(performance_score)
        return performance_score
    
    async def should_switch_policy(self, context: PolicyContext) -> Tuple[bool, str]:
        """Determine if reactive policy should be switched"""
        
        # Reactive policies switch based on immediate conditions
        
        # Check threat level changes
        current_threat_level = len(context.target_environment.get('active_threats', []))
        if current_threat_level > 5 and self.reaction_type == "passive":
            return True, "High threat level requires active reactive policy"
        
        # Check response time requirements
        if context.time_constraints.get('max_response_time_ms', 1000) < self.response_time_ms:
            return True, "Tighter response time requirements"
        
        # Check system resource availability
        if context.system_resources.get('cpu_available', 1.0) < 0.1:
            return True, "Critical resource shortage requires minimal reactive policy"
        
        return False, "Reactive policy appropriate for current conditions"


class PolicySwitchingDecisionNetwork(nn.Module):
    """Neural network for policy switching decisions"""
    
    def __init__(self,
                 context_dim: int = 256,
                 policy_dim: int = 64,
                 hidden_dim: int = 512,
                 num_policy_levels: int = 4):
        super().__init__()
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Policy state encoder
        self.policy_encoder = nn.Sequential(
            nn.Linear(policy_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Performance predictor
        self.performance_predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Switch necessity classifier
        self.switch_classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # switch / no_switch
            nn.Softmax(dim=-1)
        )
        
        # Policy level selector
        self.level_selector = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_policy_levels),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, 
                context_features: torch.Tensor,
                current_policy_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # Encode inputs
        context_encoded = self.context_encoder(context_features)
        policy_encoded = self.policy_encoder(current_policy_features)
        
        # Combine features
        combined_features = torch.cat([context_encoded, policy_encoded], dim=-1)
        
        # Generate predictions
        performance_prediction = self.performance_predictor(combined_features)
        switch_probability = self.switch_classifier(combined_features)
        level_preferences = self.level_selector(combined_features)
        
        return {
            'performance_prediction': performance_prediction,
            'switch_probability': switch_probability,
            'level_preferences': level_preferences,
            'combined_features': combined_features
        }


class HierarchicalPolicySwitchingSystem:
    """Main hierarchical policy switching system"""
    
    def __init__(self,
                 switching_model_config: Dict[str, Any] = None,
                 device: str = 'auto'):
        
        self.logger = logging.getLogger(__name__)
        
        # Device configuration
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Neural switching model
        model_config = switching_model_config or {}
        self.switching_network = PolicySwitchingDecisionNetwork(**model_config).to(self.device)
        
        # Policy registry
        self.policies: Dict[PolicyLevel, Dict[str, BasePolicy]] = {
            level: {} for level in PolicyLevel
        }
        
        # Active policies for each level
        self.active_policies: Dict[PolicyLevel, BasePolicy] = {}
        
        # Switching history
        self.switch_history: List[PolicySwitchEvent] = []
        self.max_history_size = 1000
        
        # Performance tracking
        self.level_performance: Dict[PolicyLevel, deque] = {
            level: deque(maxlen=100) for level in PolicyLevel
        }
        
        # System metrics
        self.switching_metrics = {
            'total_switches': 0,
            'switches_by_trigger': defaultdict(int),
            'switches_by_level': defaultdict(int),
            'average_switch_frequency_per_hour': 0.0,
            'successful_switches': 0,
            'switching_overhead_ms': 0.0
        }
        
        # Initialize default policies
        self._initialize_default_policies()
        
        self.logger.info(f"Hierarchical policy switching system initialized on {self.device}")
    
    def _initialize_default_policies(self):
        """Initialize default policies for each level"""
        
        # Strategic policies
        self.register_policy(StrategicPolicy("balanced_strategy", "balanced"))
        self.register_policy(StrategicPolicy("aggressive_strategy", "aggressive"))
        self.register_policy(StrategicPolicy("stealth_strategy", "stealth"))
        
        # Operational policies
        self.register_policy(OperationalPolicy("standard_ops", "standard"))
        self.register_policy(OperationalPolicy("rapid_ops", "rapid"))
        self.register_policy(OperationalPolicy("careful_ops", "careful"))
        
        # Tactical policies
        self.register_policy(TacticalPolicy("adaptive_tactics", "adaptive"))
        self.register_policy(TacticalPolicy("aggressive_tactics", "aggressive"))
        self.register_policy(TacticalPolicy("defensive_tactics", "defensive"))
        
        # Reactive policies
        self.register_policy(ReactivePolicy("standard_reactive", "balanced"))
        self.register_policy(ReactivePolicy("fast_reactive", "fast"))
        self.register_policy(ReactivePolicy("defensive_reactive", "defensive"))
        
        # Activate initial policies
        for level in PolicyLevel:
            if self.policies[level]:
                first_policy = list(self.policies[level].values())[0]
                self.active_policies[level] = first_policy
    
    def register_policy(self, policy: BasePolicy):
        """Register a policy in the system"""
        
        self.policies[policy.policy_level][policy.policy_id] = policy
        self.logger.info(f"Registered policy {policy.policy_id} at level {policy.policy_level.value}")
    
    async def evaluate_hierarchical_action(self, context: PolicyContext) -> Dict[str, Any]:
        """Evaluate actions across all policy levels"""
        
        hierarchical_actions = {}
        
        # Evaluate from strategic to reactive
        for level in [PolicyLevel.STRATEGIC, PolicyLevel.OPERATIONAL, 
                     PolicyLevel.TACTICAL, PolicyLevel.REACTIVE]:
            
            if level in self.active_policies:
                active_policy = self.active_policies[level]
                
                # Check if policy switch is needed
                await self._check_and_switch_policy(level, context)
                
                # Get action from active policy
                action = await active_policy.evaluate_action(context)
                hierarchical_actions[level.value] = action
                
                # Update context with higher-level decisions for lower levels
                if level == PolicyLevel.STRATEGIC:
                    context.campaign_objectives = action.get('success_criteria', context.campaign_objectives)
                elif level == PolicyLevel.OPERATIONAL:
                    context.time_constraints.update(action.get('execution_timeline', {}))
        
        return {
            'hierarchical_actions': hierarchical_actions,
            'active_policies': {level.value: policy.policy_id 
                              for level, policy in self.active_policies.items()},
            'context_analysis': await self._analyze_switching_context(context),
            'switch_recommendations': await self._generate_switch_recommendations(context)
        }
    
    async def _check_and_switch_policy(self, level: PolicyLevel, context: PolicyContext):
        """Check if policy should be switched and perform switch if needed"""
        
        if level not in self.active_policies:
            return
        
        current_policy = self.active_policies[level]
        
        # Check if current policy recommends switching
        should_switch, policy_reason = await current_policy.should_switch_policy(context)
        
        # Use neural network to make switching decision
        neural_decision = await self._neural_switching_decision(level, context, current_policy)
        
        # Combine policy and neural recommendations
        final_switch_decision = should_switch or neural_decision['should_switch']
        
        if final_switch_decision:
            # Select new policy
            new_policy = await self._select_new_policy(level, context, neural_decision)
            
            if new_policy and new_policy.policy_id != current_policy.policy_id:
                # Perform the switch
                await self._perform_policy_switch(
                    level, current_policy, new_policy, context,
                    PolicySwitchTrigger.PERFORMANCE_DEGRADATION if should_switch 
                    else PolicySwitchTrigger.HIERARCHICAL_SIGNAL,
                    policy_reason if should_switch else "Neural network recommendation"
                )
    
    async def _neural_switching_decision(self, 
                                       level: PolicyLevel,
                                       context: PolicyContext,
                                       current_policy: BasePolicy) -> Dict[str, Any]:
        """Use neural network to make switching decision"""
        
        # Convert context to features
        context_features = await self._context_to_features(context)
        policy_features = await self._policy_to_features(current_policy)
        
        # Get neural network prediction
        with torch.no_grad():
            network_output = self.switching_network(context_features, policy_features)
        
        # Extract decisions
        switch_prob = network_output['switch_probability'][0, 1].item()  # Probability of switching
        performance_pred = network_output['performance_prediction'][0].item()
        level_prefs = network_output['level_preferences'][0].cpu().numpy()
        
        return {
            'should_switch': switch_prob > 0.6,  # Threshold for switching
            'switch_confidence': switch_prob,
            'predicted_performance': performance_pred,
            'level_preferences': level_prefs,
            'recommended_level': PolicyLevel(list(PolicyLevel)[np.argmax(level_prefs)])
        }
    
    async def _select_new_policy(self, 
                               level: PolicyLevel,
                               context: PolicyContext,
                               neural_decision: Dict[str, Any]) -> Optional[BasePolicy]:
        """Select new policy for the given level"""
        
        available_policies = list(self.policies[level].values())
        current_policy = self.active_policies[level]
        
        # Remove current policy from candidates
        candidate_policies = [p for p in available_policies if p != current_policy]
        
        if not candidate_policies:
            return None
        
        # Score each candidate policy
        policy_scores = []
        
        for policy in candidate_policies:
            score = 0.0
            
            # Performance history score
            policy_stats = policy.get_performance_stats()
            score += 0.4 * policy_stats['average_performance']
            
            # Context appropriateness score
            context_score = await self._calculate_policy_context_score(policy, context)
            score += 0.3 * context_score
            
            # Neural network preference score
            if hasattr(policy, 'policy_type'):
                # This would be more sophisticated in production
                type_bonus = 0.1 if policy.policy_type in str(neural_decision) else 0.0
                score += type_bonus
            
            # Diversity bonus (prefer policies not used recently)
            recent_switches = [event for event in self.switch_history[-10:] 
                             if event.to_policy == policy.policy_id]
            diversity_bonus = 0.2 * (1.0 - len(recent_switches) / 10.0)
            score += diversity_bonus
            
            policy_scores.append((policy, score))
        
        # Select policy with highest score
        policy_scores.sort(key=lambda x: x[1], reverse=True)
        return policy_scores[0][0] if policy_scores else None
    
    async def _perform_policy_switch(self,
                                   level: PolicyLevel,
                                   from_policy: BasePolicy,
                                   to_policy: BasePolicy,
                                   context: PolicyContext,
                                   trigger: PolicySwitchTrigger,
                                   rationale: str):
        """Perform the actual policy switch"""
        
        switch_start_time = datetime.utcnow()
        
        # Deactivate old policy
        await from_policy.deactivate(context)
        
        # Activate new policy
        await to_policy.activate(context)
        
        # Update active policy
        self.active_policies[level] = to_policy
        
        # Calculate switch overhead
        switch_end_time = datetime.utcnow()
        switch_overhead = (switch_end_time - switch_start_time).total_seconds() * 1000  # ms
        
        # Create switch event
        switch_event = PolicySwitchEvent(
            event_id=f"switch_{level.value}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            timestamp=switch_start_time,
            from_policy=from_policy.policy_id,
            to_policy=to_policy.policy_id,
            from_level=level,
            to_level=level,
            trigger=trigger,
            trigger_confidence=0.8,  # Would be calculated more precisely
            switch_rationale=rationale,
            context_snapshot=context,
            expected_improvement=0.1,  # Would be predicted by model
            switch_overhead=switch_overhead
        )
        
        # Record switch
        self.switch_history.append(switch_event)
        if len(self.switch_history) > self.max_history_size:
            self.switch_history = self.switch_history[-self.max_history_size:]
        
        # Update metrics
        self.switching_metrics['total_switches'] += 1
        self.switching_metrics['switches_by_trigger'][trigger.value] += 1
        self.switching_metrics['switches_by_level'][level.value] += 1
        self.switching_metrics['switching_overhead_ms'] = (
            (self.switching_metrics['switching_overhead_ms'] * 
             (self.switching_metrics['total_switches'] - 1) + switch_overhead) /
            self.switching_metrics['total_switches']
        )
        
        self.logger.info(f"Policy switch completed: {from_policy.policy_id} -> {to_policy.policy_id} "
                        f"at level {level.value} (trigger: {trigger.value})")
    
    async def _context_to_features(self, context: PolicyContext) -> torch.Tensor:
        """Convert policy context to feature tensor"""
        
        features = []
        
        # Environmental features
        env_complexity = len(context.target_environment) / 20.0  # Normalize
        features.append(env_complexity)
        
        # Resource features
        features.extend([
            context.system_resources.get('cpu_available', 0.5),
            context.system_resources.get('memory_available', 0.5),
            context.system_resources.get('network_bandwidth', 0.5)
        ])
        
        # Performance features
        features.extend([
            context.recent_performance.get('success_rate', 0.5),
            context.resource_efficiency,
            context.uncertainty_level,
            context.novelty_score,
            context.complexity_estimate
        ])
        
        # Temporal features
        features.extend([
            context.time_since_last_switch / 3600.0,  # Normalize to hours
            len(context.policy_history) / 10.0,  # Normalize
            context.risk_tolerance
        ])
        
        # Pad to required dimension
        target_dim = 256
        while len(features) < target_dim:
            features.append(0.0)
        
        return torch.tensor(features[:target_dim], dtype=torch.float32, device=self.device).unsqueeze(0)
    
    async def _policy_to_features(self, policy: BasePolicy) -> torch.Tensor:
        """Convert policy to feature tensor"""
        
        features = []
        
        # Policy characteristics
        level_encoding = {
            PolicyLevel.STRATEGIC: [1, 0, 0, 0],
            PolicyLevel.OPERATIONAL: [0, 1, 0, 0],
            PolicyLevel.TACTICAL: [0, 0, 1, 0],
            PolicyLevel.REACTIVE: [0, 0, 0, 1]
        }
        features.extend(level_encoding[policy.policy_level])
        
        # Performance statistics
        stats = policy.get_performance_stats()
        features.extend([
            stats['average_performance'],
            stats['performance_variance'],
            stats['performance_trend'],
            stats['activation_count'] / 100.0,  # Normalize
            stats['average_execution_time'] / 3600.0  # Normalize to hours
        ])
        
        # Policy-specific features
        if hasattr(policy, 'strategy_type'):
            # Strategic policy features
            strategy_encoding = {
                'balanced': [1, 0, 0],
                'aggressive': [0, 1, 0],
                'stealth': [0, 0, 1]
            }
            features.extend(strategy_encoding.get(policy.strategy_type, [0, 0, 0]))
        else:
            features.extend([0, 0, 0])
        
        # Pad to required dimension
        target_dim = 64
        while len(features) < target_dim:
            features.append(0.0)
        
        return torch.tensor(features[:target_dim], dtype=torch.float32, device=self.device).unsqueeze(0)
    
    async def _calculate_policy_context_score(self, 
                                            policy: BasePolicy,
                                            context: PolicyContext) -> float:
        """Calculate how well a policy fits the current context"""
        
        score = 0.5  # Base score
        
        # Strategic level context matching
        if policy.policy_level == PolicyLevel.STRATEGIC:
            if hasattr(policy, 'strategy_type'):
                if context.risk_tolerance > 0.7 and policy.strategy_type == 'aggressive':
                    score += 0.3
                elif context.risk_tolerance < 0.3 and policy.strategy_type == 'stealth':
                    score += 0.3
                elif 0.3 <= context.risk_tolerance <= 0.7 and policy.strategy_type == 'balanced':
                    score += 0.3
        
        # Resource availability matching
        resource_availability = np.mean(list(context.system_resources.values()))
        if resource_availability < 0.3:
            # Prefer lightweight policies under resource constraints
            if policy.policy_level in [PolicyLevel.REACTIVE, PolicyLevel.TACTICAL]:
                score += 0.2
        
        # Complexity matching
        if context.complexity_estimate > 0.7:
            # Prefer more sophisticated policies for complex environments
            if policy.policy_level in [PolicyLevel.STRATEGIC, PolicyLevel.OPERATIONAL]:
                score += 0.2
        
        # Uncertainty matching
        if context.uncertainty_level > 0.6:
            # Prefer adaptive policies under high uncertainty
            if hasattr(policy, 'tactic_type') and 'adaptive' in policy.tactic_type:
                score += 0.2
        
        return min(1.0, max(0.0, score))
    
    async def _analyze_switching_context(self, context: PolicyContext) -> Dict[str, Any]:
        """Analyze context for switching patterns and insights"""
        
        analysis = {
            'context_stability': self._calculate_context_stability(context),
            'resource_pressure': self._calculate_resource_pressure(context),
            'performance_trend': self._calculate_performance_trend(context),
            'switch_frequency': self._calculate_switch_frequency(),
            'level_effectiveness': self._calculate_level_effectiveness()
        }
        
        return analysis
    
    async def _generate_switch_recommendations(self, context: PolicyContext) -> List[Dict[str, Any]]:
        """Generate recommendations for policy switching"""
        
        recommendations = []
        
        # Analyze current performance
        for level, policy in self.active_policies.items():
            policy_stats = policy.get_performance_stats()
            
            if policy_stats['average_performance'] < 0.4:
                recommendations.append({
                    'type': 'performance_improvement',
                    'level': level.value,
                    'current_policy': policy.policy_id,
                    'recommendation': f'Consider switching {level.value} policy due to low performance',
                    'priority': 'high',
                    'expected_benefit': 'performance_improvement'
                })
        
        # Analyze resource constraints
        if context.system_resources.get('cpu_available', 1.0) < 0.2:
            recommendations.append({
                'type': 'resource_optimization',
                'recommendation': 'Switch to resource-efficient policies due to CPU constraints',
                'priority': 'high',
                'affected_levels': [level.value for level in PolicyLevel]
            })
        
        # Analyze context changes
        if context.novelty_score > 0.8:
            recommendations.append({
                'type': 'context_adaptation',
                'recommendation': 'Environment has changed significantly, review all policy levels',
                'priority': 'medium',
                'novelty_score': context.novelty_score
            })
        
        return recommendations
    
    def _calculate_context_stability(self, context: PolicyContext) -> float:
        """Calculate how stable the context has been"""
        
        # Simple stability measure based on recent switches
        recent_switches = [event for event in self.switch_history[-10:]]
        switch_rate = len(recent_switches) / 10.0
        
        stability = max(0.0, 1.0 - switch_rate)
        return stability
    
    def _calculate_resource_pressure(self, context: PolicyContext) -> float:
        """Calculate resource pressure level"""
        
        resource_values = list(context.system_resources.values())
        if not resource_values:
            return 0.5
        
        # Pressure is inverse of available resources
        avg_availability = np.mean(resource_values)
        pressure = 1.0 - avg_availability
        
        return pressure
    
    def _calculate_performance_trend(self, context: PolicyContext) -> float:
        """Calculate overall performance trend across levels"""
        
        all_performance_values = []
        
        for level, performance_history in self.level_performance.items():
            if performance_history:
                all_performance_values.extend(list(performance_history))
        
        if len(all_performance_values) < 3:
            return 0.0
        
        # Calculate trend over recent performance
        recent_values = all_performance_values[-20:]  # Last 20 measurements
        x = np.arange(len(recent_values))
        
        correlation = np.corrcoef(x, recent_values)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def _calculate_switch_frequency(self) -> float:
        """Calculate recent switch frequency"""
        
        if not self.switch_history:
            return 0.0
        
        # Count switches in last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_switches = [event for event in self.switch_history 
                          if event.timestamp >= one_hour_ago]
        
        return len(recent_switches)
    
    def _calculate_level_effectiveness(self) -> Dict[str, float]:
        """Calculate effectiveness of each policy level"""
        
        effectiveness = {}
        
        for level, performance_history in self.level_performance.items():
            if performance_history:
                effectiveness[level.value] = np.mean(list(performance_history))
            else:
                effectiveness[level.value] = 0.5
        
        return effectiveness
    
    async def get_switching_system_status(self) -> Dict[str, Any]:
        """Get current status of the switching system"""
        
        return {
            'system_active': True,
            'active_policies': {level.value: policy.policy_id 
                              for level, policy in self.active_policies.items()},
            'registered_policies': {
                level.value: list(policies.keys()) 
                for level, policies in self.policies.items()
            },
            'switching_metrics': dict(self.switching_metrics),
            'recent_switches': len([event for event in self.switch_history 
                                  if (datetime.utcnow() - event.timestamp).total_seconds() < 3600]),
            'level_performance': {
                level.value: {
                    'average': np.mean(list(perf_history)) if perf_history else 0.0,
                    'count': len(perf_history)
                }
                for level, perf_history in self.level_performance.items()
            },
            'switch_history_size': len(self.switch_history)
        }


if __name__ == "__main__":
    async def main():
        # Example usage
        switching_system = HierarchicalPolicySwitchingSystem()
        
        # Create example context
        context = PolicyContext(
            target_environment={'complexity': 0.6, 'threats': ['ddos', 'malware']},
            system_resources={'cpu_available': 0.7, 'memory_available': 0.8},
            time_constraints={'max_execution_time': 3600},
            recent_performance={'success_rate': 0.6, 'efficiency': 0.7},
            success_rate=0.6,
            resource_efficiency=0.7,
            campaign_objectives=['stealth', 'persistence'],
            current_phase='exploitation',
            risk_tolerance=0.5,
            timestamp=datetime.utcnow(),
            time_since_last_switch=600,
            policy_history=['balanced_strategy', 'standard_ops'],
            uncertainty_level=0.4,
            novelty_score=0.3,
            complexity_estimate=0.6
        )
        
        # Evaluate hierarchical actions
        actions = await switching_system.evaluate_hierarchical_action(context)
        print(f"Hierarchical actions: {json.dumps(actions, indent=2, default=str)}")
        
        # Get system status
        status = await switching_system.get_switching_system_status()
        print(f"\nSwitching system status: {json.dumps(status, indent=2)}")
    
    asyncio.run(main())