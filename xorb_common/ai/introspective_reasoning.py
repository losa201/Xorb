#!/usr/bin/env python3
"""
Introspective Agent Reasoning System for Xorb 2.0

This module implements introspective reasoning capabilities that enable agents 
to reflect on their own actions, analyze decision patterns, and continuously
improve their strategies through self-examination and meta-cognitive processes.
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import json
import pickle
from pathlib import Path


class ReasoningLevel(Enum):
    """Levels of introspective reasoning"""
    REACTIVE = "reactive"           # Direct action-response
    TACTICAL = "tactical"           # Short-term planning
    STRATEGIC = "strategic"         # Long-term goal optimization
    META_COGNITIVE = "meta_cognitive"  # Reasoning about reasoning


class DecisionOutcome(Enum):
    """Outcomes of agent decisions"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ActionReflection:
    """Reflection on a single agent action"""
    action_id: str
    agent_type: str
    action_taken: str
    context_snapshot: Dict[str, Any]
    reasoning_level: ReasoningLevel
    decision_factors: Dict[str, float]
    
    # Outcomes and feedback
    outcome: DecisionOutcome
    success_metrics: Dict[str, float]
    execution_time: float
    resource_usage: Dict[str, float]
    
    # Introspective analysis
    confidence_at_decision: float
    confidence_post_outcome: float
    surprise_factor: float  # How unexpected was the outcome
    learning_opportunity_score: float
    
    # Temporal context
    timestamp: datetime
    campaign_phase: str
    system_state_hash: str
    
    # Meta-reasoning
    reasoning_process: List[str]  # Steps in decision process
    alternative_actions_considered: List[str]
    counterfactual_analysis: Optional[Dict[str, Any]] = None


@dataclass
class ReasoningPattern:
    """Identified pattern in agent reasoning"""
    pattern_id: str
    pattern_type: str  # 'success_pattern', 'failure_pattern', 'bias_pattern'
    description: str
    frequency: int
    contexts: List[str]  # When this pattern appears
    
    # Pattern characteristics
    decision_sequence: List[str]
    typical_outcomes: Dict[DecisionOutcome, float]
    resource_efficiency: float
    adaptability_score: float
    
    # Learning insights
    improvement_suggestions: List[str]
    risk_factors: List[str]
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Evolution tracking
    first_observed: datetime
    last_observed: datetime
    trend_direction: str  # 'improving', 'declining', 'stable'


class SelfReflectionEncoder(nn.Module):
    """Neural encoder for self-reflection on agent actions"""
    
    def __init__(self, 
                 context_dim: int = 128,
                 action_dim: int = 64,
                 reflection_dim: int = 256,
                 num_heads: int = 8):
        super().__init__()
        
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, reflection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(reflection_dim, reflection_dim)
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, reflection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(reflection_dim, reflection_dim)
        )
        
        # Multi-head attention for context-action interaction
        self.attention = nn.MultiheadAttention(
            reflection_dim, num_heads, batch_first=True
        )
        
        # Outcome prediction head
        self.outcome_predictor = nn.Sequential(
            nn.Linear(reflection_dim, reflection_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(reflection_dim // 2, len(DecisionOutcome)),
            nn.Softmax(dim=-1)
        )
        
        # Confidence estimation head
        self.confidence_estimator = nn.Sequential(
            nn.Linear(reflection_dim, reflection_dim // 4),
            nn.ReLU(),
            nn.Linear(reflection_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Surprise detection head
        self.surprise_detector = nn.Sequential(
            nn.Linear(reflection_dim * 2, reflection_dim),  # current + expected
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(reflection_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                context_features: torch.Tensor,
                action_features: torch.Tensor,
                expected_outcome: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Encode context and action
        context_encoded = self.context_encoder(context_features)
        action_encoded = self.action_encoder(action_features)
        
        # Apply attention between context and action
        attended_features, attention_weights = self.attention(
            query=action_encoded.unsqueeze(1),
            key=context_encoded.unsqueeze(1),
            value=context_encoded.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)
        
        # Generate predictions
        outcome_probs = self.outcome_predictor(attended_features)
        confidence = self.confidence_estimator(attended_features)
        
        results = {
            'outcome_probabilities': outcome_probs,
            'confidence': confidence,
            'attention_weights': attention_weights,
            'encoded_representation': attended_features
        }
        
        # Calculate surprise if expected outcome is provided
        if expected_outcome is not None:
            combined_features = torch.cat([attended_features, expected_outcome], dim=-1)
            surprise = self.surprise_detector(combined_features)
            results['surprise_score'] = surprise
        
        return results


class MetaCognitiveAnalyzer:
    """Analyzer for meta-cognitive reasoning patterns"""
    
    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self.reflection_history: deque = deque(maxlen=max_history_size)
        self.reasoning_patterns: Dict[str, ReasoningPattern] = {}
        self.logger = logging.getLogger(__name__)
        
        # Pattern detection parameters
        self.min_pattern_frequency = 5
        self.pattern_similarity_threshold = 0.8
        self.success_rate_threshold = 0.7
    
    def analyze_reasoning_patterns(self, 
                                 reflections: List[ActionReflection]) -> List[ReasoningPattern]:
        """Analyze reflections to identify reasoning patterns"""
        
        patterns = []
        
        # Group reflections by agent type and reasoning level
        grouped_reflections = defaultdict(list)
        for reflection in reflections:
            key = f"{reflection.agent_type}_{reflection.reasoning_level.value}"
            grouped_reflections[key].append(reflection)
        
        # Analyze each group for patterns
        for group_key, group_reflections in grouped_reflections.items():
            if len(group_reflections) < self.min_pattern_frequency:
                continue
            
            # Success patterns
            success_patterns = self._identify_success_patterns(group_reflections)
            patterns.extend(success_patterns)
            
            # Failure patterns
            failure_patterns = self._identify_failure_patterns(group_reflections)
            patterns.extend(failure_patterns)
            
            # Bias patterns
            bias_patterns = self._identify_bias_patterns(group_reflections)
            patterns.extend(bias_patterns)
        
        return patterns
    
    def _identify_success_patterns(self, 
                                 reflections: List[ActionReflection]) -> List[ReasoningPattern]:
        """Identify patterns that lead to successful outcomes"""
        
        successful_reflections = [
            r for r in reflections 
            if r.outcome == DecisionOutcome.SUCCESS
        ]
        
        if len(successful_reflections) < self.min_pattern_frequency:
            return []
        
        # Cluster successful reflections by decision factors
        pattern_clusters = self._cluster_by_decision_factors(successful_reflections)
        
        patterns = []
        for cluster_id, cluster_reflections in pattern_clusters.items():
            if len(cluster_reflections) < self.min_pattern_frequency:
                continue
            
            # Calculate pattern characteristics
            success_rate = len([r for r in cluster_reflections 
                              if r.outcome == DecisionOutcome.SUCCESS]) / len(cluster_reflections)
            
            if success_rate >= self.success_rate_threshold:
                pattern = ReasoningPattern(
                    pattern_id=f"success_{cluster_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    pattern_type="success_pattern",
                    description=f"Successful reasoning pattern for {cluster_reflections[0].agent_type}",
                    frequency=len(cluster_reflections),
                    contexts=[r.campaign_phase for r in cluster_reflections],
                    decision_sequence=self._extract_decision_sequence(cluster_reflections),
                    typical_outcomes=self._calculate_outcome_distribution(cluster_reflections),
                    resource_efficiency=np.mean([
                        sum(r.resource_usage.values()) for r in cluster_reflections
                    ]),
                    adaptability_score=self._calculate_adaptability_score(cluster_reflections),
                    improvement_suggestions=self._generate_improvement_suggestions(cluster_reflections),
                    risk_factors=self._identify_risk_factors(cluster_reflections),
                    confidence_intervals=self._calculate_confidence_intervals(cluster_reflections),
                    first_observed=min(r.timestamp for r in cluster_reflections),
                    last_observed=max(r.timestamp for r in cluster_reflections),
                    trend_direction=self._analyze_trend(cluster_reflections)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _identify_failure_patterns(self, 
                                 reflections: List[ActionReflection]) -> List[ReasoningPattern]:
        """Identify patterns that lead to failures"""
        
        failed_reflections = [
            r for r in reflections 
            if r.outcome in [DecisionOutcome.FAILURE, DecisionOutcome.ERROR, DecisionOutcome.TIMEOUT]
        ]
        
        if len(failed_reflections) < self.min_pattern_frequency:
            return []
        
        # Similar clustering approach but for failures
        pattern_clusters = self._cluster_by_decision_factors(failed_reflections)
        
        patterns = []
        for cluster_id, cluster_reflections in pattern_clusters.items():
            if len(cluster_reflections) < self.min_pattern_frequency:
                continue
            
            pattern = ReasoningPattern(
                pattern_id=f"failure_{cluster_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                pattern_type="failure_pattern",
                description=f"Failure reasoning pattern for {cluster_reflections[0].agent_type}",
                frequency=len(cluster_reflections),
                contexts=[r.campaign_phase for r in cluster_reflections],
                decision_sequence=self._extract_decision_sequence(cluster_reflections),
                typical_outcomes=self._calculate_outcome_distribution(cluster_reflections),
                resource_efficiency=np.mean([
                    sum(r.resource_usage.values()) for r in cluster_reflections
                ]),
                adaptability_score=self._calculate_adaptability_score(cluster_reflections),
                improvement_suggestions=self._generate_failure_mitigations(cluster_reflections),
                risk_factors=self._identify_common_failure_factors(cluster_reflections),
                confidence_intervals=self._calculate_confidence_intervals(cluster_reflections),
                first_observed=min(r.timestamp for r in cluster_reflections),
                last_observed=max(r.timestamp for r in cluster_reflections),
                trend_direction=self._analyze_trend(cluster_reflections)
            )
            patterns.append(pattern)
        
        return patterns
    
    def _identify_bias_patterns(self, 
                              reflections: List[ActionReflection]) -> List[ReasoningPattern]:
        """Identify cognitive bias patterns in reasoning"""
        
        bias_patterns = []
        
        # Overconfidence bias detection
        overconfident_reflections = [
            r for r in reflections
            if (r.confidence_at_decision - r.confidence_post_outcome) > 0.3
        ]
        
        if len(overconfident_reflections) >= self.min_pattern_frequency:
            pattern = ReasoningPattern(
                pattern_id=f"overconfidence_bias_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                pattern_type="bias_pattern",
                description="Overconfidence bias in decision making",
                frequency=len(overconfident_reflections),
                contexts=[r.campaign_phase for r in overconfident_reflections],
                decision_sequence=["high_confidence_decision", "poor_outcome"],
                typical_outcomes=self._calculate_outcome_distribution(overconfident_reflections),
                resource_efficiency=0.6,  # Usually poor due to overcommitment
                adaptability_score=0.4,   # Low adaptability due to bias
                improvement_suggestions=[
                    "Implement confidence calibration training",
                    "Use ensemble methods for decision confidence",
                    "Add uncertainty quantification to decision process"
                ],
                risk_factors=[
                    "High stakes decisions",
                    "Familiar contexts",
                    "Time pressure"
                ],
                confidence_intervals=self._calculate_confidence_intervals(overconfident_reflections),
                first_observed=min(r.timestamp for r in overconfident_reflections),
                last_observed=max(r.timestamp for r in overconfident_reflections),
                trend_direction=self._analyze_trend(overconfident_reflections)
            )
            bias_patterns.append(pattern)
        
        # Anchoring bias detection (over-reliance on first information)
        # Would require analysis of decision_factors evolution over time
        
        return bias_patterns
    
    def _cluster_by_decision_factors(self, 
                                   reflections: List[ActionReflection]) -> Dict[str, List[ActionReflection]]:
        """Cluster reflections by similar decision factors"""
        
        clusters = defaultdict(list)
        
        for reflection in reflections:
            # Create a signature based on top decision factors
            top_factors = sorted(
                reflection.decision_factors.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]  # Top 3 factors
            
            cluster_key = "_".join([factor[0] for factor in top_factors])
            clusters[cluster_key].append(reflection)
        
        return dict(clusters)
    
    def _extract_decision_sequence(self, reflections: List[ActionReflection]) -> List[str]:
        """Extract common decision sequence from reflections"""
        
        # Find most common reasoning process steps
        all_steps = []
        for reflection in reflections:
            all_steps.extend(reflection.reasoning_process)
        
        # Count step frequencies
        step_counts = defaultdict(int)
        for step in all_steps:
            step_counts[step] += 1
        
        # Return most common steps in order
        common_steps = sorted(
            step_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 steps
        
        return [step[0] for step in common_steps]
    
    def _calculate_outcome_distribution(self, 
                                      reflections: List[ActionReflection]) -> Dict[DecisionOutcome, float]:
        """Calculate distribution of outcomes"""
        
        outcome_counts = defaultdict(int)
        for reflection in reflections:
            outcome_counts[reflection.outcome] += 1
        
        total = len(reflections)
        return {outcome: count / total for outcome, count in outcome_counts.items()}
    
    def _calculate_adaptability_score(self, reflections: List[ActionReflection]) -> float:
        """Calculate how well the agent adapts to different contexts"""
        
        if len(reflections) < 2:
            return 0.5
        
        # Measure variation in decision factors across different contexts
        context_diversity = len(set(r.campaign_phase for r in reflections))
        decision_consistency = self._measure_decision_consistency(reflections)
        
        # High adaptability = high context diversity with appropriate decision variation
        adaptability = (context_diversity / 10.0) * (1.0 - decision_consistency)
        return min(1.0, max(0.0, adaptability))
    
    def _measure_decision_consistency(self, reflections: List[ActionReflection]) -> float:
        """Measure consistency in decision making"""
        
        if len(reflections) < 2:
            return 1.0
        
        # Calculate variance in decision factors
        all_factors = []
        for reflection in reflections:
            factor_vector = [reflection.decision_factors.get(key, 0.0) 
                           for key in sorted(reflection.decision_factors.keys())]
            all_factors.append(factor_vector)
        
        if not all_factors or len(all_factors[0]) == 0:
            return 1.0
        
        factor_array = np.array(all_factors)
        consistency = 1.0 - np.mean(np.var(factor_array, axis=0))
        return max(0.0, min(1.0, consistency))
    
    def _generate_improvement_suggestions(self, 
                                        reflections: List[ActionReflection]) -> List[str]:
        """Generate suggestions for improving successful patterns"""
        
        suggestions = []
        
        # Analyze resource efficiency
        avg_resource_usage = np.mean([sum(r.resource_usage.values()) for r in reflections])
        if avg_resource_usage > 0.8:
            suggestions.append("Optimize resource usage to maintain success with lower overhead")
        
        # Analyze execution time
        avg_execution_time = np.mean([r.execution_time for r in reflections])
        if avg_execution_time > 300:  # 5 minutes
            suggestions.append("Investigate opportunities to reduce execution time")
        
        # Analyze confidence calibration
        confidence_errors = [abs(r.confidence_at_decision - r.confidence_post_outcome) 
                           for r in reflections]
        avg_confidence_error = np.mean(confidence_errors)
        if avg_confidence_error > 0.2:
            suggestions.append("Improve confidence calibration accuracy")
        
        return suggestions
    
    def _generate_failure_mitigations(self, 
                                    reflections: List[ActionReflection]) -> List[str]:
        """Generate suggestions for mitigating failure patterns"""
        
        mitigations = []
        
        # Common failure factors
        common_factors = defaultdict(int)
        for reflection in reflections:
            for factor, weight in reflection.decision_factors.items():
                if weight > 0.7:  # High weight factors in failures
                    common_factors[factor] += 1
        
        if common_factors:
            top_factor = max(common_factors.items(), key=lambda x: x[1])[0]
            mitigations.append(f"Review decision weighting for factor: {top_factor}")
        
        # Timeout failures
        timeout_count = len([r for r in reflections if r.outcome == DecisionOutcome.TIMEOUT])
        if timeout_count > len(reflections) * 0.3:
            mitigations.append("Implement better timeout handling and early termination conditions")
        
        # Error failures
        error_count = len([r for r in reflections if r.outcome == DecisionOutcome.ERROR])
        if error_count > len(reflections) * 0.2:
            mitigations.append("Add more robust error handling and validation")
        
        return mitigations
    
    def _identify_risk_factors(self, reflections: List[ActionReflection]) -> List[str]:
        """Identify risk factors from successful patterns"""
        
        risk_factors = []
        
        # High resource usage
        high_resource_reflections = [r for r in reflections 
                                   if sum(r.resource_usage.values()) > 0.8]
        if len(high_resource_reflections) > len(reflections) * 0.5:
            risk_factors.append("High resource usage - may not scale well")
        
        # Low surprise factor (may indicate overfitting to familiar scenarios)
        low_surprise_reflections = [r for r in reflections if r.surprise_factor < 0.1]
        if len(low_surprise_reflections) > len(reflections) * 0.8:
            risk_factors.append("Low adaptability to novel scenarios")
        
        return risk_factors
    
    def _identify_common_failure_factors(self, 
                                       reflections: List[ActionReflection]) -> List[str]:
        """Identify common factors leading to failures"""
        
        failure_factors = []
        
        # Analyze decision factors in failed cases
        failed_decision_factors = defaultdict(list)
        for reflection in reflections:
            for factor, weight in reflection.decision_factors.items():
                failed_decision_factors[factor].append(weight)
        
        # Find factors with consistently high weights in failures
        for factor, weights in failed_decision_factors.items():
            avg_weight = np.mean(weights)
            if avg_weight > 0.7:
                failure_factors.append(f"Over-reliance on {factor}")
        
        return failure_factors
    
    def _calculate_confidence_intervals(self, 
                                      reflections: List[ActionReflection]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for key metrics"""
        
        intervals = {}
        
        # Success rate confidence interval
        success_count = len([r for r in reflections if r.outcome == DecisionOutcome.SUCCESS])
        success_rate = success_count / len(reflections)
        success_ci = self._calculate_binomial_ci(success_count, len(reflections))
        intervals['success_rate'] = success_ci
        
        # Execution time confidence interval
        execution_times = [r.execution_time for r in reflections]
        exec_time_ci = self._calculate_normal_ci(execution_times)
        intervals['execution_time'] = exec_time_ci
        
        return intervals
    
    def _calculate_binomial_ci(self, successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate binomial confidence interval"""
        if trials == 0:
            return (0.0, 0.0)
        
        p = successes / trials
        z = 1.96  # 95% confidence
        margin = z * np.sqrt(p * (1 - p) / trials)
        
        return (max(0.0, p - margin), min(1.0, p + margin))
    
    def _calculate_normal_ci(self, values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate normal confidence interval"""
        if not values:
            return (0.0, 0.0)
        
        mean = np.mean(values)
        std = np.std(values)
        n = len(values)
        
        z = 1.96  # 95% confidence
        margin = z * std / np.sqrt(n)
        
        return (mean - margin, mean + margin)
    
    def _analyze_trend(self, reflections: List[ActionReflection]) -> str:
        """Analyze trend in pattern performance over time"""
        
        if len(reflections) < 3:
            return "stable"
        
        # Sort by timestamp
        sorted_reflections = sorted(reflections, key=lambda r: r.timestamp)
        
        # Split into early and late periods
        mid_point = len(sorted_reflections) // 2
        early_period = sorted_reflections[:mid_point]
        late_period = sorted_reflections[mid_point:]
        
        # Compare success rates
        early_success_rate = len([r for r in early_period 
                                if r.outcome == DecisionOutcome.SUCCESS]) / len(early_period)
        late_success_rate = len([r for r in late_period 
                               if r.outcome == DecisionOutcome.SUCCESS]) / len(late_period)
        
        difference = late_success_rate - early_success_rate
        
        if difference > 0.1:
            return "improving"
        elif difference < -0.1:
            return "declining"
        else:
            return "stable"


class IntrospectiveReasoningSystem:
    """Main introspective reasoning system for agents"""
    
    def __init__(self, 
                 model_config: Dict[str, Any] = None,
                 device: str = 'auto'):
        
        self.logger = logging.getLogger(__name__)
        
        # Device configuration
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Model configuration
        default_config = {
            'context_dim': 128,
            'action_dim': 64,
            'reflection_dim': 256,
            'num_heads': 8
        }
        self.model_config = {**default_config, **(model_config or {})}
        
        # Initialize neural reflection model
        self.reflection_encoder = SelfReflectionEncoder(**self.model_config).to(self.device)
        
        # Meta-cognitive analyzer
        self.meta_analyzer = MetaCognitiveAnalyzer()
        
        # Reflection storage
        self.reflection_buffer: deque = deque(maxlen=50000)
        self.active_patterns: Dict[str, ReasoningPattern] = {}
        
        # Performance metrics
        self.introspection_metrics = {
            'total_reflections': 0,
            'patterns_identified': 0,
            'improvement_actions_taken': 0,
            'meta_cognitive_insights': 0,
            'reasoning_accuracy': 0.0,
            'pattern_prediction_accuracy': 0.0
        }
        
        self.logger.info(f"Introspective reasoning system initialized on {self.device}")
    
    async def reflect_on_action(self,
                              agent_id: str,
                              agent_type: str,
                              action_taken: str,
                              context: Dict[str, Any],
                              outcome: DecisionOutcome,
                              success_metrics: Dict[str, float],
                              execution_details: Dict[str, Any]) -> ActionReflection:
        """Create a reflection on an agent action"""
        
        # Generate unique action ID
        action_id = f"{agent_id}_{action_taken}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Extract reasoning information
        reasoning_level = self._determine_reasoning_level(action_taken, context)
        decision_factors = self._extract_decision_factors(context, execution_details)
        reasoning_process = self._reconstruct_reasoning_process(execution_details)
        
        # Neural prediction for confidence and surprise
        context_features = self._context_to_features(context)
        action_features = self._action_to_features(action_taken, agent_type)
        
        with torch.no_grad():
            neural_output = self.reflection_encoder(context_features, action_features)
            predicted_confidence = neural_output['confidence'].item()
            
        # Calculate surprise factor based on prediction vs reality
        surprise_factor = self._calculate_surprise_factor(
            predicted_confidence, outcome, success_metrics
        )
        
        # Create reflection
        reflection = ActionReflection(
            action_id=action_id,
            agent_type=agent_type,
            action_taken=action_taken,
            context_snapshot=context.copy(),
            reasoning_level=reasoning_level,
            decision_factors=decision_factors,
            outcome=outcome,
            success_metrics=success_metrics,
            execution_time=execution_details.get('execution_time', 0.0),
            resource_usage=execution_details.get('resource_usage', {}),
            confidence_at_decision=execution_details.get('initial_confidence', 0.5),
            confidence_post_outcome=self._calculate_post_outcome_confidence(outcome, success_metrics),
            surprise_factor=surprise_factor,
            learning_opportunity_score=self._calculate_learning_opportunity(surprise_factor, outcome),
            timestamp=datetime.utcnow(),
            campaign_phase=context.get('campaign_phase', 'unknown'),
            system_state_hash=str(hash(str(sorted(context.items())))),
            reasoning_process=reasoning_process,
            alternative_actions_considered=execution_details.get('alternatives_considered', [])
        )
        
        # Store reflection
        self.reflection_buffer.append(reflection)
        self.introspection_metrics['total_reflections'] += 1
        
        # Trigger pattern analysis if enough new reflections
        if len(self.reflection_buffer) % 100 == 0:
            await self._update_reasoning_patterns()
        
        self.logger.debug(f"Created reflection for action {action_id}")
        return reflection
    
    async def analyze_reasoning_evolution(self, 
                                        agent_id: str,
                                        time_window_hours: int = 24) -> Dict[str, Any]:
        """Analyze how an agent's reasoning has evolved over time"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        recent_reflections = [
            r for r in self.reflection_buffer
            if r.timestamp >= cutoff_time and agent_id in r.action_id
        ]
        
        if len(recent_reflections) < 5:
            return {'status': 'insufficient_data', 'reflections_count': len(recent_reflections)}
        
        # Sort by timestamp
        sorted_reflections = sorted(recent_reflections, key=lambda r: r.timestamp)
        
        # Analyze evolution trends
        evolution_analysis = {
            'time_window_hours': time_window_hours,
            'total_reflections': len(sorted_reflections),
            'reasoning_level_evolution': self._analyze_reasoning_level_evolution(sorted_reflections),
            'confidence_calibration_trend': self._analyze_confidence_calibration(sorted_reflections),
            'decision_factor_shifts': self._analyze_decision_factor_shifts(sorted_reflections),
            'learning_trajectory': self._analyze_learning_trajectory(sorted_reflections),
            'surprise_adaptation': self._analyze_surprise_adaptation(sorted_reflections),
            'performance_trend': self._analyze_performance_trend(sorted_reflections),
            'meta_insights': self._generate_meta_insights(sorted_reflections)
        }
        
        return evolution_analysis
    
    async def generate_reasoning_recommendations(self, 
                                               agent_id: str,
                                               context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations for improving agent reasoning"""
        
        recommendations = []
        
        # Get recent reflections for this agent
        agent_reflections = [
            r for r in self.reflection_buffer
            if agent_id in r.action_id
        ][-50:]  # Last 50 reflections
        
        if len(agent_reflections) < 10:
            return [{'type': 'data_collection', 'message': 'Need more reflection data'}]
        
        # Analyze patterns
        patterns = self.meta_analyzer.analyze_reasoning_patterns(agent_reflections)
        
        # Generate recommendations based on patterns
        for pattern in patterns:
            if pattern.pattern_type == 'failure_pattern':
                recommendations.extend(self._generate_failure_recommendations(pattern))
            elif pattern.pattern_type == 'bias_pattern':
                recommendations.extend(self._generate_bias_recommendations(pattern))
            elif pattern.pattern_type == 'success_pattern':
                recommendations.extend(self._generate_optimization_recommendations(pattern))
        
        # Context-specific recommendations
        context_recommendations = self._generate_context_recommendations(agent_reflections, context)
        recommendations.extend(context_recommendations)
        
        # Prioritize recommendations
        prioritized_recommendations = sorted(
            recommendations, 
            key=lambda r: r.get('priority_score', 0.5), 
            reverse=True
        )
        
        return prioritized_recommendations[:10]  # Top 10 recommendations
    
    def _determine_reasoning_level(self, action: str, context: Dict[str, Any]) -> ReasoningLevel:
        """Determine the level of reasoning used for an action"""
        
        # Simple heuristics - in production, this could use ML classification
        if 'immediate' in action or 'react' in action:
            return ReasoningLevel.REACTIVE
        elif 'plan' in action or 'strategy' in action:
            if context.get('time_horizon', 0) > 3600:  # > 1 hour
                return ReasoningLevel.STRATEGIC
            else:
                return ReasoningLevel.TACTICAL
        elif 'analyze' in action or 'optimize' in action:
            return ReasoningLevel.META_COGNITIVE
        else:
            return ReasoningLevel.TACTICAL  # Default
    
    def _extract_decision_factors(self, 
                                context: Dict[str, Any], 
                                execution_details: Dict[str, Any]) -> Dict[str, float]:
        """Extract decision factors from context and execution details"""
        
        factors = {}
        
        # Extract from context
        factors['target_complexity'] = context.get('target_complexity', 0.5)
        factors['resource_availability'] = context.get('resource_availability', 0.5)
        factors['time_pressure'] = context.get('time_pressure', 0.5)
        factors['success_probability'] = context.get('success_probability', 0.5)
        
        # Extract from execution details
        factors['confidence_level'] = execution_details.get('initial_confidence', 0.5)
        factors['risk_tolerance'] = execution_details.get('risk_tolerance', 0.5)
        factors['novelty_preference'] = execution_details.get('novelty_preference', 0.5)
        
        return factors
    
    def _reconstruct_reasoning_process(self, execution_details: Dict[str, Any]) -> List[str]:
        """Reconstruct the reasoning process steps"""
        
        # Extract from execution details or use defaults
        process_steps = execution_details.get('reasoning_steps', [
            'context_analysis',
            'option_generation',
            'risk_assessment',
            'resource_evaluation',
            'decision_selection',
            'execution_planning'
        ])
        
        return process_steps
    
    def _context_to_features(self, context: Dict[str, Any]) -> torch.Tensor:
        """Convert context to feature tensor"""
        
        # Simple feature extraction - in production, use more sophisticated encoding
        features = [
            context.get('target_complexity', 0.5),
            context.get('resource_availability', 0.5),
            context.get('time_pressure', 0.5),
            context.get('success_probability', 0.5),
            len(context.get('available_agents', [])) / 10.0,
            context.get('campaign_progress', 0.5)
        ]
        
        # Pad to required dimension
        while len(features) < self.model_config['context_dim']:
            features.append(0.0)
        
        return torch.tensor(features[:self.model_config['context_dim']], 
                          dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def _action_to_features(self, action: str, agent_type: str) -> torch.Tensor:
        """Convert action to feature tensor"""
        
        # Simple action encoding - in production, use embedding lookup
        action_hash = hash(action) % 1000000
        agent_hash = hash(agent_type) % 1000000
        
        features = [
            action_hash / 1000000.0,
            agent_hash / 1000000.0,
            len(action) / 100.0,
            1.0 if 'attack' in action else 0.0,
            1.0 if 'recon' in action else 0.0,
            1.0 if 'exploit' in action else 0.0
        ]
        
        # Pad to required dimension
        while len(features) < self.model_config['action_dim']:
            features.append(0.0)
        
        return torch.tensor(features[:self.model_config['action_dim']], 
                          dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def _calculate_surprise_factor(self, 
                                 predicted_confidence: float,
                                 actual_outcome: DecisionOutcome,
                                 success_metrics: Dict[str, float]) -> float:
        """Calculate how surprising the outcome was"""
        
        # Convert outcome to expected success probability
        outcome_success_prob = {
            DecisionOutcome.SUCCESS: 0.9,
            DecisionOutcome.PARTIAL_SUCCESS: 0.6,
            DecisionOutcome.FAILURE: 0.1,
            DecisionOutcome.ERROR: 0.0,
            DecisionOutcome.TIMEOUT: 0.2,
            DecisionOutcome.UNKNOWN: 0.5
        }.get(actual_outcome, 0.5)
        
        # Calculate surprise as difference between prediction and reality
        surprise = abs(predicted_confidence - outcome_success_prob)
        
        # Adjust based on success metrics variance
        if success_metrics:
            metrics_variance = np.var(list(success_metrics.values()))
            surprise *= (1.0 + metrics_variance)  # Higher variance = more surprising
        
        return min(1.0, surprise)
    
    def _calculate_post_outcome_confidence(self,
                                         outcome: DecisionOutcome,
                                         success_metrics: Dict[str, float]) -> float:
        """Calculate confidence after seeing the outcome"""
        
        base_confidence = {
            DecisionOutcome.SUCCESS: 0.9,
            DecisionOutcome.PARTIAL_SUCCESS: 0.7,
            DecisionOutcome.FAILURE: 0.3,
            DecisionOutcome.ERROR: 0.1,
            DecisionOutcome.TIMEOUT: 0.4,
            DecisionOutcome.UNKNOWN: 0.5
        }.get(outcome, 0.5)
        
        # Adjust based on success metrics
        if success_metrics:
            avg_metric = np.mean(list(success_metrics.values()))
            base_confidence = (base_confidence + avg_metric) / 2.0
        
        return base_confidence
    
    def _calculate_learning_opportunity(self, 
                                      surprise_factor: float,
                                      outcome: DecisionOutcome) -> float:
        """Calculate learning opportunity score"""
        
        # High surprise = high learning opportunity
        opportunity = surprise_factor
        
        # Failed outcomes with high surprise = highest learning opportunity
        if outcome in [DecisionOutcome.FAILURE, DecisionOutcome.ERROR]:
            opportunity *= 1.5
        
        # Successful outcomes with high surprise = moderate learning opportunity
        elif outcome == DecisionOutcome.SUCCESS:
            opportunity *= 1.2
        
        return min(1.0, opportunity)
    
    async def _update_reasoning_patterns(self):
        """Update reasoning patterns based on recent reflections"""
        
        recent_reflections = list(self.reflection_buffer)[-500:]  # Last 500 reflections
        new_patterns = self.meta_analyzer.analyze_reasoning_patterns(recent_reflections)
        
        # Update active patterns
        for pattern in new_patterns:
            self.active_patterns[pattern.pattern_id] = pattern
        
        # Remove old patterns (keep only recent ones)
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        self.active_patterns = {
            pid: pattern for pid, pattern in self.active_patterns.items()
            if pattern.last_observed >= cutoff_time
        }
        
        self.introspection_metrics['patterns_identified'] = len(self.active_patterns)
        self.logger.debug(f"Updated reasoning patterns: {len(new_patterns)} new, {len(self.active_patterns)} active")
    
    # Analysis methods for reasoning evolution
    def _analyze_reasoning_level_evolution(self, reflections: List[ActionReflection]) -> Dict[str, Any]:
        """Analyze how reasoning levels have evolved"""
        
        level_timeline = [(r.timestamp, r.reasoning_level) for r in reflections]
        level_timeline.sort()
        
        # Count level usage over time
        early_half = level_timeline[:len(level_timeline)//2]
        late_half = level_timeline[len(level_timeline)//2:]
        
        early_counts = defaultdict(int)
        late_counts = defaultdict(int)
        
        for _, level in early_half:
            early_counts[level.value] += 1
        
        for _, level in late_half:
            late_counts[level.value] += 1
        
        return {
            'early_period_distribution': dict(early_counts),
            'late_period_distribution': dict(late_counts),
            'complexity_trend': self._calculate_complexity_trend(early_counts, late_counts)
        }
    
    def _analyze_confidence_calibration(self, reflections: List[ActionReflection]) -> Dict[str, Any]:
        """Analyze confidence calibration over time"""
        
        calibration_errors = []
        for reflection in reflections:
            error = abs(reflection.confidence_at_decision - reflection.confidence_post_outcome)
            calibration_errors.append((reflection.timestamp, error))
        
        # Split into time periods
        mid_point = len(calibration_errors) // 2
        early_errors = [error for _, error in calibration_errors[:mid_point]]
        late_errors = [error for _, error in calibration_errors[mid_point:]]
        
        return {
            'early_average_error': np.mean(early_errors) if early_errors else 0.0,
            'late_average_error': np.mean(late_errors) if late_errors else 0.0,
            'improvement': (np.mean(early_errors) - np.mean(late_errors)) if early_errors and late_errors else 0.0,
            'calibration_trend': 'improving' if np.mean(early_errors) > np.mean(late_errors) else 'stable'
        }
    
    def _analyze_decision_factor_shifts(self, reflections: List[ActionReflection]) -> Dict[str, Any]:
        """Analyze how decision factors have shifted over time"""
        
        factor_evolution = defaultdict(list)
        
        for reflection in reflections:
            for factor, weight in reflection.decision_factors.items():
                factor_evolution[factor].append((reflection.timestamp, weight))
        
        # Calculate trends for each factor
        factor_trends = {}
        for factor, timeline in factor_evolution.items():
            if len(timeline) >= 3:
                timestamps = [t.timestamp() for t, _ in timeline]
                weights = [w for _, w in timeline]
                
                # Simple linear trend
                correlation = np.corrcoef(timestamps, weights)[0, 1] if len(set(weights)) > 1 else 0
                factor_trends[factor] = {
                    'correlation': correlation,
                    'trend': 'increasing' if correlation > 0.1 else 'decreasing' if correlation < -0.1 else 'stable',
                    'average_weight': np.mean(weights)
                }
        
        return factor_trends
    
    def _analyze_learning_trajectory(self, reflections: List[ActionReflection]) -> Dict[str, Any]:
        """Analyze the agent's learning trajectory"""
        
        learning_scores = [(r.timestamp, r.learning_opportunity_score) for r in reflections]
        surprise_scores = [(r.timestamp, r.surprise_factor) for r in reflections]
        
        # Calculate moving averages
        window_size = min(10, len(learning_scores) // 3)
        if window_size >= 3:
            learning_ma = self._moving_average([score for _, score in learning_scores], window_size)
            surprise_ma = self._moving_average([score for _, score in surprise_scores], window_size)
            
            learning_trend = 'improving' if learning_ma[-1] > learning_ma[0] else 'declining'
            adaptation_trend = 'improving' if surprise_ma[0] > surprise_ma[-1] else 'stable'
        else:
            learning_trend = 'insufficient_data'
            adaptation_trend = 'insufficient_data'
        
        return {
            'learning_trend': learning_trend,
            'adaptation_trend': adaptation_trend,
            'average_learning_opportunity': np.mean([score for _, score in learning_scores]),
            'average_surprise_factor': np.mean([score for _, score in surprise_scores])
        }
    
    def _analyze_surprise_adaptation(self, reflections: List[ActionReflection]) -> Dict[str, Any]:
        """Analyze how well the agent adapts to surprising outcomes"""
        
        high_surprise_reflections = [r for r in reflections if r.surprise_factor > 0.5]
        if len(high_surprise_reflections) < 3:
            return {'status': 'insufficient_surprise_data'}
        
        # Analyze performance following high surprise events
        surprise_recovery = []
        for i, reflection in enumerate(reflections):
            if reflection.surprise_factor > 0.5 and i < len(reflections) - 1:
                next_reflection = reflections[i + 1]
                recovery_score = self._calculate_recovery_score(reflection, next_reflection)
                surprise_recovery.append(recovery_score)
        
        return {
            'high_surprise_events': len(high_surprise_reflections),
            'average_recovery_score': np.mean(surprise_recovery) if surprise_recovery else 0.0,
            'adaptation_capability': 'good' if np.mean(surprise_recovery) > 0.6 else 'needs_improvement'
        }
    
    def _analyze_performance_trend(self, reflections: List[ActionReflection]) -> Dict[str, Any]:
        """Analyze overall performance trend"""
        
        success_timeline = []
        for reflection in reflections:
            success_score = 1.0 if reflection.outcome == DecisionOutcome.SUCCESS else 0.0
            success_timeline.append((reflection.timestamp, success_score))
        
        # Calculate success rate trend
        mid_point = len(success_timeline) // 2
        early_success_rate = np.mean([score for _, score in success_timeline[:mid_point]])
        late_success_rate = np.mean([score for _, score in success_timeline[mid_point:]])
        
        return {
            'early_success_rate': early_success_rate,
            'late_success_rate': late_success_rate,
            'improvement': late_success_rate - early_success_rate,
            'trend': 'improving' if late_success_rate > early_success_rate else 'declining'
        }
    
    def _generate_meta_insights(self, reflections: List[ActionReflection]) -> List[str]:
        """Generate high-level insights about reasoning patterns"""
        
        insights = []
        
        # Analyze reasoning complexity
        meta_cognitive_count = len([r for r in reflections 
                                  if r.reasoning_level == ReasoningLevel.META_COGNITIVE])
        if meta_cognitive_count > len(reflections) * 0.3:
            insights.append("Agent demonstrates strong meta-cognitive reasoning capabilities")
        
        # Analyze adaptability
        unique_contexts = len(set(r.campaign_phase for r in reflections))
        if unique_contexts > 5:
            insights.append("Agent shows good adaptability across diverse contexts")
        
        # Analyze learning efficiency
        avg_learning_score = np.mean([r.learning_opportunity_score for r in reflections])
        if avg_learning_score > 0.7:
            insights.append("Agent efficiently identifies learning opportunities")
        
        return insights
    
    # Helper methods
    def _calculate_complexity_trend(self, early_counts, late_counts) -> str:
        """Calculate if reasoning complexity is increasing"""
        
        complexity_weights = {
            'reactive': 1,
            'tactical': 2,
            'strategic': 3,
            'meta_cognitive': 4
        }
        
        early_complexity = sum(complexity_weights.get(level, 0) * count 
                             for level, count in early_counts.items())
        late_complexity = sum(complexity_weights.get(level, 0) * count 
                            for level, count in late_counts.items())
        
        early_total = sum(early_counts.values())
        late_total = sum(late_counts.values())
        
        if early_total > 0 and late_total > 0:
            early_avg = early_complexity / early_total
            late_avg = late_complexity / late_total
            
            if late_avg > early_avg + 0.5:
                return 'increasing'
            elif late_avg < early_avg - 0.5:
                return 'decreasing'
        
        return 'stable'
    
    def _moving_average(self, data: List[float], window_size: int) -> List[float]:
        """Calculate moving average"""
        if len(data) < window_size:
            return data
        
        moving_avg = []
        for i in range(len(data) - window_size + 1):
            avg = np.mean(data[i:i + window_size])
            moving_avg.append(avg)
        
        return moving_avg
    
    def _calculate_recovery_score(self, 
                                surprise_reflection: ActionReflection,
                                next_reflection: ActionReflection) -> float:
        """Calculate how well the agent recovered from a surprising outcome"""
        
        # Base recovery on outcome improvement
        outcome_scores = {
            DecisionOutcome.SUCCESS: 1.0,
            DecisionOutcome.PARTIAL_SUCCESS: 0.7,
            DecisionOutcome.FAILURE: 0.3,
            DecisionOutcome.ERROR: 0.1,
            DecisionOutcome.TIMEOUT: 0.2,
            DecisionOutcome.UNKNOWN: 0.5
        }
        
        surprise_score = outcome_scores.get(surprise_reflection.outcome, 0.5)
        recovery_score = outcome_scores.get(next_reflection.outcome, 0.5)
        
        # Good recovery is improvement + reduced surprise
        recovery_improvement = max(0, recovery_score - surprise_score)
        surprise_reduction = max(0, surprise_reflection.surprise_factor - next_reflection.surprise_factor)
        
        return (recovery_improvement + surprise_reduction) / 2.0
    
    # Recommendation generation methods
    def _generate_failure_recommendations(self, pattern: ReasoningPattern) -> List[Dict[str, Any]]:
        """Generate recommendations for failure patterns"""
        
        recommendations = []
        
        for suggestion in pattern.improvement_suggestions:
            recommendations.append({
                'type': 'failure_mitigation',
                'pattern_id': pattern.pattern_id,
                'recommendation': suggestion,
                'priority_score': 0.8,
                'implementation_complexity': 'medium',
                'expected_impact': 'high'
            })
        
        return recommendations
    
    def _generate_bias_recommendations(self, pattern: ReasoningPattern) -> List[Dict[str, Any]]:
        """Generate recommendations for bias patterns"""
        
        recommendations = []
        
        if 'overconfidence' in pattern.pattern_id:
            recommendations.append({
                'type': 'bias_correction',
                'pattern_id': pattern.pattern_id,
                'recommendation': 'Implement confidence calibration using ensemble methods',
                'priority_score': 0.9,
                'implementation_complexity': 'high',
                'expected_impact': 'high'
            })
        
        return recommendations
    
    def _generate_optimization_recommendations(self, pattern: ReasoningPattern) -> List[Dict[str, Any]]:
        """Generate recommendations for optimizing successful patterns"""
        
        recommendations = []
        
        for suggestion in pattern.improvement_suggestions:
            recommendations.append({
                'type': 'optimization',
                'pattern_id': pattern.pattern_id,
                'recommendation': suggestion,
                'priority_score': 0.6,
                'implementation_complexity': 'low',
                'expected_impact': 'medium'
            })
        
        return recommendations
    
    def _generate_context_recommendations(self, 
                                        reflections: List[ActionReflection],
                                        current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate context-specific recommendations"""
        
        recommendations = []
        
        # Find similar contexts in reflection history
        similar_contexts = [
            r for r in reflections
            if self._calculate_context_similarity(r.context_snapshot, current_context) > 0.7
        ]
        
        if similar_contexts:
            # Analyze successful actions in similar contexts
            successful_actions = [
                r.action_taken for r in similar_contexts
                if r.outcome == DecisionOutcome.SUCCESS
            ]
            
            if successful_actions:
                most_successful_action = max(set(successful_actions), key=successful_actions.count)
                recommendations.append({
                    'type': 'context_specific',
                    'recommendation': f'Consider using action: {most_successful_action}',
                    'priority_score': 0.7,
                    'context_similarity': 'high',
                    'historical_success_rate': successful_actions.count(most_successful_action) / len(similar_contexts)
                })
        
        return recommendations
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts"""
        
        # Simple similarity based on common keys and values
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = context1[key], context2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2), 1e-6)
                similarity = 1.0 - abs(val1 - val2) / max_val
                similarities.append(max(0.0, similarity))
            elif val1 == val2:
                similarities.append(1.0)
            else:
                similarities.append(0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    async def get_introspection_status(self) -> Dict[str, Any]:
        """Get current status of the introspection system"""
        
        return {
            'system_active': True,
            'total_reflections': len(self.reflection_buffer),
            'active_patterns': len(self.active_patterns),
            'metrics': self.introspection_metrics.copy(),
            'pattern_summary': {
                pattern_id: {
                    'type': pattern.pattern_type,
                    'frequency': pattern.frequency,
                    'trend': pattern.trend_direction
                }
                for pattern_id, pattern in self.active_patterns.items()
            },
            'recent_activity': {
                'reflections_last_hour': len([
                    r for r in self.reflection_buffer
                    if (datetime.utcnow() - r.timestamp).total_seconds() < 3600
                ]),
                'unique_agents_analyzed': len(set(
                    r.action_id.split('_')[0] for r in list(self.reflection_buffer)[-100:]
                ))
            }
        }


if __name__ == "__main__":
    async def main():
        # Example usage
        reasoning_system = IntrospectiveReasoningSystem()
        
        # Simulate agent reflections
        for i in range(20):
            agent_id = f"agent_{i % 3}"
            agent_type = ["recon_agent", "exploit_agent", "persistence_agent"][i % 3]
            action = ["scan_target", "exploit_vulnerability", "establish_persistence"][i % 3]
            
            context = {
                'target_complexity': np.random.random(),
                'resource_availability': np.random.random(),
                'time_pressure': np.random.random(),
                'campaign_phase': ['initial', 'active', 'cleanup'][i % 3]
            }
            
            outcome = np.random.choice(list(DecisionOutcome))
            success_metrics = {'accuracy': np.random.random(), 'efficiency': np.random.random()}
            execution_details = {
                'execution_time': np.random.random() * 300,
                'resource_usage': {'cpu': np.random.random(), 'memory': np.random.random()},
                'initial_confidence': np.random.random()
            }
            
            reflection = await reasoning_system.reflect_on_action(
                agent_id, agent_type, action, context, outcome, success_metrics, execution_details
            )
            
            print(f"Created reflection {reflection.action_id}: {outcome.value}")
        
        # Analyze reasoning evolution
        evolution = await reasoning_system.analyze_reasoning_evolution("agent_0", 24)
        print(f"\nReasoning evolution: {json.dumps(evolution, indent=2, default=str)}")
        
        # Generate recommendations
        recommendations = await reasoning_system.generate_reasoning_recommendations("agent_0", context)
        print(f"\nRecommendations: {json.dumps(recommendations, indent=2)}")
        
        # Get status
        status = await reasoning_system.get_introspection_status()
        print(f"\nSystem status: {json.dumps(status, indent=2)}")
    
    asyncio.run(main())