#!/usr/bin/env python3
"""
Adaptive Anomaly Reinforcement Shaper for Xorb 2.0

This module implements adaptive reward shaping based on anomaly detection,
enabling the RL system to dynamically adjust rewards when encountering
novel attack patterns or unexpected system behaviors.
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import json
from pathlib import Path

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available for anomaly detection")


class AnomalyType(Enum):
    """Types of anomalies that can be detected"""
    STATISTICAL_OUTLIER = "statistical_outlier"
    TEMPORAL_ANOMALY = "temporal_anomaly"
    BEHAVIORAL_SHIFT = "behavioral_shift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    NOVEL_PATTERN = "novel_pattern"
    SECURITY_INCIDENT = "security_incident"
    RESOURCE_ANOMALY = "resource_anomaly"


@dataclass
class AnomalyEvent:
    """Represents a detected anomaly"""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    affected_components: List[str]
    detection_timestamp: datetime
    
    # Context information
    state_vector: np.ndarray
    reward_context: Dict[str, float]
    campaign_context: Dict[str, Any]
    
    # Anomaly-specific data
    anomaly_score: float
    baseline_deviation: float
    temporal_pattern: Optional[Dict[str, Any]] = None
    
    # Resolution tracking
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    adaptation_applied: bool = False


@dataclass
class RewardShapingRule:
    """Rule for shaping rewards based on anomaly patterns"""
    rule_id: str
    anomaly_pattern: AnomalyType
    shaping_function: str  # Function name or lambda string
    magnitude: float  # Multiplier for reward adjustment
    duration_minutes: int  # How long the rule stays active
    
    # Conditions
    min_confidence: float = 0.7
    min_severity: float = 0.5
    max_applications: int = 100
    
    # State tracking
    applications_count: int = 0
    last_applied: Optional[datetime] = None
    effectiveness_score: float = 0.5
    created_at: datetime = field(default_factory=datetime.utcnow)


class OnlineAnomalyDetector:
    """Online anomaly detection system for real-time adaptation"""
    
    def __init__(self, 
                 window_size: int = 1000,
                 contamination: float = 0.1,
                 n_estimators: int = 100):
        
        self.window_size = window_size
        self.contamination = contamination
        self.n_estimators = n_estimators
        
        # Sliding window for online detection
        self.state_history = deque(maxlen=window_size)
        self.reward_history = deque(maxlen=window_size)
        self.timestamp_history = deque(maxlen=window_size)
        
        # Detection models
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Retain 95% variance
        
        # Baseline statistics
        self.baseline_stats = {
            'mean_state': None,
            'std_state': None,
            'mean_reward': 0.0,
            'std_reward': 1.0,
            'update_count': 0
        }
        
        # Model update frequency
        self.model_update_interval = 100  # Update every 100 samples
        self.last_model_update = 0
        
        self.logger = logging.getLogger(__name__)
    
    def add_observation(self, 
                       state_vector: np.ndarray, 
                       reward: float, 
                       timestamp: datetime = None):
        """Add new observation to the detector"""
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.state_history.append(state_vector.copy())
        self.reward_history.append(reward)
        self.timestamp_history.append(timestamp)
        
        # Update baseline statistics
        self._update_baseline_stats(state_vector, reward)
        
        # Periodically retrain models
        if len(self.state_history) >= 50 and (len(self.state_history) - self.last_model_update) >= self.model_update_interval:
            self._update_detection_models()
    
    def _update_baseline_stats(self, state_vector: np.ndarray, reward: float):
        """Update running baseline statistics"""
        
        count = self.baseline_stats['update_count']
        
        # Update reward statistics
        if count == 0:
            self.baseline_stats['mean_reward'] = reward
            self.baseline_stats['std_reward'] = 0.0
        else:
            # Online update of mean and variance
            old_mean = self.baseline_stats['mean_reward']
            new_mean = old_mean + (reward - old_mean) / (count + 1)
            
            if count > 1:
                old_var = self.baseline_stats['std_reward'] ** 2
                new_var = ((count - 1) * old_var + (reward - old_mean) * (reward - new_mean)) / count
                self.baseline_stats['std_reward'] = np.sqrt(max(0, new_var))
            
            self.baseline_stats['mean_reward'] = new_mean
        
        # Update state statistics
        if self.baseline_stats['mean_state'] is None:
            self.baseline_stats['mean_state'] = state_vector.copy()
            self.baseline_stats['std_state'] = np.zeros_like(state_vector)
        else:
            # Online update for state vector statistics
            old_mean = self.baseline_stats['mean_state']
            new_mean = old_mean + (state_vector - old_mean) / (count + 1)
            
            if count > 1:
                old_var = self.baseline_stats['std_state'] ** 2
                new_var = ((count - 1) * old_var + (state_vector - old_mean) * (state_vector - new_mean)) / count
                self.baseline_stats['std_state'] = np.sqrt(np.maximum(new_var, 1e-8))
            
            self.baseline_stats['mean_state'] = new_mean
        
        self.baseline_stats['update_count'] = count + 1
    
    def _update_detection_models(self):
        """Update anomaly detection models with recent data"""
        
        if not SKLEARN_AVAILABLE or len(self.state_history) < 50:
            return
        
        try:
            # Convert deque to numpy array
            states = np.array(list(self.state_history))
            
            # Fit scaler and PCA on recent data
            scaled_states = self.scaler.fit_transform(states)
            
            # Apply PCA if state dimension is high
            if states.shape[1] > 10:
                reduced_states = self.pca.fit_transform(scaled_states)
            else:
                reduced_states = scaled_states
            
            # Update isolation forest
            self.isolation_forest = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=42
            )
            self.isolation_forest.fit(reduced_states)
            
            self.last_model_update = len(self.state_history)
            self.logger.debug(f"Updated anomaly detection models with {len(states)} samples")
            
        except Exception as e:
            self.logger.error(f"Error updating detection models: {e}")
    
    def detect_anomalies(self, 
                        state_vector: np.ndarray, 
                        reward: float,
                        campaign_context: Dict[str, Any] = None) -> List[AnomalyEvent]:
        """Detect anomalies in current state and reward"""
        
        anomalies = []
        campaign_context = campaign_context or {}
        
        # Statistical outlier detection
        stat_anomaly = self._detect_statistical_anomaly(state_vector, reward)
        if stat_anomaly:
            anomalies.append(stat_anomaly)
        
        # Temporal anomaly detection
        temporal_anomaly = self._detect_temporal_anomaly(state_vector, reward)
        if temporal_anomaly:
            anomalies.append(temporal_anomaly)
        
        # Behavioral shift detection
        behavioral_anomaly = self._detect_behavioral_shift(state_vector, reward)
        if behavioral_anomaly:
            anomalies.append(behavioral_anomaly)
        
        # Resource anomaly detection (EPYC-specific)
        resource_anomaly = self._detect_resource_anomaly(state_vector, campaign_context)
        if resource_anomaly:
            anomalies.append(resource_anomaly)
        
        return anomalies
    
    def _detect_statistical_anomaly(self, 
                                  state_vector: np.ndarray, 
                                  reward: float) -> Optional[AnomalyEvent]:
        """Detect statistical outliers using isolation forest"""
        
        if not SKLEARN_AVAILABLE or self.isolation_forest is None:
            return None
        
        try:
            # Prepare input
            input_vector = state_vector.reshape(1, -1)
            scaled_input = self.scaler.transform(input_vector)
            
            if hasattr(self.pca, 'transform') and state_vector.shape[0] > 10:
                reduced_input = self.pca.transform(scaled_input)
            else:
                reduced_input = scaled_input
            
            # Get anomaly score
            anomaly_score = self.isolation_forest.decision_function(reduced_input)[0]
            is_outlier = self.isolation_forest.predict(reduced_input)[0] == -1
            
            if is_outlier:
                # Calculate severity based on how far from normal
                severity = min(1.0, abs(anomaly_score) / 0.5)  # Normalize to 0-1
                confidence = min(1.0, abs(anomaly_score) / 0.3)
                
                return AnomalyEvent(
                    anomaly_id=f"stat_{datetime.utcnow().isoformat()}",
                    anomaly_type=AnomalyType.STATISTICAL_OUTLIER,
                    severity=severity,
                    confidence=confidence,
                    affected_components=["state_space"],
                    detection_timestamp=datetime.utcnow(),
                    state_vector=state_vector,
                    reward_context={"current_reward": reward},
                    campaign_context={},
                    anomaly_score=float(anomaly_score),
                    baseline_deviation=severity
                )
                
        except Exception as e:
            self.logger.debug(f"Error in statistical anomaly detection: {e}")
        
        return None
    
    def _detect_temporal_anomaly(self, 
                               state_vector: np.ndarray, 
                               reward: float) -> Optional[AnomalyEvent]:
        """Detect temporal anomalies in reward patterns"""
        
        if len(self.reward_history) < 20:
            return None
        
        recent_rewards = list(self.reward_history)[-20:]
        current_reward = reward
        
        # Check for sudden reward changes
        recent_mean = np.mean(recent_rewards)
        recent_std = np.std(recent_rewards)
        
        if recent_std > 0:
            z_score = abs(current_reward - recent_mean) / recent_std
            
            if z_score > 3.0:  # 3-sigma rule
                severity = min(1.0, z_score / 5.0)
                confidence = min(1.0, z_score / 4.0)
                
                # Analyze temporal pattern
                timestamps = list(self.timestamp_history)[-20:]
                time_diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                            for i in range(1, len(timestamps))]
                
                temporal_pattern = {
                    'reward_trend': 'spike' if current_reward > recent_mean else 'drop',
                    'z_score': float(z_score),
                    'recent_mean': float(recent_mean),
                    'recent_std': float(recent_std),
                    'avg_time_diff': float(np.mean(time_diffs)) if time_diffs else 0.0
                }
                
                return AnomalyEvent(
                    anomaly_id=f"temp_{datetime.utcnow().isoformat()}",
                    anomaly_type=AnomalyType.TEMPORAL_ANOMALY,
                    severity=severity,
                    confidence=confidence,
                    affected_components=["reward_system"],
                    detection_timestamp=datetime.utcnow(),
                    state_vector=state_vector,
                    reward_context={"current_reward": reward, "recent_mean": recent_mean},
                    campaign_context={},
                    anomaly_score=float(z_score),
                    baseline_deviation=severity,
                    temporal_pattern=temporal_pattern
                )
        
        return None
    
    def _detect_behavioral_shift(self, 
                               state_vector: np.ndarray, 
                               reward: float) -> Optional[AnomalyEvent]:
        """Detect shifts in behavioral patterns"""
        
        if len(self.state_history) < 100:
            return None
        
        # Compare recent states vs historical baseline
        recent_states = np.array(list(self.state_history)[-50:])
        older_states = np.array(list(self.state_history)[-100:-50])
        
        # Calculate distribution shift using KL divergence approximation
        recent_mean = np.mean(recent_states, axis=0)
        older_mean = np.mean(older_states, axis=0)
        
        # Simple shift detection using Euclidean distance
        shift_magnitude = np.linalg.norm(recent_mean - older_mean)
        
        # Normalize by historical standard deviation
        if self.baseline_stats['std_state'] is not None:
            normalized_shift = shift_magnitude / (np.mean(self.baseline_stats['std_state']) + 1e-8)
        else:
            normalized_shift = shift_magnitude
        
        if normalized_shift > 2.0:  # Threshold for significant shift
            severity = min(1.0, normalized_shift / 5.0)
            confidence = min(1.0, normalized_shift / 3.0)
            
            return AnomalyEvent(
                anomaly_id=f"behav_{datetime.utcnow().isoformat()}",
                anomaly_type=AnomalyType.BEHAVIORAL_SHIFT,
                severity=severity,
                confidence=confidence,
                affected_components=["behavior_pattern"],
                detection_timestamp=datetime.utcnow(),
                state_vector=state_vector,
                reward_context={"current_reward": reward},
                campaign_context={},
                anomaly_score=float(normalized_shift),
                baseline_deviation=severity
            )
        
        return None
    
    def _detect_resource_anomaly(self, 
                               state_vector: np.ndarray, 
                               campaign_context: Dict[str, Any]) -> Optional[AnomalyEvent]:
        """Detect EPYC-specific resource anomalies"""
        
        # Extract resource-related features from state vector
        # Assuming specific indices for resource metrics
        if len(state_vector) < 8:
            return None
        
        # EPYC resource features (based on common patterns)
        cpu_util = state_vector[0] if len(state_vector) > 0 else 0.0
        memory_util = state_vector[1] if len(state_vector) > 1 else 0.0
        numa_locality = state_vector[2] if len(state_vector) > 2 else 0.0
        thermal_state = state_vector[3] if len(state_vector) > 3 else 0.0
        
        anomaly_indicators = []
        
        # Check for resource exhaustion patterns
        if cpu_util > 0.95:
            anomaly_indicators.append("cpu_exhaustion")
        
        if memory_util > 0.90:
            anomaly_indicators.append("memory_pressure")
        
        if numa_locality < 0.3:
            anomaly_indicators.append("numa_thrashing")
        
        if thermal_state > 0.85:
            anomaly_indicators.append("thermal_throttling")
        
        if anomaly_indicators:
            severity = len(anomaly_indicators) / 4.0  # Normalize by max indicators
            confidence = 0.8  # High confidence for resource metrics
            
            return AnomalyEvent(
                anomaly_id=f"resource_{datetime.utcnow().isoformat()}",
                anomaly_type=AnomalyType.RESOURCE_ANOMALY,
                severity=severity,
                confidence=confidence,
                affected_components=anomaly_indicators,
                detection_timestamp=datetime.utcnow(),
                state_vector=state_vector,
                reward_context={},
                campaign_context=campaign_context,
                anomaly_score=severity,
                baseline_deviation=severity
            )
        
        return None


class AdaptiveRewardShaper:
    """Adaptive reward shaping based on detected anomalies"""
    
    def __init__(self, 
                 base_reward_scale: float = 1.0,
                 adaptation_rate: float = 0.1,
                 max_adaptation_magnitude: float = 2.0):
        
        self.base_reward_scale = base_reward_scale
        self.adaptation_rate = adaptation_rate
        self.max_adaptation_magnitude = max_adaptation_magnitude
        
        # Shaping rules
        self.active_rules: Dict[str, RewardShapingRule] = {}
        self.rule_history: List[RewardShapingRule] = []
        
        # Adaptation state
        self.adaptation_factors = defaultdict(float)
        self.shaping_history = deque(maxlen=1000)
        
        # Performance tracking
        self.effectiveness_metrics = {
            'total_adaptations': 0,
            'successful_adaptations': 0,
            'false_positive_rate': 0.0,
            'average_improvement': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize default shaping rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default reward shaping rules"""
        
        default_rules = [
            RewardShapingRule(
                rule_id="statistical_outlier_penalty",
                anomaly_pattern=AnomalyType.STATISTICAL_OUTLIER,
                shaping_function="negative_exponential",
                magnitude=0.5,  # Reduce reward by up to 50%
                duration_minutes=10,
                min_confidence=0.8
            ),
            RewardShapingRule(
                rule_id="temporal_spike_dampening",
                anomaly_pattern=AnomalyType.TEMPORAL_ANOMALY,
                shaping_function="dampening",
                magnitude=0.3,
                duration_minutes=5,
                min_confidence=0.7
            ),
            RewardShapingRule(
                rule_id="behavioral_shift_exploration",
                anomaly_pattern=AnomalyType.BEHAVIORAL_SHIFT,
                shaping_function="exploration_bonus",
                magnitude=1.2,  # 20% bonus for exploration
                duration_minutes=30,
                min_confidence=0.6
            ),
            RewardShapingRule(
                rule_id="resource_anomaly_penalty",
                anomaly_pattern=AnomalyType.RESOURCE_ANOMALY,
                shaping_function="resource_penalty",
                magnitude=0.7,
                duration_minutes=15,
                min_confidence=0.8
            ),
            RewardShapingRule(
                rule_id="novel_pattern_bonus",
                anomaly_pattern=AnomalyType.NOVEL_PATTERN,
                shaping_function="novelty_bonus",
                magnitude=1.5,  # 50% bonus for novel patterns
                duration_minutes=20,
                min_confidence=0.5
            )
        ]
        
        for rule in default_rules:
            self.active_rules[rule.rule_id] = rule
    
    def shape_reward(self, 
                    base_reward: float,
                    anomalies: List[AnomalyEvent],
                    context: Dict[str, Any] = None) -> Tuple[float, Dict[str, Any]]:
        """Apply adaptive reward shaping based on detected anomalies"""
        
        shaped_reward = base_reward
        applied_rules = []
        shaping_metadata = {
            'base_reward': base_reward,
            'anomalies_detected': len(anomalies),
            'shaping_factors': {}
        }
        
        # Apply shaping rules for each detected anomaly
        for anomaly in anomalies:
            applicable_rules = self._get_applicable_rules(anomaly)
            
            for rule in applicable_rules:
                if self._should_apply_rule(rule, anomaly):
                    shaping_factor = self._calculate_shaping_factor(rule, anomaly)
                    shaped_reward = self._apply_shaping_function(
                        shaped_reward, rule.shaping_function, shaping_factor
                    )
                    
                    # Track rule application
                    rule.applications_count += 1
                    rule.last_applied = datetime.utcnow()
                    applied_rules.append(rule.rule_id)
                    
                    shaping_metadata['shaping_factors'][rule.rule_id] = shaping_factor
        
        # Apply global adaptation factors
        global_factor = self._get_global_adaptation_factor(context)
        shaped_reward *= global_factor
        
        # Record shaping history
        shaping_record = {
            'timestamp': datetime.utcnow(),
            'base_reward': base_reward,
            'shaped_reward': shaped_reward,
            'applied_rules': applied_rules,
            'anomalies': [a.anomaly_type.value for a in anomalies],
            'global_factor': global_factor
        }
        self.shaping_history.append(shaping_record)
        
        # Update metrics
        self.effectiveness_metrics['total_adaptations'] += 1
        
        shaping_metadata.update({
            'shaped_reward': shaped_reward,
            'applied_rules': applied_rules,
            'global_factor': global_factor,
            'shaping_magnitude': abs(shaped_reward - base_reward) / (abs(base_reward) + 1e-8)
        })
        
        return shaped_reward, shaping_metadata
    
    def _get_applicable_rules(self, anomaly: AnomalyEvent) -> List[RewardShapingRule]:
        """Get rules applicable to a specific anomaly"""
        
        applicable = []
        
        for rule in self.active_rules.values():
            if (rule.anomaly_pattern == anomaly.anomaly_type and
                anomaly.confidence >= rule.min_confidence and
                anomaly.severity >= rule.min_severity):
                applicable.append(rule)
        
        return applicable
    
    def _should_apply_rule(self, rule: RewardShapingRule, anomaly: AnomalyEvent) -> bool:
        """Determine if a rule should be applied"""
        
        # Check application count limit
        if rule.applications_count >= rule.max_applications:
            return False
        
        # Check time-based constraints
        if rule.last_applied:
            time_since_last = (datetime.utcnow() - rule.last_applied).total_seconds() / 60
            if time_since_last < rule.duration_minutes / 10:  # Cooldown period
                return False
        
        # Check rule effectiveness
        if rule.effectiveness_score < 0.3:  # Don't apply ineffective rules
            return False
        
        return True
    
    def _calculate_shaping_factor(self, rule: RewardShapingRule, anomaly: AnomalyEvent) -> float:
        """Calculate the shaping factor for a rule-anomaly pair"""
        
        # Base factor from rule magnitude
        base_factor = rule.magnitude
        
        # Adjust based on anomaly severity and confidence
        severity_adjustment = anomaly.severity
        confidence_adjustment = anomaly.confidence
        
        # Calculate final factor
        if rule.shaping_function in ["negative_exponential", "dampening", "resource_penalty"]:
            # For penalty functions, higher severity = lower factor
            factor = base_factor * (1.0 - severity_adjustment * confidence_adjustment)
        else:
            # For bonus functions, higher severity = higher factor
            factor = base_factor * (1.0 + severity_adjustment * confidence_adjustment)
        
        # Clamp to reasonable bounds
        return max(0.1, min(self.max_adaptation_magnitude, factor))
    
    def _apply_shaping_function(self, 
                              current_reward: float, 
                              function_name: str, 
                              factor: float) -> float:
        """Apply specific shaping function to reward"""
        
        if function_name == "negative_exponential":
            # Exponential penalty based on factor
            return current_reward * np.exp(-(1.0 - factor))
        
        elif function_name == "dampening":
            # Simple multiplicative dampening
            return current_reward * factor
        
        elif function_name == "exploration_bonus":
            # Additive bonus for exploration
            bonus = abs(current_reward) * (factor - 1.0)
            return current_reward + bonus
        
        elif function_name == "resource_penalty":
            # Penalty based on resource usage
            penalty = abs(current_reward) * (1.0 - factor)
            return current_reward - penalty
        
        elif function_name == "novelty_bonus":
            # Bonus for novel patterns
            bonus = abs(current_reward) * (factor - 1.0) * 0.5  # Moderate bonus
            return current_reward + bonus
        
        else:
            # Default: simple multiplication
            return current_reward * factor
    
    def _get_global_adaptation_factor(self, context: Dict[str, Any] = None) -> float:
        """Calculate global adaptation factor based on system state"""
        
        context = context or {}
        factor = 1.0
        
        # Adapt based on recent performance
        if len(self.shaping_history) >= 10:
            recent_improvements = []
            for record in list(self.shaping_history)[-10:]:
                if record['base_reward'] != 0:
                    improvement = (record['shaped_reward'] - record['base_reward']) / abs(record['base_reward'])
                    recent_improvements.append(improvement)
            
            if recent_improvements:
                avg_improvement = np.mean(recent_improvements)
                # Adjust factor based on recent success
                factor *= (1.0 + avg_improvement * self.adaptation_rate)
        
        # Adapt based on system load (if available)
        system_load = context.get('system_load', 0.5)
        if system_load > 0.8:
            factor *= 0.9  # Reduce aggressiveness under high load
        
        # Clamp factor to reasonable bounds
        return max(0.5, min(1.5, factor))
    
    def update_rule_effectiveness(self, 
                                rule_id: str, 
                                effectiveness_score: float):
        """Update the effectiveness score of a shaping rule"""
        
        if rule_id in self.active_rules:
            rule = self.active_rules[rule_id]
            
            # Use exponential moving average
            alpha = 0.1
            rule.effectiveness_score = (alpha * effectiveness_score + 
                                      (1 - alpha) * rule.effectiveness_score)
            
            # Update global effectiveness metrics
            if effectiveness_score > 0.5:
                self.effectiveness_metrics['successful_adaptations'] += 1
            
            # Calculate rolling average improvement
            improvements = []
            for record in list(self.shaping_history)[-50:]:
                if rule_id in record.get('applied_rules', []):
                    base = record['base_reward']
                    shaped = record['shaped_reward']
                    if base != 0:
                        improvement = (shaped - base) / abs(base)
                        improvements.append(improvement)
            
            if improvements:
                avg_improvement = np.mean(improvements)
                self.effectiveness_metrics['average_improvement'] = avg_improvement
            
            self.logger.debug(f"Updated rule {rule_id} effectiveness: {rule.effectiveness_score:.3f}")
    
    def add_custom_rule(self, rule: RewardShapingRule) -> bool:
        """Add a custom reward shaping rule"""
        
        if rule.rule_id in self.active_rules:
            self.logger.warning(f"Rule {rule.rule_id} already exists")
            return False
        
        self.active_rules[rule.rule_id] = rule
        self.logger.info(f"Added custom shaping rule: {rule.rule_id}")
        return True
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a reward shaping rule"""
        
        if rule_id in self.active_rules:
            removed_rule = self.active_rules.pop(rule_id)
            self.rule_history.append(removed_rule)
            self.logger.info(f"Removed shaping rule: {rule_id}")
            return True
        
        return False
    
    def get_shaping_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about reward shaping"""
        
        stats = {
            'active_rules': len(self.active_rules),
            'total_applications': sum(rule.applications_count for rule in self.active_rules.values()),
            'effectiveness_metrics': self.effectiveness_metrics.copy(),
            'rule_effectiveness': {
                rule_id: rule.effectiveness_score 
                for rule_id, rule in self.active_rules.items()
            },
            'recent_shaping_frequency': 0.0,
            'average_shaping_magnitude': 0.0
        }
        
        # Calculate recent activity
        if len(self.shaping_history) >= 10:
            recent_records = list(self.shaping_history)[-50:]
            
            # Frequency of shaping
            shaped_count = sum(1 for r in recent_records if len(r.get('applied_rules', [])) > 0)
            stats['recent_shaping_frequency'] = shaped_count / len(recent_records)
            
            # Average magnitude
            magnitudes = [r.get('shaping_magnitude', 0.0) for r in recent_records]
            stats['average_shaping_magnitude'] = np.mean(magnitudes) if magnitudes else 0.0
        
        return stats


class AdaptiveAnomalyRLShaper:
    """Main class integrating anomaly detection with adaptive reward shaping"""
    
    def __init__(self, 
                 detector_config: Dict[str, Any] = None,
                 shaper_config: Dict[str, Any] = None):
        
        # Initialize components
        detector_config = detector_config or {}
        shaper_config = shaper_config or {}
        
        self.anomaly_detector = OnlineAnomalyDetector(**detector_config)
        self.reward_shaper = AdaptiveRewardShaper(**shaper_config)
        
        # Integration state
        self.anomaly_history = deque(maxlen=1000)
        self.performance_tracker = deque(maxlen=100)
        
        # Feedback learning
        self.feedback_buffer = []
        self.learning_rate = 0.01
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Adaptive Anomaly RL Shaper initialized")
    
    async def process_experience(self, 
                               state_vector: np.ndarray,
                               action_taken: int,
                               base_reward: float,
                               next_state_vector: np.ndarray,
                               done: bool,
                               campaign_context: Dict[str, Any] = None) -> Tuple[float, Dict[str, Any]]:
        """Process an RL experience with anomaly detection and reward shaping"""
        
        campaign_context = campaign_context or {}
        
        # Add observation to detector
        self.anomaly_detector.add_observation(state_vector, base_reward)
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(
            state_vector, base_reward, campaign_context
        )
        
        # Store anomalies
        self.anomaly_history.extend(anomalies)
        
        # Shape reward based on anomalies
        shaped_reward, shaping_metadata = self.reward_shaper.shape_reward(
            base_reward, anomalies, campaign_context
        )
        
        # Track performance
        performance_record = {
            'timestamp': datetime.utcnow(),
            'base_reward': base_reward,
            'shaped_reward': shaped_reward,
            'anomalies_count': len(anomalies),
            'shaping_applied': len(shaping_metadata.get('applied_rules', [])) > 0
        }
        self.performance_tracker.append(performance_record)
        
        # Prepare comprehensive metadata
        metadata = {
            'anomalies_detected': [
                {
                    'type': a.anomaly_type.value,
                    'severity': a.severity,
                    'confidence': a.confidence,
                    'components': a.affected_components
                } for a in anomalies
            ],
            'reward_shaping': shaping_metadata,
            'detector_stats': {
                'baseline_mean_reward': self.anomaly_detector.baseline_stats['mean_reward'],
                'baseline_std_reward': self.anomaly_detector.baseline_stats['std_reward'],
                'observations_count': len(self.anomaly_detector.state_history)
            }
        }
        
        return shaped_reward, metadata
    
    async def provide_feedback(self, 
                             campaign_id: str, 
                             final_performance: float,
                             applied_rules: List[str]):
        """Provide feedback on the effectiveness of applied shaping rules"""
        
        # Update rule effectiveness based on final performance
        for rule_id in applied_rules:
            # Simple effectiveness calculation based on performance
            effectiveness = min(1.0, max(0.0, final_performance / 100.0))  # Normalize
            self.reward_shaper.update_rule_effectiveness(rule_id, effectiveness)
        
        # Store feedback for learning
        feedback_record = {
            'campaign_id': campaign_id,
            'final_performance': final_performance,
            'applied_rules': applied_rules,
            'timestamp': datetime.utcnow()
        }
        self.feedback_buffer.append(feedback_record)
        
        # Limit buffer size
        if len(self.feedback_buffer) > 1000:
            self.feedback_buffer = self.feedback_buffer[-1000:]
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the adaptive shaping system"""
        
        return {
            'anomaly_detector': {
                'observations_count': len(self.anomaly_detector.state_history),
                'model_updated': self.anomaly_detector.isolation_forest is not None,
                'baseline_established': self.anomaly_detector.baseline_stats['update_count'] > 50
            },
            'reward_shaper': self.reward_shaper.get_shaping_statistics(),
            'recent_activity': {
                'anomalies_detected_24h': len([
                    a for a in self.anomaly_history 
                    if (datetime.utcnow() - a.detection_timestamp).total_seconds() < 86400
                ]),
                'performance_records': len(self.performance_tracker),
                'feedback_records': len(self.feedback_buffer)
            },
            'system_health': {
                'detection_active': len(self.anomaly_detector.state_history) > 0,
                'shaping_active': len(self.reward_shaper.active_rules) > 0,
                'learning_active': len(self.feedback_buffer) > 0
            }
        }


if __name__ == "__main__":
    async def main():
        # Example usage
        shaper = AdaptiveAnomalyRLShaper()
        
        # Simulate some RL experiences
        for i in range(100):
            state = np.random.randn(10)
            action = i % 5
            reward = np.random.normal(10, 3)  # Base reward with some noise
            next_state = np.random.randn(10)
            done = (i % 10 == 9)
            
            # Add some anomalous rewards occasionally
            if i % 20 == 0:
                reward += np.random.normal(0, 10)  # Anomalous spike
            
            shaped_reward, metadata = await shaper.process_experience(
                state, action, reward, next_state, done
            )
            
            if i % 20 == 0:
                print(f"Step {i}: Base reward: {reward:.2f}, Shaped: {shaped_reward:.2f}")
                if metadata['anomalies_detected']:
                    print(f"  Anomalies: {[a['type'] for a in metadata['anomalies_detected']]}")
        
        # Get status
        status = shaper.get_comprehensive_status()
        print(f"\nSystem Status:")
        print(f"  Observations: {status['anomaly_detector']['observations_count']}")
        print(f"  Active rules: {status['reward_shaper']['active_rules']}")
        print(f"  Recent anomalies: {status['recent_activity']['anomalies_detected_24h']}")
    
    asyncio.run(main())