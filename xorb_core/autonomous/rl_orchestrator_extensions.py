#!/usr/bin/env python3
"""
Reinforcement Learning Extensions for Autonomous Orchestrator
Enhanced with task preemption, confidence scoring, and adaptive learning
"""

import asyncio
import logging
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import heapq
import random

import structlog
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

from xorb_common.agents.base_agent import AgentCapability, AgentTask, AgentResult
from .models import AutonomousDecision, WorkloadProfile


@dataclass 
class PreemptionEvent:
    """Event for task preemption with context"""
    task_id: str
    agent_id: str
    reason: str
    priority_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    recovery_strategy: Optional[str] = None


@dataclass
class ConfidenceMetrics:
    """Worker confidence scoring metrics"""
    worker_id: str
    task_type_confidences: Dict[str, float] = field(default_factory=dict)
    historical_accuracy: float = 0.8
    resource_reliability: float = 0.9
    failure_recovery_rate: float = 0.85
    overall_confidence: float = 0.8
    confidence_trend: float = 0.0  # +/- trending
    last_updated: datetime = field(default_factory=datetime.utcnow)


class ReinforcementLearningAgent:
    """Reinforcement learning agent for orchestrator optimization"""
    
    def __init__(self, state_size: int = 50, action_size: int = 10):
        self.logger = structlog.get_logger("ReinforcementLearningAgent")
        self.state_size = state_size
        self.action_size = action_size
        
        # Q-learning parameters
        self.learning_rate = 0.01
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate
        
        # Q-table (simplified - would use neural networks in production)
        self.q_table: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        
        # Experience replay buffer
        self.experience_buffer: List[Tuple] = []
        self.buffer_size = 10000
        
        # State and action mappings
        self.action_mapping = {
            0: 'increase_concurrency',
            1: 'decrease_concurrency', 
            2: 'prioritize_high_confidence_tasks',
            3: 'preempt_low_priority_tasks',
            4: 'allocate_more_resources',
            5: 'reduce_resource_allocation',
            6: 'adjust_timeout_thresholds',
            7: 'switch_agent_selection_strategy',
            8: 'increase_learning_rate',
            9: 'maintain_current_policy'
        }
        
    def get_state_representation(self, system_metrics: Dict[str, Any]) -> str:
        """Convert system metrics to state representation"""
        
        # Discretize continuous metrics into state bins
        cpu_bin = min(4, int(system_metrics.get('cpu_usage', 0.0) * 5))
        memory_bin = min(4, int(system_metrics.get('memory_usage', 0.0) * 5))
        queue_bin = min(4, int(min(system_metrics.get('queue_depth', 0) / 10, 4)))
        success_rate_bin = min(4, int(system_metrics.get('success_rate', 0.8) * 5))
        active_tasks_bin = min(4, int(min(system_metrics.get('active_tasks', 0) / 8, 4)))
        
        # Create state string
        state = f"{cpu_bin}_{memory_bin}_{queue_bin}_{success_rate_bin}_{active_tasks_bin}"
        return state
        
    def select_action(self, state: str) -> int:
        """Select action using epsilon-greedy policy"""
        
        if time.time() % 1.0 < self.epsilon:  # Exploration
            return int(time.time() * 1000) % self.action_size
        else:  # Exploitation
            q_values = self.q_table[state]
            if not q_values:
                return 0  # Default action
            return max(q_values.keys(), key=lambda k: q_values[k])
    
    def update_q_value(self, state: str, action: int, reward: float, next_state: str):
        """Update Q-value using Q-learning algorithm"""
        
        current_q = self.q_table[state][action]
        
        # Get maximum Q-value for next state
        next_q_values = self.q_table[next_state]
        max_next_q = max(next_q_values.values()) if next_q_values else 0.0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q
        
        # Store experience for replay
        experience = (state, action, reward, next_state)
        self.experience_buffer.append(experience)
        
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
    
    def calculate_reward(self, previous_metrics: Dict[str, Any], 
                        current_metrics: Dict[str, Any],
                        action_taken: int) -> float:
        """Calculate reward based on system improvement"""
        
        reward = 0.0
        
        # Reward for improved success rate
        success_rate_improvement = (
            current_metrics.get('success_rate', 0.8) - 
            previous_metrics.get('success_rate', 0.8)
        )
        reward += success_rate_improvement * 10.0
        
        # Reward for reduced queue depth
        queue_improvement = (
            previous_metrics.get('queue_depth', 0) - 
            current_metrics.get('queue_depth', 0)
        )
        reward += queue_improvement * 0.1
        
        # Penalty for high resource usage
        cpu_usage = current_metrics.get('cpu_usage', 0.0)
        memory_usage = current_metrics.get('memory_usage', 0.0)
        
        if cpu_usage > 0.9 or memory_usage > 0.9:
            reward -= 5.0
        elif cpu_usage > 0.8 or memory_usage > 0.8:
            reward -= 1.0
        
        # Reward for optimal resource utilization (70-80%)
        if 0.7 <= cpu_usage <= 0.8 and 0.7 <= memory_usage <= 0.8:
            reward += 2.0
        
        # Penalty for security violations
        security_violations = current_metrics.get('security_violations', 0)
        reward -= security_violations * 10.0
        
        return reward
    
    def get_action_description(self, action: int) -> str:
        """Get human-readable description of action"""
        return self.action_mapping.get(action, f'unknown_action_{action}')
    
    def experience_replay(self, batch_size: int = 32):
        """Perform experience replay to improve learning"""
        
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample random batch from experience buffer
        batch = random.sample(self.experience_buffer, batch_size)
        
        # Update Q-values for sampled experiences
        for state, action, reward, next_state in batch:
            self.update_q_value(state, action, reward, next_state)
    
    def decay_epsilon(self, decay_rate: float = 0.995):
        """Decay exploration rate over time"""
        self.epsilon = max(0.01, self.epsilon * decay_rate)
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            'q_table_size': len(self.q_table),
            'experience_buffer_size': len(self.experience_buffer),
            'current_epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'total_states_explored': len(self.q_table)
        }


class ConfidenceTracker:
    """Track and manage worker confidence scores"""
    
    def __init__(self):
        self.logger = structlog.get_logger("ConfidenceTracker")
        self.worker_confidences: Dict[str, ConfidenceMetrics] = {}
        
    def update_confidence(self, worker_id: str, confidence_metrics: ConfidenceMetrics):
        """Update confidence metrics for a worker"""
        self.worker_confidences[worker_id] = confidence_metrics
        
    def get_worker_confidence(self, worker_id: str) -> ConfidenceMetrics:
        """Get confidence metrics for a worker"""
        return self.worker_confidences.get(worker_id, ConfidenceMetrics(worker_id=worker_id))
        
    def get_task_confidence(self, worker_id: str, task_type: str) -> float:
        """Get confidence score for a specific task type"""
        worker_confidence = self.get_worker_confidence(worker_id)
        return worker_confidence.task_type_confidences.get(task_type, 0.5)
        
    def get_top_confident_workers(self, task_type: str, limit: int = 5) -> List[str]:
        """Get workers with highest confidence for a task type"""
        worker_scores = []
        
        for worker_id, confidence in self.worker_confidences.items():
            task_confidence = confidence.task_type_confidences.get(task_type, 0.5)
            worker_scores.append((worker_id, task_confidence))
        
        worker_scores.sort(key=lambda x: x[1], reverse=True)
        return [worker_id for worker_id, _ in worker_scores[:limit]]


class LearningFeedbackLoop:
    """Process feedback and apply learning improvements"""
    
    def __init__(self):
        self.logger = structlog.get_logger("LearningFeedbackLoop")
        self.feedback_history: List[Dict[str, Any]] = []
        self.learning_models: Dict[str, Any] = {}
        
    async def process_feedback(self, feedback_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process feedback data and generate improvement recommendations"""
        
        improvements = []
        
        # Analyze decision accuracy patterns
        decision_accuracy = self._analyze_decision_accuracy(feedback_data)
        if decision_accuracy < 0.7:
            improvements.append({
                'type': 'decision_threshold_adjustment',
                'description': 'Increase decision confidence threshold',
                'parameters': {'new_threshold': 0.8},
                'priority': 'high'
            })
        
        # Analyze resource allocation effectiveness
        resource_effectiveness = self._analyze_resource_effectiveness(feedback_data)
        if resource_effectiveness < 0.6:
            improvements.append({
                'type': 'resource_allocation_optimization',
                'description': 'Optimize resource allocation strategy',
                'parameters': {'allocation_factor': 0.9},
                'priority': 'medium'
            })
        
        # Analyze task preemption outcomes
        preemption_success = self._analyze_preemption_outcomes(feedback_data)
        if preemption_success < 0.5:
            improvements.append({
                'type': 'preemption_strategy_adjustment',
                'description': 'Adjust preemption scoring algorithm',
                'parameters': {'score_threshold': 0.8},
                'priority': 'medium'
            })
        
        # Store feedback for future analysis
        self.feedback_history.extend(feedback_data)
        if len(self.feedback_history) > 10000:
            self.feedback_history = self.feedback_history[-8000:]
        
        return improvements
    
    def _analyze_decision_accuracy(self, feedback_data: List[Dict[str, Any]]) -> float:
        """Analyze accuracy of autonomous decisions"""
        if not feedback_data:
            return 0.8  # Default assumption
            
        accurate_decisions = sum(1 for f in feedback_data if f.get('feedback_score', 0) > 0.7)
        return accurate_decisions / len(feedback_data)
    
    def _analyze_resource_effectiveness(self, feedback_data: List[Dict[str, Any]]) -> float:
        """Analyze effectiveness of resource allocation decisions"""
        resource_decisions = [f for f in feedback_data if f.get('decision_type') == 'resource_allocation']
        if not resource_decisions:
            return 0.8
            
        effective_decisions = sum(1 for f in resource_decisions if f.get('feedback_score', 0) > 0.6)
        return effective_decisions / len(resource_decisions)
    
    def _analyze_preemption_outcomes(self, feedback_data: List[Dict[str, Any]]) -> float:
        """Analyze outcomes of task preemption decisions"""
        preemption_decisions = [f for f in feedback_data if f.get('decision_type') == 'task_preemption']
        if not preemption_decisions:
            return 0.7
            
        successful_preemptions = sum(1 for f in preemption_decisions if f.get('feedback_score', 0) > 0.5)
        return successful_preemptions / len(preemption_decisions)


class BayesianTaskOptimizer:
    """Bayesian optimization for task scheduling and resource allocation"""
    
    def __init__(self):
        self.logger = structlog.get_logger("BayesianTaskOptimizer")
        self.optimization_history: List[Dict[str, Any]] = []
        self.parameter_bounds = {
            'cpu_threshold': (0.5, 0.95),
            'memory_threshold': (0.5, 0.95),
            'confidence_threshold': (0.3, 0.9),
            'preemption_score_threshold': (0.4, 0.9)
        }
        
    async def update_models(self, feedback_data: List[Dict[str, Any]]):
        """Update Bayesian optimization models with new feedback"""
        
        # Process feedback into optimization objectives
        objectives = self._extract_optimization_objectives(feedback_data)
        
        # Update parameter effectiveness estimates
        for objective in objectives:
            self.optimization_history.append(objective)
            
        # Limit history size
        if len(self.optimization_history) > 5000:
            self.optimization_history = self.optimization_history[-4000:]
            
        # Perform Bayesian parameter optimization (simplified)
        optimal_params = await self._optimize_parameters()
        
        self.logger.info("Updated Bayesian optimization models",
                        objectives_processed=len(objectives),
                        optimal_params=optimal_params)
    
    def _extract_optimization_objectives(self, feedback_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract optimization objectives from feedback data"""
        
        objectives = []
        
        for feedback in feedback_data:
            if 'context' in feedback and 'parameters' in feedback['context']:
                objective = {
                    'parameters': feedback['context']['parameters'],
                    'objective_value': feedback.get('feedback_score', 0.5),
                    'timestamp': datetime.utcnow().isoformat()
                }
                objectives.append(objective)
        
        return objectives
    
    async def _optimize_parameters(self) -> Dict[str, float]:
        """Perform Bayesian parameter optimization"""
        
        # Simplified Bayesian optimization - in production would use libraries like scikit-optimize
        optimal_params = {}
        
        for param_name, (min_val, max_val) in self.parameter_bounds.items():
            # Find parameter values with best historical performance
            param_performances = []
            
            for record in self.optimization_history[-1000:]:  # Recent history
                if param_name in record.get('parameters', {}):
                    param_value = record['parameters'][param_name]
                    objective_value = record['objective_value']
                    param_performances.append((param_value, objective_value))
            
            if param_performances:
                # Simple optimization: take weighted average of best performing values
                param_performances.sort(key=lambda x: x[1], reverse=True)
                top_performers = param_performances[:min(10, len(param_performances) // 4)]
                
                if top_performers:
                    weighted_sum = sum(value * performance for value, performance in top_performers)
                    weight_sum = sum(performance for _, performance in top_performers)
                    optimal_params[param_name] = weighted_sum / weight_sum
                else:
                    optimal_params[param_name] = (min_val + max_val) / 2
            else:
                optimal_params[param_name] = (min_val + max_val) / 2
        
        return optimal_params
    
    async def suggest_parameters(self, optimization_target: str) -> Dict[str, float]:
        """Suggest optimal parameters for a given optimization target"""
        
        # Get current optimal parameters
        optimal_params = await self._optimize_parameters()
        
        # Add exploration noise for better optimization
        exploration_factor = 0.1
        suggested_params = {}
        
        for param_name, optimal_value in optimal_params.items():
            min_val, max_val = self.parameter_bounds[param_name]
            noise = (max_val - min_val) * exploration_factor * (0.5 - time.time() % 1.0)
            suggested_value = max(min_val, min(max_val, optimal_value + noise))
            suggested_params[param_name] = suggested_value
        
        return suggested_params


class TaskPreemptor:
    """Manage task preemption with sophisticated strategies"""
    
    def __init__(self):
        self.logger = structlog.get_logger("TaskPreemptor")
        self.preemption_strategies: Dict[str, Callable] = {
            'priority_based': self._priority_based_preemption,
            'resource_based': self._resource_based_preemption,
            'confidence_based': self._confidence_based_preemption,
            'deadline_based': self._deadline_based_preemption
        }
        
    async def evaluate_preemption_opportunity(self, 
                                            waiting_tasks: List[AgentTask],
                                            running_tasks: List[AgentTask],
                                            system_state: Dict[str, Any]) -> List[PreemptionEvent]:
        """Evaluate preemption opportunities using multiple strategies"""
        
        preemption_events = []
        
        # Apply each preemption strategy
        for strategy_name, strategy_func in self.preemption_strategies.items():
            try:
                strategy_events = await strategy_func(waiting_tasks, running_tasks, system_state)
                preemption_events.extend(strategy_events)
            except Exception as e:
                self.logger.error(f"Preemption strategy {strategy_name} failed", error=str(e))
        
        # Deduplicate and rank preemption events
        unique_events = self._deduplicate_preemption_events(preemption_events)
        ranked_events = sorted(unique_events, key=lambda x: x.priority_score, reverse=True)
        
        return ranked_events
    
    async def _priority_based_preemption(self, waiting_tasks, running_tasks, system_state) -> List[PreemptionEvent]:
        """Preemption based on task priority differences"""
        events = []
        
        for waiting_task in waiting_tasks:
            for running_task in running_tasks:
                if waiting_task.priority > running_task.priority + 2:
                    events.append(PreemptionEvent(
                        task_id=running_task.task_id,
                        agent_id=running_task.agent_id,
                        reason=f"Priority preemption: {waiting_task.priority} > {running_task.priority}",
                        priority_score=0.8,
                        recovery_strategy="requeue_with_priority_boost"
                    ))
        
        return events
    
    async def _resource_based_preemption(self, waiting_tasks, running_tasks, system_state) -> List[PreemptionEvent]:
        """Preemption based on resource utilization"""
        events = []
        
        cpu_usage = system_state.get('cpu_usage', 0.0)
        memory_usage = system_state.get('memory_usage', 0.0)
        
        if cpu_usage > 0.85 or memory_usage > 0.85:
            # Find resource-intensive running tasks
            for running_task in running_tasks:
                estimated_cpu = running_task.parameters.get('estimated_cpu', 0.1)
                estimated_memory = running_task.parameters.get('estimated_memory', 0.05)
                
                if estimated_cpu > 0.2 or estimated_memory > 0.15:  # High resource tasks
                    events.append(PreemptionEvent(
                        task_id=running_task.task_id,
                        agent_id=running_task.agent_id,
                        reason=f"Resource pressure: CPU {cpu_usage:.2f}, Memory {memory_usage:.2f}",
                        priority_score=0.7,
                        recovery_strategy="retry_with_resource_limits"
                    ))
        
        return events
    
    async def _confidence_based_preemption(self, waiting_tasks, running_tasks, system_state) -> List[PreemptionEvent]:
        """Preemption based on worker confidence scores"""
        events = []
        
        confidence_tracker = system_state.get('confidence_tracker')
        if not confidence_tracker:
            return events
        
        for running_task in running_tasks:
            task_confidence = confidence_tracker.get_task_confidence(
                running_task.agent_id, running_task.task_type
            )
            
            if task_confidence < 0.3:  # Very low confidence
                events.append(PreemptionEvent(
                    task_id=running_task.task_id,
                    agent_id=running_task.agent_id,
                    reason=f"Low confidence task: {task_confidence:.2f}",
                    priority_score=0.6,
                    recovery_strategy="retry_with_high_confidence_agent"
                ))
        
        return events
    
    async def _deadline_based_preemption(self, waiting_tasks, running_tasks, system_state) -> List[PreemptionEvent]:
        """Preemption based on task deadlines"""
        events = []
        
        current_time = datetime.utcnow()
        
        for waiting_task in waiting_tasks:
            if waiting_task.deadline:
                time_to_deadline = (waiting_task.deadline - current_time).total_seconds()
                
                if time_to_deadline < 300:  # Less than 5 minutes to deadline
                    # Find preemptable running tasks
                    for running_task in running_tasks:
                        if (not running_task.deadline or 
                            running_task.deadline > waiting_task.deadline + timedelta(minutes=10)):
                            
                            events.append(PreemptionEvent(
                                task_id=running_task.task_id,
                                agent_id=running_task.agent_id,
                                reason=f"Deadline urgency: {time_to_deadline:.0f}s remaining",
                                priority_score=0.9,
                                recovery_strategy="requeue_with_extended_deadline"
                            ))
        
        return events
    
    def _deduplicate_preemption_events(self, events: List[PreemptionEvent]) -> List[PreemptionEvent]:
        """Remove duplicate preemption events"""
        seen_tasks = set()
        unique_events = []
        
        for event in events:
            if event.task_id not in seen_tasks:
                seen_tasks.add(event.task_id)
                unique_events.append(event)
        
        return unique_events


class ExecutionGraph:
    """Track task execution relationships and dependencies"""
    
    def __init__(self):
        self.logger = structlog.get_logger("ExecutionGraph")
        self.task_relationships: Dict[str, Set[str]] = defaultdict(set)
        self.execution_history: List[Dict[str, Any]] = []
        
    def add_task_relationship(self, parent_task_id: str, child_task_id: str):
        """Add dependency relationship between tasks"""
        self.task_relationships[parent_task_id].add(child_task_id)
        
    def get_dependent_tasks(self, task_id: str) -> Set[str]:
        """Get tasks that depend on the given task"""
        return self.task_relationships.get(task_id, set())
        
    def record_execution(self, task_id: str, execution_data: Dict[str, Any]):
        """Record task execution data"""
        execution_record = {
            'task_id': task_id,
            'timestamp': datetime.utcnow().isoformat(),
            **execution_data
        }
        self.execution_history.append(execution_record)
        
        # Limit history size
        if len(self.execution_history) > 10000:
            self.execution_history = self.execution_history[-8000:]


class EnhancedAutonomousMetrics:
    """Enhanced metrics for RL-enabled autonomous orchestrator"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # RL-specific metrics
        self.rl_rewards = Histogram(
            'xorb_rl_rewards',
            'RL agent rewards distribution',
            ['action_type'],
            registry=self.registry
        )
        
        self.rl_actions = Counter(
            'xorb_rl_actions_total',
            'Total RL actions taken',
            ['action', 'state'],
            registry=self.registry
        )
        
        self.confidence_scores = Histogram(
            'xorb_worker_confidence_scores',
            'Worker confidence score distribution',
            ['worker_id', 'task_type'],
            registry=self.registry
        )
        
        self.preemption_events = Counter(
            'xorb_task_preemptions_total',
            'Total task preemptions',
            ['reason', 'recovery_strategy'],
            registry=self.registry
        )
        
        self.learning_improvements = Counter(
            'xorb_learning_improvements_total',
            'Learning-based improvements applied',
            ['improvement_type'],
            registry=self.registry
        )
        
        self.bayesian_optimizations = Counter(
            'xorb_bayesian_optimizations_total',
            'Bayesian optimization cycles',
            ['optimization_target'],
            registry=self.registry
        )