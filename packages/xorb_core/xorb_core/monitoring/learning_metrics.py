"""
Learning Metrics Collector
Prometheus metrics for ML system observability
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
import prometheus_client

logger = logging.getLogger(__name__)

class MetricType(Enum):
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency_ms: float
    confidence: float

class LearningMetricsCollector:
    """Collects and exposes learning-specific Prometheus metrics"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or prometheus_client.REGISTRY
        self._initialize_metrics()
        
    def _initialize_metrics(self):
        """Initialize all learning-related Prometheus metrics"""
        
        # Model performance metrics
        self.model_predictions_total = Counter(
            'xorb_model_predictions_total',
            'Total number of model predictions',
            ['model_type', 'algorithm', 'environment'],
            registry=self.registry
        )
        
        self.model_prediction_duration = Histogram(
            'xorb_model_prediction_duration_seconds',
            'Time spent on model predictions',
            ['model_type', 'algorithm'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        self.agent_selection_accuracy = Gauge(
            'xorb_agent_selection_accuracy',
            'Accuracy of agent selection decisions',
            ['algorithm', 'time_window'],
            registry=self.registry
        )
        
        self.workflow_success_rate = Gauge(
            'xorb_workflow_success_rate',
            'Workflow execution success rate',
            ['template_id', 'environment'],
            registry=self.registry
        )
        
        # Learning system metrics
        self.policy_updates_total = Counter(
            'xorb_policy_updates_total',
            'Number of policy updates',
            ['algorithm', 'update_type'],
            registry=self.registry
        )
        
        self.training_loss = Gauge(
            'xorb_training_loss',
            'Current training loss',
            ['algorithm', 'loss_type'],
            registry=self.registry
        )
        
        self.model_performance_score = Gauge(
            'xorb_model_performance_score',
            'Model performance score (0-1)',
            ['model_id', 'metric_type'],
            registry=self.registry
        )
        
        # Bandit algorithm metrics
        self.bandit_selections_total = Counter(
            'xorb_bandit_selections_total',
            'Total agent selections by bandit algorithm',
            ['algorithm', 'agent_id'],
            registry=self.registry
        )
        
        self.bandit_reward = Histogram(
            'xorb_bandit_reward',
            'Reward values from bandit algorithm',
            ['algorithm', 'agent_id'],
            buckets=[-1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
            registry=self.registry
        )
        
        self.exploration_rate = Gauge(
            'xorb_exploration_rate',
            'Current exploration rate for bandit algorithms',
            ['algorithm'],
            registry=self.registry
        )
        
        # LLM usage metrics
        self.llm_requests_total = Counter(
            'xorb_llm_requests_total',
            'Total LLM requests',
            ['provider', 'model', 'priority'],
            registry=self.registry
        )
        
        self.llm_tokens_used_total = Counter(
            'xorb_llm_tokens_used_total',
            'Total LLM tokens consumed',
            ['provider', 'model', 'priority'],
            registry=self.registry
        )
        
        self.llm_request_duration = Histogram(
            'xorb_llm_request_duration_seconds',
            'Duration of LLM requests',
            ['provider', 'model'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        self.llm_cost_total = Counter(
            'xorb_llm_cost_usd_total',
            'Total LLM costs in USD',
            ['provider', 'model'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits_total = Counter(
            'xorb_cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses_total = Counter(
            'xorb_cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_size = Gauge(
            'xorb_cache_size',
            'Current cache size',
            ['cache_type'],
            registry=self.registry
        )
        
        # Experience store metrics
        self.experience_records_total = Counter(
            'xorb_experience_records_total',
            'Total experience records stored',
            ['record_type', 'agent_id'],
            registry=self.registry
        )
        
        self.experience_store_latency = Histogram(
            'xorb_experience_store_latency_seconds',
            'Experience store operation latency',
            ['operation'],
            registry=self.registry
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            'xorb_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['service_name'],
            registry=self.registry
        )
        
        self.circuit_breaker_failures = Counter(
            'xorb_circuit_breaker_failures_total',
            'Total circuit breaker failures',
            ['service_name'],
            registry=self.registry
        )
        
        # Drift detection metrics
        self.concept_drift_detected = Counter(
            'xorb_concept_drift_detected_total',
            'Number of concept drift detections',
            ['detector_type', 'severity'],
            registry=self.registry
        )
        
        self.drift_detection_score = Gauge(
            'xorb_drift_detection_score',
            'Current drift detection score',
            ['detector_type'],
            registry=self.registry
        )
        
        logger.info("Initialized learning metrics collector")
    
    async def record_model_prediction(self, model_type: str, algorithm: str, 
                                    environment: str, duration_seconds: float):
        """Record model prediction metrics"""
        self.model_predictions_total.labels(
            model_type=model_type,
            algorithm=algorithm,
            environment=environment
        ).inc()
        
        self.model_prediction_duration.labels(
            model_type=model_type,
            algorithm=algorithm
        ).observe(duration_seconds)
    
    async def record_agent_selection_accuracy(self, algorithm: str, 
                                            time_window: str, accuracy: float):
        """Record agent selection accuracy"""
        self.agent_selection_accuracy.labels(
            algorithm=algorithm,
            time_window=time_window
        ).set(accuracy)
    
    async def record_workflow_success_rate(self, template_id: str, 
                                         environment: str, success_rate: float):
        """Record workflow success rate"""
        self.workflow_success_rate.labels(
            template_id=template_id,
            environment=environment
        ).set(success_rate)
    
    async def record_policy_update(self, algorithm: str, update_type: str):
        """Record policy update event"""
        self.policy_updates_total.labels(
            algorithm=algorithm,
            update_type=update_type
        ).inc()
    
    async def record_training_metrics(self, metrics: Dict[str, float]):
        """Record training metrics"""
        for algorithm, losses in metrics.items():
            if isinstance(losses, dict):
                for loss_type, value in losses.items():
                    self.training_loss.labels(
                        algorithm=algorithm,
                        loss_type=loss_type
                    ).set(value)
            else:
                self.training_loss.labels(
                    algorithm=algorithm,
                    loss_type='total'
                ).set(losses)
    
    async def record_model_performance(self, model_id: str, metrics: ModelMetrics):
        """Record model performance metrics"""
        performance_metrics = {
            'accuracy': metrics.accuracy,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1_score': metrics.f1_score,
            'confidence': metrics.confidence
        }
        
        for metric_type, value in performance_metrics.items():
            self.model_performance_score.labels(
                model_id=model_id,
                metric_type=metric_type
            ).set(value)
    
    async def record_bandit_selection(self, agent_id: str, context: Dict[str, Any], 
                                    algorithm: str):
        """Record bandit algorithm selection"""
        self.bandit_selections_total.labels(
            algorithm=algorithm,
            agent_id=agent_id
        ).inc()
    
    async def record_bandit_reward(self, agent_id: str, reward: float, algorithm: str):
        """Record bandit algorithm reward"""
        self.bandit_reward.labels(
            algorithm=algorithm,
            agent_id=agent_id
        ).observe(reward)
    
    async def record_exploration_rate(self, algorithm: str, rate: float):
        """Record current exploration rate"""
        self.exploration_rate.labels(algorithm=algorithm).set(rate)
    
    async def record_llm_request(self, provider: str, model: str, priority: str,
                               tokens_used: int, duration_seconds: float, cost_usd: float):
        """Record LLM request metrics"""
        self.llm_requests_total.labels(
            provider=provider,
            model=model,
            priority=priority
        ).inc()
        
        self.llm_tokens_used_total.labels(
            provider=provider,
            model=model,
            priority=priority
        ).inc(tokens_used)
        
        self.llm_request_duration.labels(
            provider=provider,
            model=model
        ).observe(duration_seconds)
        
        self.llm_cost_total.labels(
            provider=provider,
            model=model
        ).inc(cost_usd)
    
    async def record_cache_operation(self, cache_type: str, hit: bool, size: int):
        """Record cache operation metrics"""
        if hit:
            self.cache_hits_total.labels(cache_type=cache_type).inc()
        else:
            self.cache_misses_total.labels(cache_type=cache_type).inc()
        
        self.cache_size.labels(cache_type=cache_type).set(size)
    
    async def record_experience_stored(self, record_type: str, agent_id: str):
        """Record experience storage event"""
        self.experience_records_total.labels(
            record_type=record_type,
            agent_id=agent_id
        ).inc()
    
    async def record_experience_store_latency(self, operation: str, duration_seconds: float):
        """Record experience store operation latency"""
        self.experience_store_latency.labels(operation=operation).observe(duration_seconds)
    
    async def record_circuit_breaker_state(self, service_name: str, state: str):
        """Record circuit breaker state"""
        state_mapping = {'closed': 0, 'open': 1, 'half_open': 2}
        state_value = state_mapping.get(state, 0)
        
        self.circuit_breaker_state.labels(service_name=service_name).set(state_value)
    
    async def record_circuit_breaker_failure(self, service_name: str):
        """Record circuit breaker failure"""
        self.circuit_breaker_failures.labels(service_name=service_name).inc()
    
    async def record_concept_drift(self, detector_type: str, severity: str, score: float):
        """Record concept drift detection"""
        self.concept_drift_detected.labels(
            detector_type=detector_type,
            severity=severity
        ).inc()
        
        self.drift_detection_score.labels(detector_type=detector_type).set(score)
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values"""
        try:
            # Generate metrics in Prometheus format
            metrics_data = generate_latest(self.registry).decode('utf-8')
            
            # Parse and return structured data
            metrics_dict = {}
            for line in metrics_data.split('\n'):
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    metric_name = parts[0]
                    metric_value = parts[1]
                    
                    # Extract base metric name (remove labels)
                    base_name = metric_name.split('{')[0]
                    if base_name not in metrics_dict:
                        metrics_dict[base_name] = []
                    
                    metrics_dict[base_name].append({
                        'labels': metric_name,
                        'value': float(metric_value)
                    })
            
            return metrics_dict
            
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return {}
    
    async def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning system performance"""
        try:
            summary = {
                'timestamp': time.time(),
                'model_predictions': {},
                'policy_updates': {},
                'llm_usage': {},
                'cache_performance': {},
                'system_health': {}
            }
            
            # Get current metric values
            current_metrics = await self.get_current_metrics()
            
            # Extract key metrics for summary
            for metric_name, metric_data in current_metrics.items():
                if 'model_predictions_total' in metric_name:
                    total_predictions = sum(item['value'] for item in metric_data)
                    summary['model_predictions']['total'] = total_predictions
                
                elif 'policy_updates_total' in metric_name:
                    total_updates = sum(item['value'] for item in metric_data)
                    summary['policy_updates']['total'] = total_updates
                
                elif 'llm_tokens_used_total' in metric_name:
                    total_tokens = sum(item['value'] for item in metric_data)
                    summary['llm_usage']['total_tokens'] = total_tokens
                
                elif 'cache_hits_total' in metric_name:
                    cache_hits = sum(item['value'] for item in metric_data)
                    summary['cache_performance']['hits'] = cache_hits
                
                elif 'cache_misses_total' in metric_name:
                    cache_misses = sum(item['value'] for item in metric_data)
                    summary['cache_performance']['misses'] = cache_misses
            
            # Calculate derived metrics
            if 'hits' in summary['cache_performance'] and 'misses' in summary['cache_performance']:
                total_requests = summary['cache_performance']['hits'] + summary['cache_performance']['misses']
                if total_requests > 0:
                    summary['cache_performance']['hit_rate'] = summary['cache_performance']['hits'] / total_requests
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting learning summary: {e}")
            return {'error': str(e)}
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')
    
    async def reset_metrics(self):
        """Reset all metrics (for testing)"""
        logger.warning("Resetting all learning metrics")
        
        # Clear the registry and reinitialize
        self.registry._collector_to_names.clear()
        self.registry._names_to_collectors.clear()
        self._initialize_metrics()
    
    async def export_metrics_to_file(self, filepath: str):
        """Export current metrics to file"""
        try:
            metrics_data = self.get_prometheus_metrics()
            with open(filepath, 'w') as f:
                f.write(metrics_data)
            logger.info(f"Exported metrics to {filepath}")
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            raise