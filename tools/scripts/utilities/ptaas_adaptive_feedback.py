#!/usr/bin/env python3
"""
XORB PTaaS Adaptive Feedback System
Dynamic agent configuration and optimization based on learning insights
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import numpy as np
from collections import defaultdict, deque
import hashlib

import aiohttp
import aiokafka
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
import circuit_breaker
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="XORB PTaaS Adaptive Feedback System",
    description="Dynamic agent optimization based on learning insights and performance data",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FeedbackType(Enum):
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    STRATEGY_ADJUSTMENT = "strategy_adjustment" 
    THRESHOLD_TUNING = "threshold_tuning"
    RESOURCE_SCALING = "resource_scaling"
    PRIORITY_REBALANCING = "priority_rebalancing"
    ERROR_CORRECTION = "error_correction"

class OptimizationGoal(Enum):
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    MINIMIZE_FALSE_POSITIVES = "minimize_false_positives"
    OPTIMIZE_SPEED = "optimize_speed"
    BALANCE_COVERAGE = "balance_coverage"
    REDUCE_RESOURCE_USAGE = "reduce_resource_usage"

@dataclass
class AgentConfiguration:
    agent_id: str
    config_version: str
    parameters: Dict[str, Any]
    performance_targets: Dict[str, float]
    constraints: Dict[str, Any]
    last_updated: datetime
    applied_optimizations: List[str] = field(default_factory=list)

@dataclass
class FeedbackAction:
    action_id: str
    feedback_type: FeedbackType
    target_agent: str
    optimization_goal: OptimizationGoal
    parameters: Dict[str, Any]
    expected_impact: Dict[str, float]
    confidence_score: float
    priority: float
    created_at: datetime
    applied_at: Optional[datetime] = None
    rollback_data: Optional[Dict[str, Any]] = None

@dataclass
class PerformanceSnapshot:
    agent_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    configuration_hash: str
    test_context: Dict[str, Any]
    outcomes: Dict[str, Any]

class MLOptimizer:
    """Machine learning optimizer for agent configuration tuning"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            'success_rate', 'false_positive_rate', 'avg_detection_time',
            'cpu_usage', 'memory_usage', 'scan_depth', 'timeout_value',
            'parallel_workers', 'retry_count', 'confidence_threshold'
        ]
        self.target_columns = [
            'accuracy_improvement', 'speed_improvement', 'resource_efficiency'
        ]
        
        # Initialize models for each agent
        self.agent_ids = [
            "AGENT-WEB-SCANNER-001",
            "AGENT-NETWORK-RECON-002", 
            "AGENT-VULN-ASSESSMENT-003",
            "AGENT-EXPLOITATION-004",
            "AGENT-DB-TESTER-005"
        ]
        
        for agent_id in self.agent_ids:
            self.models[agent_id] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.scalers[agent_id] = StandardScaler()
        
        self.training_data = defaultdict(list)
        self.is_trained = defaultdict(bool)
    
    def add_training_sample(self, agent_id: str, features: Dict[str, float], 
                          targets: Dict[str, float]):
        """Add training sample for agent optimization model"""
        if agent_id not in self.agent_ids:
            return
        
        feature_vector = [features.get(col, 0.0) for col in self.feature_columns]
        target_vector = [targets.get(col, 0.0) for col in self.target_columns]
        
        self.training_data[agent_id].append({
            'features': feature_vector,
            'targets': target_vector,
            'timestamp': datetime.now()
        })
        
        # Retrain if we have enough samples
        if len(self.training_data[agent_id]) >= 50:
            asyncio.create_task(self._retrain_model(agent_id))
    
    async def _retrain_model(self, agent_id: str):
        """Retrain optimization model for specific agent"""
        try:
            samples = self.training_data[agent_id]
            if len(samples) < 10:
                return
            
            # Prepare training data
            X = np.array([sample['features'] for sample in samples])
            y = np.array([sample['targets'] for sample in samples])
            
            # Scale features
            X_scaled = self.scalers[agent_id].fit_transform(X)
            
            # Train model
            self.models[agent_id].fit(X_scaled, y)
            self.is_trained[agent_id] = True
            
            logger.info(f"Retrained optimization model for {agent_id} with {len(samples)} samples")
            
        except Exception as e:
            logger.error(f"Error retraining model for {agent_id}: {e}")
    
    def predict_optimization(self, agent_id: str, current_config: Dict[str, float], 
                           proposed_changes: Dict[str, float]) -> Dict[str, float]:
        """Predict impact of configuration changes"""
        if agent_id not in self.models or not self.is_trained[agent_id]:
            # Return default predictions if model not trained
            return {
                'accuracy_improvement': 0.05,
                'speed_improvement': 0.02,
                'resource_efficiency': 0.03
            }
        
        try:
            # Create feature vector with proposed changes
            modified_config = current_config.copy()
            modified_config.update(proposed_changes)
            
            feature_vector = np.array([[modified_config.get(col, 0.0) for col in self.feature_columns]])
            feature_vector_scaled = self.scalers[agent_id].transform(feature_vector)
            
            # Predict improvements
            predictions = self.models[agent_id].predict(feature_vector_scaled)[0]
            
            return {
                target: max(0.0, pred) for target, pred in zip(self.target_columns, predictions)
            }
            
        except Exception as e:
            logger.error(f"Error predicting optimization for {agent_id}: {e}")
            return {'accuracy_improvement': 0.0, 'speed_improvement': 0.0, 'resource_efficiency': 0.0}

class AdaptiveFeedbackEngine:
    """Core adaptive feedback engine for PTaaS agent optimization"""
    
    def __init__(self):
        # Connections
        self.redis_client = None
        self.kafka_producer = None
        self.kafka_consumer = None
        
        # Components
        self.ml_optimizer = MLOptimizer()
        
        # State tracking
        self.agent_configurations: Dict[str, AgentConfiguration] = {}
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.active_feedback_actions: Dict[str, FeedbackAction] = {}
        self.optimization_queue = asyncio.Queue(maxsize=1000)
        
        # Circuit breakers
        self.circuit_breakers = {
            'kafka': circuit_breaker.CircuitBreaker(failure_threshold=5, recovery_timeout=60),
            'redis': circuit_breaker.CircuitBreaker(failure_threshold=3, recovery_timeout=30),
            'ptaas_api': circuit_breaker.CircuitBreaker(failure_threshold=10, recovery_timeout=120)
        }
        
        # Metrics
        self.metrics_registry = CollectorRegistry()
        self._setup_metrics()
        
        # Background tasks
        self.worker_tasks = []
        self.is_running = False
        
        # Initialize agent configurations
        self._initialize_agent_configurations()
    
    def _setup_metrics(self):
        """Setup Prometheus metrics for feedback system"""
        self.feedback_actions_counter = Counter(
            'ptaas_feedback_actions_total',
            'Total feedback actions generated',
            ['feedback_type', 'agent_id', 'optimization_goal'],
            registry=self.metrics_registry
        )
        
        self.optimization_latency = Histogram(
            'ptaas_optimization_latency_seconds',
            'Time taken for optimization decisions',
            ['agent_id'],
            registry=self.metrics_registry
        )
        
        self.performance_improvement = Gauge(
            'ptaas_performance_improvement',
            'Measured performance improvement after optimization',
            ['agent_id', 'metric_type'],
            registry=self.metrics_registry
        )
        
        self.configuration_changes = Counter(
            'ptaas_configuration_changes_total',
            'Total configuration changes applied',
            ['agent_id', 'parameter_type'],
            registry=self.metrics_registry
        )
        
        self.feedback_effectiveness = Gauge(
            'ptaas_feedback_effectiveness',
            'Effectiveness score of feedback actions',
            ['agent_id', 'feedback_type'],
            registry=self.metrics_registry
        )
    
    def _initialize_agent_configurations(self):
        """Initialize default configurations for all PTaaS agents"""
        default_configs = {
            "AGENT-WEB-SCANNER-001": {
                "scan_depth": 3,
                "timeout_seconds": 300,
                "parallel_workers": 4,
                "retry_count": 2,
                "confidence_threshold": 0.7,
                "rate_limit_per_second": 10,
                "max_payloads_per_check": 50,
                "follow_redirects": True,
                "scan_forms": True,
                "scan_cookies": True
            },
            "AGENT-NETWORK-RECON-002": {
                "port_range": "1-65535",
                "scan_intensity": "normal",
                "timeout_seconds": 180,
                "parallel_workers": 8,
                "retry_count": 1,
                "os_detection": True,
                "service_detection": True,
                "version_detection": True,
                "stealth_mode": False
            },
            "AGENT-VULN-ASSESSMENT-003": {
                "scan_depth": 5,
                "timeout_seconds": 600,
                "parallel_workers": 6,
                "false_positive_reduction": True,
                "confidence_threshold": 0.8,
                "exploit_verification": True,
                "severity_threshold": "medium",
                "max_vulnerabilities": 1000
            },
            "AGENT-EXPLOITATION-004": {
                "exploit_timeout": 120,
                "parallel_workers": 2,
                "safety_checks": True,
                "auto_exploit": False,
                "payload_customization": True,
                "success_threshold": 0.9,
                "rollback_on_failure": True,
                "evidence_collection": True
            },
            "AGENT-DB-TESTER-005": {
                "connection_timeout": 30,
                "query_timeout": 60,
                "parallel_connections": 5,
                "injection_payloads": 100,
                "blind_sql_detection": True,
                "time_based_detection": True,
                "error_based_detection": True,
                "boolean_based_detection": True
            }
        }
        
        for agent_id, config in default_configs.items():
            self.agent_configurations[agent_id] = AgentConfiguration(
                agent_id=agent_id,
                config_version="1.0.0",
                parameters=config,
                performance_targets={
                    "success_rate": 0.85,
                    "false_positive_rate": 0.15,
                    "avg_detection_time": 120.0,
                    "resource_utilization": 0.7
                },
                constraints={
                    "max_memory_mb": 2048,
                    "max_cpu_percent": 80,
                    "max_runtime_minutes": 60
                },
                last_updated=datetime.now()
            )
    
    async def initialize(self):
        """Initialize all connections and components"""
        try:
            # Redis connection
            self.redis_client = redis.from_url(
                "redis://localhost:6379",
                decode_responses=True,
                max_connections=20
            )
            await self.redis_client.ping()
            logger.info("Redis connection established for adaptive feedback")
            
            # Kafka connections
            self.kafka_producer = aiokafka.AIOKafkaProducer(
                bootstrap_servers='localhost:9092',
                value_serializer=lambda x: json.dumps(x, default=str).encode()
            )
            await self.kafka_producer.start()
            
            self.kafka_consumer = aiokafka.AIOKafkaConsumer(
                'ptaas-performance-metrics',
                'ptaas-learning-insights',
                bootstrap_servers='localhost:9092',
                value_deserializer=lambda x: json.loads(x.decode())
            )
            await self.kafka_consumer.start()
            
            logger.info("Kafka connections established for adaptive feedback")
            
        except Exception as e:
            logger.error(f"Failed to initialize adaptive feedback system: {e}")
            raise
    
    async def start_feedback_engine(self):
        """Start the adaptive feedback engine with worker tasks"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start worker tasks
        self.worker_tasks = [
            asyncio.create_task(self._performance_monitoring_worker()),
            asyncio.create_task(self._optimization_worker()),
            asyncio.create_task(self._feedback_application_worker()),
            asyncio.create_task(self._kafka_consumer_worker()),
            asyncio.create_task(self._performance_analysis_worker())
        ]
        
        logger.info("Adaptive feedback engine started with 5 worker tasks")
    
    async def stop_feedback_engine(self):
        """Stop the adaptive feedback engine gracefully"""
        self.is_running = False
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        logger.info("Adaptive feedback engine stopped gracefully")
    
    async def _performance_monitoring_worker(self):
        """Worker task for monitoring agent performance"""
        logger.info("Starting performance monitoring worker")
        
        while self.is_running:
            try:
                # Collect performance data for all agents
                for agent_id in self.agent_configurations.keys():
                    await self._collect_agent_performance(agent_id)
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitoring worker: {e}")
                await asyncio.sleep(60)
    
    async def _optimization_worker(self):
        """Worker task for processing optimization requests"""
        logger.info("Starting optimization worker")
        
        while self.is_running:
            try:
                # Get optimization request from queue
                optimization_request = await asyncio.wait_for(
                    self.optimization_queue.get(),
                    timeout=5.0
                )
                
                # Process optimization
                await self._process_optimization_request(optimization_request)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in optimization worker: {e}")
                await asyncio.sleep(5)
    
    async def _feedback_application_worker(self):
        """Worker task for applying feedback actions"""
        logger.info("Starting feedback application worker")
        
        while self.is_running:
            try:
                # Check for pending feedback actions
                pending_actions = [
                    action for action in self.active_feedback_actions.values()
                    if action.applied_at is None
                ]
                
                # Sort by priority and apply highest priority actions
                pending_actions.sort(key=lambda x: x.priority, reverse=True)
                
                for action in pending_actions[:5]:  # Process up to 5 actions per cycle
                    await self._apply_feedback_action(action)
                
                await asyncio.sleep(10)  # Apply feedback every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in feedback application worker: {e}")
                await asyncio.sleep(30)
    
    async def _kafka_consumer_worker(self):
        """Worker task for consuming Kafka messages"""
        logger.info("Starting Kafka consumer worker for adaptive feedback")
        
        try:
            async for message in self.kafka_consumer:
                try:
                    topic = message.topic
                    data = message.value
                    
                    if topic == 'ptaas-performance-metrics':
                        await self._handle_performance_metric(data)
                    elif topic == 'ptaas-learning-insights':
                        await self._handle_learning_insight(data)
                        
                except Exception as e:
                    logger.error(f"Error processing Kafka message: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Kafka consumer error in adaptive feedback: {e}")
            # Implement exponential backoff retry
            await asyncio.sleep(10)
            if self.is_running:
                await self._kafka_consumer_worker()
    
    async def _performance_analysis_worker(self):
        """Worker task for analyzing performance trends and generating insights"""
        logger.info("Starting performance analysis worker")
        
        while self.is_running:
            try:
                # Analyze performance trends for each agent
                for agent_id in self.agent_configurations.keys():
                    await self._analyze_agent_performance_trends(agent_id)
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance analysis worker: {e}")
                await asyncio.sleep(300)
    
    async def _collect_agent_performance(self, agent_id: str):
        """Collect current performance metrics for an agent"""
        try:
            # Get performance data from PTaaS orchestrator
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:8084/api/v1/agents/{agent_id}/performance") as response:
                    if response.status == 200:
                        performance_data = await response.json()
                        
                        # Create performance snapshot
                        snapshot = PerformanceSnapshot(
                            agent_id=agent_id,
                            timestamp=datetime.now(),
                            metrics=performance_data.get('metrics', {}),
                            configuration_hash=self._get_config_hash(agent_id),
                            test_context=performance_data.get('context', {}),
                            outcomes=performance_data.get('outcomes', {})
                        )
                        
                        # Add to history
                        self.performance_history[agent_id].append(snapshot)
                        
                        # Store in Redis for learning engine
                        await self._store_performance_data(snapshot)
                        
        except Exception as e:
            logger.error(f"Error collecting performance for {agent_id}: {e}")
    
    def _get_config_hash(self, agent_id: str) -> str:
        """Generate hash of current agent configuration"""
        if agent_id not in self.agent_configurations:
            return ""
        
        config = self.agent_configurations[agent_id]
        config_str = json.dumps(config.parameters, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    async def _store_performance_data(self, snapshot: PerformanceSnapshot):
        """Store performance snapshot in Redis"""
        if not self.redis_client:
            return
        
        try:
            data = {
                'agent_id': snapshot.agent_id,
                'timestamp': snapshot.timestamp.isoformat(),
                'metrics': snapshot.metrics,
                'configuration_hash': snapshot.configuration_hash,
                'test_context': snapshot.test_context,
                'outcomes': snapshot.outcomes
            }
            
            await self.redis_client.lpush(
                f'ptaas:performance:{snapshot.agent_id}',
                json.dumps(data)
            )
            await self.redis_client.expire(f'ptaas:performance:{snapshot.agent_id}', 86400)  # 24 hours
            
        except Exception as e:
            logger.error(f"Error storing performance data: {e}")
    
    async def _handle_performance_metric(self, data: Dict[str, Any]):
        """Handle incoming performance metric from Kafka"""
        try:
            agent_id = data.get('agent_id')
            if not agent_id or agent_id not in self.agent_configurations:
                return
            
            # Create performance snapshot from metric data
            snapshot = PerformanceSnapshot(
                agent_id=agent_id,
                timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
                metrics=data.get('metrics', {}),
                configuration_hash=data.get('configuration_hash', ''),
                test_context=data.get('context', {}),
                outcomes=data.get('outcomes', {})
            )
            
            # Add to history
            self.performance_history[agent_id].append(snapshot)
            
            # Check if optimization is needed
            if await self._should_optimize_agent(agent_id, snapshot):
                await self.optimization_queue.put({
                    'type': 'performance_optimization',
                    'agent_id': agent_id,
                    'snapshot': snapshot,
                    'timestamp': datetime.now()
                })
                
        except Exception as e:
            logger.error(f"Error handling performance metric: {e}")
    
    async def _handle_learning_insight(self, data: Dict[str, Any]):
        """Handle learning insight from the learning engine"""
        try:
            insight_type = data.get('insight_type')
            agent_id = data.get('agent_id')
            
            if not agent_id or agent_id not in self.agent_configurations:
                return
            
            # Generate feedback action based on insight
            feedback_action = await self._generate_feedback_from_insight(data)
            
            if feedback_action:
                self.active_feedback_actions[feedback_action.action_id] = feedback_action
                
                # Update metrics
                self.feedback_actions_counter.labels(
                    feedback_type=feedback_action.feedback_type.value,
                    agent_id=agent_id,
                    optimization_goal=feedback_action.optimization_goal.value
                ).inc()
                
        except Exception as e:
            logger.error(f"Error handling learning insight: {e}")
    
    async def _should_optimize_agent(self, agent_id: str, snapshot: PerformanceSnapshot) -> bool:
        """Determine if agent needs optimization based on performance"""
        if len(self.performance_history[agent_id]) < 5:
            return False  # Need sufficient history
        
        config = self.agent_configurations[agent_id]
        current_metrics = snapshot.metrics
        
        # Check if performance is below targets
        performance_issues = []
        
        if current_metrics.get('success_rate', 1.0) < config.performance_targets['success_rate']:
            performance_issues.append('low_success_rate')
        
        if current_metrics.get('false_positive_rate', 0.0) > config.performance_targets['false_positive_rate']:
            performance_issues.append('high_false_positive_rate')
        
        if current_metrics.get('avg_detection_time', 0.0) > config.performance_targets['avg_detection_time']:
            performance_issues.append('slow_detection')
        
        # Check for performance degradation trend
        recent_snapshots = list(self.performance_history[agent_id])[-5:]
        if len(recent_snapshots) >= 3:
            success_rates = [s.metrics.get('success_rate', 0.0) for s in recent_snapshots]
            if len(success_rates) >= 3 and all(success_rates[i] >= success_rates[i+1] for i in range(len(success_rates)-1)):
                performance_issues.append('declining_performance')
        
        return len(performance_issues) > 0
    
    async def _process_optimization_request(self, request: Dict[str, Any]):
        """Process optimization request and generate feedback actions"""
        start_time = time.time()
        
        try:
            agent_id = request['agent_id']
            optimization_type = request['type']
            
            # Generate optimization recommendations
            feedback_actions = await self._generate_optimization_feedback(request)
            
            # Store feedback actions
            for action in feedback_actions:
                self.active_feedback_actions[action.action_id] = action
            
            # Record processing latency
            self.optimization_latency.labels(agent_id=agent_id).observe(
                time.time() - start_time
            )
            
            logger.info(f"Generated {len(feedback_actions)} optimization actions for {agent_id}")
            
        except Exception as e:
            logger.error(f"Error processing optimization request: {e}")
    
    async def _generate_optimization_feedback(self, request: Dict[str, Any]) -> List[FeedbackAction]:
        """Generate optimization feedback actions based on request"""
        agent_id = request['agent_id']
        snapshot = request.get('snapshot')
        
        if not snapshot or agent_id not in self.agent_configurations:
            return []
        
        actions = []
        current_config = self.agent_configurations[agent_id]
        current_metrics = snapshot.metrics
        
        # Generate different types of optimization actions
        
        # 1. Threshold tuning for false positive reduction
        if current_metrics.get('false_positive_rate', 0.0) > 0.2:
            actions.append(FeedbackAction(
                action_id=str(uuid.uuid4()),
                feedback_type=FeedbackType.THRESHOLD_TUNING,
                target_agent=agent_id,
                optimization_goal=OptimizationGoal.MINIMIZE_FALSE_POSITIVES,
                parameters={
                    'confidence_threshold': min(0.9, current_config.parameters.get('confidence_threshold', 0.7) + 0.1)
                },
                expected_impact={'false_positive_reduction': 0.3, 'accuracy_improvement': 0.15},
                confidence_score=0.8,
                priority=0.85,
                created_at=datetime.now()
            ))
        
        # 2. Performance optimization for slow agents
        if current_metrics.get('avg_detection_time', 0.0) > current_config.performance_targets['avg_detection_time']:
            actions.append(FeedbackAction(
                action_id=str(uuid.uuid4()),
                feedback_type=FeedbackType.PERFORMANCE_OPTIMIZATION,
                target_agent=agent_id,
                optimization_goal=OptimizationGoal.OPTIMIZE_SPEED,
                parameters={
                    'parallel_workers': min(16, current_config.parameters.get('parallel_workers', 4) + 2),
                    'timeout_seconds': max(60, current_config.parameters.get('timeout_seconds', 300) - 30)
                },
                expected_impact={'speed_improvement': 0.25, 'resource_efficiency': -0.1},
                confidence_score=0.75,
                priority=0.7,
                created_at=datetime.now()
            ))
        
        # 3. Resource scaling based on utilization
        resource_utilization = current_metrics.get('resource_utilization', 0.5)
        if resource_utilization < 0.3:  # Under-utilized
            actions.append(FeedbackAction(
                action_id=str(uuid.uuid4()),
                feedback_type=FeedbackType.RESOURCE_SCALING,
                target_agent=agent_id,
                optimization_goal=OptimizationGoal.BALANCE_COVERAGE,
                parameters={
                    'scan_depth': min(10, current_config.parameters.get('scan_depth', 3) + 1),
                    'parallel_workers': min(12, current_config.parameters.get('parallel_workers', 4) + 1)
                },
                expected_impact={'coverage_improvement': 0.2, 'resource_efficiency': -0.15},
                confidence_score=0.65,
                priority=0.6,
                created_at=datetime.now()
            ))
        elif resource_utilization > 0.8:  # Over-utilized
            actions.append(FeedbackAction(
                action_id=str(uuid.uuid4()),
                feedback_type=FeedbackType.RESOURCE_SCALING,
                target_agent=agent_id,
                optimization_goal=OptimizationGoal.REDUCE_RESOURCE_USAGE,
                parameters={
                    'parallel_workers': max(1, current_config.parameters.get('parallel_workers', 4) - 1),
                    'scan_depth': max(1, current_config.parameters.get('scan_depth', 3) - 1)
                },
                expected_impact={'resource_efficiency': 0.3, 'coverage_reduction': -0.1},
                confidence_score=0.7,
                priority=0.75,
                created_at=datetime.now()
            ))
        
        return actions
    
    async def _generate_feedback_from_insight(self, insight_data: Dict[str, Any]) -> Optional[FeedbackAction]:
        """Generate feedback action from learning engine insight"""
        try:
            insight_type = insight_data.get('insight_type')
            agent_id = insight_data.get('agent_id')
            confidence = insight_data.get('confidence', 0.5)
            
            if insight_type == 'strategy_optimization':
                return FeedbackAction(
                    action_id=str(uuid.uuid4()),
                    feedback_type=FeedbackType.STRATEGY_ADJUSTMENT,
                    target_agent=agent_id,
                    optimization_goal=OptimizationGoal.MAXIMIZE_ACCURACY,
                    parameters=insight_data.get('recommended_parameters', {}),
                    expected_impact=insight_data.get('expected_impact', {}),
                    confidence_score=confidence,
                    priority=confidence * 0.9,
                    created_at=datetime.now()
                )
            
            elif insight_type == 'performance_degradation':
                return FeedbackAction(
                    action_id=str(uuid.uuid4()),
                    feedback_type=FeedbackType.ERROR_CORRECTION,
                    target_agent=agent_id,
                    optimization_goal=OptimizationGoal.MAXIMIZE_ACCURACY,
                    parameters={
                        'reset_to_baseline': True,
                        'increase_monitoring': True
                    },
                    expected_impact={'stability_improvement': 0.4},
                    confidence_score=confidence,
                    priority=0.95,  # High priority for error correction
                    created_at=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating feedback from insight: {e}")
            return None
    
    @circuit_breaker.circuit
    async def _apply_feedback_action(self, action: FeedbackAction):
        """Apply feedback action to target agent"""
        try:
            agent_id = action.target_agent
            
            if agent_id not in self.agent_configurations:
                logger.warning(f"Unknown agent ID in feedback action: {agent_id}")
                return
            
            # Store rollback data
            current_config = self.agent_configurations[agent_id]
            action.rollback_data = {
                'previous_parameters': current_config.parameters.copy(),
                'previous_version': current_config.config_version
            }
            
            # Apply parameter changes
            updated_parameters = current_config.parameters.copy()
            updated_parameters.update(action.parameters)
            
            # Update configuration
            new_version = f"{current_config.config_version.split('.')[0]}.{int(current_config.config_version.split('.')[1]) + 1}.0"
            
            self.agent_configurations[agent_id] = AgentConfiguration(
                agent_id=agent_id,
                config_version=new_version,
                parameters=updated_parameters,
                performance_targets=current_config.performance_targets,
                constraints=current_config.constraints,
                last_updated=datetime.now(),
                applied_optimizations=current_config.applied_optimizations + [action.action_id]
            )
            
            # Send configuration update to PTaaS agent
            await self._send_config_update(agent_id, updated_parameters)
            
            # Mark action as applied
            action.applied_at = datetime.now()
            
            # Update metrics
            for param_name in action.parameters.keys():
                self.configuration_changes.labels(
                    agent_id=agent_id,
                    parameter_type=param_name
                ).inc()
            
            logger.info(f"Applied feedback action {action.action_id} to agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Error applying feedback action: {e}")
    
    @circuit_breaker.circuit
    async def _send_config_update(self, agent_id: str, parameters: Dict[str, Any]):
        """Send configuration update to PTaaS agent"""
        try:
            if self.kafka_producer:
                update_message = {
                    'type': 'configuration_update',
                    'agent_id': agent_id,
                    'parameters': parameters,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'adaptive_feedback_engine'
                }
                
                await self.kafka_producer.send(
                    'ptaas-agent-configs',
                    value=update_message
                )
                
                logger.info(f"Sent configuration update to agent {agent_id}")
                
        except Exception as e:
            logger.error(f"Error sending config update to {agent_id}: {e}")
    
    async def _analyze_agent_performance_trends(self, agent_id: str):
        """Analyze performance trends and generate ML training data"""
        try:
            if len(self.performance_history[agent_id]) < 10:
                return  # Need sufficient history
            
            snapshots = list(self.performance_history[agent_id])[-20:]  # Last 20 snapshots
            
            # Calculate performance improvements/degradations
            for i in range(1, len(snapshots)):
                prev_snapshot = snapshots[i-1]
                curr_snapshot = snapshots[i]
                
                # Extract features
                features = {
                    'success_rate': curr_snapshot.metrics.get('success_rate', 0.0),
                    'false_positive_rate': curr_snapshot.metrics.get('false_positive_rate', 0.0),
                    'avg_detection_time': curr_snapshot.metrics.get('avg_detection_time', 0.0),
                    'cpu_usage': curr_snapshot.metrics.get('cpu_usage', 0.0),
                    'memory_usage': curr_snapshot.metrics.get('memory_usage', 0.0),
                    'scan_depth': curr_snapshot.test_context.get('scan_depth', 3),
                    'timeout_value': curr_snapshot.test_context.get('timeout_seconds', 300),
                    'parallel_workers': curr_snapshot.test_context.get('parallel_workers', 4),
                    'retry_count': curr_snapshot.test_context.get('retry_count', 2),
                    'confidence_threshold': curr_snapshot.test_context.get('confidence_threshold', 0.7)
                }
                
                # Calculate improvements
                targets = {
                    'accuracy_improvement': (
                        curr_snapshot.metrics.get('success_rate', 0.0) - 
                        prev_snapshot.metrics.get('success_rate', 0.0)
                    ),
                    'speed_improvement': (
                        prev_snapshot.metrics.get('avg_detection_time', 0.0) - 
                        curr_snapshot.metrics.get('avg_detection_time', 0.0)
                    ) / max(1, prev_snapshot.metrics.get('avg_detection_time', 1.0)),
                    'resource_efficiency': (
                        prev_snapshot.metrics.get('resource_utilization', 0.0) - 
                        curr_snapshot.metrics.get('resource_utilization', 0.0)
                    )
                }
                
                # Add training sample
                self.ml_optimizer.add_training_sample(agent_id, features, targets)
            
        except Exception as e:
            logger.error(f"Error analyzing performance trends for {agent_id}: {e}")
    
    async def get_feedback_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the adaptive feedback system"""
        return {
            'system_status': {
                'is_running': self.is_running,
                'worker_tasks': len(self.worker_tasks),
                'active_feedback_actions': len(self.active_feedback_actions),
                'optimization_queue_size': self.optimization_queue.qsize()
            },
            'agent_configurations': {
                agent_id: {
                    'config_version': config.config_version,
                    'last_updated': config.last_updated.isoformat(),
                    'applied_optimizations_count': len(config.applied_optimizations),
                    'parameters_count': len(config.parameters)
                }
                for agent_id, config in self.agent_configurations.items()
            },
            'performance_history': {
                agent_id: len(history) for agent_id, history in self.performance_history.items()
            },
            'ml_optimizer_status': {
                agent_id: {
                    'is_trained': self.ml_optimizer.is_trained[agent_id],
                    'training_samples': len(self.ml_optimizer.training_data[agent_id])
                }
                for agent_id in self.ml_optimizer.agent_ids
            },
            'connection_status': {
                'redis': self.redis_client is not None,
                'kafka_producer': self.kafka_producer is not None,
                'kafka_consumer': self.kafka_consumer is not None
            }
        }

# Global feedback engine instance
feedback_engine = AdaptiveFeedbackEngine()

@app.on_event("startup")
async def startup_event():
    """Initialize adaptive feedback engine on startup"""
    await feedback_engine.initialize()
    await feedback_engine.start_feedback_engine()

@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully shutdown adaptive feedback engine"""
    await feedback_engine.stop_feedback_engine()

@app.get("/health")
async def health_check():
    """Comprehensive health check for adaptive feedback system"""
    status = await feedback_engine.get_feedback_system_status()
    
    return {
        "status": "healthy" if status['system_status']['is_running'] else "unhealthy",
        "service": "ptaas_adaptive_feedback",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "details": status
    }

@app.get("/api/v1/feedback/status")
async def get_feedback_status():
    """Get detailed status of the adaptive feedback system"""
    return await feedback_engine.get_feedback_system_status()

@app.get("/api/v1/feedback/configurations")
async def get_agent_configurations():
    """Get current configurations for all agents"""
    return {
        agent_id: asdict(config) 
        for agent_id, config in feedback_engine.agent_configurations.items()
    }

@app.get("/api/v1/feedback/configurations/{agent_id}")
async def get_agent_configuration(agent_id: str):
    """Get current configuration for specific agent"""
    if agent_id not in feedback_engine.agent_configurations:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return asdict(feedback_engine.agent_configurations[agent_id])

@app.get("/api/v1/feedback/actions")
async def get_feedback_actions():
    """Get all active feedback actions"""
    return {
        "active_actions": [
            asdict(action) for action in feedback_engine.active_feedback_actions.values()
        ],
        "total_count": len(feedback_engine.active_feedback_actions)
    }

@app.get("/api/v1/feedback/performance/{agent_id}")
async def get_agent_performance_history(agent_id: str):
    """Get performance history for specific agent"""
    if agent_id not in feedback_engine.performance_history:
        raise HTTPException(status_code=404, detail="Agent performance history not found")
    
    history = list(feedback_engine.performance_history[agent_id])[-50:]  # Last 50 snapshots
    
    return {
        "agent_id": agent_id,
        "performance_history": [
            {
                'timestamp': snapshot.timestamp.isoformat(),
                'metrics': snapshot.metrics,
                'configuration_hash': snapshot.configuration_hash,
                'test_context': snapshot.test_context,
                'outcomes': snapshot.outcomes
            }
            for snapshot in history
        ],
        "total_snapshots": len(feedback_engine.performance_history[agent_id])
    }

@app.post("/api/v1/feedback/optimize/{agent_id}")
async def trigger_agent_optimization(agent_id: str, optimization_goal: str):
    """Manually trigger optimization for specific agent"""
    if agent_id not in feedback_engine.agent_configurations:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    try:
        goal = OptimizationGoal(optimization_goal)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid optimization goal")
    
    # Add optimization request to queue
    optimization_request = {
        'type': 'manual_optimization',
        'agent_id': agent_id,
        'optimization_goal': goal,
        'timestamp': datetime.now(),
        'triggered_by': 'api_request'
    }
    
    try:
        feedback_engine.optimization_queue.put_nowait(optimization_request)
        return {
            "status": "optimization_queued",
            "agent_id": agent_id,
            "optimization_goal": optimization_goal,
            "queue_position": feedback_engine.optimization_queue.qsize()
        }
    except asyncio.QueueFull:
        raise HTTPException(status_code=503, detail="Optimization queue is full")

@app.websocket("/ws/feedback/realtime")
async def feedback_websocket(websocket: WebSocket):
    """Real-time WebSocket endpoint for feedback system updates"""
    await websocket.accept()
    
    try:
        while True:
            status = await feedback_engine.get_feedback_system_status()
            await websocket.send_json({
                "type": "feedback_status",
                "data": status,
                "timestamp": datetime.now().isoformat()
            })
            
            await asyncio.sleep(10)  # Send updates every 10 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(
        "ptaas_adaptive_feedback:app",
        host="0.0.0.0",
        port=8088,
        reload=False,
        log_level="info"
    )