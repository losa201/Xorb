#!/usr/bin/env python3
"""
XORB PTaaS-Learning Engine Integration
Enterprise-grade integration between PTaaS platform and Autonomous Learning Engine
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import numpy as np
from collections import deque, defaultdict
import aiohttp
import aiokafka
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import circuit_breaker

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="XORB PTaaS-Learning Integration Engine",
    description="Real-time integration between PTaaS platform and Autonomous Learning Engine",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LearningEventType(Enum):
    TEST_STARTED = "test_started"
    TEST_COMPLETED = "test_completed"
    VULNERABILITY_FOUND = "vulnerability_found"
    AGENT_PERFORMANCE = "agent_performance"
    FALSE_POSITIVE = "false_positive"
    TEST_FAILED = "test_failed"
    ANOMALY_DETECTED = "anomaly_detected"
    STRATEGY_OPTIMIZED = "strategy_optimized"

class ActionType(Enum):
    AGENT_CONFIG_UPDATE = "agent_config_update"
    STRATEGY_ADJUSTMENT = "strategy_adjustment"
    PRIORITY_REWEIGHT = "priority_reweight"
    THRESHOLD_UPDATE = "threshold_update"
    SCAN_INTENSITY_CHANGE = "scan_intensity_change"

@dataclass
class LearningEvent:
    event_id: str
    event_type: LearningEventType
    timestamp: datetime
    agent_id: str
    test_id: str
    data: Dict[str, Any]
    context: Dict[str, Any]
    reward: float
    confidence: float

@dataclass
class AdaptiveAction:
    action_id: str
    action_type: ActionType
    target_agent: str
    parameters: Dict[str, Any]
    expected_improvement: float
    priority: float
    created_at: datetime
    applied_at: Optional[datetime] = None

@dataclass
class AgentPerformanceMetrics:
    agent_id: str
    success_rate: float
    false_positive_rate: float
    avg_detection_time: float
    accuracy_score: float
    efficiency_score: float
    learning_velocity: float
    last_updated: datetime

class MultiArmedBandit:
    """Multi-armed bandit for agent selection optimization"""
    
    def __init__(self, n_arms: int, decay_factor: float = 0.95):
        self.n_arms = n_arms
        self.decay_factor = decay_factor
        self.arm_counts = np.zeros(n_arms)
        self.arm_values = np.zeros(n_arms)
        self.total_count = 0
        self.epsilon = 0.1  # Exploration rate
        
    def select_arm(self) -> int:
        """Select arm using epsilon-greedy strategy with UCB"""
        if np.random.random() < self.epsilon:
            # Exploration: random selection
            return np.random.randint(self.n_arms)
        else:
            # Exploitation with Upper Confidence Bound
            if self.total_count == 0:
                return np.random.randint(self.n_arms)
            
            ucb_values = np.zeros(self.n_arms)
            for arm in range(self.n_arms):
                if self.arm_counts[arm] == 0:
                    ucb_values[arm] = float('inf')
                else:
                    confidence = np.sqrt(2 * np.log(self.total_count) / self.arm_counts[arm])
                    ucb_values[arm] = self.arm_values[arm] + confidence
            
            return np.argmax(ucb_values)
    
    def update_arm(self, arm: int, reward: float):
        """Update arm statistics with reward"""
        self.arm_counts[arm] += 1
        self.total_count += 1
        
        # Exponential moving average
        alpha = 1.0 / self.arm_counts[arm]
        self.arm_values[arm] += alpha * (reward - self.arm_values[arm])
        
        # Apply decay to old values
        self.arm_values *= self.decay_factor
        self.arm_counts *= self.decay_factor

class ReinforcementLearningOptimizer:
    """Q-Learning optimizer for testing strategy adaptation"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = 0.95
        self.epsilon = 0.1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Q-table initialization
        self.q_table = np.random.uniform(low=-0.1, high=0.1, 
                                       size=(state_size, action_size))
        self.experience_buffer = deque(maxlen=10000)
        
    def get_state_vector(self, context: Dict[str, Any]) -> int:
        """Convert context to state vector"""
        # Simplified state encoding based on context features
        features = [
            context.get('target_complexity', 0.5),
            context.get('previous_success_rate', 0.5),
            context.get('time_pressure', 0.5),
            context.get('resource_utilization', 0.5)
        ]
        
        # Discretize continuous features into state bins
        state_bins = [min(int(f * 10), 9) for f in features]
        state_index = sum(bin_val * (10 ** i) for i, bin_val in enumerate(state_bins))
        return state_index % self.state_size
    
    def select_action(self, state: int) -> int:
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])
    
    def update_q_value(self, state: int, action: int, reward: float, next_state: int):
        """Update Q-value using Q-learning update rule"""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def store_experience(self, state: int, action: int, reward: float, next_state: int):
        """Store experience for potential replay learning"""
        self.experience_buffer.append((state, action, reward, next_state))

class PTaaSLearningIntegration:
    """Core integration engine between PTaaS and Learning systems"""
    
    def __init__(self):
        self.redis_client = None
        self.kafka_producer = None
        self.kafka_consumer = None
        
        # Learning components
        self.bandit = MultiArmedBandit(n_arms=5)  # 5 PTaaS agents
        self.rl_optimizer = ReinforcementLearningOptimizer(state_size=10000, action_size=10)
        
        # Performance tracking
        self.agent_metrics: Dict[str, AgentPerformanceMetrics] = {}
        self.learning_events: deque = deque(maxlen=1000)
        self.adaptive_actions: Dict[str, AdaptiveAction] = {}
        
        # Circuit breakers for fault tolerance
        self.circuit_breakers = {
            'kafka': circuit_breaker.CircuitBreaker(failure_threshold=5, recovery_timeout=60),
            'redis': circuit_breaker.CircuitBreaker(failure_threshold=3, recovery_timeout=30),
            'ptaas_api': circuit_breaker.CircuitBreaker(failure_threshold=10, recovery_timeout=120)
        }
        
        # Metrics
        self.metrics_registry = CollectorRegistry()
        self.learning_events_counter = Counter(
            'ptaas_learning_events_total',
            'Total learning events processed',
            ['event_type', 'agent_id'],
            registry=self.metrics_registry
        )
        self.adaptation_latency = Histogram(
            'ptaas_adaptation_latency_seconds',
            'Time taken for adaptive responses',
            registry=self.metrics_registry
        )
        self.agent_performance_gauge = Gauge(
            'ptaas_agent_performance_score',
            'Current agent performance scores',
            ['agent_id', 'metric_type'],
            registry=self.metrics_registry
        )
        
        # Initialize agent performance tracking
        self._initialize_agent_metrics()
    
    def _initialize_agent_metrics(self):
        """Initialize performance metrics for all PTaaS agents"""
        agent_ids = [
            "AGENT-WEB-SCANNER-001",
            "AGENT-NETWORK-RECON-002", 
            "AGENT-VULN-ASSESSMENT-003",
            "AGENT-EXPLOITATION-004",
            "AGENT-DB-TESTER-005"
        ]
        
        for agent_id in agent_ids:
            self.agent_metrics[agent_id] = AgentPerformanceMetrics(
                agent_id=agent_id,
                success_rate=0.85,  # Initial baseline
                false_positive_rate=0.15,
                avg_detection_time=120.0,
                accuracy_score=0.80,
                efficiency_score=0.75,
                learning_velocity=0.1,
                last_updated=datetime.now()
            )
    
    async def initialize_connections(self):
        """Initialize all external connections with fault tolerance"""
        try:
            # Redis connection
            self.redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
            await self.redis_client.ping()
            logger.info("Redis connection established for learning integration")
            
            # Kafka connections
            self.kafka_producer = aiokafka.AIOKafkaProducer(
                bootstrap_servers='localhost:9092',
                value_serializer=lambda x: json.dumps(x, default=str).encode()
            )
            await self.kafka_producer.start()
            
            self.kafka_consumer = aiokafka.AIOKafkaConsumer(
                'ptaas-events',
                'ptaas-metrics',
                bootstrap_servers='localhost:9092',
                value_deserializer=lambda x: json.loads(x.decode())
            )
            await self.kafka_consumer.start()
            logger.info("Kafka connections established for real-time data pipelines")
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            # Implement graceful degradation
            await self._handle_connection_failure(e)
    
    async def _handle_connection_failure(self, error: Exception):
        """Handle connection failures with graceful degradation"""
        logger.warning(f"Connection failure handled: {error}")
        # Continue with reduced functionality
        pass
    
    @circuit_breaker.circuit
    async def process_learning_event(self, event: LearningEvent) -> Optional[AdaptiveAction]:
        """Process learning event and generate adaptive action"""
        start_time = time.time()
        
        try:
            # Store event
            self.learning_events.append(event)
            self.learning_events_counter.labels(
                event_type=event.event_type.value,
                agent_id=event.agent_id
            ).inc()
            
            # Update agent performance metrics
            await self._update_agent_metrics(event)
            
            # Generate adaptive action based on event
            adaptive_action = await self._generate_adaptive_action(event)
            
            # Store adaptive action
            if adaptive_action:
                self.adaptive_actions[adaptive_action.action_id] = adaptive_action
                
                # Send to PTaaS agents via Kafka
                await self._send_adaptive_action(adaptive_action)
            
            # Update learning algorithms
            await self._update_learning_algorithms(event, adaptive_action)
            
            # Record processing latency
            self.adaptation_latency.observe(time.time() - start_time)
            
            return adaptive_action
            
        except Exception as e:
            logger.error(f"Error processing learning event: {e}")
            return None
    
    async def _update_agent_metrics(self, event: LearningEvent):
        """Update agent performance metrics based on learning event"""
        agent_id = event.agent_id
        
        if agent_id not in self.agent_metrics:
            return
        
        metrics = self.agent_metrics[agent_id]
        alpha = 0.1  # Learning rate for exponential moving average
        
        if event.event_type == LearningEventType.VULNERABILITY_FOUND:
            # Positive outcome - improve success rate
            metrics.success_rate = (1 - alpha) * metrics.success_rate + alpha * 1.0
            metrics.accuracy_score = (1 - alpha) * metrics.accuracy_score + alpha * event.confidence
            
        elif event.event_type == LearningEventType.FALSE_POSITIVE:
            # Negative outcome - increase false positive rate
            metrics.false_positive_rate = (1 - alpha) * metrics.false_positive_rate + alpha * 1.0
            metrics.accuracy_score = (1 - alpha) * metrics.accuracy_score + alpha * (1.0 - event.confidence)
            
        elif event.event_type == LearningEventType.TEST_COMPLETED:
            # Update detection time and efficiency
            detection_time = event.data.get('detection_time', metrics.avg_detection_time)
            metrics.avg_detection_time = (1 - alpha) * metrics.avg_detection_time + alpha * detection_time
            
            efficiency = event.data.get('efficiency_score', metrics.efficiency_score)
            metrics.efficiency_score = (1 - alpha) * metrics.efficiency_score + alpha * efficiency
        
        # Update learning velocity based on recent performance changes
        metrics.learning_velocity = alpha * abs(event.reward)
        metrics.last_updated = datetime.now()
        
        # Update Prometheus metrics
        self.agent_performance_gauge.labels(
            agent_id=agent_id, metric_type='success_rate'
        ).set(metrics.success_rate)
        self.agent_performance_gauge.labels(
            agent_id=agent_id, metric_type='accuracy_score'
        ).set(metrics.accuracy_score)
        self.agent_performance_gauge.labels(
            agent_id=agent_id, metric_type='efficiency_score'
        ).set(metrics.efficiency_score)
    
    async def _generate_adaptive_action(self, event: LearningEvent) -> Optional[AdaptiveAction]:
        """Generate adaptive action based on learning event analysis"""
        
        # Analyze event context and determine optimal action
        if event.event_type == LearningEventType.FALSE_POSITIVE and event.confidence > 0.8:
            # High-confidence false positive - adjust detection thresholds
            return AdaptiveAction(
                action_id=str(uuid.uuid4()),
                action_type=ActionType.THRESHOLD_UPDATE,
                target_agent=event.agent_id,
                parameters={
                    'detection_threshold': 0.1,  # Increase threshold to reduce FPs
                    'confidence_boost': -0.05
                },
                expected_improvement=0.15,
                priority=0.8,
                created_at=datetime.now()
            )
        
        elif event.event_type == LearningEventType.VULNERABILITY_FOUND and event.reward > 0.9:
            # High-value vulnerability found - increase scan intensity for similar targets
            return AdaptiveAction(
                action_id=str(uuid.uuid4()),
                action_type=ActionType.SCAN_INTENSITY_CHANGE,
                target_agent=event.agent_id,
                parameters={
                    'intensity_multiplier': 1.2,
                    'focus_areas': event.data.get('vulnerability_categories', []),
                    'duration_extension': 300  # 5 more minutes
                },
                expected_improvement=0.25,
                priority=0.9,
                created_at=datetime.now()
            )
        
        elif event.event_type == LearningEventType.AGENT_PERFORMANCE:
            # Performance-based configuration optimization
            performance_score = event.data.get('performance_score', 0.5)
            
            if performance_score < 0.6:  # Underperforming agent
                return AdaptiveAction(
                    action_id=str(uuid.uuid4()),
                    action_type=ActionType.AGENT_CONFIG_UPDATE,
                    target_agent=event.agent_id,
                    parameters={
                        'timeout_adjustment': 1.5,  # Increase timeout
                        'retry_count': 2,  # Add retries
                        'parallel_workers': max(1, event.data.get('workers', 2) - 1)
                    },
                    expected_improvement=0.2,
                    priority=0.7,
                    created_at=datetime.now()
                )
        
        return None
    
    @circuit_breaker.circuit
    async def _send_adaptive_action(self, action: AdaptiveAction):
        """Send adaptive action to PTaaS agents via Kafka"""
        try:
            if self.kafka_producer:
                message = {
                    'action_id': action.action_id,
                    'action_type': action.action_type.value,
                    'target_agent': action.target_agent,
                    'parameters': action.parameters,
                    'priority': action.priority,
                    'timestamp': action.created_at.isoformat()
                }
                
                await self.kafka_producer.send(
                    'ptaas-adaptive-actions',
                    value=message
                )
                
                logger.info(f"Sent adaptive action {action.action_id} to agent {action.target_agent}")
                
        except Exception as e:
            logger.error(f"Failed to send adaptive action: {e}")
    
    async def _update_learning_algorithms(self, event: LearningEvent, action: Optional[AdaptiveAction]):
        """Update reinforcement learning and multi-armed bandit algorithms"""
        
        # Update multi-armed bandit for agent selection
        agent_index = self._get_agent_index(event.agent_id)
        if agent_index is not None:
            self.bandit.update_arm(agent_index, event.reward)
        
        # Update Q-learning optimizer
        if action:
            state = self.rl_optimizer.get_state_vector(event.context)
            action_index = self._get_action_index(action.action_type)
            
            # Simulate next state based on expected improvement
            next_state = state  # Simplified - could be more sophisticated
            
            self.rl_optimizer.update_q_value(state, action_index, event.reward, next_state)
            self.rl_optimizer.store_experience(state, action_index, event.reward, next_state)
    
    def _get_agent_index(self, agent_id: str) -> Optional[int]:
        """Map agent ID to bandit arm index"""
        agent_mapping = {
            "AGENT-WEB-SCANNER-001": 0,
            "AGENT-NETWORK-RECON-002": 1,
            "AGENT-VULN-ASSESSMENT-003": 2,
            "AGENT-EXPLOITATION-004": 3,
            "AGENT-DB-TESTER-005": 4
        }
        return agent_mapping.get(agent_id)
    
    def _get_action_index(self, action_type: ActionType) -> int:
        """Map action type to RL action index"""
        action_mapping = {
            ActionType.AGENT_CONFIG_UPDATE: 0,
            ActionType.STRATEGY_ADJUSTMENT: 1,
            ActionType.PRIORITY_REWEIGHT: 2,
            ActionType.THRESHOLD_UPDATE: 3,
            ActionType.SCAN_INTENSITY_CHANGE: 4
        }
        return action_mapping.get(action_type, 0)
    
    async def get_optimal_agent_selection(self, context: Dict[str, Any]) -> Tuple[str, float]:
        """Get optimal agent selection using multi-armed bandit"""
        optimal_arm = self.bandit.select_arm()
        
        agent_mapping = [
            "AGENT-WEB-SCANNER-001",
            "AGENT-NETWORK-RECON-002",
            "AGENT-VULN-ASSESSMENT-003",
            "AGENT-EXPLOITATION-004",
            "AGENT-DB-TESTER-005"
        ]
        
        selected_agent = agent_mapping[optimal_arm]
        confidence = self.bandit.arm_values[optimal_arm]
        
        return selected_agent, confidence
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get comprehensive learning insights and analytics"""
        insights = {
            'timestamp': datetime.now().isoformat(),
            'total_events_processed': len(self.learning_events),
            'active_adaptations': len(self.adaptive_actions),
            'agent_performance': {},
            'learning_algorithm_stats': {
                'bandit': {
                    'arm_values': self.bandit.arm_values.tolist(),
                    'arm_counts': self.bandit.arm_counts.tolist(),
                    'total_selections': self.bandit.total_count,
                    'epsilon': self.bandit.epsilon
                },
                'q_learning': {
                    'epsilon': self.rl_optimizer.epsilon,
                    'experience_buffer_size': len(self.rl_optimizer.experience_buffer),
                    'q_table_shape': self.q_table.shape if hasattr(self.rl_optimizer, 'q_table') else None
                }
            },
            'recent_events': [
                {
                    'event_type': event.event_type.value,
                    'agent_id': event.agent_id,
                    'reward': event.reward,
                    'confidence': event.confidence,
                    'timestamp': event.timestamp.isoformat()
                }
                for event in list(self.learning_events)[-10:]
            ]
        }
        
        # Add agent performance metrics
        for agent_id, metrics in self.agent_metrics.items():
            insights['agent_performance'][agent_id] = {
                'success_rate': metrics.success_rate,
                'false_positive_rate': metrics.false_positive_rate,
                'avg_detection_time': metrics.avg_detection_time,
                'accuracy_score': metrics.accuracy_score,
                'efficiency_score': metrics.efficiency_score,
                'learning_velocity': metrics.learning_velocity,
                'last_updated': metrics.last_updated.isoformat()
            }
        
        return insights
    
    async def kafka_consumer_loop(self):
        """Main Kafka consumer loop for processing PTaaS events"""
        logger.info("Starting Kafka consumer loop for PTaaS-Learning integration")
        
        try:
            async for message in self.kafka_consumer:
                try:
                    # Parse message
                    event_data = message.value
                    
                    # Create learning event
                    learning_event = LearningEvent(
                        event_id=str(uuid.uuid4()),
                        event_type=LearningEventType(event_data.get('event_type', 'test_completed')),
                        timestamp=datetime.fromisoformat(event_data.get('timestamp', datetime.now().isoformat())),
                        agent_id=event_data.get('agent_id', 'unknown'),
                        test_id=event_data.get('test_id', 'unknown'),
                        data=event_data.get('data', {}),
                        context=event_data.get('context', {}),
                        reward=event_data.get('reward', 0.0),
                        confidence=event_data.get('confidence', 0.5)
                    )
                    
                    # Process event
                    await self.process_learning_event(learning_event)
                    
                except Exception as e:
                    logger.error(f"Error processing Kafka message: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Kafka consumer error: {e}")
            # Implement exponential backoff retry
            await asyncio.sleep(5)
            await self.kafka_consumer_loop()

# Global integration engine
integration_engine = PTaaSLearningIntegration()

@app.on_event("startup")
async def startup_event():
    """Initialize integration engine on startup"""
    await integration_engine.initialize_connections()
    
    # Start Kafka consumer in background
    asyncio.create_task(integration_engine.kafka_consumer_loop())

@app.get("/health")
async def health_check():
    """Comprehensive health check for integration service"""
    health_status = {
        "status": "healthy",
        "service": "ptaas_learning_integration",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "connections": {
            "redis": "connected" if integration_engine.redis_client else "disconnected",
            "kafka_producer": "connected" if integration_engine.kafka_producer else "disconnected",
            "kafka_consumer": "connected" if integration_engine.kafka_consumer else "disconnected"
        },
        "circuit_breakers": {
            name: "closed" if cb.state == circuit_breaker.CircuitBreakerState.CLOSED else "open"
            for name, cb in integration_engine.circuit_breakers.items()
        },
        "learning_stats": {
            "events_processed": len(integration_engine.learning_events),
            "active_adaptations": len(integration_engine.adaptive_actions),
            "agent_count": len(integration_engine.agent_metrics)
        }
    }
    
    return health_status

@app.post("/api/v1/learning/events")
async def submit_learning_event(event_data: dict):
    """Submit learning event for processing"""
    try:
        learning_event = LearningEvent(
            event_id=str(uuid.uuid4()),
            event_type=LearningEventType(event_data.get('event_type')),
            timestamp=datetime.fromisoformat(event_data.get('timestamp', datetime.now().isoformat())),
            agent_id=event_data.get('agent_id'),
            test_id=event_data.get('test_id'),
            data=event_data.get('data', {}),
            context=event_data.get('context', {}),
            reward=event_data.get('reward', 0.0),
            confidence=event_data.get('confidence', 0.5)
        )
        
        adaptive_action = await integration_engine.process_learning_event(learning_event)
        
        return {
            "status": "processed",
            "event_id": learning_event.event_id,
            "adaptive_action": asdict(adaptive_action) if adaptive_action else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/learning/insights")
async def get_learning_insights():
    """Get comprehensive learning insights and analytics"""
    return await integration_engine.get_learning_insights()

@app.get("/api/v1/learning/agents/optimal")
async def get_optimal_agent(context: dict = None):
    """Get optimal agent selection recommendation"""
    context = context or {}
    agent_id, confidence = await integration_engine.get_optimal_agent_selection(context)
    
    return {
        "optimal_agent": agent_id,
        "confidence": confidence,
        "selection_method": "multi_armed_bandit",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/learning/performance/{agent_id}")
async def get_agent_performance(agent_id: str):
    """Get detailed performance metrics for specific agent"""
    if agent_id not in integration_engine.agent_metrics:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    metrics = integration_engine.agent_metrics[agent_id]
    return {
        "agent_id": agent_id,
        "performance_metrics": asdict(metrics),
        "recommendation": await integration_engine._generate_performance_recommendation(agent_id)
    }

@app.get("/api/v1/learning/adaptations")
async def get_active_adaptations():
    """Get all active adaptive actions"""
    return {
        "active_adaptations": [
            asdict(action) for action in integration_engine.adaptive_actions.values()
        ],
        "total_count": len(integration_engine.adaptive_actions)
    }

@app.websocket("/ws/learning/realtime")
async def learning_websocket(websocket: WebSocket):
    """Real-time WebSocket endpoint for learning events and adaptations"""
    await websocket.accept()
    
    try:
        last_event_count = 0
        
        while True:
            # Send updates when new events are processed
            current_event_count = len(integration_engine.learning_events)
            
            if current_event_count > last_event_count:
                insights = await integration_engine.get_learning_insights()
                await websocket.send_json({
                    "type": "learning_update",
                    "insights": insights,
                    "timestamp": datetime.now().isoformat()
                })
                last_event_count = current_event_count
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(
        "ptaas_learning_integration:app",
        host="0.0.0.0",
        port=8086,
        reload=False,
        log_level="info"
    )