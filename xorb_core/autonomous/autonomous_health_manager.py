#!/usr/bin/env python3
"""
XORB Autonomous Health Manager

Principal infrastructure system for full autonomous lifecycle management,
including health monitoring, self-diagnosis, and auto-remediation of failing
components in real-time.

Features:
- Real-time health monitoring via Prometheus and Docker APIs
- Intelligent error classification using ML models
- Self-healing workflows with graceful failover
- Episodic memory integration for failure pattern learning
- Autonomous remediation with escalation paths

Author: XORB Autonomous Systems
Version: 2.0.0
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import aiohttp
import docker
import numpy as np
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import websockets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import redis.asyncio as redis

# XORB Core Imports
from xorb_core.knowledge_fabric.episodic_memory_system import EpisodicMemorySystem
from xorb_core.orchestration.intelligent_orchestrator import IntelligentOrchestrator
from xorb_core.ai.external_intelligence_api import ExternalIntelligenceAPI

logger = logging.getLogger(__name__)


class ServiceState(Enum):
    """Service health states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"


class RemediationType(Enum):
    """Types of remediation actions"""
    RESTART = "restart"
    ROLLBACK = "rollback"
    SCALE_UP = "scale_up"
    FAILOVER = "failover"
    ESCALATE = "escalate"
    IGNORE = "ignore"


class ErrorCategory(Enum):
    """ML-classified error categories"""
    NETWORK = "network"
    MEMORY = "memory"
    DEPENDENCY = "dependency"
    LOGIC = "logic"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


@dataclass
class HealthMetrics:
    """Container for service health metrics"""
    service_name: str
    timestamp: datetime
    state: ServiceState
    cpu_usage: float
    memory_usage: float
    response_time: float
    error_rate: float
    uptime: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class FailureEvent:
    """Container for failure event data"""
    service_name: str
    timestamp: datetime
    error_category: ErrorCategory
    error_message: str
    stack_trace: Optional[str]
    metrics_snapshot: HealthMetrics
    remediation_attempted: List[RemediationType] = field(default_factory=list)
    resolution_time: Optional[float] = None
    success: bool = False


class AutonomousHealthManager:
    """
    Principal autonomous health management system for XORB ecosystem.
    
    Provides real-time monitoring, intelligent diagnosis, and automated
    remediation of infrastructure failures.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.docker_client = docker.from_env()
        self.redis_client = None
        self.episodic_memory = EpisodicMemorySystem()
        self.orchestrator = IntelligentOrchestrator()
        self.external_intel = ExternalIntelligenceAPI()
        
        # Health monitoring state
        self.service_states: Dict[str, ServiceState] = {}
        self.health_history: Dict[str, List[HealthMetrics]] = {}
        self.active_failures: Dict[str, FailureEvent] = {}
        self.remediation_queue = asyncio.Queue()
        
        # ML models for error classification
        self.error_classifier: Optional[LogisticRegression] = None
        self.remediation_model: Optional[SVC] = None
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Configuration
        self.monitoring_interval = config.get('monitoring_interval', 30)
        self.health_threshold = config.get('health_threshold', 0.8)
        self.max_remediation_attempts = config.get('max_remediation_attempts', 3)
        self.escalation_timeout = config.get('escalation_timeout', 300)
        
        logger.info("🤖 Autonomous Health Manager initialized")
    
    def _setup_prometheus_metrics(self):
        """Initialize Prometheus metrics for health monitoring"""
        self.metrics = {
            'service_health_status': Gauge(
                'xorb_service_health_status',
                'Current health status of XORB services',
                ['service', 'state']
            ),
            'agent_failure_count': Counter(
                'xorb_agent_failure_count',
                'Total number of agent failures',
                ['agent_type', 'error_category']
            ),
            'self_healing_trigger_total': Counter(
                'xorb_self_healing_trigger_total',
                'Total number of self-healing triggers',
                ['service', 'remediation_type']
            ),
            'autonomous_repair_success_ratio': Gauge(
                'xorb_autonomous_repair_success_ratio',
                'Success ratio of autonomous repairs',
                ['service', 'time_window']
            ),
            'service_response_time': Histogram(
                'xorb_service_response_time_seconds',
                'Service response time in seconds',
                ['service'],
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            ),
            'remediation_duration': Histogram(
                'xorb_remediation_duration_seconds',
                'Time taken for remediation actions',
                ['service', 'remediation_type'],
                buckets=[1, 5, 10, 30, 60, 300]
            ),
            'memory_usage_percent': Gauge(
                'xorb_service_memory_usage_percent',
                'Memory usage percentage by service',
                ['service']
            ),
            'cpu_usage_percent': Gauge(
                'xorb_service_cpu_usage_percent',
                'CPU usage percentage by service',
                ['service']
            )
        }
    
    async def initialize(self):
        """Initialize async components and start monitoring"""
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                decode_responses=True
            )
            
            # Load or train ML models
            await self._initialize_ml_models()
            
            # Start Prometheus metrics server
            start_http_server(self.config.get('metrics_port', 9091))
            
            # Start monitoring tasks
            await asyncio.gather(
                self._health_monitoring_loop(),
                self._remediation_worker(),
                self._metrics_aggregator(),
                self._websocket_health_stream()
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize health manager: {e}")
            raise
    
    async def _initialize_ml_models(self):
        """Initialize or load ML models for error classification"""
        try:
            # Try to load existing models
            self.error_classifier = joblib.load('models/error_classifier.pkl')
            self.remediation_model = joblib.load('models/remediation_model.pkl')
            self.vectorizer = joblib.load('models/error_vectorizer.pkl')
            logger.info("✅ Loaded pre-trained ML models")
            
        except FileNotFoundError:
            # Train new models with historical data
            logger.info("🧠 Training new ML models for error classification")
            await self._train_error_classification_models()
    
    async def _train_error_classification_models(self):
        """Train ML models using historical failure data"""
        # Get historical failure patterns from episodic memory
        failure_history = await self.episodic_memory.get_failure_patterns(limit=1000)
        
        if len(failure_history) < 10:
            # Use synthetic training data for initial deployment
            failure_history = self._generate_synthetic_training_data()
        
        # Prepare training data
        error_messages = [f['error_message'] for f in failure_history]
        error_categories = [f['category'] for f in failure_history]
        remediation_actions = [f['successful_remediation'] for f in failure_history]
        
        # Vectorize error messages
        X = self.vectorizer.fit_transform(error_messages)
        
        # Train error classifier
        self.error_classifier = LogisticRegression(random_state=42)
        self.error_classifier.fit(X, error_categories)
        
        # Train remediation model
        self.remediation_model = SVC(kernel='rbf', probability=True, random_state=42)
        combined_features = np.hstack([X.toarray(), 
                                     [[f['severity']] for f in failure_history]])
        self.remediation_model.fit(combined_features, remediation_actions)
        
        # Save models
        import os
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.error_classifier, 'models/error_classifier.pkl')
        joblib.dump(self.remediation_model, 'models/remediation_model.pkl')
        joblib.dump(self.vectorizer, 'models/error_vectorizer.pkl')
        
        logger.info("🎯 ML models trained and saved successfully")
    
    def _generate_synthetic_training_data(self) -> List[Dict]:
        """Generate synthetic training data for initial model training"""
        synthetic_data = [
            {
                'error_message': 'Connection refused to database',
                'category': ErrorCategory.NETWORK.value,
                'successful_remediation': RemediationType.RESTART.value,
                'severity': 0.8
            },
            {
                'error_message': 'Out of memory error in container',
                'category': ErrorCategory.MEMORY.value,
                'successful_remediation': RemediationType.SCALE_UP.value,
                'severity': 0.9
            },
            {
                'error_message': 'Service dependency unavailable',
                'category': ErrorCategory.DEPENDENCY.value,
                'successful_remediation': RemediationType.FAILOVER.value,
                'severity': 0.7
            },
            {
                'error_message': 'Null pointer exception in handler',
                'category': ErrorCategory.LOGIC.value,
                'successful_remediation': RemediationType.ROLLBACK.value,
                'severity': 0.6
            },
            {
                'error_message': 'CPU usage exceeds limits',
                'category': ErrorCategory.RESOURCE.value,
                'successful_remediation': RemediationType.SCALE_UP.value,
                'severity': 0.8
            }
        ] * 20  # Replicate for more training samples
        
        return synthetic_data
    
    async def _health_monitoring_loop(self):
        """Main health monitoring loop"""
        logger.info("🔍 Starting autonomous health monitoring loop")
        
        while True:
            try:
                # Get all services to monitor
                services = await self._discover_services()
                
                # Monitor each service
                health_tasks = [
                    self._monitor_service_health(service) 
                    for service in services
                ]
                health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
                
                # Process health results
                for service, result in zip(services, health_results):
                    if isinstance(result, Exception):
                        logger.error(f"Health monitoring failed for {service}: {result}")
                        continue
                    
                    await self._process_health_metrics(service, result)
                
                # Update Prometheus metrics
                await self._update_prometheus_metrics()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _discover_services(self) -> List[str]:
        """Discover all XORB services to monitor"""
        services = []
        
        try:
            # Get Docker containers
            containers = self.docker_client.containers.list()
            for container in containers:
                if any(label in container.name.lower() for label in ['xorb', 'api', 'worker', 'orchestrator']):
                    services.append(container.name)
            
            # Add external services
            external_services = [
                'prometheus', 'grafana', 'postgres', 'redis', 
                'temporal', 'neo4j', 'qdrant', 'nats'
            ]
            services.extend(external_services)
            
        except Exception as e:
            logger.error(f"Service discovery failed: {e}")
        
        return services
    
    async def _monitor_service_health(self, service_name: str) -> HealthMetrics:
        """Monitor health of a specific service"""
        timestamp = datetime.now()
        
        try:
            # Get container stats if it's a Docker service
            container_stats = await self._get_container_stats(service_name)
            
            # Get service-specific health metrics
            health_endpoint = f"http://localhost:{self._get_service_port(service_name)}/health"
            response_time, status_code = await self._check_health_endpoint(health_endpoint)
            
            # Calculate health metrics
            cpu_usage = container_stats.get('cpu_usage', 0.0)
            memory_usage = container_stats.get('memory_usage', 0.0)
            error_rate = await self._calculate_error_rate(service_name)
            uptime = container_stats.get('uptime', 0.0)
            
            # Determine service state
            state = self._determine_service_state(
                cpu_usage, memory_usage, response_time, error_rate, status_code
            )
            
            return HealthMetrics(
                service_name=service_name,
                timestamp=timestamp,
                state=state,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                response_time=response_time,
                error_rate=error_rate,
                uptime=uptime,
                custom_metrics=container_stats.get('custom', {})
            )
            
        except Exception as e:
            logger.error(f"Health monitoring failed for {service_name}: {e}")
            return HealthMetrics(
                service_name=service_name,
                timestamp=timestamp,
                state=ServiceState.UNKNOWN,
                cpu_usage=0.0,
                memory_usage=0.0,
                response_time=float('inf'),
                error_rate=1.0,
                uptime=0.0
            )
    
    async def _get_container_stats(self, service_name: str) -> Dict[str, float]:
        """Get Docker container statistics"""
        try:
            container = self.docker_client.containers.get(service_name)
            stats = container.stats(stream=False)
            
            # Calculate CPU usage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            cpu_usage = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0
            
            # Calculate memory usage
            memory_usage = (stats['memory_stats']['usage'] / 
                          stats['memory_stats']['limit']) * 100.0
            
            # Calculate uptime
            created_time = datetime.fromisoformat(
                container.attrs['Created'].replace('Z', '+00:00')
            )
            uptime = (datetime.now().replace(tzinfo=created_time.tzinfo) - created_time).total_seconds()
            
            return {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'uptime': uptime,
                'status': container.status
            }
            
        except Exception as e:
            logger.warning(f"Could not get container stats for {service_name}: {e}")
            return {'cpu_usage': 0.0, 'memory_usage': 0.0, 'uptime': 0.0}
    
    def _get_service_port(self, service_name: str) -> int:
        """Map service names to their health check ports"""
        port_mapping = {
            'api': 8000,
            'worker': 9000,
            'orchestrator': 8080,
            'scanner-go': 8004,
            'prometheus': 9090,
            'grafana': 3000,
            'temporal': 8233,
            'postgres': 5432,
            'redis': 6379,
            'neo4j': 7474,
            'qdrant': 6333,
            'nats': 8222
        }
        
        for key, port in port_mapping.items():
            if key in service_name.lower():
                return port
        
        return 8080  # Default port
    
    async def _check_health_endpoint(self, url: str) -> Tuple[float, int]:
        """Check service health endpoint and measure response time"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as response:
                    response_time = time.time() - start_time
                    return response_time, response.status
                    
        except asyncio.TimeoutError:
            return float('inf'), 408
        except Exception:
            return float('inf'), 503
    
    async def _calculate_error_rate(self, service_name: str) -> float:
        """Calculate error rate from recent logs or metrics"""
        try:
            # Query Prometheus for error rate
            prometheus_url = "http://localhost:9090/api/v1/query"
            query = f'rate(http_requests_total{{job="{service_name}",status=~"5.."}}[5m]) / rate(http_requests_total{{job="{service_name}"}}[5m])'
            
            async with aiohttp.ClientSession() as session:
                async with session.get(prometheus_url, params={'query': query}) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data['data']['result']:
                            return float(data['data']['result'][0]['value'][1])
            
            return 0.0  # Default to no errors if can't determine
            
        except Exception:
            return 0.0
    
    def _determine_service_state(self, cpu_usage: float, memory_usage: float, 
                               response_time: float, error_rate: float, 
                               status_code: int) -> ServiceState:
        """Determine service health state based on metrics"""
        
        if status_code >= 500 or response_time == float('inf'):
            return ServiceState.FAILED
        
        if status_code >= 400 or error_rate > 0.1 or response_time > 5.0:
            return ServiceState.CRITICAL
        
        if cpu_usage > 80 or memory_usage > 80 or error_rate > 0.05:
            return ServiceState.DEGRADED
        
        if status_code == 200 and cpu_usage < 70 and memory_usage < 70:
            return ServiceState.HEALTHY
        
        return ServiceState.UNKNOWN
    
    async def _process_health_metrics(self, service_name: str, metrics: HealthMetrics):
        """Process health metrics and trigger remediation if needed"""
        # Store metrics history
        if service_name not in self.health_history:
            self.health_history[service_name] = []
        
        self.health_history[service_name].append(metrics)
        
        # Keep only recent history (last 100 measurements)
        self.health_history[service_name] = self.health_history[service_name][-100:]
        
        # Update current state
        previous_state = self.service_states.get(service_name, ServiceState.UNKNOWN)
        self.service_states[service_name] = metrics.state
        
        # Detect state changes and trigger remediation
        if metrics.state in [ServiceState.CRITICAL, ServiceState.FAILED]:
            if service_name not in self.active_failures:
                failure_event = FailureEvent(
                    service_name=service_name,
                    timestamp=metrics.timestamp,
                    error_category=ErrorCategory.UNKNOWN,
                    error_message=f"Service state changed to {metrics.state.value}",
                    stack_trace=None,
                    metrics_snapshot=metrics
                )
                
                self.active_failures[service_name] = failure_event
                await self.remediation_queue.put(failure_event)
                
                logger.warning(f"🚨 Service {service_name} entered {metrics.state.value} state")
        
        elif metrics.state == ServiceState.HEALTHY and service_name in self.active_failures:
            # Service recovered
            failure_event = self.active_failures.pop(service_name)
            failure_event.success = True
            failure_event.resolution_time = (
                metrics.timestamp - failure_event.timestamp
            ).total_seconds()
            
            # Store successful recovery in episodic memory
            await self.episodic_memory.store_failure_resolution(failure_event)
            
            logger.info(f"✅ Service {service_name} recovered successfully")
    
    async def _remediation_worker(self):
        """Background worker for processing remediation actions"""
        logger.info("🔧 Starting remediation worker")
        
        while True:
            try:
                failure_event = await self.remediation_queue.get()
                await self._handle_service_failure(failure_event)
                
            except Exception as e:
                logger.error(f"Remediation worker error: {e}")
                await asyncio.sleep(1)
    
    async def _handle_service_failure(self, failure_event: FailureEvent):
        """Handle a service failure with intelligent remediation"""
        service_name = failure_event.service_name
        
        logger.info(f"🔧 Handling failure for {service_name}")
        
        try:
            # Classify the error using ML
            error_category = await self._classify_error(failure_event.error_message)
            failure_event.error_category = error_category
            
            # Get similar past failures for learning
            similar_failures = await self.episodic_memory.get_similar_failures(
                failure_event, similarity_threshold=0.7
            )
            
            # Determine best remediation strategy
            remediation_type = await self._determine_remediation_strategy(
                failure_event, similar_failures
            )
            
            # Execute remediation
            start_time = time.time()
            success = await self._execute_remediation(service_name, remediation_type)
            duration = time.time() - start_time
            
            # Update metrics
            self.metrics['self_healing_trigger_total'].labels(
                service=service_name,
                remediation_type=remediation_type.value
            ).inc()
            
            self.metrics['remediation_duration'].labels(
                service=service_name,
                remediation_type=remediation_type.value
            ).observe(duration)
            
            failure_event.remediation_attempted.append(remediation_type)
            
            if success:
                logger.info(f"✅ Remediation successful for {service_name}")
                failure_event.success = True
            else:
                logger.warning(f"⚠️ Remediation failed for {service_name}")
                
                # Try escalation if max attempts not reached
                if len(failure_event.remediation_attempted) < self.max_remediation_attempts:
                    await self._escalate_remediation(failure_event)
                else:
                    logger.error(f"❌ Max remediation attempts reached for {service_name}")
                    await self._notify_human_intervention(failure_event)
            
        except Exception as e:
            logger.error(f"Error handling failure for {service_name}: {e}")
    
    async def _classify_error(self, error_message: str) -> ErrorCategory:
        """Classify error using ML model"""
        try:
            if self.error_classifier and self.vectorizer:
                X = self.vectorizer.transform([error_message])
                prediction = self.error_classifier.predict(X)[0]
                return ErrorCategory(prediction)
            
        except Exception as e:
            logger.warning(f"Error classification failed: {e}")
        
        # Fallback to rule-based classification
        message_lower = error_message.lower()
        
        if any(term in message_lower for term in ['connection', 'network', 'timeout', 'refused']):
            return ErrorCategory.NETWORK
        elif any(term in message_lower for term in ['memory', 'oom', 'out of memory']):
            return ErrorCategory.MEMORY
        elif any(term in message_lower for term in ['dependency', 'unavailable', 'not found']):
            return ErrorCategory.DEPENDENCY
        elif any(term in message_lower for term in ['cpu', 'load', 'resource']):
            return ErrorCategory.RESOURCE
        else:
            return ErrorCategory.UNKNOWN
    
    async def _determine_remediation_strategy(self, failure_event: FailureEvent, 
                                            similar_failures: List[Dict]) -> RemediationType:
        """Determine the best remediation strategy"""
        
        # Use ML model if available
        if self.remediation_model and self.vectorizer:
            try:
                X_text = self.vectorizer.transform([failure_event.error_message])
                X_severity = [[failure_event.metrics_snapshot.error_rate]]
                X_combined = np.hstack([X_text.toarray(), X_severity])
                
                prediction = self.remediation_model.predict(X_combined)[0]
                return RemediationType(prediction)
                
            except Exception as e:
                logger.warning(f"ML remediation prediction failed: {e}")
        
        # Fallback to rule-based strategy
        error_category = failure_event.error_category
        
        if error_category == ErrorCategory.NETWORK:
            return RemediationType.RESTART
        elif error_category == ErrorCategory.MEMORY:
            return RemediationType.SCALE_UP
        elif error_category == ErrorCategory.DEPENDENCY:
            return RemediationType.FAILOVER
        elif error_category == ErrorCategory.LOGIC:
            return RemediationType.ROLLBACK
        else:
            return RemediationType.RESTART
    
    async def _execute_remediation(self, service_name: str, 
                                 remediation_type: RemediationType) -> bool:
        """Execute the specified remediation action"""
        
        logger.info(f"🔧 Executing {remediation_type.value} for {service_name}")
        
        try:
            if remediation_type == RemediationType.RESTART:
                return await self._restart_service(service_name)
            
            elif remediation_type == RemediationType.ROLLBACK:
                return await self._rollback_service(service_name)
            
            elif remediation_type == RemediationType.SCALE_UP:
                return await self._scale_up_service(service_name)
            
            elif remediation_type == RemediationType.FAILOVER:
                return await self._failover_service(service_name)
            
            elif remediation_type == RemediationType.ESCALATE:
                return await self._escalate_to_human(service_name)
            
            else:
                logger.warning(f"Unknown remediation type: {remediation_type}")
                return False
                
        except Exception as e:
            logger.error(f"Remediation execution failed: {e}")
            return False
    
    async def _restart_service(self, service_name: str) -> bool:
        """Restart a Docker service with graceful handling"""
        try:
            # Get container
            container = self.docker_client.containers.get(service_name)
            
            # Graceful shutdown first
            container.stop(timeout=30)
            
            # Wait a moment
            await asyncio.sleep(2)
            
            # Restart
            container.start()
            
            # Wait for health check
            await asyncio.sleep(10)
            
            # Verify restart success
            container.reload()
            return container.status == 'running'
            
        except Exception as e:
            logger.error(f"Service restart failed for {service_name}: {e}")
            return False
    
    async def _rollback_service(self, service_name: str) -> bool:
        """Rollback service to previous version"""
        try:
            # This would typically involve docker-compose operations
            # For now, we'll simulate with a restart
            logger.info(f"Rolling back {service_name} to previous version")
            return await self._restart_service(service_name)
            
        except Exception as e:
            logger.error(f"Service rollback failed for {service_name}: {e}")
            return False
    
    async def _scale_up_service(self, service_name: str) -> bool:
        """Scale up service resources"""
        try:
            # Update container resource limits
            logger.info(f"Scaling up resources for {service_name}")
            
            # This would involve docker-compose scale operations
            # For now, we'll restart with updated resource limits
            return await self._restart_service(service_name)
            
        except Exception as e:
            logger.error(f"Service scale up failed for {service_name}: {e}")
            return False
    
    async def _failover_service(self, service_name: str) -> bool:
        """Failover to backup instance or alternative service"""
        try:
            logger.info(f"Initiating failover for {service_name}")
            
            # Notify orchestrator about failover
            await self.orchestrator.handle_service_failover(service_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Service failover failed for {service_name}: {e}")
            return False
    
    async def _escalate_to_human(self, service_name: str) -> bool:
        """Escalate issue to human intervention"""
        try:
            logger.critical(f"🚨 Escalating {service_name} failure to human intervention")
            
            # Send notification through external intelligence API
            await self.external_intel.send_alert({
                'type': 'service_failure_escalation',
                'service': service_name,
                'timestamp': datetime.now().isoformat(),
                'severity': 'critical'
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Escalation failed for {service_name}: {e}")
            return False
    
    async def _escalate_remediation(self, failure_event: FailureEvent):
        """Escalate to more aggressive remediation strategies"""
        service_name = failure_event.service_name
        
        # Try more aggressive strategies
        if RemediationType.RESTART in failure_event.remediation_attempted:
            await self.remediation_queue.put(FailureEvent(
                service_name=service_name,
                timestamp=datetime.now(),
                error_category=failure_event.error_category,
                error_message="Escalated remediation after restart failure",
                stack_trace=None,
                metrics_snapshot=failure_event.metrics_snapshot
            ))
    
    async def _notify_human_intervention(self, failure_event: FailureEvent):
        """Notify for human intervention"""
        await self.external_intel.send_critical_alert({
            'service': failure_event.service_name,
            'error_category': failure_event.error_category.value,
            'failed_remediations': [r.value for r in failure_event.remediation_attempted],
            'timestamp': failure_event.timestamp.isoformat()
        })
    
    async def _update_prometheus_metrics(self):
        """Update Prometheus metrics with current health status"""
        for service_name, state in self.service_states.items():
            # Update health status
            for s in ServiceState:
                value = 1 if state == s else 0
                self.metrics['service_health_status'].labels(
                    service=service_name, state=s.value
                ).set(value)
            
            # Update resource metrics if available
            if service_name in self.health_history and self.health_history[service_name]:
                latest_metrics = self.health_history[service_name][-1]
                
                self.metrics['memory_usage_percent'].labels(
                    service=service_name
                ).set(latest_metrics.memory_usage)
                
                self.metrics['cpu_usage_percent'].labels(
                    service=service_name
                ).set(latest_metrics.cpu_usage)
                
                self.metrics['service_response_time'].labels(
                    service=service_name
                ).observe(latest_metrics.response_time)
    
    async def _metrics_aggregator(self):
        """Aggregate and calculate success ratios"""
        while True:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                # Calculate success ratios for different time windows
                for window in ['1h', '24h', '7d']:
                    for service_name in self.service_states.keys():
                        success_ratio = await self._calculate_success_ratio(service_name, window)
                        self.metrics['autonomous_repair_success_ratio'].labels(
                            service=service_name, time_window=window
                        ).set(success_ratio)
                
            except Exception as e:
                logger.error(f"Metrics aggregation error: {e}")
    
    async def _calculate_success_ratio(self, service_name: str, time_window: str) -> float:
        """Calculate repair success ratio for a time window"""
        try:
            # Get failure history from episodic memory
            failures = await self.episodic_memory.get_service_failures(
                service_name, time_window
            )
            
            if not failures:
                return 1.0  # No failures = 100% success
            
            successful = sum(1 for f in failures if f.get('resolved', False))
            return successful / len(failures)
            
        except Exception:
            return 0.0
    
    async def _websocket_health_stream(self):
        """WebSocket server for real-time health streaming"""
        async def health_handler(websocket, path):
            try:
                while True:
                    health_summary = {
                        'timestamp': datetime.now().isoformat(),
                        'services': {
                            name: {
                                'state': state.value,
                                'metrics': self.health_history.get(name, [])[-1].__dict__ 
                                         if name in self.health_history and self.health_history[name] 
                                         else None
                            }
                            for name, state in self.service_states.items()
                        },
                        'active_failures': len(self.active_failures),
                        'remediation_queue_size': self.remediation_queue.qsize()
                    }
                    
                    await websocket.send(json.dumps(health_summary, default=str))
                    await asyncio.sleep(5)
                    
            except websockets.exceptions.ConnectionClosed:
                pass
            except Exception as e:
                logger.error(f"WebSocket health stream error: {e}")
        
        # Start WebSocket server
        start_server = websockets.serve(
            health_handler, 
            'localhost', 
            self.config.get('websocket_port', 9092)
        )
        
        await start_server
        logger.info("🔌 WebSocket health stream server started on port 9092")


# Factory function for easy initialization
async def create_autonomous_health_manager(config: Dict[str, Any]) -> AutonomousHealthManager:
    """Factory function to create and initialize health manager"""
    manager = AutonomousHealthManager(config)
    await manager.initialize()
    return manager


if __name__ == "__main__":
    # Example configuration
    config = {
        'monitoring_interval': 30,
        'health_threshold': 0.8,
        'max_remediation_attempts': 3,
        'escalation_timeout': 300,
        'metrics_port': 9091,
        'websocket_port': 9092,
        'redis_host': 'localhost',
        'redis_port': 6379
    }
    
    # Initialize and run
    async def main():
        manager = await create_autonomous_health_manager(config)
        await manager.initialize()
    
    asyncio.run(main())