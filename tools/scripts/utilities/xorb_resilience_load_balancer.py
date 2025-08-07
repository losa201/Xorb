#!/usr/bin/env python3
"""
XORB Resilience & Scalability Layer - Distributed Load Balancer
Advanced adaptive load distribution with auto-scaling and health monitoring
"""

import asyncio
import json
import time
import logging
import psutil
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import aiohttp
import numpy as np
from collections import defaultdict, deque
import uuid
import statistics
import socket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    ADAPTIVE_PERFORMANCE = "adaptive_performance"
    CONSISTENT_HASH = "consistent_hash"

class ScalingAction(Enum):
    """Auto-scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"

@dataclass
class ServiceInstance:
    """Service instance information"""
    instance_id: str
    service_name: str
    host: str
    port: int
    status: ServiceStatus
    weight: float = 1.0
    current_connections: int = 0
    max_connections: int = 1000
    response_time_avg: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    last_health_check: datetime = field(default_factory=datetime.now)
    health_check_failures: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LoadBalancingMetrics:
    """Load balancing performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    requests_per_second: float = 0.0
    active_connections: int = 0
    circuit_breaker_trips: int = 0
    auto_scaling_events: int = 0
    load_distribution_efficiency: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AutoScalingConfig:
    """Auto-scaling configuration"""
    min_instances: int = 2
    max_instances: int = 20
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    target_response_time_ms: float = 500.0
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    evaluation_period: int = 60  # seconds

class XORBResilienceLoadBalancer:
    """Advanced distributed load balancer with auto-scaling and resilience"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.load_balancer_id = str(uuid.uuid4())
        
        # Service registry
        self.service_instances: Dict[str, List[ServiceInstance]] = defaultdict(list)
        self.service_configs: Dict[str, Dict[str, Any]] = {}
        
        # Load balancing state
        self.current_strategy = LoadBalancingStrategy.ADAPTIVE_PERFORMANCE
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
        self.consistent_hash_rings: Dict[str, List[Tuple[int, str]]] = {}
        
        # Metrics and monitoring
        self.metrics: LoadBalancingMetrics = LoadBalancingMetrics()
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=60))
        
        # Auto-scaling
        self.auto_scaling_configs: Dict[str, AutoScalingConfig] = {}
        self.scaling_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.last_scale_actions: Dict[str, datetime] = {}
        
        # Health monitoring
        self.health_check_interval = self.config.get('health_check_interval', 30)
        self.health_check_timeout = self.config.get('health_check_timeout', 5)
        self.max_health_check_failures = self.config.get('max_health_check_failures', 3)
        
        # Performance optimization
        self.connection_pool_size = self.config.get('connection_pool_size', 100)
        self.request_timeout = self.config.get('request_timeout', 30)
        self.retry_attempts = self.config.get('retry_attempts', 3)
        
        # Initialize components
        self._initialize_default_services()
        self._initialize_auto_scaling_configs()
        
        logger.info(f"XORB Resilience Load Balancer initialized: {self.load_balancer_id}")
    
    def _initialize_default_services(self):
        """Initialize default XORB service configurations"""
        default_services = {
            "neural_orchestrator": {
                "port_range": [8003],
                "health_endpoint": "/health",
                "max_connections": 500,
                "expected_response_time": 200
            },
            "learning_service": {
                "port_range": [8004],
                "health_endpoint": "/health", 
                "max_connections": 300,
                "expected_response_time": 150
            },
            "threat_detection": {
                "port_range": [8005],
                "health_endpoint": "/health",
                "max_connections": 1000,
                "expected_response_time": 100
            },
            "agent_cluster": {
                "port_range": [8006],
                "health_endpoint": "/health",
                "max_connections": 800,
                "expected_response_time": 250
            },
            "intelligence_fusion": {
                "port_range": [8007],
                "health_endpoint": "/health",
                "max_connections": 400,
                "expected_response_time": 300
            },
            "evolution_accelerator": {
                "port_range": [8008],
                "health_endpoint": "/health",
                "max_connections": 200,
                "expected_response_time": 400
            }
        }
        
        for service_name, config in default_services.items():
            self.service_configs[service_name] = config
            
            # Register initial instances
            for port in config["port_range"]:
                instance = ServiceInstance(
                    instance_id=f"{service_name}_{port}_{int(time.time())}",
                    service_name=service_name,
                    host="localhost",
                    port=port,
                    status=ServiceStatus.HEALTHY,
                    max_connections=config["max_connections"],
                    metadata={"expected_response_time": config["expected_response_time"]}
                )
                self.service_instances[service_name].append(instance)
    
    def _initialize_auto_scaling_configs(self):
        """Initialize auto-scaling configurations for services"""
        for service_name in self.service_configs.keys():
            self.auto_scaling_configs[service_name] = AutoScalingConfig(
                min_instances=1,
                max_instances=5,
                target_cpu_utilization=75.0,
                target_memory_utilization=85.0,
                target_response_time_ms=self.service_configs[service_name]["expected_response_time"]
            )
    
    async def register_service_instance(self, service_name: str, host: str, port: int, 
                                      weight: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Register a new service instance"""
        try:
            instance_id = f"{service_name}_{host}_{port}_{int(time.time())}"
            
            instance = ServiceInstance(
                instance_id=instance_id,
                service_name=service_name,
                host=host,
                port=port,
                status=ServiceStatus.HEALTHY,
                weight=weight,
                metadata=metadata or {}
            )
            
            self.service_instances[service_name].append(instance)
            
            # Update consistent hash ring
            await self._update_consistent_hash_ring(service_name)
            
            logger.info(f"Registered service instance: {instance_id}")
            return instance_id
            
        except Exception as e:
            logger.error(f"Failed to register service instance: {e}")
            raise e
    
    async def deregister_service_instance(self, service_name: str, instance_id: str) -> bool:
        """Deregister a service instance"""
        try:
            instances = self.service_instances[service_name]
            instance_found = False
            
            for i, instance in enumerate(instances):
                if instance.instance_id == instance_id:
                    instances.pop(i)
                    instance_found = True
                    break
            
            if instance_found:
                # Update consistent hash ring
                await self._update_consistent_hash_ring(service_name)
                logger.info(f"Deregistered service instance: {instance_id}")
                return True
            else:
                logger.warning(f"Service instance not found: {instance_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to deregister service instance: {e}")
            return False
    
    async def select_service_instance(self, service_name: str, request_context: Optional[Dict[str, Any]] = None) -> Optional[ServiceInstance]:
        """Select the best service instance based on current strategy"""
        try:
            instances = [inst for inst in self.service_instances[service_name] 
                        if inst.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]]
            
            if not instances:
                logger.error(f"No healthy instances available for service: {service_name}")
                return None
            
            if self.current_strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return await self._select_round_robin(service_name, instances)
            elif self.current_strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return await self._select_weighted_round_robin(service_name, instances)
            elif self.current_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return await self._select_least_connections(instances)
            elif self.current_strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return await self._select_least_response_time(instances)
            elif self.current_strategy == LoadBalancingStrategy.ADAPTIVE_PERFORMANCE:
                return await self._select_adaptive_performance(instances)
            elif self.current_strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                return await self._select_consistent_hash(service_name, instances, request_context)
            else:
                return instances[0]  # Fallback
                
        except Exception as e:
            logger.error(f"Failed to select service instance: {e}")
            return None
    
    async def _select_round_robin(self, service_name: str, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round-robin selection"""
        counter = self.round_robin_counters[service_name]
        selected = instances[counter % len(instances)]
        self.round_robin_counters[service_name] = (counter + 1) % len(instances)
        return selected
    
    async def _select_weighted_round_robin(self, service_name: str, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted round-robin selection"""
        total_weight = sum(inst.weight for inst in instances)
        if total_weight == 0:
            return instances[0]
        
        # Create weighted selection based on weights
        weighted_instances = []
        for instance in instances:
            weight_ratio = int((instance.weight / total_weight) * 100)
            weighted_instances.extend([instance] * max(1, weight_ratio))
        
        counter = self.round_robin_counters[service_name]
        selected = weighted_instances[counter % len(weighted_instances)]
        self.round_robin_counters[service_name] = (counter + 1) % len(weighted_instances)
        return selected
    
    async def _select_least_connections(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least connections selection"""
        return min(instances, key=lambda inst: inst.current_connections)
    
    async def _select_least_response_time(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least response time selection"""
        return min(instances, key=lambda inst: inst.response_time_avg)
    
    async def _select_adaptive_performance(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Adaptive performance-based selection"""
        best_score = float('inf')
        best_instance = instances[0]
        
        for instance in instances:
            # Calculate composite score based on multiple metrics
            connection_score = instance.current_connections / instance.max_connections
            response_time_score = instance.response_time_avg / 1000.0  # Normalize to seconds
            cpu_score = instance.cpu_usage / 100.0
            memory_score = instance.memory_usage / 100.0
            error_score = instance.error_rate
            
            # Weighted composite score (lower is better)
            composite_score = (
                connection_score * 0.25 +
                response_time_score * 0.3 +
                cpu_score * 0.2 +
                memory_score * 0.15 +
                error_score * 0.1
            )
            
            if composite_score < best_score:
                best_score = composite_score
                best_instance = instance
        
        return best_instance
    
    async def _select_consistent_hash(self, service_name: str, instances: List[ServiceInstance], 
                                    request_context: Optional[Dict[str, Any]]) -> ServiceInstance:
        """Consistent hash selection for session affinity"""
        if not request_context or 'client_id' not in request_context:
            # Fall back to adaptive performance if no client context
            return await self._select_adaptive_performance(instances)
        
        client_id = request_context['client_id']
        hash_value = int(hashlib.md5(client_id.encode()).hexdigest(), 16)
        
        # Find the instance in the consistent hash ring
        if service_name not in self.consistent_hash_rings:
            await self._update_consistent_hash_ring(service_name)
        
        ring = self.consistent_hash_rings[service_name]
        if not ring:
            return instances[0]
        
        # Find the first instance with hash >= client hash
        for ring_hash, instance_id in ring:
            if ring_hash >= hash_value:
                for instance in instances:
                    if instance.instance_id == instance_id:
                        return instance
        
        # Wrap around to the first instance
        first_instance_id = ring[0][1]
        for instance in instances:
            if instance.instance_id == first_instance_id:
                return instance
        
        return instances[0]  # Fallback
    
    async def _update_consistent_hash_ring(self, service_name: str):
        """Update consistent hash ring for a service"""
        instances = self.service_instances[service_name]
        ring = []
        
        # Create multiple hash points for each instance (virtual nodes)
        virtual_nodes = 150
        for instance in instances:
            for i in range(virtual_nodes):
                hash_key = f"{instance.instance_id}:{i}"
                hash_value = int(hashlib.md5(hash_key.encode()).hexdigest(), 16)
                ring.append((hash_value, instance.instance_id))
        
        # Sort by hash value
        ring.sort(key=lambda x: x[0])
        self.consistent_hash_rings[service_name] = ring
    
    async def execute_request(self, service_name: str, endpoint: str, method: str = "GET", 
                            data: Optional[Dict[str, Any]] = None, 
                            request_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a request with load balancing and resilience"""
        start_time = time.time()
        
        try:
            # Select service instance
            instance = await self.select_service_instance(service_name, request_context)
            if not instance:
                return {
                    'success': False,
                    'error': 'No healthy service instances available',
                    'service_name': service_name
                }
            
            # Increment connection count
            instance.current_connections += 1
            
            # Execute request with retries
            result = await self._execute_request_with_retries(instance, endpoint, method, data)
            
            # Update metrics
            response_time = (time.time() - start_time) * 1000  # ms
            await self._update_instance_metrics(instance, response_time, result['success'])
            await self._update_load_balancer_metrics(result['success'])
            
            return result
            
        except Exception as e:
            logger.error(f"Request execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'service_name': service_name
            }
        finally:
            # Decrement connection count
            if 'instance' in locals() and instance:
                instance.current_connections = max(0, instance.current_connections - 1)
    
    async def _execute_request_with_retries(self, instance: ServiceInstance, endpoint: str, 
                                          method: str, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute request with retry logic"""
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                url = f"http://{instance.host}:{instance.port}{endpoint}"
                
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.request_timeout)
                ) as session:
                    if method.upper() == "GET":
                        async with session.get(url) as response:
                            result_data = await response.json() if response.content_type == 'application/json' else await response.text()
                            return {
                                'success': response.status < 400,
                                'status_code': response.status,
                                'data': result_data,
                                'instance_id': instance.instance_id,
                                'attempt': attempt + 1
                            }
                    elif method.upper() == "POST":
                        async with session.post(url, json=data) as response:
                            result_data = await response.json() if response.content_type == 'application/json' else await response.text()
                            return {
                                'success': response.status < 400,
                                'status_code': response.status,
                                'data': result_data,
                                'instance_id': instance.instance_id,
                                'attempt': attempt + 1
                            }
                    else:
                        return {
                            'success': False,
                            'error': f'Unsupported HTTP method: {method}',
                            'instance_id': instance.instance_id
                        }
                        
            except asyncio.TimeoutError:
                last_error = f"Request timeout (attempt {attempt + 1})"
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                last_error = f"Request failed: {str(e)} (attempt {attempt + 1})"
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return {
            'success': False,
            'error': f"All retry attempts failed. Last error: {last_error}",
            'instance_id': instance.instance_id
        }
    
    async def _update_instance_metrics(self, instance: ServiceInstance, response_time: float, success: bool):
        """Update instance performance metrics"""
        try:
            # Update response time (moving average)
            if instance.response_time_avg == 0:
                instance.response_time_avg = response_time
            else:
                instance.response_time_avg = (instance.response_time_avg * 0.9) + (response_time * 0.1)
            
            # Update error rate
            error_occurred = not success
            if hasattr(instance, 'error_count'):
                instance.error_count = (instance.error_count * 0.95) + (1 if error_occurred else 0)
                instance.error_rate = instance.error_count / 20  # Normalize
            else:
                instance.error_count = 1 if error_occurred else 0
                instance.error_rate = instance.error_count
            
            # Update system metrics
            try:
                instance.cpu_usage = psutil.cpu_percent(interval=None)
                instance.memory_usage = psutil.virtual_memory().percent
            except:
                pass  # Metrics collection is optional
            
            # Store response time for analysis
            service_name = instance.service_name
            self.response_times[service_name].append(response_time)
            
        except Exception as e:
            logger.error(f"Failed to update instance metrics: {e}")
    
    async def _update_load_balancer_metrics(self, success: bool):
        """Update load balancer metrics"""
        try:
            self.metrics.total_requests += 1
            if success:
                self.metrics.successful_requests += 1
            else:
                self.metrics.failed_requests += 1
            
            # Update requests per second
            current_time = time.time()
            if not hasattr(self, '_last_rps_update'):
                self._last_rps_update = current_time
                self._request_count_window = 0
            
            self._request_count_window += 1
            
            if current_time - self._last_rps_update >= 1.0:  # Update every second
                self.metrics.requests_per_second = self._request_count_window / (current_time - self._last_rps_update)
                self._last_rps_update = current_time
                self._request_count_window = 0
            
            # Update average response time
            all_response_times = []
            for service_times in self.response_times.values():
                all_response_times.extend(list(service_times))
            
            if all_response_times:
                self.metrics.avg_response_time = statistics.mean(all_response_times)
            
            self.metrics.timestamp = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to update load balancer metrics: {e}")
    
    async def perform_health_checks(self):
        """Perform health checks on all service instances"""
        try:
            health_check_tasks = []
            
            for service_name, instances in self.service_instances.items():
                for instance in instances:
                    task = self._health_check_instance(service_name, instance)
                    health_check_tasks.append(task)
            
            if health_check_tasks:
                await asyncio.gather(*health_check_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Health check execution failed: {e}")
    
    async def _health_check_instance(self, service_name: str, instance: ServiceInstance):
        """Perform health check on a single instance"""
        try:
            health_endpoint = self.service_configs.get(service_name, {}).get('health_endpoint', '/health')
            url = f"http://{instance.host}:{instance.port}{health_endpoint}"
            
            start_time = time.time()
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.health_check_timeout)
            ) as session:
                async with session.get(url) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        instance.status = ServiceStatus.HEALTHY
                        instance.health_check_failures = 0
                        instance.last_health_check = datetime.now()
                        
                        # Update response time
                        if instance.response_time_avg == 0:
                            instance.response_time_avg = response_time
                        else:
                            instance.response_time_avg = (instance.response_time_avg * 0.8) + (response_time * 0.2)
                    else:
                        await self._handle_health_check_failure(instance)
                        
        except asyncio.TimeoutError:
            logger.warning(f"Health check timeout for {instance.instance_id}")
            await self._handle_health_check_failure(instance)
        except Exception as e:
            logger.warning(f"Health check failed for {instance.instance_id}: {e}")
            await self._handle_health_check_failure(instance)
    
    async def _handle_health_check_failure(self, instance: ServiceInstance):
        """Handle health check failure"""
        instance.health_check_failures += 1
        instance.last_health_check = datetime.now()
        
        if instance.health_check_failures >= self.max_health_check_failures:
            if instance.status != ServiceStatus.OFFLINE:
                logger.error(f"Marking instance as offline: {instance.instance_id}")
                instance.status = ServiceStatus.OFFLINE
        elif instance.health_check_failures >= self.max_health_check_failures // 2:
            if instance.status != ServiceStatus.UNHEALTHY:
                logger.warning(f"Marking instance as unhealthy: {instance.instance_id}")
                instance.status = ServiceStatus.UNHEALTHY
    
    async def evaluate_auto_scaling(self):
        """Evaluate auto-scaling needs for all services"""
        try:
            for service_name in self.service_instances.keys():
                if service_name in self.auto_scaling_configs:
                    await self._evaluate_service_scaling(service_name)
                    
        except Exception as e:
            logger.error(f"Auto-scaling evaluation failed: {e}")
    
    async def _evaluate_service_scaling(self, service_name: str):
        """Evaluate scaling needs for a specific service"""
        try:
            config = self.auto_scaling_configs[service_name]
            instances = [inst for inst in self.service_instances[service_name] 
                        if inst.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]]
            
            if not instances:
                return
            
            # Calculate current metrics averages
            avg_cpu = statistics.mean([inst.cpu_usage for inst in instances])
            avg_memory = statistics.mean([inst.memory_usage for inst in instances])
            avg_response_time = statistics.mean([inst.response_time_avg for inst in instances])
            
            # Determine scaling action
            scaling_action = ScalingAction.MAINTAIN
            
            # Scale up conditions
            if (avg_cpu > config.target_cpu_utilization * config.scale_up_threshold or
                avg_memory > config.target_memory_utilization * config.scale_up_threshold or
                avg_response_time > config.target_response_time_ms * config.scale_up_threshold):
                
                if len(instances) < config.max_instances:
                    scaling_action = ScalingAction.SCALE_UP
            
            # Scale down conditions
            elif (avg_cpu < config.target_cpu_utilization * config.scale_down_threshold and
                  avg_memory < config.target_memory_utilization * config.scale_down_threshold and
                  avg_response_time < config.target_response_time_ms * config.scale_down_threshold):
                
                if len(instances) > config.min_instances:
                    scaling_action = ScalingAction.SCALE_DOWN
            
            # Execute scaling action if cooldown period has passed
            await self._execute_scaling_action(service_name, scaling_action, config)
            
        except Exception as e:
            logger.error(f"Service scaling evaluation failed for {service_name}: {e}")
    
    async def _execute_scaling_action(self, service_name: str, action: ScalingAction, config: AutoScalingConfig):
        """Execute scaling action with cooldown enforcement"""
        try:
            if action == ScalingAction.MAINTAIN:
                return
            
            current_time = datetime.now()
            last_action_time = self.last_scale_actions.get(service_name)
            
            # Check cooldown period
            if last_action_time:
                time_since_last_action = (current_time - last_action_time).total_seconds()
                required_cooldown = config.scale_up_cooldown if action == ScalingAction.SCALE_UP else config.scale_down_cooldown
                
                if time_since_last_action < required_cooldown:
                    logger.debug(f"Scaling action {action.value} for {service_name} skipped due to cooldown")
                    return
            
            # Execute scaling action
            if action == ScalingAction.SCALE_UP:
                await self._scale_up_service(service_name)
            elif action == ScalingAction.SCALE_DOWN:
                await self._scale_down_service(service_name)
            
            # Record scaling event
            self.last_scale_actions[service_name] = current_time
            self.metrics.auto_scaling_events += 1
            
            scaling_event = {
                'timestamp': current_time.isoformat(),
                'service_name': service_name,
                'action': action.value,
                'instance_count_before': len(self.service_instances[service_name]),
                'instance_count_after': len(self.service_instances[service_name])
            }
            
            self.scaling_history[service_name].append(scaling_event)
            
            logger.info(f"Executed scaling action {action.value} for service {service_name}")
            
        except Exception as e:
            logger.error(f"Scaling action execution failed: {e}")
    
    async def _scale_up_service(self, service_name: str):
        """Scale up a service by adding a new instance"""
        try:
            # For demo purposes, we'll simulate adding a new instance
            # In production, this would trigger container orchestration
            
            existing_instances = self.service_instances[service_name]
            if not existing_instances:
                return
            
            # Use the configuration to determine new port
            base_port = existing_instances[0].port
            new_port = base_port + len(existing_instances)
            
            # Register new instance
            await self.register_service_instance(
                service_name=service_name,
                host="localhost",
                port=new_port,
                weight=1.0,
                metadata={"scaled_instance": True, "created_at": datetime.now().isoformat()}
            )
            
            logger.info(f"Scaled up {service_name}: added instance on port {new_port}")
            
        except Exception as e:
            logger.error(f"Scale up failed for {service_name}: {e}")
    
    async def _scale_down_service(self, service_name: str):
        """Scale down a service by removing an instance"""
        try:
            instances = [inst for inst in self.service_instances[service_name] 
                        if inst.status != ServiceStatus.OFFLINE]
            
            if len(instances) <= self.auto_scaling_configs[service_name].min_instances:
                return
            
            # Remove the instance with lowest performance score
            instance_to_remove = min(instances, key=lambda inst: (
                inst.current_connections + 
                inst.response_time_avg / 1000 + 
                inst.cpu_usage / 100 + 
                inst.memory_usage / 100
            ))
            
            await self.deregister_service_instance(service_name, instance_to_remove.instance_id)
            
            logger.info(f"Scaled down {service_name}: removed instance {instance_to_remove.instance_id}")
            
        except Exception as e:
            logger.error(f"Scale down failed for {service_name}: {e}")
    
    async def get_load_balancer_status(self) -> Dict[str, Any]:
        """Get comprehensive load balancer status"""
        try:
            service_stats = {}
            
            for service_name, instances in self.service_instances.items():
                healthy_count = len([inst for inst in instances if inst.status == ServiceStatus.HEALTHY])
                degraded_count = len([inst for inst in instances if inst.status == ServiceStatus.DEGRADED])
                unhealthy_count = len([inst for inst in instances if inst.status == ServiceStatus.UNHEALTHY])
                offline_count = len([inst for inst in instances if inst.status == ServiceStatus.OFFLINE])
                
                total_connections = sum(inst.current_connections for inst in instances)
                avg_response_time = statistics.mean([inst.response_time_avg for inst in instances]) if instances else 0.0
                avg_cpu = statistics.mean([inst.cpu_usage for inst in instances]) if instances else 0.0
                avg_memory = statistics.mean([inst.memory_usage for inst in instances]) if instances else 0.0
                
                service_stats[service_name] = {
                    'total_instances': len(instances),
                    'healthy_instances': healthy_count,
                    'degraded_instances': degraded_count,
                    'unhealthy_instances': unhealthy_count,
                    'offline_instances': offline_count,
                    'total_connections': total_connections,
                    'avg_response_time_ms': avg_response_time,
                    'avg_cpu_usage': avg_cpu,
                    'avg_memory_usage': avg_memory,
                    'recent_scaling_events': len([event for event in self.scaling_history[service_name] 
                                                if (datetime.now() - datetime.fromisoformat(event['timestamp'])).total_seconds() < 3600])
                }
            
            return {
                'load_balancer_id': self.load_balancer_id,
                'current_strategy': self.current_strategy.value,
                'metrics': asdict(self.metrics),
                'service_statistics': service_stats,
                'auto_scaling_configs': {name: asdict(config) for name, config in self.auto_scaling_configs.items()},
                'health_check_config': {
                    'interval_seconds': self.health_check_interval,
                    'timeout_seconds': self.health_check_timeout,
                    'max_failures': self.max_health_check_failures
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get load balancer status: {e}")
            return {'error': str(e)}
    
    async def start_background_tasks(self):
        """Start background monitoring and scaling tasks"""
        try:
            # Health check task
            async def health_check_loop():
                while True:
                    await self.perform_health_checks()
                    await asyncio.sleep(self.health_check_interval)
            
            # Auto-scaling evaluation task
            async def auto_scaling_loop():
                while True:
                    await self.evaluate_auto_scaling()
                    await asyncio.sleep(60)  # Evaluate every minute
            
            # Start background tasks
            asyncio.create_task(health_check_loop())
            asyncio.create_task(auto_scaling_loop())
            
            logger.info("Background monitoring and scaling tasks started")
            
        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")

# Example usage and testing
async def main():
    """Example usage of XORB Resilience Load Balancer"""
    try:
        # Initialize load balancer
        load_balancer = XORBResilienceLoadBalancer({
            'health_check_interval': 30,
            'health_check_timeout': 5,
            'max_health_check_failures': 3,
            'retry_attempts': 3
        })
        
        print("🔄 XORB Resilience Load Balancer initialized")
        
        # Start background tasks
        await load_balancer.start_background_tasks()
        
        print("⚡ Background monitoring tasks started")
        
        # Perform initial health checks
        print("\n🏥 Performing initial health checks...")
        await load_balancer.perform_health_checks()
        
        # Test load balancing
        print("\n🎯 Testing load balancing...")
        for i in range(5):
            result = await load_balancer.execute_request(
                service_name="neural_orchestrator",
                endpoint="/health",
                method="GET"
            )
            
            if result['success']:
                print(f"✅ Request {i+1}: Success - Instance {result.get('instance_id', 'unknown')}")
            else:
                print(f"❌ Request {i+1}: Failed - {result.get('error', 'Unknown error')}")
        
        # Test auto-scaling evaluation
        print("\n📈 Evaluating auto-scaling...")
        await load_balancer.evaluate_auto_scaling()
        
        # Get status
        status = await load_balancer.get_load_balancer_status()
        print(f"\n📊 Load Balancer Status:")
        print(f"- Strategy: {status['current_strategy']}")
        print(f"- Total Requests: {status['metrics']['total_requests']}")
        print(f"- Success Rate: {status['metrics']['successful_requests']/max(1, status['metrics']['total_requests'])*100:.1f}%")
        print(f"- Avg Response Time: {status['metrics']['avg_response_time']:.2f}ms")
        print(f"- Auto-scaling Events: {status['metrics']['auto_scaling_events']}")
        
        for service_name, stats in status['service_statistics'].items():
            print(f"\n🔧 {service_name}:")
            print(f"  - Instances: {stats['healthy_instances']}/{stats['total_instances']} healthy")
            print(f"  - Connections: {stats['total_connections']}")
            print(f"  - Avg Response: {stats['avg_response_time_ms']:.2f}ms")
        
        print(f"\n✅ XORB Resilience Load Balancer demonstration completed!")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())