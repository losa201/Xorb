"""
Dynamic Resource Allocation and Scaling Manager

This module provides intelligent resource allocation, auto-scaling, and workload management
for the XORB ecosystem with support for EPYC optimization, Kubernetes integration,
and predictive scaling based on workload patterns.
"""

import asyncio
import json
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict, deque

import psutil
import structlog
from prometheus_client import Counter, Gauge, Histogram

# Metrics
RESOURCE_ALLOCATION_COUNTER = Counter('xorb_resource_allocation_total', 'Resource allocations', ['type', 'status'])
CPU_UTILIZATION_GAUGE = Gauge('xorb_cpu_utilization', 'CPU utilization percentage')
MEMORY_UTILIZATION_GAUGE = Gauge('xorb_memory_utilization', 'Memory utilization percentage')
ACTIVE_CAMPAIGNS_GAUGE = Gauge('xorb_active_campaigns', 'Number of active campaigns')
AGENT_QUEUE_SIZE_GAUGE = Gauge('xorb_agent_queue_size', 'Agent queue size')
SCALING_DECISIONS_COUNTER = Counter('xorb_scaling_decisions_total', 'Scaling decisions', ['direction'])
RESOURCE_ALLOCATION_TIME = Histogram('xorb_resource_allocation_duration_seconds', 'Resource allocation time')

logger = structlog.get_logger(__name__)


class ResourceType(Enum):
    """Resource types for allocation."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    AGENTS = "agents"


class ScalingDirection(Enum):
    """Scaling direction enumeration."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class WorkloadPattern(Enum):
    """Workload pattern types."""
    STEADY = "steady"
    BURST = "burst"
    CYCLICAL = "cyclical"
    UNPREDICTABLE = "unpredictable"


@dataclass
class ResourceQuota:
    """Resource quota definition."""
    cpu_cores: float
    memory_gb: float
    disk_gb: float
    max_agents: int
    network_bandwidth_mbps: Optional[float] = None
    gpu_count: Optional[int] = None


@dataclass
class ResourceUsage:
    """Current resource usage metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io_mbps: float
    active_agents: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    min_instances: int
    max_instances: int
    target_cpu_percent: float = 70.0
    target_memory_percent: float = 80.0
    scale_up_threshold: float = 85.0
    scale_down_threshold: float = 50.0
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    predictive_scaling: bool = True
    burst_detection: bool = True


@dataclass
class WorkloadMetrics:
    """Workload performance metrics."""
    throughput: float  # operations per second
    latency_p95: float  # 95th percentile latency in ms
    error_rate: float  # error rate percentage
    queue_depth: int
    timestamp: float = field(default_factory=time.time)


class IResourceProvider(ABC):
    """Interface for resource providers."""
    
    @abstractmethod
    async def get_available_resources(self) -> ResourceQuota:
        """Get available resources."""
        pass
    
    @abstractmethod
    async def allocate_resources(self, quota: ResourceQuota) -> bool:
        """Allocate resources."""
        pass
    
    @abstractmethod
    async def deallocate_resources(self, quota: ResourceQuota) -> bool:
        """Deallocate resources."""
        pass
    
    @abstractmethod
    async def scale_instances(self, target_count: int) -> bool:
        """Scale instances to target count."""
        pass


class LocalResourceProvider(IResourceProvider):
    """Local resource provider for single-node deployments."""
    
    def __init__(self, max_quota: ResourceQuota):
        self.max_quota = max_quota
        self.allocated_quota = ResourceQuota(0, 0, 0, 0)
    
    async def get_available_resources(self) -> ResourceQuota:
        """Get available local resources."""
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return ResourceQuota(
            cpu_cores=cpu_count - self.allocated_quota.cpu_cores,
            memory_gb=(memory.total - memory.used) / (1024**3),
            disk_gb=(disk.total - disk.used) / (1024**3),
            max_agents=self.max_quota.max_agents - self.allocated_quota.max_agents
        )
    
    async def allocate_resources(self, quota: ResourceQuota) -> bool:
        """Allocate local resources."""
        available = await self.get_available_resources()
        
        if (quota.cpu_cores <= available.cpu_cores and
            quota.memory_gb <= available.memory_gb and
            quota.max_agents <= available.max_agents):
            
            self.allocated_quota.cpu_cores += quota.cpu_cores
            self.allocated_quota.memory_gb += quota.memory_gb
            self.allocated_quota.max_agents += quota.max_agents
            
            RESOURCE_ALLOCATION_COUNTER.labels(type="local", status="success").inc()
            return True
        
        RESOURCE_ALLOCATION_COUNTER.labels(type="local", status="failed").inc()
        return False
    
    async def deallocate_resources(self, quota: ResourceQuota) -> bool:
        """Deallocate local resources."""
        self.allocated_quota.cpu_cores = max(0, self.allocated_quota.cpu_cores - quota.cpu_cores)
        self.allocated_quota.memory_gb = max(0, self.allocated_quota.memory_gb - quota.memory_gb)
        self.allocated_quota.max_agents = max(0, self.allocated_quota.max_agents - quota.max_agents)
        return True
    
    async def scale_instances(self, target_count: int) -> bool:
        """Scale instances (no-op for local provider)."""
        return True


class KubernetesResourceProvider(IResourceProvider):
    """Kubernetes resource provider for container orchestration."""
    
    def __init__(self, namespace: str = "xorb", deployment_name: str = "xorb-workers"):
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.k8s_client = None
    
    async def _init_k8s_client(self):
        """Initialize Kubernetes client."""
        try:
            from kubernetes import client, config
            config.load_incluster_config()  # For in-cluster usage
            self.k8s_client = client.AppsV1Api()
        except Exception:
            try:
                config.load_kube_config()  # For local development
                self.k8s_client = client.AppsV1Api()
            except Exception as e:
                logger.error("Failed to initialize Kubernetes client", error=str(e))
                raise
    
    async def get_available_resources(self) -> ResourceQuota:
        """Get available Kubernetes resources."""
        if not self.k8s_client:
            await self._init_k8s_client()
        
        try:
            # Get cluster resources (simplified)
            return ResourceQuota(
                cpu_cores=1000,  # Large cluster assumption
                memory_gb=1000,
                disk_gb=10000,
                max_agents=1000
            )
        except Exception as e:
            logger.error("Failed to get K8s resources", error=str(e))
            return ResourceQuota(0, 0, 0, 0)
    
    async def allocate_resources(self, quota: ResourceQuota) -> bool:
        """Allocate Kubernetes resources."""
        # Implementation would create/update deployments with resource requests
        RESOURCE_ALLOCATION_COUNTER.labels(type="kubernetes", status="success").inc()
        return True
    
    async def deallocate_resources(self, quota: ResourceQuota) -> bool:
        """Deallocate Kubernetes resources."""
        return True
    
    async def scale_instances(self, target_count: int) -> bool:
        """Scale Kubernetes deployment."""
        if not self.k8s_client:
            await self._init_k8s_client()
        
        try:
            # Update deployment replica count
            body = {'spec': {'replicas': target_count}}
            self.k8s_client.patch_namespaced_deployment_scale(
                name=self.deployment_name,
                namespace=self.namespace,
                body=body
            )
            
            SCALING_DECISIONS_COUNTER.labels(direction="kubernetes").inc()
            logger.info("Scaled Kubernetes deployment", target_count=target_count)
            return True
            
        except Exception as e:
            logger.error("Failed to scale K8s deployment", error=str(e))
            return False


class WorkloadPredictor:
    """Predictive workload analyzer for proactive scaling."""
    
    def __init__(self, history_size: int = 1440):  # 24 hours of minute-by-minute data
        self.history_size = history_size
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.workload_history = deque(maxlen=history_size)
        self.pattern_cache = {}
    
    def add_sample(self, usage: ResourceUsage, workload: WorkloadMetrics):
        """Add a new sample to the history."""
        self.cpu_history.append(usage.cpu_percent)
        self.memory_history.append(usage.memory_percent)
        self.workload_history.append(workload)
    
    def detect_pattern(self) -> WorkloadPattern:
        """Detect workload pattern from historical data."""
        if len(self.cpu_history) < 60:  # Need at least 1 hour of data
            return WorkloadPattern.UNPREDICTABLE
        
        # Calculate pattern metrics
        cpu_variance = self._calculate_variance(list(self.cpu_history))
        cpu_trend = self._calculate_trend(list(self.cpu_history))
        
        # Detect cyclical patterns
        if self._detect_cyclical_pattern():
            return WorkloadPattern.CYCLICAL
        
        # Detect burst patterns
        if cpu_variance > 200 and self._detect_burst_pattern():
            return WorkloadPattern.BURST
        
        # Detect steady patterns
        if cpu_variance < 50 and abs(cpu_trend) < 0.1:
            return WorkloadPattern.STEADY
        
        return WorkloadPattern.UNPREDICTABLE
    
    def predict_next_usage(self, horizon_minutes: int = 30) -> Tuple[float, float]:
        """Predict CPU and memory usage for the next horizon."""
        if len(self.cpu_history) < 30:
            # Not enough data, use current values
            return (
                self.cpu_history[-1] if self.cpu_history else 50.0,
                self.memory_history[-1] if self.memory_history else 50.0
            )
        
        pattern = self.detect_pattern()
        
        if pattern == WorkloadPattern.CYCLICAL:
            return self._predict_cyclical(horizon_minutes)
        elif pattern == WorkloadPattern.BURST:
            return self._predict_burst(horizon_minutes)
        elif pattern == WorkloadPattern.STEADY:
            return self._predict_steady(horizon_minutes)
        else:
            return self._predict_unpredictable(horizon_minutes)
    
    def _calculate_variance(self, data: List[float]) -> float:
        """Calculate variance of data."""
        if not data:
            return 0
        mean = sum(data) / len(data)
        return sum((x - mean) ** 2 for x in data) / len(data)
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate trend (slope) of data."""
        if len(data) < 2:
            return 0
        
        n = len(data)
        x_sum = sum(range(n))
        y_sum = sum(data)
        xy_sum = sum(i * data[i] for i in range(n))
        x2_sum = sum(i ** 2 for i in range(n))
        
        denominator = n * x2_sum - x_sum ** 2
        if denominator == 0:
            return 0
        
        return (n * xy_sum - x_sum * y_sum) / denominator
    
    def _detect_cyclical_pattern(self) -> bool:
        """Detect if there's a cyclical pattern in the data."""
        # Simple autocorrelation check for daily patterns
        if len(self.cpu_history) < 1440:  # Need full day
            return False
        
        data = list(self.cpu_history)
        daily_correlation = self._autocorrelation(data, 1440)
        hourly_correlation = self._autocorrelation(data, 60)
        
        return daily_correlation > 0.7 or hourly_correlation > 0.8
    
    def _detect_burst_pattern(self) -> bool:
        """Detect burst patterns in the data."""
        recent_data = list(self.cpu_history)[-60:]  # Last hour
        if len(recent_data) < 10:
            return False
        
        mean_usage = sum(recent_data) / len(recent_data)
        spikes = [x for x in recent_data if x > mean_usage * 1.5]
        
        return len(spikes) > len(recent_data) * 0.1  # More than 10% spikes
    
    def _autocorrelation(self, data: List[float], lag: int) -> float:
        """Calculate autocorrelation at given lag."""
        if len(data) < lag + 1:
            return 0
        
        n = len(data) - lag
        mean = sum(data) / len(data)
        
        numerator = sum((data[i] - mean) * (data[i + lag] - mean) for i in range(n))
        denominator = sum((x - mean) ** 2 for x in data)
        
        return numerator / denominator if denominator > 0 else 0
    
    def _predict_cyclical(self, horizon_minutes: int) -> Tuple[float, float]:
        """Predict for cyclical patterns."""
        # Use historical data from same time of day/week
        current_minute = int(time.time() / 60) % 1440
        target_minute = (current_minute + horizon_minutes) % 1440
        
        # Find similar time periods
        similar_periods = []
        for i, usage in enumerate(self.cpu_history):
            if i % 1440 == target_minute:
                similar_periods.append((usage, self.memory_history[i]))
        
        if similar_periods:
            cpu_pred = sum(x[0] for x in similar_periods) / len(similar_periods)
            mem_pred = sum(x[1] for x in similar_periods) / len(similar_periods)
            return cpu_pred, mem_pred
        
        return self._predict_steady(horizon_minutes)
    
    def _predict_burst(self, horizon_minutes: int) -> Tuple[float, float]:
        """Predict for burst patterns."""
        # Conservative prediction - assume burst could happen
        recent_cpu = list(self.cpu_history)[-30:]
        recent_mem = list(self.memory_history)[-30:]
        
        if recent_cpu and recent_mem:
            max_cpu = max(recent_cpu)
            max_mem = max(recent_mem)
            avg_cpu = sum(recent_cpu) / len(recent_cpu)
            avg_mem = sum(recent_mem) / len(recent_mem)
            
            # Predict between average and max (conservative)
            cpu_pred = avg_cpu + (max_cpu - avg_cpu) * 0.7
            mem_pred = avg_mem + (max_mem - avg_mem) * 0.7
            
            return cpu_pred, mem_pred
        
        return 75.0, 75.0  # Conservative default
    
    def _predict_steady(self, horizon_minutes: int) -> Tuple[float, float]:
        """Predict for steady patterns."""
        if self.cpu_history and self.memory_history:
            # Use moving average with slight trend
            recent_cpu = list(self.cpu_history)[-30:]
            recent_mem = list(self.memory_history)[-30:]
            
            cpu_pred = sum(recent_cpu) / len(recent_cpu)
            mem_pred = sum(recent_mem) / len(recent_mem)
            
            # Apply trend
            cpu_trend = self._calculate_trend(recent_cpu)
            mem_trend = self._calculate_trend(recent_mem)
            
            cpu_pred += cpu_trend * horizon_minutes
            mem_pred += mem_trend * horizon_minutes
            
            return cpu_pred, mem_pred
        
        return 50.0, 50.0
    
    def _predict_unpredictable(self, horizon_minutes: int) -> Tuple[float, float]:
        """Predict for unpredictable patterns."""
        # Use conservative estimates
        if self.cpu_history and self.memory_history:
            recent_cpu = list(self.cpu_history)[-10:]
            recent_mem = list(self.memory_history)[-10:]
            
            cpu_pred = sum(recent_cpu) / len(recent_cpu) * 1.2  # 20% buffer
            mem_pred = sum(recent_mem) / len(recent_mem) * 1.2
            
            return min(cpu_pred, 95.0), min(mem_pred, 95.0)
        
        return 60.0, 60.0


class DynamicResourceManager:
    """Dynamic resource allocation and scaling manager."""
    
    def __init__(self, resource_provider: IResourceProvider, scaling_policy: ScalingPolicy):
        self.resource_provider = resource_provider
        self.scaling_policy = scaling_policy
        self.predictor = WorkloadPredictor()
        
        self.current_instances = scaling_policy.min_instances
        self.last_scale_up = 0
        self.last_scale_down = 0
        self.running = False
        
        # Resource allocation tracking
        self.allocated_quotas: Dict[str, ResourceQuota] = {}
        self.resource_history = deque(maxlen=1440)  # 24 hours
        
    async def start_resource_management(self):
        """Start the resource management service."""
        self.running = True
        
        # Start management tasks
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        scaling_task = asyncio.create_task(self._scaling_loop())
        
        logger.info("Dynamic resource management started")
        
        try:
            await asyncio.gather(monitoring_task, scaling_task)
        except asyncio.CancelledError:
            logger.info("Dynamic resource management stopped")
    
    async def stop_resource_management(self):
        """Stop the resource management service."""
        self.running = False
    
    async def _monitoring_loop(self):
        """Resource monitoring loop."""
        while self.running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(60)  # Collect metrics every minute
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(60)
    
    async def _scaling_loop(self):
        """Auto-scaling decision loop."""
        while self.running:
            try:
                if self.scaling_policy.predictive_scaling:
                    await self._predictive_scaling_decision()
                else:
                    await self._reactive_scaling_decision()
                
                await asyncio.sleep(120)  # Check scaling every 2 minutes
            except Exception as e:
                logger.error("Error in scaling loop", error=str(e))
                await asyncio.sleep(120)
    
    async def _collect_metrics(self):
        """Collect current resource usage metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            usage = ResourceUsage(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent / disk.total * 100,
                network_io_mbps=0,  # Simplified
                active_agents=len(self.allocated_quotas)
            )
            
            # Workload metrics (simplified)
            workload = WorkloadMetrics(
                throughput=100,  # Would be calculated from actual metrics
                latency_p95=50,
                error_rate=0.1,
                queue_depth=10
            )
            
            # Update metrics
            CPU_UTILIZATION_GAUGE.set(usage.cpu_percent)
            MEMORY_UTILIZATION_GAUGE.set(usage.memory_percent)
            ACTIVE_CAMPAIGNS_GAUGE.set(len(self.allocated_quotas))
            
            # Add to history and predictor
            self.resource_history.append(usage)
            self.predictor.add_sample(usage, workload)
            
        except Exception as e:
            logger.error("Failed to collect metrics", error=str(e))
    
    async def _reactive_scaling_decision(self):
        """Make reactive scaling decisions based on current metrics."""
        if not self.resource_history:
            return
        
        current_usage = self.resource_history[-1]
        current_time = time.time()
        
        # Scale up decision
        if (current_usage.cpu_percent > self.scaling_policy.scale_up_threshold or
            current_usage.memory_percent > self.scaling_policy.scale_up_threshold):
            
            if (current_time - self.last_scale_up > self.scaling_policy.scale_up_cooldown and
                self.current_instances < self.scaling_policy.max_instances):
                
                await self._scale_up()
        
        # Scale down decision
        elif (current_usage.cpu_percent < self.scaling_policy.scale_down_threshold and
              current_usage.memory_percent < self.scaling_policy.scale_down_threshold):
            
            if (current_time - self.last_scale_down > self.scaling_policy.scale_down_cooldown and
                self.current_instances > self.scaling_policy.min_instances):
                
                await self._scale_down()
    
    async def _predictive_scaling_decision(self):
        """Make predictive scaling decisions based on forecasted metrics."""
        if len(self.resource_history) < 30:  # Need sufficient history
            await self._reactive_scaling_decision()
            return
        
        # Predict usage for next 30 minutes
        predicted_cpu, predicted_memory = self.predictor.predict_next_usage(30)
        current_time = time.time()
        
        logger.debug("Predictive scaling", 
                    predicted_cpu=predicted_cpu, 
                    predicted_memory=predicted_memory)
        
        # Predictive scale up
        if (predicted_cpu > self.scaling_policy.scale_up_threshold or
            predicted_memory > self.scaling_policy.scale_up_threshold):
            
            if (current_time - self.last_scale_up > self.scaling_policy.scale_up_cooldown and
                self.current_instances < self.scaling_policy.max_instances):
                
                logger.info("Predictive scale up triggered", 
                           predicted_cpu=predicted_cpu,
                           predicted_memory=predicted_memory)
                await self._scale_up()
        
        # Predictive scale down (more conservative)
        elif (predicted_cpu < self.scaling_policy.scale_down_threshold * 0.8 and
              predicted_memory < self.scaling_policy.scale_down_threshold * 0.8):
            
            if (current_time - self.last_scale_down > self.scaling_policy.scale_down_cooldown and
                self.current_instances > self.scaling_policy.min_instances):
                
                logger.info("Predictive scale down triggered",
                           predicted_cpu=predicted_cpu,
                           predicted_memory=predicted_memory)
                await self._scale_down()
    
    async def _scale_up(self):
        """Scale up resources."""
        new_count = min(self.current_instances + 1, self.scaling_policy.max_instances)
        
        if await self.resource_provider.scale_instances(new_count):
            self.current_instances = new_count
            self.last_scale_up = time.time()
            
            SCALING_DECISIONS_COUNTER.labels(direction="up").inc()
            logger.info("Scaled up", instances=new_count)
    
    async def _scale_down(self):
        """Scale down resources."""
        new_count = max(self.current_instances - 1, self.scaling_policy.min_instances)
        
        if await self.resource_provider.scale_instances(new_count):
            self.current_instances = new_count
            self.last_scale_down = time.time()
            
            SCALING_DECISIONS_COUNTER.labels(direction="down").inc()
            logger.info("Scaled down", instances=new_count)
    
    @RESOURCE_ALLOCATION_TIME.time()
    async def allocate_campaign_resources(self, campaign_id: str, requirements: Dict[str, Any]) -> Optional[ResourceQuota]:
        """Allocate resources for a campaign."""
        try:
            # Calculate resource requirements
            quota = self._calculate_quota_from_requirements(requirements)
            
            # Check if resources are available
            if await self.resource_provider.allocate_resources(quota):
                self.allocated_quotas[campaign_id] = quota
                logger.info("Resources allocated", campaign_id=campaign_id, quota=quota)
                return quota
            else:
                logger.warning("Resource allocation failed", campaign_id=campaign_id, quota=quota)
                return None
                
        except Exception as e:
            logger.error("Error allocating resources", campaign_id=campaign_id, error=str(e))
            return None
    
    async def deallocate_campaign_resources(self, campaign_id: str) -> bool:
        """Deallocate resources for a campaign."""
        if campaign_id in self.allocated_quotas:
            quota = self.allocated_quotas[campaign_id]
            
            if await self.resource_provider.deallocate_resources(quota):
                del self.allocated_quotas[campaign_id]
                logger.info("Resources deallocated", campaign_id=campaign_id)
                return True
        
        return False
    
    def _calculate_quota_from_requirements(self, requirements: Dict[str, Any]) -> ResourceQuota:
        """Calculate resource quota from campaign requirements."""
        # Default resource allocation per agent
        base_cpu = 0.5  # cores per agent
        base_memory = 1.0  # GB per agent
        base_disk = 1.0  # GB per agent
        
        agent_count = requirements.get('max_agents', 1)
        complexity_multiplier = requirements.get('complexity_multiplier', 1.0)
        
        return ResourceQuota(
            cpu_cores=agent_count * base_cpu * complexity_multiplier,
            memory_gb=agent_count * base_memory * complexity_multiplier,
            disk_gb=agent_count * base_disk,
            max_agents=agent_count
        )
    
    def get_resource_statistics(self) -> Dict[str, Any]:
        """Get comprehensive resource statistics."""
        current_usage = self.resource_history[-1] if self.resource_history else None
        
        stats = {
            'current_instances': self.current_instances,
            'min_instances': self.scaling_policy.min_instances,
            'max_instances': self.scaling_policy.max_instances,
            'allocated_campaigns': len(self.allocated_quotas),
            'scaling_policy': {
                'predictive_enabled': self.scaling_policy.predictive_scaling,
                'scale_up_threshold': self.scaling_policy.scale_up_threshold,
                'scale_down_threshold': self.scaling_policy.scale_down_threshold
            }
        }
        
        if current_usage:
            stats['current_usage'] = {
                'cpu_percent': current_usage.cpu_percent,
                'memory_percent': current_usage.memory_percent,
                'disk_percent': current_usage.disk_percent,
                'active_agents': current_usage.active_agents
            }
        
        if len(self.resource_history) > 0:
            pattern = self.predictor.detect_pattern()
            predicted_cpu, predicted_memory = self.predictor.predict_next_usage(30)
            
            stats['predictions'] = {
                'workload_pattern': pattern.value,
                'predicted_cpu_30min': predicted_cpu,
                'predicted_memory_30min': predicted_memory
            }
        
        return stats


def create_epyc_optimized_policy() -> ScalingPolicy:
    """Create scaling policy optimized for EPYC processors."""
    return ScalingPolicy(
        min_instances=4,
        max_instances=32,  # EPYC optimization
        target_cpu_percent=60.0,  # Conservative for EPYC
        target_memory_percent=70.0,
        scale_up_threshold=75.0,
        scale_down_threshold=40.0,
        scale_up_cooldown=180,  # Faster scaling for EPYC
        scale_down_cooldown=300,
        predictive_scaling=True,
        burst_detection=True
    )


def create_development_policy() -> ScalingPolicy:
    """Create scaling policy for development environment."""
    return ScalingPolicy(
        min_instances=1,
        max_instances=4,
        target_cpu_percent=80.0,
        target_memory_percent=85.0,
        scale_up_threshold=90.0,
        scale_down_threshold=30.0,
        scale_up_cooldown=300,
        scale_down_cooldown=600,
        predictive_scaling=False,
        burst_detection=False
    )