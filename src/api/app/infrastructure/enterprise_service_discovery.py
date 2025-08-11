"""
Enterprise Service Discovery System for XORB Platform
Advanced service registry with health propagation and dependency management
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import weakref
import socket
import aiohttp

from .observability import get_metrics_collector, add_trace_context

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Enhanced service status"""
    DISCOVERING = "discovering"
    REGISTERING = "registering"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    DEREGISTERING = "deregistering"
    OFFLINE = "offline"

class ServiceCapability(Enum):
    """Service capability types"""
    API_ENDPOINT = "api_endpoint"
    MESSAGE_QUEUE = "message_queue"
    DATABASE = "database"
    CACHE = "cache"
    MONITORING = "monitoring"
    SECURITY = "security"
    ML_INFERENCE = "ml_inference"
    STORAGE = "storage"

@dataclass
class ServiceEndpoint:
    """Service endpoint information"""
    protocol: str  # http, https, tcp, udp
    host: str
    port: int
    path: Optional[str] = None
    health_check_path: Optional[str] = "/health"
    
    @property
    def url(self) -> str:
        if self.protocol in ["http", "https"]:
            path = self.path or ""
            return f"{self.protocol}://{self.host}:{self.port}{path}"
        return f"{self.protocol}://{self.host}:{self.port}"

@dataclass
class ServiceInstance:
    """Enhanced service instance information"""
    service_id: str
    instance_id: str
    service_name: str
    version: str
    endpoints: List[ServiceEndpoint]
    capabilities: List[ServiceCapability]
    metadata: Dict[str, Any]
    health_status: ServiceStatus
    last_heartbeat: datetime
    registration_time: datetime
    dependency_services: List[str]
    dependent_services: List[str]
    load_metrics: Dict[str, float]
    tags: List[str]
    environment: str  # dev, staging, prod
    datacenter: str
    zone: str

@dataclass
class ServiceDependency:
    """Service dependency relationship"""
    service_id: str
    dependent_service_id: str
    dependency_type: str  # required, optional, preferred
    health_impact: float  # 0.0 to 1.0 - how much dependency health affects this service
    last_checked: datetime
    status: str  # healthy, unhealthy, unknown

class HealthPropagationRule:
    """Rules for health propagation through service dependencies"""
    def __init__(self, 
                 rule_id: str,
                 condition: Callable[[ServiceInstance], bool],
                 action: Callable[[ServiceInstance, List[ServiceInstance]], ServiceStatus],
                 priority: int = 0):
        self.rule_id = rule_id
        self.condition = condition
        self.action = action
        self.priority = priority

class EnterpriseServiceDiscovery:
    """
    Enterprise Service Discovery with advanced features:
    - Automatic service registration and deregistration
    - Health check propagation through dependency graph
    - Load balancing and service selection
    - Service capability matching
    - Circuit breaker integration
    - Multi-environment support
    - Service mesh integration
    """
    
    def __init__(self, cluster_name: str = "xorb-enterprise"):
        self.cluster_name = cluster_name
        
        # Service registry
        self.services: Dict[str, ServiceInstance] = {}
        self.service_groups: Dict[str, List[str]] = {}  # service_name -> [instance_ids]
        
        # Dependency graph
        self.dependencies: Dict[str, List[ServiceDependency]] = {}
        self.dependents: Dict[str, List[str]] = {}  # service_id -> [dependent_service_ids]
        
        # Health propagation
        self.health_rules: List[HealthPropagationRule] = []
        self.health_history: Dict[str, List[Tuple[datetime, ServiceStatus]]] = {}
        
        # Load balancing
        self.load_balancers: Dict[str, Any] = {}
        self.service_weights: Dict[str, Dict[str, float]] = {}  # service_name -> {instance_id: weight}
        
        # Monitoring and metrics
        self.metrics = get_metrics_collector()
        self.event_callbacks: List[Callable] = []
        
        # Configuration
        self.heartbeat_interval = 30  # seconds
        self.health_check_timeout = 5  # seconds
        self.stale_service_threshold = 90  # seconds
        
        # Background tasks
        self._health_monitor_task = None
        self._cleanup_task = None
        self._dependency_monitor_task = None
        
        self._initialized = False
        self._setup_default_health_rules()
    
    def _setup_default_health_rules(self):
        """Setup default health propagation rules"""
        
        # Rule 1: If all required dependencies are unhealthy, mark service as unhealthy
        def all_required_deps_unhealthy_condition(service: ServiceInstance) -> bool:
            required_deps = [
                dep for dep in self.dependencies.get(service.service_id, [])
                if dep.dependency_type == "required"
            ]
            if not required_deps:
                return False
            
            for dep in required_deps:
                dep_service = self.services.get(dep.dependent_service_id)
                if dep_service and dep_service.health_status == ServiceStatus.HEALTHY:
                    return False
            return True
        
        def mark_unhealthy_action(service: ServiceInstance, deps: List[ServiceInstance]) -> ServiceStatus:
            return ServiceStatus.UNHEALTHY
        
        self.health_rules.append(HealthPropagationRule(
            "required_deps_unhealthy",
            all_required_deps_unhealthy_condition,
            mark_unhealthy_action,
            priority=100
        ))
        
        # Rule 2: If majority of optional dependencies are unhealthy, mark as degraded
        def optional_deps_degraded_condition(service: ServiceInstance) -> bool:
            optional_deps = [
                dep for dep in self.dependencies.get(service.service_id, [])
                if dep.dependency_type == "optional"
            ]
            if len(optional_deps) < 2:
                return False
            
            unhealthy_count = 0
            for dep in optional_deps:
                dep_service = self.services.get(dep.dependent_service_id)
                if dep_service and dep_service.health_status != ServiceStatus.HEALTHY:
                    unhealthy_count += 1
            
            return unhealthy_count > len(optional_deps) / 2
        
        def mark_degraded_action(service: ServiceInstance, deps: List[ServiceInstance]) -> ServiceStatus:
            return ServiceStatus.DEGRADED
        
        self.health_rules.append(HealthPropagationRule(
            "optional_deps_degraded",
            optional_deps_degraded_condition,
            mark_degraded_action,
            priority=50
        ))
    
    async def initialize(self) -> bool:
        """Initialize the service discovery system"""
        if self._initialized:
            return True
        
        try:
            logger.info("Initializing Enterprise Service Discovery...")
            
            # Start background monitoring tasks
            self._health_monitor_task = asyncio.create_task(self._health_monitor())
            self._cleanup_task = asyncio.create_task(self._cleanup_stale_services())
            self._dependency_monitor_task = asyncio.create_task(self._dependency_monitor())
            
            self._initialized = True
            logger.info("Enterprise Service Discovery initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize service discovery: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the service discovery system"""
        try:
            # Cancel background tasks
            for task in [self._health_monitor_task, self._cleanup_task, self._dependency_monitor_task]:
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Deregister all services
            for service_id in list(self.services.keys()):
                await self.deregister_service(service_id)
            
            logger.info("Service Discovery shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during service discovery shutdown: {e}")
    
    async def register_service(self,
                             service_name: str,
                             version: str,
                             endpoints: List[ServiceEndpoint],
                             capabilities: List[ServiceCapability],
                             dependencies: List[str] = None,
                             metadata: Dict[str, Any] = None,
                             environment: str = "prod",
                             tags: List[str] = None) -> str:
        """Register a new service instance"""
        try:
            # Generate unique instance ID
            instance_id = f"{service_name}-{int(time.time())}-{hash(str(endpoints)) % 10000}"
            service_id = f"{service_name}:{instance_id}"
            
            # Create service instance
            service_instance = ServiceInstance(
                service_id=service_id,
                instance_id=instance_id,
                service_name=service_name,
                version=version,
                endpoints=endpoints,
                capabilities=capabilities,
                metadata=metadata or {},
                health_status=ServiceStatus.REGISTERING,
                last_heartbeat=datetime.now(),
                registration_time=datetime.now(),
                dependency_services=dependencies or [],
                dependent_services=[],
                load_metrics={},
                tags=tags or [],
                environment=environment,
                datacenter=metadata.get("datacenter", "default") if metadata else "default",
                zone=metadata.get("zone", "default") if metadata else "default"
            )
            
            # Register service
            self.services[service_id] = service_instance
            
            # Update service groups
            if service_name not in self.service_groups:
                self.service_groups[service_name] = []
            self.service_groups[service_name].append(service_id)
            
            # Register dependencies
            if dependencies:
                self.dependencies[service_id] = []
                for dep_service_name in dependencies:
                    dependency = ServiceDependency(
                        service_id=service_id,
                        dependent_service_id=dep_service_name,
                        dependency_type="required",  # Default to required
                        health_impact=1.0,
                        last_checked=datetime.now(),
                        status="unknown"
                    )
                    self.dependencies[service_id].append(dependency)
                    
                    # Update dependents
                    if dep_service_name not in self.dependents:
                        self.dependents[dep_service_name] = []
                    self.dependents[dep_service_name].append(service_id)
            
            # Perform initial health check
            await self._perform_health_check(service_id)
            
            # Trigger service registration event
            await self._trigger_event("service_registered", service_instance)
            
            # Record metrics
            self.metrics.record_service_registration(
                service_name=service_name,
                instance_id=instance_id,
                environment=environment
            )
            
            logger.info(f"Service registered: {service_name} ({instance_id})")
            
            return service_id
            
        except Exception as e:
            logger.error(f"Service registration failed: {e}")
            raise
    
    async def deregister_service(self, service_id: str) -> bool:
        """Deregister a service instance"""
        try:
            if service_id not in self.services:
                return False
            
            service = self.services[service_id]
            service.health_status = ServiceStatus.DEREGISTERING
            
            # Remove from service groups
            if service.service_name in self.service_groups:
                self.service_groups[service.service_name] = [
                    sid for sid in self.service_groups[service.service_name] 
                    if sid != service_id
                ]
                
                # Clean up empty groups
                if not self.service_groups[service.service_name]:
                    del self.service_groups[service.service_name]
            
            # Clean up dependencies
            if service_id in self.dependencies:
                del self.dependencies[service_id]
            
            # Clean up dependents
            for dependent_services in self.dependents.values():
                if service_id in dependent_services:
                    dependent_services.remove(service_id)
            
            # Remove service
            del self.services[service_id]
            
            # Trigger deregistration event
            await self._trigger_event("service_deregistered", service)
            
            logger.info(f"Service deregistered: {service.service_name} ({service.instance_id})")
            
            return True
            
        except Exception as e:
            logger.error(f"Service deregistration failed: {e}")
            return False
    
    async def discover_services(self,
                              service_name: str = None,
                              capabilities: List[ServiceCapability] = None,
                              environment: str = None,
                              tags: List[str] = None,
                              health_status: ServiceStatus = None) -> List[ServiceInstance]:
        """Discover services matching criteria"""
        try:
            matching_services = []
            
            for service in self.services.values():
                # Filter by service name
                if service_name and service.service_name != service_name:
                    continue
                
                # Filter by capabilities
                if capabilities:
                    if not all(cap in service.capabilities for cap in capabilities):
                        continue
                
                # Filter by environment
                if environment and service.environment != environment:
                    continue
                
                # Filter by tags
                if tags:
                    if not all(tag in service.tags for tag in tags):
                        continue
                
                # Filter by health status
                if health_status and service.health_status != health_status:
                    continue
                
                matching_services.append(service)
            
            # Sort by health status and load
            matching_services.sort(key=lambda s: (
                s.health_status.value,
                s.load_metrics.get("cpu_usage", 0.0)
            ))
            
            return matching_services
            
        except Exception as e:
            logger.error(f"Service discovery failed: {e}")
            return []
    
    async def get_service_endpoint(self,
                                 service_name: str,
                                 protocol: str = "http",
                                 load_balance: bool = True) -> Optional[ServiceEndpoint]:
        """Get an endpoint for a service with load balancing"""
        try:
            services = await self.discover_services(
                service_name=service_name,
                health_status=ServiceStatus.HEALTHY
            )
            
            if not services:
                return None
            
            if load_balance:
                # Use weighted round-robin load balancing
                service = await self._select_service_by_load(services)
            else:
                service = services[0]
            
            # Find matching endpoint
            for endpoint in service.endpoints:
                if endpoint.protocol == protocol:
                    return endpoint
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get service endpoint: {e}")
            return None
    
    async def update_service_health(self, service_id: str, status: ServiceStatus, metrics: Dict[str, Any] = None):
        """Update service health status and metrics"""
        try:
            if service_id not in self.services:
                return
            
            service = self.services[service_id]
            old_status = service.health_status
            
            service.health_status = status
            service.last_heartbeat = datetime.now()
            
            if metrics:
                service.load_metrics.update(metrics)
            
            # Record health history
            if service_id not in self.health_history:
                self.health_history[service_id] = []
            
            self.health_history[service_id].append((datetime.now(), status))
            
            # Keep only recent history (last 100 entries)
            if len(self.health_history[service_id]) > 100:
                self.health_history[service_id] = self.health_history[service_id][-100:]
            
            # Trigger health change event
            if old_status != status:
                await self._trigger_event("health_changed", service, {
                    "old_status": old_status,
                    "new_status": status
                })
                
                # Propagate health changes to dependent services
                await self._propagate_health_change(service_id)
            
            # Record metrics
            self.metrics.record_service_health_update(
                service_name=service.service_name,
                instance_id=service.instance_id,
                status=status.value
            )
            
        except Exception as e:
            logger.error(f"Failed to update service health: {e}")
    
    async def get_service_topology(self) -> Dict[str, Any]:
        """Get the complete service topology and dependency graph"""
        try:
            topology = {
                "services": {},
                "dependencies": [],
                "clusters": {},
                "statistics": {
                    "total_services": len(self.services),
                    "healthy_services": len([s for s in self.services.values() if s.health_status == ServiceStatus.HEALTHY]),
                    "service_groups": len(self.service_groups),
                    "dependency_relationships": sum(len(deps) for deps in self.dependencies.values())
                }
            }
            
            # Add service information
            for service_id, service in self.services.items():
                topology["services"][service_id] = {
                    "service_name": service.service_name,
                    "instance_id": service.instance_id,
                    "version": service.version,
                    "health_status": service.health_status.value,
                    "capabilities": [cap.value for cap in service.capabilities],
                    "endpoints": [asdict(ep) for ep in service.endpoints],
                    "environment": service.environment,
                    "load_metrics": service.load_metrics,
                    "last_heartbeat": service.last_heartbeat.isoformat()
                }
            
            # Add dependency information
            for service_id, deps in self.dependencies.items():
                for dep in deps:
                    topology["dependencies"].append({
                        "from": service_id,
                        "to": dep.dependent_service_id,
                        "type": dep.dependency_type,
                        "health_impact": dep.health_impact,
                        "status": dep.status
                    })
            
            # Group services by name
            for service_name, instance_ids in self.service_groups.items():
                topology["clusters"][service_name] = {
                    "instances": instance_ids,
                    "total_instances": len(instance_ids),
                    "healthy_instances": len([
                        sid for sid in instance_ids 
                        if self.services[sid].health_status == ServiceStatus.HEALTHY
                    ])
                }
            
            return topology
            
        except Exception as e:
            logger.error(f"Failed to get service topology: {e}")
            return {"error": str(e)}
    
    async def _health_monitor(self):
        """Background task for health monitoring"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Check health of all registered services
                health_check_tasks = []
                for service_id in self.services.keys():
                    task = asyncio.create_task(self._perform_health_check(service_id))
                    health_check_tasks.append(task)
                
                # Wait for all health checks to complete
                if health_check_tasks:
                    await asyncio.gather(*health_check_tasks, return_exceptions=True)
                
                # Apply health propagation rules
                await self._apply_health_propagation_rules()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _perform_health_check(self, service_id: str):
        """Perform health check on a specific service"""
        try:
            if service_id not in self.services:
                return
            
            service = self.services[service_id]
            
            # Find HTTP endpoint with health check
            health_endpoint = None
            for endpoint in service.endpoints:
                if endpoint.protocol in ["http", "https"] and endpoint.health_check_path:
                    health_endpoint = endpoint
                    break
            
            if not health_endpoint:
                # No health check endpoint, assume healthy if recently updated
                time_since_heartbeat = (datetime.now() - service.last_heartbeat).total_seconds()
                if time_since_heartbeat > self.stale_service_threshold:
                    await self.update_service_health(service_id, ServiceStatus.UNHEALTHY)
                return
            
            # Perform HTTP health check
            health_url = f"{health_endpoint.url.rstrip('/')}{health_endpoint.health_check_path}"
            
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.health_check_timeout)) as session:
                    async with session.get(health_url) as response:
                        if response.status == 200:
                            # Get load metrics from health response
                            try:
                                health_data = await response.json()
                                load_metrics = health_data.get("metrics", {})
                                await self.update_service_health(service_id, ServiceStatus.HEALTHY, load_metrics)
                            except:
                                await self.update_service_health(service_id, ServiceStatus.HEALTHY)
                        else:
                            await self.update_service_health(service_id, ServiceStatus.UNHEALTHY)
            
            except asyncio.TimeoutError:
                await self.update_service_health(service_id, ServiceStatus.UNHEALTHY)
            except Exception as e:
                logger.warning(f"Health check failed for {service_id}: {e}")
                await self.update_service_health(service_id, ServiceStatus.UNHEALTHY)
                
        except Exception as e:
            logger.error(f"Health check error for {service_id}: {e}")
    
    async def _select_service_by_load(self, services: List[ServiceInstance]) -> ServiceInstance:
        """Select service instance using weighted load balancing"""
        if len(services) == 1:
            return services[0]
        
        # Calculate weights based on inverse load
        weights = []
        for service in services:
            cpu_usage = service.load_metrics.get("cpu_usage", 0.5)
            memory_usage = service.load_metrics.get("memory_usage", 0.5)
            load_score = (cpu_usage + memory_usage) / 2
            weight = 1.0 / (load_score + 0.1)  # Avoid division by zero
            weights.append(weight)
        
        # Weighted random selection
        import random
        total_weight = sum(weights)
        random_value = random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if random_value <= cumulative_weight:
                return services[i]
        
        return services[0]  # Fallback


# Global service discovery instance
service_discovery: Optional[EnterpriseServiceDiscovery] = None

async def get_service_discovery() -> EnterpriseServiceDiscovery:
    """Get the global service discovery instance"""
    global service_discovery
    if not service_discovery:
        raise RuntimeError("Service discovery not initialized")
    return service_discovery

async def init_service_discovery(cluster_name: str = "xorb-enterprise") -> bool:
    """Initialize the service discovery system"""
    global service_discovery
    
    if service_discovery:
        return True
    
    service_discovery = EnterpriseServiceDiscovery(cluster_name=cluster_name)
    return await service_discovery.initialize()