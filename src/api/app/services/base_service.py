"""
Base Service Classes - Foundation for all XORB services
Principal Auditor Implementation: Enterprise-grade service foundation
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service status enumeration"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class ServiceType(Enum):
    """Service type enumeration"""
    CORE = "core"
    SECURITY = "security"
    SECURITY_TESTING = "security_testing"
    INTELLIGENCE = "intelligence"
    ORCHESTRATION = "orchestration"
    INFRASTRUCTURE = "infrastructure"
    MONITORING = "monitoring"
    REPORTING = "reporting"
    COMPLIANCE = "compliance"

@dataclass
class ServiceHealth:
    """Service health information"""
    service_id: str
    status: str
    message: str
    timestamp: datetime
    metrics: Dict[str, Any] = None
    dependencies: List[str] = None
    response_time_ms: Optional[float] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class ServiceMetrics:
    """Service metrics data"""
    service_id: str
    requests_total: int = 0
    requests_successful: int = 0
    requests_failed: int = 0
    avg_response_time_ms: float = 0.0
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.requests_total == 0:
            return 100.0
        return (self.requests_successful / self.requests_total) * 100.0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage"""
        if self.requests_total == 0:
            return 0.0
        return (self.requests_failed / self.requests_total) * 100.0

class BaseService(ABC):
    """
    Base service class providing common functionality for all XORB services
    
    Features:
    - Service lifecycle management
    - Health checking and monitoring
    - Metrics collection and reporting
    - Configuration management
    - Dependency tracking
    - Error handling and logging
    """
    
    def __init__(
        self,
        service_id: str,
        service_type: ServiceType = ServiceType.CORE,
        dependencies: List[str] = None,
        config: Dict[str, Any] = None
    ):
        self.service_id = service_id
        self.service_type = service_type
        self.dependencies = dependencies or []
        self.config = config or {}
        
        # Service state
        self.status = ServiceStatus.INITIALIZING
        self.start_time: Optional[datetime] = None
        self.last_health_check: Optional[datetime] = None
        
        # Metrics
        self.metrics = ServiceMetrics(service_id=service_id)
        
        # Health tracking
        self.health_history: List[ServiceHealth] = []
        self.max_health_history = 100
        
        # Error tracking
        self.error_count = 0
        self.last_error: Optional[Exception] = None
        self.last_error_time: Optional[datetime] = None
        
        logger.info(f"Initialized {self.service_type.value} service: {service_id}")
    
    async def initialize(self) -> bool:
        """
        Initialize the service - to be implemented by subclasses
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.start_time = datetime.utcnow()
            self.status = ServiceStatus.RUNNING
            logger.info(f"Service {self.service_id} initialized successfully")
            return True
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.last_error = e
            self.last_error_time = datetime.utcnow()
            logger.error(f"Failed to initialize service {self.service_id}: {e}")
            return False
    
    async def shutdown(self) -> bool:
        """
        Shutdown the service gracefully
        
        Returns:
            bool: True if shutdown successful, False otherwise
        """
        try:
            self.status = ServiceStatus.STOPPING
            logger.info(f"Shutting down service {self.service_id}")
            
            # Perform service-specific shutdown
            await self._perform_shutdown()
            
            self.status = ServiceStatus.STOPPED
            logger.info(f"Service {self.service_id} shutdown complete")
            return True
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.last_error = e
            self.last_error_time = datetime.utcnow()
            logger.error(f"Failed to shutdown service {self.service_id}: {e}")
            return False
    
    async def _perform_shutdown(self):
        """Override in subclasses for specific shutdown logic"""
        pass
    
    async def health_check(self) -> ServiceHealth:
        """
        Perform health check on the service
        
        Returns:
            ServiceHealth: Current health status
        """
        start_time = datetime.utcnow()
        
        try:
            # Perform service-specific health checks
            health_data = await self._perform_health_check()
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            health = ServiceHealth(
                service_id=self.service_id,
                status="healthy",
                message="Service is operating normally",
                timestamp=datetime.utcnow(),
                metrics=health_data.get("metrics", {}),
                dependencies=self.dependencies,
                response_time_ms=response_time
            )
            
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            health = ServiceHealth(
                service_id=self.service_id,
                status="unhealthy",
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                response_time_ms=response_time
            )
            
            self.error_count += 1
            self.last_error = e
            self.last_error_time = datetime.utcnow()
        
        # Store health history
        self.health_history.append(health)
        if len(self.health_history) > self.max_health_history:
            self.health_history.pop(0)
        
        self.last_health_check = datetime.utcnow()
        
        return health
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Override in subclasses for specific health check logic"""
        return {
            "status": "healthy",
            "metrics": {
                "uptime_seconds": self.get_uptime_seconds(),
                "status": self.status.value
            }
        }
    
    def get_uptime_seconds(self) -> float:
        """Get service uptime in seconds"""
        if self.start_time:
            return (datetime.utcnow() - self.start_time).total_seconds()
        return 0.0
    
    def get_metrics(self) -> ServiceMetrics:
        """Get current service metrics"""
        self.metrics.uptime_seconds = self.get_uptime_seconds()
        self.metrics.last_updated = datetime.utcnow()
        return self.metrics
    
    def increment_request_count(self, successful: bool = True):
        """Increment request counters"""
        self.metrics.requests_total += 1
        if successful:
            self.metrics.requests_successful += 1
        else:
            self.metrics.requests_failed += 1
    
    def record_response_time(self, response_time_ms: float):
        """Record response time for metrics"""
        if self.metrics.requests_total > 0:
            # Calculate rolling average
            current_avg = self.metrics.avg_response_time_ms
            total_requests = self.metrics.requests_total
            self.metrics.avg_response_time_ms = (
                (current_avg * (total_requests - 1) + response_time_ms) / total_requests
            )
        else:
            self.metrics.avg_response_time_ms = response_time_ms
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get comprehensive service information"""
        return {
            "service_id": self.service_id,
            "service_type": self.service_type.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": self.get_uptime_seconds(),
            "dependencies": self.dependencies,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "error_count": self.error_count,
            "last_error": str(self.last_error) if self.last_error else None,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
            "metrics": asdict(self.metrics)
        }
    
    def get_health_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent health check history"""
        recent_health = self.health_history[-limit:] if self.health_history else []
        return [asdict(health) for health in recent_health]


class SecurityService(BaseService):
    """
    Base class for security-related services
    
    Provides additional security-specific functionality and monitoring
    """
    
    def __init__(
        self,
        service_id: str,
        dependencies: List[str] = None,
        config: Dict[str, Any] = None
    ):
        super().__init__(
            service_id=service_id,
            service_type=ServiceType.SECURITY,
            dependencies=dependencies,
            config=config
        )
        
        # Security-specific metrics
        self.security_events_processed = 0
        self.threats_detected = 0
        self.false_positives = 0
        self.scan_sessions_active = 0
        
        # Security configuration
        self.security_config = config.get("security", {}) if config else {}
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Security service health check"""
        base_health = await super()._perform_health_check()
        
        # Add security-specific health metrics
        base_health["metrics"].update({
            "security_events_processed": self.security_events_processed,
            "threats_detected": self.threats_detected,
            "false_positives": self.false_positives,
            "scan_sessions_active": self.scan_sessions_active,
            "detection_rate": (
                self.threats_detected / max(1, self.security_events_processed) * 100
                if self.security_events_processed > 0 else 0.0
            )
        })
        
        return base_health
    
    def record_security_event(self, event_type: str, threat_detected: bool = False):
        """Record security event for metrics"""
        self.security_events_processed += 1
        if threat_detected:
            self.threats_detected += 1
    
    def record_false_positive(self):
        """Record false positive detection"""
        self.false_positives += 1
    
    def increment_active_scans(self):
        """Increment active scan counter"""
        self.scan_sessions_active += 1
    
    def decrement_active_scans(self):
        """Decrement active scan counter"""
        self.scan_sessions_active = max(0, self.scan_sessions_active - 1)


class XORBService(BaseService):
    """
    XORB-specific service base class
    
    Provides XORB platform-specific functionality and integrations
    """
    
    def __init__(
        self,
        service_id: str,
        service_type: ServiceType = ServiceType.CORE,
        dependencies: List[str] = None,
        config: Dict[str, Any] = None
    ):
        super().__init__(
            service_id=service_id,
            service_type=service_type,
            dependencies=dependencies,
            config=config
        )
        
        # XORB-specific configuration
        self.xorb_config = config.get("xorb", {}) if config else {}
        
        # Integration settings
        self.vault_enabled = config.get("vault_enabled", False) if config else False
        self.temporal_enabled = config.get("temporal_enabled", False) if config else False
        self.redis_enabled = config.get("redis_enabled", True) if config else True
        
        # Performance tracking
        self.api_calls_made = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """XORB service health check with integration status"""
        base_health = await super()._perform_health_check()
        
        # Add XORB-specific health metrics
        base_health["metrics"].update({
            "api_calls_made": self.api_calls_made,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / max(1, self.cache_hits + self.cache_misses) * 100
                if (self.cache_hits + self.cache_misses) > 0 else 0.0
            ),
            "integrations": {
                "vault": self.vault_enabled,
                "temporal": self.temporal_enabled,
                "redis": self.redis_enabled
            }
        })
        
        return base_health
    
    def record_api_call(self):
        """Record API call for metrics"""
        self.api_calls_made += 1
    
    def record_cache_hit(self):
        """Record cache hit"""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        self.cache_misses += 1


# Global service registry for monitoring and management
class ServiceRegistry:
    """Global service registry for tracking all services"""
    
    def __init__(self):
        self.services: Dict[str, BaseService] = {}
        self.service_dependencies: Dict[str, List[str]] = {}
    
    def register_service(self, service: BaseService):
        """Register a service in the global registry"""
        self.services[service.service_id] = service
        self.service_dependencies[service.service_id] = service.dependencies
        logger.info(f"Registered service: {service.service_id}")
    
    def unregister_service(self, service_id: str):
        """Unregister a service from the global registry"""
        if service_id in self.services:
            del self.services[service_id]
            del self.service_dependencies[service_id]
            logger.info(f"Unregistered service: {service_id}")
    
    def get_service(self, service_id: str) -> Optional[BaseService]:
        """Get a service by ID"""
        return self.services.get(service_id)
    
    def get_all_services(self) -> Dict[str, BaseService]:
        """Get all registered services"""
        return self.services.copy()
    
    def get_services_by_type(self, service_type: ServiceType) -> List[BaseService]:
        """Get all services of a specific type"""
        return [
            service for service in self.services.values()
            if service.service_type == service_type
        ]
    
    async def health_check_all(self) -> Dict[str, ServiceHealth]:
        """Perform health check on all registered services"""
        health_results = {}
        
        for service_id, service in self.services.items():
            try:
                health = await service.health_check()
                health_results[service_id] = health
            except Exception as e:
                health_results[service_id] = ServiceHealth(
                    service_id=service_id,
                    status="error",
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.utcnow()
                )
        
        return health_results
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        service_types = {}
        total_services = len(self.services)
        
        for service in self.services.values():
            service_type = service.service_type.value
            service_types[service_type] = service_types.get(service_type, 0) + 1
        
        return {
            "total_services": total_services,
            "service_types": service_types,
            "service_ids": list(self.services.keys())
        }


# Global service registry instance
service_registry = ServiceRegistry()