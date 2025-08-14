"""
Base service classes for XORB platform
Provides common interface and functionality for all services
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Type
from uuid import UUID, uuid4
from dataclasses import dataclass

from ..infrastructure.observability import add_trace_context, get_metrics_collector


class ServiceStatus(Enum):
    """Service lifecycle status"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class ServiceType(Enum):
    """Service classification types"""
    CORE = "core"
    ANALYTICS = "analytics"
    SECURITY = "security"
    INTELLIGENCE = "intelligence"
    INTEGRATION = "integration"
    MONITORING = "monitoring"


@dataclass
class ServiceHealth:
    """Service health information"""
    status: ServiceStatus
    message: str
    timestamp: datetime
    checks: Dict[str, Any]
    uptime_seconds: float = 0.0
    last_error: Optional[str] = None


@dataclass
class ServiceMetrics:
    """Service performance metrics"""
    service_id: str
    requests_total: int = 0
    requests_per_second: float = 0.0
    avg_response_time_ms: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    custom_metrics: Dict[str, Any] = None


class XORBService(ABC):
    """
    Abstract base class for all XORB platform services

    Provides:
    - Standardized lifecycle management (initialize, shutdown)
    - Health checking and status reporting
    - Metrics collection and monitoring
    - Error handling and logging
    - Dependency management
    - Configuration validation
    """

    def __init__(self,
                 service_id: Optional[str] = None,
                 service_type: ServiceType = ServiceType.CORE,
                 dependencies: List[str] = None,
                 config: Dict[str, Any] = None):
        # Service identity
        self.service_id = service_id or self.__class__.__name__
        self.service_type = service_type
        self.instance_id = str(uuid4())

        # Dependencies and configuration
        self.dependencies = dependencies or []
        self.config = config or {}

        # Service state
        self.status = ServiceStatus.INITIALIZING
        self.start_time: Optional[datetime] = None
        self.last_health_check: Optional[datetime] = None
        self.error_count = 0
        self.restart_count = 0

        # Logging and metrics
        self.logger = logging.getLogger(f"xorb.{self.service_id}")
        self.metrics_collector = get_metrics_collector()

        # Health check configuration
        self.health_check_interval = 30  # seconds
        self.health_check_timeout = 5   # seconds
        self.max_consecutive_failures = 3

        self.logger.info(f"Service {self.service_id} created (instance: {self.instance_id})")

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the service

        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Gracefully shutdown the service

        Returns:
            bool: True if shutdown successful, False otherwise
        """
        pass

    @abstractmethod
    async def health_check(self) -> ServiceHealth:
        """
        Perform health check

        Returns:
            ServiceHealth: Current health status
        """
        pass

    async def get_metrics(self) -> ServiceMetrics:
        """
        Get service performance metrics

        Returns:
            ServiceMetrics: Current service metrics
        """
        uptime = 0.0
        if self.start_time:
            uptime = (datetime.utcnow() - self.start_time).total_seconds()

        return ServiceMetrics(
            service_id=self.service_id,
            uptime_seconds=uptime,
            error_count=self.error_count,
            custom_metrics=await self._get_custom_metrics()
        )

    async def _get_custom_metrics(self) -> Dict[str, Any]:
        """Override in subclasses to provide service-specific metrics"""
        return {}

    async def start(self) -> bool:
        """
        Start the service (public interface)

        Returns:
            bool: True if service started successfully
        """
        try:
            self.logger.info(f"Starting service {self.service_id}")
            self.status = ServiceStatus.INITIALIZING

            # Validate configuration
            if not await self._validate_config():
                self.logger.error("Configuration validation failed")
                self.status = ServiceStatus.UNHEALTHY
                return False

            # Check dependencies
            if not await self._check_dependencies():
                self.logger.error("Dependency check failed")
                self.status = ServiceStatus.UNHEALTHY
                return False

            # Initialize service
            if not await self.initialize():
                self.logger.error("Service initialization failed")
                self.status = ServiceStatus.UNHEALTHY
                return False

            # Service started successfully
            self.start_time = datetime.utcnow()
            self.status = ServiceStatus.HEALTHY

            # Add tracing context
            add_trace_context(
                operation="service_start",
                service_id=self.service_id,
                instance_id=self.instance_id
            )

            # Record metrics
            self.metrics_collector.record_job_execution(
                f"service_start_{self.service_id}", 0, True
            )

            self.logger.info(f"Service {self.service_id} started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start service {self.service_id}: {e}")
            self.status = ServiceStatus.UNHEALTHY
            self.error_count += 1
            return False

    async def stop(self) -> bool:
        """
        Stop the service (public interface)

        Returns:
            bool: True if service stopped successfully
        """
        try:
            self.logger.info(f"Stopping service {self.service_id}")
            self.status = ServiceStatus.SHUTTING_DOWN

            # Shutdown service
            if not await self.shutdown():
                self.logger.error("Service shutdown failed")
                return False

            # Update status
            self.status = ServiceStatus.STOPPED

            # Add tracing context
            add_trace_context(
                operation="service_stop",
                service_id=self.service_id,
                instance_id=self.instance_id
            )

            self.logger.info(f"Service {self.service_id} stopped successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop service {self.service_id}: {e}")
            self.error_count += 1
            return False

    async def restart(self) -> bool:
        """
        Restart the service

        Returns:
            bool: True if service restarted successfully
        """
        self.restart_count += 1
        self.logger.info(f"Restarting service {self.service_id} (restart #{self.restart_count})")

        # Stop then start
        await self.stop()
        await asyncio.sleep(1)  # Brief pause
        return await self.start()

    async def _validate_config(self) -> bool:
        """
        Validate service configuration
        Override in subclasses for service-specific validation

        Returns:
            bool: True if configuration is valid
        """
        # Basic validation - check required config keys exist
        required_keys = getattr(self, 'REQUIRED_CONFIG_KEYS', [])

        for key in required_keys:
            if key not in self.config:
                self.logger.error(f"Missing required configuration key: {key}")
                return False

        return True

    async def _check_dependencies(self) -> bool:
        """
        Check service dependencies are available
        Override in subclasses for service-specific dependency checks

        Returns:
            bool: True if all dependencies are available
        """
        # Basic dependency check - this would be enhanced with actual dependency resolution
        if not self.dependencies:
            return True

        # For now, just log the dependencies
        self.logger.info(f"Service {self.service_id} depends on: {self.dependencies}")
        return True

    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            "service_id": self.service_id,
            "service_type": self.service_type.value,
            "instance_id": self.instance_id,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0,
            "restart_count": self.restart_count,
            "error_count": self.error_count,
            "dependencies": self.dependencies
        }


class DatabaseService(XORBService):
    """Base class for database-related services"""

    def __init__(self, **kwargs):
        super().__init__(service_type=ServiceType.CORE, **kwargs)
        self.REQUIRED_CONFIG_KEYS = ['database_url']

    @abstractmethod
    async def get_connection_pool_status(self) -> Dict[str, Any]:
        """Get database connection pool status"""
        pass


class AnalyticsService(XORBService):
    """Base class for analytics services"""

    def __init__(self, **kwargs):
        super().__init__(service_type=ServiceType.ANALYTICS, **kwargs)

    @abstractmethod
    async def process_batch(self, data: List[Any]) -> Dict[str, Any]:
        """Process a batch of data for analytics"""
        pass

    @abstractmethod
    async def get_analytics_metrics(self) -> Dict[str, Any]:
        """Get analytics-specific metrics"""
        pass


class SecurityService(XORBService):
    """Base class for security services"""

    def __init__(self, **kwargs):
        super().__init__(service_type=ServiceType.SECURITY, **kwargs)

    @abstractmethod
    async def process_security_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process a security event"""
        pass

    @abstractmethod
    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security-specific metrics"""
        pass


class IntelligenceService(XORBService):
    """Base class for AI/ML intelligence services"""

    def __init__(self, **kwargs):
        super().__init__(service_type=ServiceType.INTELLIGENCE, **kwargs)

    @abstractmethod
    async def analyze(self, data: Any) -> Dict[str, Any]:
        """Perform AI/ML analysis on data"""
        pass

    @abstractmethod
    async def get_model_status(self) -> Dict[str, Any]:
        """Get AI/ML model status"""
        pass


class IntegrationService(XORBService):
    """Base class for integration services"""

    def __init__(self, **kwargs):
        super().__init__(service_type=ServiceType.INTEGRATION, **kwargs)

    @abstractmethod
    async def sync_data(self) -> bool:
        """Synchronize data with external system"""
        pass

    @abstractmethod
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status"""
        pass


# Service factory for creating services
class ServiceFactory:
    """Factory for creating service instances"""

    _service_classes: Dict[str, Type[XORBService]] = {}

    @classmethod
    def register_service(cls, service_id: str, service_class: Type[XORBService]):
        """Register a service class"""
        cls._service_classes[service_id] = service_class

    @classmethod
    def create_service(cls,
                      service_id: str,
                      config: Dict[str, Any] = None,
                      dependencies: List[str] = None) -> Optional[XORBService]:
        """Create a service instance"""
        if service_id not in cls._service_classes:
            return None

        service_class = cls._service_classes[service_id]
        return service_class(
            service_id=service_id,
            config=config or {},
            dependencies=dependencies or []
        )

    @classmethod
    def get_registered_services(cls) -> List[str]:
        """Get list of registered service IDs"""
        return list(cls._service_classes.keys())
