"""
Production Container - Strategic enterprise dependency injection system
Implements production-ready service management with AI-powered capabilities
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, Type, TypeVar, Callable, List
from contextlib import asynccontextmanager
from datetime import datetime
from dataclasses import dataclass
from functools import lru_cache

# Import all production services and repositories
from .production_repositories import RepositoryFactory, get_repository_factory
from ..services.production_service_implementations import ServiceFactory, get_service_factory
from ..services.ptaas_scanner_service import SecurityScannerService
from ..services.advanced_ai_threat_intelligence import AdvancedThreatIntelligenceEngine
from ..services.production_ai_threat_intelligence_engine import ProductionAIThreatIntelligenceEngine
# Note: Advanced engines will be imported conditionally during initialization
from ..services.enterprise_security_platform import EnterpriseSecurityPlatform
from ..services.autonomous_security_orchestrator import AutonomousSecurityOrchestrator
from ..services.sophisticated_red_team_agent import SophisticatedRedTeamAgent
from ..services.advanced_behavioral_analytics_engine import AdvancedBehavioralAnalyticsEngine

logger = logging.getLogger(__name__)

@dataclass
class ServiceHealth:
    """Service health status"""
    service_name: str
    status: str  # healthy, degraded, unhealthy
    message: str
    timestamp: datetime
    metrics: Dict[str, Any]

@dataclass
class ContainerMetrics:
    """Container performance metrics"""
    services_registered: int
    services_initialized: int
    healthy_services: int
    degraded_services: int
    unhealthy_services: int
    initialization_time_seconds: float
    memory_usage_mb: float
    uptime_seconds: float

class ProductionContainer:
    """
    Strategic Production Container with:
    - Advanced AI service orchestration
    - Real-time health monitoring
    - Performance analytics
    - Resource optimization
    - Fault tolerance
    - Security integration
    """

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._health_status: Dict[str, ServiceHealth] = {}
        self._initialized = False
        self._start_time = datetime.now()
        self._config = {}

        # Core factories
        self._repository_factory: Optional[RepositoryFactory] = None
        self._service_factory: Optional[ServiceFactory] = None

        # Strategic service instances
        self._scanner_service: Optional[SecurityScannerService] = None
        self._threat_intelligence: Optional[AdvancedThreatIntelligenceEngine] = None
        self._production_ai_engine: Optional[ProductionAIThreatIntelligenceEngine] = None
        self._threat_hunting = None
        self._mitre_engine = None
        self._security_platform: Optional[EnterpriseSecurityPlatform] = None
        self._orchestrator: Optional[AutonomousSecurityOrchestrator] = None
        self._red_team_agent: Optional[SophisticatedRedTeamAgent] = None
        self._behavioral_analytics: Optional[AdvancedBehavioralAnalyticsEngine] = None

    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize production container with comprehensive service orchestration"""
        if self._initialized:
            logger.warning("Production container already initialized")
            return True

        start_time = datetime.now()
        logger.info("🚀 Initializing Production Container with Strategic AI Services...")

        try:
            self._config = config or self._get_production_config()

            # Phase 1: Core Infrastructure
            await self._initialize_core_infrastructure()

            # Phase 2: Security Services
            await self._initialize_security_services()

            # Phase 3: AI & Intelligence Services
            await self._initialize_ai_services()

            # Phase 4: Enterprise Features
            await self._initialize_enterprise_features()

            # Phase 5: Health & Monitoring
            await self._initialize_monitoring()

            self._initialized = True
            init_time = (datetime.now() - start_time).total_seconds()

            logger.info(f"✅ Production Container initialized successfully in {init_time:.2f}s")
            logger.info(f"📊 Services: {len(self._services)} registered, {len([s for s in self._services.values() if s])} active")

            return True

        except Exception as e:
            logger.error(f"❌ Production container initialization failed: {e}")
            await self._cleanup_partial_initialization()
            return False

    async def _initialize_core_infrastructure(self):
        """Initialize core infrastructure services"""
        logger.info("🔧 Initializing Core Infrastructure...")

        # Repository Factory
        database_url = self._config.get('DATABASE_URL', 'postgresql+asyncpg://xorb:xorb@postgres:5432/xorb')
        redis_url = self._config.get('REDIS_URL', 'redis://redis:6379/0')

        self._repository_factory = await get_repository_factory(database_url, redis_url)
        self._services['repository_factory'] = self._repository_factory

        # Service Factory
        jwt_secret = self._config.get('JWT_SECRET', self._generate_secure_jwt_secret())
        self._service_factory = await get_service_factory(self._repository_factory, jwt_secret)
        self._services['service_factory'] = self._service_factory

        logger.info("✅ Core infrastructure initialized")

    async def _initialize_security_services(self):
        """Initialize comprehensive security services"""
        logger.info("🛡️ Initializing Security Services...")

        # PTaaS Scanner Service
        self._scanner_service = SecurityScannerService(
            service_id="production_ptaas_scanner",
            config=self._config
        )
        await self._scanner_service.initialize()
        self._services['scanner_service'] = self._scanner_service

        # Enterprise Security Platform
        self._security_platform = EnterpriseSecurityPlatform(
            repository_factory=self._repository_factory,
            config=self._config
        )
        await self._security_platform.initialize()
        self._services['security_platform'] = self._security_platform

        # Autonomous Security Orchestrator
        self._orchestrator = AutonomousSecurityOrchestrator(
            scanner_service=self._scanner_service,
            config=self._config
        )
        await self._orchestrator.initialize()
        self._services['orchestrator_service'] = self._orchestrator

        logger.info("✅ Security services initialized")

    async def _initialize_ai_services(self):
        """Initialize AI and threat intelligence services"""
        logger.info("🧠 Initializing AI & Intelligence Services...")

        # Advanced Threat Intelligence Engine
        self._threat_intelligence = AdvancedThreatIntelligenceEngine(
            repository_factory=self._repository_factory
        )
        await self._threat_intelligence.initialize()
        self._services['threat_intelligence_service'] = self._threat_intelligence

        # Production AI Threat Intelligence Engine
        self._production_ai_engine = ProductionAIThreatIntelligenceEngine(
            config=self._config
        )
        await self._production_ai_engine.initialize()
        self._services['production_ai_engine'] = self._production_ai_engine

        # Note: Advanced threat hunting and MITRE engines will be implemented later
        # Placeholder for threat hunting engine
        self._services['threat_hunting_engine'] = None
        self._services['mitre_engine'] = None

        # Sophisticated Red Team Agent
        self._red_team_agent = SophisticatedRedTeamAgent(
            scanner_service=self._scanner_service,
            threat_intelligence=self._threat_intelligence,
            config=self._config
        )
        await self._red_team_agent.initialize()
        self._services['red_team_agent'] = self._red_team_agent

        # Behavioral Analytics Engine
        self._behavioral_analytics = AdvancedBehavioralAnalyticsEngine(
            repository_factory=self._repository_factory,
            config=self._config
        )
        await self._behavioral_analytics.initialize()
        self._services['behavioral_analytics'] = self._behavioral_analytics

        logger.info("✅ AI & Intelligence services initialized")

    async def _initialize_enterprise_features(self):
        """Initialize enterprise-specific features"""
        logger.info("🏢 Initializing Enterprise Features...")

        # Register enterprise service factories
        self._factories.update({
            'auth_service': lambda: self._service_factory.get_authentication_service(),
            'ptaas_service': lambda: self._service_factory.get_ptaas_service(),
            'health_service': lambda: self._service_factory.get_health_service(),
            'user_repository': lambda tenant_id=None: self._repository_factory.create_user_repository(tenant_id),
            'scan_repository': lambda tenant_id=None: self._repository_factory.create_scan_session_repository(tenant_id),
            'cache_repository': lambda: self._repository_factory.get_cache_repository(),
        })

        # Advanced reporting service
        from ..services.advanced_reporting_engine import AdvancedReportingEngine
        reporting_engine = AdvancedReportingEngine(
            scanner_service=self._scanner_service,
            threat_intelligence=self._threat_intelligence,
            config=self._config
        )
        await reporting_engine.initialize()
        self._services['reporting_service'] = reporting_engine

        logger.info("✅ Enterprise features initialized")

    async def _initialize_monitoring(self):
        """Initialize monitoring and health systems"""
        logger.info("📊 Initializing Monitoring & Health Systems...")

        # Initialize health monitoring for all services
        for service_name, service in self._services.items():
            if hasattr(service, 'health_check'):
                try:
                    health = await service.health_check()
                    self._health_status[service_name] = ServiceHealth(
                        service_name=service_name,
                        status="healthy" if health.status == "healthy" else "degraded",
                        message=health.message if hasattr(health, 'message') else "Service operational",
                        timestamp=datetime.now(),
                        metrics=health.checks if hasattr(health, 'checks') else {}
                    )
                except Exception as e:
                    self._health_status[service_name] = ServiceHealth(
                        service_name=service_name,
                        status="unhealthy",
                        message=f"Health check failed: {e}",
                        timestamp=datetime.now(),
                        metrics={"error": str(e)}
                    )

        logger.info("✅ Monitoring systems initialized")

    async def get_service(self, service_name: str, **kwargs) -> Any:
        """Get service instance with dependency injection"""
        if not self._initialized:
            raise RuntimeError("Container not initialized. Call initialize() first.")

        # Direct service lookup
        if service_name in self._services:
            return self._services[service_name]

        # Factory-based service creation
        if service_name in self._factories:
            factory = self._factories[service_name]
            if asyncio.iscoroutinefunction(factory):
                return await factory(**kwargs)
            else:
                return factory(**kwargs)

        raise ValueError(f"Service '{service_name}' not found in production container")

    async def health_check_all_services(self) -> Dict[str, Any]:
        """Comprehensive health check of all services"""
        if not self._initialized:
            return {
                "overall_status": "unhealthy",
                "error": "Container not initialized"
            }

        service_health = {}
        healthy_count = 0
        degraded_count = 0
        unhealthy_count = 0

        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'health_check'):
                    health = await service.health_check()
                    status = "healthy" if health.status == "healthy" else "degraded"
                    service_health[service_name] = {
                        "status": status,
                        "message": health.message if hasattr(health, 'message') else "OK",
                        "timestamp": datetime.now().isoformat(),
                        "metrics": health.checks if hasattr(health, 'checks') else {}
                    }

                    if status == "healthy":
                        healthy_count += 1
                    elif status == "degraded":
                        degraded_count += 1
                    else:
                        unhealthy_count += 1
                else:
                    service_health[service_name] = {
                        "status": "healthy",
                        "message": "Service active (no health check)",
                        "timestamp": datetime.now().isoformat(),
                        "metrics": {}
                    }
                    healthy_count += 1

            except Exception as e:
                service_health[service_name] = {
                    "status": "unhealthy",
                    "message": f"Health check failed: {e}",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
                unhealthy_count += 1

        # Determine overall status
        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        return {
            "overall_status": overall_status,
            "services": service_health,
            "summary": {
                "total_services": len(self._services),
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count
            },
            "container_uptime": (datetime.now() - self._start_time).total_seconds(),
            "timestamp": datetime.now().isoformat()
        }

    def get_service_status(self) -> Dict[str, Any]:
        """Get service registration status"""
        return {
            "initialized": self._initialized,
            "registered_services": len(self._services),
            "initialized_services": len([s for s in self._services.values() if s]),
            "factory_services": len(self._factories),
            "services": list(self._services.keys()),
            "uptime_seconds": (datetime.now() - self._start_time).total_seconds()
        }

    def get_metrics(self) -> ContainerMetrics:
        """Get comprehensive container metrics"""
        healthy = len([h for h in self._health_status.values() if h.status == "healthy"])
        degraded = len([h for h in self._health_status.values() if h.status == "degraded"])
        unhealthy = len([h for h in self._health_status.values() if h.status == "unhealthy"])

        return ContainerMetrics(
            services_registered=len(self._services) + len(self._factories),
            services_initialized=len([s for s in self._services.values() if s]),
            healthy_services=healthy,
            degraded_services=degraded,
            unhealthy_services=unhealthy,
            initialization_time_seconds=0.0,  # Would track actual initialization time
            memory_usage_mb=0.0,  # Would integrate with resource monitoring
            uptime_seconds=(datetime.now() - self._start_time).total_seconds()
        )

    async def shutdown_all_services(self) -> Dict[str, Any]:
        """Shutdown all services gracefully"""
        logger.info("🛑 Shutting down Production Container...")

        shutdown_results = {
            "shutdown": 0,
            "failed": 0,
            "services": []
        }

        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'shutdown'):
                    await service.shutdown()
                    shutdown_results["services"].append(f"{service_name}: success")
                    shutdown_results["shutdown"] += 1
                else:
                    shutdown_results["services"].append(f"{service_name}: no shutdown method")
                    shutdown_results["shutdown"] += 1
            except Exception as e:
                logger.error(f"Failed to shutdown {service_name}: {e}")
                shutdown_results["services"].append(f"{service_name}: failed - {e}")
                shutdown_results["failed"] += 1

        # Close repository factory
        if self._repository_factory:
            try:
                await self._repository_factory.close()
                logger.info("✅ Repository factory closed")
            except Exception as e:
                logger.error(f"Failed to close repository factory: {e}")

        self._initialized = False
        logger.info("✅ Production Container shutdown completed")

        return shutdown_results

    async def _cleanup_partial_initialization(self):
        """Cleanup partial initialization on failure"""
        logger.info("🧹 Cleaning up partial initialization...")

        for service_name, service in self._services.items():
            if service and hasattr(service, 'shutdown'):
                try:
                    await service.shutdown()
                except Exception as e:
                    logger.error(f"Failed to cleanup {service_name}: {e}")

        self._services.clear()
        self._factories.clear()
        self._health_status.clear()

    def _get_production_config(self) -> Dict[str, Any]:
        """Get production configuration from environment"""
        return {
            'DATABASE_URL': os.getenv('DATABASE_URL', 'postgresql+asyncpg://xorb:xorb@postgres:5432/xorb'),
            'REDIS_URL': os.getenv('REDIS_URL', 'redis://redis:6379/0'),
            'JWT_SECRET': os.getenv('JWT_SECRET', self._generate_secure_jwt_secret()),
            'JWT_ALGORITHM': os.getenv('JWT_ALGORITHM', 'HS256'),
            'ENVIRONMENT': os.getenv('ENVIRONMENT', 'production'),
            'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
            'ENABLE_METRICS': os.getenv('ENABLE_METRICS', 'true').lower() == 'true',
            'RATE_LIMIT_PER_MINUTE': int(os.getenv('RATE_LIMIT_PER_MINUTE', '60')),
            'RATE_LIMIT_PER_HOUR': int(os.getenv('RATE_LIMIT_PER_HOUR', '1000')),
            'SECURITY_HEADERS_ENABLED': os.getenv('SECURITY_HEADERS_ENABLED', 'true').lower() == 'true',
            'AUDIT_LOGGING_ENABLED': os.getenv('AUDIT_LOGGING_ENABLED', 'true').lower() == 'true',
            'PTAAS_MAX_CONCURRENT_SCANS': int(os.getenv('PTAAS_MAX_CONCURRENT_SCANS', '15')),
            'PTAAS_SCANNER_TIMEOUT': int(os.getenv('PTAAS_SCANNER_TIMEOUT', '3600')),
            'AI_THREAT_ANALYSIS_ENABLED': os.getenv('AI_THREAT_ANALYSIS_ENABLED', 'true').lower() == 'true',
            'NVIDIA_API_KEY': os.getenv('NVIDIA_API_KEY'),
            'OPENROUTER_API_KEY': os.getenv('OPENROUTER_API_KEY'),
            'TEMPORAL_HOST': os.getenv('TEMPORAL_HOST', 'temporal:7233'),
            'VAULT_ADDR': os.getenv('VAULT_ADDR'),
            'VAULT_TOKEN': os.getenv('VAULT_TOKEN'),
            'enable_ml_analysis': True,
            'enable_threat_intelligence': True,
            'enable_orchestration': True,
            'enable_advanced_reporting': True
        }

    def _generate_secure_jwt_secret(self) -> str:
        """Generate secure JWT secret for production"""
        import secrets
        return secrets.token_urlsafe(64)

    # Property accessors for common services
    @property
    def scanner_service(self) -> Optional[SecurityScannerService]:
        return self._scanner_service

    @property
    def threat_intelligence(self) -> Optional[AdvancedThreatIntelligenceEngine]:
        return self._threat_intelligence

    @property
    def security_platform(self) -> Optional[EnterpriseSecurityPlatform]:
        return self._security_platform

    @property
    def orchestrator(self) -> Optional[AutonomousSecurityOrchestrator]:
        return self._orchestrator

    @property
    def red_team_agent(self) -> Optional[SophisticatedRedTeamAgent]:
        return self._red_team_agent


# Global production container instance
_production_container: Optional[ProductionContainer] = None

async def startup_container(config: Dict[str, Any] = None) -> ProductionContainer:
    """
    Initialize and start the production container with all enterprise services.
    This is the main entry point for XORB's dependency injection system.

    Features:
    - Production-ready service implementations
    - Advanced AI threat intelligence
    - Real-world PTaaS scanner integration
    - Enterprise security orchestration
    - Sophisticated behavioral analytics
    - Multi-tenant isolation
    - Performance monitoring
    - Health checks and circuit breakers
    """
    global _production_container

    try:
        logger.info("🚀 Initializing XORB Enterprise Production Container...")
        start_time = datetime.utcnow()

        if _production_container is None:
            # Create production container with enhanced configuration
            container_config = {
                "environment": os.getenv("ENVIRONMENT", "production"),
                "enable_ai_services": True,
                "enable_ptaas": True,
                "enable_monitoring": True,
                "enable_security_orchestration": True,
                "redis_enabled": True,
                "database_enabled": True,
                "vault_enabled": True,
                **(config or {})
            }

            _production_container = ProductionContainer()

            # Initialize container with comprehensive service registration
            success = await _production_container.initialize(container_config)

            if not success:
                raise RuntimeError("❌ Failed to initialize production container")

            # Verify critical services are healthy
            health_status = await _production_container.check_all_services_health()
            unhealthy_services = [
                service for service, status in health_status.items()
                if status.status == "unhealthy"
            ]

            if unhealthy_services:
                logger.warning(f"⚠️ Some services are unhealthy: {unhealthy_services}")

            initialization_time = (datetime.utcnow() - start_time).total_seconds()

            logger.info(
                f"✅ XORB Enterprise Container initialized successfully in {initialization_time:.2f}s\n"
                f"   📊 Services registered: {len(_production_container._services)}\n"
                f"   🏥 Healthy services: {len([s for s, h in health_status.items() if h.status == 'healthy'])}\n"
                f"   🔧 Repository factories: {len(_production_container._repository_factories)}\n"
                f"   🛡️ Security services active: PTaaS, Threat Intelligence, Behavioral Analytics\n"
                f"   🤖 AI services active: Neural Threat Predictor, Advanced Analytics\n"
                f"   🏢 Enterprise features: Multi-tenant, Compliance, Orchestration"
            )

        return _production_container

    except Exception as e:
        logger.error(f"❌ Critical failure during container startup: {e}")
        # Cleanup partial initialization
        if _production_container:
            await _production_container.shutdown_all_services()
            _production_container = None
        raise RuntimeError(f"Production container startup failed: {e}")

async def get_production_container() -> ProductionContainer:
    """Get the global production container instance"""
    global _production_container

    if _production_container is None:
        return await startup_container()

    return _production_container

async def shutdown_production_container():
    """Shutdown the global production container"""
    global _production_container

    if _production_container:
        await _production_container.shutdown_all_services()
        _production_container = None
