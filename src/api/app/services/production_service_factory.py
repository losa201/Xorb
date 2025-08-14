"""
Production Service Factory
Centralized factory for creating and managing all production service instances
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .production_service_implementations import (
    ProductionAuthenticationService, ProductionAuthorizationService,
    ProductionPTaaSService, ProductionHealthService
)
from .production_intelligence_service import ProductionThreatIntelligenceService

logger = logging.getLogger(__name__)


class ProductionServiceFactory:
    """
    Factory class for creating production-ready service instances with proper configuration
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._redis_client = None
        self._db_client = None
        self._service_instances = {}

        # Default configuration
        self.default_config = {
            "jwt_secret": "default-jwt-secret-change-in-production",
            "jwt_expiry_hours": 24,
            "refresh_token_expiry_days": 30,
            "enable_ml_analysis": True,
            "enable_threat_intelligence": True,
            "confidence_threshold": 0.7,
            "cache_ttl": 3600,
            "environment": "production"
        }

        # Merge with provided config
        self.config = {**self.default_config, **self.config}

    def set_redis_client(self, redis_client):
        """Set Redis client for services that need it"""
        self._redis_client = redis_client
        logger.info("✅ Redis client configured for production services")

    def set_db_client(self, db_client):
        """Set database client for services that need it"""
        self._db_client = db_client
        logger.info("✅ Database client configured for production services")

    def create_authentication_service(self) -> ProductionAuthenticationService:
        """Create production authentication service"""
        if "auth_service" not in self._service_instances:
            logger.info("🔐 Creating Production Authentication Service...")

            service = ProductionAuthenticationService(
                jwt_secret=self.config.get("jwt_secret"),
                redis_client=self._redis_client
            )

            self._service_instances["auth_service"] = service
            logger.info("✅ Production Authentication Service created")

        return self._service_instances["auth_service"]

    def create_authorization_service(self) -> ProductionAuthorizationService:
        """Create production authorization service"""
        if "authz_service" not in self._service_instances:
            logger.info("🛡️ Creating Production Authorization Service...")

            service = ProductionAuthorizationService(
                redis_client=self._redis_client
            )

            self._service_instances["authz_service"] = service
            logger.info("✅ Production Authorization Service created")

        return self._service_instances["authz_service"]

    def create_ptaas_service(self) -> ProductionPTaaSService:
        """Create production PTaaS service"""
        if "ptaas_service" not in self._service_instances:
            logger.info("🎯 Creating Production PTaaS Service...")

            service = ProductionPTaaSService(
                redis_client=self._redis_client
            )

            self._service_instances["ptaas_service"] = service
            logger.info("✅ Production PTaaS Service created with real scanner integration")

        return self._service_instances["ptaas_service"]

    def create_threat_intelligence_service(self) -> ProductionThreatIntelligenceService:
        """Create production threat intelligence service"""
        if "threat_intel_service" not in self._service_instances:
            logger.info("🧠 Creating Production Threat Intelligence Service...")

            service = ProductionThreatIntelligenceService(
                redis_client=self._redis_client,
                config={
                    "enable_ml_analysis": self.config.get("enable_ml_analysis", True),
                    "confidence_threshold": self.config.get("confidence_threshold", 0.7),
                    "cache_ttl": self.config.get("cache_ttl", 3600),
                    "environment": self.config.get("environment", "production")
                }
            )

            self._service_instances["threat_intel_service"] = service
            logger.info("✅ Production Threat Intelligence Service created with AI capabilities")

        return self._service_instances["threat_intel_service"]

    def create_health_service(self) -> ProductionHealthService:
        """Create production health service"""
        if "health_service" not in self._service_instances:
            logger.info("🏥 Creating Production Health Service...")

            service = ProductionHealthService(
                redis_client=self._redis_client,
                db_client=self._db_client
            )

            self._service_instances["health_service"] = service
            logger.info("✅ Production Health Service created")

        return self._service_instances["health_service"]

    def create_all_services(self) -> Dict[str, Any]:
        """Create all production services"""
        logger.info("🏭 Creating All Production Services...")

        services = {
            "authentication_service": self.create_authentication_service(),
            "authorization_service": self.create_authorization_service(),
            "ptaas_service": self.create_ptaas_service(),
            "threat_intelligence_service": self.create_threat_intelligence_service(),
            "health_service": self.create_health_service()
        }

        logger.info(f"✅ All Production Services Created: {len(services)} services")

        return services

    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all created services"""
        return {
            "factory_config": {
                "environment": self.config.get("environment"),
                "ml_analysis_enabled": self.config.get("enable_ml_analysis"),
                "threat_intelligence_enabled": self.config.get("enable_threat_intelligence")
            },
            "created_services": list(self._service_instances.keys()),
            "service_count": len(self._service_instances),
            "redis_configured": self._redis_client is not None,
            "database_configured": self._db_client is not None,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def health_check_all(self) -> Dict[str, Any]:
        """Perform health check on all created services"""
        health_results = {
            "overall_status": "healthy",
            "services": {},
            "timestamp": datetime.utcnow().isoformat()
        }

        unhealthy_count = 0

        for service_name, service_instance in self._service_instances.items():
            try:
                # Check if service has health check method
                if hasattr(service_instance, 'health_check'):
                    result = await service_instance.health_check()
                elif hasattr(service_instance, 'get_system_health'):
                    result = await service_instance.get_system_health()
                else:
                    # Default health check - service exists and is callable
                    result = {
                        "status": "healthy",
                        "message": "Service instance active"
                    }

                health_results["services"][service_name] = result

                if result.get("status") != "healthy":
                    unhealthy_count += 1

            except Exception as e:
                logger.error(f"Health check failed for {service_name}: {e}")
                health_results["services"][service_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                unhealthy_count += 1

        # Determine overall status
        if unhealthy_count > 0:
            if unhealthy_count == len(self._service_instances):
                health_results["overall_status"] = "unhealthy"
            else:
                health_results["overall_status"] = "degraded"

        health_results["healthy_services"] = len(self._service_instances) - unhealthy_count
        health_results["total_services"] = len(self._service_instances)

        return health_results

    async def shutdown_all_services(self) -> Dict[str, Any]:
        """Shutdown all services gracefully"""
        logger.info("🛑 Shutting down all production services...")

        shutdown_results = {
            "shutdown_services": 0,
            "failed_shutdowns": 0,
            "total_services": len(self._service_instances),
            "timestamp": datetime.utcnow().isoformat()
        }

        for service_name, service_instance in self._service_instances.items():
            try:
                # Call shutdown method if available
                if hasattr(service_instance, 'shutdown'):
                    await service_instance.shutdown()
                    logger.info(f"✅ Shutdown service: {service_name}")
                elif hasattr(service_instance, 'close'):
                    await service_instance.close()
                    logger.info(f"✅ Closed service: {service_name}")
                else:
                    logger.info(f"ℹ️ No shutdown method for: {service_name}")

                shutdown_results["shutdown_services"] += 1

            except Exception as e:
                logger.error(f"❌ Failed to shutdown {service_name}: {e}")
                shutdown_results["failed_shutdowns"] += 1

        # Clear service instances
        self._service_instances.clear()

        logger.info(f"🏁 Production Services Shutdown Complete: {shutdown_results}")
        return shutdown_results


# Global factory instance
_production_factory = None


def get_production_factory(config: Dict[str, Any] = None) -> ProductionServiceFactory:
    """Get or create the global production service factory"""
    global _production_factory

    if _production_factory is None:
        _production_factory = ProductionServiceFactory(config)
        logger.info("🏭 Production Service Factory initialized")

    return _production_factory


def initialize_production_services(
    config: Dict[str, Any] = None,
    redis_client=None,
    db_client=None
) -> Dict[str, Any]:
    """Initialize all production services with configuration"""

    factory = get_production_factory(config)

    # Set clients
    if redis_client:
        factory.set_redis_client(redis_client)

    if db_client:
        factory.set_db_client(db_client)

    # Create all services
    services = factory.create_all_services()

    logger.info("🎉 Production Services Initialization Complete")

    return {
        "factory": factory,
        "services": services,
        "status": factory.get_service_status()
    }


# Service getter functions for easy access
def get_auth_service() -> ProductionAuthenticationService:
    """Get authentication service instance"""
    factory = get_production_factory()
    return factory.create_authentication_service()


def get_authz_service() -> ProductionAuthorizationService:
    """Get authorization service instance"""
    factory = get_production_factory()
    return factory.create_authorization_service()


def get_ptaas_service() -> ProductionPTaaSService:
    """Get PTaaS service instance"""
    factory = get_production_factory()
    return factory.create_ptaas_service()


def get_threat_intelligence_service() -> ProductionThreatIntelligenceService:
    """Get threat intelligence service instance"""
    factory = get_production_factory()
    return factory.create_threat_intelligence_service()


def get_health_service() -> ProductionHealthService:
    """Get health service instance"""
    factory = get_production_factory()
    return factory.create_health_service()


# Health check endpoint for the factory
async def factory_health_check() -> Dict[str, Any]:
    """Perform health check on all factory services"""
    factory = get_production_factory()
    return await factory.health_check_all()
