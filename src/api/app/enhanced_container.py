"""
Enhanced Dependency Injection Container - Production-ready service management
Manages all services, repositories, and dependencies with proper lifecycle management
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, Type, TypeVar, Callable
from contextlib import asynccontextmanager
from functools import lru_cache

from .infrastructure.production_repositories import (
    RepositoryFactory, 
    ProductionUserRepository,
    ProductionScanSessionRepository, 
    ProductionRedisCache,
    get_repository_factory
)
from .services.production_service_implementations import (
    ServiceFactory,
    ProductionAuthenticationService,
    # ProductionPTaaSService,  # Not implemented
    # ProductionHealthService,  # Not implemented
    get_service_factory
)
from .services.advanced_ai_threat_intelligence import AdvancedThreatIntelligenceEngine
from .services.ptaas_scanner_service import get_scanner_service
from .infrastructure.redis_compatibility import get_redis_client

logger = logging.getLogger(__name__)

T = TypeVar('T')


class EnhancedContainer:
    """
    Production-ready dependency injection container with:
    - Singleton and transient service management
    - Async initialization and cleanup
    - Health monitoring integration
    - Configuration management
    - Service lifecycle management
    """
    
    def __init__(self):
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._initialized = False
        self._repository_factory: Optional[RepositoryFactory] = None
        self._service_factory: Optional[ServiceFactory] = None
        self._config = {}
        
    async def initialize(self, config: Dict[str, Any] = None):
        """Initialize container with configuration"""
        if self._initialized:
            logger.warning("Container already initialized")
            return
            
        try:
            self._config = config or self._get_default_config()
            
            # Initialize repository factory
            await self._init_repository_factory()
            
            # Initialize service factory
            await self._init_service_factory()
            
            # Register core services
            await self._register_core_services()
            
            # Register enhanced services
            await self._register_enhanced_services()
            
            self._initialized = True
            logger.info("Enhanced container initialized successfully")
            
        except Exception as e:
            logger.error(f"Container initialization failed: {e}")
            raise
    
    async def _init_repository_factory(self):
        """Initialize repository factory with configuration"""
        database_url = self._config.get('DATABASE_URL', 'sqlite+aiosqlite:///./xorb.db')
        redis_url = self._config.get('REDIS_URL', 'redis://localhost:6379/0')
        
        self._repository_factory = await get_repository_factory(database_url, redis_url)
        self._singletons['repository_factory'] = self._repository_factory
        
        logger.info("Repository factory initialized")
    
    async def _init_service_factory(self):
        """Initialize service factory with configuration"""
        jwt_secret = self._config.get('JWT_SECRET', 'dev-secret-key-change-in-production')
        
        if not jwt_secret or jwt_secret == 'dev-secret-key-change-in-production':
            logger.warning("Using default JWT secret - change in production!")
        
        self._service_factory = await get_service_factory(self._repository_factory, jwt_secret)
        self._singletons['service_factory'] = self._service_factory
        
        logger.info("Service factory initialized")
    
    async def _register_core_services(self):
        """Register core XORB services"""
        # Authentication Service
        self._factories['auth_service'] = lambda: self._service_factory.get_authentication_service()
        
        # PTaaS Service
        self._factories['ptaas_service'] = lambda: self._service_factory.get_ptaas_service()
        
        # Health Service
        self._factories['health_service'] = lambda: self._service_factory.get_health_service()
        
        # Scanner Service
        self._factories['scanner_service'] = get_scanner_service
        
        logger.info("Core services registered")
    
    async def _register_enhanced_services(self):
        """Register enhanced AI and intelligence services"""
        # AI Threat Intelligence Engine
        async def create_threat_intelligence():
            engine = AdvancedThreatIntelligenceEngine(self._repository_factory)
            return engine
        
        self._factories['threat_intelligence'] = create_threat_intelligence
        
        # Enhanced Repository Services
        self._factories['user_repository'] = lambda tenant_id=None: self._repository_factory.create_user_repository(tenant_id)
        self._factories['scan_repository'] = lambda tenant_id=None: self._repository_factory.create_scan_session_repository(tenant_id)
        self._factories['cache_repository'] = lambda: self._repository_factory.get_cache_repository()
        
        logger.info("Enhanced services registered")
    
    async def get(self, service_name: str, **kwargs) -> Any:
        """Get service instance with dependency injection"""
        if not self._initialized:
            raise RuntimeError("Container not initialized. Call initialize() first.")
        
        # Check singletons first
        if service_name in self._singletons:
            return self._singletons[service_name]
        
        # Check factories
        if service_name in self._factories:
            factory = self._factories[service_name]
            
            # Handle async factories
            if asyncio.iscoroutinefunction(factory):
                instance = await factory(**kwargs)
            else:
                instance = factory(**kwargs)
            
            # Cache singletons (services without parameters)
            if not kwargs and service_name.endswith('_service'):
                self._singletons[service_name] = instance
            
            return instance
        
        raise ValueError(f"Service '{service_name}' not found in container")
    
    def register_singleton(self, name: str, instance: Any):
        """Register singleton instance"""
        self._singletons[name] = instance
        logger.debug(f"Singleton '{name}' registered")
    
    def register_factory(self, name: str, factory: Callable):
        """Register factory function"""
        self._factories[name] = factory
        logger.debug(f"Factory '{name}' registered")
    
    async def get_auth_service(self) -> ProductionAuthenticationService:
        """Get authentication service"""
        return await self.get('auth_service')
    
    async def get_ptaas_service(self) -> Any:  # ProductionPTaaSService not implemented
        """Get PTaaS service"""
        return await self.get('ptaas_service')
    
    async def get_health_service(self) -> Any:  # ProductionHealthService not implemented
        """Get health service"""
        return await self.get('health_service')
    
    async def get_threat_intelligence(self) -> AdvancedThreatIntelligenceEngine:
        """Get threat intelligence engine"""
        return await self.get('threat_intelligence')
    
    async def get_scanner_service(self):
        """Get scanner service"""
        return await self.get('scanner_service')
    
    def get_user_repository(self, tenant_id: Optional[str] = None) -> ProductionUserRepository:
        """Get user repository"""
        return self._repository_factory.create_user_repository(tenant_id)
    
    def get_scan_repository(self, tenant_id: Optional[str] = None) -> ProductionScanSessionRepository:
        """Get scan session repository"""
        return self._repository_factory.create_scan_session_repository(tenant_id)
    
    def get_cache_repository(self) -> ProductionRedisCache:
        """Get cache repository"""
        return self._repository_factory.get_cache_repository()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of all services"""
        if not self._initialized:
            return {
                "status": "unhealthy",
                "error": "Container not initialized"
            }
        
        try:
            health_service = await self.get_health_service()
            return await health_service.get_system_health()
        except Exception as e:
            logger.error(f"Container health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "container_status": "degraded"
            }
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration"""
        self._config.update(updates)
        logger.info(f"Configuration updated with {len(updates)} keys")
    
    async def shutdown(self):
        """Shutdown container and cleanup resources"""
        try:
            logger.info("Shutting down enhanced container...")
            
            # Close repository factory
            if self._repository_factory:
                await self._repository_factory.close()
            
            # Clear caches
            self._singletons.clear()
            self._factories.clear()
            
            self._initialized = False
            logger.info("Container shutdown completed")
            
        except Exception as e:
            logger.error(f"Container shutdown failed: {e}")
            raise
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration from environment"""
        return {
            'DATABASE_URL': os.getenv('DATABASE_URL', 'sqlite+aiosqlite:///./xorb.db'),
            'REDIS_URL': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
            'JWT_SECRET': os.getenv('JWT_SECRET', 'dev-secret-key-change-in-production'),
            'JWT_ALGORITHM': os.getenv('JWT_ALGORITHM', 'HS256'),
            'ENVIRONMENT': os.getenv('ENVIRONMENT', 'development'),
            'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
            'ENABLE_METRICS': os.getenv('ENABLE_METRICS', 'true').lower() == 'true',
            'RATE_LIMIT_PER_MINUTE': int(os.getenv('RATE_LIMIT_PER_MINUTE', '60')),
            'RATE_LIMIT_PER_HOUR': int(os.getenv('RATE_LIMIT_PER_HOUR', '1000')),
            'CORS_ALLOW_ORIGINS': os.getenv('CORS_ALLOW_ORIGINS', '*').split(','),
            'SECURITY_HEADERS_ENABLED': os.getenv('SECURITY_HEADERS_ENABLED', 'true').lower() == 'true',
            'AUDIT_LOGGING_ENABLED': os.getenv('AUDIT_LOGGING_ENABLED', 'true').lower() == 'true',
            'PTAAS_MAX_CONCURRENT_SCANS': int(os.getenv('PTAAS_MAX_CONCURRENT_SCANS', '10')),
            'PTAAS_SCANNER_TIMEOUT': int(os.getenv('PTAAS_SCANNER_TIMEOUT', '1800')),
            'AI_THREAT_ANALYSIS_ENABLED': os.getenv('AI_THREAT_ANALYSIS_ENABLED', 'true').lower() == 'true',
            'NVIDIA_API_KEY': os.getenv('NVIDIA_API_KEY'),
            'OPENROUTER_API_KEY': os.getenv('OPENROUTER_API_KEY'),
            'TEMPORAL_HOST': os.getenv('TEMPORAL_HOST', 'localhost:7233'),
            'VAULT_ADDR': os.getenv('VAULT_ADDR'),
            'VAULT_TOKEN': os.getenv('VAULT_TOKEN')
        }
    
    @property
    def is_initialized(self) -> bool:
        """Check if container is initialized"""
        return self._initialized
    
    @property
    def service_count(self) -> int:
        """Get count of registered services"""
        return len(self._singletons) + len(self._factories)
    
    def list_services(self) -> Dict[str, str]:
        """List all registered services"""
        services = {}
        
        for name in self._singletons:
            services[name] = "singleton"
        
        for name in self._factories:
            services[name] = "factory"
        
        return services


# Global container instance
_container: Optional[EnhancedContainer] = None


async def get_container(config: Dict[str, Any] = None) -> EnhancedContainer:
    """Get global container instance"""
    global _container
    
    if _container is None:
        _container = EnhancedContainer()
        await _container.initialize(config)
    
    return _container


async def shutdown_container():
    """Shutdown global container"""
    global _container
    if _container:
        await _container.shutdown()
        _container = None


@asynccontextmanager
async def container_context(config: Dict[str, Any] = None):
    """Context manager for container lifecycle"""
    container = None
    try:
        container = await get_container(config)
        yield container
    finally:
        if container:
            await shutdown_container()


class ServiceProvider:
    """Service provider for easier dependency injection in routers"""
    
    def __init__(self, container: EnhancedContainer):
        self.container = container
    
    async def get_auth_service(self) -> ProductionAuthenticationService:
        return await self.container.get_auth_service()
    
    async def get_ptaas_service(self) -> Any:  # ProductionPTaaSService not implemented
        return await self.container.get_ptaas_service()
    
    async def get_health_service(self) -> Any:  # ProductionHealthService not implemented
        return await self.container.get_health_service()
    
    async def get_threat_intelligence(self) -> AdvancedThreatIntelligenceEngine:
        return await self.container.get_threat_intelligence()
    
    def get_user_repository(self, tenant_id: Optional[str] = None):
        return self.container.get_user_repository(tenant_id)
    
    def get_scan_repository(self, tenant_id: Optional[str] = None):
        return self.container.get_scan_repository(tenant_id)
    
    def get_cache_repository(self):
        return self.container.get_cache_repository()


async def get_service_provider() -> ServiceProvider:
    """Get service provider instance"""
    container = await get_container()
    return ServiceProvider(container)