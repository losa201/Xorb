"""
Dependency injection container for managing service dependencies
"""

import os
from typing import Dict, Any, TypeVar, Type, Optional

from .domain.repositories import (
    UserRepository, OrganizationRepository, EmbeddingRepository,
    DiscoveryRepository, AuthTokenRepository, CacheRepository
)
from .services.interfaces import (
    AuthenticationService, EmbeddingService, DiscoveryService
)
from .services.auth_service import AuthenticationServiceImpl
from .services.embedding_service import EmbeddingServiceImpl
from .services.discovery_service import DiscoveryServiceImpl
from .infrastructure.repositories import (
    InMemoryUserRepository, InMemoryOrganizationRepository,
    InMemoryEmbeddingRepository, InMemoryDiscoveryRepository,
    InMemoryAuthTokenRepository, RedisCacheRepository
)

T = TypeVar('T')


class Container:
    """Dependency injection container"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._config = self._load_config()
        
        # Register default implementations
        self._register_repositories()
        self._register_services()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        return {
            'secret_key': os.getenv('SECRET_KEY', 'default_secret_key'),
            'algorithm': os.getenv('ALGORITHM', 'HS256'),
            'access_token_expire_minutes': int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', '30')),
            'nvidia_api_key': os.getenv('NVIDIA_API_KEY', 'your_nvidia_api_key_here'),
            'nvidia_base_url': os.getenv('NVIDIA_BASE_URL', 'https://integrate.api.nvidia.com/v1'),
            'temporal_url': os.getenv('TEMPORAL_URL', 'temporal:7233'),
            'redis_url': os.getenv('REDIS_URL', 'redis://redis:6379/0'),
            'task_queue': os.getenv('TASK_QUEUE', 'xorb-task-queue')
        }
    
    def _register_repositories(self):
        """Register repository implementations"""
        
        # For development/testing, use in-memory repositories
        # In production, these would be replaced with database implementations
        self.register_singleton(UserRepository, InMemoryUserRepository)
        self.register_singleton(OrganizationRepository, InMemoryOrganizationRepository)
        self.register_singleton(EmbeddingRepository, InMemoryEmbeddingRepository)
        self.register_singleton(DiscoveryRepository, InMemoryDiscoveryRepository)
        self.register_singleton(AuthTokenRepository, InMemoryAuthTokenRepository)
        
        # Redis cache repository
        self.register_singleton(
            CacheRepository, 
            lambda: RedisCacheRepository(self._config['redis_url'])
        )
    
    def _register_services(self):
        """Register service implementations"""
        
        # Authentication service
        self.register_singleton(
            AuthenticationService,
            lambda: AuthenticationServiceImpl(
                user_repository=self.get(UserRepository),
                token_repository=self.get(AuthTokenRepository),
                secret_key=self._config['secret_key'],
                algorithm=self._config['algorithm'],
                access_token_expire_minutes=self._config['access_token_expire_minutes']
            )
        )
        
        # Embedding service
        self.register_singleton(
            EmbeddingService,
            lambda: EmbeddingServiceImpl(
                embedding_repository=self.get(EmbeddingRepository),
                nvidia_api_key=self._config['nvidia_api_key'],
                nvidia_base_url=self._config['nvidia_base_url']
            )
        )
        
        # Discovery service
        self.register_singleton(
            DiscoveryService,
            lambda: DiscoveryServiceImpl(
                discovery_repository=self.get(DiscoveryRepository),
                temporal_url=self._config['temporal_url'],
                task_queue=self._config['task_queue']
            )
        )
    
    def register_singleton(self, interface: Type[T], implementation) -> None:
        """Register a singleton service"""
        key = self._get_key(interface)
        self._services[key] = ('singleton', implementation)
    
    def register_transient(self, interface: Type[T], implementation) -> None:
        """Register a transient service"""
        key = self._get_key(interface)
        self._services[key] = ('transient', implementation)
    
    def get(self, interface: Type[T]) -> T:
        """Get service instance"""
        key = self._get_key(interface)
        
        if key not in self._services:
            raise ValueError(f"Service {interface.__name__} not registered")
        
        service_type, implementation = self._services[key]
        
        if service_type == 'singleton':
            if key not in self._singletons:
                if callable(implementation):
                    self._singletons[key] = implementation()
                else:
                    self._singletons[key] = implementation
            return self._singletons[key]
        
        else:  # transient
            if callable(implementation):
                return implementation()
            else:
                return implementation
    
    def _get_key(self, interface: Type[T]) -> str:
        """Get string key for interface"""
        return f"{interface.__module__}.{interface.__name__}"
    
    def override(self, interface: Type[T], implementation) -> None:
        """Override a service registration (useful for testing)"""
        key = self._get_key(interface)
        # Remove from singletons if it exists
        if key in self._singletons:
            del self._singletons[key]
        # Register new implementation as singleton
        self.register_singleton(interface, implementation)
    
    async def initialize(self):
        """Initialize services that require async setup"""
        
        # Initialize cache repository
        cache_repo = self.get(CacheRepository)
        if hasattr(cache_repo, 'initialize'):
            await cache_repo.initialize()
        
        # Add any other async initialization here
        
        # Seed with default data for development
        await self._seed_development_data()
    
    async def _seed_development_data(self):
        """Seed with default data for development"""
        
        # Create default user
        user_repo = self.get(UserRepository)
        org_repo = self.get(OrganizationRepository)
        
        # Check if default user already exists
        existing_user = await user_repo.get_by_username("admin")
        if not existing_user:
            from .domain.entities import User, Organization
            
            # Create default user
            default_user = User.create(
                username="admin",
                email="admin@xorb.com",
                roles=["admin", "user"]
            )
            await user_repo.create(default_user)
            
            # Create default organization
            default_org = Organization.create(
                name="Default Organization",
                plan_type="Enterprise"
            )
            await org_repo.create(default_org)
            
            # Associate user with organization
            if hasattr(org_repo, 'add_user_to_organization'):
                await org_repo.add_user_to_organization(default_user.id, default_org.id)


# Global container instance
container = Container()


def get_container() -> Container:
    """Get the global container instance"""
    return container