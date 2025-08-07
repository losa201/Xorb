"""
Service interfaces - Define contracts for business operations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID

from ..domain.entities import (
    User, Organization, EmbeddingRequest, EmbeddingResult, 
    DiscoveryWorkflow, AuthToken
)
from ..domain.value_objects import UsageStats, RateLimitInfo


class AuthenticationService(ABC):
    """Interface for authentication operations"""
    
    @abstractmethod
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with credentials"""
        pass
    
    @abstractmethod
    async def create_access_token(self, user: User) -> str:
        """Create access token for user"""
        pass
    
    @abstractmethod
    async def validate_token(self, token: str) -> Optional[User]:
        """Validate access token and return user"""
        pass
    
    @abstractmethod
    async def revoke_token(self, token: str) -> bool:
        """Revoke an access token"""
        pass


class AuthorizationService(ABC):
    """Interface for authorization operations"""
    
    @abstractmethod
    async def check_permission(self, user: User, resource: str, action: str) -> bool:
        """Check if user has permission for resource action"""
        pass
    
    @abstractmethod
    async def get_user_permissions(self, user: User) -> Dict[str, List[str]]:
        """Get all permissions for user"""
        pass


class EmbeddingService(ABC):
    """Interface for embedding operations"""
    
    @abstractmethod
    async def generate_embeddings(
        self,
        texts: List[str],
        model: str,
        input_type: str,
        user: User,
        org: Organization
    ) -> EmbeddingResult:
        """Generate embeddings for texts"""
        pass
    
    @abstractmethod
    async def compute_similarity(
        self,
        text1: str,
        text2: str,
        model: str,
        user: User
    ) -> float:
        """Compute similarity between two texts"""
        pass
    
    @abstractmethod
    async def batch_embeddings(
        self,
        texts: List[str],
        model: str,
        batch_size: int,
        input_type: str,
        user: User,
        org: Organization
    ) -> EmbeddingResult:
        """Process large batches of texts"""
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available embedding models"""
        pass


class DiscoveryService(ABC):
    """Interface for discovery operations"""
    
    @abstractmethod
    async def start_discovery(
        self,
        domain: str,
        user: User,
        org: Organization
    ) -> DiscoveryWorkflow:
        """Start a new discovery workflow"""
        pass
    
    @abstractmethod
    async def get_discovery_results(
        self,
        workflow_id: str,
        user: User
    ) -> Optional[DiscoveryWorkflow]:
        """Get results from discovery workflow"""
        pass
    
    @abstractmethod
    async def get_user_workflows(
        self,
        user: User,
        limit: int = 50,
        offset: int = 0
    ) -> List[DiscoveryWorkflow]:
        """Get workflows for user"""
        pass


class RateLimitService(ABC):
    """Interface for rate limiting operations"""
    
    @abstractmethod
    async def check_rate_limit(
        self,
        org: Organization,
        resource_type: str,
        action: str
    ) -> RateLimitInfo:
        """Check rate limit for organization and resource"""
        pass
    
    @abstractmethod
    async def increment_usage(
        self,
        org: Organization,
        resource_type: str,
        amount: int = 1
    ) -> None:
        """Increment resource usage"""
        pass
    
    @abstractmethod
    async def get_usage_stats(
        self,
        org: Organization
    ) -> Dict[str, Any]:
        """Get usage statistics for organization"""
        pass


class UserService(ABC):
    """Interface for user management operations"""
    
    @abstractmethod
    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: List[str]
    ) -> User:
        """Create a new user"""
        pass
    
    @abstractmethod
    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID"""
        pass
    
    @abstractmethod
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        pass
    
    @abstractmethod
    async def update_user(self, user: User) -> User:
        """Update user information"""
        pass
    
    @abstractmethod
    async def deactivate_user(self, user_id: UUID) -> bool:
        """Deactivate a user"""
        pass


class OrganizationService(ABC):
    """Interface for organization management operations"""
    
    @abstractmethod
    async def create_organization(
        self,
        name: str,
        plan_type: str,
        owner: User
    ) -> Organization:
        """Create a new organization"""
        pass
    
    @abstractmethod
    async def get_organization_by_id(self, org_id: UUID) -> Optional[Organization]:
        """Get organization by ID"""
        pass
    
    @abstractmethod
    async def update_organization(self, organization: Organization) -> Organization:
        """Update organization information"""
        pass
    
    @abstractmethod
    async def get_user_organizations(self, user: User) -> List[Organization]:
        """Get organizations for user"""
        pass


class NotificationService(ABC):
    """Interface for notification operations"""
    
    @abstractmethod
    async def send_notification(
        self,
        user: User,
        message: str,
        notification_type: str
    ) -> bool:
        """Send notification to user"""
        pass
    
    @abstractmethod
    async def send_webhook(
        self,
        org: Organization,
        event: str,
        data: Dict[str, Any]
    ) -> bool:
        """Send webhook to organization"""
        pass


class HealthService(ABC):
    """Interface for health check operations"""
    
    @abstractmethod
    async def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service"""
        pass
    
    @abstractmethod
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        pass