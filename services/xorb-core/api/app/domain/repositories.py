"""
Repository interfaces - Define contracts for data access without implementation details.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from .entities import (
    User, Organization, EmbeddingRequest, EmbeddingResult, 
    DiscoveryWorkflow, AuthToken
)


class UserRepository(ABC):
    """Interface for user data access"""
    
    @abstractmethod
    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID"""
        pass
    
    @abstractmethod
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        pass
    
    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        pass
    
    @abstractmethod
    async def create(self, user: User) -> User:
        """Create a new user"""
        pass
    
    @abstractmethod
    async def update(self, user: User) -> User:
        """Update an existing user"""
        pass
    
    @abstractmethod
    async def delete(self, user_id: UUID) -> bool:
        """Delete a user"""
        pass


class OrganizationRepository(ABC):
    """Interface for organization data access"""
    
    @abstractmethod
    async def get_by_id(self, org_id: UUID) -> Optional[Organization]:
        """Get organization by ID"""
        pass
    
    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[Organization]:
        """Get organization by name"""
        pass
    
    @abstractmethod
    async def create(self, organization: Organization) -> Organization:
        """Create a new organization"""
        pass
    
    @abstractmethod
    async def update(self, organization: Organization) -> Organization:
        """Update an existing organization"""
        pass
    
    @abstractmethod
    async def get_user_organizations(self, user_id: UUID) -> List[Organization]:
        """Get organizations for a user"""
        pass


class EmbeddingRepository(ABC):
    """Interface for embedding data access"""
    
    @abstractmethod
    async def save_request(self, request: EmbeddingRequest) -> EmbeddingRequest:
        """Save an embedding request"""
        pass
    
    @abstractmethod
    async def save_result(self, result: EmbeddingResult) -> EmbeddingResult:
        """Save embedding results"""
        pass
    
    @abstractmethod
    async def get_request_by_id(self, request_id: UUID) -> Optional[EmbeddingRequest]:
        """Get embedding request by ID"""
        pass
    
    @abstractmethod
    async def get_result_by_request_id(self, request_id: UUID) -> Optional[EmbeddingResult]:
        """Get embedding result by request ID"""
        pass
    
    @abstractmethod
    async def get_user_requests(
        self, 
        user_id: UUID, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[EmbeddingRequest]:
        """Get embedding requests for a user"""
        pass


class DiscoveryRepository(ABC):
    """Interface for discovery workflow data access"""
    
    @abstractmethod
    async def save_workflow(self, workflow: DiscoveryWorkflow) -> DiscoveryWorkflow:
        """Save a discovery workflow"""
        pass
    
    @abstractmethod
    async def get_by_id(self, workflow_id: UUID) -> Optional[DiscoveryWorkflow]:
        """Get workflow by ID"""
        pass
    
    @abstractmethod
    async def get_by_workflow_id(self, workflow_id: str) -> Optional[DiscoveryWorkflow]:
        """Get workflow by external workflow ID"""
        pass
    
    @abstractmethod
    async def update_workflow(self, workflow: DiscoveryWorkflow) -> DiscoveryWorkflow:
        """Update workflow status and results"""
        pass
    
    @abstractmethod
    async def get_user_workflows(
        self, 
        user_id: UUID, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[DiscoveryWorkflow]:
        """Get workflows for a user"""
        pass


class AuthTokenRepository(ABC):
    """Interface for auth token data access"""
    
    @abstractmethod
    async def save_token(self, token: AuthToken) -> AuthToken:
        """Save an auth token"""
        pass
    
    @abstractmethod
    async def get_by_token(self, token: str) -> Optional[AuthToken]:
        """Get token info by token string"""
        pass
    
    @abstractmethod
    async def revoke_token(self, token: str) -> bool:
        """Revoke a token"""
        pass
    
    @abstractmethod
    async def revoke_user_tokens(self, user_id: UUID) -> int:
        """Revoke all tokens for a user"""
        pass
    
    @abstractmethod
    async def cleanup_expired_tokens(self) -> int:
        """Clean up expired tokens"""
        pass


class CacheRepository(ABC):
    """Interface for caching operations"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: str, ttl: int = None) -> bool:
        """Set value in cache with optional TTL"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a counter"""
        pass
    
    @abstractmethod
    async def decrement(self, key: str, amount: int = 1) -> int:
        """Decrement a counter"""
        pass