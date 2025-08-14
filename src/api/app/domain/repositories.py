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
        raise NotImplementedError("get_by_id must be implemented by subclass")

    @abstractmethod
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        raise NotImplementedError("get_by_username must be implemented by subclass")

    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        raise NotImplementedError("get_by_email must be implemented by subclass")

    @abstractmethod
    async def create(self, user: User) -> User:
        """Create a new user"""
        raise NotImplementedError("create must be implemented by subclass")

    @abstractmethod
    async def update(self, user: User) -> User:
        """Update an existing user"""
        raise NotImplementedError("update must be implemented by subclass")

    @abstractmethod
    async def delete(self, user_id: UUID) -> bool:
        """Delete a user"""
        raise NotImplementedError("delete must be implemented by subclass")


class OrganizationRepository(ABC):
    """Interface for organization data access"""

    @abstractmethod
    async def get_by_id(self, org_id: UUID) -> Optional[Organization]:
        """Get organization by ID"""
        raise NotImplementedError("get_by_id must be implemented by subclass")

    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[Organization]:
        """Get organization by name"""
        raise NotImplementedError("get_by_name must be implemented by subclass")

    @abstractmethod
    async def create(self, organization: Organization) -> Organization:
        """Create a new organization"""
        raise NotImplementedError("create must be implemented by subclass")

    @abstractmethod
    async def update(self, organization: Organization) -> Organization:
        """Update an existing organization"""
        raise NotImplementedError("update must be implemented by subclass")

    @abstractmethod
    async def get_user_organizations(self, user_id: UUID) -> List[Organization]:
        """Get organizations for a user"""
        raise NotImplementedError("get_user_organizations must be implemented by subclass")


class EmbeddingRepository(ABC):
    """Interface for embedding data access"""

    @abstractmethod
    async def save_request(self, request: EmbeddingRequest) -> EmbeddingRequest:
        """Save an embedding request"""
        raise NotImplementedError("save_request must be implemented by subclass")

    @abstractmethod
    async def save_result(self, result: EmbeddingResult) -> EmbeddingResult:
        """Save embedding results"""
        raise NotImplementedError("save_result must be implemented by subclass")

    @abstractmethod
    async def get_request_by_id(self, request_id: UUID) -> Optional[EmbeddingRequest]:
        """Get embedding request by ID"""
        raise NotImplementedError("get_request_by_id must be implemented by subclass")

    @abstractmethod
    async def get_result_by_request_id(self, request_id: UUID) -> Optional[EmbeddingResult]:
        """Get embedding result by request ID"""
        raise NotImplementedError("get_result_by_request_id must be implemented by subclass")

    @abstractmethod
    async def get_user_requests(
        self,
        user_id: UUID,
        limit: int = 50,
        offset: int = 0
    ) -> List[EmbeddingRequest]:
        """Get embedding requests for a user"""
        raise NotImplementedError("get_user_requests must be implemented by subclass")


class DiscoveryRepository(ABC):
    """Interface for discovery workflow data access"""

    @abstractmethod
    async def save_workflow(self, workflow: DiscoveryWorkflow) -> DiscoveryWorkflow:
        """Save a discovery workflow"""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    async def get_by_id(self, workflow_id: UUID) -> Optional[DiscoveryWorkflow]:
        """Get workflow by ID"""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    async def get_by_workflow_id(self, workflow_id: str) -> Optional[DiscoveryWorkflow]:
        """Get workflow by external workflow ID"""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    async def update_workflow(self, workflow: DiscoveryWorkflow) -> DiscoveryWorkflow:
        """Update workflow status and results"""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    async def get_user_workflows(
        self,
        user_id: UUID,
        limit: int = 50,
        offset: int = 0
    ) -> List[DiscoveryWorkflow]:
        """Get workflows for a user"""
        raise NotImplementedError("Method must be implemented by subclass")


class AuthTokenRepository(ABC):
    """Interface for auth token data access"""

    @abstractmethod
    async def save_token(self, token: AuthToken) -> AuthToken:
        """Save an auth token"""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    async def get_by_token(self, token: str) -> Optional[AuthToken]:
        """Get token info by token string"""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    async def revoke_token(self, token: str) -> bool:
        """Revoke a token"""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    async def revoke_user_tokens(self, user_id: UUID) -> int:
        """Revoke all tokens for a user"""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    async def cleanup_expired_tokens(self) -> int:
        """Clean up expired tokens"""
        raise NotImplementedError("Method must be implemented by subclass")


class CacheRepository(ABC):
    """Interface for caching operations"""

    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    async def set(self, key: str, value: str, ttl: int = None) -> bool:
        """Set value in cache with optional TTL"""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a counter"""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    async def decrement(self, key: str, amount: int = 1) -> int:
        """Decrement a counter"""
        raise NotImplementedError("Method must be implemented by subclass")


class ScanSessionRepository(ABC):
    """Interface for scan session data access"""

    @abstractmethod
    async def create_session(self, session_data: dict) -> dict:
        """Create a new scan session"""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    async def get_session(self, session_id: UUID) -> Optional[dict]:
        """Get scan session by ID"""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    async def update_session(self, session_id: UUID, updates: dict) -> bool:
        """Update scan session"""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    async def get_user_sessions(self, user_id: UUID) -> List[dict]:
        """Get scan sessions for a user"""
        raise NotImplementedError("Method must be implemented by subclass")


class TenantRepository(ABC):
    """Interface for tenant data access"""

    @abstractmethod
    async def create_tenant(self, tenant_data: dict) -> dict:
        """Create a new tenant"""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    async def get_tenant(self, tenant_id: UUID) -> Optional[dict]:
        """Get tenant by ID"""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    async def update_tenant(self, tenant_id: UUID, updates: dict) -> bool:
        """Update tenant"""
        raise NotImplementedError("Method must be implemented by subclass")

    @abstractmethod
    async def get_tenant_by_name(self, name: str) -> Optional[dict]:
        """Get tenant by name"""
        raise NotImplementedError("Method must be implemented by subclass")
