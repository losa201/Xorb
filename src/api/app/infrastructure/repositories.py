"""
Repository implementations for data access
"""

import json
from typing import List, Optional, Dict, Any
from uuid import UUID

# Optional dependencies
try:
    import aioredis
    AIOREDIS_AVAILABLE = True
except (ImportError, TypeError):
    # TypeError can happen with version conflicts
    AIOREDIS_AVAILABLE = False
    aioredis = None

try:
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import select, update, delete
    from sqlalchemy.orm import selectinload
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from ..domain.entities import (
    User, Organization, EmbeddingRequest, EmbeddingResult,
    DiscoveryWorkflow, AuthToken
)
from ..domain.repositories import (
    UserRepository, OrganizationRepository, EmbeddingRepository,
    DiscoveryRepository, AuthTokenRepository, CacheRepository
)


class InMemoryUserRepository(UserRepository):
    """In-memory user repository for testing/development"""
    
    def __init__(self):
        self._users: Dict[UUID, User] = {}
        self._username_index: Dict[str, UUID] = {}
        self._email_index: Dict[str, UUID] = {}
    
    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        return self._users.get(user_id)
    
    async def get_by_username(self, username: str) -> Optional[User]:
        user_id = self._username_index.get(username)
        return self._users.get(user_id) if user_id else None
    
    async def get_by_email(self, email: str) -> Optional[User]:
        user_id = self._email_index.get(email)
        return self._users.get(user_id) if user_id else None
    
    async def create(self, user: User) -> User:
        self._users[user.id] = user
        self._username_index[user.username] = user.id
        self._email_index[user.email] = user.id
        return user
    
    async def update(self, user: User) -> User:
        if user.id in self._users:
            old_user = self._users[user.id]
            # Update indexes if username/email changed
            if old_user.username != user.username:
                del self._username_index[old_user.username]
                self._username_index[user.username] = user.id
            if old_user.email != user.email:
                del self._email_index[old_user.email]
                self._email_index[user.email] = user.id
            
            self._users[user.id] = user
        return user
    
    async def delete(self, user_id: UUID) -> bool:
        if user_id in self._users:
            user = self._users[user_id]
            del self._users[user_id]
            del self._username_index[user.username]
            del self._email_index[user.email]
            return True
        return False


class InMemoryOrganizationRepository(OrganizationRepository):
    """In-memory organization repository for testing/development"""
    
    def __init__(self):
        self._organizations: Dict[UUID, Organization] = {}
        self._name_index: Dict[str, UUID] = {}
        self._user_orgs: Dict[UUID, List[UUID]] = {}
    
    async def get_by_id(self, org_id: UUID) -> Optional[Organization]:
        return self._organizations.get(org_id)
    
    async def get_by_name(self, name: str) -> Optional[Organization]:
        org_id = self._name_index.get(name)
        return self._organizations.get(org_id) if org_id else None
    
    async def create(self, organization: Organization) -> Organization:
        self._organizations[organization.id] = organization
        self._name_index[organization.name] = organization.id
        return organization
    
    async def update(self, organization: Organization) -> Organization:
        if organization.id in self._organizations:
            old_org = self._organizations[organization.id]
            if old_org.name != organization.name:
                del self._name_index[old_org.name]
                self._name_index[organization.name] = organization.id
            
            self._organizations[organization.id] = organization
        return organization
    
    async def get_user_organizations(self, user_id: UUID) -> List[Organization]:
        org_ids = self._user_orgs.get(user_id, [])
        return [self._organizations[org_id] for org_id in org_ids if org_id in self._organizations]
    
    async def add_user_to_organization(self, user_id: UUID, org_id: UUID):
        """Helper method to associate user with organization"""
        if user_id not in self._user_orgs:
            self._user_orgs[user_id] = []
        if org_id not in self._user_orgs[user_id]:
            self._user_orgs[user_id].append(org_id)


class InMemoryEmbeddingRepository(EmbeddingRepository):
    """In-memory embedding repository for testing/development"""
    
    def __init__(self):
        self._requests: Dict[UUID, EmbeddingRequest] = {}
        self._results: Dict[UUID, EmbeddingResult] = {}
        self._user_requests: Dict[UUID, List[UUID]] = {}
    
    async def save_request(self, request: EmbeddingRequest) -> EmbeddingRequest:
        self._requests[request.id] = request
        
        # Update user index
        if request.user_id not in self._user_requests:
            self._user_requests[request.user_id] = []
        if request.id not in self._user_requests[request.user_id]:
            self._user_requests[request.user_id].append(request.id)
        
        return request
    
    async def save_result(self, result: EmbeddingResult) -> EmbeddingResult:
        self._results[result.id] = result
        return result
    
    async def get_request_by_id(self, request_id: UUID) -> Optional[EmbeddingRequest]:
        return self._requests.get(request_id)
    
    async def get_result_by_request_id(self, request_id: UUID) -> Optional[EmbeddingResult]:
        for result in self._results.values():
            if result.request_id == request_id:
                return result
        return None
    
    async def get_user_requests(
        self, 
        user_id: UUID, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[EmbeddingRequest]:
        request_ids = self._user_requests.get(user_id, [])
        sorted_ids = sorted(request_ids, key=lambda x: self._requests[x].created_at, reverse=True)
        page_ids = sorted_ids[offset:offset + limit]
        return [self._requests[req_id] for req_id in page_ids]


class InMemoryDiscoveryRepository(DiscoveryRepository):
    """In-memory discovery repository for testing/development"""
    
    def __init__(self):
        self._workflows: Dict[UUID, DiscoveryWorkflow] = {}
        self._workflow_id_index: Dict[str, UUID] = {}
        self._user_workflows: Dict[UUID, List[UUID]] = {}
    
    async def save_workflow(self, workflow: DiscoveryWorkflow) -> DiscoveryWorkflow:
        self._workflows[workflow.id] = workflow
        self._workflow_id_index[workflow.workflow_id] = workflow.id
        
        # Update user index
        if workflow.user_id not in self._user_workflows:
            self._user_workflows[workflow.user_id] = []
        if workflow.id not in self._user_workflows[workflow.user_id]:
            self._user_workflows[workflow.user_id].append(workflow.id)
        
        return workflow
    
    async def get_by_id(self, workflow_id: UUID) -> Optional[DiscoveryWorkflow]:
        return self._workflows.get(workflow_id)
    
    async def get_by_workflow_id(self, workflow_id: str) -> Optional[DiscoveryWorkflow]:
        internal_id = self._workflow_id_index.get(workflow_id)
        return self._workflows.get(internal_id) if internal_id else None
    
    async def update_workflow(self, workflow: DiscoveryWorkflow) -> DiscoveryWorkflow:
        if workflow.id in self._workflows:
            self._workflows[workflow.id] = workflow
        return workflow
    
    async def get_user_workflows(
        self, 
        user_id: UUID, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[DiscoveryWorkflow]:
        workflow_ids = self._user_workflows.get(user_id, [])
        sorted_ids = sorted(workflow_ids, key=lambda x: self._workflows[x].created_at, reverse=True)
        page_ids = sorted_ids[offset:offset + limit]
        return [self._workflows[wf_id] for wf_id in page_ids]


class InMemoryAuthTokenRepository(AuthTokenRepository):
    """In-memory auth token repository for testing/development"""
    
    def __init__(self):
        self._tokens: Dict[str, AuthToken] = {}
        self._user_tokens: Dict[UUID, List[str]] = {}
    
    async def save_token(self, token: AuthToken) -> AuthToken:
        self._tokens[token.token] = token
        
        # Update user index
        if token.user_id not in self._user_tokens:
            self._user_tokens[token.user_id] = []
        if token.token not in self._user_tokens[token.user_id]:
            self._user_tokens[token.user_id].append(token.token)
        
        return token
    
    async def get_by_token(self, token: str) -> Optional[AuthToken]:
        return self._tokens.get(token)
    
    async def revoke_token(self, token: str) -> bool:
        if token in self._tokens:
            self._tokens[token].revoke()
            return True
        return False
    
    async def revoke_user_tokens(self, user_id: UUID) -> int:
        tokens = self._user_tokens.get(user_id, [])
        count = 0
        for token in tokens:
            if token in self._tokens and not self._tokens[token].is_revoked:
                self._tokens[token].revoke()
                count += 1
        return count
    
    async def cleanup_expired_tokens(self) -> int:
        count = 0
        expired_tokens = []
        
        for token_str, token in self._tokens.items():
            if not token.is_valid():
                expired_tokens.append(token_str)
        
        for token_str in expired_tokens:
            del self._tokens[token_str]
            # Also remove from user index
            for user_id, tokens in self._user_tokens.items():
                if token_str in tokens:
                    tokens.remove(token_str)
            count += 1
        
        return count


class InMemoryRedisLike:
    """Simple in-memory Redis-like implementation for testing"""
    
    def __init__(self):
        self._data = {}
    
    async def get(self, key: str):
        value = self._data.get(key)
        return value.encode() if value else None
    
    async def set(self, key: str, value: str):
        self._data[key] = value
        return True
    
    async def setex(self, key: str, ttl: int, value: str):
        # For simplicity, ignore TTL in tests
        self._data[key] = value
        return True
    
    async def delete(self, key: str):
        if key in self._data:
            del self._data[key]
            return 1
        return 0
    
    async def exists(self, key: str):
        return 1 if key in self._data else 0
    
    async def incrby(self, key: str, amount: int):
        current = int(self._data.get(key, 0))
        new_value = current + amount
        self._data[key] = str(new_value)
        return new_value
    
    async def decrby(self, key: str, amount: int):
        current = int(self._data.get(key, 0))
        new_value = max(0, current - amount)
        self._data[key] = str(new_value)
        return new_value


class RedisCacheRepository(CacheRepository):
    """Redis-based cache repository"""
    
    def __init__(self, redis_url: str = "redis://redis:6379/0"):
        self.redis_url = redis_url
        self.redis = None
    
    async def initialize(self):
        """Initialize Redis connection"""
        if AIOREDIS_AVAILABLE:
            self.redis = await aioredis.from_url(self.redis_url)
        else:
            # Use in-memory fallback
            self.redis = InMemoryRedisLike()
    
    async def get(self, key: str) -> Optional[str]:
        if not self.redis:
            await self.initialize()
        result = await self.redis.get(key)
        return result.decode() if result else None
    
    async def set(self, key: str, value: str, ttl: int = None) -> bool:
        if not self.redis:
            await self.initialize()
        
        if ttl:
            return await self.redis.setex(key, ttl, value)
        else:
            return await self.redis.set(key, value)
    
    async def delete(self, key: str) -> bool:
        if not self.redis:
            await self.initialize()
        result = await self.redis.delete(key)
        return result > 0
    
    async def exists(self, key: str) -> bool:
        if not self.redis:
            await self.initialize()
        result = await self.redis.exists(key)
        return result > 0
    
    async def increment(self, key: str, amount: int = 1) -> int:
        if not self.redis:
            await self.initialize()
        return await self.redis.incrby(key, amount)
    
    async def decrement(self, key: str, amount: int = 1) -> int:
        if not self.redis:
            await self.initialize()
        return await self.redis.decrby(key, amount)