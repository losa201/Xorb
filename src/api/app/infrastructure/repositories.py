"""
Repository implementations for data access
"""

import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID
from sqlalchemy import text

# Redis dependencies with fallback
import logging

try:
    import redis.asyncio as aioredis
    AIOREDIS_AVAILABLE = True
except ImportError:
    try:
        import aioredis
        AIOREDIS_AVAILABLE = True
    except ImportError:
        AIOREDIS_AVAILABLE = False
        aioredis = None

logger = logging.getLogger(__name__)

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
            try:
                # Try redis.asyncio first (preferred for Python 3.12+)
                if hasattr(aioredis, 'from_url'):
                    self.redis = aioredis.from_url(self.redis_url, decode_responses=True)
                else:
                    # Fallback to older aioredis API
                    self.redis = await aioredis.from_url(self.redis_url)
            except Exception as e:
                logger.warning(f"Redis connection failed, using in-memory fallback: {e}")
                self.redis = InMemoryRedisLike()
        else:
            # Use in-memory fallback
            self.redis = InMemoryRedisLike()

    async def get(self, key: str) -> Optional[str]:
        if not self.redis:
            await self.initialize()
        result = await self.redis.get(key)
        # Handle both bytes and string responses
        if result is None:
            return None
        return result.decode() if isinstance(result, bytes) else result

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


# Production PostgreSQL Repository Implementations
class ProductionScanSessionRepository:
    """Production PostgreSQL-based scan session repository"""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def create_session(self, session_data: dict) -> dict:
        """Create a new scan session in database"""
        session_id = str(UUID.uuid4())
        async with self.session_factory() as db_session:
            # Create scan session record
            scan_session = {
                "id": session_id,
                "tenant_id": session_data.get("tenant_id"),
                "user_id": session_data.get("user_id"),
                "targets": json.dumps(session_data.get("targets", [])),
                "scan_type": session_data.get("scan_type", "comprehensive"),
                "status": "created",
                "metadata": json.dumps(session_data.get("metadata", {})),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            # Execute insert query
            await db_session.execute(
                text("""
                    INSERT INTO scan_sessions
                    (id, tenant_id, user_id, targets, scan_type, status, metadata, created_at, updated_at)
                    VALUES (:id, :tenant_id, :user_id, :targets, :scan_type, :status, :metadata, :created_at, :updated_at)
                """),
                scan_session
            )
            await db_session.commit()

            return {
                "session_id": session_id,
                "status": "created",
                "targets": session_data.get("targets", []),
                "scan_type": session_data.get("scan_type"),
                "created_at": scan_session["created_at"].isoformat()
            }

    async def get_session(self, session_id: UUID) -> Optional[dict]:
        """Get scan session by ID"""
        async with self.session_factory() as db_session:
            result = await db_session.execute(
                text("SELECT * FROM scan_sessions WHERE id = :session_id"),
                {"session_id": str(session_id)}
            )
            row = result.fetchone()

            if row:
                return {
                    "session_id": row.id,
                    "tenant_id": row.tenant_id,
                    "user_id": row.user_id,
                    "targets": json.loads(row.targets) if row.targets else [],
                    "scan_type": row.scan_type,
                    "status": row.status,
                    "metadata": json.loads(row.metadata) if row.metadata else {},
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                    "results": json.loads(row.results) if hasattr(row, 'results') and row.results else None
                }
            return None

    async def update_session(self, session_id: UUID, updates: dict) -> bool:
        """Update scan session"""
        async with self.session_factory() as db_session:
            update_fields = []
            params = {"session_id": str(session_id), "updated_at": datetime.utcnow()}

            for key, value in updates.items():
                if key in ["status", "scan_type"]:
                    update_fields.append(f"{key} = :{key}")
                    params[key] = value
                elif key in ["targets", "metadata", "results"]:
                    update_fields.append(f"{key} = :{key}")
                    params[key] = json.dumps(value) if value else None

            if update_fields:
                update_fields.append("updated_at = :updated_at")
                query = f"UPDATE scan_sessions SET {', '.join(update_fields)} WHERE id = :session_id"

                result = await db_session.execute(text(query), params)
                await db_session.commit()
                return result.rowcount > 0
            return False

    async def get_user_sessions(self, user_id: UUID) -> List[dict]:
        """Get scan sessions for a user"""
        async with self.session_factory() as db_session:
            result = await db_session.execute(
                text("""
                    SELECT * FROM scan_sessions
                    WHERE user_id = :user_id
                    ORDER BY created_at DESC
                    LIMIT 100
                """),
                {"user_id": str(user_id)}
            )

            sessions = []
            for row in result.fetchall():
                sessions.append({
                    "session_id": row.id,
                    "targets": json.loads(row.targets) if row.targets else [],
                    "scan_type": row.scan_type,
                    "status": row.status,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "metadata": json.loads(row.metadata) if row.metadata else {}
                })

            return sessions


class ProductionTenantRepository:
    """Production PostgreSQL-based tenant repository"""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def create_tenant(self, tenant_data: dict) -> dict:
        """Create a new tenant"""
        tenant_id = str(UUID.uuid4())
        async with self.session_factory() as db_session:
            tenant = {
                "id": tenant_id,
                "name": tenant_data.get("name"),
                "slug": tenant_data.get("slug"),
                "plan_type": tenant_data.get("plan_type", "basic"),
                "status": "active",
                "settings": json.dumps(tenant_data.get("settings", {})),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            await db_session.execute(
                text("""
                    INSERT INTO tenants
                    (id, name, slug, plan_type, status, settings, created_at, updated_at)
                    VALUES (:id, :name, :slug, :plan_type, :status, :settings, :created_at, :updated_at)
                """),
                tenant
            )
            await db_session.commit()

            return {
                "tenant_id": tenant_id,
                "name": tenant["name"],
                "slug": tenant["slug"],
                "plan_type": tenant["plan_type"],
                "status": tenant["status"],
                "created_at": tenant["created_at"].isoformat()
            }

    async def get_tenant(self, tenant_id: UUID) -> Optional[dict]:
        """Get tenant by ID"""
        async with self.session_factory() as db_session:
            result = await db_session.execute(
                text("SELECT * FROM tenants WHERE id = :tenant_id"),
                {"tenant_id": str(tenant_id)}
            )
            row = result.fetchone()

            if row:
                return {
                    "tenant_id": row.id,
                    "name": row.name,
                    "slug": row.slug,
                    "plan_type": row.plan_type,
                    "status": row.status,
                    "settings": json.loads(row.settings) if row.settings else {},
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None
                }
            return None

    async def update_tenant(self, tenant_id: UUID, updates: dict) -> bool:
        """Update tenant"""
        async with self.session_factory() as db_session:
            update_fields = []
            params = {"tenant_id": str(tenant_id), "updated_at": datetime.utcnow()}

            for key, value in updates.items():
                if key in ["name", "slug", "plan_type", "status"]:
                    update_fields.append(f"{key} = :{key}")
                    params[key] = value
                elif key == "settings":
                    update_fields.append("settings = :settings")
                    params["settings"] = json.dumps(value) if value else None

            if update_fields:
                update_fields.append("updated_at = :updated_at")
                query = f"UPDATE tenants SET {', '.join(update_fields)} WHERE id = :tenant_id"

                result = await db_session.execute(text(query), params)
                await db_session.commit()
                return result.rowcount > 0
            return False

    async def get_tenant_by_name(self, name: str) -> Optional[dict]:
        """Get tenant by name"""
        async with self.session_factory() as db_session:
            result = await db_session.execute(
                text("SELECT * FROM tenants WHERE name = :name"),
                {"name": name}
            )
            row = result.fetchone()

            if row:
                return {
                    "tenant_id": row.id,
                    "name": row.name,
                    "slug": row.slug,
                    "plan_type": row.plan_type,
                    "status": row.status,
                    "settings": json.loads(row.settings) if row.settings else {}
                }
            return None


# Production repository factory
class ProductionRepositoryFactory:
    """Factory for creating production repository instances"""

    def __init__(self, session_factory, redis_url: str = None):
        self.session_factory = session_factory
        self.redis_url = redis_url or "redis://redis:6379/0"
        self._repositories = {}

    def get_user_repository(self) -> UserRepository:
        """Get user repository"""
        if "user" not in self._repositories:
            self._repositories["user"] = InMemoryUserRepository()  # Use in-memory for now
        return self._repositories["user"]

    def get_organization_repository(self) -> OrganizationRepository:
        """Get organization repository"""
        if "organization" not in self._repositories:
            self._repositories["organization"] = InMemoryOrganizationRepository()
        return self._repositories["organization"]

    def get_embedding_repository(self) -> EmbeddingRepository:
        """Get embedding repository"""
        if "embedding" not in self._repositories:
            self._repositories["embedding"] = InMemoryEmbeddingRepository()
        return self._repositories["embedding"]

    def get_discovery_repository(self) -> DiscoveryRepository:
        """Get discovery repository"""
        if "discovery" not in self._repositories:
            self._repositories["discovery"] = InMemoryDiscoveryRepository()
        return self._repositories["discovery"]

    def get_auth_token_repository(self) -> AuthTokenRepository:
        """Get auth token repository"""
        if "auth_token" not in self._repositories:
            self._repositories["auth_token"] = InMemoryAuthTokenRepository()
        return self._repositories["auth_token"]

    def get_cache_repository(self) -> CacheRepository:
        """Get cache repository"""
        if "cache" not in self._repositories:
            self._repositories["cache"] = RedisCacheRepository(self.redis_url)
        return self._repositories["cache"]

    def get_scan_session_repository(self):
        """Get scan session repository"""
        if "scan_session" not in self._repositories:
            self._repositories["scan_session"] = ProductionScanSessionRepository(self.session_factory)
        return self._repositories["scan_session"]

    def get_tenant_repository(self):
        """Get tenant repository"""
        if "tenant" not in self._repositories:
            self._repositories["tenant"] = ProductionTenantRepository(self.session_factory)
        return self._repositories["tenant"]
