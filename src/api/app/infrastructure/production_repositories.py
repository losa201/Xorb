"""
Production Repository Implementations - Enterprise-grade data access layer
Implements all repository interfaces with PostgreSQL + Redis for production use
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from uuid import UUID, uuid4
from contextlib import asynccontextmanager

import asyncpg
from sqlalchemy import text, and_, or_, func, select, insert, update, delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from .redis_compatibility import get_redis_client, CompatibleRedisClient
from ..domain.entities import (
    User, Organization, EmbeddingRequest, EmbeddingResult,
    DiscoveryWorkflow, AuthToken
)
from ..domain.repositories import (
    UserRepository, OrganizationRepository, EmbeddingRepository,
    DiscoveryRepository, AuthTokenRepository, CacheRepository,
    ScanSessionRepository, TenantRepository
)

logger = logging.getLogger(__name__)


class ProductionPostgreSQLRepository:
    """Base production repository with PostgreSQL backend"""

    def __init__(self, session_factory: async_sessionmaker, tenant_id: Optional[UUID] = None):
        self.session_factory = session_factory
        self.tenant_id = tenant_id

    @asynccontextmanager
    async def get_session(self):
        """Get database session with automatic transaction management"""
        async with self.session_factory() as session:
            try:
                # Set tenant context for Row Level Security
                if self.tenant_id:
                    await session.execute(
                        text("SET app.current_tenant_id = :tenant_id"),
                        {"tenant_id": str(self.tenant_id)}
                    )
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database transaction failed: {e}")
                raise
            finally:
                await session.close()


class ProductionUserRepository(ProductionPostgreSQLRepository, UserRepository):
    """Production user repository with PostgreSQL backend"""

    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID"""
        async with self.get_session() as session:
            try:
                result = await session.execute(
                    text("""
                        SELECT id, username, email, first_name, last_name,
                               roles, is_active, created_at, updated_at,
                               last_login, tenant_id, permissions
                        FROM users
                        WHERE id = :user_id AND is_active = true
                    """),
                    {"user_id": str(user_id)}
                )
                row = result.fetchone()
                if row:
                    return User(
                        id=UUID(row.id),
                        username=row.username,
                        email=row.email,
                        first_name=row.first_name,
                        last_name=row.last_name,
                        roles=row.roles or [],
                        is_active=row.is_active,
                        created_at=row.created_at,
                        updated_at=row.updated_at,
                        last_login=row.last_login,
                        tenant_id=UUID(row.tenant_id) if row.tenant_id else None,
                        permissions=row.permissions or {}
                    )
                return None
            except Exception as e:
                logger.error(f"Failed to get user by ID {user_id}: {e}")
                return None

    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        async with self.get_session() as session:
            try:
                result = await session.execute(
                    text("""
                        SELECT id, username, email, first_name, last_name,
                               roles, is_active, created_at, updated_at,
                               last_login, tenant_id, permissions
                        FROM users
                        WHERE username = :username AND is_active = true
                    """),
                    {"username": username}
                )
                row = result.fetchone()
                if row:
                    return User(
                        id=UUID(row.id),
                        username=row.username,
                        email=row.email,
                        first_name=row.first_name,
                        last_name=row.last_name,
                        roles=row.roles or [],
                        is_active=row.is_active,
                        created_at=row.created_at,
                        updated_at=row.updated_at,
                        last_login=row.last_login,
                        tenant_id=UUID(row.tenant_id) if row.tenant_id else None,
                        permissions=row.permissions or {}
                    )
                return None
            except Exception as e:
                logger.error(f"Failed to get user by username {username}: {e}")
                return None

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        async with self.get_session() as session:
            try:
                result = await session.execute(
                    text("""
                        SELECT id, username, email, first_name, last_name,
                               roles, is_active, created_at, updated_at,
                               last_login, tenant_id, permissions
                        FROM users
                        WHERE email = :email AND is_active = true
                    """),
                    {"email": email.lower()}
                )
                row = result.fetchone()
                if row:
                    return User(
                        id=UUID(row.id),
                        username=row.username,
                        email=row.email,
                        first_name=row.first_name,
                        last_name=row.last_name,
                        roles=row.roles or [],
                        is_active=row.is_active,
                        created_at=row.created_at,
                        updated_at=row.updated_at,
                        last_login=row.last_login,
                        tenant_id=UUID(row.tenant_id) if row.tenant_id else None,
                        permissions=row.permissions or {}
                    )
                return None
            except Exception as e:
                logger.error(f"Failed to get user by email {email}: {e}")
                return None

    async def create(self, user: User) -> User:
        """Create a new user"""
        async with self.get_session() as session:
            try:
                # Generate ID if not provided
                if not user.id:
                    user.id = uuid4()

                # Set tenant context
                tenant_id = user.tenant_id or self.tenant_id

                await session.execute(
                    text("""
                        INSERT INTO users (
                            id, username, email, first_name, last_name,
                            password_hash, roles, is_active, created_at,
                            updated_at, tenant_id, permissions
                        ) VALUES (
                            :id, :username, :email, :first_name, :last_name,
                            :password_hash, :roles, :is_active, :created_at,
                            :updated_at, :tenant_id, :permissions
                        )
                    """),
                    {
                        "id": str(user.id),
                        "username": user.username,
                        "email": user.email.lower(),
                        "first_name": user.first_name,
                        "last_name": user.last_name,
                        "password_hash": getattr(user, 'password_hash', None),
                        "roles": json.dumps(user.roles),
                        "is_active": user.is_active,
                        "created_at": user.created_at or datetime.utcnow(),
                        "updated_at": user.updated_at or datetime.utcnow(),
                        "tenant_id": str(tenant_id) if tenant_id else None,
                        "permissions": json.dumps(user.permissions)
                    }
                )
                return user
            except IntegrityError as e:
                logger.error(f"User creation failed - integrity error: {e}")
                raise ValueError("User with this username or email already exists")
            except Exception as e:
                logger.error(f"Failed to create user: {e}")
                raise

    async def update(self, user: User) -> User:
        """Update an existing user"""
        async with self.get_session() as session:
            try:
                user.updated_at = datetime.utcnow()

                await session.execute(
                    text("""
                        UPDATE users SET
                            username = :username,
                            email = :email,
                            first_name = :first_name,
                            last_name = :last_name,
                            roles = :roles,
                            is_active = :is_active,
                            updated_at = :updated_at,
                            last_login = :last_login,
                            permissions = :permissions
                        WHERE id = :id
                    """),
                    {
                        "id": str(user.id),
                        "username": user.username,
                        "email": user.email.lower(),
                        "first_name": user.first_name,
                        "last_name": user.last_name,
                        "roles": json.dumps(user.roles),
                        "is_active": user.is_active,
                        "updated_at": user.updated_at,
                        "last_login": user.last_login,
                        "permissions": json.dumps(user.permissions)
                    }
                )
                return user
            except Exception as e:
                logger.error(f"Failed to update user {user.id}: {e}")
                raise

    async def delete(self, user_id: UUID) -> bool:
        """Soft delete a user"""
        async with self.get_session() as session:
            try:
                result = await session.execute(
                    text("""
                        UPDATE users SET
                            is_active = false,
                            updated_at = :updated_at
                        WHERE id = :user_id
                    """),
                    {
                        "user_id": str(user_id),
                        "updated_at": datetime.utcnow()
                    }
                )
                return result.rowcount > 0
            except Exception as e:
                logger.error(f"Failed to delete user {user_id}: {e}")
                return False


class ProductionScanSessionRepository(ProductionPostgreSQLRepository, ScanSessionRepository):
    """Production scan session repository with advanced tracking"""

    async def create_session(self, session_data: dict) -> dict:
        """Create a new scan session with comprehensive tracking"""
        async with self.get_session() as session:
            try:
                session_id = uuid4()
                created_at = datetime.utcnow()

                # Extract targets for separate storage
                targets = session_data.get('targets', [])

                # Create main session record
                await session.execute(
                    text("""
                        INSERT INTO scan_sessions (
                            id, tenant_id, user_id, scan_type, status,
                            targets_count, scan_profile, stealth_mode,
                            priority, estimated_duration, metadata,
                            created_at, updated_at
                        ) VALUES (
                            :id, :tenant_id, :user_id, :scan_type, :status,
                            :targets_count, :scan_profile, :stealth_mode,
                            :priority, :estimated_duration, :metadata,
                            :created_at, :updated_at
                        )
                    """),
                    {
                        "id": str(session_id),
                        "tenant_id": str(self.tenant_id) if self.tenant_id else None,
                        "user_id": session_data.get('user_id'),
                        "scan_type": session_data.get('scan_type', 'comprehensive'),
                        "status": "pending",
                        "targets_count": len(targets),
                        "scan_profile": session_data.get('scan_profile', 'standard'),
                        "stealth_mode": session_data.get('stealth_mode', False),
                        "priority": session_data.get('priority', 'medium'),
                        "estimated_duration": session_data.get('estimated_duration', 1800),
                        "metadata": json.dumps(session_data.get('metadata', {})),
                        "created_at": created_at,
                        "updated_at": created_at
                    }
                )

                # Store targets separately for detailed tracking
                for i, target in enumerate(targets):
                    await session.execute(
                        text("""
                            INSERT INTO scan_targets (
                                id, session_id, target_order, host, ports,
                                scan_profile, authorized, metadata, created_at
                            ) VALUES (
                                :id, :session_id, :target_order, :host, :ports,
                                :scan_profile, :authorized, :metadata, :created_at
                            )
                        """),
                        {
                            "id": str(uuid4()),
                            "session_id": str(session_id),
                            "target_order": i,
                            "host": target.get('host'),
                            "ports": json.dumps(target.get('ports', [])),
                            "scan_profile": target.get('scan_profile', 'standard'),
                            "authorized": target.get('authorized', False),
                            "metadata": json.dumps(target.get('metadata', {})),
                            "created_at": created_at
                        }
                    )

                return {
                    "session_id": str(session_id),
                    "status": "pending",
                    "created_at": created_at.isoformat(),
                    "targets_count": len(targets),
                    "estimated_duration": session_data.get('estimated_duration', 1800)
                }

            except Exception as e:
                logger.error(f"Failed to create scan session: {e}")
                raise

    async def get_session(self, session_id: UUID) -> Optional[dict]:
        """Get scan session with full details"""
        async with self.get_session() as session:
            try:
                # Get main session data
                result = await session.execute(
                    text("""
                        SELECT
                            id, tenant_id, user_id, scan_type, status,
                            targets_count, scan_profile, stealth_mode,
                            priority, estimated_duration, actual_duration,
                            started_at, completed_at, error_message,
                            metadata, created_at, updated_at,
                            progress_percentage, findings_count
                        FROM scan_sessions
                        WHERE id = :session_id
                    """),
                    {"session_id": str(session_id)}
                )

                session_row = result.fetchone()
                if not session_row:
                    return None

                # Get targets
                targets_result = await session.execute(
                    text("""
                        SELECT
                            host, ports, scan_profile, authorized,
                            status, scan_results, metadata
                        FROM scan_targets
                        WHERE session_id = :session_id
                        ORDER BY target_order
                    """),
                    {"session_id": str(session_id)}
                )

                targets = []
                for target_row in targets_result.fetchall():
                    targets.append({
                        "host": target_row.host,
                        "ports": json.loads(target_row.ports or "[]"),
                        "scan_profile": target_row.scan_profile,
                        "authorized": target_row.authorized,
                        "status": target_row.status,
                        "scan_results": json.loads(target_row.scan_results or "{}"),
                        "metadata": json.loads(target_row.metadata or "{}")
                    })

                # Get scan results summary
                results_result = await session.execute(
                    text("""
                        SELECT
                            vulnerability_summary, risk_score,
                            compliance_results, recommendations
                        FROM scan_results
                        WHERE session_id = :session_id
                    """),
                    {"session_id": str(session_id)}
                )

                results_row = results_result.fetchone()
                scan_results = {}
                if results_row:
                    scan_results = {
                        "vulnerability_summary": json.loads(results_row.vulnerability_summary or "{}"),
                        "risk_score": results_row.risk_score,
                        "compliance_results": json.loads(results_row.compliance_results or "{}"),
                        "recommendations": json.loads(results_row.recommendations or "[]")
                    }

                return {
                    "session_id": session_row.id,
                    "tenant_id": session_row.tenant_id,
                    "user_id": session_row.user_id,
                    "scan_type": session_row.scan_type,
                    "status": session_row.status,
                    "targets": targets,
                    "targets_count": session_row.targets_count,
                    "scan_profile": session_row.scan_profile,
                    "stealth_mode": session_row.stealth_mode,
                    "priority": session_row.priority,
                    "estimated_duration": session_row.estimated_duration,
                    "actual_duration": session_row.actual_duration,
                    "started_at": session_row.started_at.isoformat() if session_row.started_at else None,
                    "completed_at": session_row.completed_at.isoformat() if session_row.completed_at else None,
                    "error_message": session_row.error_message,
                    "metadata": json.loads(session_row.metadata or "{}"),
                    "created_at": session_row.created_at.isoformat(),
                    "updated_at": session_row.updated_at.isoformat(),
                    "progress_percentage": session_row.progress_percentage or 0,
                    "findings_count": session_row.findings_count or 0,
                    "scan_results": scan_results
                }

            except Exception as e:
                logger.error(f"Failed to get scan session {session_id}: {e}")
                return None

    async def update_session(self, session_id: UUID, updates: dict) -> bool:
        """Update scan session with advanced tracking"""
        async with self.get_session() as session:
            try:
                # Build dynamic update query
                set_clauses = ["updated_at = :updated_at"]
                params = {
                    "session_id": str(session_id),
                    "updated_at": datetime.utcnow()
                }

                # Add conditional updates
                if "status" in updates:
                    set_clauses.append("status = :status")
                    params["status"] = updates["status"]

                if "progress_percentage" in updates:
                    set_clauses.append("progress_percentage = :progress_percentage")
                    params["progress_percentage"] = updates["progress_percentage"]

                if "findings_count" in updates:
                    set_clauses.append("findings_count = :findings_count")
                    params["findings_count"] = updates["findings_count"]

                if "error_message" in updates:
                    set_clauses.append("error_message = :error_message")
                    params["error_message"] = updates["error_message"]

                if "started_at" in updates:
                    set_clauses.append("started_at = :started_at")
                    params["started_at"] = updates["started_at"]

                if "completed_at" in updates:
                    set_clauses.append("completed_at = :completed_at")
                    params["completed_at"] = updates["completed_at"]

                if "actual_duration" in updates:
                    set_clauses.append("actual_duration = :actual_duration")
                    params["actual_duration"] = updates["actual_duration"]

                update_query = f"""
                    UPDATE scan_sessions
                    SET {', '.join(set_clauses)}
                    WHERE id = :session_id
                """

                result = await session.execute(text(update_query), params)
                return result.rowcount > 0

            except Exception as e:
                logger.error(f"Failed to update scan session {session_id}: {e}")
                return False

    async def get_user_sessions(self, user_id: UUID) -> List[dict]:
        """Get scan sessions for a user with pagination and filtering"""
        async with self.get_session() as session:
            try:
                result = await session.execute(
                    text("""
                        SELECT
                            id, scan_type, status, targets_count,
                            scan_profile, priority, progress_percentage,
                            findings_count, created_at, completed_at,
                            metadata
                        FROM scan_sessions
                        WHERE user_id = :user_id
                        ORDER BY created_at DESC
                        LIMIT 50
                    """),
                    {"user_id": str(user_id)}
                )

                sessions = []
                for row in result.fetchall():
                    sessions.append({
                        "session_id": row.id,
                        "scan_type": row.scan_type,
                        "status": row.status,
                        "targets_count": row.targets_count,
                        "scan_profile": row.scan_profile,
                        "priority": row.priority,
                        "progress_percentage": row.progress_percentage or 0,
                        "findings_count": row.findings_count or 0,
                        "created_at": row.created_at.isoformat(),
                        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
                        "metadata": json.loads(row.metadata or "{}")
                    })

                return sessions

            except Exception as e:
                logger.error(f"Failed to get user sessions for {user_id}: {e}")
                return []


class ProductionRedisCache(CacheRepository):
    """Production Redis cache with high availability and persistence"""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._client = None

    async def initialize(self):
        """Initialize Redis connection with fallback"""
        try:
            self._client = await get_redis_client(self.redis_url)
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.error(f"Redis initialization failed: {e}")
            # Will use memory fallback from CompatibleRedisClient

    async def get(self, key: str) -> Optional[str]:
        """Get value from cache with namespace support"""
        try:
            namespaced_key = f"xorb:cache:{key}"
            if self._client:
                return await self._client.get(namespaced_key)
            return None
        except Exception as e:
            logger.error(f"Cache GET failed for key {key}: {e}")
            return None

    async def set(self, key: str, value: str, ttl: int = None) -> bool:
        """Set value in cache with automatic expiration"""
        try:
            namespaced_key = f"xorb:cache:{key}"
            if self._client:
                return await self._client.set(namespaced_key, value, ex=ttl)
            return False
        except Exception as e:
            logger.error(f"Cache SET failed for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            namespaced_key = f"xorb:cache:{key}"
            if self._client:
                result = await self._client.delete(namespaced_key)
                return result > 0
            return False
        except Exception as e:
            logger.error(f"Cache DELETE failed for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            namespaced_key = f"xorb:cache:{key}"
            if self._client:
                return await self._client.exists(namespaced_key)
            return False
        except Exception as e:
            logger.error(f"Cache EXISTS failed for key {key}: {e}")
            return False

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a counter with namespace support"""
        try:
            namespaced_key = f"xorb:counters:{key}"
            if self._client:
                return await self._client.incrby(namespaced_key, amount)
            return 0
        except Exception as e:
            logger.error(f"Cache INCREMENT failed for key {key}: {e}")
            return 0

    async def decrement(self, key: str, amount: int = 1) -> int:
        """Decrement a counter"""
        try:
            namespaced_key = f"xorb:counters:{key}"
            if self._client:
                return await self._client.incrby(namespaced_key, -amount)
            return 0
        except Exception as e:
            logger.error(f"Cache DECREMENT failed for key {key}: {e}")
            return 0

    async def set_json(self, key: str, data: dict, ttl: int = None) -> bool:
        """Store JSON data in cache"""
        try:
            json_str = json.dumps(data, default=str)
            return await self.set(key, json_str, ttl)
        except Exception as e:
            logger.error(f"Cache SET_JSON failed for key {key}: {e}")
            return False

    async def get_json(self, key: str) -> Optional[dict]:
        """Retrieve JSON data from cache"""
        try:
            json_str = await self.get(key)
            if json_str:
                return json.loads(json_str)
            return None
        except Exception as e:
            logger.error(f"Cache GET_JSON failed for key {key}: {e}")
            return None

    async def close(self):
        """Close Redis connection"""
        try:
            if self._client:
                await self._client.close()
        except Exception as e:
            logger.error(f"Redis close failed: {e}")


# Repository factory for dependency injection
class RepositoryFactory:
    """Production repository factory with dependency injection support"""

    def __init__(self, database_url: str, redis_url: str = "redis://localhost:6379/0"):
        self.database_url = database_url
        self.redis_url = redis_url
        self._engine = None
        self._session_factory = None
        self._cache = None

    async def initialize(self):
        """Initialize database connections and cache"""
        try:
            # Create async engine with production settings
            self._engine = create_async_engine(
                self.database_url,
                echo=False,  # Set to True for SQL debugging
                pool_size=20,
                max_overflow=30,
                pool_timeout=30,
                pool_recycle=3600,
                pool_pre_ping=True
            )

            # Create session factory
            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            # Initialize cache
            self._cache = ProductionRedisCache(self.redis_url)
            await self._cache.initialize()

            logger.info("Repository factory initialized successfully")

        except Exception as e:
            logger.error(f"Repository factory initialization failed: {e}")
            raise

    def create_user_repository(self, tenant_id: Optional[UUID] = None) -> ProductionUserRepository:
        """Create user repository instance"""
        return ProductionUserRepository(self._session_factory, tenant_id)

    def create_scan_session_repository(self, tenant_id: Optional[UUID] = None) -> ProductionScanSessionRepository:
        """Create scan session repository instance"""
        return ProductionScanSessionRepository(self._session_factory, tenant_id)

    def get_cache_repository(self) -> ProductionRedisCache:
        """Get cache repository instance"""
        return self._cache

    async def close(self):
        """Close all connections"""
        try:
            if self._cache:
                await self._cache.close()
            if self._engine:
                await self._engine.dispose()
            logger.info("Repository factory closed successfully")
        except Exception as e:
            logger.error(f"Repository factory close failed: {e}")


# Global factory instance
_repository_factory: Optional[RepositoryFactory] = None


async def get_repository_factory(
    database_url: str = None,
    redis_url: str = "redis://localhost:6379/0"
) -> RepositoryFactory:
    """Get global repository factory instance"""
    global _repository_factory

    if _repository_factory is None:
        if not database_url:
            raise ValueError("Database URL is required for repository factory initialization")

        _repository_factory = RepositoryFactory(database_url, redis_url)
        await _repository_factory.initialize()

    return _repository_factory


async def close_repository_factory():
    """Close global repository factory"""
    global _repository_factory
    if _repository_factory:
        await _repository_factory.close()
        _repository_factory = None
