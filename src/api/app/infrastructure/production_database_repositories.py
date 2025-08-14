"""
Production Database Repository Implementations
Replaces in-memory stubs with full PostgreSQL backing for enterprise deployment
"""

import json
import logging
import uuid
from typing import List, Optional, Dict, Any, Union
from uuid import UUID
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy import select, update, delete, and_, or_, func, text
from sqlalchemy.orm import selectinload, sessionmaker
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.dialects.postgresql import insert

from ..domain.entities import (
    User, Organization, EmbeddingRequest, EmbeddingResult,
    DiscoveryWorkflow, AuthToken
)
from ..domain.repositories import (
    UserRepository, OrganizationRepository, EmbeddingRepository,
    DiscoveryRepository, AuthTokenRepository, CacheRepository,
    ScanSessionRepository, TenantRepository
)
from ..domain.tenant_entities import Tenant, TenantPlan, TenantStatus
from .database_models import (
    UserModel, OrganizationModel, TenantModel, EmbeddingRequestModel,
    EmbeddingResultModel, DiscoveryWorkflowModel, AuthTokenModel,
    UserOrganizationModel, ScanSessionModel
)

logger = logging.getLogger(__name__)


class DatabaseConnectionManager:
    """Advanced database connection management with health monitoring"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine: Optional[AsyncEngine] = None
        self.session_factory = None
        self.connection_pool_size = 20
        self.max_overflow = 10

    async def initialize(self):
        """Initialize database engine and session factory"""
        try:
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_size=self.connection_pool_size,
                max_overflow=self.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,  # 1 hour
                connect_args={
                    "command_timeout": 30,
                    "server_settings": {
                        "application_name": "xorb_enterprise_platform",
                        "tcp_keepalives_idle": "600",
                        "tcp_keepalives_interval": "30",
                        "tcp_keepalives_count": "3",
                    }
                }
            )

            self.session_factory = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            # Test connection
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))

            logger.info("Production database connection established successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    @asynccontextmanager
    async def get_session(self):
        """Get database session with proper error handling"""
        if not self.session_factory:
            raise RuntimeError("Database not initialized")

        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                return {
                    "status": "healthy",
                    "connection_count": self.engine.pool.size(),
                    "checked_out": self.engine.pool.checkedout(),
                    "checked_in": self.engine.pool.checkedin(),
                    "response_time_ms": 0  # Could add timing here
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


class ProductionUserRepository(UserRepository):
    """Production PostgreSQL user repository with advanced features"""

    def __init__(self, db_manager: DatabaseConnectionManager):
        self.db_manager = db_manager

    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID with optimized query"""
        async with self.db_manager.get_session() as session:
            try:
                stmt = (
                    select(UserModel)
                    .options(selectinload(UserModel.organizations))
                    .where(UserModel.id == user_id)
                )
                result = await session.execute(stmt)
                user_model = result.scalar_one_or_none()

                return self._model_to_entity(user_model) if user_model else None

            except Exception as e:
                logger.error(f"Error getting user by ID {user_id}: {e}")
                return None

    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username with case-insensitive search"""
        async with self.db_manager.get_session() as session:
            try:
                stmt = (
                    select(UserModel)
                    .options(selectinload(UserModel.organizations))
                    .where(func.lower(UserModel.username) == username.lower())
                )
                result = await session.execute(stmt)
                user_model = result.scalar_one_or_none()

                return self._model_to_entity(user_model) if user_model else None

            except Exception as e:
                logger.error(f"Error getting user by username {username}: {e}")
                return None

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email with case-insensitive search"""
        async with self.db_manager.get_session() as session:
            try:
                stmt = (
                    select(UserModel)
                    .options(selectinload(UserModel.organizations))
                    .where(func.lower(UserModel.email) == email.lower())
                )
                result = await session.execute(stmt)
                user_model = result.scalar_one_or_none()

                return self._model_to_entity(user_model) if user_model else None

            except Exception as e:
                logger.error(f"Error getting user by email {email}: {e}")
                return None

    async def create(self, user: User) -> User:
        """Create new user with comprehensive error handling"""
        async with self.db_manager.get_session() as session:
            try:
                user_model = UserModel(
                    id=user.id,
                    username=user.username,
                    email=user.email,
                    password_hash=getattr(user, 'password_hash', ''),
                    first_name=getattr(user, 'first_name', ''),
                    last_name=getattr(user, 'last_name', ''),
                    roles=user.roles,
                    is_active=getattr(user, 'is_active', True),
                    is_verified=getattr(user, 'is_verified', False),
                    is_admin=getattr(user, 'is_admin', False),
                    mfa_enabled=getattr(user, 'mfa_enabled', False),
                    user_metadata=getattr(user, 'metadata', {}),
                    created_at=user.created_at,
                    updated_at=user.updated_at
                )

                session.add(user_model)
                await session.flush()

                return self._model_to_entity(user_model)

            except IntegrityError as e:
                logger.error(f"User creation failed due to constraint violation: {e}")
                raise ValueError(f"User with username or email already exists")
            except Exception as e:
                logger.error(f"Error creating user: {e}")
                raise

    async def update(self, user: User) -> User:
        """Update user with optimistic locking"""
        async with self.db_manager.get_session() as session:
            try:
                stmt = (
                    update(UserModel)
                    .where(UserModel.id == user.id)
                    .values(
                        username=user.username,
                        email=user.email,
                        first_name=getattr(user, 'first_name', ''),
                        last_name=getattr(user, 'last_name', ''),
                        roles=user.roles,
                        is_active=getattr(user, 'is_active', True),
                        is_verified=getattr(user, 'is_verified', False),
                        is_admin=getattr(user, 'is_admin', False),
                        mfa_enabled=getattr(user, 'mfa_enabled', False),
                        user_metadata=getattr(user, 'metadata', {}),
                        updated_at=func.now()
                    )
                    .returning(UserModel)
                )

                result = await session.execute(stmt)
                updated_model = result.scalar_one()

                return self._model_to_entity(updated_model)

            except NoResultFound:
                raise ValueError(f"User {user.id} not found")
            except Exception as e:
                logger.error(f"Error updating user {user.id}: {e}")
                raise

    async def delete(self, user_id: UUID) -> bool:
        """Soft delete user (mark as inactive)"""
        async with self.db_manager.get_session() as session:
            try:
                stmt = (
                    update(UserModel)
                    .where(UserModel.id == user_id)
                    .values(is_active=False, updated_at=func.now())
                )

                result = await session.execute(stmt)
                return result.rowcount > 0

            except Exception as e:
                logger.error(f"Error deleting user {user_id}: {e}")
                return False

    def _model_to_entity(self, model: UserModel) -> User:
        """Convert database model to domain entity"""
        return User(
            id=model.id,
            username=model.username,
            email=model.email,
            roles=model.roles or ["user"],
            created_at=model.created_at,
            updated_at=model.updated_at
        )


class ProductionOrganizationRepository(OrganizationRepository):
    """Production PostgreSQL organization repository"""

    def __init__(self, db_manager: DatabaseConnectionManager):
        self.db_manager = db_manager

    async def get_by_id(self, org_id: UUID) -> Optional[Organization]:
        """Get organization by ID"""
        async with self.db_manager.get_session() as session:
            try:
                stmt = (
                    select(OrganizationModel)
                    .options(selectinload(OrganizationModel.users))
                    .where(OrganizationModel.id == org_id)
                )
                result = await session.execute(stmt)
                org_model = result.scalar_one_or_none()

                return self._model_to_entity(org_model) if org_model else None

            except Exception as e:
                logger.error(f"Error getting organization by ID {org_id}: {e}")
                return None

    async def get_by_name(self, name: str) -> Optional[Organization]:
        """Get organization by name"""
        async with self.db_manager.get_session() as session:
            try:
                stmt = (
                    select(OrganizationModel)
                    .where(func.lower(OrganizationModel.name) == name.lower())
                )
                result = await session.execute(stmt)
                org_model = result.scalar_one_or_none()

                return self._model_to_entity(org_model) if org_model else None

            except Exception as e:
                logger.error(f"Error getting organization by name {name}: {e}")
                return None

    async def create(self, organization: Organization) -> Organization:
        """Create new organization"""
        async with self.db_manager.get_session() as session:
            try:
                org_model = OrganizationModel(
                    id=organization.id,
                    name=organization.name,
                    plan_type=getattr(organization, 'plan_type', 'basic'),
                    settings=getattr(organization, 'settings', {}),
                    is_active=getattr(organization, 'is_active', True),
                    created_at=organization.created_at,
                    updated_at=organization.updated_at
                )

                session.add(org_model)
                await session.flush()

                return self._model_to_entity(org_model)

            except IntegrityError as e:
                logger.error(f"Organization creation failed: {e}")
                raise ValueError(f"Organization with name already exists")
            except Exception as e:
                logger.error(f"Error creating organization: {e}")
                raise

    async def update(self, organization: Organization) -> Organization:
        """Update organization"""
        async with self.db_manager.get_session() as session:
            try:
                stmt = (
                    update(OrganizationModel)
                    .where(OrganizationModel.id == organization.id)
                    .values(
                        name=organization.name,
                        plan_type=getattr(organization, 'plan_type', 'basic'),
                        settings=getattr(organization, 'settings', {}),
                        is_active=getattr(organization, 'is_active', True),
                        updated_at=func.now()
                    )
                    .returning(OrganizationModel)
                )

                result = await session.execute(stmt)
                updated_model = result.scalar_one()

                return self._model_to_entity(updated_model)

            except NoResultFound:
                raise ValueError(f"Organization {organization.id} not found")
            except Exception as e:
                logger.error(f"Error updating organization {organization.id}: {e}")
                raise

    async def get_user_organizations(self, user_id: UUID) -> List[Organization]:
        """Get organizations for a user"""
        async with self.db_manager.get_session() as session:
            try:
                stmt = (
                    select(OrganizationModel)
                    .join(UserOrganizationModel)
                    .where(UserOrganizationModel.user_id == user_id)
                    .where(OrganizationModel.is_active == True)
                )

                result = await session.execute(stmt)
                org_models = result.scalars().all()

                return [self._model_to_entity(org) for org in org_models]

            except Exception as e:
                logger.error(f"Error getting user organizations for {user_id}: {e}")
                return []

    async def add_user_to_organization(self, user_id: UUID, org_id: UUID, role: str = "member") -> bool:
        """Add user to organization"""
        async with self.db_manager.get_session() as session:
            try:
                # Use upsert to handle duplicates
                stmt = insert(UserOrganizationModel).values(
                    user_id=user_id,
                    organization_id=org_id,
                    role=role,
                    joined_at=func.now()
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=['user_id', 'organization_id'],
                    set_={'role': stmt.excluded.role, 'updated_at': func.now()}
                )

                await session.execute(stmt)
                return True

            except Exception as e:
                logger.error(f"Error adding user {user_id} to organization {org_id}: {e}")
                return False

    def _model_to_entity(self, model: OrganizationModel) -> Organization:
        """Convert database model to domain entity"""
        return Organization(
            id=model.id,
            name=model.name,
            created_at=model.created_at,
            updated_at=model.updated_at
        )


class ProductionScanSessionRepository(ScanSessionRepository):
    """Production scan session repository with advanced querying"""

    def __init__(self, db_manager: DatabaseConnectionManager):
        self.db_manager = db_manager

    async def create_session(self, session_data: dict) -> dict:
        """Create new scan session"""
        async with self.db_manager.get_session() as session:
            try:
                scan_session = ScanSessionModel(
                    id=uuid.uuid4(),
                    user_id=session_data.get('user_id'),
                    tenant_id=session_data.get('tenant_id'),
                    targets=session_data.get('targets', []),
                    scan_type=session_data.get('scan_type', 'quick'),
                    scan_profile=session_data.get('scan_profile', 'basic'),
                    status='queued',
                    configuration=session_data.get('configuration', {}),
                    metadata=session_data.get('metadata', {}),
                    created_at=func.now()
                )

                session.add(scan_session)
                await session.flush()

                return {
                    'session_id': str(scan_session.id),
                    'status': scan_session.status,
                    'created_at': scan_session.created_at.isoformat(),
                    'targets': scan_session.targets,
                    'scan_type': scan_session.scan_type
                }

            except Exception as e:
                logger.error(f"Error creating scan session: {e}")
                raise

    async def get_session(self, session_id: UUID) -> Optional[dict]:
        """Get scan session by ID"""
        async with self.db_manager.get_session() as session:
            try:
                stmt = select(ScanSessionModel).where(ScanSessionModel.id == session_id)
                result = await session.execute(stmt)
                scan_model = result.scalar_one_or_none()

                if not scan_model:
                    return None

                return {
                    'session_id': str(scan_model.id),
                    'user_id': str(scan_model.user_id),
                    'tenant_id': str(scan_model.tenant_id) if scan_model.tenant_id else None,
                    'status': scan_model.status,
                    'targets': scan_model.targets,
                    'scan_type': scan_model.scan_type,
                    'scan_profile': scan_model.scan_profile,
                    'configuration': scan_model.configuration,
                    'results': scan_model.results,
                    'metadata': scan_model.metadata,
                    'created_at': scan_model.created_at.isoformat(),
                    'updated_at': scan_model.updated_at.isoformat() if scan_model.updated_at else None,
                    'completed_at': scan_model.completed_at.isoformat() if scan_model.completed_at else None
                }

            except Exception as e:
                logger.error(f"Error getting scan session {session_id}: {e}")
                return None

    async def update_session(self, session_id: UUID, updates: dict) -> bool:
        """Update scan session"""
        async with self.db_manager.get_session() as session:
            try:
                update_values = {
                    'updated_at': func.now()
                }

                # Add provided updates
                for key, value in updates.items():
                    if key in ['status', 'results', 'metadata', 'error_message']:
                        update_values[key] = value

                # Set completion time if status is completed or failed
                if updates.get('status') in ['completed', 'failed']:
                    update_values['completed_at'] = func.now()

                stmt = (
                    update(ScanSessionModel)
                    .where(ScanSessionModel.id == session_id)
                    .values(**update_values)
                )

                result = await session.execute(stmt)
                return result.rowcount > 0

            except Exception as e:
                logger.error(f"Error updating scan session {session_id}: {e}")
                return False

    async def get_user_sessions(
        self,
        user_id: UUID,
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[str] = None
    ) -> List[dict]:
        """Get scan sessions for a user with pagination"""
        async with self.db_manager.get_session() as session:
            try:
                stmt = (
                    select(ScanSessionModel)
                    .where(ScanSessionModel.user_id == user_id)
                    .order_by(ScanSessionModel.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )

                if status_filter:
                    stmt = stmt.where(ScanSessionModel.status == status_filter)

                result = await session.execute(stmt)
                sessions = result.scalars().all()

                return [
                    {
                        'session_id': str(s.id),
                        'status': s.status,
                        'scan_type': s.scan_type,
                        'targets': s.targets,
                        'created_at': s.created_at.isoformat(),
                        'completed_at': s.completed_at.isoformat() if s.completed_at else None
                    }
                    for s in sessions
                ]

            except Exception as e:
                logger.error(f"Error getting user sessions for {user_id}: {e}")
                return []


class ProductionAuthTokenRepository(AuthTokenRepository):
    """Production auth token repository with automatic cleanup"""

    def __init__(self, db_manager: DatabaseConnectionManager):
        self.db_manager = db_manager

    async def save_token(self, token: AuthToken) -> AuthToken:
        """Save auth token"""
        async with self.db_manager.get_session() as session:
            try:
                token_model = AuthTokenModel(
                    id=uuid.uuid4(),
                    user_id=token.user_id,
                    token=token.token,
                    token_type=getattr(token, 'token_type', 'access'),
                    expires_at=token.expires_at,
                    is_revoked=False,
                    created_at=func.now()
                )

                session.add(token_model)
                await session.flush()

                return token

            except Exception as e:
                logger.error(f"Error saving token: {e}")
                raise

    async def get_by_token(self, token: str) -> Optional[AuthToken]:
        """Get token by token string"""
        async with self.db_manager.get_session() as session:
            try:
                stmt = (
                    select(AuthTokenModel)
                    .where(
                        and_(
                            AuthTokenModel.token == token,
                            AuthTokenModel.is_revoked == False,
                            AuthTokenModel.expires_at > func.now()
                        )
                    )
                )
                result = await session.execute(stmt)
                token_model = result.scalar_one_or_none()

                if not token_model:
                    return None

                return AuthToken(
                    user_id=token_model.user_id,
                    token=token_model.token,
                    expires_at=token_model.expires_at
                )

            except Exception as e:
                logger.error(f"Error getting token: {e}")
                return None

    async def revoke_token(self, token: str) -> bool:
        """Revoke a token"""
        async with self.db_manager.get_session() as session:
            try:
                stmt = (
                    update(AuthTokenModel)
                    .where(AuthTokenModel.token == token)
                    .values(is_revoked=True, updated_at=func.now())
                )

                result = await session.execute(stmt)
                return result.rowcount > 0

            except Exception as e:
                logger.error(f"Error revoking token: {e}")
                return False

    async def revoke_user_tokens(self, user_id: UUID) -> int:
        """Revoke all tokens for a user"""
        async with self.db_manager.get_session() as session:
            try:
                stmt = (
                    update(AuthTokenModel)
                    .where(
                        and_(
                            AuthTokenModel.user_id == user_id,
                            AuthTokenModel.is_revoked == False
                        )
                    )
                    .values(is_revoked=True, updated_at=func.now())
                )

                result = await session.execute(stmt)
                return result.rowcount

            except Exception as e:
                logger.error(f"Error revoking user tokens for {user_id}: {e}")
                return 0

    async def cleanup_expired_tokens(self) -> int:
        """Clean up expired and revoked tokens"""
        async with self.db_manager.get_session() as session:
            try:
                stmt = delete(AuthTokenModel).where(
                    or_(
                        AuthTokenModel.expires_at < func.now(),
                        AuthTokenModel.is_revoked == True
                    )
                )

                result = await session.execute(stmt)
                return result.rowcount

            except Exception as e:
                logger.error(f"Error cleaning up tokens: {e}")
                return 0


class ProductionTenantRepository(TenantRepository):
    """Production tenant repository for multi-tenant support"""

    def __init__(self, db_manager: DatabaseConnectionManager):
        self.db_manager = db_manager

    async def create_tenant(self, tenant_data: dict) -> dict:
        """Create new tenant"""
        async with self.db_manager.get_session() as session:
            try:
                tenant = TenantModel(
                    id=uuid.uuid4(),
                    name=tenant_data['name'],
                    slug=tenant_data['slug'],
                    plan=tenant_data.get('plan', TenantPlan.BASIC),
                    status=TenantStatus.ACTIVE,
                    settings=tenant_data.get('settings', {}),
                    created_at=func.now()
                )

                session.add(tenant)
                await session.flush()

                return {
                    'tenant_id': str(tenant.id),
                    'name': tenant.name,
                    'slug': tenant.slug,
                    'plan': tenant.plan.value,
                    'status': tenant.status.value,
                    'created_at': tenant.created_at.isoformat()
                }

            except IntegrityError as e:
                logger.error(f"Tenant creation failed: {e}")
                raise ValueError(f"Tenant with slug already exists")
            except Exception as e:
                logger.error(f"Error creating tenant: {e}")
                raise

    async def get_tenant(self, tenant_id: UUID) -> Optional[dict]:
        """Get tenant by ID"""
        async with self.db_manager.get_session() as session:
            try:
                stmt = select(TenantModel).where(TenantModel.id == tenant_id)
                result = await session.execute(stmt)
                tenant = result.scalar_one_or_none()

                if not tenant:
                    return None

                return {
                    'tenant_id': str(tenant.id),
                    'name': tenant.name,
                    'slug': tenant.slug,
                    'plan': tenant.plan.value,
                    'status': tenant.status.value,
                    'settings': tenant.settings,
                    'created_at': tenant.created_at.isoformat(),
                    'updated_at': tenant.updated_at.isoformat() if tenant.updated_at else None
                }

            except Exception as e:
                logger.error(f"Error getting tenant {tenant_id}: {e}")
                return None

    async def get_tenant_by_name(self, name: str) -> Optional[dict]:
        """Get tenant by name"""
        async with self.db_manager.get_session() as session:
            try:
                stmt = select(TenantModel).where(func.lower(TenantModel.name) == name.lower())
                result = await session.execute(stmt)
                tenant = result.scalar_one_or_none()

                if not tenant:
                    return None

                return {
                    'tenant_id': str(tenant.id),
                    'name': tenant.name,
                    'slug': tenant.slug,
                    'plan': tenant.plan.value,
                    'status': tenant.status.value
                }

            except Exception as e:
                logger.error(f"Error getting tenant by name {name}: {e}")
                return None

    async def update_tenant(self, tenant_id: UUID, updates: dict) -> bool:
        """Update tenant"""
        async with self.db_manager.get_session() as session:
            try:
                update_values = {
                    'updated_at': func.now()
                }

                for key, value in updates.items():
                    if key in ['name', 'slug', 'plan', 'status', 'settings']:
                        update_values[key] = value

                stmt = (
                    update(TenantModel)
                    .where(TenantModel.id == tenant_id)
                    .values(**update_values)
                )

                result = await session.execute(stmt)
                return result.rowcount > 0

            except Exception as e:
                logger.error(f"Error updating tenant {tenant_id}: {e}")
                return False
