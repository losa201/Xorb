"""Tenant management service."""
import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from ..domain.tenant_entities import (
    Tenant, TenantUser, TenantCreate, TenantUpdate,
    TenantUserCreate, TenantUserUpdate, TenantStatus
)
from ..infrastructure.database import get_async_session
from ..auth.models import Role, UserClaims


logger = logging.getLogger(__name__)


class TenantService:
    """Service for tenant management operations."""

    def __init__(self):
        self.session_factory = get_async_session

    async def create_tenant(self, tenant_data: TenantCreate) -> Tenant:
        """Create a new tenant."""
        async with self.session_factory() as session:
            try:
                tenant = Tenant(
                    name=tenant_data.name,
                    slug=tenant_data.slug,
                    plan=tenant_data.plan.value,
                    contact_email=tenant_data.contact_email,
                    contact_name=tenant_data.contact_name,
                    max_users=tenant_data.max_users,
                    max_storage_gb=tenant_data.max_storage_gb,
                    settings=tenant_data.settings
                )

                session.add(tenant)
                await session.commit()
                await session.refresh(tenant)

                logger.info(f"Created tenant: {tenant.name} ({tenant.id})")
                return tenant

            except IntegrityError as e:
                await session.rollback()
                logger.error(f"Failed to create tenant: {e}")
                raise ValueError(f"Tenant with slug '{tenant_data.slug}' already exists")

    async def get_tenant(self, tenant_id: UUID) -> Optional[Tenant]:
        """Get tenant by ID."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(Tenant).where(Tenant.id == tenant_id)
            )
            return result.scalar_one_or_none()

    async def get_tenant_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get tenant by slug."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(Tenant).where(Tenant.slug == slug)
            )
            return result.scalar_one_or_none()

    async def list_tenants(
        self,
        status: Optional[TenantStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Tenant]:
        """List tenants with optional filtering."""
        async with self.session_factory() as session:
            query = select(Tenant)

            if status:
                query = query.where(Tenant.status == status.value)

            query = query.limit(limit).offset(offset)
            result = await session.execute(query)
            return result.scalars().all()

    async def update_tenant(
        self,
        tenant_id: UUID,
        tenant_data: TenantUpdate
    ) -> Optional[Tenant]:
        """Update tenant."""
        async with self.session_factory() as session:
            try:
                # Build update data
                update_data = {}
                for field, value in tenant_data.model_dump(exclude_unset=True).items():
                    if value is not None:
                        if field == 'status' and isinstance(value, TenantStatus):
                            update_data[field] = value.value
                        else:
                            update_data[field] = value

                if not update_data:
                    return await self.get_tenant(tenant_id)

                result = await session.execute(
                    update(Tenant)
                    .where(Tenant.id == tenant_id)
                    .values(**update_data)
                    .returning(Tenant)
                )

                tenant = result.scalar_one_or_none()
                if tenant:
                    await session.commit()
                    logger.info(f"Updated tenant: {tenant_id}")
                else:
                    await session.rollback()

                return tenant

            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to update tenant {tenant_id}: {e}")
                raise

    async def delete_tenant(self, tenant_id: UUID) -> bool:
        """Soft delete tenant (mark as archived)."""
        async with self.session_factory() as session:
            try:
                result = await session.execute(
                    update(Tenant)
                    .where(Tenant.id == tenant_id)
                    .values(status=TenantStatus.ARCHIVED.value)
                    .returning(Tenant.id)
                )

                deleted_id = result.scalar_one_or_none()
                if deleted_id:
                    await session.commit()
                    logger.info(f"Archived tenant: {tenant_id}")
                    return True
                else:
                    await session.rollback()
                    return False

            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to archive tenant {tenant_id}: {e}")
                raise

    async def add_user_to_tenant(
        self,
        tenant_id: UUID,
        user_data: TenantUserCreate
    ) -> TenantUser:
        """Add user to tenant."""
        async with self.session_factory() as session:
            try:
                tenant_user = TenantUser(
                    tenant_id=tenant_id,
                    user_id=user_data.user_id,
                    email=user_data.email,
                    roles=user_data.roles,
                    is_active=True
                )

                session.add(tenant_user)
                await session.commit()
                await session.refresh(tenant_user)

                logger.info(f"Added user {user_data.user_id} to tenant {tenant_id}")
                return tenant_user

            except IntegrityError as e:
                await session.rollback()
                logger.error(f"Failed to add user to tenant: {e}")
                raise ValueError("User already exists in this tenant")

    async def get_tenant_user(
        self,
        tenant_id: UUID,
        user_id: str
    ) -> Optional[TenantUser]:
        """Get tenant user relationship."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(TenantUser).where(
                    TenantUser.tenant_id == tenant_id,
                    TenantUser.user_id == user_id
                )
            )
            return result.scalar_one_or_none()

    async def list_tenant_users(
        self,
        tenant_id: UUID,
        active_only: bool = True
    ) -> List[TenantUser]:
        """List users for a tenant."""
        async with self.session_factory() as session:
            # Set tenant context for RLS
            await session.execute(
                "SELECT set_config('app.tenant_id', :tenant_id, false)",
                {"tenant_id": str(tenant_id)}
            )

            query = select(TenantUser).where(TenantUser.tenant_id == tenant_id)

            if active_only:
                query = query.where(TenantUser.is_active == True)

            result = await session.execute(query)
            return result.scalars().all()

    async def update_tenant_user(
        self,
        tenant_id: UUID,
        user_id: str,
        user_data: TenantUserUpdate
    ) -> Optional[TenantUser]:
        """Update tenant user."""
        async with self.session_factory() as session:
            try:
                # Set tenant context for RLS
                await session.execute(
                    "SELECT set_config('app.tenant_id', :tenant_id, false)",
                    {"tenant_id": str(tenant_id)}
                )

                update_data = user_data.model_dump(exclude_unset=True)
                if not update_data:
                    return await self.get_tenant_user(tenant_id, user_id)

                result = await session.execute(
                    update(TenantUser)
                    .where(
                        TenantUser.tenant_id == tenant_id,
                        TenantUser.user_id == user_id
                    )
                    .values(**update_data)
                    .returning(TenantUser)
                )

                tenant_user = result.scalar_one_or_none()
                if tenant_user:
                    await session.commit()
                    logger.info(f"Updated tenant user: {tenant_id}/{user_id}")
                else:
                    await session.rollback()

                return tenant_user

            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to update tenant user {tenant_id}/{user_id}: {e}")
                raise

    async def remove_user_from_tenant(
        self,
        tenant_id: UUID,
        user_id: str
    ) -> bool:
        """Remove user from tenant."""
        async with self.session_factory() as session:
            try:
                # Set tenant context for RLS
                await session.execute(
                    "SELECT set_config('app.tenant_id', :tenant_id, false)",
                    {"tenant_id": str(tenant_id)}
                )

                result = await session.execute(
                    delete(TenantUser).where(
                        TenantUser.tenant_id == tenant_id,
                        TenantUser.user_id == user_id
                    )
                )

                if result.rowcount > 0:
                    await session.commit()
                    logger.info(f"Removed user {user_id} from tenant {tenant_id}")
                    return True
                else:
                    await session.rollback()
                    return False

            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to remove user from tenant: {e}")
                raise

    async def validate_tenant_access(
        self,
        user_claims: UserClaims,
        tenant_id: UUID
    ) -> bool:
        """Validate that user has access to tenant."""
        # Super admins have access to all tenants
        if user_claims.is_super_admin():
            return True

        # Check if user's tenant matches requested tenant
        if user_claims.tenant_id == tenant_id:
            return True

        # Check if user is explicitly granted access to this tenant
        tenant_user = await self.get_tenant_user(tenant_id, user_claims.sub)
        return tenant_user is not None and tenant_user.is_active

    async def set_database_context(
        self,
        session: AsyncSession,
        tenant_id: UUID,
        user_role: Optional[str] = None
    ) -> None:
        """Set database context for tenant isolation."""
        await session.execute(
            "SELECT set_config('app.tenant_id', :tenant_id, false)",
            {"tenant_id": str(tenant_id)}
        )

        if user_role:
            await session.execute(
                "SELECT set_config('app.user_role', :user_role, false)",
                {"user_role": user_role}
            )
