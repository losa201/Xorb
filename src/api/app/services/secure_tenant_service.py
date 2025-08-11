"""
Secure Tenant Service
Production-grade tenant management with security enforcement
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from ..core.secure_tenant_context import TenantContext, SecureTenantContextManager
from ..domain.tenant_entities import (
    Tenant, TenantUser, TenantCreate, TenantUpdate, 
    TenantUserCreate, TenantUserUpdate, TenantStatus
)
from ..infrastructure.secure_repositories import SecureRepositoryBase, create_secure_repository
from ..infrastructure.secure_query_builder import SecureQueryBuilder, secure_select, secure_update
from ..auth.models import UserClaims, Role
from ..core.logging import get_logger

logger = get_logger(__name__)


class SecureTenantService:
    """
    Secure tenant management service with mandatory tenant isolation
    
    This service replaces the vulnerable TenantService with security hardening:
    - All operations require validated tenant context
    - User-tenant relationship validation
    - Comprehensive audit logging
    - SQL injection prevention
    - Cross-tenant access prevention
    """
    
    def __init__(
        self, 
        session: AsyncSession,
        tenant_context: TenantContext,
        tenant_manager: SecureTenantContextManager
    ):
        self.session = session
        self.tenant_context = tenant_context
        self.tenant_manager = tenant_manager
        self.query_builder = SecureQueryBuilder(tenant_context)
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    async def get_tenant(self, tenant_id: UUID) -> Optional[Tenant]:
        """
        Get tenant by ID with security validation
        
        Args:
            tenant_id: Tenant ID to retrieve
            
        Returns:
            Tenant object if authorized, None otherwise
            
        Raises:
            HTTPException: If access denied
        """
        # Validate access to requested tenant
        if tenant_id != self.tenant_context.tenant_id:
            # Only super admins can access other tenants
            if not self._is_super_admin():
                self.logger.warning(
                    f"Unauthorized tenant access attempt: user={self.tenant_context.user_id}, "
                    f"requested={tenant_id}, authorized={self.tenant_context.tenant_id}"
                )
                return None
        
        try:
            result = await secure_select(
                self.session,
                self.tenant_context,
                "tenants",
                where_conditions={"id": tenant_id}
            )
            
            row = result.first()
            if row:
                # Convert row to Tenant object
                return self._row_to_tenant(row)
            
            return None
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error getting tenant {tenant_id}: {e}")
            raise
    
    async def get_tenant_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get tenant by slug with validation"""
        try:
            result = await secure_select(
                self.session,
                self.tenant_context,
                "tenants",
                where_conditions={"slug": slug}
            )
            
            row = result.first()
            if row:
                tenant = self._row_to_tenant(row)
                
                # Validate access to this tenant
                if tenant.id != self.tenant_context.tenant_id and not self._is_super_admin():
                    self.logger.warning(
                        f"Unauthorized tenant access by slug: user={self.tenant_context.user_id}, "
                        f"slug={slug}, tenant_id={tenant.id}"
                    )
                    return None
                
                return tenant
            
            return None
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error getting tenant by slug {slug}: {e}")
            raise
    
    async def create_tenant(self, tenant_data: TenantCreate, admin_user_id: str) -> Tenant:
        """
        Create a new tenant (super admin only)
        
        Args:
            tenant_data: Tenant creation data
            admin_user_id: ID of admin creating the tenant
            
        Returns:
            Created tenant
            
        Raises:
            ValueError: If not authorized or constraint violation
        """
        # Only super admins can create tenants
        if not self._is_super_admin():
            raise ValueError("Only super administrators can create tenants")
        
        try:
            # Prepare tenant data
            create_data = {
                "name": tenant_data.name,
                "slug": tenant_data.slug,
                "plan": tenant_data.plan.value,
                "contact_email": tenant_data.contact_email,
                "contact_name": tenant_data.contact_name,
                "max_users": tenant_data.max_users,
                "max_storage_gb": tenant_data.max_storage_gb,
                "settings": tenant_data.settings,
                "status": TenantStatus.ACTIVE.value,
                "created_at": datetime.utcnow()
            }
            
            # Create tenant without tenant context (since it's a new tenant)
            # This requires special handling for super admin operations
            result = await self.session.execute(
                """
                INSERT INTO tenants (name, slug, plan, contact_email, contact_name, 
                                   max_users, max_storage_gb, settings, status, created_at)
                VALUES (:name, :slug, :plan, :contact_email, :contact_name,
                        :max_users, :max_storage_gb, :settings, :status, :created_at)
                RETURNING *
                """,
                create_data
            )
            
            tenant_row = result.first()
            tenant = self._row_to_tenant(tenant_row)
            
            await self.session.commit()
            
            self.logger.info(
                f"Tenant created: {tenant.name} ({tenant.id}) by admin {admin_user_id}"
            )
            
            return tenant
            
        except IntegrityError as e:
            await self.session.rollback()
            self.logger.error(f"Tenant creation constraint violation: {e}")
            raise ValueError(f"Tenant with slug '{tenant_data.slug}' already exists")
        except SQLAlchemyError as e:
            await self.session.rollback()
            self.logger.error(f"Database error creating tenant: {e}")
            raise
    
    async def update_tenant(
        self, 
        tenant_id: UUID, 
        tenant_data: TenantUpdate
    ) -> Optional[Tenant]:
        """
        Update tenant with security validation
        
        Args:
            tenant_id: Tenant to update
            tenant_data: Update data
            
        Returns:
            Updated tenant if successful
        """
        # Validate access to tenant
        if tenant_id != self.tenant_context.tenant_id:
            if not self._is_super_admin():
                self.logger.warning(
                    f"Unauthorized tenant update attempt: user={self.tenant_context.user_id}, "
                    f"tenant={tenant_id}"
                )
                raise ValueError("Access denied to update this tenant")
        
        try:
            # Build update data, excluding None values
            update_data = {}
            for field, value in tenant_data.model_dump(exclude_unset=True).items():
                if value is not None:
                    if field == 'status' and isinstance(value, TenantStatus):
                        update_data[field] = value.value
                    else:
                        update_data[field] = value
            
            if not update_data:
                return await self.get_tenant(tenant_id)
            
            # Add updated timestamp
            update_data['updated_at'] = datetime.utcnow()
            
            # Use secure update
            result = await secure_update(
                self.session,
                self.tenant_context,
                "tenants",
                update_data,
                {"id": tenant_id},
                returning=["*"]
            )
            
            row = result.first()
            if row:
                await self.session.commit()
                tenant = self._row_to_tenant(row)
                
                self.logger.info(
                    f"Tenant updated: {tenant_id} by user {self.tenant_context.user_id}"
                )
                
                return tenant
            
            return None
            
        except SQLAlchemyError as e:
            await self.session.rollback()
            self.logger.error(f"Database error updating tenant {tenant_id}: {e}")
            raise
    
    async def list_tenant_users(
        self, 
        tenant_id: UUID,
        active_only: bool = True
    ) -> List[TenantUser]:
        """
        List users for a tenant with security validation
        
        Args:
            tenant_id: Tenant to list users for
            active_only: Whether to include only active users
            
        Returns:
            List of tenant users
        """
        # Validate access to tenant
        if tenant_id != self.tenant_context.tenant_id:
            if not self._is_super_admin():
                self.logger.warning(
                    f"Unauthorized tenant user list attempt: user={self.tenant_context.user_id}, "
                    f"tenant={tenant_id}"
                )
                return []
        
        try:
            where_conditions = {"tenant_id": tenant_id}
            if active_only:
                where_conditions["is_active"] = True
            
            result = await secure_select(
                self.session,
                self.tenant_context,
                "tenant_users",
                where_conditions=where_conditions
            )
            
            users = []
            for row in result:
                users.append(self._row_to_tenant_user(row))
            
            return users
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error listing tenant users: {e}")
            raise
    
    async def add_user_to_tenant(
        self, 
        tenant_id: UUID, 
        user_data: TenantUserCreate
    ) -> TenantUser:
        """
        Add user to tenant with security validation
        
        Args:
            tenant_id: Target tenant
            user_data: User data
            
        Returns:
            Created tenant user relationship
        """
        # Validate access to tenant
        if tenant_id != self.tenant_context.tenant_id:
            if not self._is_super_admin():
                raise ValueError("Access denied to add user to this tenant")
        
        try:
            create_data = {
                "tenant_id": tenant_id,
                "user_id": user_data.user_id,
                "email": user_data.email,
                "roles": user_data.roles,
                "is_active": True,
                "joined_at": datetime.utcnow(),
                "created_at": datetime.utcnow()
            }
            
            # Use secure insert but manually handle tenant_id for this cross-tenant operation
            # if performed by super admin
            result = await self.session.execute(
                """
                INSERT INTO tenant_users (tenant_id, user_id, email, roles, is_active, joined_at, created_at)
                VALUES (:tenant_id, :user_id, :email, :roles, :is_active, :joined_at, :created_at)
                RETURNING *
                """,
                create_data
            )
            
            row = result.first()
            tenant_user = self._row_to_tenant_user(row)
            
            await self.session.commit()
            
            self.logger.info(
                f"User {user_data.user_id} added to tenant {tenant_id} "
                f"by {self.tenant_context.user_id}"
            )
            
            return tenant_user
            
        except IntegrityError as e:
            await self.session.rollback()
            self.logger.error(f"User-tenant relationship constraint violation: {e}")
            raise ValueError("User already exists in this tenant")
        except SQLAlchemyError as e:
            await self.session.rollback()
            self.logger.error(f"Database error adding user to tenant: {e}")
            raise
    
    async def validate_user_tenant_access(
        self, 
        user_id: str, 
        tenant_id: UUID
    ) -> bool:
        """
        Validate user has access to tenant
        
        Args:
            user_id: User to validate
            tenant_id: Tenant to check access for
            
        Returns:
            True if user has access
        """
        try:
            result = await secure_select(
                self.session,
                self.tenant_context,
                "tenant_users",
                where_conditions={
                    "user_id": user_id,
                    "tenant_id": tenant_id,
                    "is_active": True
                }
            )
            
            return result.first() is not None
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error validating user tenant access: {e}")
            return False
    
    def _is_super_admin(self) -> bool:
        """Check if current user is super admin"""
        return "super_admin" in self.tenant_context.permissions
    
    def _row_to_tenant(self, row) -> Tenant:
        """Convert database row to Tenant object"""
        return Tenant(
            id=row.id,
            name=row.name,
            slug=row.slug,
            status=row.status,
            plan=row.plan,
            settings=row.settings or {},
            created_at=row.created_at,
            updated_at=row.updated_at,
            contact_email=row.contact_email,
            contact_name=row.contact_name
        )
    
    def _row_to_tenant_user(self, row) -> TenantUser:
        """Convert database row to TenantUser object"""
        return TenantUser(
            id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
            email=row.email,
            roles=row.roles or [],
            is_active=row.is_active,
            joined_at=row.joined_at,
            last_login=row.last_login,
            created_at=row.created_at,
            updated_at=row.updated_at
        )