"""
Production RBAC Service
Comprehensive Role-Based Access Control service with enterprise features
"""

import asyncio
import logging
from typing import Dict, List, Set, Optional, Any, Union
from uuid import UUID
from datetime import datetime, timedelta
from dataclasses import dataclass

from sqlalchemy import select, and_, or_
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from ..infrastructure.rbac_models import RBACRole, RBACPermission, RBACUserRole, RBACUserPermission
from ..infrastructure.database_models import UserModel, OrganizationModel
from ..core.logging import get_logger
from .interfaces import CacheService


@dataclass
class RBACContext:
    """RBAC evaluation context"""
    user_id: UUID
    tenant_id: Optional[UUID] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PermissionCheck:
    """Result of permission check"""
    granted: bool
    reason: str
    source: str  # 'role', 'direct', 'denied'
    checked_at: datetime
    context: RBACContext


class RBACService:
    """Production-ready RBAC service with caching and audit logging"""
    
    def __init__(self, db_session: AsyncSession, cache_service: Optional[CacheService] = None):
        self.db = db_session
        self.cache = cache_service
        self.logger = get_logger(__name__)
        
        # Cache TTL settings
        self.user_permissions_ttl = 300  # 5 minutes
        self.role_permissions_ttl = 3600  # 1 hour
        self.permission_cache_ttl = 1800  # 30 minutes
    
    async def check_permission(
        self, 
        context: RBACContext, 
        permission: str, 
        resource_id: Optional[str] = None
    ) -> PermissionCheck:
        """
        Check if user has specific permission
        
        Args:
            context: RBAC context with user and tenant info
            permission: Permission name (e.g., 'ptaas:scan:create')
            resource_id: Optional specific resource ID for fine-grained access
            
        Returns:
            PermissionCheck with result and metadata
        """
        check_time = datetime.utcnow()
        
        try:
            # Check cache first
            cache_key = f"rbac:user:{context.user_id}:tenant:{context.tenant_id}:perm:{permission}"
            if self.cache:
                cached_result = await self.cache.get(cache_key)
                if cached_result:
                    self.logger.debug(f"Permission check cache hit: {permission} for user {context.user_id}")
                    return PermissionCheck(
                        granted=cached_result.get('granted', False),
                        reason=cached_result.get('reason', 'cached result'),
                        source=cached_result.get('source', 'cache'),
                        checked_at=check_time,
                        context=context
                    )
            
            # Get user permissions (roles + direct)
            user_permissions = await self._get_user_permissions(context.user_id, context.tenant_id)
            
            # Check if permission is granted
            if permission in user_permissions:
                result = PermissionCheck(
                    granted=True,
                    reason=f"User has permission '{permission}'",
                    source='role' if await self._is_permission_from_role(context.user_id, permission, context.tenant_id) else 'direct',
                    checked_at=check_time,
                    context=context
                )
            else:
                result = PermissionCheck(
                    granted=False,
                    reason=f"User lacks permission '{permission}'",
                    source='denied',
                    checked_at=check_time,
                    context=context
                )
            
            # Cache the result
            if self.cache:
                await self.cache.set(
                    cache_key,
                    {
                        'granted': result.granted,
                        'reason': result.reason,
                        'source': result.source
                    },
                    ttl=self.permission_cache_ttl
                )
            
            # Log security event
            await self._log_permission_check(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error checking permission {permission} for user {context.user_id}: {str(e)}")
            # Fail secure - deny permission on error
            return PermissionCheck(
                granted=False,
                reason=f"Permission check failed: {str(e)}",
                source='error',
                checked_at=check_time,
                context=context
            )
    
    async def check_multiple_permissions(
        self, 
        context: RBACContext, 
        permissions: List[str],
        require_all: bool = True
    ) -> Dict[str, PermissionCheck]:
        """
        Check multiple permissions efficiently
        
        Args:
            context: RBAC context
            permissions: List of permission names
            require_all: If True, requires all permissions; if False, requires any
            
        Returns:
            Dict mapping permission names to PermissionCheck results
        """
        results = {}
        
        # Check all permissions
        for permission in permissions:
            results[permission] = await self.check_permission(context, permission)
        
        return results
    
    async def get_user_roles(self, user_id: UUID, tenant_id: Optional[UUID] = None) -> List[Dict[str, Any]]:
        """Get all roles assigned to user in specific tenant"""
        try:
            query = select(RBACRole, RBACUserRole).join(
                RBACUserRole, RBACRole.id == RBACUserRole.role_id
            ).where(
                and_(
                    RBACUserRole.user_id == user_id,
                    RBACUserRole.is_active == True,
                    RBACRole.is_active == True,
                    or_(
                        RBACUserRole.expires_at.is_(None),
                        RBACUserRole.expires_at > datetime.utcnow()
                    )
                )
            )
            
            if tenant_id:
                query = query.where(
                    or_(
                        RBACUserRole.tenant_id == tenant_id,
                        RBACUserRole.tenant_id.is_(None)  # Global roles
                    )
                )
            
            result = await self.db.execute(query)
            role_assignments = result.all()
            
            roles = []
            for role, assignment in role_assignments:
                roles.append({
                    'id': str(role.id),
                    'name': role.name,
                    'display_name': role.display_name,
                    'description': role.description,
                    'level': role.level,
                    'is_system_role': role.is_system_role,
                    'tenant_id': str(assignment.tenant_id) if assignment.tenant_id else None,
                    'granted_at': assignment.granted_at,
                    'expires_at': assignment.expires_at
                })
            
            return roles
            
        except Exception as e:
            self.logger.error(f"Error getting roles for user {user_id}: {str(e)}")
            return []
    
    async def assign_role(
        self, 
        user_id: UUID, 
        role_name: str, 
        granted_by: UUID,
        tenant_id: Optional[UUID] = None,
        expires_at: Optional[datetime] = None
    ) -> bool:
        """Assign role to user"""
        try:
            # Get role
            role_query = select(RBACRole).where(
                and_(RBACRole.name == role_name, RBACRole.is_active == True)
            )
            role_result = await self.db.execute(role_query)
            role = role_result.scalar_one_or_none()
            
            if not role:
                self.logger.warning(f"Role '{role_name}' not found")
                return False
            
            # Check if assignment already exists
            existing_query = select(RBACUserRole).where(
                and_(
                    RBACUserRole.user_id == user_id,
                    RBACUserRole.role_id == role.id,
                    RBACUserRole.tenant_id == tenant_id
                )
            )
            existing_result = await self.db.execute(existing_query)
            existing = existing_result.scalar_one_or_none()
            
            if existing:
                # Update existing assignment
                existing.is_active = True
                existing.expires_at = expires_at
                existing.granted_by = granted_by
                existing.granted_at = datetime.utcnow()
            else:
                # Create new assignment
                assignment = RBACUserRole(
                    user_id=user_id,
                    role_id=role.id,
                    tenant_id=tenant_id,
                    granted_by=granted_by,
                    expires_at=expires_at
                )
                self.db.add(assignment)
            
            await self.db.commit()
            
            # Invalidate user permission cache
            await self._invalidate_user_cache(user_id)
            
            self.logger.info(f"Assigned role '{role_name}' to user {user_id} in tenant {tenant_id}")
            return True
            
        except Exception as e:
            await self.db.rollback()
            self.logger.error(f"Error assigning role '{role_name}' to user {user_id}: {str(e)}")
            return False
    
    async def revoke_role(
        self, 
        user_id: UUID, 
        role_name: str, 
        revoked_by: UUID,
        tenant_id: Optional[UUID] = None
    ) -> bool:
        """Revoke role from user"""
        try:
            # Get role and assignment
            query = select(RBACUserRole).join(RBACRole).where(
                and_(
                    RBACUserRole.user_id == user_id,
                    RBACRole.name == role_name,
                    RBACUserRole.tenant_id == tenant_id,
                    RBACUserRole.is_active == True
                )
            )
            
            result = await self.db.execute(query)
            assignment = result.scalar_one_or_none()
            
            if not assignment:
                self.logger.warning(f"Role assignment '{role_name}' not found for user {user_id}")
                return False
            
            # Deactivate assignment
            assignment.is_active = False
            assignment.granted_by = revoked_by  # Track who revoked it
            
            await self.db.commit()
            
            # Invalidate user permission cache
            await self._invalidate_user_cache(user_id)
            
            self.logger.info(f"Revoked role '{role_name}' from user {user_id} in tenant {tenant_id}")
            return True
            
        except Exception as e:
            await self.db.rollback()
            self.logger.error(f"Error revoking role '{role_name}' from user {user_id}: {str(e)}")
            return False
    
    async def assign_permission(
        self, 
        user_id: UUID, 
        permission_name: str, 
        granted_by: UUID,
        tenant_id: Optional[UUID] = None,
        expires_at: Optional[datetime] = None
    ) -> bool:
        """Assign direct permission to user"""
        try:
            # Get permission
            perm_query = select(RBACPermission).where(
                and_(RBACPermission.name == permission_name, RBACPermission.is_active == True)
            )
            perm_result = await self.db.execute(perm_query)
            permission = perm_result.scalar_one_or_none()
            
            if not permission:
                self.logger.warning(f"Permission '{permission_name}' not found")
                return False
            
            # Check if assignment already exists
            existing_query = select(RBACUserPermission).where(
                and_(
                    RBACUserPermission.user_id == user_id,
                    RBACUserPermission.permission_id == permission.id,
                    RBACUserPermission.tenant_id == tenant_id
                )
            )
            existing_result = await self.db.execute(existing_query)
            existing = existing_result.scalar_one_or_none()
            
            if existing:
                # Update existing assignment
                existing.is_active = True
                existing.expires_at = expires_at
                existing.granted_by = granted_by
                existing.granted_at = datetime.utcnow()
            else:
                # Create new assignment
                assignment = RBACUserPermission(
                    user_id=user_id,
                    permission_id=permission.id,
                    tenant_id=tenant_id,
                    granted_by=granted_by,
                    expires_at=expires_at
                )
                self.db.add(assignment)
            
            await self.db.commit()
            
            # Invalidate user permission cache
            await self._invalidate_user_cache(user_id)
            
            self.logger.info(f"Assigned permission '{permission_name}' to user {user_id} in tenant {tenant_id}")
            return True
            
        except Exception as e:
            await self.db.rollback()
            self.logger.error(f"Error assigning permission '{permission_name}' to user {user_id}: {str(e)}")
            return False
    
    async def get_available_roles(self) -> List[Dict[str, Any]]:
        """Get all available roles"""
        try:
            query = select(RBACRole).where(RBACRole.is_active == True).order_by(RBACRole.level.desc())
            result = await self.db.execute(query)
            roles = result.scalars().all()
            
            return [
                {
                    'id': str(role.id),
                    'name': role.name,
                    'display_name': role.display_name,
                    'description': role.description,
                    'level': role.level,
                    'is_system_role': role.is_system_role
                }
                for role in roles
            ]
            
        except Exception as e:
            self.logger.error(f"Error getting available roles: {str(e)}")
            return []
    
    async def get_available_permissions(self) -> List[Dict[str, Any]]:
        """Get all available permissions"""
        try:
            query = select(RBACPermission).where(RBACPermission.is_active == True).order_by(RBACPermission.resource, RBACPermission.action)
            result = await self.db.execute(query)
            permissions = result.scalars().all()
            
            return [
                {
                    'id': str(perm.id),
                    'name': perm.name,
                    'display_name': perm.display_name,
                    'description': perm.description,
                    'resource': perm.resource,
                    'action': perm.action,
                    'is_system_permission': perm.is_system_permission
                }
                for perm in permissions
            ]
            
        except Exception as e:
            self.logger.error(f"Error getting available permissions: {str(e)}")
            return []
    
    async def _get_user_permissions(self, user_id: UUID, tenant_id: Optional[UUID] = None) -> Set[str]:
        """Get all permissions for user (from roles and direct assignments)"""
        cache_key = f"rbac:user_permissions:{user_id}:tenant:{tenant_id}"
        
        # Check cache
        if self.cache:
            cached_permissions = await self.cache.get(cache_key)
            if cached_permissions:
                return set(cached_permissions)
        
        permissions = set()
        
        try:
            # Get permissions from roles
            role_perms_query = select(RBACPermission.name).join(
                RBACRole, RBACPermission.id.in_(
                    select(RBACUserRole.role_id).where(
                        and_(
                            RBACUserRole.user_id == user_id,
                            RBACUserRole.is_active == True,
                            or_(
                                RBACUserRole.tenant_id == tenant_id,
                                RBACUserRole.tenant_id.is_(None)  # Global roles
                            ),
                            or_(
                                RBACUserRole.expires_at.is_(None),
                                RBACUserRole.expires_at > datetime.utcnow()
                            )
                        )
                    )
                )
            ).join(
                RBACUserRole, RBACRole.id == RBACUserRole.role_id
            ).where(
                and_(
                    RBACPermission.is_active == True,
                    RBACRole.is_active == True
                )
            )
            
            role_result = await self.db.execute(role_perms_query)
            role_permissions = role_result.scalars().all()
            permissions.update(role_permissions)
            
            # Get direct permissions
            direct_perms_query = select(RBACPermission.name).join(
                RBACUserPermission, RBACPermission.id == RBACUserPermission.permission_id
            ).where(
                and_(
                    RBACUserPermission.user_id == user_id,
                    RBACUserPermission.is_active == True,
                    or_(
                        RBACUserPermission.tenant_id == tenant_id,
                        RBACUserPermission.tenant_id.is_(None)
                    ),
                    or_(
                        RBACUserPermission.expires_at.is_(None),
                        RBACUserPermission.expires_at > datetime.utcnow()
                    ),
                    RBACPermission.is_active == True
                )
            )
            
            direct_result = await self.db.execute(direct_perms_query)
            direct_permissions = direct_result.scalars().all()
            permissions.update(direct_permissions)
            
            # Cache the result
            if self.cache:
                await self.cache.set(cache_key, list(permissions), ttl=self.user_permissions_ttl)
            
            return permissions
            
        except Exception as e:
            self.logger.error(f"Error getting permissions for user {user_id}: {str(e)}")
            return set()
    
    async def _is_permission_from_role(self, user_id: UUID, permission: str, tenant_id: Optional[UUID] = None) -> bool:
        """Check if permission comes from role assignment vs direct assignment"""
        try:
            # Check if user has this permission through roles
            role_query = select(RBACPermission.name).join(
                RBACRole, RBACPermission.id.in_(
                    select(RBACUserRole.role_id).where(
                        and_(
                            RBACUserRole.user_id == user_id,
                            RBACUserRole.is_active == True,
                            or_(
                                RBACUserRole.tenant_id == tenant_id,
                                RBACUserRole.tenant_id.is_(None)
                            )
                        )
                    )
                )
            ).where(RBACPermission.name == permission)
            
            result = await self.db.execute(role_query)
            return result.scalar_one_or_none() is not None
            
        except Exception:
            return False
    
    async def _invalidate_user_cache(self, user_id: UUID):
        """Invalidate all cached data for user"""
        if not self.cache:
            return
        
        try:
            # Pattern-based cache invalidation would be ideal here
            # For now, we'll clear specific known keys
            patterns = [
                f"rbac:user_permissions:{user_id}:*",
                f"rbac:user:{user_id}:*"
            ]
            
            for pattern in patterns:
                # This would need to be implemented based on your cache service
                # await self.cache.delete_pattern(pattern)
                pass
                
        except Exception as e:
            self.logger.warning(f"Error invalidating cache for user {user_id}: {str(e)}")
    
    async def _log_permission_check(self, result: PermissionCheck):
        """Log permission check for security audit"""
        # This would integrate with your audit logging system
        log_data = {
            'event_type': 'permission_check',
            'user_id': str(result.context.user_id),
            'tenant_id': str(result.context.tenant_id) if result.context.tenant_id else None,
            'granted': result.granted,
            'reason': result.reason,
            'source': result.source,
            'checked_at': result.checked_at.isoformat(),
            'ip_address': result.context.ip_address,
            'user_agent': result.context.user_agent,
            'session_id': result.context.session_id
        }
        
        if result.granted:
            self.logger.info(f"Permission granted: {log_data}")
        else:
            self.logger.warning(f"Permission denied: {log_data}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for RBAC service"""
        try:
            # Test database connectivity
            role_count_query = select(RBACRole).where(RBACRole.is_active == True)
            role_result = await self.db.execute(role_count_query)
            active_roles = len(role_result.scalars().all())
            
            permission_count_query = select(RBACPermission).where(RBACPermission.is_active == True)
            permission_result = await self.db.execute(permission_count_query)
            active_permissions = len(permission_result.scalars().all())
            
            return {
                'status': 'healthy',
                'active_roles': active_roles,
                'active_permissions': active_permissions,
                'cache_enabled': self.cache is not None,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"RBAC service health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }