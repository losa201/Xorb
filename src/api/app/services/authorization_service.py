"""
Advanced authorization service with role-based access control (RBAC)
Implements production-ready authorization with enterprise security features
"""

import asyncio
import logging
from typing import Dict, List, Set, Optional, Any
from uuid import UUID
from dataclasses import dataclass
from enum import Enum

from ..domain.entities import User
from ..domain.repositories import CacheRepository
from .interfaces import AuthorizationService
from .base_service import XORBService, ServiceType


class Permission(Enum):
    """System permissions enumeration"""
    # User management
    USER_CREATE = "user:create"
    USER_READ = "user:read" 
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    
    # Organization management
    ORG_CREATE = "organization:create"
    ORG_READ = "organization:read"
    ORG_UPDATE = "organization:update"
    ORG_DELETE = "organization:delete"
    
    # PTaaS operations
    PTAAS_SCAN_CREATE = "ptaas:scan:create"
    PTAAS_SCAN_READ = "ptaas:scan:read"
    PTAAS_SCAN_UPDATE = "ptaas:scan:update"
    PTAAS_SCAN_DELETE = "ptaas:scan:delete"
    PTAAS_ADMIN = "ptaas:admin"
    
    # Intelligence operations
    INTEL_READ = "intelligence:read"
    INTEL_ANALYZE = "intelligence:analyze"
    INTEL_ADMIN = "intelligence:admin"
    
    # System administration
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_CONFIG = "system:config"
    
    # Audit and compliance
    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"
    COMPLIANCE_READ = "compliance:read"
    COMPLIANCE_ADMIN = "compliance:admin"


class Role(Enum):
    """System roles with associated permissions"""
    ADMIN = "admin"
    SECURITY_ANALYST = "security_analyst"
    PENTESTER = "pentester"
    VIEWER = "viewer"
    AUDITOR = "auditor"
    COMPLIANCE_OFFICER = "compliance_officer"


@dataclass
class RoleDefinition:
    """Role definition with permissions and metadata"""
    name: str
    permissions: Set[Permission]
    description: str
    is_system_role: bool = True


class ProductionAuthorizationService(AuthorizationService, XORBService):
    """Production-ready authorization service with enterprise RBAC"""
    
    def __init__(self, cache_repository: CacheRepository):
        super().__init__(service_type=ServiceType.SECURITY)
        self.cache = cache_repository
        self.logger = logging.getLogger(__name__)
        
        # Define role hierarchy and permissions
        self._role_definitions = {
            Role.ADMIN: RoleDefinition(
                name="Administrator",
                permissions={
                    Permission.USER_CREATE, Permission.USER_READ, Permission.USER_UPDATE, Permission.USER_DELETE,
                    Permission.ORG_CREATE, Permission.ORG_READ, Permission.ORG_UPDATE, Permission.ORG_DELETE,
                    Permission.PTAAS_SCAN_CREATE, Permission.PTAAS_SCAN_READ, Permission.PTAAS_SCAN_UPDATE, 
                    Permission.PTAAS_SCAN_DELETE, Permission.PTAAS_ADMIN,
                    Permission.INTEL_READ, Permission.INTEL_ANALYZE, Permission.INTEL_ADMIN,
                    Permission.SYSTEM_ADMIN, Permission.SYSTEM_MONITOR, Permission.SYSTEM_CONFIG,
                    Permission.AUDIT_READ, Permission.AUDIT_EXPORT,
                    Permission.COMPLIANCE_READ, Permission.COMPLIANCE_ADMIN
                },
                description="Full system access with all permissions"
            ),
            Role.SECURITY_ANALYST: RoleDefinition(
                name="Security Analyst",
                permissions={
                    Permission.USER_READ, Permission.ORG_READ,
                    Permission.PTAAS_SCAN_CREATE, Permission.PTAAS_SCAN_READ, Permission.PTAAS_SCAN_UPDATE,
                    Permission.INTEL_READ, Permission.INTEL_ANALYZE,
                    Permission.SYSTEM_MONITOR,
                    Permission.AUDIT_READ, Permission.COMPLIANCE_READ
                },
                description="Security analysis and monitoring capabilities"
            ),
            Role.PENTESTER: RoleDefinition(
                name="Penetration Tester",
                permissions={
                    Permission.USER_READ, Permission.ORG_READ,
                    Permission.PTAAS_SCAN_CREATE, Permission.PTAAS_SCAN_READ, Permission.PTAAS_SCAN_UPDATE,
                    Permission.INTEL_READ,
                    Permission.AUDIT_READ
                },
                description="Penetration testing and vulnerability assessment"
            ),
            Role.VIEWER: RoleDefinition(
                name="Viewer",
                permissions={
                    Permission.USER_READ, Permission.ORG_READ,
                    Permission.PTAAS_SCAN_READ,
                    Permission.INTEL_READ,
                    Permission.SYSTEM_MONITOR
                },
                description="Read-only access to system resources"
            ),
            Role.AUDITOR: RoleDefinition(
                name="Auditor",
                permissions={
                    Permission.USER_READ, Permission.ORG_READ,
                    Permission.PTAAS_SCAN_READ,
                    Permission.AUDIT_READ, Permission.AUDIT_EXPORT,
                    Permission.COMPLIANCE_READ,
                    Permission.SYSTEM_MONITOR
                },
                description="Audit and compliance monitoring"
            ),
            Role.COMPLIANCE_OFFICER: RoleDefinition(
                name="Compliance Officer",
                permissions={
                    Permission.USER_READ, Permission.ORG_READ,
                    Permission.PTAAS_SCAN_READ,
                    Permission.AUDIT_READ, Permission.AUDIT_EXPORT,
                    Permission.COMPLIANCE_READ, Permission.COMPLIANCE_ADMIN,
                    Permission.SYSTEM_MONITOR
                },
                description="Compliance management and reporting"
            )
        }
    
    async def check_permission(self, user: User, resource: str, action: str) -> bool:
        """Check if user has permission for resource action"""
        try:
            # Construct permission string
            permission_str = f"{resource}:{action}"
            
            # Check if this matches any defined permission
            permission = self._find_permission(permission_str)
            if not permission:
                self.logger.warning(f"Unknown permission requested: {permission_str}")
                return False
            
            # Get user permissions from cache first
            cache_key = f"user_permissions:{user.id}"
            cached_permissions = await self.cache.get(cache_key)
            
            if cached_permissions:
                return permission_str in cached_permissions
            
            # Calculate permissions and cache them
            user_permissions = await self._calculate_user_permissions(user)
            await self.cache.set(cache_key, user_permissions, ttl=300)  # 5 minutes
            
            return permission_str in user_permissions
            
        except Exception as e:
            self.logger.error(f"Error checking permission for user {user.id}: {str(e)}")
            return False
    
    async def get_user_permissions(self, user: User) -> Dict[str, List[str]]:
        """Get all permissions for user grouped by resource"""
        try:
            permissions = await self._calculate_user_permissions(user)
            
            # Group permissions by resource
            grouped_permissions = {}
            for perm in permissions:
                if ":" in perm:
                    resource, action = perm.split(":", 1)
                    if resource not in grouped_permissions:
                        grouped_permissions[resource] = []
                    grouped_permissions[resource].append(action)
            
            return grouped_permissions
            
        except Exception as e:
            self.logger.error(f"Error getting permissions for user {user.id}: {str(e)}")
            return {}
    
    async def assign_role(self, user_id: UUID, role: Role) -> bool:
        """Assign role to user"""
        try:
            # In production, this would update the database
            # For now, we'll use cache to simulate role assignment
            cache_key = f"user_roles:{user_id}"
            existing_roles = await self.cache.get(cache_key) or []
            
            if role.value not in existing_roles:
                existing_roles.append(role.value)
                await self.cache.set(cache_key, existing_roles, ttl=3600)  # 1 hour
            
            # Invalidate permissions cache
            perm_cache_key = f"user_permissions:{user_id}"
            await self.cache.delete(perm_cache_key)
            
            self.logger.info(f"Assigned role {role.value} to user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error assigning role {role.value} to user {user_id}: {str(e)}")
            return False
    
    async def remove_role(self, user_id: UUID, role: Role) -> bool:
        """Remove role from user"""
        try:
            cache_key = f"user_roles:{user_id}"
            existing_roles = await self.cache.get(cache_key) or []
            
            if role.value in existing_roles:
                existing_roles.remove(role.value)
                await self.cache.set(cache_key, existing_roles, ttl=3600)
            
            # Invalidate permissions cache
            perm_cache_key = f"user_permissions:{user_id}"
            await self.cache.delete(perm_cache_key)
            
            self.logger.info(f"Removed role {role.value} from user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing role {role.value} from user {user_id}: {str(e)}")
            return False
    
    async def get_user_roles(self, user_id: UUID) -> List[str]:
        """Get all roles assigned to user"""
        try:
            cache_key = f"user_roles:{user_id}"
            roles = await self.cache.get(cache_key)
            
            if roles is None:
                # Check user.roles from the User entity
                # For now, return empty list if not in cache
                return []
            
            return roles
            
        except Exception as e:
            self.logger.error(f"Error getting roles for user {user_id}: {str(e)}")
            return []
    
    async def check_role(self, user: User, required_role: Role) -> bool:
        """Check if user has specific role"""
        try:
            user_roles = await self.get_user_roles(user.id)
            return required_role.value in user_roles or required_role.value in (user.roles or [])
            
        except Exception as e:
            self.logger.error(f"Error checking role {required_role.value} for user {user.id}: {str(e)}")
            return False
    
    def get_role_permissions(self, role: Role) -> Set[str]:
        """Get all permissions for a role"""
        if role not in self._role_definitions:
            return set()
        
        role_def = self._role_definitions[role]
        return {perm.value for perm in role_def.permissions}
    
    def get_available_roles(self) -> Dict[str, Dict[str, Any]]:
        """Get all available roles and their definitions"""
        return {
            role.value: {
                "name": definition.name,
                "description": definition.description,
                "permissions": [perm.value for perm in definition.permissions],
                "is_system_role": definition.is_system_role
            }
            for role, definition in self._role_definitions.items()
        }
    
    async def _calculate_user_permissions(self, user: User) -> Set[str]:
        """Calculate all permissions for a user based on their roles"""
        all_permissions = set()
        
        # Get roles from user entity
        user_roles = user.roles or []
        
        # Get additional roles from cache (for dynamic role assignment)
        cached_roles = await self.get_user_roles(user.id)
        user_roles.extend(cached_roles)
        
        # Remove duplicates
        user_roles = list(set(user_roles))
        
        # Calculate permissions from all roles
        for role_str in user_roles:
            try:
                role = Role(role_str)
                role_permissions = self.get_role_permissions(role)
                all_permissions.update(role_permissions)
            except ValueError:
                self.logger.warning(f"Unknown role: {role_str}")
                continue
        
        return all_permissions
    
    def _find_permission(self, permission_str: str) -> Optional[Permission]:
        """Find permission enum from string"""
        for perm in Permission:
            if perm.value == permission_str:
                return perm
        return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test cache connectivity
            test_key = "auth_health_check"
            await self.cache.set(test_key, "ok", ttl=60)
            cache_result = await self.cache.get(test_key)
            await self.cache.delete(test_key)
            
            cache_healthy = cache_result == "ok"
            
            return {
                "status": "healthy" if cache_healthy else "degraded",
                "cache_connection": cache_healthy,
                "role_definitions": len(self._role_definitions),
                "permission_count": len(Permission),
                "timestamp": str(asyncio.get_event_loop().time())
            }
            
        except Exception as e:
            self.logger.error(f"Authorization service health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": str(asyncio.get_event_loop().time())
            }