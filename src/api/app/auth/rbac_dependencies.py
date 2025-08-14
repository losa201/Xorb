"""
Production RBAC Dependencies
FastAPI dependencies for Role-Based Access Control
"""

import asyncio
from functools import wraps
from typing import List, Optional, Union, Callable, Any
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..services.rbac_service import RBACService, RBACContext, PermissionCheck
from ..services.interfaces import AuthenticationService
from ..container import get_container
from ..core.logging import get_logger
from .dependencies import get_current_user
from .models import UserClaims


logger = get_logger(__name__)
security = HTTPBearer(auto_error=False)


async def get_rbac_service() -> RBACService:
    """Get RBAC service instance"""
    container = get_container()
    return container.get(RBACService)


async def get_rbac_context(request: Request) -> Optional[RBACContext]:
    """Get RBAC context from request state"""
    return getattr(request.state, 'rbac_context', None)


def require_permission(permission: str, tenant_specific: bool = True):
    """
    Dependency to require specific permission
    
    Args:
        permission: Permission name (e.g., 'ptaas:scan:create')
        tenant_specific: Whether to check permission in current tenant context
    """
    async def permission_dependency(
        request: Request,
        current_user: UserClaims = Depends(get_current_user),
        rbac_service: RBACService = Depends(get_rbac_service)
    ) -> UserClaims:
        
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        # Create RBAC context
        context = RBACContext(
            user_id=UUID(current_user.user_id) if isinstance(current_user.user_id, str) else current_user.user_id,
            tenant_id=UUID(current_user.tenant_id) if current_user.tenant_id and tenant_specific else None,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get('User-Agent'),
            metadata={
                'method': request.method,
                'path': request.url.path,
                'permission_check': permission
            }
        )
        
        # Check permission
        result = await rbac_service.check_permission(context, permission)
        
        if not result.granted:
            logger.warning(f"Permission denied: {permission} for user {current_user.user_id} - {result.reason}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required. {result.reason}"
            )
        
        logger.debug(f"Permission granted: {permission} for user {current_user.user_id}")
        return current_user
    
    return permission_dependency


def require_permissions(permissions: List[str], require_all: bool = True, tenant_specific: bool = True):
    """
    Dependency to require multiple permissions
    
    Args:
        permissions: List of permission names
        require_all: If True, requires all permissions; if False, requires any
        tenant_specific: Whether to check permissions in current tenant context
    """
    async def permissions_dependency(
        request: Request,
        current_user: UserClaims = Depends(get_current_user),
        rbac_service: RBACService = Depends(get_rbac_service)
    ) -> UserClaims:
        
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        # Create RBAC context
        context = RBACContext(
            user_id=UUID(current_user.user_id) if isinstance(current_user.user_id, str) else current_user.user_id,
            tenant_id=UUID(current_user.tenant_id) if current_user.tenant_id and tenant_specific else None,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get('User-Agent'),
            metadata={
                'method': request.method,
                'path': request.url.path,
                'permission_check': permissions
            }
        )
        
        # Check all permissions
        results = await rbac_service.check_multiple_permissions(context, permissions, require_all)
        
        # Evaluate results
        granted_count = sum(1 for result in results.values() if result.granted)
        
        if require_all and granted_count < len(permissions):
            # Need all permissions
            denied_perms = [perm for perm, result in results.items() if not result.granted]
            logger.warning(f"Missing required permissions: {denied_perms} for user {current_user.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permissions: {', '.join(denied_perms)}"
            )
        elif not require_all and granted_count == 0:
            # Need at least one permission
            logger.warning(f"No required permissions granted: {permissions} for user {current_user.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of: {', '.join(permissions)}"
            )
        
        logger.debug(f"Permissions granted: {granted_count}/{len(permissions)} for user {current_user.user_id}")
        return current_user
    
    return permissions_dependency


def require_role(role: str, tenant_specific: bool = True):
    """
    Dependency to require specific role
    
    Args:
        role: Role name (e.g., 'admin', 'security_analyst')
        tenant_specific: Whether to check role in current tenant context
    """
    async def role_dependency(
        request: Request,
        current_user: UserClaims = Depends(get_current_user),
        rbac_service: RBACService = Depends(get_rbac_service)
    ) -> UserClaims:
        
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        # Get user roles
        tenant_id = UUID(current_user.tenant_id) if current_user.tenant_id and tenant_specific else None
        user_roles = await rbac_service.get_user_roles(
            UUID(current_user.user_id) if isinstance(current_user.user_id, str) else current_user.user_id,
            tenant_id
        )
        
        # Check if user has required role
        has_role = any(r['name'] == role for r in user_roles)
        
        if not has_role:
            logger.warning(f"Role denied: {role} for user {current_user.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required"
            )
        
        logger.debug(f"Role granted: {role} for user {current_user.user_id}")
        return current_user
    
    return role_dependency


def require_any_role(roles: List[str], tenant_specific: bool = True):
    """
    Dependency to require any of the specified roles
    """
    async def roles_dependency(
        request: Request,
        current_user: UserClaims = Depends(get_current_user),
        rbac_service: RBACService = Depends(get_rbac_service)
    ) -> UserClaims:
        
        if not current_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        # Get user roles
        tenant_id = UUID(current_user.tenant_id) if current_user.tenant_id and tenant_specific else None
        user_roles = await rbac_service.get_user_roles(
            UUID(current_user.user_id) if isinstance(current_user.user_id, str) else current_user.user_id,
            tenant_id
        )
        
        user_role_names = {r['name'] for r in user_roles}
        
        # Check if user has any required role
        has_any_role = bool(user_role_names.intersection(set(roles)))
        
        if not has_any_role:
            logger.warning(f"Roles denied: {roles} for user {current_user.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {', '.join(roles)}"
            )
        
        logger.debug(f"Role granted from: {roles} for user {current_user.user_id}")
        return current_user
    
    return roles_dependency


# Predefined role dependencies for common use cases
require_admin = require_role("super_admin")
require_tenant_admin = require_any_role(["super_admin", "tenant_admin"])
require_security_manager = require_any_role(["super_admin", "tenant_admin", "security_manager"])
require_security_analyst = require_any_role(["super_admin", "tenant_admin", "security_manager", "security_analyst"])


# Predefined permission dependencies for common operations
require_user_management = require_permission("user:manage_roles")
require_ptaas_scan = require_permission("ptaas:scan:create")
require_ptaas_read = require_permission("ptaas:scan:read")
require_intelligence_read = require_permission("intelligence:read")
require_system_admin = require_permission("system:admin")
require_audit_read = require_permission("audit:read")


def rbac_decorator(
    permissions: Optional[List[str]] = None,
    roles: Optional[List[str]] = None,
    require_all_permissions: bool = True,
    require_any_role: bool = True,
    tenant_specific: bool = True
):
    """
    Decorator for applying RBAC to route functions
    
    Usage:
        @router.get("/sensitive")
        @rbac_decorator(permissions=["ptaas:scan:read"])
        async def get_scan_results():
            ...
            
        @router.post("/admin")
        @rbac_decorator(roles=["admin", "tenant_admin"], require_any_role=True)
        async def admin_action():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from function arguments
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                # Try to find request in kwargs
                request = kwargs.get('request')
            
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Request object not found for RBAC check"
                )
            
            # Get current user from request state
            if not hasattr(request.state, 'user') or not request.state.user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            current_user = request.state.user
            container = get_container()
            rbac_service = container.get(RBACService)
            
            # Create RBAC context
            context = RBACContext(
                user_id=UUID(current_user.user_id) if isinstance(current_user.user_id, str) else current_user.user_id,
                tenant_id=UUID(current_user.tenant_id) if current_user.tenant_id and tenant_specific else None,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get('User-Agent'),
                metadata={
                    'method': request.method,
                    'path': request.url.path,
                    'function': func.__name__
                }
            )
            
            # Check permissions if specified
            if permissions:
                results = await rbac_service.check_multiple_permissions(context, permissions, require_all_permissions)
                granted_count = sum(1 for result in results.values() if result.granted)
                
                if require_all_permissions and granted_count < len(permissions):
                    denied_perms = [perm for perm, result in results.items() if not result.granted]
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Missing required permissions: {', '.join(denied_perms)}"
                    )
                elif not require_all_permissions and granted_count == 0:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Requires one of: {', '.join(permissions)}"
                    )
            
            # Check roles if specified
            if roles:
                tenant_id = UUID(current_user.tenant_id) if current_user.tenant_id and tenant_specific else None
                user_roles = await rbac_service.get_user_roles(
                    UUID(current_user.user_id) if isinstance(current_user.user_id, str) else current_user.user_id,
                    tenant_id
                )
                user_role_names = {r['name'] for r in user_roles}
                
                if require_any_role:
                    if not user_role_names.intersection(set(roles)):
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"Requires one of roles: {', '.join(roles)}"
                        )
                else:
                    if not set(roles).issubset(user_role_names):
                        missing_roles = set(roles) - user_role_names
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail=f"Missing required roles: {', '.join(missing_roles)}"
                        )
            
            # Call original function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator