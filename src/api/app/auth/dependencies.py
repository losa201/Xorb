"""Authentication dependencies for FastAPI."""
from functools import wraps
from typing import Callable, List, Optional, Set, Union

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .models import Permission, Role, UserClaims
from ..container import get_container
from ..services.interfaces import AuthenticationService


security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[UserClaims]:
    """Extract current user from JWT token using unified auth service."""
    if not credentials:
        return None
    
    try:
        # Get unified authentication service
        container = get_container()
        auth_service: AuthenticationService = container.get(AuthenticationService)
        
        # Validate token using unified service
        validation_result = await auth_service.validate_token(credentials.credentials)
        
        if not validation_result.valid:
            return None
        
        # Create user context from validation result
        user_context = await auth_service.create_user_context(
            validation_result.user, 
            validation_result.claims
        )
        
        # Create UserClaims from user context for compatibility
        user_claims = UserClaims(
            user_id=user_context.user_id,
            username=user_context.username,
            tenant_id=user_context.tenant_id,
            roles=user_context.roles,
            permissions=user_context.permissions,
            is_admin=user_context.is_admin
        )
        
        # Store in request state for downstream middleware
        request.state.user = user_claims
        return user_claims
        
    except Exception:
        return None


async def require_auth(
    current_user: Optional[UserClaims] = Depends(get_current_user)
) -> UserClaims:
    """Require authenticated user."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return current_user


def require_permissions(
    *permissions: Permission
) -> Callable[[UserClaims], UserClaims]:
    """Require specific permissions."""
    def dependency(
        current_user: UserClaims = Depends(require_auth)
    ) -> UserClaims:
        missing_perms = set(permissions) - current_user.permissions
        if missing_perms:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permissions: {', '.join(missing_perms)}"
            )
        return current_user
    
    return dependency


def require_roles(
    *roles: Role
) -> Callable[[UserClaims], UserClaims]:
    """Require specific roles."""
    def dependency(
        current_user: UserClaims = Depends(require_auth)
    ) -> UserClaims:
        if not any(role in current_user.roles for role in roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {', '.join(roles)}"
            )
        return current_user
    
    return dependency


def require_tenant_admin(
    current_user: UserClaims = Depends(require_auth)
) -> UserClaims:
    """Require tenant admin role."""
    if not (current_user.has_role(Role.TENANT_ADMIN) or current_user.has_role(Role.SUPER_ADMIN)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant admin access required"
        )
    return current_user


def require_super_admin(
    current_user: UserClaims = Depends(require_auth)
) -> UserClaims:
    """Require super admin role."""
    if not current_user.has_role(Role.SUPER_ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin access required"
        )
    return current_user


# RBAC Decorator for route functions
def rbac(
    permissions: Optional[List[Permission]] = None,
    roles: Optional[List[Role]] = None,
    require_tenant_admin: bool = False,
    require_super_admin: bool = False
):
    """RBAC decorator for route functions.
    
    Usage:
        @app.get("/sensitive")
        @rbac(permissions=[Permission.EVIDENCE_READ])
        async def get_evidence():
            ...
            
        @app.post("/admin")
        @rbac(require_tenant_admin=True)
        async def admin_action():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user from request state (set by get_current_user)
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request or not hasattr(request.state, 'user'):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            user: UserClaims = request.state.user
            
            # Check super admin requirement
            if require_super_admin and not user.has_role(Role.SUPER_ADMIN):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Super admin access required"
                )
            
            # Check tenant admin requirement
            if require_tenant_admin and not (
                user.has_role(Role.TENANT_ADMIN) or user.has_role(Role.SUPER_ADMIN)
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Tenant admin access required"
                )
            
            # Check role requirements
            if roles and not any(user.has_role(role) for role in roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Requires one of roles: {', '.join(roles)}"
                )
            
            # Check permission requirements
            if permissions:
                missing_perms = set(permissions) - user.permissions
                if missing_perms:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Missing permissions: {', '.join(missing_perms)}"
                    )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Backwards compatibility shims
async def require_user() -> Optional[UserClaims]:
    """Legacy compatibility shim."""
    return None  # Will be replaced by proper dependency injection


async def require_reader() -> Optional[UserClaims]:
    """Legacy compatibility shim."""
    return None  # Will be replaced by proper dependency injection