"""
Dependency Injection
FastAPI dependencies for services and authentication
"""

from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import aioredis

from src.api.services.sso_service import SSOService
from src.api.services.auth_service import AuthService
from src.common.models.auth import User
from src.common.config import get_settings

# Security scheme
security = HTTPBearer()

# Global instances (in production, use proper DI container)
_redis_client: Optional[aioredis.Redis] = None
_sso_service: Optional[SSOService] = None
_auth_service: Optional[AuthService] = None


async def get_redis_client() -> aioredis.Redis:
    """Get Redis client instance"""
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        redis_url = getattr(settings, 'redis_url', 'redis://localhost:6379')
        _redis_client = await aioredis.from_url(redis_url)
    return _redis_client


async def get_sso_service(redis_client: aioredis.Redis = Depends(get_redis_client)) -> SSOService:
    """Get SSO service instance"""
    global _sso_service
    if _sso_service is None:
        settings = get_settings()

        # SSO configuration
        sso_config = {
            "sso_azure": {
                "enabled": getattr(settings, 'azure_sso_enabled', False),
                "client_id": getattr(settings, 'azure_client_id', ''),
                "client_secret": getattr(settings, 'azure_client_secret', ''),
                "tenant": getattr(settings, 'azure_tenant', 'common')
            },
            "sso_google": {
                "enabled": getattr(settings, 'google_sso_enabled', False),
                "client_id": getattr(settings, 'google_client_id', ''),
                "client_secret": getattr(settings, 'google_client_secret', '')
            },
            "sso_okta": {
                "enabled": getattr(settings, 'okta_sso_enabled', False),
                "client_id": getattr(settings, 'okta_client_id', ''),
                "client_secret": getattr(settings, 'okta_client_secret', ''),
                "domain": getattr(settings, 'okta_domain', '')
            },
            "sso_github": {
                "enabled": getattr(settings, 'github_sso_enabled', False),
                "client_id": getattr(settings, 'github_client_id', ''),
                "client_secret": getattr(settings, 'github_client_secret', '')
            }
        }

        _sso_service = SSOService(redis_client, sso_config)
    return _sso_service


async def get_auth_service(redis_client: aioredis.Redis = Depends(get_redis_client)) -> AuthService:
    """Get auth service instance"""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService(redis_client)
    return _auth_service


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> User:
    """Get current authenticated user"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )

    user = await auth_service.validate_jwt_token(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return user


async def get_current_user_optional(
    request: Request,
    auth_service: AuthService = Depends(get_auth_service)
) -> Optional[User]:
    """Get current user if authenticated (optional)"""
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        token = auth_header.split(" ")[1]
        user = await auth_service.validate_jwt_token(token)
        return user
    except Exception:
        return None


def require_permission(permission: str):
    """Dependency to require specific permission"""
    async def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if not current_user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    return permission_checker


def require_role(role: str):
    """Dependency to require specific role"""
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        user_role = current_user.role.value if hasattr(current_user.role, 'value') else current_user.role
        if user_role != role and not current_user.is_admin():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required"
            )
        return current_user
    return role_checker


def require_admin():
    """Dependency to require admin role"""
    return require_role("admin")


async def get_request_context(request: Request) -> Dict[str, Any]:
    """Get request context information"""
    return {
        "ip_address": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("User-Agent", "unknown"),
        "request_id": request.headers.get("X-Request-ID", "unknown"),
        "timestamp": request.state.__dict__.get("start_time"),
        "method": request.method,
        "url": str(request.url)
    }
