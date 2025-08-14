"""
FastAPI dependencies for clean architecture
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from .container import get_container
from .services.interfaces import AuthenticationService
from .domain.entities import User, Organization
from .domain.exceptions import DomainException

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current authenticated user"""
    try:
        container = get_container()
        auth_service = container.get(AuthenticationService)

        user = await auth_service.validate_token(token)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return user

    except DomainException as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )


async def get_current_organization(
    org_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
) -> Organization:
    """Get current organization - simplified implementation"""

    # In a real implementation, this would:
    # 1. Extract org_id from headers or query params
    # 2. Validate user has access to the organization
    # 3. Return the organization from the repository

    # For now, return a default organization
    return Organization.create(
        name="Default Organization",
        plan_type="Enterprise"
    )


def require_role(required_role: str):
    """Dependency factory for role-based access control"""

    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if not current_user.has_role(required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {required_role}"
            )
        return current_user

    return role_checker


# Convenience dependencies for common roles
require_admin = require_role("admin")
require_user = require_role("user")
require_reader = require_role("reader")


async def get_authentication_service() -> AuthenticationService:
    """Get authentication service instance"""
    container = get_container()
    return container.get(AuthenticationService)


async def get_embedding_service():
    """Get embedding service instance"""
    from .services.interfaces import EmbeddingService
    container = get_container()
    return container.get(EmbeddingService)


async def get_discovery_service():
    """Get discovery service instance"""
    from .services.interfaces import DiscoveryService
    container = get_container()
    return container.get(DiscoveryService)


# Health check dependency that doesn't require authentication
async def health_check_allowed() -> bool:
    """Allow health checks without authentication"""
    return True
