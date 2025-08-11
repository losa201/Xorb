"""Tenant context middleware for multi-tenant isolation."""
import logging
from typing import Optional
from uuid import UUID

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

try:
    from ..auth.models import UserClaims
except ImportError:
    # Fallback UserClaims
    class UserClaims:
        def __init__(self, user_id: str = "anonymous", tenant_id: str = None):
            self.user_id = user_id
            self.tenant_id = tenant_id

try:
    from ..infrastructure.database import get_async_session
except ImportError:
    def get_async_session():
        return None


logger = logging.getLogger(__name__)


class TenantContextMiddleware(BaseHTTPMiddleware):
    """Middleware to set tenant context for database operations."""

    BYPASS_PATHS = {
        "/health",
        "/readiness", 
        "/metrics",
        "/docs",
        "/openapi.json",
        "/auth/login",
        "/auth/callback",
        "/auth/logout"
    }

    async def dispatch(self, request: Request, call_next):
        """Set tenant context and call next middleware."""
        # Skip tenant context for bypass paths
        if any(request.url.path.startswith(path) for path in self.BYPASS_PATHS):
            return await call_next(request)

        tenant_id = None
        
        try:
            # Extract tenant from user claims if authenticated
            if hasattr(request.state, 'user') and request.state.user:
                user: UserClaims = request.state.user
                tenant_id = user.tenant_id
            else:
                # Try to extract from headers (for service-to-service calls)
                tenant_header = request.headers.get("X-Tenant-ID")
                if tenant_header:
                    tenant_id = UUID(tenant_header)

            # Set tenant context in request state
            request.state.tenant_id = tenant_id

            # Set up database session with tenant context
            if tenant_id:
                await self._set_database_tenant_context(tenant_id)

            response = await call_next(request)
            return response

        except Exception as e:
            logger.error(f"Tenant context middleware error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Tenant context error"
            )

    async def _set_database_tenant_context(self, tenant_id: UUID) -> None:
        """Set tenant context in database session."""
        try:
            # Get database session
            async with get_async_session() as session:
                # Set the app.tenant_id session variable for RLS
                await session.execute(
                    "SELECT set_config('app.tenant_id', :tenant_id, false)",
                    {"tenant_id": str(tenant_id)}
                )
                await session.commit()
                
        except Exception as e:
            logger.error(f"Failed to set database tenant context: {e}")
            # Don't fail the request, but log the error
            pass


def get_current_tenant_id(request: Request) -> Optional[UUID]:
    """Get current tenant ID from request state."""
    return getattr(request.state, 'tenant_id', None)


def require_tenant_context(request: Request) -> UUID:
    """Require tenant context to be set."""
    tenant_id = get_current_tenant_id(request)
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context required"
        )
    return tenant_id