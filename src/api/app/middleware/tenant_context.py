"""
DEPRECATED: Vulnerable tenant context middleware - DO NOT USE
This middleware has been replaced with SecureTenantMiddleware for security reasons.

SECURITY VULNERABILITIES:
- Allows tenant switching via headers (X-Tenant-ID)
- Silent failure on database context errors
- No validation of user-tenant relationships
- Missing security event logging

Use: src/api/app/middleware/secure_tenant_middleware.py instead
"""
import logging
from typing import Optional
from uuid import UUID

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

# Import secure replacement
from .secure_tenant_middleware import SecureTenantMiddleware

logger = logging.getLogger(__name__)

# Issue deprecation warning
logger.warning(
    "SECURITY WARNING: tenant_context.py is deprecated due to security vulnerabilities. "
    "Use secure_tenant_middleware.py instead."
)


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