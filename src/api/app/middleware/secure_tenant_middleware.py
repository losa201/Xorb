"""
Secure Tenant Context Middleware
Production-grade middleware for tenant isolation enforcement
"""

import logging
from typing import Optional, Callable
from uuid import UUID

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..core.secure_tenant_context import (
    SecureTenantContextManager, 
    TenantContext, 
    TenantSecurityEvent,
    TenantContextViolationType,
    get_tenant_context_manager
)
from ..auth.models import UserClaims
from ..core.logging import get_logger


logger = get_logger(__name__)


class SecureTenantMiddleware(BaseHTTPMiddleware):
    """
    Secure tenant context middleware with enhanced security controls
    
    Features:
    - Mandatory tenant validation for all protected endpoints
    - Prevention of header-based tenant switching (security vulnerability)
    - Comprehensive security event logging
    - Emergency security controls
    - Performance monitoring
    """
    
    def __init__(self, app, tenant_manager: Optional[SecureTenantContextManager] = None):
        super().__init__(app)
        self.tenant_manager = tenant_manager or get_tenant_context_manager()
        
        # Paths that bypass tenant context (very limited)
        self.bypass_paths = {
            "/health",
            "/readiness", 
            "/metrics",
            "/docs",
            "/openapi.json",
            "/auth/login",
            "/auth/callback",
            "/auth/logout",
            "/auth/refresh"
        }
        
        # Paths that require tenant context
        self.protected_patterns = {
            "/api/v1/ptaas",
            "/api/v1/intelligence", 
            "/api/v1/findings",
            "/api/v1/evidence",
            "/api/v1/scans",
            "/api/v1/tenants"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with secure tenant context enforcement
        
        Args:
            request: FastAPI request object
            call_next: Next middleware in chain
            
        Returns:
            Response: HTTP response
        """
        start_time = logger.performance_timer()
        
        try:
            # Check if path should be bypassed
            if self._should_bypass_tenant_check(request.url.path):
                response = await call_next(request)
                logger.performance_metric("tenant_middleware_bypass", start_time)
                return response
            
            # Detect suspicious header manipulation attempts
            await self._detect_header_manipulation(request)
            
            # Get authenticated user (must be set by auth middleware)
            user_claims = self._get_authenticated_user(request)
            if not user_claims:
                logger.warning(f"Unauthenticated request to protected path: {request.url.path}")
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Authentication required for tenant-scoped resources"}
                )
            
            # Establish secure tenant context
            tenant_context = await self.tenant_manager.establish_secure_context(request, user_claims)
            
            # Store context in request state
            request.state.tenant_context = tenant_context
            request.state.tenant_id = tenant_context.tenant_id
            
            # Log successful context establishment
            logger.debug(
                f"Secure tenant context established: user={user_claims.user_id}, "
                f"tenant={tenant_context.tenant_id}, endpoint={request.url.path}"
            )
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            self._add_security_headers(response, tenant_context)
            
            logger.performance_metric("tenant_middleware_success", start_time)
            return response
            
        except HTTPException as e:
            # Log HTTP exceptions for security monitoring
            await self._log_request_error(request, e)
            logger.performance_metric("tenant_middleware_error", start_time)
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail}
            )
            
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Tenant middleware unexpected error: {e}")
            await self._log_request_error(request, e)
            logger.performance_metric("tenant_middleware_exception", start_time)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal tenant context error"}
            )
    
    def _should_bypass_tenant_check(self, path: str) -> bool:
        """Check if path should bypass tenant context enforcement"""
        # Exact match for bypass paths
        if path in self.bypass_paths:
            return True
        
        # Check if path starts with bypass pattern
        for bypass_path in self.bypass_paths:
            if path.startswith(bypass_path):
                return True
        
        return False
    
    def _requires_tenant_context(self, path: str) -> bool:
        """Check if path requires tenant context"""
        for pattern in self.protected_patterns:
            if path.startswith(pattern):
                return True
        return False
    
    async def _detect_header_manipulation(self, request: Request) -> None:
        """
        Detect suspicious header manipulation attempts
        
        This addresses the critical vulnerability where attackers could
        set X-Tenant-ID headers to bypass tenant isolation.
        """
        suspicious_headers = [
            "X-Tenant-ID",
            "X-Tenant",
            "Tenant-ID", 
            "Tenant",
            "X-Organization-ID",
            "Organization-ID"
        ]
        
        for header in suspicious_headers:
            if header in request.headers:
                # Log security violation
                await self.tenant_manager._log_security_event(
                    TenantSecurityEvent(
                        violation_type=TenantContextViolationType.HEADER_MANIPULATION,
                        ip_address=request.client.host if request.client else None,
                        user_agent=request.headers.get('User-Agent'),
                        endpoint=request.url.path,
                        details={
                            "suspicious_header": header,
                            "header_value": request.headers.get(header),
                            "all_headers": dict(request.headers)
                        }
                    )
                )
                
                logger.warning(
                    f"Suspicious header manipulation detected: {header}={request.headers.get(header)} "
                    f"from IP {request.client.host if request.client else 'unknown'}"
                )
                
                # In production, you might want to block such requests
                if self.tenant_manager.enable_strict_validation:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Tenant context manipulation via headers is not permitted"
                    )
    
    def _get_authenticated_user(self, request: Request) -> Optional[UserClaims]:
        """Get authenticated user from request state"""
        if hasattr(request.state, 'user'):
            return request.state.user
        
        # Fallback: try to get from headers (for service-to-service)
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            # This would need to integrate with your JWT validation
            # For now, return None to require proper auth middleware
            pass
        
        return None
    
    def _add_security_headers(self, response: Response, context: TenantContext) -> None:
        """Add security headers to response"""
        # Add tenant context hash for client validation
        tenant_hash = hash(f"{context.tenant_id}:{context.user_id}:{context.validated_at}")
        response.headers["X-Tenant-Context-Hash"] = str(abs(tenant_hash))
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Add tenant-specific CSP if needed
        # response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    async def _log_request_error(self, request: Request, error: Exception) -> None:
        """Log request error for security monitoring"""
        error_details = {
            "method": request.method,
            "path": request.url.path,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "ip_address": request.client.host if request.client else None,
            "user_agent": request.headers.get('User-Agent')
        }
        
        if hasattr(request.state, 'user') and request.state.user:
            error_details["user_id"] = request.state.user.user_id
            error_details["user_tenant"] = request.state.user.tenant_id
        
        logger.error(f"Tenant middleware error: {error_details}")


class TenantContextDependency:
    """
    FastAPI dependency for accessing secure tenant context
    
    Usage:
        @router.get("/protected")
        async def protected_endpoint(
            tenant_context: TenantContext = Depends(require_tenant_context)
        ):
            # Access tenant-scoped data using tenant_context.tenant_id
            pass
    """
    
    def __init__(self, required: bool = True):
        self.required = required
    
    async def __call__(self, request: Request) -> Optional[TenantContext]:
        if hasattr(request.state, 'tenant_context'):
            context: TenantContext = request.state.tenant_context
            
            # Verify context is still valid
            if not context.is_valid():
                if self.required:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Tenant context expired"
                    )
                return None
            
            return context
        
        if self.required:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant context not available"
            )
        
        return None


# Dependency instances
require_tenant_context = TenantContextDependency(required=True)
optional_tenant_context = TenantContextDependency(required=False)


def get_tenant_id(request: Request) -> UUID:
    """
    Get tenant ID from secure context
    
    Args:
        request: FastAPI request object
        
    Returns:
        UUID: Validated tenant ID
        
    Raises:
        HTTPException: If tenant context not available
    """
    if hasattr(request.state, 'tenant_context'):
        context: TenantContext = request.state.tenant_context
        if context.is_valid():
            return context.tenant_id
    
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Valid tenant context required"
    )


def get_user_id(request: Request) -> str:
    """
    Get user ID from secure context
    
    Args:
        request: FastAPI request object
        
    Returns:
        str: User ID
        
    Raises:
        HTTPException: If tenant context not available
    """
    if hasattr(request.state, 'tenant_context'):
        context: TenantContext = request.state.tenant_context
        if context.is_valid():
            return context.user_id
    
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Valid tenant context required"
    )


async def validate_tenant_access(
    request: Request, 
    tenant_id: UUID
) -> bool:
    """
    Validate that current user can access specified tenant
    
    Args:
        request: FastAPI request object
        tenant_id: Tenant to validate access for
        
    Returns:
        bool: True if access allowed
    """
    if hasattr(request.state, 'tenant_context'):
        context: TenantContext = request.state.tenant_context
        if context.is_valid() and context.tenant_id == tenant_id:
            return True
    
    return False