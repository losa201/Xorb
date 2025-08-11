"""
RBAC Middleware
Production-ready Role-Based Access Control middleware for FastAPI
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from uuid import UUID
from datetime import datetime

from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware

from ..services.rbac_service import RBACService, RBACContext
from ..services.interfaces import AuthenticationService
from ..container import get_container
from ..core.logging import get_logger


class RBACMiddleware(BaseHTTPMiddleware):
    """RBAC middleware for automatic permission checking"""
    
    def __init__(self, app, exclude_paths: Optional[List[str]] = None):
        super().__init__(app)
        self.logger = get_logger(__name__)
        self.exclude_paths = exclude_paths or [
            '/health',
            '/readiness', 
            '/docs',
            '/openapi.json',
            '/api/v1/auth/token',
            '/api/v1/auth/logout'
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through RBAC middleware"""
        
        # Skip middleware for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        try:
            # Create RBAC context from request
            rbac_context = await self._create_rbac_context(request)
            
            # Store context in request state for use by decorators
            request.state.rbac_context = rbac_context
            
            # Continue processing
            response = await call_next(request)
            
            # Add security headers
            self._add_security_headers(response)
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"RBAC middleware error: {str(e)}")
            # Fail secure - reject request on error
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authorization system error"
            )
    
    async def _create_rbac_context(self, request: Request) -> Optional[RBACContext]:
        """Create RBAC context from request"""
        try:
            # Extract user information from request state (set by auth middleware)
            if not hasattr(request.state, 'user') or not request.state.user:
                return None
            
            user = request.state.user
            
            return RBACContext(
                user_id=UUID(user.user_id) if isinstance(user.user_id, str) else user.user_id,
                tenant_id=UUID(user.tenant_id) if user.tenant_id and isinstance(user.tenant_id, str) else user.tenant_id,
                session_id=getattr(user, 'session_id', None),
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get('User-Agent'),
                metadata={
                    'method': request.method,
                    'path': request.url.path,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Error creating RBAC context: {str(e)}")
            return None
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response"""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"