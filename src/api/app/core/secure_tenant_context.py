"""
Secure Tenant Context Manager
Production-grade tenant isolation enforcement for XORB platform
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Set
from uuid import UUID
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

from fastapi import HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select
from sqlalchemy.exc import SQLAlchemyError

from ..domain.tenant_entities import Tenant
from ..core.logging import get_logger
from ..auth.models import UserClaims


class TenantContextViolationType(str, Enum):
    """Types of tenant context violations"""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MISSING_CONTEXT = "missing_context"
    INVALID_TENANT = "invalid_tenant"
    CROSS_TENANT_ATTEMPT = "cross_tenant_attempt"
    RLS_FAILURE = "rls_failure"
    HEADER_MANIPULATION = "header_manipulation"


@dataclass
class TenantSecurityEvent:
    """Security event for tenant context violations"""
    violation_type: TenantContextViolationType
    user_id: Optional[str] = None
    tenant_id: Optional[UUID] = None
    attempted_tenant: Optional[UUID] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TenantContext:
    """Secure tenant context container"""
    tenant_id: UUID
    user_id: str
    session_id: Optional[str] = None
    validated_at: datetime = field(default_factory=datetime.utcnow)
    permissions: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self, max_age_minutes: int = 30) -> bool:
        """Check if context is still valid"""
        age = datetime.utcnow() - self.validated_at
        return age < timedelta(minutes=max_age_minutes)
    
    def refresh(self):
        """Refresh validation timestamp"""
        self.validated_at = datetime.utcnow()


class SecureTenantContextManager:
    """
    Production-grade tenant context manager with security enforcement
    
    Features:
    - Mandatory tenant validation for all operations
    - Cross-tenant access prevention
    - Database-level Row Level Security (RLS) enforcement
    - Security event logging and monitoring
    - Context validation and refresh
    - Emergency security controls
    """
    
    def __init__(
        self, 
        db_session_factory,
        cache_service=None,
        logger: Optional[logging.Logger] = None
    ):
        self.db_session_factory = db_session_factory
        self.cache = cache_service
        self.logger = logger or get_logger(__name__)
        
        # Security settings
        self.max_context_age_minutes = 30
        self.enable_strict_validation = True
        self.log_all_violations = True
        self.fail_on_rls_error = True
        
        # Bypass paths (very limited)
        self.bypass_paths = {
            "/health",
            "/readiness", 
            "/metrics",
            "/docs",
            "/openapi.json"
        }
        
        # Security monitoring
        self.violation_events: List[TenantSecurityEvent] = []
        self.suspicious_ips: Set[str] = set()
    
    async def validate_user_tenant_access(
        self, 
        user_claims: UserClaims, 
        requested_tenant_id: UUID,
        session: Optional[AsyncSession] = None
    ) -> bool:
        """
        Validate that user has legitimate access to tenant
        
        Args:
            user_claims: Authenticated user claims
            requested_tenant_id: Tenant being accessed
            session: Optional DB session
            
        Returns:
            bool: True if access is authorized
            
        Raises:
            HTTPException: If access denied
        """
        try:
            # Super admins can access any tenant (with logging)
            if user_claims.is_super_admin():
                self.logger.warning(
                    f"Super admin {user_claims.user_id} accessing tenant {requested_tenant_id}"
                )
                return True
            
            # Check if user's primary tenant matches
            if user_claims.tenant_id == str(requested_tenant_id):
                return True
            
            # Check database for explicit tenant access
            if session is None:
                async with self.db_session_factory() as session:
                    return await self._check_tenant_membership(session, user_claims, requested_tenant_id)
            else:
                return await self._check_tenant_membership(session, user_claims, requested_tenant_id)
                
        except Exception as e:
            self.logger.error(f"Tenant validation error: {e}")
            await self._log_security_event(
                TenantSecurityEvent(
                    violation_type=TenantContextViolationType.UNAUTHORIZED_ACCESS,
                    user_id=user_claims.user_id,
                    tenant_id=UUID(user_claims.tenant_id) if user_claims.tenant_id else None,
                    attempted_tenant=requested_tenant_id,
                    details={"error": str(e)}
                )
            )
            return False
    
    async def _check_tenant_membership(
        self, 
        session: AsyncSession, 
        user_claims: UserClaims, 
        tenant_id: UUID
    ) -> bool:
        """Check if user is member of tenant via database"""
        try:
            # Use parameterized query to prevent injection
            result = await session.execute(
                text("""
                    SELECT 1 FROM tenant_users 
                    WHERE user_id = :user_id 
                    AND tenant_id = :tenant_id 
                    AND is_active = true
                """),
                {
                    "user_id": user_claims.user_id,
                    "tenant_id": str(tenant_id)
                }
            )
            
            return result.scalar() is not None
            
        except SQLAlchemyError as e:
            self.logger.error(f"Database error checking tenant membership: {e}")
            return False
    
    async def establish_secure_context(
        self, 
        request: Request, 
        user_claims: UserClaims
    ) -> TenantContext:
        """
        Establish secure tenant context for request
        
        Args:
            request: FastAPI request object
            user_claims: Authenticated user claims
            
        Returns:
            TenantContext: Validated tenant context
            
        Raises:
            HTTPException: If context cannot be established securely
        """
        try:
            # Determine tenant ID - NEVER trust headers
            tenant_id = None
            
            # Primary source: user claims
            if user_claims.tenant_id:
                tenant_id = UUID(user_claims.tenant_id)
            
            # Secondary: extract from URL path if tenant-scoped endpoint
            if not tenant_id:
                tenant_id = await self._extract_tenant_from_path(request.url.path)
            
            if not tenant_id:
                await self._log_security_event(
                    TenantSecurityEvent(
                        violation_type=TenantContextViolationType.MISSING_CONTEXT,
                        user_id=user_claims.user_id,
                        ip_address=request.client.host if request.client else None,
                        user_agent=request.headers.get('User-Agent'),
                        endpoint=request.url.path
                    )
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Tenant context required but not available"
                )
            
            # Validate access
            if not await self.validate_user_tenant_access(user_claims, tenant_id):
                await self._log_security_event(
                    TenantSecurityEvent(
                        violation_type=TenantContextViolationType.UNAUTHORIZED_ACCESS,
                        user_id=user_claims.user_id,
                        tenant_id=UUID(user_claims.tenant_id) if user_claims.tenant_id else None,
                        attempted_tenant=tenant_id,
                        ip_address=request.client.host if request.client else None,
                        user_agent=request.headers.get('User-Agent'),
                        endpoint=request.url.path
                    )
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to requested tenant"
                )
            
            # Create secure context
            context = TenantContext(
                tenant_id=tenant_id,
                user_id=user_claims.user_id,
                session_id=getattr(request.state, 'session_id', None),
                permissions=set(user_claims.permissions) if user_claims.permissions else set(),
                metadata={
                    'ip_address': request.client.host if request.client else None,
                    'user_agent': request.headers.get('User-Agent'),
                    'endpoint': request.url.path,
                    'method': request.method
                }
            )
            
            self.logger.debug(
                f"Established secure tenant context: user={user_claims.user_id}, tenant={tenant_id}"
            )
            
            return context
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to establish tenant context: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to establish secure tenant context"
            )
    
    async def _extract_tenant_from_path(self, path: str) -> Optional[UUID]:
        """Extract tenant ID from URL path if tenant-scoped endpoint"""
        # Look for patterns like /api/v1/tenants/{tenant_id}/...
        path_parts = path.split('/')
        for i, part in enumerate(path_parts):
            if part == 'tenants' and i + 1 < len(path_parts):
                try:
                    return UUID(path_parts[i + 1])
                except ValueError:
                    continue
        return None
    
    @asynccontextmanager
    async def secure_database_session(
        self, 
        context: TenantContext,
        read_only: bool = False
    ):
        """
        Create database session with enforced tenant context
        
        Args:
            context: Validated tenant context
            read_only: Whether session should be read-only
            
        Yields:
            AsyncSession: Database session with tenant context set
        """
        if not context.is_valid(self.max_context_age_minutes):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Tenant context expired, re-authentication required"
            )
        
        async with self.db_session_factory() as session:
            try:
                # Set tenant context in database session
                await self._set_database_tenant_context(session, context)
                
                # Verify RLS is enforced
                if self.fail_on_rls_error:
                    await self._verify_rls_enforcement(session, context.tenant_id)
                
                yield session
                
            except Exception as e:
                self.logger.error(f"Database session error: {e}")
                await self._log_security_event(
                    TenantSecurityEvent(
                        violation_type=TenantContextViolationType.RLS_FAILURE,
                        user_id=context.user_id,
                        tenant_id=context.tenant_id,
                        details={"error": str(e)}
                    )
                )
                raise
    
    async def _set_database_tenant_context(
        self, 
        session: AsyncSession, 
        context: TenantContext
    ) -> None:
        """Set tenant context variables in database session"""
        try:
            # Set tenant context for RLS
            await session.execute(
                text("SELECT set_config('app.tenant_id', :tenant_id, true)"),
                {"tenant_id": str(context.tenant_id)}
            )
            
            # Set user context
            await session.execute(
                text("SELECT set_config('app.user_id', :user_id, true)"),
                {"user_id": context.user_id}
            )
            
            # Set session context
            if context.session_id:
                await session.execute(
                    text("SELECT set_config('app.session_id', :session_id, true)"),
                    {"session_id": context.session_id}
                )
            
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to set database tenant context: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to set secure database context"
            )
    
    async def _verify_rls_enforcement(
        self, 
        session: AsyncSession, 
        tenant_id: UUID
    ) -> None:
        """Verify that Row Level Security is properly enforced"""
        try:
            # Check if RLS is enabled on key tables
            result = await session.execute(
                text("""
                    SELECT tablename, rowsecurity 
                    FROM pg_tables t
                    JOIN pg_class c ON c.relname = t.tablename
                    WHERE t.schemaname = 'public' 
                    AND t.tablename IN ('tenants', 'tenant_users', 'findings', 'evidence')
                    AND c.relrowsecurity = false
                """)
            )
            
            unprotected_tables = result.fetchall()
            if unprotected_tables:
                table_names = [row[0] for row in unprotected_tables]
                self.logger.error(f"RLS not enabled on tables: {table_names}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database security configuration error"
                )
            
            # Verify tenant context is set
            context_check = await session.execute(
                text("SELECT current_setting('app.tenant_id', true)")
            )
            current_tenant = context_check.scalar()
            
            if not current_tenant or current_tenant != str(tenant_id):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database tenant context verification failed"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"RLS verification failed: {e}")
            if self.fail_on_rls_error:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database security verification failed"
                )
    
    async def _log_security_event(self, event: TenantSecurityEvent) -> None:
        """Log security event for monitoring and alerting"""
        self.violation_events.append(event)
        
        # Log to application logger
        self.logger.warning(
            f"Tenant security violation: {event.violation_type.value} "
            f"user={event.user_id} tenant={event.tenant_id} "
            f"attempted={event.attempted_tenant} ip={event.ip_address}"
        )
        
        # Track suspicious IPs
        if event.ip_address and event.violation_type in [
            TenantContextViolationType.UNAUTHORIZED_ACCESS,
            TenantContextViolationType.CROSS_TENANT_ATTEMPT
        ]:
            self.suspicious_ips.add(event.ip_address)
        
        # Store in cache for monitoring
        if self.cache:
            try:
                await self.cache.lpush(
                    "security:tenant_violations",
                    event.__dict__,
                    expire=86400  # 24 hours
                )
            except Exception as e:
                self.logger.error(f"Failed to cache security event: {e}")
    
    async def get_recent_violations(
        self, 
        hours: int = 24
    ) -> List[TenantSecurityEvent]:
        """Get recent tenant security violations"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            event for event in self.violation_events 
            if event.timestamp >= cutoff
        ]
    
    def is_ip_suspicious(self, ip_address: str) -> bool:
        """Check if IP address has been flagged as suspicious"""
        return ip_address in self.suspicious_ips
    
    async def emergency_disable_tenant(
        self, 
        tenant_id: UUID, 
        reason: str,
        admin_user_id: str
    ) -> None:
        """Emergency disable tenant access"""
        self.logger.critical(
            f"EMERGENCY: Disabling tenant {tenant_id} - {reason} - by {admin_user_id}"
        )
        
        # This would integrate with tenant service to suspend the tenant
        # Implementation depends on tenant management system
        pass


# Global instance for dependency injection
_tenant_context_manager: Optional[SecureTenantContextManager] = None


def get_tenant_context_manager() -> SecureTenantContextManager:
    """Get singleton tenant context manager"""
    global _tenant_context_manager
    if _tenant_context_manager is None:
        raise RuntimeError("Tenant context manager not initialized")
    return _tenant_context_manager


def initialize_tenant_context_manager(
    db_session_factory,
    cache_service=None,
    logger=None
) -> SecureTenantContextManager:
    """Initialize global tenant context manager"""
    global _tenant_context_manager
    _tenant_context_manager = SecureTenantContextManager(
        db_session_factory=db_session_factory,
        cache_service=cache_service,
        logger=logger
    )
    return _tenant_context_manager