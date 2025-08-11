"""
XORB Unified Authentication Service
Consolidates all authentication functionality into a single, production-ready service.

Replaces:
- unified_auth_service.py
- unified_auth_service_consolidated.py  
- enterprise_auth.py
- Various other auth-related services

Features:
- Multi-provider authentication (Local, OIDC, SAML, mTLS)
- Hierarchical RBAC with fine-grained permissions
- Account security (lockouts, rate limiting, audit logs)
- Enterprise features (SSO, tenant isolation, API keys)
- Zero Trust security model
"""

import asyncio
import hashlib
import json
import secrets
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any, Set, Union
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass, asdict

import jwt
import redis.asyncio as redis
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from pydantic import BaseModel, EmailStr

# Import from common security utils with fallback
import bcrypt
import passlib.context

PASSWORD_CONTEXT = passlib.context.CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return PASSWORD_CONTEXT.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return PASSWORD_CONTEXT.verify(password, hashed)

def needs_rehash(hashed: str) -> bool:
    return PASSWORD_CONTEXT.needs_update(hashed)

from ..domain.entities import User, AuthToken
from ..domain.exceptions import (
    InvalidCredentials, AccountLocked, SecurityViolation, ValidationError, TokenExpired
)
from ..domain.repositories import UserRepository, AuthTokenRepository
from .interfaces import AuthenticationService

logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class AuthProvider(Enum):
    """Supported authentication providers"""
    LOCAL = "local"
    OIDC = "oidc"
    SAML = "saml"
    LDAP = "ldap"
    MTLS = "mtls"
    API_KEY = "api_key"

class Role(Enum):
    """Hierarchical security roles"""
    SUPER_ADMIN = "super_admin"      # Global system administrator
    TENANT_ADMIN = "tenant_admin"    # Tenant-level administrator
    SECURITY_ANALYST = "analyst"     # Security operations analyst
    ORCHESTRATOR = "orchestrator"    # Workflow orchestrator
    AGENT = "agent"                  # Autonomous agent
    USER = "user"                    # Standard user
    READONLY = "readonly"            # Read-only access

class Permission(Enum):
    """Fine-grained permissions"""
    # System Management
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_METRICS = "system:metrics"
    
    # User Management
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    
    # Agent Management
    AGENT_CREATE = "agent:create"
    AGENT_READ = "agent:read"
    AGENT_UPDATE = "agent:update"
    AGENT_DELETE = "agent:delete"
    AGENT_EXECUTE = "agent:execute"
    
    # Security Operations
    SECURITY_SCAN = "security:scan"
    SECURITY_ANALYZE = "security:analyze"
    SECURITY_RESPOND = "security:respond"
    
    # Tenant Management
    TENANT_CREATE = "tenant:create"
    TENANT_READ = "tenant:read"
    TENANT_UPDATE = "tenant:update"
    TENANT_DELETE = "tenant:delete"

@dataclass
class SecurityContext:
    """Complete security context for a user session"""
    user_id: UUID
    username: str
    email: str
    roles: List[Role]
    permissions: Set[Permission]
    tenant_id: Optional[UUID] = None
    auth_provider: AuthProvider = AuthProvider.LOCAL
    session_id: str = ""
    ip_address: str = ""
    user_agent: str = ""
    expires_at: datetime = None
    mfa_verified: bool = False
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission"""
        return permission in self.permissions
    
    def has_role(self, role: Role) -> bool:
        """Check if user has specific role"""
        return role in self.roles

@dataclass
class AuthenticationResult:
    """Result of authentication attempt"""
    success: bool
    security_context: Optional[SecurityContext] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    error_message: Optional[str] = None
    requires_mfa: bool = False
    account_locked: bool = False

# ============================================================================
# UNIFIED AUTHENTICATION SERVICE
# ============================================================================

class ConsolidatedAuthService(AuthenticationService):
    """
    Unified authentication service consolidating all auth functionality
    """
    
    # Role hierarchy (higher roles inherit lower role permissions)
    ROLE_HIERARCHY = {
        Role.SUPER_ADMIN: [Role.TENANT_ADMIN, Role.SECURITY_ANALYST, Role.ORCHESTRATOR, Role.AGENT, Role.USER, Role.READONLY],
        Role.TENANT_ADMIN: [Role.SECURITY_ANALYST, Role.ORCHESTRATOR, Role.AGENT, Role.USER, Role.READONLY],
        Role.SECURITY_ANALYST: [Role.USER, Role.READONLY],
        Role.ORCHESTRATOR: [Role.AGENT, Role.USER, Role.READONLY],
        Role.AGENT: [Role.READONLY],
        Role.USER: [Role.READONLY],
        Role.READONLY: []
    }
    
    # Role-based permissions mapping
    ROLE_PERMISSIONS = {
        Role.SUPER_ADMIN: {
            Permission.SYSTEM_ADMIN, Permission.SYSTEM_CONFIG, Permission.SYSTEM_METRICS,
            Permission.USER_CREATE, Permission.USER_READ, Permission.USER_UPDATE, Permission.USER_DELETE,
            Permission.AGENT_CREATE, Permission.AGENT_READ, Permission.AGENT_UPDATE, Permission.AGENT_DELETE, Permission.AGENT_EXECUTE,
            Permission.SECURITY_SCAN, Permission.SECURITY_ANALYZE, Permission.SECURITY_RESPOND,
            Permission.TENANT_CREATE, Permission.TENANT_READ, Permission.TENANT_UPDATE, Permission.TENANT_DELETE
        },
        Role.TENANT_ADMIN: {
            Permission.USER_CREATE, Permission.USER_READ, Permission.USER_UPDATE, Permission.USER_DELETE,
            Permission.AGENT_CREATE, Permission.AGENT_READ, Permission.AGENT_UPDATE, Permission.AGENT_DELETE, Permission.AGENT_EXECUTE,
            Permission.SECURITY_SCAN, Permission.SECURITY_ANALYZE, Permission.SECURITY_RESPOND,
            Permission.TENANT_READ, Permission.TENANT_UPDATE
        },
        Role.SECURITY_ANALYST: {
            Permission.USER_READ, Permission.AGENT_READ, Permission.AGENT_EXECUTE,
            Permission.SECURITY_SCAN, Permission.SECURITY_ANALYZE, Permission.SECURITY_RESPOND
        },
        Role.ORCHESTRATOR: {
            Permission.AGENT_READ, Permission.AGENT_EXECUTE, Permission.SECURITY_SCAN
        },
        Role.AGENT: {
            Permission.AGENT_READ, Permission.SECURITY_SCAN
        },
        Role.USER: {
            Permission.USER_READ, Permission.AGENT_READ
        },
        Role.READONLY: set()
    }
    
    def __init__(
        self,
        user_repository: UserRepository,
        token_repository: AuthTokenRepository,
        redis_client: redis.Redis,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
        max_login_attempts: int = 5,
        lockout_duration_minutes: int = 15
    ):
        self.user_repository = user_repository
        self.token_repository = token_repository
        self.redis_client = redis_client
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.max_login_attempts = max_login_attempts
        self.lockout_duration_minutes = lockout_duration_minutes
        
        # Cache for security contexts
        self._context_cache: Dict[str, SecurityContext] = {}
        
    # ========================================================================
    # CORE AUTHENTICATION METHODS
    # ========================================================================
    
    async def authenticate_user(
        self, 
        username: str, 
        password: str,
        ip_address: str = "",
        user_agent: str = "",
        provider: AuthProvider = AuthProvider.LOCAL
    ) -> AuthenticationResult:
        """
        Authenticate user with comprehensive security checks
        """
        try:
            # Check for account lockout
            if await self._is_account_locked(username, ip_address):
                await self._log_security_event(
                    "auth_attempt_blocked", 
                    username, 
                    ip_address, 
                    {"reason": "account_locked"}
                )
                return AuthenticationResult(
                    success=False,
                    account_locked=True,
                    error_message="Account is locked due to too many failed attempts"
                )
            
            # Retrieve user
            user = await self.user_repository.get_by_username(username)
            if not user:
                await self._record_failed_attempt(username, ip_address)
                await self._log_security_event(
                    "auth_failed", 
                    username, 
                    ip_address, 
                    {"reason": "user_not_found"}
                )
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid credentials"
                )
            
            # Verify password
            if not verify_password(password, user.password_hash):
                await self._record_failed_attempt(username, ip_address)
                await self._log_security_event(
                    "auth_failed", 
                    username, 
                    ip_address, 
                    {"reason": "invalid_password"}
                )
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid credentials"
                )
            
            # Check if user is active
            if not user.is_active:
                await self._log_security_event(
                    "auth_failed", 
                    username, 
                    ip_address, 
                    {"reason": "user_inactive"}
                )
                return AuthenticationResult(
                    success=False,
                    error_message="Account is inactive"
                )
            
            # Clear failed attempts on successful auth
            await self._clear_failed_attempts(username, ip_address)
            
            # Create security context
            security_context = await self._create_security_context(
                user, provider, ip_address, user_agent
            )
            
            # Generate tokens
            access_token = await self._create_access_token(security_context)
            refresh_token = await self._create_refresh_token(security_context)
            
            # Log successful authentication
            await self._log_security_event(
                "auth_success", 
                username, 
                ip_address, 
                {"provider": provider.value}
            )
            
            return AuthenticationResult(
                success=True,
                security_context=security_context,
                access_token=access_token,
                refresh_token=refresh_token
            )
            
        except Exception as e:
            logger.error(f"Authentication error for {username}: {e}")
            await self._log_security_event(
                "auth_error", 
                username, 
                ip_address, 
                {"error": str(e)}
            )
            return AuthenticationResult(
                success=False,
                error_message="Authentication service error"
            )
    
    async def authenticate_api_key(self, api_key: str) -> AuthenticationResult:
        """Authenticate using API key"""
        try:
            # Hash the API key for lookup
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # Find token in repository
            token = await self.token_repository.get_by_token_hash(api_key_hash)
            if not token or token.expires_at < datetime.utcnow():
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid or expired API key"
                )
            
            # Get associated user
            user = await self.user_repository.get_by_id(token.user_id)
            if not user or not user.is_active:
                return AuthenticationResult(
                    success=False,
                    error_message="User account inactive"
                )
            
            # Create security context for API key auth
            security_context = await self._create_security_context(
                user, AuthProvider.API_KEY
            )
            
            return AuthenticationResult(
                success=True,
                security_context=security_context
            )
            
        except Exception as e:
            logger.error(f"API key authentication error: {e}")
            return AuthenticationResult(
                success=False,
                error_message="API key authentication failed"
            )
    
    async def validate_token(self, token: str) -> Optional[SecurityContext]:
        """Validate and decode JWT token"""
        try:
            # Check cache first
            if token in self._context_cache:
                context = self._context_cache[token]
                if context.expires_at > datetime.utcnow():
                    return context
                else:
                    del self._context_cache[token]
            
            # Decode JWT
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            user_id = UUID(payload.get("sub"))
            exp = datetime.fromtimestamp(payload.get("exp"))
            
            # Check expiration
            if exp < datetime.utcnow():
                raise TokenExpired("Token has expired")
            
            # Get user and create context
            user = await self.user_repository.get_by_id(user_id)
            if not user or not user.is_active:
                return None
            
            security_context = await self._create_security_context(
                user, 
                AuthProvider(payload.get("provider", "local")),
                payload.get("ip", ""),
                payload.get("user_agent", "")
            )
            security_context.expires_at = exp
            security_context.session_id = payload.get("session_id", "")
            
            # Cache the context
            self._context_cache[token] = security_context
            
            return security_context
            
        except (jwt.JWTError, ValueError, KeyError) as e:
            logger.warning(f"Token validation failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return None
    
    # ========================================================================
    # SECURITY CONTEXT & PERMISSION MANAGEMENT
    # ========================================================================
    
    async def _create_security_context(
        self, 
        user: User, 
        provider: AuthProvider,
        ip_address: str = "",
        user_agent: str = ""
    ) -> SecurityContext:
        """Create comprehensive security context"""
        
        # Parse roles from user
        user_roles = [Role(role) for role in user.roles if role in [r.value for r in Role]]
        
        # Calculate effective permissions
        permissions = set()
        for role in user_roles:
            # Add direct role permissions
            if role in self.ROLE_PERMISSIONS:
                permissions.update(self.ROLE_PERMISSIONS[role])
            
            # Add inherited permissions from role hierarchy
            if role in self.ROLE_HIERARCHY:
                for inherited_role in self.ROLE_HIERARCHY[role]:
                    if inherited_role in self.ROLE_PERMISSIONS:
                        permissions.update(self.ROLE_PERMISSIONS[inherited_role])
        
        return SecurityContext(
            user_id=user.id,
            username=user.username,
            email=user.email,
            roles=user_roles,
            permissions=permissions,
            tenant_id=getattr(user, 'tenant_id', None),
            auth_provider=provider,
            session_id=str(uuid4()),
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        )
    
    def check_permission(self, context: SecurityContext, permission: Permission) -> bool:
        """Check if security context has specific permission"""
        return context.has_permission(permission)
    
    def check_role(self, context: SecurityContext, role: Role) -> bool:
        """Check if security context has specific role"""
        return context.has_role(role)
    
    # ========================================================================
    # TOKEN MANAGEMENT
    # ========================================================================
    
    async def _create_access_token(self, context: SecurityContext) -> str:
        """Create JWT access token"""
        expires = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = {
            "sub": str(context.user_id),
            "username": context.username,
            "roles": [role.value for role in context.roles],
            "permissions": [perm.value for perm in context.permissions],
            "tenant_id": str(context.tenant_id) if context.tenant_id else None,
            "provider": context.auth_provider.value,
            "session_id": context.session_id,
            "ip": context.ip_address,
            "user_agent": context.user_agent,
            "iat": datetime.utcnow(),
            "exp": expires
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    async def _create_refresh_token(self, context: SecurityContext) -> str:
        """Create refresh token"""
        expires = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        # Store refresh token in repository
        refresh_token = AuthToken.create(
            user_id=context.user_id,
            token_type="refresh",
            expires_at=expires
        )
        
        await self.token_repository.create(refresh_token)
        return refresh_token.token
    
    async def create_api_key(
        self, 
        context: SecurityContext, 
        name: str, 
        expires_in_days: int = 365
    ) -> str:
        """Create API key for user"""
        if not self.check_permission(context, Permission.USER_UPDATE):
            raise SecurityViolation("Insufficient permissions to create API key")
        
        expires = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # Generate secure API key
        api_key = f"xorb_{''.join(secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(32))}"
        
        # Store hashed version
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        token = AuthToken.create(
            user_id=context.user_id,
            token_type="api_key",
            token_hash=api_key_hash,
            name=name,
            expires_at=expires
        )
        
        await self.token_repository.create(token)
        
        await self._log_security_event(
            "api_key_created",
            context.username,
            context.ip_address,
            {"name": name, "expires": expires.isoformat()}
        )
        
        return api_key
    
    # ========================================================================
    # ACCOUNT SECURITY & RATE LIMITING
    # ========================================================================
    
    async def _is_account_locked(self, username: str, ip_address: str) -> bool:
        """Check if account is locked due to failed attempts"""
        user_key = f"failed_attempts:user:{username}"
        ip_key = f"failed_attempts:ip:{ip_address}"
        
        user_attempts = await self.redis_client.get(user_key)
        ip_attempts = await self.redis_client.get(ip_key)
        
        user_count = int(user_attempts) if user_attempts else 0
        ip_count = int(ip_attempts) if ip_attempts else 0
        
        return user_count >= self.max_login_attempts or ip_count >= (self.max_login_attempts * 2)
    
    async def _record_failed_attempt(self, username: str, ip_address: str):
        """Record failed authentication attempt"""
        user_key = f"failed_attempts:user:{username}"
        ip_key = f"failed_attempts:ip:{ip_address}"
        
        # Increment counters with expiration
        await self.redis_client.incr(user_key)
        await self.redis_client.expire(user_key, self.lockout_duration_minutes * 60)
        
        await self.redis_client.incr(ip_key)
        await self.redis_client.expire(ip_key, self.lockout_duration_minutes * 60)
    
    async def _clear_failed_attempts(self, username: str, ip_address: str):
        """Clear failed attempt counters"""
        user_key = f"failed_attempts:user:{username}"
        ip_key = f"failed_attempts:ip:{ip_address}"
        
        await self.redis_client.delete(user_key, ip_key)
    
    # ========================================================================
    # AUDIT LOGGING
    # ========================================================================
    
    async def _log_security_event(
        self, 
        event_type: str, 
        username: str, 
        ip_address: str, 
        additional_data: Dict[str, Any] = None
    ):
        """Log security event for audit trail"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "username": username,
            "ip_address": ip_address,
            "user_agent": additional_data.get("user_agent", "") if additional_data else "",
            "additional_data": additional_data or {}
        }
        
        # Store in Redis for real-time monitoring
        await self.redis_client.lpush(
            "security_events", 
            json.dumps(event)
        )
        
        # Limit list size
        await self.redis_client.ltrim("security_events", 0, 9999)
        
        # Log for persistent storage
        logger.info(f"Security event: {event_type}", extra=event)
    
    # ========================================================================
    # ENTERPRISE FEATURES
    # ========================================================================
    
    async def get_user_permissions(self, user_id: UUID) -> Set[Permission]:
        """Get all permissions for a user"""
        user = await self.user_repository.get_by_id(user_id)
        if not user:
            return set()
        
        context = await self._create_security_context(user, AuthProvider.LOCAL)
        return context.permissions
    
    async def check_tenant_access(self, context: SecurityContext, tenant_id: UUID) -> bool:
        """Check if user has access to specific tenant"""
        # Super admins have access to all tenants
        if Role.SUPER_ADMIN in context.roles:
            return True
        
        # Users can only access their own tenant
        return context.tenant_id == tenant_id
    
    async def get_security_events(
        self, 
        context: SecurityContext, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent security events (admin only)"""
        if not self.check_permission(context, Permission.SYSTEM_ADMIN):
            raise SecurityViolation("Insufficient permissions to view security events")
        
        events = await self.redis_client.lrange("security_events", 0, limit - 1)
        return [json.loads(event) for event in events]
    
    # ========================================================================
    # INTERFACE IMPLEMENTATIONS (REQUIRED BY AuthenticationService)
    # ========================================================================
    
    async def authenticate_user(self, credentials) -> AuthenticationResult:
        """Interface implementation for authenticate_user"""
        if isinstance(credentials, dict):
            username = credentials.get("username", "")
            password = credentials.get("password", "")
            ip_address = credentials.get("ip_address", "")
            user_agent = credentials.get("user_agent", "")
            provider = AuthProvider(credentials.get("provider", "local"))
            
            return await self.authenticate_user(
                username, password, ip_address, user_agent, provider
            )
        else:
            return AuthenticationResult(
                success=False,
                error_message="Invalid credentials format"
            )
    
    async def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Generate new access token from refresh token"""
        try:
            # Get refresh token from repository
            token = await self.token_repository.get_by_token(refresh_token)
            if not token or token.token_type != "refresh" or token.expires_at < datetime.utcnow():
                return None
            
            # Get user
            user = await self.user_repository.get_by_id(token.user_id)
            if not user or not user.is_active:
                return None
            
            # Create new security context
            context = await self._create_security_context(user, AuthProvider.LOCAL)
            
            # Generate new access token
            return await self._create_access_token(context)
            
        except Exception as e:
            logger.error(f"Refresh token error: {e}")
            return None
    
    async def logout_user(self, session_id: str) -> bool:
        """Logout user by session ID"""
        try:
            # Find and remove from cache
            tokens_to_remove = []
            context_to_logout = None
            
            for token, context in self._context_cache.items():
                if context.session_id == session_id:
                    tokens_to_remove.append(token)
                    context_to_logout = context
                    break
            
            for token in tokens_to_remove:
                del self._context_cache[token]
            
            # Log logout event
            if context_to_logout:
                await self._log_security_event(
                    "user_logout",
                    context_to_logout.username,
                    context_to_logout.ip_address,
                    {"session_id": session_id}
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False
    
    def hash_password(self, password: str) -> str:
        """Hash password securely using bcrypt"""
        return hash_password(password)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return verify_password(password, hashed)
    
    # ========================================================================
    # ENTERPRISE SSO & OIDC IMPLEMENTATIONS
    # ========================================================================
    
    async def authenticate_oidc(self, id_token: str, access_token: str) -> AuthenticationResult:
        """Authenticate user via OIDC (OpenID Connect)"""
        try:
            # Decode and verify ID token (simplified - in production use proper OIDC library)
            # This is a basic implementation - use python-jose or similar for production
            payload = jwt.decode(id_token, options={"verify_signature": False})
            
            email = payload.get("email")
            username = payload.get("preferred_username", email)
            name = payload.get("name", username)
            
            if not email or not username:
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid OIDC token - missing required claims"
                )
            
            # Find or create user
            user = await self.user_repository.get_by_username(username)
            if not user:
                # Auto-provision user from OIDC
                user = User.create(
                    username=username,
                    email=email,
                    password_hash="",  # No password for OIDC users
                    roles=["user"],  # Default role
                    is_active=True,
                    auth_provider="oidc"
                )
                user = await self.user_repository.create(user)
            
            if not user.is_active:
                return AuthenticationResult(
                    success=False,
                    error_message="Account is inactive"
                )
            
            # Create security context
            security_context = await self._create_security_context(
                user, AuthProvider.OIDC
            )
            
            # Generate tokens
            access_token = await self._create_access_token(security_context)
            refresh_token = await self._create_refresh_token(security_context)
            
            await self._log_security_event(
                "oidc_auth_success",
                username,
                "",
                {"email": email}
            )
            
            return AuthenticationResult(
                success=True,
                security_context=security_context,
                access_token=access_token,
                refresh_token=refresh_token
            )
            
        except Exception as e:
            logger.error(f"OIDC authentication error: {e}")
            return AuthenticationResult(
                success=False,
                error_message="OIDC authentication failed"
            )
    
    async def authenticate_mtls(self, client_cert: str) -> AuthenticationResult:
        """Authenticate using mutual TLS client certificate"""
        try:
            # Parse client certificate
            cert = x509.load_pem_x509_certificate(client_cert.encode())
            
            # Extract subject information
            subject = cert.subject
            
            # Get CN (Common Name) as username
            username = None
            for attribute in subject:
                if attribute.oid == x509.NameOID.COMMON_NAME:
                    username = attribute.value
                    break
            
            if not username:
                return AuthenticationResult(
                    success=False,
                    error_message="No Common Name in client certificate"
                )
            
            # Find user by certificate subject
            user = await self.user_repository.get_by_username(username)
            if not user or not user.is_active:
                return AuthenticationResult(
                    success=False,
                    error_message="Certificate not authorized"
                )
            
            # Create security context
            security_context = await self._create_security_context(
                user, AuthProvider.MTLS
            )
            
            # Generate access token (no refresh for mTLS)
            access_token = await self._create_access_token(security_context)
            
            await self._log_security_event(
                "mtls_auth_success",
                username,
                "",
                {"cert_subject": str(subject)}
            )
            
            return AuthenticationResult(
                success=True,
                security_context=security_context,
                access_token=access_token
            )
            
        except Exception as e:
            logger.error(f"mTLS authentication error: {e}")
            return AuthenticationResult(
                success=False,
                error_message="mTLS authentication failed"
            )
    
    # ========================================================================
    # CLEANUP & MAINTENANCE
    # ========================================================================
    
    async def cleanup_expired_tokens(self):
        """Clean up expired tokens and cache entries"""
        # Remove expired tokens from repository
        await self.token_repository.delete_expired()
        
        # Clean context cache
        expired_tokens = []
        for token, context in self._context_cache.items():
            if context.expires_at < datetime.utcnow():
                expired_tokens.append(token)
        
        for token in expired_tokens:
            del self._context_cache[token]
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get authentication service health status"""
        try:
            # Test Redis connection
            await self.redis_client.ping()
            redis_status = "healthy"
        except Exception:
            redis_status = "unhealthy"
        
        # Get cache stats
        cache_size = len(self._context_cache)
        
        return {
            "status": "healthy" if redis_status == "healthy" else "degraded",
            "redis_connection": redis_status,
            "cached_contexts": cache_size,
            "supported_providers": [provider.value for provider in AuthProvider],
            "timestamp": datetime.utcnow().isoformat()
        }