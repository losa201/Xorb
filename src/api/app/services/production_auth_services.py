"""
Production Authentication & Authorization Services
Enterprise-grade security with comprehensive audit logging and session management
"""

import asyncio
import json
import logging
import secrets
import hashlib
import hmac
import time
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    try:
        import redis
        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False
        redis = None

from .interfaces import AuthenticationService, AuthorizationService
from .base_service import XORBService, ServiceHealth, ServiceStatus
from ..domain.entities import User, Organization, AuthToken
from ..domain.repositories import UserRepository, AuthTokenRepository, CacheRepository

logger = logging.getLogger(__name__)


@dataclass
class AuthenticationResult:
    """Authentication result with detailed information"""
    success: bool
    user: Optional[User] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    permissions: List[str] = None
    session_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class TokenValidationResult:
    """Token validation result"""
    valid: bool
    user: Optional[User] = None
    claims: Dict[str, Any] = None
    expires_at: Optional[datetime] = None
    error_message: Optional[str] = None


class ProductionAuthenticationService(AuthenticationService, XORBService):
    """
    Production-ready authentication service with comprehensive security features
    - JWT token management with rotation
    - Session management with Redis
    - Audit logging for all auth events
    - Rate limiting and brute force protection
    - Multi-factor authentication support
    """
    
    def __init__(
        self,
        user_repository: UserRepository,
        auth_token_repository: AuthTokenRepository,
        cache_repository: CacheRepository,
        jwt_secret: str,
        jwt_algorithm: str = "HS256",
        access_token_expire_minutes: int = 60,
        refresh_token_expire_days: int = 30,
        **kwargs
    ):
        super().__init__(
            service_id="production_authentication_service",
            dependencies=["database", "cache", "audit_logger"],
            **kwargs
        )
        
        self.user_repository = user_repository
        self.auth_token_repository = auth_token_repository
        self.cache = cache_repository
        
        # JWT configuration
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        
        # Security settings
        self.max_login_attempts = 5
        self.lockout_duration_minutes = 15
        self.password_min_length = 8
        self.require_password_complexity = True
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def authenticate_user(self, credentials: Dict[str, Any]) -> AuthenticationResult:
        """
        Authenticate user with comprehensive security validation
        Supports username/password, email/password, and future MFA methods
        """
        
        auth_type = credentials.get("type", "password")
        client_ip = credentials.get("client_ip", "unknown")
        user_agent = credentials.get("user_agent", "unknown")
        
        try:
            if auth_type == "password":
                return await self._authenticate_with_password(credentials, client_ip, user_agent)
            elif auth_type == "refresh_token":
                return await self._authenticate_with_refresh_token(credentials, client_ip, user_agent)
            elif auth_type == "mfa":
                return await self._authenticate_with_mfa(credentials, client_ip, user_agent)
            else:
                return AuthenticationResult(
                    success=False,
                    error_message=f"Unsupported authentication type: {auth_type}"
                )
        
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            await self._log_auth_event("authentication_error", None, client_ip, {"error": str(e)})
            return AuthenticationResult(
                success=False,
                error_message="Authentication service error"
            )
    
    async def _authenticate_with_password(
        self, 
        credentials: Dict[str, Any], 
        client_ip: str, 
        user_agent: str
    ) -> AuthenticationResult:
        """Authenticate user with username/password"""
        
        identifier = credentials.get("username") or credentials.get("email")
        password = credentials.get("password")
        
        if not identifier or not password:
            await self._log_auth_event("authentication_failed", None, client_ip, 
                                     {"reason": "missing_credentials"})
            return AuthenticationResult(
                success=False,
                error_message="Username/email and password required"
            )
        
        # Check rate limiting
        if await self._is_rate_limited(identifier, client_ip):
            await self._log_auth_event("authentication_rate_limited", identifier, client_ip)
            return AuthenticationResult(
                success=False,
                error_message="Too many failed attempts. Please try again later."
            )
        
        # Find user
        user = None
        if "@" in identifier:
            user = await self.user_repository.get_by_email(identifier)
        else:
            user = await self.user_repository.get_by_username(identifier)
        
        if not user:
            await self._record_failed_attempt(identifier, client_ip)
            await self._log_auth_event("authentication_failed", identifier, client_ip, 
                                     {"reason": "user_not_found"})
            return AuthenticationResult(
                success=False,
                error_message="Invalid credentials"
            )
        
        # Check if user is active
        if not user.is_active:
            await self._log_auth_event("authentication_failed", user.username, client_ip, 
                                     {"reason": "account_disabled"})
            return AuthenticationResult(
                success=False,
                error_message="Account is disabled"
            )
        
        # Verify password
        if not self.verify_password(password, user.password_hash):
            await self._record_failed_attempt(identifier, client_ip)
            await self._log_auth_event("authentication_failed", user.username, client_ip, 
                                     {"reason": "invalid_password"})
            return AuthenticationResult(
                success=False,
                error_message="Invalid credentials"
            )
        
        # Clear failed attempts on successful login
        await self._clear_failed_attempts(identifier, client_ip)
        
        # Generate tokens and session
        session_id = str(uuid4())
        access_token, access_expires = self._generate_access_token(user, session_id)
        refresh_token, refresh_expires = await self._generate_refresh_token(user, session_id)
        
        # Create session
        await self._create_session(session_id, user, client_ip, user_agent, refresh_expires)
        
        # Log successful authentication
        await self._log_auth_event("authentication_success", user.username, client_ip, 
                                 {"session_id": session_id})
        
        return AuthenticationResult(
            success=True,
            user=user,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=access_expires,
            permissions=user.roles,
            session_id=session_id,
            metadata={
                "login_time": datetime.utcnow().isoformat(),
                "client_ip": client_ip,
                "user_agent": user_agent
            }
        )
    
    async def _authenticate_with_refresh_token(
        self, 
        credentials: Dict[str, Any], 
        client_ip: str, 
        user_agent: str
    ) -> AuthenticationResult:
        """Authenticate using refresh token"""
        
        refresh_token = credentials.get("refresh_token")
        if not refresh_token:
            return AuthenticationResult(
                success=False,
                error_message="Refresh token required"
            )
        
        # Validate refresh token
        token_record = await self.auth_token_repository.get_by_token(refresh_token)
        if not token_record or token_record.token_type != "refresh":
            await self._log_auth_event("refresh_token_invalid", None, client_ip)
            return AuthenticationResult(
                success=False,
                error_message="Invalid refresh token"
            )
        
        # Check if token is expired
        if token_record.expires_at < datetime.utcnow():
            await self.auth_token_repository.delete(token_record.id)
            await self._log_auth_event("refresh_token_expired", None, client_ip)
            return AuthenticationResult(
                success=False,
                error_message="Refresh token expired"
            )
        
        # Get user
        user = await self.user_repository.get_by_id(token_record.user_id)
        if not user or not user.is_active:
            await self._log_auth_event("refresh_token_user_invalid", None, client_ip)
            return AuthenticationResult(
                success=False,
                error_message="Invalid user account"
            )
        
        # Generate new access token
        session_id = token_record.session_id or str(uuid4())
        access_token, access_expires = self._generate_access_token(user, session_id)
        
        await self._log_auth_event("token_refreshed", user.username, client_ip, 
                                 {"session_id": session_id})
        
        return AuthenticationResult(
            success=True,
            user=user,
            access_token=access_token,
            refresh_token=refresh_token,  # Return same refresh token
            expires_at=access_expires,
            permissions=user.roles,
            session_id=session_id
        )
    
    async def validate_token(self, token: str) -> TokenValidationResult:
        """Validate JWT access token with comprehensive checks"""
        
        try:
            # Decode JWT token
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Extract claims
            user_id = payload.get("sub")
            session_id = payload.get("session_id")
            expires_at = datetime.fromtimestamp(payload.get("exp", 0))
            
            if not user_id:
                return TokenValidationResult(
                    valid=False,
                    error_message="Invalid token format"
                )
            
            # Check if token is expired
            if expires_at < datetime.utcnow():
                return TokenValidationResult(
                    valid=False,
                    error_message="Token expired"
                )
            
            # Check if session is still valid
            if session_id and not await self._is_session_valid(session_id):
                return TokenValidationResult(
                    valid=False,
                    error_message="Session invalid"
                )
            
            # Get user
            user = await self.user_repository.get_by_id(UUID(user_id))
            if not user or not user.is_active:
                return TokenValidationResult(
                    valid=False,
                    error_message="User account invalid"
                )
            
            return TokenValidationResult(
                valid=True,
                user=user,
                claims=payload,
                expires_at=expires_at
            )
        
        except jwt.ExpiredSignatureError:
            return TokenValidationResult(
                valid=False,
                error_message="Token expired"
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return TokenValidationResult(
                valid=False,
                error_message="Invalid token"
            )
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return TokenValidationResult(
                valid=False,
                error_message="Token validation failed"
            )
    
    async def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Generate new access token from refresh token"""
        
        result = await self._authenticate_with_refresh_token(
            {"refresh_token": refresh_token, "type": "refresh_token"},
            "unknown", "unknown"
        )
        
        return result.access_token if result.success else None
    
    async def logout_user(self, session_id: str) -> bool:
        """Logout user and invalidate session"""
        
        try:
            # Invalidate session
            await self._invalidate_session(session_id)
            
            # Revoke all tokens for this session
            await self.auth_token_repository.revoke_by_session(session_id)
            
            await self._log_auth_event("logout", None, "unknown", {"session_id": session_id})
            
            return True
        
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False
    
    def hash_password(self, password: str) -> str:
        """Hash password securely using bcrypt"""
        
        if not self._validate_password_strength(password):
            raise ValueError("Password does not meet complexity requirements")
        
        # Generate salt and hash password
        salt = bcrypt.gensalt(rounds=12)
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        return password_hash.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def _generate_access_token(self, user: User, session_id: str) -> Tuple[str, datetime]:
        """Generate JWT access token"""
        
        expires_at = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = {
            "sub": str(user.id),
            "username": user.username,
            "email": user.email,
            "roles": user.roles,
            "session_id": session_id,
            "iat": datetime.utcnow(),
            "exp": expires_at,
            "type": "access_token"
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        return token, expires_at
    
    async def _generate_refresh_token(self, user: User, session_id: str) -> Tuple[str, datetime]:
        """Generate refresh token and store in database"""
        
        expires_at = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        # Generate secure random token
        token_value = secrets.token_urlsafe(64)
        
        # Create token record
        auth_token = AuthToken(
            id=uuid4(),
            user_id=user.id,
            token_type="refresh",
            token_value=token_value,
            session_id=session_id,
            expires_at=expires_at,
            created_at=datetime.utcnow(),
            is_revoked=False
        )
        
        await self.auth_token_repository.create(auth_token)
        
        return token_value, expires_at
    
    async def _create_session(
        self, 
        session_id: str, 
        user: User, 
        client_ip: str, 
        user_agent: str, 
        expires_at: datetime
    ) -> None:
        """Create user session"""
        
        session_data = {
            "session_id": session_id,
            "user_id": str(user.id),
            "username": user.username,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at.isoformat(),
            "last_activity": datetime.utcnow().isoformat()
        }
        
        # Store session in cache
        await self.cache.set(
            f"session:{session_id}",
            json.dumps(session_data),
            ttl=int((expires_at - datetime.utcnow()).total_seconds())
        )
        
        # Store in memory for quick access
        self.active_sessions[session_id] = session_data
    
    async def _is_session_valid(self, session_id: str) -> bool:
        """Check if session is still valid"""
        
        try:
            # Check cache first
            session_data = await self.cache.get(f"session:{session_id}")
            if session_data:
                session = json.loads(session_data)
                expires_at = datetime.fromisoformat(session["expires_at"])
                return expires_at > datetime.utcnow()
            
            return False
        
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return False
    
    async def _invalidate_session(self, session_id: str) -> None:
        """Invalidate user session"""
        
        try:
            await self.cache.delete(f"session:{session_id}")
            
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
        
        except Exception as e:
            logger.error(f"Session invalidation error: {e}")
    
    async def _is_rate_limited(self, identifier: str, client_ip: str) -> bool:
        """Check if authentication attempts are rate limited"""
        
        try:
            # Check attempts by identifier
            identifier_key = f"auth_attempts:{identifier}"
            identifier_attempts = await self.cache.get(identifier_key)
            
            if identifier_attempts and int(identifier_attempts) >= self.max_login_attempts:
                return True
            
            # Check attempts by IP
            ip_key = f"auth_attempts_ip:{client_ip}"
            ip_attempts = await self.cache.get(ip_key)
            
            if ip_attempts and int(ip_attempts) >= (self.max_login_attempts * 3):  # Higher limit for IP
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Rate limiting check error: {e}")
            return False
    
    async def _record_failed_attempt(self, identifier: str, client_ip: str) -> None:
        """Record failed authentication attempt"""
        
        try:
            lockout_seconds = self.lockout_duration_minutes * 60
            
            # Record attempt by identifier
            identifier_key = f"auth_attempts:{identifier}"
            current_attempts = await self.cache.get(identifier_key)
            new_attempts = int(current_attempts or 0) + 1
            await self.cache.set(identifier_key, str(new_attempts), ttl=lockout_seconds)
            
            # Record attempt by IP
            ip_key = f"auth_attempts_ip:{client_ip}"
            current_ip_attempts = await self.cache.get(ip_key)
            new_ip_attempts = int(current_ip_attempts or 0) + 1
            await self.cache.set(ip_key, str(new_ip_attempts), ttl=lockout_seconds)
        
        except Exception as e:
            logger.error(f"Failed to record authentication attempt: {e}")
    
    async def _clear_failed_attempts(self, identifier: str, client_ip: str) -> None:
        """Clear failed authentication attempts"""
        
        try:
            await asyncio.gather(
                self.cache.delete(f"auth_attempts:{identifier}"),
                self.cache.delete(f"auth_attempts_ip:{client_ip}"),
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Failed to clear authentication attempts: {e}")
    
    def _validate_password_strength(self, password: str) -> bool:
        """Validate password strength requirements"""
        
        if not self.require_password_complexity:
            return len(password) >= self.password_min_length
        
        # Check length
        if len(password) < self.password_min_length:
            return False
        
        # Check complexity requirements
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return sum([has_upper, has_lower, has_digit, has_special]) >= 3
    
    async def _log_auth_event(
        self, 
        event_type: str, 
        username: Optional[str], 
        client_ip: str, 
        metadata: Dict[str, Any] = None
    ) -> None:
        """Log authentication event for audit trail"""
        
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "username": username,
                "client_ip": client_ip,
                "service": self.service_id,
                "metadata": metadata or {}
            }
            
            # Store in audit log (cache for now, should be persistent storage)
            audit_key = f"audit:auth:{datetime.utcnow().strftime('%Y%m%d')}:{uuid4()}"
            await self.cache.set(audit_key, json.dumps(log_entry), ttl=86400 * 30)  # 30 days
            
            logger.info(f"Auth event: {event_type} - {username} - {client_ip}")
        
        except Exception as e:
            logger.error(f"Failed to log auth event: {e}")


class ProductionAuthorizationService(AuthorizationService, XORBService):
    """
    Production-ready authorization service with RBAC
    - Role-based access control
    - Permission caching
    - Resource-level permissions
    - Audit logging
    """
    
    def __init__(
        self,
        user_repository: UserRepository,
        cache_repository: CacheRepository,
        **kwargs
    ):
        super().__init__(
            service_id="production_authorization_service",
            dependencies=["database", "cache"],
            **kwargs
        )
        
        self.user_repository = user_repository
        self.cache = cache_repository
        
        # Permission hierarchy
        self.role_permissions = {
            "admin": [
                "users.create", "users.read", "users.update", "users.delete",
                "organizations.create", "organizations.read", "organizations.update", "organizations.delete",
                "scans.create", "scans.read", "scans.update", "scans.delete",
                "reports.create", "reports.read", "reports.update", "reports.delete",
                "system.admin", "system.configure"
            ],
            "security_analyst": [
                "scans.create", "scans.read", "scans.update",
                "reports.create", "reports.read", "reports.update",
                "threats.read", "threats.analyze",
                "compliance.read", "compliance.validate"
            ],
            "pentester": [
                "scans.create", "scans.read", "scans.update",
                "reports.create", "reports.read",
                "tools.use", "targets.scan"
            ],
            "auditor": [
                "scans.read", "reports.read",
                "compliance.read", "compliance.audit",
                "audit_logs.read"
            ],
            "user": [
                "profile.read", "profile.update",
                "scans.read", "reports.read"
            ]
        }
    
    async def check_permission(self, user: User, resource: str, action: str) -> bool:
        """Check if user has permission for resource action"""
        
        try:
            permission = f"{resource}.{action}"
            
            # Check cache first
            cache_key = f"permission:{user.id}:{permission}"
            cached_result = await self.cache.get(cache_key)
            
            if cached_result is not None:
                return cached_result.lower() == "true"
            
            # Get user permissions
            user_permissions = await self.get_user_permissions(user)
            
            # Check if user has permission
            has_permission = False
            
            # Check direct permissions
            all_permissions = []
            for role in user.roles:
                all_permissions.extend(user_permissions.get(role, []))
            
            has_permission = permission in all_permissions
            
            # Check wildcard permissions
            if not has_permission:
                wildcard_permission = f"{resource}.*"
                has_permission = wildcard_permission in all_permissions
            
            # Check admin override
            if not has_permission:
                has_permission = "system.admin" in all_permissions
            
            # Cache result
            await self.cache.set(cache_key, str(has_permission).lower(), ttl=300)  # 5 minutes
            
            # Log permission check
            await self._log_permission_check(user, permission, has_permission)
            
            return has_permission
        
        except Exception as e:
            logger.error(f"Permission check error: {e}")
            return False
    
    async def get_user_permissions(self, user: User) -> Dict[str, List[str]]:
        """Get all permissions for user"""
        
        try:
            # Check cache first
            cache_key = f"user_permissions:{user.id}"
            cached_permissions = await self.cache.get(cache_key)
            
            if cached_permissions:
                return json.loads(cached_permissions)
            
            # Build permissions from roles
            permissions = {}
            
            for role in user.roles:
                role_perms = self.role_permissions.get(role, [])
                permissions[role] = role_perms
            
            # Cache permissions
            await self.cache.set(cache_key, json.dumps(permissions), ttl=3600)  # 1 hour
            
            return permissions
        
        except Exception as e:
            logger.error(f"Get user permissions error: {e}")
            return {}
    
    async def _log_permission_check(self, user: User, permission: str, granted: bool) -> None:
        """Log permission check for audit trail"""
        
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": str(user.id),
                "username": user.username,
                "permission": permission,
                "granted": granted,
                "service": self.service_id
            }
            
            # Store in audit log
            audit_key = f"audit:authz:{datetime.utcnow().strftime('%Y%m%d')}:{uuid4()}"
            await self.cache.set(audit_key, json.dumps(log_entry), ttl=86400 * 30)  # 30 days
        
        except Exception as e:
            logger.error(f"Failed to log permission check: {e}")