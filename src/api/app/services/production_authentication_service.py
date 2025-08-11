"""
Production Authentication Service Implementation
Sophisticated authentication with multi-factor support, JWT management, and security hardening
"""

import asyncio
import hashlib
import secrets
import jwt
import bcrypt
# Optional dependencies
try:
    import pyotp
    PYOTP_AVAILABLE = True
except ImportError:
    PYOTP_AVAILABLE = False
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re
import json
from ipaddress import ip_address, AddressValueError

from .interfaces import AuthenticationService, AuthorizationService
from .base_service import XORBService, ServiceHealth, ServiceStatus
from ..domain.entities import User, Organization

logger = logging.getLogger(__name__)


class AuthMethod(Enum):
    """Authentication methods supported"""
    PASSWORD = "password"
    MFA_TOTP = "mfa_totp"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    SSO_SAML = "sso_saml"
    SSO_OAUTH2 = "sso_oauth2"


class SessionStatus(Enum):
    """Session status types"""
    ACTIVE = "active"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"
    SUSPICIOUS = "suspicious"


@dataclass
class AuthenticationRequest:
    """Authentication request structure"""
    username: str
    auth_method: AuthMethod
    credentials: Dict[str, Any]
    client_info: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class AuthenticationResult:
    """Authentication result with detailed information"""
    success: bool
    user_id: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    session_id: Optional[str] = None
    auth_methods_used: List[AuthMethod] = None
    security_context: Dict[str, Any] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.auth_methods_used is None:
            self.auth_methods_used = []
        if self.security_context is None:
            self.security_context = {}


@dataclass
class UserSession:
    """User session management"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    status: SessionStatus
    auth_methods: List[AuthMethod]
    security_flags: Dict[str, Any]
    metadata: Dict[str, Any]


class ProductionAuthenticationService(XORBService, AuthenticationService, AuthorizationService):
    """Production-ready authentication service with advanced security features"""
    
    def __init__(self, **kwargs):
        super().__init__(
            service_id="production_auth_service",
            **kwargs
        )
        
        # Security configuration
        self.jwt_secret = kwargs.get("jwt_secret", secrets.token_hex(32))
        self.jwt_algorithm = "HS256"
        self.access_token_expiry = timedelta(hours=1)
        self.refresh_token_expiry = timedelta(days=7)
        self.session_timeout = timedelta(hours=8)
        
        # Password security
        self.password_min_length = 12
        self.password_complexity_required = True
        self.password_history_length = 5
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        
        # MFA configuration
        self.mfa_required_roles = ["admin", "security_analyst"]
        self.totp_issuer = "XORB Security Platform"
        
        # Session management
        self.active_sessions: Dict[str, UserSession] = {}
        self.blacklisted_tokens: set = set()
        self.failed_attempts: Dict[str, List[datetime]] = {}
        
        # Security monitoring
        self.suspicious_activities: List[Dict[str, Any]] = []
        self.security_events: List[Dict[str, Any]] = []
        
        # User data store (in production, this would be a database)
        self.users: Dict[str, Dict[str, Any]] = {}
        self.user_permissions: Dict[str, Dict[str, List[str]]] = {}
        
    async def initialize(self) -> bool:
        """Initialize authentication service with security hardening"""
        try:
            logger.info("Initializing Production Authentication Service...")
            
            # Initialize security monitoring
            asyncio.create_task(self._security_monitoring_task())
            asyncio.create_task(self._session_cleanup_task())
            asyncio.create_task(self._failed_attempt_cleanup_task())
            
            # Load security policies
            await self._load_security_policies()
            
            # Initialize MFA subsystem
            await self._initialize_mfa_system()
            
            # Setup audit logging
            await self._setup_audit_logging()
            
            logger.info("Production Authentication Service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize authentication service: {e}")
            return False
    
    # AuthenticationService interface implementation
    
    async def authenticate_user(self, credentials: Dict[str, Any]) -> AuthenticationResult:
        """Sophisticated multi-factor authentication with security monitoring"""
        try:
            # Create authentication request
            auth_request = AuthenticationRequest(
                username=credentials.get("username", ""),
                auth_method=AuthMethod(credentials.get("auth_method", "password")),
                credentials=credentials,
                client_info=credentials.get("client_info", {}),
                ip_address=credentials.get("ip_address"),
                user_agent=credentials.get("user_agent")
            )
            
            # Security pre-checks
            security_check = await self._perform_security_checks(auth_request)
            if not security_check["allowed"]:
                return AuthenticationResult(
                    success=False,
                    error_code="SECURITY_VIOLATION",
                    error_message=security_check["reason"]
                )
            
            # Rate limiting check
            if not await self._check_rate_limiting(auth_request.username, auth_request.ip_address):
                return AuthenticationResult(
                    success=False,
                    error_code="RATE_LIMITED",
                    error_message="Too many authentication attempts"
                )
            
            # Primary authentication
            primary_auth = await self._authenticate_primary(auth_request)
            if not primary_auth.success:
                await self._record_failed_attempt(auth_request)
                return primary_auth
            
            # Multi-factor authentication if required
            user_data = self.users.get(primary_auth.user_id, {})
            requires_mfa = await self._requires_mfa(user_data, auth_request)
            
            if requires_mfa:
                mfa_result = await self._authenticate_mfa(auth_request, user_data)
                if not mfa_result.success:
                    return mfa_result
                
                primary_auth.auth_methods_used.extend(mfa_result.auth_methods_used)
            
            # Generate tokens and session
            tokens = await self._generate_tokens(primary_auth.user_id, auth_request)
            session = await self._create_user_session(primary_auth.user_id, auth_request, tokens)
            
            # Record successful authentication
            await self._record_successful_auth(auth_request, primary_auth.user_id)
            
            # Build final result
            result = AuthenticationResult(
                success=True,
                user_id=primary_auth.user_id,
                access_token=tokens["access_token"],
                refresh_token=tokens["refresh_token"],
                expires_at=tokens["expires_at"],
                session_id=session.session_id,
                auth_methods_used=primary_auth.auth_methods_used,
                security_context={
                    "auth_timestamp": datetime.utcnow().isoformat(),
                    "client_ip": auth_request.ip_address,
                    "user_agent": auth_request.user_agent,
                    "security_level": await self._calculate_security_level(auth_request, user_data),
                    "session_info": asdict(session)
                }
            )
            
            logger.info(f"User {auth_request.username} authenticated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Authentication failed for user {credentials.get('username', 'unknown')}: {e}")
            return AuthenticationResult(
                success=False,
                error_code="AUTHENTICATION_ERROR",
                error_message=str(e)
            )
    
    async def validate_token(self, token: str) -> Dict[str, Any]:
        """Advanced JWT token validation with security checks"""
        try:
            # Check token blacklist
            if token in self.blacklisted_tokens:
                return {
                    "valid": False,
                    "error": "TOKEN_BLACKLISTED",
                    "message": "Token has been revoked"
                }
            
            # Decode and validate JWT
            try:
                payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            except jwt.ExpiredSignatureError:
                return {
                    "valid": False,
                    "error": "TOKEN_EXPIRED",
                    "message": "Token has expired"
                }
            except jwt.InvalidTokenError as e:
                return {
                    "valid": False,
                    "error": "TOKEN_INVALID",
                    "message": f"Invalid token: {e}"
                }
            
            # Validate token claims
            required_claims = ["user_id", "exp", "iat", "session_id"]
            for claim in required_claims:
                if claim not in payload:
                    return {
                        "valid": False,
                        "error": "INVALID_CLAIMS",
                        "message": f"Missing required claim: {claim}"
                    }
            
            # Validate session
            session_id = payload.get("session_id")
            if session_id not in self.active_sessions:
                return {
                    "valid": False,
                    "error": "SESSION_INVALID",
                    "message": "Session no longer valid"
                }
            
            session = self.active_sessions[session_id]
            if session.status != SessionStatus.ACTIVE:
                return {
                    "valid": False,
                    "error": "SESSION_INACTIVE",
                    "message": f"Session status: {session.status.value}"
                }
            
            # Update session last activity
            session.last_activity = datetime.utcnow()
            
            # Validate user still exists and is active
            user_id = payload.get("user_id")
            user_data = self.users.get(user_id)
            if not user_data or not user_data.get("active", True):
                return {
                    "valid": False,
                    "error": "USER_INACTIVE",
                    "message": "User account is inactive"
                }
            
            return {
                "valid": True,
                "user_id": user_id,
                "session_id": session_id,
                "payload": payload,
                "user_data": user_data,
                "session": asdict(session)
            }
            
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return {
                "valid": False,
                "error": "VALIDATION_ERROR",
                "message": str(e)
            }
    
    async def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Secure token refresh with rotation"""
        try:
            # Validate refresh token
            validation_result = await self.validate_token(refresh_token)
            if not validation_result["valid"]:
                return None
            
            user_id = validation_result["user_id"]
            session_id = validation_result["session_id"]
            
            # Generate new access token
            new_token_data = await self._generate_access_token(user_id, session_id)
            
            # Optionally rotate refresh token for enhanced security
            if await self._should_rotate_refresh_token(refresh_token):
                # Blacklist old refresh token
                self.blacklisted_tokens.add(refresh_token)
                
                # Generate new refresh token
                new_refresh_token = await self._generate_refresh_token(user_id, session_id)
                return {
                    "access_token": new_token_data["token"],
                    "refresh_token": new_refresh_token["token"],
                    "expires_at": new_token_data["expires_at"].isoformat()
                }
            
            return new_token_data["token"]
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return None
    
    async def logout_user(self, session_id: str) -> bool:
        """Secure logout with session cleanup"""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            
            # Mark session as invalidated
            session.status = SessionStatus.INVALIDATED
            
            # Blacklist associated tokens
            user_tokens = await self._get_user_tokens(session.user_id, session_id)
            for token in user_tokens:
                self.blacklisted_tokens.add(token)
            
            # Remove session
            del self.active_sessions[session_id]
            
            # Record logout event
            await self._record_logout_event(session)
            
            logger.info(f"User {session.user_id} logged out successfully")
            return True
            
        except Exception as e:
            logger.error(f"Logout failed for session {session_id}: {e}")
            return False
    
    def hash_password(self, password: str) -> str:
        """Secure password hashing with bcrypt"""
        try:
            # Generate salt and hash password
            salt = bcrypt.gensalt(rounds=12)  # High cost factor for security
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Password hashing failed: {e}")
            raise
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Secure password verification"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    # AuthorizationService interface implementation
    
    async def check_permission(self, user: User, resource: str, action: str) -> bool:
        """Advanced role-based access control with context awareness"""
        try:
            user_id = str(user.id) if hasattr(user, 'id') else str(user)
            
            # Get user permissions
            user_perms = self.user_permissions.get(user_id, {})
            resource_perms = user_perms.get(resource, [])
            
            # Check direct permission
            if action in resource_perms or "*" in resource_perms:
                return True
            
            # Check role-based permissions
            user_data = self.users.get(user_id, {})
            user_roles = user_data.get("roles", [])
            
            for role in user_roles:
                role_perms = await self._get_role_permissions(role)
                role_resource_perms = role_perms.get(resource, [])
                
                if action in role_resource_perms or "*" in role_resource_perms:
                    return True
            
            # Check hierarchical permissions
            if await self._check_hierarchical_permissions(user_data, resource, action):
                return True
            
            # Check context-based permissions
            if await self._check_context_permissions(user_data, resource, action):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False
    
    async def get_user_permissions(self, user: User) -> Dict[str, List[str]]:
        """Get comprehensive user permissions including inherited ones"""
        try:
            user_id = str(user.id) if hasattr(user, 'id') else str(user)
            user_data = self.users.get(user_id, {})
            
            # Start with direct permissions
            permissions = self.user_permissions.get(user_id, {}).copy()
            
            # Add role-based permissions
            user_roles = user_data.get("roles", [])
            for role in user_roles:
                role_perms = await self._get_role_permissions(role)
                for resource, actions in role_perms.items():
                    if resource not in permissions:
                        permissions[resource] = []
                    permissions[resource].extend(actions)
                    permissions[resource] = list(set(permissions[resource]))  # Remove duplicates
            
            # Add organization-based permissions
            org_id = user_data.get("organization_id")
            if org_id:
                org_perms = await self._get_organization_permissions(org_id)
                for resource, actions in org_perms.items():
                    if resource not in permissions:
                        permissions[resource] = []
                    permissions[resource].extend(actions)
                    permissions[resource] = list(set(permissions[resource]))
            
            return permissions
            
        except Exception as e:
            logger.error(f"Failed to get user permissions: {e}")
            return {}
    
    # Advanced helper methods
    
    async def _perform_security_checks(self, auth_request: AuthenticationRequest) -> Dict[str, Any]:
        """Comprehensive security checks before authentication"""
        checks = {
            "allowed": True,
            "reason": "",
            "security_flags": []
        }
        
        try:
            # IP-based security checks
            if auth_request.ip_address:
                if await self._is_ip_blacklisted(auth_request.ip_address):
                    checks["allowed"] = False
                    checks["reason"] = "IP address is blacklisted"
                    return checks
                
                if await self._is_ip_suspicious(auth_request.ip_address):
                    checks["security_flags"].append("suspicious_ip")
            
            # User agent analysis
            if auth_request.user_agent:
                if await self._is_user_agent_suspicious(auth_request.user_agent):
                    checks["security_flags"].append("suspicious_user_agent")
            
            # Geolocation checks
            if auth_request.ip_address:
                geo_info = await self._get_geo_location(auth_request.ip_address)
                if await self._is_geo_location_risky(geo_info, auth_request.username):
                    checks["security_flags"].append("risky_geolocation")
            
            # Time-based analysis
            if await self._is_unusual_time_access(auth_request.username):
                checks["security_flags"].append("unusual_time_access")
            
            return checks
            
        except Exception as e:
            logger.error(f"Security checks failed: {e}")
            checks["allowed"] = False
            checks["reason"] = "Security check error"
            return checks
    
    async def _authenticate_primary(self, auth_request: AuthenticationRequest) -> AuthenticationResult:
        """Primary authentication based on method"""
        try:
            if auth_request.auth_method == AuthMethod.PASSWORD:
                return await self._authenticate_password(auth_request)
            elif auth_request.auth_method == AuthMethod.API_KEY:
                return await self._authenticate_api_key(auth_request)
            elif auth_request.auth_method == AuthMethod.CERTIFICATE:
                return await self._authenticate_certificate(auth_request)
            else:
                return AuthenticationResult(
                    success=False,
                    error_code="UNSUPPORTED_AUTH_METHOD",
                    error_message=f"Authentication method {auth_request.auth_method.value} not supported"
                )
                
        except Exception as e:
            logger.error(f"Primary authentication failed: {e}")
            return AuthenticationResult(
                success=False,
                error_code="PRIMARY_AUTH_ERROR",
                error_message=str(e)
            )
    
    async def _authenticate_password(self, auth_request: AuthenticationRequest) -> AuthenticationResult:
        """Password-based authentication with security checks"""
        username = auth_request.username
        password = auth_request.credentials.get("password", "")
        
        # Find user
        user_data = None
        user_id = None
        for uid, data in self.users.items():
            if data.get("username") == username or data.get("email") == username:
                user_data = data
                user_id = uid
                break
        
        if not user_data:
            return AuthenticationResult(
                success=False,
                error_code="USER_NOT_FOUND",
                error_message="User not found"
            )
        
        # Check if user is active
        if not user_data.get("active", True):
            return AuthenticationResult(
                success=False,
                error_code="USER_INACTIVE",
                error_message="User account is inactive"
            )
        
        # Check if account is locked
        if await self._is_account_locked(username):
            return AuthenticationResult(
                success=False,
                error_code="ACCOUNT_LOCKED",
                error_message="Account is temporarily locked"
            )
        
        # Verify password
        stored_hash = user_data.get("password_hash", "")
        if not self.verify_password(password, stored_hash):
            return AuthenticationResult(
                success=False,
                error_code="INVALID_CREDENTIALS",
                error_message="Invalid username or password"
            )
        
        # Check password age and complexity
        password_warnings = await self._check_password_policy(user_data, password)
        
        return AuthenticationResult(
            success=True,
            user_id=user_id,
            auth_methods_used=[AuthMethod.PASSWORD],
            security_context={
                "password_warnings": password_warnings,
                "last_login": user_data.get("last_login"),
                "login_count": user_data.get("login_count", 0) + 1
            }
        )
    
    async def health_check(self) -> ServiceHealth:
        """Comprehensive authentication service health check"""
        try:
            checks = {
                "active_sessions": len(self.active_sessions),
                "blacklisted_tokens": len(self.blacklisted_tokens),
                "failed_attempts_tracking": len(self.failed_attempts),
                "registered_users": len(self.users),
                "security_events_recent": len([
                    event for event in self.security_events
                    if (datetime.utcnow() - datetime.fromisoformat(event["timestamp"])).total_seconds() < 3600
                ])
            }
            
            # Check for security issues
            critical_issues = []
            
            # Check for excessive failed attempts
            recent_failures = sum(
                len([attempt for attempt in attempts if (datetime.utcnow() - attempt).total_seconds() < 3600])
                for attempts in self.failed_attempts.values()
            )
            
            if recent_failures > 100:  # More than 100 failed attempts in last hour
                critical_issues.append("High number of failed authentication attempts")
            
            # Check session health
            expired_sessions = [
                s for s in self.active_sessions.values()
                if (datetime.utcnow() - s.last_activity) > self.session_timeout
            ]
            
            if len(expired_sessions) > len(self.active_sessions) * 0.5:
                critical_issues.append("High number of expired sessions")
            
            # Determine overall health
            if critical_issues:
                status = ServiceStatus.DEGRADED
                message = f"Security issues detected: {'; '.join(critical_issues)}"
            else:
                status = ServiceStatus.HEALTHY
                message = "Authentication service operational"
            
            return ServiceHealth(
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                checks=checks
            )
            
        except Exception as e:
            return ServiceHealth(
                status=ServiceStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                timestamp=datetime.utcnow(),
                checks={"error": str(e)}
            )
    
    async def _load_security_policies(self) -> None:
        """Load security policies and configurations"""
        try:
            logger.info("Loading security policies...")
            
            # Load security policies from environment variables with secure defaults
            self.security_policies = {
                "password_policy": {
                    "min_length": int(os.getenv("PASSWORD_MIN_LENGTH", "12")),
                    "require_uppercase": os.getenv("PASSWORD_REQUIRE_UPPERCASE", "true").lower() == "true",
                    "require_lowercase": os.getenv("PASSWORD_REQUIRE_LOWERCASE", "true").lower() == "true",
                    "require_numbers": os.getenv("PASSWORD_REQUIRE_NUMBERS", "true").lower() == "true",
                    "require_special_chars": os.getenv("PASSWORD_REQUIRE_SPECIAL", "true").lower() == "true",
                    "max_age_days": int(os.getenv("PASSWORD_MAX_AGE_DAYS", "90")),
                    "history_count": int(os.getenv("PASSWORD_HISTORY_COUNT", "5"))
                },
                "session_policy": {
                    "max_duration_hours": int(os.getenv("SESSION_MAX_DURATION_HOURS", "8")),
                    "idle_timeout_minutes": int(os.getenv("SESSION_IDLE_TIMEOUT_MINUTES", "30")),
                    "concurrent_sessions": int(os.getenv("SESSION_CONCURRENT_LIMIT", "3")),
                    "secure_cookie": os.getenv("SESSION_SECURE_COOKIE", "true").lower() == "true"
                },
                "lockout_policy": {
                    "max_attempts": int(os.getenv("AUTH_MAX_ATTEMPTS", "5")),
                    "lockout_duration_minutes": int(os.getenv("AUTH_LOCKOUT_DURATION_MINUTES", "30")),
                    "progressive_delay": os.getenv("AUTH_PROGRESSIVE_DELAY", "true").lower() == "true"
                },
                "audit_policy": {
                    "log_all_attempts": os.getenv("AUDIT_LOG_ALL_ATTEMPTS", "true").lower() == "true",
                    "log_successful_logins": os.getenv("AUDIT_LOG_SUCCESS", "true").lower() == "true",
                    "log_failed_logins": os.getenv("AUDIT_LOG_FAILURES", "true").lower() == "true",
                    "retention_days": int(os.getenv("AUDIT_RETENTION_DAYS", "365"))
                }
            }
            
            # Log policy configuration (without sensitive values)
            logger.info(f"Security policies loaded - Password min length: {self.security_policies['password_policy']['min_length']}, "
                       f"Session timeout: {self.security_policies['session_policy']['idle_timeout_minutes']}min, "
                       f"Max login attempts: {self.security_policies['lockout_policy']['max_attempts']}")
        except Exception as e:
            logger.error(f"Failed to load security policies: {e}")
    
    async def _initialize_mfa_system(self) -> None:
        """Initialize MFA subsystem"""
        try:
            if PYOTP_AVAILABLE:
                logger.info("Initializing MFA system with TOTP support...")
                self.mfa_enabled = True
            else:
                logger.warning("PyOTP not available - MFA disabled")
                self.mfa_enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize MFA system: {e}")
    
    async def _setup_audit_logging(self) -> None:
        """Setup audit logging for security events"""
        try:
            logger.info("Setting up audit logging...")
            
            # Initialize audit logging configuration
            self.audit_config = {
                "enabled": True,
                "log_format": os.getenv("AUDIT_LOG_FORMAT", "json"),
                "log_level": os.getenv("AUDIT_LOG_LEVEL", "INFO"),
                "destinations": []
            }
            
            # Setup file-based audit logging
            audit_file_path = os.getenv("AUDIT_LOG_FILE", "/var/log/xorb/audit.log")
            if audit_file_path:
                self.audit_config["destinations"].append({
                    "type": "file",
                    "path": audit_file_path,
                    "rotation": os.getenv("AUDIT_LOG_ROTATION", "daily")
                })
            
            # Setup database audit logging if configured
            if os.getenv("AUDIT_DATABASE_URL"):
                self.audit_config["destinations"].append({
                    "type": "database",
                    "url": os.getenv("AUDIT_DATABASE_URL"),
                    "table": os.getenv("AUDIT_TABLE_NAME", "audit_logs")
                })
            
            # Setup SIEM integration if configured
            siem_endpoint = os.getenv("AUDIT_SIEM_ENDPOINT")
            if siem_endpoint:
                self.audit_config["destinations"].append({
                    "type": "siem",
                    "endpoint": siem_endpoint,
                    "api_key": os.getenv("AUDIT_SIEM_API_KEY"),
                    "format": os.getenv("AUDIT_SIEM_FORMAT", "cef")
                })
            
            self.audit_enabled = True
            logger.info(f"Audit logging configured with {len(self.audit_config['destinations'])} destinations")
            
        except Exception as e:
            logger.error(f"Failed to setup audit logging: {e}")
            self.audit_enabled = False
    
    async def _check_password_policy(self, user_data: Dict[str, Any], password: str) -> List[str]:
        """Check password against security policy"""
        warnings = []
        try:
            # Check password age
            password_created = user_data.get("password_created")
            if password_created:
                age_days = (datetime.utcnow() - datetime.fromisoformat(password_created)).days
                if age_days > 90:
                    warnings.append("Password is older than 90 days")
            
            # Check password complexity
            if len(password) < 12:
                warnings.append("Password should be at least 12 characters")
                
        except Exception as e:
            logger.error(f"Password policy check failed: {e}")
        
        return warnings


# Global service instance
_auth_service: Optional[ProductionAuthenticationService] = None

async def get_production_auth_service() -> ProductionAuthenticationService:
    """Get global production authentication service instance"""
    global _auth_service
    
    if _auth_service is None:
        _auth_service = ProductionAuthenticationService()
        await _auth_service.initialize()
    
    return _auth_service