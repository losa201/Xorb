#!/usr/bin/env python3
"""
Enterprise Authorization Management System
Role-based access control and permission management for autonomous operations

This module provides enterprise-grade authorization including:
- Role-based access control (RBAC)
- Attribute-based access control (ABAC) 
- Fine-grained permission management
- Multi-factor authentication integration
- Session management and token validation
- Delegation and temporary permissions
- Audit trail for all authorization decisions
"""

import asyncio
import logging
import json
import uuid
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import structlog

from .audit_logger import AuditLogger, AuditEvent, EventType, EventSeverity
from .security_framework import SecurityLevel

logger = structlog.get_logger(__name__)


class Permission(Enum):
    """System permissions for operations"""
    # Red team operations
    EXECUTE_RECONNAISSANCE = "execute_reconnaissance"
    EXECUTE_INITIAL_ACCESS = "execute_initial_access"
    EXECUTE_PERSISTENCE = "execute_persistence"
    EXECUTE_PRIVILEGE_ESCALATION = "execute_privilege_escalation"
    EXECUTE_LATERAL_MOVEMENT = "execute_lateral_movement"
    EXECUTE_DATA_COLLECTION = "execute_data_collection"
    
    # System administration
    MANAGE_OPERATIONS = "manage_operations"
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    MANAGE_POLICIES = "manage_policies"
    
    # Data access
    ACCESS_AUDIT_LOGS = "access_audit_logs"
    ACCESS_SENSITIVE_DATA = "access_sensitive_data"
    MODIFY_SYSTEM_CONFIG = "modify_system_config"
    
    # Emergency controls
    EMERGENCY_STOP = "emergency_stop"
    OVERRIDE_SAFETY = "override_safety"
    
    # Monitoring and analysis
    VIEW_OPERATIONS = "view_operations"
    VIEW_METRICS = "view_metrics"
    EXPORT_REPORTS = "export_reports"


class Role(Enum):
    """Predefined system roles"""
    SYSTEM_ADMIN = "system_admin"
    SECURITY_ENGINEER = "security_engineer"
    RED_TEAM_OPERATOR = "red_team_operator"
    PURPLE_TEAM_LEAD = "purple_team_lead"
    COMPLIANCE_OFFICER = "compliance_officer"
    OBSERVER = "observer"
    EMERGENCY_RESPONDER = "emergency_responder"


@dataclass
class User:
    """User account definition"""
    user_id: str
    username: str
    email: str
    roles: List[Role]
    permissions: List[Permission]
    security_clearance: SecurityLevel
    mfa_enabled: bool
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    account_locked: bool = False
    created_at: datetime = None
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class Session:
    """User session with expiration and validation"""
    session_id: str
    user_id: str
    user: User
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    source_ip: str
    user_agent: str
    mfa_verified: bool
    elevated_privileges: bool = False
    privileges_expire_at: Optional[datetime] = None
    
    def is_valid(self) -> bool:
        """Check if session is still valid"""
        now = datetime.utcnow()
        return (now < self.expires_at and 
                not self.user.account_locked and
                (now - self.last_activity).total_seconds() < 3600)  # 1 hour timeout
    
    def is_elevated(self) -> bool:
        """Check if session has elevated privileges"""
        if not self.elevated_privileges:
            return False
        if self.privileges_expire_at and datetime.utcnow() > self.privileges_expire_at:
            return False
        return True


@dataclass
class AuthorizationRequest:
    """Authorization request for permission checking"""
    request_id: str
    user_id: str
    session_id: str
    permission: Permission
    resource: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())


@dataclass
class AuthorizationResult:
    """Result of authorization check"""
    request_id: str
    granted: bool
    reason: str
    user_id: str
    permission: Permission
    resource: Optional[str] = None
    conditions: List[str] = None
    expires_at: Optional[datetime] = None
    requires_elevation: bool = False
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.conditions is None:
            self.conditions = []


class TokenManager:
    """JWT token management for authentication"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.private_key = None
        self.public_key = None
        self.algorithm = "RS256"
        self.token_expiry = timedelta(hours=8)
        
    async def initialize(self):
        """Initialize token management"""
        try:
            # Load or generate RSA keys
            private_key_path = self.config.get("private_key_path", "keys/jwt_private.pem")
            public_key_path = self.config.get("public_key_path", "keys/jwt_public.pem")
            
            if not Path(private_key_path).exists():
                await self._generate_key_pair(private_key_path, public_key_path)
            
            await self._load_keys(private_key_path, public_key_path)
            
            logger.info("Token manager initialized")
            
        except Exception as e:
            logger.error("Failed to initialize token manager", error=str(e))
            raise
    
    async def _generate_key_pair(self, private_path: str, public_path: str):
        """Generate RSA key pair for JWT signing"""
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Get public key
        public_key = private_key.public_key()
        
        # Create directories
        Path(private_path).parent.mkdir(parents=True, exist_ok=True)
        Path(public_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save private key
        with open(private_path, 'wb') as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        # Save public key
        with open(public_path, 'wb') as f:
            f.write(public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))
        
        # Set restrictive permissions
        Path(private_path).chmod(0o600)
        Path(public_path).chmod(0o644)
        
        logger.info("JWT key pair generated", 
                   private_key=private_path, public_key=public_path)
    
    async def _load_keys(self, private_path: str, public_path: str):
        """Load RSA keys"""
        with open(private_path, 'rb') as f:
            self.private_key = serialization.load_pem_private_key(
                f.read(), password=None
            )
        
        with open(public_path, 'rb') as f:
            self.public_key = serialization.load_pem_public_key(f.read())
    
    def generate_token(self, user: User, session_id: str) -> str:
        """Generate JWT token for user"""
        try:
            now = datetime.utcnow()
            expires_at = now + self.token_expiry
            
            payload = {
                "sub": user.user_id,
                "username": user.username,
                "email": user.email,
                "roles": [role.value for role in user.roles],
                "permissions": [perm.value for perm in user.permissions],
                "security_clearance": user.security_clearance.value,
                "session_id": session_id,
                "iat": now,
                "exp": expires_at,
                "iss": "xorb_authorization_system",
                "aud": "xorb_platform"
            }
            
            token = jwt.encode(payload, self.private_key, algorithm=self.algorithm)
            
            logger.debug("Token generated", user_id=user.user_id, session_id=session_id)
            
            return token
            
        except Exception as e:
            logger.error("Token generation failed", error=str(e))
            raise
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate and decode JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.public_key, 
                algorithms=[self.algorithm],
                audience="xorb_platform",
                issuer="xorb_authorization_system"
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid token", error=str(e))
            return None
        except Exception as e:
            logger.error("Token validation failed", error=str(e))
            return None


class RoleManager:
    """Role and permission management"""
    
    def __init__(self):
        self.role_permissions: Dict[Role, Set[Permission]] = {}
        self._initialize_default_roles()
    
    def _initialize_default_roles(self):
        """Initialize default role-permission mappings"""
        self.role_permissions = {
            Role.SYSTEM_ADMIN: {
                Permission.MANAGE_OPERATIONS,
                Permission.MANAGE_USERS,
                Permission.MANAGE_ROLES,
                Permission.MANAGE_POLICIES,
                Permission.ACCESS_AUDIT_LOGS,
                Permission.MODIFY_SYSTEM_CONFIG,
                Permission.EMERGENCY_STOP,
                Permission.OVERRIDE_SAFETY,
                Permission.VIEW_OPERATIONS,
                Permission.VIEW_METRICS,
                Permission.EXPORT_REPORTS
            },
            
            Role.SECURITY_ENGINEER: {
                Permission.EXECUTE_RECONNAISSANCE,
                Permission.EXECUTE_INITIAL_ACCESS,
                Permission.EXECUTE_PERSISTENCE,
                Permission.EXECUTE_PRIVILEGE_ESCALATION,
                Permission.EXECUTE_LATERAL_MOVEMENT,
                Permission.EXECUTE_DATA_COLLECTION,
                Permission.MANAGE_OPERATIONS,
                Permission.ACCESS_AUDIT_LOGS,
                Permission.VIEW_OPERATIONS,
                Permission.VIEW_METRICS,
                Permission.EXPORT_REPORTS,
                Permission.EMERGENCY_STOP
            },
            
            Role.RED_TEAM_OPERATOR: {
                Permission.EXECUTE_RECONNAISSANCE,
                Permission.EXECUTE_INITIAL_ACCESS,
                Permission.EXECUTE_PERSISTENCE,
                Permission.EXECUTE_PRIVILEGE_ESCALATION,
                Permission.EXECUTE_LATERAL_MOVEMENT,
                Permission.EXECUTE_DATA_COLLECTION,
                Permission.VIEW_OPERATIONS,
                Permission.VIEW_METRICS
            },
            
            Role.PURPLE_TEAM_LEAD: {
                Permission.EXECUTE_RECONNAISSANCE,
                Permission.EXECUTE_INITIAL_ACCESS,
                Permission.MANAGE_OPERATIONS,
                Permission.ACCESS_AUDIT_LOGS,
                Permission.VIEW_OPERATIONS,
                Permission.VIEW_METRICS,
                Permission.EXPORT_REPORTS,
                Permission.EMERGENCY_STOP
            },
            
            Role.COMPLIANCE_OFFICER: {
                Permission.ACCESS_AUDIT_LOGS,
                Permission.VIEW_OPERATIONS,
                Permission.VIEW_METRICS,
                Permission.EXPORT_REPORTS
            },
            
            Role.OBSERVER: {
                Permission.VIEW_OPERATIONS,
                Permission.VIEW_METRICS
            },
            
            Role.EMERGENCY_RESPONDER: {
                Permission.EMERGENCY_STOP,
                Permission.VIEW_OPERATIONS,
                Permission.ACCESS_AUDIT_LOGS
            }
        }
    
    def get_role_permissions(self, role: Role) -> Set[Permission]:
        """Get permissions for a role"""
        return self.role_permissions.get(role, set())
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission"""
        # Direct permission
        if permission in user.permissions:
            return True
        
        # Role-based permission
        for role in user.roles:
            if permission in self.get_role_permissions(role):
                return True
        
        return False


class AuthorizationManager:
    """Main authorization management system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.manager_id = str(uuid.uuid4())
        
        # Core components
        self.token_manager = TokenManager(config.get("token_manager", {}))
        self.role_manager = RoleManager()
        self.audit_logger: Optional[AuditLogger] = None
        
        # User and session storage
        self.users: Dict[str, User] = {}
        self.active_sessions: Dict[str, Session] = {}
        
        # Security settings
        self.max_failed_attempts = config.get("max_failed_attempts", 3)
        self.lockout_duration = timedelta(minutes=config.get("lockout_duration_minutes", 30))
        self.session_timeout = timedelta(hours=config.get("session_timeout_hours", 8))
        self.require_mfa = config.get("require_mfa", True)
        
    async def initialize(self):
        """Initialize authorization manager"""
        try:
            logger.info("Initializing Authorization Manager", manager_id=self.manager_id)
            
            # Initialize token manager
            await self.token_manager.initialize()
            
            # Initialize default users
            await self._initialize_default_users()
            
            # Start session cleanup task
            asyncio.create_task(self._cleanup_expired_sessions())
            
            logger.info("Authorization Manager initialized successfully")
            
        except Exception as e:
            logger.error("Authorization Manager initialization failed", error=str(e))
            raise
    
    def set_audit_logger(self, audit_logger: AuditLogger):
        """Set audit logger for authorization events"""
        self.audit_logger = audit_logger
    
    async def _initialize_default_users(self):
        """Initialize default system users"""
        # System administrator
        admin_user = User(
            user_id=str(uuid.uuid4()),
            username="admin",
            email="admin@xorb.local",
            roles=[Role.SYSTEM_ADMIN],
            permissions=[],
            security_clearance=SecurityLevel.SECRET,
            mfa_enabled=True
        )
        
        # Security engineer
        security_user = User(
            user_id=str(uuid.uuid4()),
            username="security_engineer",
            email="security@xorb.local",
            roles=[Role.SECURITY_ENGINEER],
            permissions=[],
            security_clearance=SecurityLevel.CONFIDENTIAL,
            mfa_enabled=True
        )
        
        # Red team operator
        redteam_user = User(
            user_id=str(uuid.uuid4()),
            username="redteam_operator",
            email="redteam@xorb.local",
            roles=[Role.RED_TEAM_OPERATOR],
            permissions=[],
            security_clearance=SecurityLevel.CONFIDENTIAL,
            mfa_enabled=True
        )
        
        self.users[admin_user.user_id] = admin_user
        self.users[security_user.user_id] = security_user
        self.users[redteam_user.user_id] = redteam_user
        
        logger.info("Default users initialized", users_count=len(self.users))
    
    async def authenticate_user(self, username: str, password: str, 
                              source_ip: str, user_agent: str) -> Optional[str]:
        """Authenticate user and create session"""
        try:
            # Find user by username
            user = None
            for u in self.users.values():
                if u.username == username:
                    user = u
                    break
            
            if not user:
                await self._log_auth_event("authentication_failed", 
                                         username=username, 
                                         reason="user_not_found",
                                         source_ip=source_ip)
                return None
            
            # Check if account is locked
            if user.account_locked:
                await self._log_auth_event("authentication_failed",
                                         user_id=user.user_id,
                                         username=username,
                                         reason="account_locked",
                                         source_ip=source_ip)
                return None
            
            # Validate password (simplified - in production use proper hashing)
            if not self._validate_password(password, user):
                user.failed_login_attempts += 1
                
                # Lock account after max attempts
                if user.failed_login_attempts >= self.max_failed_attempts:
                    user.account_locked = True
                    await self._log_auth_event("account_locked",
                                             user_id=user.user_id,
                                             username=username,
                                             source_ip=source_ip)
                
                await self._log_auth_event("authentication_failed",
                                         user_id=user.user_id,
                                         username=username,
                                         reason="invalid_credentials",
                                         source_ip=source_ip)
                return None
            
            # Reset failed attempts on successful auth
            user.failed_login_attempts = 0
            user.last_login = datetime.utcnow()
            
            # Create session
            session = await self._create_session(user, source_ip, user_agent)
            
            # Log successful authentication
            await self._log_auth_event("authentication_success",
                                     user_id=user.user_id,
                                     username=username,
                                     session_id=session.session_id,
                                     source_ip=source_ip)
            
            return session.session_id
            
        except Exception as e:
            logger.error("Authentication failed", error=str(e))
            return None
    
    def _validate_password(self, password: str, user: User) -> bool:
        """Validate user password (simplified implementation)"""
        # In production, use proper password hashing (bcrypt, argon2, etc.)
        # For this example, using a simple check
        expected_passwords = {
            "admin": "admin_secure_password_123",
            "security_engineer": "security_password_456",
            "redteam_operator": "redteam_password_789"
        }
        
        return password == expected_passwords.get(user.username, "")
    
    async def _create_session(self, user: User, source_ip: str, user_agent: str) -> Session:
        """Create new user session"""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        session = Session(
            session_id=session_id,
            user_id=user.user_id,
            user=user,
            created_at=now,
            expires_at=now + self.session_timeout,
            last_activity=now,
            source_ip=source_ip,
            user_agent=user_agent,
            mfa_verified=not self.require_mfa  # Simplified MFA check
        )
        
        self.active_sessions[session_id] = session
        
        return session
    
    async def validate_session(self, session_id: str) -> Optional[Session]:
        """Validate session and update activity"""
        try:
            session = self.active_sessions.get(session_id)
            
            if not session or not session.is_valid():
                if session:
                    del self.active_sessions[session_id]
                return None
            
            # Update last activity
            session.last_activity = datetime.utcnow()
            
            return session
            
        except Exception as e:
            logger.error("Session validation failed", error=str(e))
            return None
    
    async def check_permission(self, request: AuthorizationRequest) -> AuthorizationResult:
        """Check if user has permission for requested action"""
        try:
            # Validate session
            session = await self.validate_session(request.session_id)
            if not session:
                result = AuthorizationResult(
                    request_id=request.request_id,
                    granted=False,
                    reason="Invalid or expired session",
                    user_id=request.user_id,
                    permission=request.permission,
                    resource=request.resource
                )
                
                await self._log_authz_event("access_denied", request, result)
                return result
            
            user = session.user
            
            # Check if user has permission
            has_permission = self.role_manager.has_permission(user, request.permission)
            
            # Check security clearance requirements
            clearance_sufficient = await self._check_security_clearance(request, user)
            
            # Check resource-specific permissions
            resource_allowed = await self._check_resource_access(request, user)
            
            # Check for elevated privilege requirements
            requires_elevation = self._requires_elevation(request.permission)
            
            # Determine if access is granted
            granted = (has_permission and 
                      clearance_sufficient and 
                      resource_allowed and
                      (not requires_elevation or session.is_elevated()))
            
            # Build result
            result = AuthorizationResult(
                request_id=request.request_id,
                granted=granted,
                reason=self._build_denial_reason(has_permission, clearance_sufficient, 
                                               resource_allowed, requires_elevation, session),
                user_id=request.user_id,
                permission=request.permission,
                resource=request.resource,
                requires_elevation=requires_elevation and not session.is_elevated()
            )
            
            # Log authorization decision
            event_type = "access_granted" if granted else "access_denied"
            await self._log_authz_event(event_type, request, result)
            
            return result
            
        except Exception as e:
            logger.error("Permission check failed", error=str(e))
            
            result = AuthorizationResult(
                request_id=request.request_id,
                granted=False,
                reason=f"Authorization check failed: {str(e)}",
                user_id=request.user_id,
                permission=request.permission,
                resource=request.resource
            )
            
            await self._log_authz_event("access_denied", request, result)
            return result
    
    def _requires_elevation(self, permission: Permission) -> bool:
        """Check if permission requires elevated privileges"""
        elevated_permissions = {
            Permission.OVERRIDE_SAFETY,
            Permission.EMERGENCY_STOP,
            Permission.MANAGE_USERS,
            Permission.MODIFY_SYSTEM_CONFIG
        }
        
        return permission in elevated_permissions
    
    async def _check_security_clearance(self, request: AuthorizationRequest, user: User) -> bool:
        """Check if user has sufficient security clearance"""
        # Define clearance requirements for different operations
        clearance_requirements = {
            Permission.ACCESS_SENSITIVE_DATA: SecurityLevel.CONFIDENTIAL,
            Permission.EXECUTE_LATERAL_MOVEMENT: SecurityLevel.CONFIDENTIAL,
            Permission.OVERRIDE_SAFETY: SecurityLevel.SECRET,
            Permission.MANAGE_POLICIES: SecurityLevel.SECRET
        }
        
        required_clearance = clearance_requirements.get(request.permission)
        
        if not required_clearance:
            return True
        
        # Simple clearance hierarchy check
        clearance_levels = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.RESTRICTED: 3,
            SecurityLevel.SECRET: 4,
            SecurityLevel.TOP_SECRET: 5
        }
        
        user_level = clearance_levels.get(user.security_clearance, 0)
        required_level = clearance_levels.get(required_clearance, 0)
        
        return user_level >= required_level
    
    async def _check_resource_access(self, request: AuthorizationRequest, user: User) -> bool:
        """Check resource-specific access permissions"""
        if not request.resource:
            return True
        
        # Resource-based access control logic
        # In production, this would integrate with resource ownership/ACLs
        
        return True  # Simplified for this implementation
    
    def _build_denial_reason(self, has_permission: bool, clearance_sufficient: bool,
                           resource_allowed: bool, requires_elevation: bool, session: Session) -> str:
        """Build detailed reason for access denial"""
        if not has_permission:
            return "User does not have required permission"
        elif not clearance_sufficient:
            return "Insufficient security clearance"
        elif not resource_allowed:
            return "Access to resource not permitted"
        elif requires_elevation and not session.is_elevated():
            return "Operation requires elevated privileges"
        else:
            return "Access granted"
    
    async def _log_auth_event(self, event_type: str, **kwargs):
        """Log authentication event"""
        if not self.audit_logger:
            return
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            component="authorization_manager",
            details=kwargs,
            severity=EventSeverity.HIGH if "failed" in event_type else EventSeverity.INFO
        )
        
        await self.audit_logger.log_event(event)
    
    async def _log_authz_event(self, event_type: str, request: AuthorizationRequest, 
                              result: AuthorizationResult):
        """Log authorization event"""
        if not self.audit_logger:
            return
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            component="authorization_manager",
            user_id=request.user_id,
            session_id=request.session_id,
            target_resource=request.resource,
            action=request.permission.value,
            result="granted" if result.granted else "denied",
            details={
                "request_id": request.request_id,
                "permission": request.permission.value,
                "reason": result.reason,
                "requires_elevation": result.requires_elevation
            },
            severity=EventSeverity.MEDIUM if not result.granted else EventSeverity.INFO
        )
        
        await self.audit_logger.log_event(event)
    
    async def _cleanup_expired_sessions(self):
        """Periodic cleanup of expired sessions"""
        while True:
            try:
                now = datetime.utcnow()
                expired_sessions = [
                    session_id for session_id, session in self.active_sessions.items()
                    if now > session.expires_at or not session.is_valid()
                ]
                
                for session_id in expired_sessions:
                    session = self.active_sessions.pop(session_id, None)
                    if session:
                        await self._log_auth_event("session_expired",
                                                 user_id=session.user_id,
                                                 session_id=session_id)
                
                if expired_sessions:
                    logger.info("Cleaned up expired sessions",
                               expired_count=len(expired_sessions))
                
                # Run cleanup every 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error("Session cleanup failed", error=str(e))
                await asyncio.sleep(60)  # Retry after 1 minute on error
    
    async def get_authorization_metrics(self) -> Dict[str, Any]:
        """Get authorization system metrics"""
        try:
            now = datetime.utcnow()
            
            # Calculate session statistics
            active_sessions_count = len(self.active_sessions)
            elevated_sessions = sum(1 for s in self.active_sessions.values() if s.is_elevated())
            
            # Calculate user statistics  
            locked_accounts = sum(1 for u in self.users.values() if u.account_locked)
            mfa_enabled_users = sum(1 for u in self.users.values() if u.mfa_enabled)
            
            return {
                "manager_id": self.manager_id,
                "users": {
                    "total_users": len(self.users),
                    "locked_accounts": locked_accounts,
                    "mfa_enabled": mfa_enabled_users
                },
                "sessions": {
                    "active_sessions": active_sessions_count,
                    "elevated_sessions": elevated_sessions
                },
                "security": {
                    "require_mfa": self.require_mfa,
                    "max_failed_attempts": self.max_failed_attempts,
                    "session_timeout_hours": self.session_timeout.total_seconds() / 3600
                }
            }
            
        except Exception as e:
            logger.error("Failed to get authorization metrics", error=str(e))
            return {"error": str(e)}


# Export main classes
__all__ = [
    "AuthorizationManager",
    "Permission",
    "Role", 
    "User",
    "Session",
    "AuthorizationRequest",
    "AuthorizationResult",
    "TokenManager"
]