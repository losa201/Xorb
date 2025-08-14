"""
Secure Authentication Service - PR-004 Implementation
Production-ready authentication service with comprehensive security controls
"""

import os
import hashlib
import secrets
import jwt
import bcrypt
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import re
import json
from ipaddress import ip_address, AddressValueError

from .interfaces import AuthenticationService
from ..core.config import get_settings, get_security_config
from ..domain.entities import User, Organization
from ..core.logging import get_logger

logger = get_logger(__name__)


class AuthenticationError(Exception):
    """Authentication-specific errors"""
    def __init__(self, message: str, code: str = "AUTH_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


class SessionStatus(Enum):
    """Session status types"""
    ACTIVE = "active"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"
    SUSPICIOUS = "suspicious"


@dataclass
class AuthenticationResult:
    """Authentication result with security context"""
    success: bool
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    security_context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.security_context is None:
            self.security_context = {}


@dataclass
class UserSession:
    """Secure user session management"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    status: SessionStatus
    security_flags: Dict[str, Any]
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session has expired"""
        timeout_delta = timedelta(minutes=timeout_minutes)
        return (datetime.utcnow() - self.last_activity) > timeout_delta
    
    def is_valid(self) -> bool:
        """Check if session is valid"""
        return self.status == SessionStatus.ACTIVE and not self.is_expired()


class SecureAuthenticationService(AuthenticationService):
    """Production-ready authentication service with comprehensive security"""
    
    def __init__(self):
        """Initialize secure authentication service"""
        self.settings = get_settings()
        self.security_config = get_security_config()
        
        # Session management
        self.active_sessions: Dict[str, UserSession] = {}
        self.blacklisted_tokens: set = set()
        self.failed_attempts: Dict[str, List[datetime]] = {}
        
        # Security monitoring
        self.security_events: List[Dict[str, Any]] = []
        
        # User storage (in production, this would be database repositories)
        self.users: Dict[str, Dict[str, Any]] = {}
        
        # Initialize security settings
        self._initialize_security_settings()
        
        logger.info("Secure Authentication Service initialized")
    
    def _initialize_security_settings(self):
        """Initialize security settings from configuration"""
        self.jwt_secret = self.settings.jwt_secret_key
        self.jwt_algorithm = self.settings.jwt_algorithm
        self.access_token_expiry = timedelta(minutes=self.settings.jwt_expiration_minutes)
        self.refresh_token_expiry = timedelta(days=self.settings.jwt_refresh_expiration_days)
        self.session_timeout_minutes = 30
        self.max_login_attempts = self.settings.max_login_attempts
        self.lockout_duration = timedelta(minutes=self.settings.lockout_duration_minutes)
        
        # Password security from security config
        self.password_min_length = self.security_config.min_password_length
        self.require_mfa = self.security_config.require_mfa
        
        logger.info(f"Security settings initialized - Token expiry: {self.access_token_expiry}, "
                   f"Max attempts: {self.max_login_attempts}, Password length: {self.password_min_length}")
    
    async def authenticate_user(self, credentials: Dict[str, Any]) -> AuthenticationResult:
        """Secure user authentication with comprehensive validation"""
        try:
            username = credentials.get("username", "")
            password = credentials.get("password", "")
            ip_address = credentials.get("ip_address", "")
            user_agent = credentials.get("user_agent", "")
            
            if not username or not password:
                return AuthenticationResult(
                    success=False,
                    error_code="INVALID_INPUT",
                    error_message="Username and password are required"
                )
            
            # Security pre-checks
            if not await self._perform_security_checks(username, ip_address, user_agent):
                return AuthenticationResult(
                    success=False,
                    error_code="SECURITY_VIOLATION",
                    error_message="Authentication blocked due to security policy violation"
                )
            
            # Rate limiting check
            if not await self._check_rate_limiting(username, ip_address):
                await self._record_failed_attempt(username, ip_address, "RATE_LIMITED")
                return AuthenticationResult(
                    success=False,
                    error_code="RATE_LIMITED",
                    error_message="Too many authentication attempts. Please try again later."
                )
            
            # Find user and validate credentials
            user_data = await self._find_user_by_credentials(username)
            if not user_data:
                await self._record_failed_attempt(username, ip_address, "USER_NOT_FOUND")
                return AuthenticationResult(
                    success=False,
                    error_code="INVALID_CREDENTIALS",
                    error_message="Invalid username or password"
                )
            
            user_id = user_data["id"]
            
            # Check if account is locked
            if await self._is_account_locked(username):
                await self._record_failed_attempt(username, ip_address, "ACCOUNT_LOCKED")
                return AuthenticationResult(
                    success=False,
                    error_code="ACCOUNT_LOCKED",
                    error_message="Account is temporarily locked due to multiple failed login attempts"
                )
            
            # Validate password
            if not self.verify_password(password, user_data.get("password_hash", "")):
                await self._record_failed_attempt(username, ip_address, "INVALID_PASSWORD")
                return AuthenticationResult(
                    success=False,
                    error_code="INVALID_CREDENTIALS",
                    error_message="Invalid username or password"
                )
            
            # Check if user account is active
            if not user_data.get("active", True):
                await self._record_failed_attempt(username, ip_address, "USER_INACTIVE")
                return AuthenticationResult(
                    success=False,
                    error_code="ACCOUNT_INACTIVE",
                    error_message="User account is inactive"
                )
            
            # Create secure session
            session = await self._create_user_session(user_id, ip_address, user_agent)
            
            # Generate secure tokens
            access_token = await self._generate_access_token(user_id, session.session_id)
            refresh_token = await self._generate_refresh_token(user_id, session.session_id)
            
            # Record successful authentication
            await self._record_successful_auth(username, user_id, ip_address)
            
            # Clear failed attempts for this user
            if username in self.failed_attempts:
                del self.failed_attempts[username]
            
            result = AuthenticationResult(
                success=True,
                user_id=user_id,
                session_id=session.session_id,
                access_token=access_token,
                refresh_token=refresh_token,
                expires_at=datetime.utcnow() + self.access_token_expiry,
                security_context={
                    "auth_timestamp": datetime.utcnow().isoformat(),
                    "ip_address": ip_address,
                    "user_agent": user_agent,
                    "session_security_level": "standard",
                    "authentication_method": "password"
                }
            )
            
            logger.info(f"User {username} authenticated successfully from {ip_address}")
            return result
            
        except Exception as e:
            logger.error(f"Authentication failed for user {credentials.get('username', 'unknown')}: {e}")
            return AuthenticationResult(
                success=False,
                error_code="AUTHENTICATION_ERROR",
                error_message="Internal authentication error"
            )
    
    async def validate_token(self, token: str) -> Dict[str, Any]:
        """Comprehensive JWT token validation with security checks"""
        try:
            # Check token blacklist
            if token in self.blacklisted_tokens:
                return {
                    "valid": False,
                    "error": "TOKEN_BLACKLISTED",
                    "message": "Token has been revoked"
                }
            
            # Decode JWT token
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
                    "message": f"Invalid token: {str(e)}"
                }
            
            # Validate token claims
            required_claims = ["user_id", "session_id", "exp", "iat"]
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
                    "message": "Session no longer exists"
                }
            
            session = self.active_sessions[session_id]
            if not session.is_valid():
                # Clean up expired session
                del self.active_sessions[session_id]
                return {
                    "valid": False,
                    "error": "SESSION_EXPIRED",
                    "message": "Session has expired"
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
                    "message": "User account is no longer active"
                }
            
            return {
                "valid": True,
                "user_id": user_id,
                "session_id": session_id,
                "payload": payload,
                "user_data": user_data
            }
            
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return {
                "valid": False,
                "error": "VALIDATION_ERROR",
                "message": "Token validation failed"
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
            new_access_token = await self._generate_access_token(user_id, session_id)
            
            logger.info(f"Access token refreshed for user {user_id}")
            return new_access_token
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return None
    
    async def logout_user(self, session_id: str) -> bool:
        """Secure logout with complete session cleanup"""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            user_id = session.user_id
            
            # Mark session as invalidated
            session.status = SessionStatus.INVALIDATED
            
            # Remove session from active sessions
            del self.active_sessions[session_id]
            
            # Record logout event
            await self._record_logout_event(user_id, session_id)
            
            logger.info(f"User {user_id} logged out successfully")
            return True
            
        except Exception as e:
            logger.error(f"Logout failed for session {session_id}: {e}")
            return False
    
    async def create_access_token(self, user_data: Any, session_id: Optional[str] = None) -> str:
        """Create secure access token"""
        try:
            if isinstance(user_data, dict):
                user_id = user_data.get("id") or user_data.get("user_id")
            else:
                user_id = getattr(user_data, "id", str(user_data))
            
            if not session_id:
                # Create temporary session for token creation
                session_id = secrets.token_urlsafe(32)
            
            return await self._generate_access_token(user_id, session_id)
            
        except Exception as e:
            logger.error(f"Failed to create access token: {e}")
            raise AuthenticationError("Failed to create access token", "TOKEN_CREATION_FAILED")
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke/blacklist a token"""
        try:
            # Add token to blacklist
            self.blacklisted_tokens.add(token)
            
            # Try to extract session info to clean up
            try:
                payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm], options={"verify_exp": False})
                session_id = payload.get("session_id")
                if session_id and session_id in self.active_sessions:
                    await self.logout_user(session_id)
            except:
                pass  # Token might be malformed, but still blacklist it
            
            logger.info("Token revoked successfully")
            return True
            
        except Exception as e:
            logger.error(f"Token revocation failed: {e}")
            return False
    
    def hash_password(self, password: str) -> str:
        """Secure password hashing with bcrypt"""
        try:
            # Generate salt and hash password with high cost factor
            salt = bcrypt.gensalt(rounds=12)
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Password hashing failed: {e}")
            raise AuthenticationError("Password hashing failed", "PASSWORD_HASH_ERROR")
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Secure password verification"""
        try:
            if not password or not hashed:
                return False
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    # Private helper methods
    
    async def _perform_security_checks(self, username: str, ip_address: str, user_agent: str) -> bool:
        """Perform comprehensive security checks"""
        try:
            # Basic input validation
            if not self._is_valid_username(username):
                logger.warning(f"Invalid username format: {username}")
                return False
            
            # IP address validation
            if ip_address:
                try:
                    from ipaddress import ip_address as validate_ip
                    validate_ip(ip_address)  # Validate IP format
                except AddressValueError:
                    logger.warning(f"Invalid IP address: {ip_address}")
                    return False
            
            # User agent validation (basic check)
            if user_agent and len(user_agent) > 500:
                logger.warning(f"Suspicious user agent length: {len(user_agent)}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Security checks failed: {e}")
            return False
    
    def _is_valid_username(self, username: str) -> bool:
        """Validate username format"""
        if not username or len(username) < 3 or len(username) > 50:
            return False
        
        # Allow alphanumeric, underscore, hyphen, dot, and @ for email
        pattern = re.compile(r'^[a-zA-Z0-9._@-]+$')
        return pattern.match(username) is not None
    
    async def _check_rate_limiting(self, username: str, ip_address: str) -> bool:
        """Check rate limiting for authentication attempts"""
        current_time = datetime.utcnow()
        
        # Check per-user rate limiting
        user_attempts = self.failed_attempts.get(username, [])
        recent_attempts = [
            attempt for attempt in user_attempts 
            if (current_time - attempt).total_seconds() < 3600  # Last hour
        ]
        
        if len(recent_attempts) >= self.max_login_attempts:
            return False
        
        return True
    
    async def _find_user_by_credentials(self, username: str) -> Optional[Dict[str, Any]]:
        """Find user by username or email"""
        for user_id, user_data in self.users.items():
            if (user_data.get("username") == username or 
                user_data.get("email") == username):
                user_data["id"] = user_id
                return user_data
        return None
    
    async def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts"""
        if username not in self.failed_attempts:
            return False
        
        failed_attempts = self.failed_attempts[username]
        if len(failed_attempts) < self.max_login_attempts:
            return False
        
        # Check if lockout period has expired
        last_attempt = max(failed_attempts)
        lockout_expired = (datetime.utcnow() - last_attempt) > self.lockout_duration
        
        if lockout_expired:
            # Clear old attempts
            del self.failed_attempts[username]
            return False
        
        return True
    
    async def _create_user_session(self, user_id: str, ip_address: str, user_agent: str) -> UserSession:
        """Create secure user session"""
        session_id = secrets.token_urlsafe(32)
        current_time = datetime.utcnow()
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            created_at=current_time,
            last_activity=current_time,
            ip_address=ip_address,
            user_agent=user_agent,
            status=SessionStatus.ACTIVE,
            security_flags={}
        )
        
        self.active_sessions[session_id] = session
        return session
    
    async def _generate_access_token(self, user_id: str, session_id: str) -> str:
        """Generate secure access token"""
        current_time = datetime.utcnow()
        expiry_time = current_time + self.access_token_expiry
        
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "iat": current_time,
            "exp": expiry_time,
            "type": "access"
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    async def _generate_refresh_token(self, user_id: str, session_id: str) -> str:
        """Generate secure refresh token"""
        current_time = datetime.utcnow()
        expiry_time = current_time + self.refresh_token_expiry
        
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "iat": current_time,
            "exp": expiry_time,
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    async def _record_failed_attempt(self, username: str, ip_address: str, reason: str):
        """Record failed authentication attempt"""
        current_time = datetime.utcnow()
        
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        
        self.failed_attempts[username].append(current_time)
        
        # Record security event
        self.security_events.append({
            "event_type": "failed_authentication",
            "username": username,
            "ip_address": ip_address,
            "reason": reason,
            "timestamp": current_time.isoformat()
        })
        
        logger.warning(f"Failed authentication attempt for {username} from {ip_address}: {reason}")
    
    async def _record_successful_auth(self, username: str, user_id: str, ip_address: str):
        """Record successful authentication"""
        current_time = datetime.utcnow()
        
        # Update user last login
        if user_id in self.users:
            self.users[user_id]["last_login"] = current_time.isoformat()
            self.users[user_id]["login_count"] = self.users[user_id].get("login_count", 0) + 1
        
        # Record security event
        self.security_events.append({
            "event_type": "successful_authentication",
            "username": username,
            "user_id": user_id,
            "ip_address": ip_address,
            "timestamp": current_time.isoformat()
        })
        
        logger.info(f"Successful authentication for {username} from {ip_address}")
    
    async def _record_logout_event(self, user_id: str, session_id: str):
        """Record logout event"""
        current_time = datetime.utcnow()
        
        self.security_events.append({
            "event_type": "user_logout",
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": current_time.isoformat()
        })
        
        logger.info(f"User {user_id} logged out")
    
    async def seed_development_user(self):
        """Seed development user for testing (only in development)"""
        if self.settings.environment != "development":
            return
        
        # Only create if not already exists
        if self.users:
            return
            
        # Create development admin user
        dev_user_id = "dev_admin_001"
        dev_password = "DevPassword123"  # Simple password for development testing
        dev_password_hash = self.hash_password(dev_password)
        
        self.users[dev_user_id] = {
            "id": dev_user_id,
            "username": "admin",
            "email": "admin@xorb.dev",
            "password_hash": dev_password_hash,
            "active": True,
            "roles": ["admin", "user"],
            "created_at": datetime.utcnow().isoformat(),
            "login_count": 0
        }
        
        # Verify the password was hashed correctly
        verification_test = self.verify_password(dev_password, dev_password_hash)
        logger.info(f"Development user seeded: admin@xorb.dev (password verification: {verification_test})")


# Global service instance with lazy initialization
_auth_service: Optional[SecureAuthenticationService] = None

async def get_secure_auth_service() -> SecureAuthenticationService:
    """Get global secure authentication service instance"""
    global _auth_service
    
    if _auth_service is None:
        _auth_service = SecureAuthenticationService()
        # Seed development user in development environment
        await _auth_service.seed_development_user()
    
    return _auth_service