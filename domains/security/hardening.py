#!/usr/bin/env python3

import asyncio
import logging
import hashlib
import secrets
import time
import json
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import uuid

try:
    import bcrypt
    import jwt
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("Cryptography dependencies not available. Security features limited.")

import redis.asyncio as redis


class SecurityLevel(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    HIGH_SECURITY = "high_security"


class AccessLevel(str, Enum):
    READ_ONLY = "read_only"
    OPERATOR = "operator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


@dataclass
class SecurityPolicy:
    max_requests_per_hour: int = 1000
    max_concurrent_campaigns: int = 5
    allowed_ip_ranges: List[str] = field(default_factory=list)
    blocked_ip_addresses: Set[str] = field(default_factory=set)
    require_mfa: bool = False
    session_timeout_minutes: int = 480  # 8 hours
    password_min_length: int = 12
    password_require_special: bool = True
    audit_all_actions: bool = True
    encrypt_sensitive_data: bool = True


@dataclass
class UserSession:
    session_id: str
    user_id: str
    access_level: AccessLevel
    ip_address: str
    created_at: datetime
    last_activity: datetime
    mfa_verified: bool = False
    session_data: Dict[str, Any] = field(default_factory=dict)


class CryptoManager:
    """Handles encryption/decryption of sensitive data"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        self.logger = logging.getLogger(__name__)
        
        if not CRYPTO_AVAILABLE:
            self.logger.error("Cryptography not available - encryption disabled")
            self.cipher = None
            return
        
        if master_key:
            self.key = master_key
        else:
            # Generate key from environment or create new one
            self.key = self._generate_key()
        
        try:
            self.cipher = Fernet(base64.urlsafe_b64encode(self.key))
            self.logger.info("Encryption manager initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption: {e}")
            self.cipher = None

    def _generate_key(self) -> bytes:
        """Generate encryption key"""
        # In production, this should come from secure key management
        password = secrets.token_bytes(32)
        salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = kdf.derive(password)
        
        # Store salt securely (in production, use proper key management)
        with open('.crypto_salt', 'wb') as f:
            f.write(salt)
        
        return key

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not self.cipher:
            return data  # Return unencrypted if crypto unavailable
        
        try:
            encrypted_bytes = self.cipher.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_bytes).decode()
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return data

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not self.cipher:
            return encrypted_data  # Return as-is if crypto unavailable
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
            return decrypted_bytes.decode()
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            return encrypted_data


class RateLimiter:
    """Token bucket rate limiter with Redis backend"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.logger = logging.getLogger(__name__)

    async def check_rate_limit(self, 
                              key: str, 
                              max_requests: int, 
                              window_seconds: int) -> Tuple[bool, int]:
        """Check if request is within rate limit"""
        try:
            current_time = int(time.time())
            window_start = current_time - window_seconds
            
            # Use sliding window counter
            pipe = self.redis.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(f"rate_limit:{key}", 0, window_start)
            
            # Count current requests in window
            pipe.zcard(f"rate_limit:{key}")
            
            # Add current request
            pipe.zadd(f"rate_limit:{key}", {str(uuid.uuid4()): current_time})
            
            # Set expiry
            pipe.expire(f"rate_limit:{key}", window_seconds + 60)
            
            results = await pipe.execute()
            current_count = results[1]
            
            # Check if under limit (subtract 1 because we already added current request)
            allowed = (current_count - 1) < max_requests
            remaining = max(0, max_requests - current_count)
            
            return allowed, remaining
            
        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            return True, max_requests  # Fail open

    async def get_rate_limit_status(self, key: str, window_seconds: int) -> Dict[str, Any]:
        """Get detailed rate limit status"""
        try:
            current_time = int(time.time())
            window_start = current_time - window_seconds
            
            # Get all requests in current window
            requests = await self.redis.zrangebyscore(
                f"rate_limit:{key}", window_start, current_time, withscores=True
            )
            
            return {
                "current_count": len(requests),
                "window_start": window_start,
                "window_end": current_time,
                "requests_timeline": [(req[0].decode(), int(req[1])) for req in requests[-10:]]
            }
            
        except Exception as e:
            self.logger.error(f"Rate limit status check failed: {e}")
            return {"error": str(e)}


class SessionManager:
    """Secure session management with Redis storage"""
    
    def __init__(self, redis_client: redis.Redis, crypto_manager: CryptoManager):
        self.redis = redis_client
        self.crypto = crypto_manager
        self.sessions: Dict[str, UserSession] = {}
        self.logger = logging.getLogger(__name__)

    async def create_session(self, 
                           user_id: str, 
                           access_level: AccessLevel, 
                           ip_address: str,
                           timeout_minutes: int = 480) -> str:
        """Create new authenticated session"""
        session_id = secrets.token_urlsafe(32)
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            access_level=access_level,
            ip_address=ip_address,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow()
        )
        
        # Store in Redis with expiration
        session_data = {
            "user_id": user_id,
            "access_level": access_level.value,
            "ip_address": ip_address,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "mfa_verified": False
        }
        
        # Encrypt sensitive session data
        encrypted_data = self.crypto.encrypt_data(json.dumps(session_data))
        
        await self.redis.setex(
            f"session:{session_id}",
            timeout_minutes * 60,
            encrypted_data
        )
        
        self.sessions[session_id] = session
        self.logger.info(f"Created session for user {user_id} from {ip_address}")
        
        return session_id

    async def validate_session(self, session_id: str, ip_address: str) -> Optional[UserSession]:
        """Validate session and check IP"""
        try:
            # Check Redis first
            encrypted_data = await self.redis.get(f"session:{session_id}")
            if not encrypted_data:
                return None
            
            # Decrypt session data
            session_json = self.crypto.decrypt_data(encrypted_data.decode())
            session_data = json.loads(session_json)
            
            # Verify IP address matches
            if session_data["ip_address"] != ip_address:
                self.logger.warning(f"IP mismatch for session {session_id}: {ip_address} vs {session_data['ip_address']}")
                await self.invalidate_session(session_id)
                return None
            
            # Update last activity
            session_data["last_activity"] = datetime.utcnow().isoformat()
            
            # Re-encrypt and store
            updated_encrypted = self.crypto.encrypt_data(json.dumps(session_data))
            await self.redis.setex(
                f"session:{session_id}",
                480 * 60,  # Refresh timeout
                updated_encrypted
            )
            
            # Return session object
            session = UserSession(
                session_id=session_id,
                user_id=session_data["user_id"],
                access_level=AccessLevel(session_data["access_level"]),
                ip_address=session_data["ip_address"],
                created_at=datetime.fromisoformat(session_data["created_at"]),
                last_activity=datetime.fromisoformat(session_data["last_activity"]),
                mfa_verified=session_data.get("mfa_verified", False)
            )
            
            self.sessions[session_id] = session
            return session
            
        except Exception as e:
            self.logger.error(f"Session validation failed: {e}")
            return None

    async def invalidate_session(self, session_id: str):
        """Invalidate a session"""
        await self.redis.delete(f"session:{session_id}")
        self.sessions.pop(session_id, None)
        self.logger.info(f"Invalidated session {session_id}")

    async def cleanup_expired_sessions(self):
        """Remove expired sessions from memory"""
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if (current_time - session.last_activity).total_seconds() > 8 * 3600:  # 8 hours
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


class IPAccessControl:
    """IP-based access control and geofencing"""
    
    def __init__(self):
        self.allowed_networks: List[ipaddress.IPv4Network] = []
        self.blocked_ips: Set[ipaddress.IPv4Address] = set()
        self.suspicious_ips: Dict[str, List[datetime]] = {}
        self.logger = logging.getLogger(__name__)

    def add_allowed_network(self, network_cidr: str):
        """Add allowed network range"""
        try:
            network = ipaddress.IPv4Network(network_cidr, strict=False)
            self.allowed_networks.append(network)
            self.logger.info(f"Added allowed network: {network_cidr}")
        except Exception as e:
            self.logger.error(f"Invalid network CIDR {network_cidr}: {e}")

    def block_ip(self, ip_address: str):
        """Block specific IP address"""
        try:
            ip = ipaddress.IPv4Address(ip_address)
            self.blocked_ips.add(ip)
            self.logger.warning(f"Blocked IP: {ip_address}")
        except Exception as e:
            self.logger.error(f"Invalid IP address {ip_address}: {e}")

    def is_ip_allowed(self, ip_address: str) -> Tuple[bool, str]:
        """Check if IP is allowed"""
        try:
            ip = ipaddress.IPv4Address(ip_address)
            
            # Check if blocked
            if ip in self.blocked_ips:
                return False, "IP is blocked"
            
            # Check if in allowed networks (if any configured)
            if self.allowed_networks:
                for network in self.allowed_networks:
                    if ip in network:
                        return True, "IP in allowed network"
                return False, "IP not in allowed networks"
            
            # If no networks configured, allow all (except blocked)
            return True, "No network restrictions"
            
        except Exception as e:
            self.logger.error(f"IP validation failed for {ip_address}: {e}")
            return False, "Invalid IP format"

    def record_suspicious_activity(self, ip_address: str):
        """Record suspicious activity from IP"""
        current_time = datetime.utcnow()
        
        if ip_address not in self.suspicious_ips:
            self.suspicious_ips[ip_address] = []
        
        self.suspicious_ips[ip_address].append(current_time)
        
        # Keep only last 24 hours
        cutoff_time = current_time - timedelta(hours=24)
        self.suspicious_ips[ip_address] = [
            t for t in self.suspicious_ips[ip_address] if t > cutoff_time
        ]
        
        # Auto-block if too many suspicious activities
        if len(self.suspicious_ips[ip_address]) > 10:
            self.block_ip(ip_address)
            self.logger.warning(f"Auto-blocked IP {ip_address} for suspicious activity")


class AuditLogger:
    """Comprehensive audit logging for security events"""
    
    def __init__(self, log_file: str = "xorb_security_audit.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup dedicated audit logger
        self.audit_logger = logging.getLogger("xorb_audit")
        self.audit_logger.setLevel(logging.INFO)
        
        # Create file handler
        handler = logging.FileHandler(self.log_file)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s|%(name)s|%(levelname)s|%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S UTC'
        )
        handler.setFormatter(formatter)
        
        self.audit_logger.addHandler(handler)

    def log_security_event(self, 
                          event_type: str, 
                          user_id: str, 
                          ip_address: str, 
                          details: Dict[str, Any]):
        """Log security event"""
        event_data = {
            "event_type": event_type,
            "user_id": user_id,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details
        }
        
        self.audit_logger.info(json.dumps(event_data))

    def log_access_attempt(self, 
                          user_id: str, 
                          ip_address: str, 
                          success: bool, 
                          reason: str = ""):
        """Log authentication attempt"""
        self.log_security_event(
            "authentication",
            user_id,
            ip_address,
            {
                "success": success,
                "reason": reason,
                "action": "login_attempt"
            }
        )

    def log_admin_action(self, 
                        user_id: str, 
                        ip_address: str, 
                        action: str, 
                        resource: str,
                        details: Dict[str, Any] = None):
        """Log administrative action"""
        self.log_security_event(
            "admin_action",
            user_id,
            ip_address,
            {
                "action": action,
                "resource": resource,
                "details": details or {}
            }
        )

    def log_rate_limit_violation(self, 
                                ip_address: str, 
                                endpoint: str, 
                                limit_exceeded: int):
        """Log rate limit violation"""
        self.log_security_event(
            "rate_limit_violation",
            "system",
            ip_address,
            {
                "endpoint": endpoint,
                "limit_exceeded_by": limit_exceeded
            }
        )


class XORBSecurityManager:
    """Main security manager coordinating all security components"""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379",
                 security_level: SecurityLevel = SecurityLevel.PRODUCTION):
        
        self.security_level = security_level
        self.redis_client = redis.from_url(redis_url)
        
        # Initialize security components
        self.crypto_manager = CryptoManager()
        self.rate_limiter = RateLimiter(self.redis_client)
        self.session_manager = SessionManager(self.redis_client, self.crypto_manager)
        self.ip_access_control = IPAccessControl()
        self.audit_logger = AuditLogger()
        
        # Load security policy
        self.security_policy = self._get_security_policy_for_level(security_level)
        
        # Initialize security state
        self.active_sessions: Dict[str, UserSession] = {}
        self.failed_auth_attempts: Dict[str, List[datetime]] = {}
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Security manager initialized with level: {security_level.value}")

    def _get_security_policy_for_level(self, level: SecurityLevel) -> SecurityPolicy:
        """Get security policy based on security level"""
        policies = {
            SecurityLevel.DEVELOPMENT: SecurityPolicy(
                max_requests_per_hour=10000,
                max_concurrent_campaigns=10,
                require_mfa=False,
                session_timeout_minutes=1440,  # 24 hours
                password_min_length=8,
                password_require_special=False,
                audit_all_actions=False,
                encrypt_sensitive_data=False
            ),
            
            SecurityLevel.STAGING: SecurityPolicy(
                max_requests_per_hour=5000,
                max_concurrent_campaigns=8,
                require_mfa=False,
                session_timeout_minutes=720,  # 12 hours
                password_min_length=10,
                password_require_special=True,
                audit_all_actions=True,
                encrypt_sensitive_data=True
            ),
            
            SecurityLevel.PRODUCTION: SecurityPolicy(
                max_requests_per_hour=1000,
                max_concurrent_campaigns=5,
                require_mfa=True,
                session_timeout_minutes=480,  # 8 hours
                password_min_length=12,
                password_require_special=True,
                audit_all_actions=True,
                encrypt_sensitive_data=True
            ),
            
            SecurityLevel.HIGH_SECURITY: SecurityPolicy(
                max_requests_per_hour=500,
                max_concurrent_campaigns=3,
                require_mfa=True,
                session_timeout_minutes=240,  # 4 hours
                password_min_length=16,
                password_require_special=True,
                audit_all_actions=True,
                encrypt_sensitive_data=True
            )
        }
        
        return policies.get(level, policies[SecurityLevel.PRODUCTION])

    async def authenticate_user(self, 
                               username: str, 
                               password: str, 
                               ip_address: str,
                               mfa_token: Optional[str] = None) -> Tuple[bool, Optional[str], str]:
        """Authenticate user with comprehensive security checks"""
        
        # Check IP access
        ip_allowed, ip_reason = self.ip_access_control.is_ip_allowed(ip_address)
        if not ip_allowed:
            self.audit_logger.log_access_attempt(username, ip_address, False, f"IP blocked: {ip_reason}")
            return False, None, f"Access denied: {ip_reason}"
        
        # Check for brute force attempts
        if await self._is_brute_force_attempt(username, ip_address):
            self.audit_logger.log_access_attempt(username, ip_address, False, "Brute force detected")
            self.ip_access_control.record_suspicious_activity(ip_address)
            return False, None, "Too many failed attempts. Access temporarily blocked."
        
        # Validate credentials (simplified - in production use proper user DB)
        if not await self._validate_credentials(username, password):
            await self._record_failed_attempt(username, ip_address)
            self.audit_logger.log_access_attempt(username, ip_address, False, "Invalid credentials")
            return False, None, "Invalid credentials"
        
        # Check MFA if required
        if self.security_policy.require_mfa and not await self._validate_mfa(username, mfa_token):
            self.audit_logger.log_access_attempt(username, ip_address, False, "MFA required")
            return False, None, "MFA token required"
        
        # Create session
        access_level = await self._get_user_access_level(username)
        session_id = await self.session_manager.create_session(
            username, access_level, ip_address, self.security_policy.session_timeout_minutes
        )
        
        self.audit_logger.log_access_attempt(username, ip_address, True, "Successful login")
        
        return True, session_id, "Authentication successful"

    async def check_request_authorization(self, 
                                        session_id: str, 
                                        ip_address: str,
                                        endpoint: str,
                                        required_access_level: AccessLevel = AccessLevel.READ_ONLY) -> Tuple[bool, str]:
        """Check if request is authorized"""
        
        # Validate session
        session = await self.session_manager.validate_session(session_id, ip_address)
        if not session:
            return False, "Invalid or expired session"
        
        # Check access level
        access_levels = {
            AccessLevel.READ_ONLY: 1,
            AccessLevel.OPERATOR: 2,
            AccessLevel.ADMIN: 3,
            AccessLevel.SUPER_ADMIN: 4
        }
        
        user_level = access_levels.get(session.access_level, 0)
        required_level = access_levels.get(required_access_level, 1)
        
        if user_level < required_level:
            self.audit_logger.log_security_event(
                "authorization_denied",
                session.user_id,
                ip_address,
                {
                    "endpoint": endpoint,
                    "user_access_level": session.access_level.value,
                    "required_access_level": required_access_level.value
                }
            )
            return False, "Insufficient access level"
        
        # Check rate limits
        rate_limit_key = f"{session.user_id}:{ip_address}"
        allowed, remaining = await self.rate_limiter.check_rate_limit(
            rate_limit_key,
            self.security_policy.max_requests_per_hour,
            3600  # 1 hour
        )
        
        if not allowed:
            self.audit_logger.log_rate_limit_violation(ip_address, endpoint, remaining)
            return False, "Rate limit exceeded"
        
        return True, "Authorized"

    async def _is_brute_force_attempt(self, username: str, ip_address: str) -> bool:
        """Check for brute force attempts"""
        current_time = datetime.utcnow()
        key = f"{username}:{ip_address}"
        
        if key in self.failed_auth_attempts:
            # Remove attempts older than 1 hour
            cutoff_time = current_time - timedelta(hours=1)
            self.failed_auth_attempts[key] = [
                attempt for attempt in self.failed_auth_attempts[key]
                if attempt > cutoff_time
            ]
            
            # Check if too many attempts
            if len(self.failed_auth_attempts[key]) >= 5:
                return True
        
        return False

    async def _record_failed_attempt(self, username: str, ip_address: str):
        """Record failed authentication attempt"""
        key = f"{username}:{ip_address}"
        current_time = datetime.utcnow()
        
        if key not in self.failed_auth_attempts:
            self.failed_auth_attempts[key] = []
        
        self.failed_auth_attempts[key].append(current_time)

    async def _validate_credentials(self, username: str, password: str) -> bool:
        """Validate user credentials (simplified implementation)"""
        # In production, this would check against secure user database
        # For demo purposes, accept specific test credentials
        test_users = {
            "admin": "SecureP@ssw0rd123",
            "operator": "Op3rat0rP@ss",
            "viewer": "Vi3w3rP@ss"
        }
        
        return username in test_users and test_users[username] == password

    async def _validate_mfa(self, username: str, mfa_token: Optional[str]) -> bool:
        """Validate MFA token (simplified implementation)"""
        if not mfa_token:
            return False
        
        # In production, this would validate TOTP/SMS/hardware tokens
        # For demo, accept any 6-digit number
        return len(mfa_token) == 6 and mfa_token.isdigit()

    async def _get_user_access_level(self, username: str) -> AccessLevel:
        """Get user's access level (simplified)"""
        access_mapping = {
            "admin": AccessLevel.ADMIN,
            "operator": AccessLevel.OPERATOR,
            "viewer": AccessLevel.READ_ONLY
        }
        
        return access_mapping.get(username, AccessLevel.READ_ONLY)

    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        return {
            "security_level": self.security_level.value,
            "active_sessions": len(self.session_manager.sessions),
            "blocked_ips": len(self.ip_access_control.blocked_ips),
            "suspicious_ips": len(self.ip_access_control.suspicious_ips),
            "crypto_available": CRYPTO_AVAILABLE,
            "security_policy": {
                "max_requests_per_hour": self.security_policy.max_requests_per_hour,
                "max_concurrent_campaigns": self.security_policy.max_concurrent_campaigns,
                "require_mfa": self.security_policy.require_mfa,
                "session_timeout_minutes": self.security_policy.session_timeout_minutes,
                "audit_enabled": self.security_policy.audit_all_actions,
                "encryption_enabled": self.security_policy.encrypt_sensitive_data
            }
        }

    async def shutdown(self):
        """Clean shutdown of security manager"""
        self.logger.info("Shutting down security manager")
        
        # Close Redis connections
        await self.redis_client.close()
        
        # Clear sensitive data from memory
        self.session_manager.sessions.clear()
        self.failed_auth_attempts.clear()


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    async def demo_security():
        """Demonstrate security features"""
        print("=== XORB Security Manager Demo ===")
        
        # Initialize security manager
        security_manager = XORBSecurityManager(
            security_level=SecurityLevel.PRODUCTION
        )
        
        # Test IP access control
        security_manager.ip_access_control.add_allowed_network("192.168.1.0/24")
        security_manager.ip_access_control.add_allowed_network("10.0.0.0/8")
        
        test_ip = "192.168.1.100"
        allowed, reason = security_manager.ip_access_control.is_ip_allowed(test_ip)
        print(f"IP {test_ip} allowed: {allowed} ({reason})")
        
        # Test authentication
        success, session_id, message = await security_manager.authenticate_user(
            "admin", "SecureP@ssw0rd123", test_ip
        )
        print(f"Authentication: {success} - {message}")
        
        if success:
            print(f"Session ID: {session_id}")
            
            # Test authorization
            authorized, auth_message = await security_manager.check_request_authorization(
                session_id, test_ip, "/api/campaigns", AccessLevel.ADMIN
            )
            print(f"Authorization: {authorized} - {auth_message}")
        
        # Show security status
        status = await security_manager.get_security_status()
        print(f"Security Status: {json.dumps(status, indent=2)}")
        
        await security_manager.shutdown()
    
    if "--demo" in sys.argv:
        asyncio.run(demo_security())
    elif "--install-deps" in sys.argv:
        print("To install security dependencies:")
        print("pip install bcrypt PyJWT cryptography")
    else:
        print("XORB Security Manager")
        print("Usage:")
        print("  python hardening.py --demo        # Run demo")
        print("  python hardening.py --install-deps # Show installation instructions")