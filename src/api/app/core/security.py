"""
Production security hardening and best practices implementation
"""

import secrets
import hashlib
import hmac
import base64
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import re
import ipaddress
from pathlib import Path

import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import jwt
from passlib.context import CryptContext
from passlib.hash import argon2

from .logging import get_logger, security_logger

logger = get_logger(__name__)


class SecurityLevel(Enum):
    """Security level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    # JWT settings
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    jwt_refresh_expiration_days: int = 30
    
    # Password policy
    min_password_length: int = 12
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_special_chars: bool = True
    password_history_count: int = 5
    
    # Rate limiting
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    rate_limit_window_seconds: int = 3600
    
    # Session management
    session_timeout_minutes: int = 480  # 8 hours
    concurrent_sessions_limit: int = 3
    require_mfa: bool = True
    
    # API security
    api_key_length: int = 32
    webhook_signature_tolerance_seconds: int = 300
    cors_max_age: int = 3600
    
    # Encryption
    encryption_key_rotation_days: int = 90
    enable_field_level_encryption: bool = True
    
    # Content security
    max_file_upload_size_mb: int = 10
    allowed_file_extensions: List[str] = None
    enable_content_scanning: bool = True


class PasswordValidator:
    """Advanced password validation and strength checking"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.common_passwords = self._load_common_passwords()
    
    def _load_common_passwords(self) -> set:
        """Load common passwords for blacklist checking"""
        # In production, load from file or external service
        return {
            "password", "123456", "password123", "admin", "letmein",
            "welcome", "monkey", "1234567890", "qwerty", "abc123"
        }
    
    def validate_password(self, password: str, username: str = "") -> Tuple[bool, List[str]]:
        """
        Validate password against security policy
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Length check
        if len(password) < self.config.min_password_length:
            errors.append(f"Password must be at least {self.config.min_password_length} characters long")
        
        # Character requirements
        if self.config.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.config.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.config.require_numbers and not re.search(r'\d', password):
            errors.append("Password must contain at least one number")
        
        if self.config.require_special_chars and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        # Common password check
        if password.lower() in self.common_passwords:
            errors.append("Password is too common")
        
        # Username similarity check
        if username and username.lower() in password.lower():
            errors.append("Password must not contain username")
        
        # Repeated character check
        if len(set(password)) < len(password) * 0.6:
            errors.append("Password has too many repeated characters")
        
        # Sequential character check
        if self._has_sequential_chars(password):
            errors.append("Password must not contain sequential characters")
        
        return len(errors) == 0, errors
    
    def _has_sequential_chars(self, password: str) -> bool:
        """Check for sequential characters like 123 or abc"""
        sequences = ["0123456789", "abcdefghijklmnopqrstuvwxyz", "qwertyuiop"]
        for i in range(len(password) - 2):
            substr = password[i:i+3].lower()
            for seq in sequences:
                if substr in seq or substr[::-1] in seq:
                    return True
        return False
    
    def calculate_strength_score(self, password: str) -> Tuple[int, str]:
        """
        Calculate password strength score (0-100)
        
        Returns:
            Tuple of (score, strength_level)
        """
        score = 0
        
        # Length scoring
        if len(password) >= 12:
            score += 25
        elif len(password) >= 8:
            score += 15
        
        # Character diversity
        if re.search(r'[a-z]', password):
            score += 10
        if re.search(r'[A-Z]', password):
            score += 10
        if re.search(r'\d', password):
            score += 10
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            score += 15
        
        # Unique characters
        unique_ratio = len(set(password)) / len(password)
        score += int(unique_ratio * 20)
        
        # No common patterns
        if not self._has_sequential_chars(password):
            score += 10
        
        if score >= 80:
            strength = "STRONG"
        elif score >= 60:
            strength = "MEDIUM"
        elif score >= 40:
            strength = "WEAK"
        else:
            strength = "VERY_WEAK"
        
        return min(score, 100), strength


class CryptographyService:
    """Advanced cryptography service for data protection"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.pwd_context = CryptContext(
            schemes=["argon2", "bcrypt"],
            deprecated="auto",
            argon2__memory_cost=102400,  # 100MB
            argon2__time_cost=2,
            argon2__parallelism=8,
        )
        self._setup_encryption_keys()
    
    def _setup_encryption_keys(self):
        """Setup encryption keys for field-level encryption"""
        # In production, keys should be stored in secure key management service
        password = self.config.jwt_secret_key.encode()
        
        # Generate or retrieve salt from secure storage
        salt = self._get_or_generate_salt()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.fernet = Fernet(key)
    
    def _get_or_generate_salt(self) -> bytes:
        """Get or generate salt for key derivation"""
        import os
        
        # Try to get salt from environment or Vault
        salt_b64 = os.getenv('ENCRYPTION_SALT')
        if salt_b64:
            try:
                return base64.urlsafe_b64decode(salt_b64)
            except Exception as e:
                logger.warning("Failed to decode salt from environment", error=str(e))
        
        # Generate new salt if not available
        # In production, this should be stored in Vault or secure key management
        salt = secrets.token_bytes(32)
        salt_b64 = base64.urlsafe_b64encode(salt).decode()
        
        logger.warning(
            "Generated new encryption salt. Store this securely: ENCRYPTION_SALT=%s",
            salt_b64
        )
        
        return salt
    
    def hash_password(self, password: str) -> str:
        """Hash password using Argon2"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            return self.pwd_context.verify(password, hashed)
        except Exception as e:
            logger.error("Password verification failed", error=str(e))
            return False
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data for storage"""
        if not self.config.enable_field_level_encryption:
            return data
        
        try:
            encrypted = self.fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error("Data encryption failed", error=str(e))
            return data
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not self.config.enable_field_level_encryption:
            return encrypted_data
        
        try:
            decoded = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            logger.error("Data decryption failed", error=str(e))
            return encrypted_data
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        return secrets.token_urlsafe(length)
    
    def generate_api_key(self) -> str:
        """Generate secure API key"""
        return self.generate_secure_token(self.config.api_key_length)
    
    def create_webhook_signature(self, payload: str, secret: str) -> str:
        """Create HMAC signature for webhook verification"""
        signature = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    def verify_webhook_signature(
        self,
        payload: str,
        signature: str,
        secret: str,
        timestamp: Optional[int] = None
    ) -> bool:
        """Verify webhook signature with timing attack protection"""
        if timestamp and abs(time.time() - timestamp) > self.config.webhook_signature_tolerance_seconds:
            security_logger.warning("Webhook signature timestamp out of tolerance")
            return False
        
        expected_signature = self.create_webhook_signature(payload, secret)
        return hmac.compare_digest(signature, expected_signature)


class JWTService:
    """Secure JWT token management with revocation support"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.crypto_service = CryptographyService(config)
        self.revoked_tokens: Set[str] = set()  # In production, use Redis
        self._redis_client = None
    
    def create_access_token(
        self,
        subject: str,
        additional_claims: Optional[Dict[str, Any]] = None,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token"""
        now = datetime.utcnow()
        expire = now + (expires_delta or timedelta(minutes=self.config.jwt_expiration_minutes))
        
        claims = {
            "sub": subject,
            "iat": now,
            "exp": expire,
            "jti": self.crypto_service.generate_secure_token(16),  # JWT ID for revocation
            "type": "access"
        }
        
        if additional_claims:
            claims.update(additional_claims)
        
        token = jwt.encode(
            claims,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )
        
        security_logger.info("Access token created", user_id=subject)
        return token
    
    def create_refresh_token(self, subject: str) -> str:
        """Create JWT refresh token"""
        now = datetime.utcnow()
        expire = now + timedelta(days=self.config.jwt_refresh_expiration_days)
        
        claims = {
            "sub": subject,
            "iat": now,
            "exp": expire,
            "jti": self.crypto_service.generate_secure_token(16),
            "type": "refresh"
        }
        
        token = jwt.encode(
            claims,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )
        
        security_logger.info("Refresh token created", user_id=subject)
        return token
    
    def verify_token(self, token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token with revocation check"""
        try:
            # Check if token is revoked first (before expensive cryptographic verification)
            if self.is_token_revoked(token):
                security_logger.warning("Token is revoked")
                return None
            
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm]
            )
            
            if payload.get("type") != token_type:
                security_logger.warning("Invalid token type", expected=token_type, actual=payload.get("type"))
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            security_logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            security_logger.warning("Invalid token", error=str(e))
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Create new access token from refresh token"""
        payload = self.verify_token(refresh_token, "refresh")
        if not payload:
            return None
        
        subject = payload.get("sub")
        if not subject:
            return None
        
        return self.create_access_token(subject)
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a JWT token by adding it to blacklist"""
        try:
            # Decode token to get JTI without verification (since it might be expired)
            payload = jwt.decode(token, options={"verify_signature": False})
            jti = payload.get("jti")
            
            if not jti:
                security_logger.warning("Token revocation failed: no JTI found")
                return False
            
            # Add to revocation list
            if self._redis_client:
                # In production, store in Redis with expiration
                expiry_time = payload.get("exp", 0)
                ttl = max(0, expiry_time - int(time.time()))
                if ttl > 0:
                    self._redis_client.setex(f"revoked_token:{jti}", ttl, "1")
            else:
                # Fallback to in-memory storage
                self.revoked_tokens.add(jti)
            
            security_logger.info("Token revoked successfully", jti=jti)
            return True
            
        except Exception as e:
            security_logger.error("Token revocation failed", error=str(e))
            return False
    
    def is_token_revoked(self, token: str) -> bool:
        """Check if token is revoked"""
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            jti = payload.get("jti")
            
            if not jti:
                return False
            
            # Check revocation status
            if self._redis_client:
                return self._redis_client.exists(f"revoked_token:{jti}")
            else:
                return jti in self.revoked_tokens
                
        except Exception:
            return False
    
    def revoke_all_user_tokens(self, user_id: str) -> bool:
        """Revoke all tokens for a specific user"""
        try:
            # In production, this would query all active sessions for the user
            # and revoke them individually
            if self._redis_client:
                pattern = f"user_session:{user_id}:*"
                keys = self._redis_client.keys(pattern)
                if keys:
                    self._redis_client.delete(*keys)
            
            security_logger.info("All user tokens revoked", user_id=user_id)
            return True
            
        except Exception as e:
            security_logger.error("Failed to revoke user tokens", user_id=user_id, error=str(e))
            return False


class RateLimitService:
    """Advanced rate limiting with multiple strategies"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.attempts: Dict[str, List[float]] = {}
        self.locked_keys: Dict[str, float] = {}
    
    def is_rate_limited(self, key: str, max_attempts: Optional[int] = None) -> bool:
        """Check if key is rate limited"""
        current_time = time.time()
        max_attempts = max_attempts or self.config.max_login_attempts
        window = self.config.rate_limit_window_seconds
        
        # Check if locked
        if key in self.locked_keys:
            if current_time < self.locked_keys[key]:
                return True
            else:
                del self.locked_keys[key]
        
        # Clean old attempts
        if key in self.attempts:
            self.attempts[key] = [
                attempt_time for attempt_time in self.attempts[key]
                if current_time - attempt_time < window
            ]
        else:
            self.attempts[key] = []
        
        # Check rate limit
        if len(self.attempts[key]) >= max_attempts:
            # Lock the key
            self.locked_keys[key] = current_time + (self.config.lockout_duration_minutes * 60)
            security_logger.warning("Rate limit exceeded", key=key, attempts=len(self.attempts[key]))
            return True
        
        return False
    
    def record_attempt(self, key: str, success: bool = False):
        """Record an attempt for rate limiting"""
        current_time = time.time()
        
        if key not in self.attempts:
            self.attempts[key] = []
        
        self.attempts[key].append(current_time)
        
        # Clear attempts on successful authentication
        if success and key in self.attempts:
            del self.attempts[key]
            if key in self.locked_keys:
                del self.locked_keys[key]
    
    def get_remaining_attempts(self, key: str, max_attempts: Optional[int] = None) -> int:
        """Get remaining attempts before rate limit"""
        if self.is_rate_limited(key, max_attempts):
            return 0
        
        max_attempts = max_attempts or self.config.max_login_attempts
        current_attempts = len(self.attempts.get(key, []))
        return max(0, max_attempts - current_attempts)


class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.malicious_patterns = [
            r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script tags
            r'javascript:',  # JavaScript protocol
            r'on\w+\s*=',  # Event handlers
            r'<iframe\b[^>]*>',  # Iframe tags
            r'<object\b[^>]*>',  # Object tags
            r'<embed\b[^>]*>',  # Embed tags
            r'eval\s*\(',  # eval() calls
            r'setTimeout\s*\(',  # setTimeout calls
            r'setInterval\s*\(',  # setInterval calls
        ]
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def validate_ip_address(self, ip: str) -> bool:
        """Validate IP address format"""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    def validate_domain(self, domain: str) -> bool:
        """Validate domain name format"""
        pattern = r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$'
        return bool(re.match(pattern, domain))
    
    def sanitize_html(self, html: str) -> str:
        """Remove potentially dangerous HTML content"""
        # Basic HTML sanitization - use bleach library for production
        for pattern in self.malicious_patterns:
            html = re.sub(pattern, '', html, flags=re.IGNORECASE)
        return html
    
    def validate_file_upload(self, filename: str, content: bytes) -> Tuple[bool, List[str]]:
        """Validate file upload"""
        errors = []
        
        # Size check
        if len(content) > self.config.max_file_upload_size_mb * 1024 * 1024:
            errors.append(f"File size exceeds {self.config.max_file_upload_size_mb}MB limit")
        
        # Extension check
        if self.config.allowed_file_extensions:
            extension = Path(filename).suffix.lower()
            if extension not in self.config.allowed_file_extensions:
                errors.append(f"File extension {extension} not allowed")
        
        # Content scanning
        if self.config.enable_content_scanning:
            if self._contains_malicious_content(content):
                errors.append("File contains potentially malicious content")
        
        return len(errors) == 0, errors
    
    def _contains_malicious_content(self, content: bytes) -> bool:
        """Basic malicious content detection"""
        # Convert to string for pattern matching
        try:
            text = content.decode('utf-8', errors='ignore')
        except:
            return False
        
        # Check for malicious patterns
        for pattern in self.malicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False


class SecurityHeadersMiddleware:
    """Security headers middleware for FastAPI"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def get_security_headers(self) -> Dict[str, str]:
        """Get comprehensive security headers for responses"""
        return {
            # Content type protection
            "X-Content-Type-Options": "nosniff",
            
            # Frame protection
            "X-Frame-Options": "DENY",
            
            # XSS protection (legacy, but still useful)
            "X-XSS-Protection": "1; mode=block",
            
            # HTTPS enforcement
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            
            # Content Security Policy (enhanced)
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data: https: blob:; "
                "font-src 'self' https://cdn.jsdelivr.net; "
                "connect-src 'self' wss: ws:; "
                "media-src 'self'; "
                "object-src 'none'; "
                "base-uri 'self'; "
                "form-action 'self'; "
                "frame-ancestors 'none'; "
                "upgrade-insecure-requests;"
            ),
            
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Permissions policy (enhanced)
            "Permissions-Policy": (
                "accelerometer=(), ambient-light-sensor=(), autoplay=(), "
                "battery=(), camera=(), cross-origin-isolated=(), "
                "display-capture=(), document-domain=(), encrypted-media=(), "
                "execution-while-not-rendered=(), execution-while-out-of-viewport=(), "
                "fullscreen=(), geolocation=(), gyroscope=(), "
                "keyboard-map=(), magnetometer=(), microphone=(), "
                "midi=(), navigation-override=(), payment=(), "
                "picture-in-picture=(), publickey-credentials-get=(), "
                "screen-wake-lock=(), sync-xhr=(), usb=(), "
                "web-share=(), xr-spatial-tracking=()"
            ),
            
            # Additional security headers
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Resource-Policy": "same-origin",
            
            # Prevent DNS prefetching
            "X-DNS-Prefetch-Control": "off",
            
            # Prevent MIME type sniffing downloads
            "X-Download-Options": "noopen",
            
            # Prevent Adobe Flash from loading
            "X-Permitted-Cross-Domain-Policies": "none",
            
            # Security headers for API responses
            "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            
            # Clear site data on logout (when appropriate)
            # "Clear-Site-Data": '"cache", "cookies", "storage"',  # Uncomment for logout endpoints
        }


class SecurityService:
    """Main security service orchestrator"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.password_validator = PasswordValidator(config)
        self.crypto_service = CryptographyService(config)
        self.jwt_service = JWTService(config)
        self.rate_limit_service = RateLimitService(config)
        self.input_validator = InputValidator(config)
        self.security_headers = SecurityHeadersMiddleware(config)
        
        security_logger.info("Security service initialized", config_level=config.__dict__)
    
    def validate_and_hash_password(self, password: str, username: str = "") -> Tuple[bool, str, List[str]]:
        """Validate password and return hash if valid"""
        is_valid, errors = self.password_validator.validate_password(password, username)
        
        if is_valid:
            password_hash = self.crypto_service.hash_password(password)
            return True, password_hash, []
        
        return False, "", errors
    
    def authenticate_user(self, username: str, password: str, password_hash: str) -> bool:
        """Authenticate user with rate limiting"""
        # Check rate limiting
        if self.rate_limit_service.is_rate_limited(f"login:{username}"):
            security_logger.warning("Login rate limited", username=username)
            return False
        
        # Verify password
        is_valid = self.crypto_service.verify_password(password, password_hash)
        
        # Record attempt
        self.rate_limit_service.record_attempt(f"login:{username}", is_valid)
        
        if is_valid:
            security_logger.info("User authenticated successfully", username=username)
        else:
            security_logger.warning("Authentication failed", username=username)
        
        return is_valid
    
    def create_user_tokens(self, user_id: str, additional_claims: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Create access and refresh tokens for user"""
        access_token = self.jwt_service.create_access_token(user_id, additional_claims)
        refresh_token = self.jwt_service.create_refresh_token(user_id)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
    
    def verify_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify access token and return payload"""
        return self.jwt_service.verify_token(token, "access")
    
    def refresh_tokens(self, refresh_token: str) -> Optional[str]:
        """Refresh access token"""
        return self.jwt_service.refresh_access_token(refresh_token)


# Global security service instance
_security_service: Optional[SecurityService] = None


def get_security_service() -> Optional[SecurityService]:
    """Get global security service instance"""
    return _security_service


def setup_security(config: SecurityConfig) -> SecurityService:
    """Setup global security service"""
    global _security_service
    _security_service = SecurityService(config)
    return _security_service