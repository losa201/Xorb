"""
Enhanced authentication security service with Argon2 hashing
Implements secure password handling, account lockout, and security monitoring
"""

import asyncio
import hashlib
import json
import secrets
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from uuid import UUID

from passlib.context import CryptContext
from passlib.hash import argon2
import redis.asyncio as redis

from ..domain.entities import User
from ..domain.exceptions import (
    InvalidCredentials, AccountLocked, SecurityViolation, ValidationError
)
from ..domain.repositories import UserRepository


class AuthSecurityService:
    """Enhanced authentication security service"""
    
    def __init__(
        self,
        user_repository: UserRepository,
        redis_client: redis.Redis,
        max_failed_attempts: int = 5,
        lockout_duration_minutes: int = 30,
        password_min_length: int = 12,
        require_special_chars: bool = True,
        argon2_time_cost: int = 3,
        argon2_memory_cost: int = 65536,  # 64MB
        argon2_parallelism: int = 2
    ):
        self.user_repository = user_repository
        self.redis_client = redis_client
        self.max_failed_attempts = max_failed_attempts
        self.lockout_duration_minutes = lockout_duration_minutes
        self.password_min_length = password_min_length
        self.require_special_chars = require_special_chars
        
        # Configure Argon2 with enhanced security parameters
        self.pwd_context = CryptContext(
            schemes=["argon2"],
            deprecated="auto",
            argon2__time_cost=argon2_time_cost,
            argon2__memory_cost=argon2_memory_cost,
            argon2__parallelism=argon2_parallelism,
            argon2__hash_len=32,
            argon2__salt_len=16
        )
    
    async def hash_password(self, password: str, user_id: Optional[str] = None) -> str:
        """
        Hash password using Argon2id with enhanced security
        """
        # Validate password strength
        self._validate_password_strength(password)
        
        # Add user-specific salt if provided
        if user_id:
            # Create user-specific pepper
            pepper = hashlib.sha256(f"{user_id}:{secrets.token_hex(16)}".encode()).hexdigest()[:16]
            password = f"{password}{pepper}"
        
        # Hash with Argon2id
        hashed = self.pwd_context.hash(password)
        
        # Log security event
        await self._log_security_event("password_hashed", user_id, {
            "timestamp": datetime.utcnow().isoformat(),
            "algorithm": "argon2id"
        })
        
        return hashed
    
    async def verify_password(
        self, 
        password: str, 
        hashed_password: str, 
        user: User,
        client_ip: str
    ) -> bool:
        """
        Verify password with security monitoring and account lockout protection
        """
        user_id = str(user.id)
        
        # Check if account is locked
        if await self._is_account_locked(user_id):
            await self._log_security_event("login_attempt_locked_account", user_id, {
                "client_ip": client_ip,
                "timestamp": datetime.utcnow().isoformat()
            })
            raise AccountLocked("Account is temporarily locked due to too many failed attempts")
        
        # Add user-specific pepper if needed (for backward compatibility)
        original_password = password
        if user_id and len(hashed_password) > 100:  # Peppered password indicator
            pepper = hashlib.sha256(f"{user_id}:{secrets.token_hex(16)}".encode()).hexdigest()[:16]
            password = f"{password}{pepper}"
        
        try:
            # Verify password
            is_valid = self.pwd_context.verify(password, hashed_password)
            
            if is_valid:
                # Reset failed attempts on successful login
                await self._reset_failed_attempts(user_id)
                await self._log_security_event("login_success", user_id, {
                    "client_ip": client_ip,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Check if password needs rehashing (security upgrade)
                if self.pwd_context.needs_update(hashed_password):
                    new_hash = await self.hash_password(original_password, user_id)
                    await self._update_user_password_hash(user, new_hash)
                
                return True
            else:
                # Handle failed attempt
                await self._handle_failed_login_attempt(user_id, client_ip)
                return False
                
        except Exception as e:
            await self._log_security_event("password_verification_error", user_id, {
                "client_ip": client_ip,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            raise SecurityViolation("Password verification failed")
    
    async def check_password_strength(self, password: str) -> Dict[str, bool]:
        """
        Check password strength and return detailed analysis
        """
        checks = {
            "min_length": len(password) >= self.password_min_length,
            "has_uppercase": any(c.isupper() for c in password),
            "has_lowercase": any(c.islower() for c in password),
            "has_digit": any(c.isdigit() for c in password),
            "has_special": any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password),
            "no_common_patterns": not self._has_common_patterns(password),
            "not_breached": await self._check_breach_database(password)
        }
        
        checks["is_strong"] = all(checks.values())
        return checks
    
    async def generate_secure_password(self, length: int = 16) -> str:
        """
        Generate a cryptographically secure password
        """
        if length < self.password_min_length:
            length = self.password_min_length
        
        # Character sets
        lowercase = "abcdefghijklmnopqrstuvwxyz"
        uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        digits = "0123456789"
        special = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        # Ensure at least one character from each set
        password = [
            secrets.choice(lowercase),
            secrets.choice(uppercase),
            secrets.choice(digits),
            secrets.choice(special)
        ]
        
        # Fill remaining length with random characters from all sets
        all_chars = lowercase + uppercase + digits + special
        for _ in range(length - 4):
            password.append(secrets.choice(all_chars))
        
        # Shuffle the password
        secrets.SystemRandom().shuffle(password)
        
        return ''.join(password)
    
    async def _handle_failed_login_attempt(self, user_id: str, client_ip: str):
        """Handle failed login attempt with progressive delays"""
        key = f"failed_attempts:{user_id}"
        
        # Increment failed attempts
        failed_count = await self.redis_client.incr(key)
        await self.redis_client.expire(key, self.lockout_duration_minutes * 60)
        
        # Log the failed attempt
        await self._log_security_event("login_failed", user_id, {
            "client_ip": client_ip,
            "failed_count": failed_count,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Check if account should be locked
        if failed_count >= self.max_failed_attempts:
            await self._lock_account(user_id)
            await self._log_security_event("account_locked", user_id, {
                "client_ip": client_ip,
                "failed_count": failed_count,
                "lockout_duration": self.lockout_duration_minutes,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Progressive delay to slow down brute force attacks
        delay = min(failed_count * 2, 30)  # Max 30 seconds
        await asyncio.sleep(delay)
    
    async def _is_account_locked(self, user_id: str) -> bool:
        """Check if account is currently locked"""
        lock_key = f"account_locked:{user_id}"
        return await self.redis_client.exists(lock_key) > 0
    
    async def _lock_account(self, user_id: str):
        """Lock user account temporarily"""
        lock_key = f"account_locked:{user_id}"
        await self.redis_client.setex(
            lock_key, 
            self.lockout_duration_minutes * 60, 
            datetime.utcnow().isoformat()
        )
    
    async def _reset_failed_attempts(self, user_id: str):
        """Reset failed login attempts counter"""
        failed_key = f"failed_attempts:{user_id}"
        await self.redis_client.delete(failed_key)
    
    def _validate_password_strength(self, password: str):
        """Validate password meets minimum security requirements"""
        if len(password) < self.password_min_length:
            raise ValidationError(f"Password must be at least {self.password_min_length} characters long")
        
        if not any(c.isupper() for c in password):
            raise ValidationError("Password must contain at least one uppercase letter")
        
        if not any(c.islower() for c in password):
            raise ValidationError("Password must contain at least one lowercase letter")
        
        if not any(c.isdigit() for c in password):
            raise ValidationError("Password must contain at least one digit")
        
        if self.require_special_chars and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            raise ValidationError("Password must contain at least one special character")
        
        if self._has_common_patterns(password):
            raise ValidationError("Password contains common patterns and is not secure")
    
    def _has_common_patterns(self, password: str) -> bool:
        """Check for common password patterns"""
        common_patterns = [
            "123456", "password", "admin", "qwerty", "abc123",
            "letmein", "welcome", "monkey", "dragon", "master"
        ]
        
        password_lower = password.lower()
        return any(pattern in password_lower for pattern in common_patterns)
    
    async def _check_breach_database(self, password: str) -> bool:
        """Check if password appears in known breach databases (placeholder)"""
        # In a real implementation, this would check against APIs like HaveIBeenPwned
        # For now, return True (not breached) as placeholder
        return True
    
    async def _update_user_password_hash(self, user: User, new_hash: str):
        """Update user's password hash in database"""
        # Update the user entity
        user.password_hash = new_hash
        user.password_updated_at = datetime.utcnow()
        
        # Save to repository
        await self.user_repository.update(user)
        
        await self._log_security_event("password_rehashed", str(user.id), {
            "timestamp": datetime.utcnow().isoformat(),
            "reason": "security_upgrade"
        })
    
    async def _log_security_event(self, event_type: str, user_id: Optional[str], details: Dict):
        """Log security events for monitoring and audit"""
        event_data = {
            "event_type": event_type,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details
        }
        
        # Store in Redis for real-time monitoring
        event_key = f"security_events:{datetime.utcnow().strftime('%Y-%m-%d')}"
        await self.redis_client.lpush(event_key, str(event_data))
        await self.redis_client.expire(event_key, 86400 * 30)  # Keep for 30 days
        
        # In a production system, you might also:
        # - Send to SIEM system
        # - Store in dedicated audit log database
        # - Trigger alerts for suspicious activities
    
    async def get_security_stats(self, days: int = 7) -> Dict:
        """Get security statistics for monitoring dashboard"""
        stats = {
            "failed_logins": 0,
            "successful_logins": 0,
            "locked_accounts": 0,
            "password_changes": 0,
            "security_events": []
        }
        
        # Aggregate stats from Redis
        for day_offset in range(days):
            date_key = (datetime.utcnow() - timedelta(days=day_offset)).strftime('%Y-%m-%d')
            event_key = f"security_events:{date_key}"
            
            events = await self.redis_client.lrange(event_key, 0, -1)
            for event_str in events:
                try:
                    event_data = json.loads(event_str)
                    event_type = event_data.get("event_type", "")
                    
                    if event_type == "login_failed":
                        stats["failed_logins"] += 1
                    elif event_type == "login_success":
                        stats["successful_logins"] += 1
                    elif event_type == "account_locked":
                        stats["locked_accounts"] += 1
                    elif event_type == "password_hashed":
                        stats["password_changes"] += 1
                        
                    stats["security_events"].append(event_data)
                except:
                    continue
        
        return stats