"""
Secure JWT Secret Management

This module provides secure JWT secret management with:
- Cryptographically secure secret generation
- Entropy validation for secret strength
- Automatic secret rotation
- HashiCorp Vault integration
- Fallback to environment variables for development

Security Features:
- Minimum 64-character secrets with high entropy
- 24-hour automatic rotation in production
- Vault-based secret storage and retrieval
- Audit logging for all secret operations
"""

import os
import time
import secrets
import hashlib
import math
from typing import Optional, Dict, Any
from collections import Counter
from dataclasses import dataclass

try:
    from .vault_client import VaultClient
except ImportError:
    VaultClient = None

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class JWTSecretMetadata:
    """Metadata for JWT secret tracking"""
    created_at: float
    entropy: float
    length: int
    rotation_count: int
    source: str  # 'vault', 'env', 'generated'


class SecurityError(Exception):
    """Security-related error"""
    pass


class SecureJWTManager:
    """Secure JWT secret management with Vault integration"""
    
    MIN_SECRET_LENGTH = 64
    MIN_ENTROPY = 5.0
    ROTATION_INTERVAL = 24 * 60 * 60  # 24 hours in seconds
    
    def __init__(self, environment: str = "development"):
        """Initialize secure JWT manager"""
        self.environment = environment
        self.vault_client = None
        self._current_secret = None
        self._secret_metadata = None
        self._last_rotation_check = 0
        
        # Initialize Vault client for production
        if environment == "production" and VaultClient:
            try:
                self.vault_client = VaultClient()
                logger.info("Vault client initialized for JWT secret management")
            except Exception as e:
                logger.error(f"Failed to initialize Vault client: {e}")
                if environment == "production":
                    raise SecurityError("Vault initialization failed in production")
        
        # Load initial secret
        self._load_secret()
    
    def get_signing_key(self) -> str:
        """Get current JWT signing key with automatic rotation check"""
        current_time = time.time()
        
        # Check if rotation is needed (throttle checks to every 5 minutes)
        if current_time - self._last_rotation_check > 300:  # 5 minutes
            if self._needs_rotation():
                self._rotate_secret()
            self._last_rotation_check = current_time
        
        if not self._current_secret:
            raise SecurityError("No JWT secret available")
        
        return self._current_secret
    
    def _load_secret(self):
        """Load JWT secret from Vault or environment"""
        try:
            if self.vault_client and self.vault_client.is_authenticated():
                self._load_from_vault()
            else:
                self._load_from_environment()
        except Exception as e:
            logger.error(f"Failed to load JWT secret: {e}")
            if self.environment == "production":
                raise SecurityError("Failed to load JWT secret in production")
            else:
                logger.warning("Generating temporary secret for development")
                self._generate_temporary_secret()
    
    def _load_from_vault(self):
        """Load JWT secret from HashiCorp Vault"""
        try:
            secret_data = self.vault_client.get_secret("jwt-signing")
            if secret_data:
                secret = secret_data.get("key")
                created_at = secret_data.get("created_at", time.time())
                rotation_count = secret_data.get("rotation_count", 0)
                
                if secret and self._validate_secret(secret):
                    self._current_secret = secret
                    self._secret_metadata = JWTSecretMetadata(
                        created_at=created_at,
                        entropy=self._calculate_entropy(secret),
                        length=len(secret),
                        rotation_count=rotation_count,
                        source="vault"
                    )
                    logger.info("JWT secret loaded from Vault")
                    return
            
            # Generate new secret if none exists or validation fails
            self._generate_and_store_secret()
            
        except Exception as e:
            logger.error(f"Failed to load secret from Vault: {e}")
            raise
    
    def _load_from_environment(self):
        """Load JWT secret from environment variable"""
        secret = os.getenv("JWT_SECRET")
        
        if not secret:
            if self.environment == "production":
                raise SecurityError("JWT_SECRET environment variable required in production")
            else:
                logger.warning("No JWT_SECRET found, generating temporary secret")
                self._generate_temporary_secret()
                return
        
        if not self._validate_secret(secret):
            raise SecurityError("JWT_SECRET does not meet security requirements")
        
        self._current_secret = secret
        self._secret_metadata = JWTSecretMetadata(
            created_at=time.time(),
            entropy=self._calculate_entropy(secret),
            length=len(secret),
            rotation_count=0,
            source="env"
        )
        logger.info("JWT secret loaded from environment")
    
    def _validate_secret(self, secret: str) -> bool:
        """Validate JWT secret meets security requirements"""
        if len(secret) < self.MIN_SECRET_LENGTH:
            logger.error(f"JWT secret too short: {len(secret)} < {self.MIN_SECRET_LENGTH}")
            return False
        
        entropy = self._calculate_entropy(secret)
        if entropy < self.MIN_ENTROPY:
            logger.error(f"JWT secret entropy too low: {entropy} < {self.MIN_ENTROPY}")
            return False
        
        # Check for common weak patterns
        if self._has_weak_patterns(secret):
            logger.error("JWT secret contains weak patterns")
            return False
        
        return True
    
    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of string"""
        if not data:
            return 0.0
        
        counts = Counter(data)
        length = len(data)
        
        # Calculate Shannon entropy
        entropy = -sum((count/length) * math.log2(count/length) 
                      for count in counts.values())
        
        return entropy
    
    def _has_weak_patterns(self, secret: str) -> bool:
        """Check for weak patterns in secret"""
        # Check for repeated characters (more lenient for URL-safe base64)
        if len(set(secret)) < len(secret) * 0.4:  # Less than 40% unique chars
            return True
        
        # Check for sequential patterns (more lenient)
        sequential_count = 0
        for i in range(len(secret) - 4):  # Look for longer sequences
            consecutive = 0
            for j in range(1, 5):  # Check up to 5 consecutive chars
                if i + j < len(secret) and ord(secret[i+j]) == ord(secret[i+j-1]) + 1:
                    consecutive += 1
                else:
                    break
            if consecutive >= 4:  # 5 or more consecutive chars
                sequential_count += 1
        
        if sequential_count > 2:  # More than 2 long sequences
            return True
        
        # Check for repeated substrings (more than 20% repetition)
        for length in [3, 4, 5]:
            substrings = {}
            for i in range(len(secret) - length + 1):
                substring = secret[i:i+length]
                substrings[substring] = substrings.get(substring, 0) + 1
            
            total_repeated = sum(count for count in substrings.values() if count > 1)
            if total_repeated > len(secret) * 0.2:  # More than 20% repetition
                return True
        
        return False
    
    def _generate_secret(self) -> str:
        """Generate cryptographically secure JWT secret"""
        # Generate 512-bit secret (64 bytes)
        secret = secrets.token_urlsafe(64)
        
        # Validate generated secret
        max_attempts = 10
        attempts = 0
        
        while not self._validate_secret(secret) and attempts < max_attempts:
            secret = secrets.token_urlsafe(64)
            attempts += 1
        
        if attempts >= max_attempts:
            raise SecurityError("Failed to generate valid JWT secret after multiple attempts")
        
        return secret
    
    def _generate_and_store_secret(self):
        """Generate new secret and store in Vault"""
        secret = self._generate_secret()
        current_time = time.time()
        
        secret_data = {
            "key": secret,
            "created_at": current_time,
            "rotation_count": (self._secret_metadata.rotation_count + 1 
                             if self._secret_metadata else 1),
            "entropy": self._calculate_entropy(secret)
        }
        
        if self.vault_client:
            self.vault_client.store_secret("jwt-signing", secret_data)
            logger.info("New JWT secret generated and stored in Vault")
        
        self._current_secret = secret
        self._secret_metadata = JWTSecretMetadata(
            created_at=current_time,
            entropy=secret_data["entropy"],
            length=len(secret),
            rotation_count=secret_data["rotation_count"],
            source="vault" if self.vault_client else "generated"
        )
    
    def _generate_temporary_secret(self):
        """Generate temporary secret for development"""
        secret = self._generate_secret()
        self._current_secret = secret
        self._secret_metadata = JWTSecretMetadata(
            created_at=time.time(),
            entropy=self._calculate_entropy(secret),
            length=len(secret),
            rotation_count=0,
            source="generated"
        )
        logger.warning("Using temporary JWT secret - not suitable for production")
    
    def _needs_rotation(self) -> bool:
        """Check if secret needs rotation"""
        if not self._secret_metadata:
            return True
        
        # Don't rotate environment-based secrets
        if self._secret_metadata.source == "env":
            return False
        
        # Rotate if older than rotation interval
        age = time.time() - self._secret_metadata.created_at
        return age > self.ROTATION_INTERVAL
    
    def _rotate_secret(self):
        """Rotate JWT secret"""
        try:
            old_metadata = self._secret_metadata
            self._generate_and_store_secret()
            
            logger.info(
                "JWT secret rotated",
                old_rotation_count=old_metadata.rotation_count if old_metadata else 0,
                new_rotation_count=self._secret_metadata.rotation_count,
                age_hours=(time.time() - old_metadata.created_at) / 3600 if old_metadata else 0
            )
            
        except Exception as e:
            logger.error(f"JWT secret rotation failed: {e}")
            # Don't raise in production to avoid service disruption
            if self.environment != "production":
                raise SecurityError(f"JWT secret rotation failed: {e}")
    
    def force_rotation(self):
        """Force immediate secret rotation"""
        logger.warning("Forcing JWT secret rotation")
        self._rotate_secret()
    
    def get_secret_info(self) -> dict:
        """Get non-sensitive information about current secret"""
        if not self._secret_metadata:
            return {"status": "no_secret"}
        
        return {
            "length": self._secret_metadata.length,
            "entropy": round(self._secret_metadata.entropy, 2),
            "age_hours": round((time.time() - self._secret_metadata.created_at) / 3600, 2),
            "rotation_count": self._secret_metadata.rotation_count,
            "source": self._secret_metadata.source,
            "next_rotation_hours": round(
                (self._secret_metadata.created_at + self.ROTATION_INTERVAL - time.time()) / 3600, 2
            ) if self._secret_metadata.source != "env" else None
        }