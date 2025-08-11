"""
Enhanced JWT Service with RSA Support and Security Best Practices
"""

import secrets
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import base64

import jwt
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend

from .logging import get_logger, security_logger

logger = get_logger(__name__)


@dataclass
class JWTConfig:
    """Enhanced JWT configuration"""
    algorithm: str = "RS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7
    issuer: str = "xorb-platform"
    audience: str = "xorb-api"
    
    # RSA key configuration
    key_size: int = 2048
    private_key_path: Optional[str] = None
    public_key_path: Optional[str] = None
    
    # Security features
    enable_jti: bool = True  # JWT ID for revocation
    enable_nbf: bool = True  # Not before claim
    enable_token_binding: bool = True  # Bind tokens to client
    max_token_age_hours: int = 24  # Maximum token age regardless of refresh


class TokenRevocationStore:
    """In-memory token revocation store (use Redis in production)"""
    
    def __init__(self):
        self.revoked_tokens: set = set()
        self.revocation_expiry: Dict[str, datetime] = {}
    
    async def revoke_token(self, jti: str, expires_at: datetime):
        """Revoke a token by JTI"""
        self.revoked_tokens.add(jti)
        self.revocation_expiry[jti] = expires_at
        
        # Cleanup expired revocations
        await self._cleanup_expired()
    
    async def is_revoked(self, jti: str) -> bool:
        """Check if token is revoked"""
        await self._cleanup_expired()
        return jti in self.revoked_tokens
    
    async def _cleanup_expired(self):
        """Remove expired revocations"""
        now = datetime.utcnow()
        expired_jtis = [
            jti for jti, expires_at in self.revocation_expiry.items()
            if now > expires_at
        ]
        
        for jti in expired_jtis:
            self.revoked_tokens.discard(jti)
            del self.revocation_expiry[jti]


class RSAKeyManager:
    """RSA key management for JWT signing"""
    
    def __init__(self, config: JWTConfig):
        self.config = config
        self.private_key = None
        self.public_key = None
        self.key_id = None
        
    def generate_key_pair(self) -> Tuple[bytes, bytes]:
        """Generate new RSA key pair"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.config.key_size,
            backend=default_backend()
        )
        
        public_key = private_key.public_key()
        
        # Serialize private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Serialize public key
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    async def load_or_generate_keys(self):
        """Load existing keys or generate new ones"""
        try:
            if (self.config.private_key_path and 
                self.config.public_key_path and
                Path(self.config.private_key_path).exists() and
                Path(self.config.public_key_path).exists()):
                
                # Load existing keys
                await self._load_keys()
                logger.info("Loaded existing RSA keys")
            else:
                # Generate new keys
                await self._generate_and_save_keys()
                logger.info("Generated new RSA keys")
                
        except Exception as e:
            logger.error("Failed to load/generate RSA keys", error=str(e))
            raise
    
    async def _load_keys(self):
        """Load keys from files"""
        # Load private key
        with open(self.config.private_key_path, 'rb') as f:
            private_pem = f.read()
            self.private_key = serialization.load_pem_private_key(
                private_pem, password=None, backend=default_backend()
            )
        
        # Load public key
        with open(self.config.public_key_path, 'rb') as f:
            public_pem = f.read()
            self.public_key = serialization.load_pem_public_key(
                public_pem, backend=default_backend()
            )
        
        # Generate key ID
        self.key_id = self._generate_key_id(public_pem)
    
    async def _generate_and_save_keys(self):
        """Generate and save new keys"""
        private_pem, public_pem = self.generate_key_pair()
        
        # Save keys if paths provided
        if self.config.private_key_path:
            Path(self.config.private_key_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.private_key_path, 'wb') as f:
                f.write(private_pem)
        
        if self.config.public_key_path:
            Path(self.config.public_key_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.public_key_path, 'wb') as f:
                f.write(public_pem)
        
        # Load into memory
        self.private_key = serialization.load_pem_private_key(
            private_pem, password=None, backend=default_backend()
        )
        self.public_key = serialization.load_pem_public_key(
            public_pem, backend=default_backend()
        )
        
        self.key_id = self._generate_key_id(public_pem)
    
    def _generate_key_id(self, public_pem: bytes) -> str:
        """Generate key ID from public key"""
        import hashlib
        return hashlib.sha256(public_pem).hexdigest()[:8]
    
    def get_private_key(self):
        """Get private key for signing"""
        return self.private_key
    
    def get_public_key(self):
        """Get public key for verification"""
        return self.public_key
    
    def get_jwks(self) -> Dict[str, Any]:
        """Get JSON Web Key Set for public key distribution"""
        if not self.public_key:
            return {"keys": []}
        
        # Get public key numbers
        public_numbers = self.public_key.public_numbers()
        
        # Convert to base64url
        def int_to_base64url(num):
            byte_length = (num.bit_length() + 7) // 8
            return base64.urlsafe_b64encode(
                num.to_bytes(byte_length, 'big')
            ).decode('ascii').rstrip('=')
        
        jwk = {
            "kty": "RSA",
            "use": "sig",
            "alg": self.config.algorithm,
            "kid": self.key_id,
            "n": int_to_base64url(public_numbers.n),
            "e": int_to_base64url(public_numbers.e)
        }
        
        return {"keys": [jwk]}


class EnhancedJWTService:
    """Enhanced JWT service with RSA support and security features"""
    
    def __init__(self, config: JWTConfig):
        self.config = config
        self.key_manager = RSAKeyManager(config)
        self.revocation_store = TokenRevocationStore()
        self.token_stats = {
            "tokens_issued": 0,
            "tokens_verified": 0,
            "tokens_revoked": 0,
            "verification_failures": 0
        }
    
    async def initialize(self):
        """Initialize JWT service"""
        await self.key_manager.load_or_generate_keys()
        logger.info("Enhanced JWT service initialized")
    
    async def create_access_token(
        self,
        subject: str,
        additional_claims: Optional[Dict[str, Any]] = None,
        client_fingerprint: Optional[str] = None
    ) -> str:
        """Create access token with enhanced security"""
        
        now = datetime.utcnow()
        expires_at = now + timedelta(minutes=self.config.access_token_expire_minutes)
        
        # Standard claims
        claims = {
            "sub": subject,
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
            "token_type": "access"
        }
        
        # Security claims
        if self.config.enable_jti:
            claims["jti"] = secrets.token_urlsafe(16)
        
        if self.config.enable_nbf:
            claims["nbf"] = int(now.timestamp())
        
        # Token binding
        if self.config.enable_token_binding and client_fingerprint:
            claims["cnf"] = {"jkt": client_fingerprint}
        
        # Additional claims
        if additional_claims:
            # Validate additional claims
            reserved_claims = {"sub", "iss", "aud", "iat", "exp", "nbf", "jti", "token_type", "cnf"}
            for claim in additional_claims:
                if claim in reserved_claims:
                    logger.warning("Skipping reserved claim", claim=claim)
                    continue
                claims[claim] = additional_claims[claim]
        
        # Sign token
        token = jwt.encode(
            claims,
            self.key_manager.get_private_key(),
            algorithm=self.config.algorithm,
            headers={"kid": self.key_manager.key_id}
        )
        
        self.token_stats["tokens_issued"] += 1
        
        security_logger.info("Access token created", 
                           subject=subject, 
                           expires_at=expires_at.isoformat(),
                           jti=claims.get("jti"))
        
        return token
    
    async def create_refresh_token(
        self,
        subject: str,
        client_fingerprint: Optional[str] = None
    ) -> str:
        """Create refresh token"""
        
        now = datetime.utcnow()
        expires_at = now + timedelta(days=self.config.refresh_token_expire_days)
        
        claims = {
            "sub": subject,
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "iat": int(now.timestamp()),
            "exp": int(expires_at.timestamp()),
            "token_type": "refresh"
        }
        
        if self.config.enable_jti:
            claims["jti"] = secrets.token_urlsafe(16)
        
        if self.config.enable_nbf:
            claims["nbf"] = int(now.timestamp())
        
        if self.config.enable_token_binding and client_fingerprint:
            claims["cnf"] = {"jkt": client_fingerprint}
        
        token = jwt.encode(
            claims,
            self.key_manager.get_private_key(),
            algorithm=self.config.algorithm,
            headers={"kid": self.key_manager.key_id}
        )
        
        security_logger.info("Refresh token created", 
                           subject=subject, 
                           expires_at=expires_at.isoformat(),
                           jti=claims.get("jti"))
        
        return token
    
    async def verify_token(
        self,
        token: str,
        expected_type: str = "access",
        client_fingerprint: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Verify token with enhanced security checks"""
        
        try:
            # Decode token with verification
            payload = jwt.decode(
                token,
                self.key_manager.get_public_key(),
                algorithms=[self.config.algorithm],
                issuer=self.config.issuer,
                audience=self.config.audience,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_nbf": True,
                    "verify_iss": True,
                    "verify_aud": True
                }
            )
            
            # Verify token type
            if payload.get("token_type") != expected_type:
                security_logger.warning("Invalid token type", 
                                      expected=expected_type, 
                                      actual=payload.get("token_type"))
                return None
            
            # Check revocation
            jti = payload.get("jti")
            if jti and await self.revocation_store.is_revoked(jti):
                security_logger.warning("Revoked token used", jti=jti)
                return None
            
            # Verify token binding
            if self.config.enable_token_binding and client_fingerprint:
                token_fingerprint = payload.get("cnf", {}).get("jkt")
                if token_fingerprint and token_fingerprint != client_fingerprint:
                    security_logger.warning("Token binding verification failed")
                    return None
            
            # Check maximum token age
            iat = payload.get("iat", 0)
            max_age_seconds = self.config.max_token_age_hours * 3600
            if time.time() - iat > max_age_seconds:
                security_logger.warning("Token exceeds maximum age")
                return None
            
            self.token_stats["tokens_verified"] += 1
            return payload
            
        except jwt.ExpiredSignatureError:
            security_logger.warning("Token expired")
            self.token_stats["verification_failures"] += 1
            return None
        except jwt.InvalidTokenError as e:
            security_logger.warning("Invalid token", error=str(e))
            self.token_stats["verification_failures"] += 1
            return None
        except Exception as e:
            security_logger.error("Token verification error", error=str(e))
            self.token_stats["verification_failures"] += 1
            return None
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke a token"""
        try:
            # Decode without verification to get JTI and expiration
            payload = jwt.decode(token, options={"verify_signature": False})
            jti = payload.get("jti")
            exp = payload.get("exp")
            
            if jti and exp:
                expires_at = datetime.fromtimestamp(exp)
                await self.revocation_store.revoke_token(jti, expires_at)
                
                self.token_stats["tokens_revoked"] += 1
                security_logger.info("Token revoked", jti=jti)
                return True
                
        except Exception as e:
            logger.error("Failed to revoke token", error=str(e))
        
        return False
    
    async def refresh_access_token(
        self,
        refresh_token: str,
        client_fingerprint: Optional[str] = None
    ) -> Optional[str]:
        """Create new access token from refresh token"""
        
        payload = await self.verify_token(refresh_token, "refresh", client_fingerprint)
        if not payload:
            return None
        
        subject = payload.get("sub")
        if not subject:
            return None
        
        # Create new access token
        return await self.create_access_token(
            subject=subject,
            client_fingerprint=client_fingerprint
        )
    
    def get_jwks(self) -> Dict[str, Any]:
        """Get JSON Web Key Set for public key distribution"""
        return self.key_manager.get_jwks()
    
    async def rotate_keys(self):
        """Rotate RSA keys"""
        # Generate new key pair
        private_pem, public_pem = self.key_manager.generate_key_pair()
        
        # TODO: Implement key rotation with grace period
        # This would involve:
        # 1. Generate new keys
        # 2. Keep old keys for verification during grace period
        # 3. Switch to new keys for signing
        # 4. Remove old keys after grace period
        
        security_logger.info("Key rotation initiated")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get JWT service statistics"""
        return {
            "tokens_issued": self.token_stats["tokens_issued"],
            "tokens_verified": self.token_stats["tokens_verified"],
            "tokens_revoked": self.token_stats["tokens_revoked"],
            "verification_failures": self.token_stats["verification_failures"],
            "revoked_tokens_count": len(self.revocation_store.revoked_tokens),
            "key_id": self.key_manager.key_id
        }


# Global instance
_jwt_service: Optional[EnhancedJWTService] = None


async def get_jwt_service() -> EnhancedJWTService:
    """Get global JWT service instance"""
    global _jwt_service
    
    if _jwt_service is None:
        config = JWTConfig(
            algorithm=os.getenv("JWT_ALGORITHM", "RS256"),
            access_token_expire_minutes=int(os.getenv("JWT_EXPIRATION_MINUTES", "15")),
            refresh_token_expire_days=int(os.getenv("JWT_REFRESH_EXPIRATION_DAYS", "7")),
            private_key_path=os.getenv("JWT_PRIVATE_KEY_PATH", "/app/secrets/jwt-private.pem"),
            public_key_path=os.getenv("JWT_PUBLIC_KEY_PATH", "/app/secrets/jwt-public.pem")
        )
        
        _jwt_service = EnhancedJWTService(config)
        await _jwt_service.initialize()
    
    return _jwt_service


def generate_client_fingerprint(request_info: Dict[str, Any]) -> str:
    """Generate client fingerprint for token binding"""
    import hashlib
    
    # Combine client information
    fingerprint_data = f"{request_info.get('user_agent', '')}" \
                      f"{request_info.get('client_ip', '')}" \
                      f"{request_info.get('accept_language', '')}"
    
    return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]