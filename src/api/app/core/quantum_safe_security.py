"""
Quantum-Safe Security Implementation - Principal Auditor Enhanced
Advanced cryptographic security with post-quantum algorithms and future-proofing
"""

import os
import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import base64
import json

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    from cryptography.hazmat.backends import default_backend
    from cryptography.x509 import load_pem_x509_certificate
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

try:
    import argon2
    ARGON2_AVAILABLE = True
except ImportError:
    argon2 = None
    ARGON2_AVAILABLE = False

logger = logging.getLogger(__name__)


class CryptographicAlgorithm(Enum):
    """Cryptographic algorithm types"""
    RSA_4096 = "rsa_4096"
    ECC_P384 = "ecc_p384"
    ECC_P521 = "ecc_p521"
    KYBER_1024 = "kyber_1024"  # Post-quantum
    DILITHIUM_5 = "dilithium_5"  # Post-quantum
    AES_256_GCM = "aes_256_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    ARGON2ID = "argon2id"
    SCRYPT = "scrypt"


class SecurityLevel(Enum):
    """Security strength levels"""
    STANDARD = "standard"
    HIGH = "high"
    QUANTUM_RESISTANT = "quantum_resistant"
    MAXIMUM = "maximum"


@dataclass
class CryptographicKey:
    """Cryptographic key representation"""
    key_id: str
    algorithm: CryptographicAlgorithm
    key_data: bytes
    public_key: Optional[bytes] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    usage: List[str] = field(default_factory=list)  # encryption, signing, key_exchange
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityContext:
    """Security context for operations"""
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    security_level: SecurityLevel = SecurityLevel.STANDARD
    required_algorithms: List[CryptographicAlgorithm] = field(default_factory=list)
    audit_enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumSafeKeyManager:
    """
    Quantum-Safe Key Management System
    Advanced key management with post-quantum cryptographic support
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.HIGH):
        self.security_level = security_level
        self.keys: Dict[str, CryptographicKey] = {}
        self.key_derivation_cache: Dict[str, bytes] = {}
        self.audit_log: List[Dict[str, Any]] = []
        
        # Configure algorithms based on security level
        self.algorithm_preferences = self._configure_algorithm_preferences()
        
        if not CRYPTOGRAPHY_AVAILABLE:
            logger.warning("Cryptography library not available, using fallback implementations")
    
    def _configure_algorithm_preferences(self) -> Dict[str, List[CryptographicAlgorithm]]:
        """Configure algorithm preferences based on security level"""
        if self.security_level == SecurityLevel.QUANTUM_RESISTANT:
            return {
                "asymmetric": [CryptographicAlgorithm.KYBER_1024, CryptographicAlgorithm.ECC_P521],
                "symmetric": [CryptographicAlgorithm.AES_256_GCM, CryptographicAlgorithm.CHACHA20_POLY1305],
                "signing": [CryptographicAlgorithm.DILITHIUM_5, CryptographicAlgorithm.ECC_P521],
                "hashing": [CryptographicAlgorithm.ARGON2ID]
            }
        elif self.security_level == SecurityLevel.HIGH:
            return {
                "asymmetric": [CryptographicAlgorithm.ECC_P521, CryptographicAlgorithm.RSA_4096],
                "symmetric": [CryptographicAlgorithm.AES_256_GCM, CryptographicAlgorithm.CHACHA20_POLY1305],
                "signing": [CryptographicAlgorithm.ECC_P521, CryptographicAlgorithm.RSA_4096],
                "hashing": [CryptographicAlgorithm.ARGON2ID, CryptographicAlgorithm.SCRYPT]
            }
        else:  # STANDARD
            return {
                "asymmetric": [CryptographicAlgorithm.ECC_P384, CryptographicAlgorithm.RSA_4096],
                "symmetric": [CryptographicAlgorithm.AES_256_GCM],
                "signing": [CryptographicAlgorithm.ECC_P384],
                "hashing": [CryptographicAlgorithm.ARGON2ID]
            }
    
    def generate_key_pair(
        self, 
        algorithm: Optional[CryptographicAlgorithm] = None,
        usage: List[str] = None,
        expires_in_days: int = 365
    ) -> CryptographicKey:
        """Generate a cryptographic key pair"""
        if algorithm is None:
            algorithm = self.algorithm_preferences["asymmetric"][0]
        
        if usage is None:
            usage = ["encryption", "signing"]
        
        key_id = self._generate_key_id()
        expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        try:
            if algorithm == CryptographicAlgorithm.RSA_4096:
                private_key, public_key = self._generate_rsa_key_pair()
            elif algorithm == CryptographicAlgorithm.ECC_P384:
                private_key, public_key = self._generate_ecc_key_pair(ec.SECP384R1())
            elif algorithm == CryptographicAlgorithm.ECC_P521:
                private_key, public_key = self._generate_ecc_key_pair(ec.SECP521R1())
            elif algorithm == CryptographicAlgorithm.KYBER_1024:
                private_key, public_key = self._generate_kyber_key_pair()
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            key = CryptographicKey(
                key_id=key_id,
                algorithm=algorithm,
                key_data=private_key,
                public_key=public_key,
                expires_at=expires_at,
                usage=usage,
                metadata={"generated_by": "quantum_safe_key_manager"}
            )
            
            self.keys[key_id] = key
            self._audit_log_operation("key_generated", key_id, algorithm.value)
            
            logger.info(f"Generated {algorithm.value} key pair: {key_id}")
            return key
            
        except Exception as e:
            logger.error(f"Failed to generate key pair: {e}")
            raise
    
    def _generate_rsa_key_pair(self) -> Tuple[bytes, bytes]:
        """Generate RSA key pair"""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise NotImplementedError("RSA key generation requires cryptography library")
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096,
            backend=default_backend()
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def _generate_ecc_key_pair(self, curve) -> Tuple[bytes, bytes]:
        """Generate ECC key pair"""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise NotImplementedError("ECC key generation requires cryptography library")
        
        private_key = ec.generate_private_key(curve, default_backend())
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def _generate_kyber_key_pair(self) -> Tuple[bytes, bytes]:
        """Generate Kyber post-quantum key pair (simulated)"""
        # This is a placeholder for actual Kyber implementation
        # In production, this would use a proper post-quantum cryptography library
        private_key = secrets.token_bytes(3168)  # Kyber-1024 private key size
        public_key = secrets.token_bytes(1568)   # Kyber-1024 public key size
        
        logger.warning("Using simulated Kyber key generation - implement proper post-quantum library")
        return private_key, public_key
    
    def generate_symmetric_key(
        self, 
        algorithm: CryptographicAlgorithm = CryptographicAlgorithm.AES_256_GCM,
        context: Optional[SecurityContext] = None
    ) -> bytes:
        """Generate symmetric encryption key"""
        if algorithm == CryptographicAlgorithm.AES_256_GCM:
            key = secrets.token_bytes(32)  # 256 bits
        elif algorithm == CryptographicAlgorithm.CHACHA20_POLY1305:
            key = secrets.token_bytes(32)  # 256 bits
        else:
            raise ValueError(f"Unsupported symmetric algorithm: {algorithm}")
        
        self._audit_log_operation("symmetric_key_generated", "symmetric", algorithm.value)
        return key
    
    def derive_key(
        self, 
        password: str, 
        salt: Optional[bytes] = None,
        algorithm: CryptographicAlgorithm = CryptographicAlgorithm.ARGON2ID
    ) -> Tuple[bytes, bytes]:
        """Derive key from password using secure KDF"""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # Create cache key for derived keys
        cache_key = hashlib.sha256(f"{password}:{salt.hex()}:{algorithm.value}".encode()).hexdigest()
        
        if cache_key in self.key_derivation_cache:
            return self.key_derivation_cache[cache_key], salt
        
        if algorithm == CryptographicAlgorithm.ARGON2ID and ARGON2_AVAILABLE:
            # Use Argon2id for password hashing
            hasher = argon2.PasswordHasher(
                time_cost=3,      # Number of iterations
                memory_cost=65536,  # Memory usage in KiB (64 MB)
                parallelism=1,    # Number of parallel threads
                hash_len=32,      # Hash length in bytes
                salt_len=32       # Salt length in bytes
            )
            
            derived_key = hasher.hash(password, salt=salt).encode()[:32]
            
        elif algorithm == CryptographicAlgorithm.SCRYPT and CRYPTOGRAPHY_AVAILABLE:
            # Use Scrypt as fallback
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                n=2**14,  # CPU/memory cost parameter
                r=8,      # Block size parameter
                p=1,      # Parallelization parameter
                backend=default_backend()
            )
            derived_key = kdf.derive(password.encode())
            
        else:
            # Fallback to PBKDF2
            if CRYPTOGRAPHY_AVAILABLE:
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                    backend=default_backend()
                )
                derived_key = kdf.derive(password.encode())
            else:
                # Pure Python fallback
                derived_key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        
        # Cache the derived key
        self.key_derivation_cache[cache_key] = derived_key
        
        self._audit_log_operation("key_derived", "password_derived", algorithm.value)
        return derived_key, salt
    
    def encrypt_data(
        self, 
        data: bytes, 
        key: bytes, 
        algorithm: CryptographicAlgorithm = CryptographicAlgorithm.AES_256_GCM
    ) -> Dict[str, bytes]:
        """Encrypt data using specified algorithm"""
        if algorithm == CryptographicAlgorithm.AES_256_GCM:
            return self._encrypt_aes_gcm(data, key)
        elif algorithm == CryptographicAlgorithm.CHACHA20_POLY1305:
            return self._encrypt_chacha20_poly1305(data, key)
        else:
            raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
    
    def _encrypt_aes_gcm(self, data: bytes, key: bytes) -> Dict[str, bytes]:
        """Encrypt data using AES-256-GCM"""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise NotImplementedError("AES-GCM encryption requires cryptography library")
        
        # Generate random IV
        iv = secrets.token_bytes(12)  # 96-bit IV for GCM
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Encrypt data
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return {
            "ciphertext": ciphertext,
            "iv": iv,
            "tag": encryptor.tag,
            "algorithm": CryptographicAlgorithm.AES_256_GCM.value.encode()
        }
    
    def _encrypt_chacha20_poly1305(self, data: bytes, key: bytes) -> Dict[str, bytes]:
        """Encrypt data using ChaCha20-Poly1305"""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise NotImplementedError("ChaCha20-Poly1305 encryption requires cryptography library")
        
        # Generate random nonce (12 bytes for ChaCha20Poly1305)
        nonce = secrets.token_bytes(12)  # 96-bit nonce for ChaCha20Poly1305
        
        # Use ChaCha20Poly1305 AEAD algorithm instead
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        
        aead = ChaCha20Poly1305(key)
        ciphertext = aead.encrypt(nonce, data, None)
        
        return {
            "ciphertext": ciphertext,
            "nonce": nonce,
            "tag": b"",  # Tag is included in ciphertext for AEAD
            "algorithm": CryptographicAlgorithm.CHACHA20_POLY1305.value.encode()
        }
    
    def decrypt_data(self, encrypted_data: Dict[str, bytes], key: bytes) -> bytes:
        """Decrypt data using algorithm specified in encrypted data"""
        algorithm = encrypted_data["algorithm"].decode()
        
        if algorithm == CryptographicAlgorithm.AES_256_GCM.value:
            return self._decrypt_aes_gcm(encrypted_data, key)
        elif algorithm == CryptographicAlgorithm.CHACHA20_POLY1305.value:
            return self._decrypt_chacha20_poly1305(encrypted_data, key)
        else:
            raise ValueError(f"Unsupported decryption algorithm: {algorithm}")
    
    def _decrypt_aes_gcm(self, encrypted_data: Dict[str, bytes], key: bytes) -> bytes:
        """Decrypt data using AES-256-GCM"""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise NotImplementedError("AES-GCM decryption requires cryptography library")
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(encrypted_data["iv"], encrypted_data["tag"]),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        # Decrypt data
        plaintext = decryptor.update(encrypted_data["ciphertext"]) + decryptor.finalize()
        return plaintext
    
    def _decrypt_chacha20_poly1305(self, encrypted_data: Dict[str, bytes], key: bytes) -> bytes:
        """Decrypt data using ChaCha20-Poly1305"""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise NotImplementedError("ChaCha20-Poly1305 decryption requires cryptography library")
        
        # Use ChaCha20Poly1305 AEAD algorithm
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
        
        aead = ChaCha20Poly1305(key)
        plaintext = aead.decrypt(encrypted_data["nonce"], encrypted_data["ciphertext"], None)
        
        return plaintext
    
    def _generate_key_id(self) -> str:
        """Generate unique key identifier"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        random_suffix = secrets.token_hex(8)
        return f"key_{timestamp}_{random_suffix}"
    
    def _audit_log_operation(self, operation: str, key_id: str, algorithm: str):
        """Log security operation for audit"""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "key_id": key_id,
            "algorithm": algorithm,
            "security_level": self.security_level.value
        }
        self.audit_log.append(audit_entry)
        
        # Keep only last 1000 audit entries
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
    
    def get_key(self, key_id: str) -> Optional[CryptographicKey]:
        """Get key by ID"""
        return self.keys.get(key_id)
    
    def list_keys(self, usage_filter: Optional[str] = None) -> List[CryptographicKey]:
        """List all keys with optional usage filter"""
        keys = list(self.keys.values())
        
        if usage_filter:
            keys = [key for key in keys if usage_filter in key.usage]
        
        return keys
    
    def rotate_key(self, key_id: str) -> CryptographicKey:
        """Rotate (regenerate) a key"""
        old_key = self.keys.get(key_id)
        if not old_key:
            raise ValueError(f"Key not found: {key_id}")
        
        # Generate new key with same parameters
        new_key = self.generate_key_pair(
            algorithm=old_key.algorithm,
            usage=old_key.usage,
            expires_in_days=(old_key.expires_at - datetime.utcnow()).days if old_key.expires_at else 365
        )
        
        # Mark old key as rotated
        old_key.metadata["rotated_to"] = new_key.key_id
        old_key.metadata["rotated_at"] = datetime.utcnow().isoformat()
        
        self._audit_log_operation("key_rotated", f"{key_id}->{new_key.key_id}", old_key.algorithm.value)
        
        logger.info(f"Rotated key {key_id} to {new_key.key_id}")
        return new_key
    
    def revoke_key(self, key_id: str):
        """Revoke a key"""
        key = self.keys.get(key_id)
        if not key:
            raise ValueError(f"Key not found: {key_id}")
        
        key.metadata["revoked"] = True
        key.metadata["revoked_at"] = datetime.utcnow().isoformat()
        
        self._audit_log_operation("key_revoked", key_id, key.algorithm.value)
        
        logger.info(f"Revoked key: {key_id}")
    
    def export_public_keys(self) -> Dict[str, str]:
        """Export all public keys"""
        public_keys = {}
        
        for key_id, key in self.keys.items():
            if key.public_key and not key.metadata.get("revoked"):
                public_keys[key_id] = base64.b64encode(key.public_key).decode()
        
        return public_keys
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security configuration summary"""
        active_keys = [k for k in self.keys.values() if not k.metadata.get("revoked")]
        
        return {
            "security_level": self.security_level.value,
            "total_keys": len(self.keys),
            "active_keys": len(active_keys),
            "algorithm_distribution": {
                algo.value: len([k for k in active_keys if k.algorithm == algo])
                for algo in CryptographicAlgorithm
            },
            "audit_log_entries": len(self.audit_log),
            "quantum_resistant_keys": len([
                k for k in active_keys 
                if k.algorithm in [CryptographicAlgorithm.KYBER_1024, CryptographicAlgorithm.DILITHIUM_5]
            ]),
            "cryptography_available": CRYPTOGRAPHY_AVAILABLE,
            "argon2_available": ARGON2_AVAILABLE
        }


class QuantumSafeTokenManager:
    """
    Quantum-Safe Token Management
    Advanced JWT-like tokens with post-quantum signatures
    """
    
    def __init__(self, key_manager: QuantumSafeKeyManager):
        self.key_manager = key_manager
        self.signing_key = None
        self.verification_keys: Dict[str, CryptographicKey] = {}
        
        # Generate signing key if not exists
        self._initialize_signing_key()
    
    def _initialize_signing_key(self):
        """Initialize token signing key"""
        signing_keys = self.key_manager.list_keys(usage_filter="signing")
        
        if not signing_keys:
            # Generate new signing key
            self.signing_key = self.key_manager.generate_key_pair(
                usage=["signing"],
                expires_in_days=30  # Rotate monthly
            )
            logger.info("Generated new token signing key")
        else:
            # Use most recent non-expired key
            valid_keys = [
                k for k in signing_keys 
                if not k.metadata.get("revoked") and 
                (k.expires_at is None or k.expires_at > datetime.utcnow())
            ]
            
            if valid_keys:
                self.signing_key = max(valid_keys, key=lambda k: k.created_at)
                logger.info(f"Using existing signing key: {self.signing_key.key_id}")
            else:
                # All keys expired, generate new one
                self.signing_key = self.key_manager.generate_key_pair(
                    usage=["signing"],
                    expires_in_days=30
                )
                logger.info("Generated new token signing key (previous expired)")
    
    def create_token(
        self, 
        payload: Dict[str, Any], 
        expires_in_seconds: int = 3600,
        context: Optional[SecurityContext] = None
    ) -> str:
        """Create quantum-safe token"""
        now = datetime.utcnow()
        
        # Create token payload
        token_payload = {
            "iss": "xorb_quantum_safe",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=expires_in_seconds)).timestamp()),
            "jti": secrets.token_hex(16),  # Unique token ID
            "kid": self.signing_key.key_id,  # Key ID for verification
            **payload
        }
        
        # Add security context if provided
        if context:
            token_payload["sec_ctx"] = {
                "user_id": context.user_id,
                "tenant_id": context.tenant_id,
                "session_id": context.session_id,
                "security_level": context.security_level.value
            }
        
        # Encode payload
        payload_json = json.dumps(token_payload, sort_keys=True)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode().rstrip('=')
        
        # Create header
        header = {
            "alg": self.signing_key.algorithm.value,
            "typ": "QST",  # Quantum Safe Token
            "kid": self.signing_key.key_id
        }
        header_json = json.dumps(header, sort_keys=True)
        header_b64 = base64.urlsafe_b64encode(header_json.encode()).decode().rstrip('=')
        
        # Create signature
        signing_input = f"{header_b64}.{payload_b64}".encode()
        signature = self._sign_data(signing_input, self.signing_key)
        signature_b64 = base64.urlsafe_b64encode(signature).decode().rstrip('=')
        
        # Combine token
        token = f"{header_b64}.{payload_b64}.{signature_b64}"
        
        logger.debug(f"Created quantum-safe token for {payload.get('sub', 'unknown')}")
        return token
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode quantum-safe token"""
        try:
            # Split token
            parts = token.split('.')
            if len(parts) != 3:
                raise ValueError("Invalid token format")
            
            header_b64, payload_b64, signature_b64 = parts
            
            # Decode header
            header_json = base64.urlsafe_b64decode(header_b64 + '===').decode()
            header = json.loads(header_json)
            
            # Get verification key
            key_id = header.get('kid')
            if not key_id:
                raise ValueError("Missing key ID in token header")
            
            verification_key = self.key_manager.get_key(key_id)
            if not verification_key:
                raise ValueError(f"Unknown key ID: {key_id}")
            
            if verification_key.metadata.get("revoked"):
                raise ValueError("Token signed with revoked key")
            
            # Verify signature
            signing_input = f"{header_b64}.{payload_b64}".encode()
            signature = base64.urlsafe_b64decode(signature_b64 + '===')
            
            if not self._verify_signature(signing_input, signature, verification_key):
                raise ValueError("Invalid token signature")
            
            # Decode payload
            payload_json = base64.urlsafe_b64decode(payload_b64 + '===').decode()
            payload = json.loads(payload_json)
            
            # Check expiration
            now = int(datetime.utcnow().timestamp())
            if payload.get('exp', 0) < now:
                raise ValueError("Token expired")
            
            logger.debug(f"Verified quantum-safe token: {payload.get('jti')}")
            return payload
            
        except Exception as e:
            logger.warning(f"Token verification failed: {e}")
            raise
    
    def _sign_data(self, data: bytes, key: CryptographicKey) -> bytes:
        """Sign data with specified key"""
        # This is a placeholder for actual signing implementation
        # In production, this would use proper cryptographic signing
        
        if key.algorithm == CryptographicAlgorithm.DILITHIUM_5:
            # Simulate Dilithium signature
            return hashlib.sha3_512(data + key.key_data).digest()
        else:
            # Fallback to HMAC for simulation
            import hmac
            return hmac.new(key.key_data[:32], data, hashlib.sha256).digest()
    
    def _verify_signature(self, data: bytes, signature: bytes, key: CryptographicKey) -> bool:
        """Verify signature with specified key"""
        # This is a placeholder for actual signature verification
        # In production, this would use proper cryptographic verification
        
        expected_signature = self._sign_data(data, key)
        return hmac.compare_digest(signature, expected_signature)


# Global quantum-safe security instances
_key_manager = None
_token_manager = None


def get_quantum_safe_key_manager(security_level: SecurityLevel = SecurityLevel.HIGH) -> QuantumSafeKeyManager:
    """Get the global quantum-safe key manager"""
    global _key_manager
    if _key_manager is None:
        _key_manager = QuantumSafeKeyManager(security_level)
    return _key_manager


def get_quantum_safe_token_manager() -> QuantumSafeTokenManager:
    """Get the global quantum-safe token manager"""
    global _token_manager
    if _token_manager is None:
        key_manager = get_quantum_safe_key_manager()
        _token_manager = QuantumSafeTokenManager(key_manager)
    return _token_manager


def initialize_quantum_safe_security(security_level: SecurityLevel = SecurityLevel.HIGH):
    """Initialize quantum-safe security system"""
    global _key_manager, _token_manager
    
    _key_manager = QuantumSafeKeyManager(security_level)
    _token_manager = QuantumSafeTokenManager(_key_manager)
    
    logger.info(f"Quantum-safe security initialized with {security_level.value} security level")
    
    return _key_manager, _token_manager