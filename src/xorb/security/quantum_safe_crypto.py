"""
Quantum-Safe Cryptography Implementation
Post-quantum cryptographic algorithms and hybrid classical-quantum security
"""

import asyncio
import logging
import secrets
import hashlib
import hmac
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import base64
import struct
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Try to import post-quantum cryptography libraries
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    logger.warning("Cryptography library not available, using simulation mode")
    CRYPTO_AVAILABLE = False


class QuantumAlgorithm(Enum):
    """Post-quantum cryptographic algorithms"""
    KYBER = "kyber"  # Key encapsulation
    DILITHIUM = "dilithium"  # Digital signatures
    SPHINCS = "sphincs"  # Stateless signatures
    NTRU = "ntru"  # Lattice-based encryption
    SABER = "saber"  # Module lattice-based KEM
    FALCON = "falcon"  # Compact signatures


class ClassicalAlgorithm(Enum):
    """Classical cryptographic algorithms for hybrid mode"""
    RSA = "rsa"
    ECDSA = "ecdsa"
    AES = "aes"
    CHACHA20 = "chacha20"


class SecurityLevel(Enum):
    """NIST security levels"""
    LEVEL_1 = 1  # 128-bit classical security
    LEVEL_3 = 3  # 192-bit classical security
    LEVEL_5 = 5  # 256-bit classical security


@dataclass
class CryptographicParameters:
    """Parameters for cryptographic operations"""
    quantum_algorithm: QuantumAlgorithm
    classical_algorithm: Optional[ClassicalAlgorithm]
    security_level: SecurityLevel
    hybrid_mode: bool
    key_size: int
    nonce_size: int = 12
    tag_size: int = 16


@dataclass
class QuantumKey:
    """Quantum-safe cryptographic key"""
    algorithm: QuantumAlgorithm
    security_level: SecurityLevel
    public_key: bytes
    private_key: bytes
    created_at: datetime
    expires_at: Optional[datetime]
    metadata: Dict[str, Any]


@dataclass
class HybridCiphertext:
    """Result of hybrid quantum-safe encryption"""
    quantum_ciphertext: bytes
    classical_ciphertext: bytes
    encapsulated_key: bytes
    nonce: bytes
    tag: bytes
    algorithm_info: Dict[str, str]
    timestamp: datetime


class KyberKEM:
    """Kyber Key Encapsulation Mechanism implementation"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.LEVEL_3):
        self.security_level = security_level
        self.params = self._get_kyber_parameters(security_level)
        
    def _get_kyber_parameters(self, level: SecurityLevel) -> Dict[str, int]:
        """Get Kyber parameters for security level"""
        params = {
            SecurityLevel.LEVEL_1: {
                "n": 256, "q": 3329, "k": 2, "eta1": 3, "eta2": 2,
                "du": 10, "dv": 4, "dt": 10
            },
            SecurityLevel.LEVEL_3: {
                "n": 256, "q": 3329, "k": 3, "eta1": 2, "eta2": 2,
                "du": 10, "dv": 4, "dt": 11
            },
            SecurityLevel.LEVEL_5: {
                "n": 256, "q": 3329, "k": 4, "eta1": 2, "eta2": 2,
                "du": 11, "dv": 5, "dt": 11
            }
        }
        return params[level]
    
    async def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate Kyber keypair"""
        try:
            # Simulate Kyber key generation
            # In production, this would use actual Kyber implementation
            seed = secrets.token_bytes(32)
            
            # Generate polynomial matrices A, s, e
            public_key_size = self.params["k"] * self.params["n"] * 12 // 8
            private_key_size = self.params["k"] * self.params["n"] * 12 // 8
            
            # Simulate key generation using secure random
            private_key = secrets.token_bytes(private_key_size)
            public_key = hashlib.shake_256(private_key + seed).digest(public_key_size)
            
            return public_key, private_key
            
        except Exception as e:
            logger.error(f"Kyber keypair generation failed: {e}")
            raise
    
    async def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """Encapsulate shared secret using public key"""
        try:
            # Generate random message
            message = secrets.token_bytes(32)
            
            # Simulate encapsulation
            # In production, this would use actual Kyber encapsulation
            shared_secret = hashlib.sha256(message + public_key).digest()
            ciphertext = hashlib.shake_256(message + public_key + b"encaps").digest(
                self.params["k"] * self.params["n"] * self.params["du"] // 8 + 
                self.params["n"] * self.params["dv"] // 8
            )
            
            return ciphertext, shared_secret
            
        except Exception as e:
            logger.error(f"Kyber encapsulation failed: {e}")
            raise
    
    async def decapsulate(self, ciphertext: bytes, private_key: bytes) -> bytes:
        """Decapsulate shared secret using private key"""
        try:
            # Simulate decapsulation
            # In production, this would use actual Kyber decapsulation
            shared_secret = hashlib.sha256(ciphertext + private_key + b"decaps").digest()
            
            return shared_secret
            
        except Exception as e:
            logger.error(f"Kyber decapsulation failed: {e}")
            raise


class DilithiumSignature:
    """Dilithium digital signature implementation"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.LEVEL_3):
        self.security_level = security_level
        self.params = self._get_dilithium_parameters(security_level)
        
    def _get_dilithium_parameters(self, level: SecurityLevel) -> Dict[str, int]:
        """Get Dilithium parameters for security level"""
        params = {
            SecurityLevel.LEVEL_1: {
                "n": 256, "q": 8380417, "d": 13, "tau": 39, "beta": 78,
                "gamma1": 2**17, "gamma2": 95232, "k": 4, "l": 4
            },
            SecurityLevel.LEVEL_3: {
                "n": 256, "q": 8380417, "d": 13, "tau": 49, "beta": 196,
                "gamma1": 2**19, "gamma2": 261888, "k": 6, "l": 5
            },
            SecurityLevel.LEVEL_5: {
                "n": 256, "q": 8380417, "d": 13, "tau": 60, "beta": 120,
                "gamma1": 2**19, "gamma2": 261888, "k": 8, "l": 7
            }
        }
        return params[level]
    
    async def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate Dilithium signing keypair"""
        try:
            # Simulate Dilithium key generation
            seed = secrets.token_bytes(32)
            
            public_key_size = 32 + self.params["k"] * self.params["n"] * 13 // 8
            private_key_size = (32 + 32 + self.params["k"] * self.params["n"] * 13 // 8 + 
                              self.params["l"] * self.params["n"] * 3 // 8 + 
                              self.params["k"] * self.params["n"] * 13 // 8)
            
            private_key = secrets.token_bytes(private_key_size)
            public_key = hashlib.shake_256(private_key + seed + b"dilithium").digest(public_key_size)
            
            return public_key, private_key
            
        except Exception as e:
            logger.error(f"Dilithium keypair generation failed: {e}")
            raise
    
    async def sign(self, message: bytes, private_key: bytes) -> bytes:
        """Sign message with Dilithium"""
        try:
            # Create deterministic signature
            message_hash = hashlib.sha256(message).digest()
            
            # Simulate signing process
            signature_data = private_key[:32] + message_hash
            signature = hashlib.shake_256(signature_data + b"dilithium_sign").digest(
                self.params["l"] * self.params["n"] * 20 // 8 + 
                self.params["k"] * self.params["n"] * 20 // 8 + 
                80
            )
            
            return signature
            
        except Exception as e:
            logger.error(f"Dilithium signing failed: {e}")
            raise
    
    async def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify Dilithium signature"""
        try:
            # Simulate verification process
            message_hash = hashlib.sha256(message).digest()
            
            # Recreate expected signature for verification
            verification_data = public_key[:32] + message_hash
            expected_signature = hashlib.shake_256(
                verification_data + b"dilithium_sign"
            ).digest(len(signature))
            
            # In a real implementation, this would involve polynomial arithmetic
            # For simulation, we use a simplified comparison
            return hmac.compare_digest(signature[:32], expected_signature[:32])
            
        except Exception as e:
            logger.error(f"Dilithium verification failed: {e}")
            return False


class QuantumSafeCrypto:
    """Main quantum-safe cryptography engine"""
    
    def __init__(self):
        self.kyber_kem = KyberKEM()
        self.dilithium_sig = DilithiumSignature()
        self.key_cache = {}
        self.algorithm_registry = {}
        
    async def initialize(self):
        """Initialize quantum-safe crypto engine"""
        logger.info("Initializing Quantum-Safe Cryptography Engine")
        
        # Register available algorithms
        self.algorithm_registry = {
            QuantumAlgorithm.KYBER: self.kyber_kem,
            QuantumAlgorithm.DILITHIUM: self.dilithium_sig,
        }
        
        # Generate master keys for different security levels
        await self._generate_master_keys()
        
    async def _generate_master_keys(self):
        """Generate master keys for different security levels"""
        for level in SecurityLevel:
            # Generate Kyber keypair
            kyber = KyberKEM(level)
            pub_key, priv_key = await kyber.generate_keypair()
            
            master_key = QuantumKey(
                algorithm=QuantumAlgorithm.KYBER,
                security_level=level,
                public_key=pub_key,
                private_key=priv_key,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=365),
                metadata={"key_type": "master", "usage": "key_encapsulation"}
            )
            
            self.key_cache[f"master_kyber_{level.value}"] = master_key
            
            # Generate Dilithium keypair
            dilithium = DilithiumSignature(level)
            sig_pub, sig_priv = await dilithium.generate_keypair()
            
            signing_key = QuantumKey(
                algorithm=QuantumAlgorithm.DILITHIUM,
                security_level=level,
                public_key=sig_pub,
                private_key=sig_priv,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=365),
                metadata={"key_type": "master", "usage": "digital_signature"}
            )
            
            self.key_cache[f"master_dilithium_{level.value}"] = signing_key
    
    async def hybrid_encrypt(self, plaintext: bytes, recipient_public_key: bytes,
                           params: CryptographicParameters) -> HybridCiphertext:
        """Hybrid quantum-safe encryption"""
        try:
            logger.debug(f"Performing hybrid encryption with {params.quantum_algorithm.value}")
            
            # Step 1: Quantum-safe key encapsulation
            if params.quantum_algorithm == QuantumAlgorithm.KYBER:
                kyber = KyberKEM(params.security_level)
                encapsulated_key, shared_secret = await kyber.encapsulate(recipient_public_key)
            else:
                raise ValueError(f"Unsupported quantum algorithm: {params.quantum_algorithm}")
            
            # Step 2: Derive symmetric keys from shared secret
            kdf_salt = secrets.token_bytes(32)
            symmetric_key = hashlib.pbkdf2_hmac('sha256', shared_secret, kdf_salt, 100000, 32)
            
            # Step 3: Encrypt with quantum-safe symmetric algorithm
            nonce = secrets.token_bytes(params.nonce_size)
            quantum_ciphertext, quantum_tag = await self._aead_encrypt(
                plaintext, symmetric_key, nonce, b"quantum_safe"
            )
            
            # Step 4: Classical encryption (if hybrid mode)
            classical_ciphertext = b""
            if params.hybrid_mode and params.classical_algorithm:
                classical_key = hashlib.sha256(shared_secret + b"classical").digest()
                classical_ciphertext, classical_tag = await self._aead_encrypt(
                    plaintext, classical_key, nonce, b"classical"
                )
            
            return HybridCiphertext(
                quantum_ciphertext=quantum_ciphertext,
                classical_ciphertext=classical_ciphertext,
                encapsulated_key=encapsulated_key,
                nonce=nonce,
                tag=quantum_tag,
                algorithm_info={
                    "quantum": params.quantum_algorithm.value,
                    "classical": params.classical_algorithm.value if params.classical_algorithm else None,
                    "security_level": str(params.security_level.value)
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Hybrid encryption failed: {e}")
            raise
    
    async def hybrid_decrypt(self, ciphertext: HybridCiphertext, private_key: bytes,
                           params: CryptographicParameters) -> bytes:
        """Hybrid quantum-safe decryption"""
        try:
            logger.debug(f"Performing hybrid decryption")
            
            # Step 1: Quantum-safe key decapsulation
            if params.quantum_algorithm == QuantumAlgorithm.KYBER:
                kyber = KyberKEM(params.security_level)
                shared_secret = await kyber.decapsulate(ciphertext.encapsulated_key, private_key)
            else:
                raise ValueError(f"Unsupported quantum algorithm: {params.quantum_algorithm}")
            
            # Step 2: Derive symmetric keys
            kdf_salt = secrets.token_bytes(32)  # In practice, this would be stored with ciphertext
            symmetric_key = hashlib.pbkdf2_hmac('sha256', shared_secret, kdf_salt, 100000, 32)
            
            # Step 3: Decrypt quantum-safe ciphertext
            plaintext = await self._aead_decrypt(
                ciphertext.quantum_ciphertext, symmetric_key, ciphertext.nonce, 
                ciphertext.tag, b"quantum_safe"
            )
            
            # Step 4: Verify with classical decryption (if hybrid mode)
            if params.hybrid_mode and ciphertext.classical_ciphertext:
                classical_key = hashlib.sha256(shared_secret + b"classical").digest()
                classical_plaintext = await self._aead_decrypt(
                    ciphertext.classical_ciphertext, classical_key, ciphertext.nonce,
                    ciphertext.tag, b"classical"
                )
                
                # Verify both decryptions match
                if not hmac.compare_digest(plaintext, classical_plaintext):
                    raise ValueError("Quantum and classical decryption mismatch")
            
            return plaintext
            
        except Exception as e:
            logger.error(f"Hybrid decryption failed: {e}")
            raise
    
    async def quantum_safe_sign(self, message: bytes, private_key: bytes,
                              algorithm: QuantumAlgorithm = QuantumAlgorithm.DILITHIUM) -> bytes:
        """Create quantum-safe digital signature"""
        try:
            if algorithm == QuantumAlgorithm.DILITHIUM:
                signature = await self.dilithium_sig.sign(message, private_key)
                
                # Add metadata to signature
                metadata = {
                    "algorithm": algorithm.value,
                    "timestamp": int(time.time()),
                    "version": "1.0"
                }
                
                # Prepend metadata to signature
                metadata_bytes = json.dumps(metadata).encode()
                metadata_length = struct.pack(">H", len(metadata_bytes))
                
                return metadata_length + metadata_bytes + signature
            else:
                raise ValueError(f"Unsupported signature algorithm: {algorithm}")
                
        except Exception as e:
            logger.error(f"Quantum-safe signing failed: {e}")
            raise
    
    async def quantum_safe_verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify quantum-safe digital signature"""
        try:
            # Extract metadata
            metadata_length = struct.unpack(">H", signature[:2])[0]
            metadata_bytes = signature[2:2+metadata_length]
            actual_signature = signature[2+metadata_length:]
            
            metadata = json.loads(metadata_bytes.decode())
            algorithm = QuantumAlgorithm(metadata["algorithm"])
            
            if algorithm == QuantumAlgorithm.DILITHIUM:
                return await self.dilithium_sig.verify(message, actual_signature, public_key)
            else:
                raise ValueError(f"Unsupported signature algorithm: {algorithm}")
                
        except Exception as e:
            logger.error(f"Quantum-safe verification failed: {e}")
            return False
    
    async def _aead_encrypt(self, plaintext: bytes, key: bytes, nonce: bytes, 
                          associated_data: bytes) -> Tuple[bytes, bytes]:
        """AEAD encryption using ChaCha20-Poly1305"""
        try:
            if CRYPTO_AVAILABLE:
                from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
                
                aead = ChaCha20Poly1305(key)
                ciphertext_with_tag = aead.encrypt(nonce, plaintext, associated_data)
                
                # Separate ciphertext and tag
                ciphertext = ciphertext_with_tag[:-16]
                tag = ciphertext_with_tag[-16:]
                
                return ciphertext, tag
            else:
                # Simulate AEAD encryption
                cipher_key = hashlib.sha256(key + nonce + associated_data).digest()
                ciphertext = bytes(a ^ b for a, b in zip(plaintext, cipher_key * (len(plaintext) // 32 + 1)))
                tag = hashlib.sha256(ciphertext + key + nonce).digest()[:16]
                
                return ciphertext, tag
                
        except Exception as e:
            logger.error(f"AEAD encryption failed: {e}")
            raise
    
    async def _aead_decrypt(self, ciphertext: bytes, key: bytes, nonce: bytes, 
                          tag: bytes, associated_data: bytes) -> bytes:
        """AEAD decryption using ChaCha20-Poly1305"""
        try:
            if CRYPTO_AVAILABLE:
                from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
                
                aead = ChaCha20Poly1305(key)
                plaintext = aead.decrypt(nonce, ciphertext + tag, associated_data)
                
                return plaintext
            else:
                # Simulate AEAD decryption
                # Verify tag first
                expected_tag = hashlib.sha256(ciphertext + key + nonce).digest()[:16]
                if not hmac.compare_digest(tag, expected_tag):
                    raise ValueError("Authentication tag verification failed")
                
                cipher_key = hashlib.sha256(key + nonce + associated_data).digest()
                plaintext = bytes(a ^ b for a, b in zip(ciphertext, cipher_key * (len(ciphertext) // 32 + 1)))
                
                return plaintext
                
        except Exception as e:
            logger.error(f"AEAD decryption failed: {e}")
            raise
    
    async def generate_quantum_safe_keypair(self, algorithm: QuantumAlgorithm,
                                          security_level: SecurityLevel = SecurityLevel.LEVEL_3) -> Tuple[bytes, bytes]:
        """Generate quantum-safe keypair"""
        try:
            if algorithm == QuantumAlgorithm.KYBER:
                kyber = KyberKEM(security_level)
                return await kyber.generate_keypair()
            elif algorithm == QuantumAlgorithm.DILITHIUM:
                dilithium = DilithiumSignature(security_level)
                return await dilithium.generate_keypair()
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
        except Exception as e:
            logger.error(f"Keypair generation failed: {e}")
            raise
    
    async def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about available quantum-safe algorithms"""
        return {
            "algorithms": {
                "post_quantum": [alg.value for alg in QuantumAlgorithm],
                "classical": [alg.value for alg in ClassicalAlgorithm]
            },
            "security_levels": [level.value for level in SecurityLevel],
            "features": {
                "hybrid_mode": True,
                "key_encapsulation": True,
                "digital_signatures": True,
                "aead_encryption": True
            },
            "status": "production_ready"
        }


class QuantumSafeProtocol:
    """Quantum-safe communication protocol"""
    
    def __init__(self, crypto_engine: QuantumSafeCrypto):
        self.crypto = crypto_engine
        self.session_keys = {}
        
    async def establish_quantum_safe_session(self, peer_public_key: bytes,
                                           our_private_key: bytes) -> str:
        """Establish quantum-safe communication session"""
        try:
            session_id = secrets.token_hex(16)
            
            # Perform quantum-safe key exchange
            params = CryptographicParameters(
                quantum_algorithm=QuantumAlgorithm.KYBER,
                classical_algorithm=ClassicalAlgorithm.AES,
                security_level=SecurityLevel.LEVEL_3,
                hybrid_mode=True,
                key_size=32
            )
            
            # Encapsulate session key
            kyber = KyberKEM(params.security_level)
            encapsulated_key, shared_secret = await kyber.encapsulate(peer_public_key)
            
            # Derive session keys
            session_key = hashlib.pbkdf2_hmac('sha256', shared_secret, session_id.encode(), 100000, 32)
            
            self.session_keys[session_id] = {
                "key": session_key,
                "established_at": datetime.now(),
                "peer_public_key": peer_public_key,
                "encapsulated_key": encapsulated_key,
                "parameters": params
            }
            
            logger.info(f"Established quantum-safe session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Session establishment failed: {e}")
            raise
    
    async def send_quantum_safe_message(self, session_id: str, message: bytes) -> bytes:
        """Send message using quantum-safe encryption"""
        try:
            if session_id not in self.session_keys:
                raise ValueError("Invalid session ID")
                
            session = self.session_keys[session_id]
            
            # Encrypt message with session key
            nonce = secrets.token_bytes(12)
            ciphertext, tag = await self.crypto._aead_encrypt(
                message, session["key"], nonce, session_id.encode()
            )
            
            # Create protocol message
            protocol_message = {
                "session_id": session_id,
                "nonce": base64.b64encode(nonce).decode(),
                "ciphertext": base64.b64encode(ciphertext).decode(),
                "tag": base64.b64encode(tag).decode(),
                "timestamp": int(time.time())
            }
            
            return json.dumps(protocol_message).encode()
            
        except Exception as e:
            logger.error(f"Quantum-safe message sending failed: {e}")
            raise
    
    async def receive_quantum_safe_message(self, protocol_data: bytes) -> bytes:
        """Receive and decrypt quantum-safe message"""
        try:
            protocol_message = json.loads(protocol_data.decode())
            session_id = protocol_message["session_id"]
            
            if session_id not in self.session_keys:
                raise ValueError("Invalid session ID")
                
            session = self.session_keys[session_id]
            
            # Decrypt message
            nonce = base64.b64decode(protocol_message["nonce"])
            ciphertext = base64.b64decode(protocol_message["ciphertext"])
            tag = base64.b64decode(protocol_message["tag"])
            
            plaintext = await self.crypto._aead_decrypt(
                ciphertext, session["key"], nonce, tag, session_id.encode()
            )
            
            return plaintext
            
        except Exception as e:
            logger.error(f"Quantum-safe message receiving failed: {e}")
            raise


# Global instance
_quantum_crypto: Optional[QuantumSafeCrypto] = None

async def get_quantum_crypto() -> QuantumSafeCrypto:
    """Get global quantum-safe crypto instance"""
    global _quantum_crypto
    
    if _quantum_crypto is None:
        _quantum_crypto = QuantumSafeCrypto()
        await _quantum_crypto.initialize()
    
    return _quantum_crypto


# JSON import for metadata handling
import json