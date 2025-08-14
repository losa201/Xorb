from typing import Protocol, Optional
import hashlib
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization

class SignatureProvider(Protocol):
    """Interface for signature providers"""

    def sign(self, payload: bytes) -> str:
        """Sign a payload and return the signature as hex string"""
        pass

    def verify(self, payload: bytes, signature: str) -> bool:
        """Verify a signature against a payload"""
        pass

class Ed25519Signer(SignatureProvider):
    """Ed25519 implementation of SignatureProvider"""

    def __init__(self, private_key: Ed25519PrivateKey):
        self.private_key = private_key

    def sign(self, payload: bytes) -> str:
        """Sign payload using Ed25519 private key"""
        signature = self.private_key.sign(payload)
        return signature.hex()

    def verify(self, payload: bytes, signature: str) -> bool:
        """Verify signature using Ed25519 public key"""
        try:
            public_key = self.private_key.public_key()
            public_key.verify(bytes.fromhex(signature), payload)
            return True
        except Exception:
            return False

def create_ed25519_signer_from_pem(pem_data: str) -> Ed25519Signer:
    """Create an Ed25519Signer from PEM formatted private key data"""
    private_key = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(pem_data.replace("-----BEGIN PRIVATE KEY-----", "")
                      .replace("-----END PRIVATE KEY-----", "")
                      .strip())
    )
    return Ed25519Signer(private_key)
