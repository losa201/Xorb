"""Fixtures for issuing JWTs according to ADR-003."""

import json
import logging
from typing import Dict, Any, Callable
import uuid

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
import time

logger = logging.getLogger(__name__)

# --- Helper to create a test ES256 key pair ---
def _generate_test_es256_keypair() -> Tuple[bytes, bytes]: # (private_key_pem, public_key_pem)
    """Generates a test ES256 key pair."""
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return private_pem, public_pem

# --- Helper to create a JWT header ---
def _create_jwt_header(kid: str = "test-kid") -> Dict[str, str]:
    return {
        "alg": "ES256",
        "typ": "JWT",
        "kid": kid
    }

# --- Helper to encode JWT segments ---
def _base64url_encode(input_bytes: bytes) -> str:
    """Base64URL encode without padding."""
    import base64
    return base64.urlsafe_b64encode(input_bytes).decode('utf-8').rstrip('=')

def _encode_jwt_segment(header: Dict[str, Any], payload: Dict[str, Any]) -> str:
    """Encodes the header and payload segments of a JWT."""
    header_json = json.dumps(header, separators=(',', ':'))
    payload_json = json.dumps(payload, separators=(',', ':'))
    return _base64url_encode(header_json.encode('utf-8')) + '.' + _base64url_encode(payload_json.encode('utf-8'))

# --- Session-scoped test key pair ---
@pytest.fixture(scope="session")
def test_jwt_keypair() -> Tuple[bytes, bytes]:
    """Provides a test ES256 key pair for signing JWTs."""
    logger.info("Generating test ES256 key pair for JWTs...")
    private_pem, public_pem = _generate_test_es256_keypair()
    logger.info("Test ES256 key pair generated.")
    return private_pem, public_pem

# --- JWT Factory Fixture ---
@pytest.fixture(scope="session")
def jwt_token_factory(test_jwt_keypair: Tuple[bytes, bytes]) -> Callable[[Dict[str, Any]], str]:
    """
    Provides a factory function to create JWT tokens signed with the test key.
    The factory takes a dictionary of claims and returns a signed JWT string.
    Default headers (alg=ES256, typ=JWT, kid=test-kid) are used.
    """
    private_key_pem, _ = test_jwt_keypair
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)

    def _factory(claims: Dict[str, Any]) -> str:
        """Create a signed JWT token."""
        # Set default claims if not provided
        default_claims = {
            "iss": "https://auth.xorb.test", # Test issuer
            "exp": int(time.time()) + 3600, # 1 hour expiry
            "iat": int(time.time()),
            "jti": f"test-jti-{uuid.uuid4()}"
        }
        # Merge defaults with provided claims, letting provided claims override
        final_claims = {**default_claims, **claims}

        header = _create_jwt_header()

        # Encode header and payload
        signing_input = _encode_jwt_segment(header, final_claims)

        # Sign the signing input
        signature = private_key.sign(signing_input.encode('utf-8'), ec.ECDSA(hashes.SHA256()))

        # Encode the signature and form the final JWT
        encoded_signature = _base64url_encode(signature)
        jwt_token = f"{signing_input}.{encoded_signature}"

        logger.debug(f"Generated test JWT for sub={final_claims.get('sub', 'N/A')}")
        return jwt_token

    return _factory

# --- Helper function to create a specific JWT (can be used by tests directly if needed) ---
def create_jwt_token(claims: Dict[str, Any], keypair: Tuple[bytes, bytes]) -> str:
    """
    Standalone helper to create a JWT, useful for internal test logic.
    This is a non-fixture version of the factory logic.
    """
    private_key_pem, _ = keypair
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)

    default_claims = {
        "iss": "https://auth.xorb.test",
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
        "jti": f"test-jti-{uuid.uuid4()}"
    }
    final_claims = {**default_claims, **claims}

    header = _create_jwt_header()
    signing_input = _encode_jwt_segment(header, final_claims)
    signature = private_key.sign(signing_input.encode('utf-8'), ec.ECDSA(hashes.SHA256()))
    encoded_signature = _base64url_encode(signature)
    return f"{signing_input}.{encoded_signature}"
