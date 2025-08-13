"""Fixtures for generating mTLS certificates for testing."""

import logging
import os
import tempfile
from typing import Tuple, Generator
import uuid

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec


logger = logging.getLogger(__name__)

# --- Helper to generate a self-signed CA ---
def _generate_test_ca() -> Tuple[bytes, bytes]: # (cert_pem, key_pem)
    """Generates a simple self-signed CA for testing."""
    # Generate CA private key
    ca_private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    ca_public_key = ca_private_key.public_key()

    # Create CA subject
    ca_subject = x509.Name([
        x509.NameAttribute(x509.NameOID.COMMON_NAME, "XORB Test CA"),
        x509.NameAttribute(x509.NameOID.ORGANIZATION_NAME, "XORB Integration Tests"),
    ])

    # Create CA certificate
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_subject)
        .issuer_name(ca_subject) # Self-signed
        .public_key(ca_public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(x509.datetime.datetime.utcnow())
        .not_valid_after(x509.datetime.datetime.utcnow() + x509.datetime.timedelta(days=365))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None), critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                key_cert_sign=True, crl_sign=True, digital_signature=False,
                content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False, encipher_only=False,
                decipher_only=False
            ),
            critical=True
        )
        .sign(private_key=ca_private_key, algorithm=hashes.SHA256())
    )

    ca_cert_pem = ca_cert.public_bytes(serialization.Encoding.PEM)
    ca_key_pem = ca_private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    return ca_cert_pem, ca_key_pem

# --- Helper to generate a service/client certificate ---
def _generate_test_cert(
    ca_cert_pem: bytes, ca_key_pem: bytes, common_name: str, is_client: bool = False
) -> Tuple[bytes, bytes]: # (cert_pem, key_pem)
    """Generates a certificate signed by the test CA."""
    # Load CA
    ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)
    ca_private_key = serialization.load_pem_private_key(ca_key_pem, password=None)

    # Generate private key for the new certificate
    cert_private_key = ec.generate_private_key(ec.SECP256R1())
    cert_public_key = cert_private_key.public_key()

    # Create subject
    cert_subject = x509.Name([
        x509.NameAttribute(x509.NameOID.COMMON_NAME, common_name),
        x509.NameAttribute(x509.NameOID.ORGANIZATION_NAME, "XORB Test Service" if not is_client else "XORB Test Client"),
    ])

    # Build certificate
    builder = (
        x509.CertificateBuilder()
        .subject_name(cert_subject)
        .issuer_name(ca_cert.subject)
        .public_key(cert_public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(x509.datetime.datetime.utcnow())
        .not_valid_after(x509.datetime.datetime.utcnow() + x509.datetime.timedelta(days=30)) # Short TTL for tests
    )

    # Add extensions
    if is_client:
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([x509.ExtendedKeyUsageOID.CLIENT_AUTH]), critical=True
        )
    else: # Server
        builder = builder.add_extension(
            x509.ExtendedKeyUsage([x509.ExtendedKeyUsageOID.SERVER_AUTH]), critical=True
        )
        # Add a SAN for localhost testing
        builder = builder.add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False
        )

    builder = builder.add_extension(
        x509.BasicConstraints(ca=False, path_length=None), critical=True
    )

    cert = builder.sign(private_key=ca_private_key, algorithm=hashes.SHA256())

    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = cert_private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    return cert_pem, key_pem


# --- Session-scoped CA and Certificates ---
@pytest.fixture(scope="session")
def test_ca() -> Generator[Tuple[bytes, bytes], None, None]:
    """Provides a self-signed CA for the test session."""
    logger.info("Generating test CA...")
    ca_cert_pem, ca_key_pem = _generate_test_ca()
    logger.info("Test CA generated.")
    yield ca_cert_pem, ca_key_pem

@pytest.fixture(scope="session")
def cert_and_key_paths(test_ca: Tuple[bytes, bytes]) -> Generator[Tuple[str, str], None, None]:
    """
    Provides file paths to a test service/client certificate and its private key.
    These are signed by the session-scoped test CA.
    Files are cleaned up after the session.
    """
    ca_cert_pem, ca_key_pem = test_ca
    common_name = f"test-client-{uuid.uuid4()}.xorb.test" # Unique CN for each session

    logger.info(f"Generating test certificate for {common_name}...")
    cert_pem, key_pem = _generate_test_cert(ca_cert_pem, ca_key_pem, common_name, is_client=True)
    logger.info("Test certificate generated.")

    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        cert_path = os.path.join(tmpdir, "test_cert.pem")
        key_path = os.path.join(tmpdir, "test_key.pem")
        ca_cert_path = os.path.join(tmpdir, "test_ca.pem") # Also provide CA cert path

        with open(cert_path, "wb") as f:
            f.write(cert_pem)
        with open(key_path, "wb") as f:
            f.write(key_pem)
        with open(ca_cert_path, "wb") as f:
            f.write(ca_cert_pem)

        logger.debug(f"Test cert/key files created: {cert_path}, {key_path}")
        # Yield paths including the CA cert path for trust chain
        yield cert_path, key_path, ca_cert_path
    logger.debug("Test cert/key files cleaned up.")

# --- Trusted CA Path for Clients ---
@pytest.fixture(scope="session")
def trusted_ca_path(cert_and_key_paths: Tuple[str, str, str]) -> str:
    """
    Provides the path to the trusted CA certificate for mTLS verification.
    This is the CA that issued the test certificates.
    """
    _, _, ca_cert_path = cert_and_key_paths
    logger.debug(f"Providing trusted CA path: {ca_cert_path}")
    return ca_cert_path
