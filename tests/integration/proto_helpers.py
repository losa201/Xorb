"""Helpers for loading and working with Protobuf messages for integration tests."""

import logging
from typing import Any, Dict
import os

# Import generated protobuf classes
# These paths are relative to the test directory structure
# and assume `make proto-gen` or equivalent has been run.
try:
    # Adjust import path based on your project's proto generation output
    # If protos are generated into a `gen/` directory at the project root:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'gen', 'python'))

    from discovery.v1 import discovery_job_pb2
    from evidence.v1 import evidence_pb2
    from google.protobuf import any_pb2, timestamp_pb2
except ImportError as e:
    logging.error(f"Failed to import generated protobufs: {e}")
    raise

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
import time
import uuid

logger = logging.getLogger(__name__)

# --- Helper functions for Discovery Job Protobuf ---

def create_discovery_job_request(
    job_id: str, tenant_id: str, targets: list, profile: str
) -> discovery_job_pb2.DiscoveryJobRequest:
    """Creates a DiscoveryJobRequest protobuf message."""
    request = discovery_job_pb2.DiscoveryJobRequest()
    request.job_id = job_id
    request.tenant_id = tenant_id
    request.targets.extend(targets)
    request.profile = profile
    # Set timestamp
    request.request_time.CopyFrom(timestamp_pb2.Timestamp(seconds=int(time.time())))
    logger.debug(f"Created DiscoveryJobRequest: {request}")
    return request

def serialize_discovery_job_request(request: discovery_job_pb2.DiscoveryJobRequest) -> bytes:
    """Serializes a DiscoveryJobRequest to bytes."""
    return request.SerializeToString()

def load_discovery_job_request_from_bytes(data: bytes) -> discovery_job_pb2.DiscoveryJobRequest:
    """Loads a DiscoveryJobRequest from bytes."""
    request = discovery_job_pb2.DiscoveryJobRequest()
    request.ParseFromString(data)
    return request

# --- Helper functions for Evidence Protobuf ---

def create_signed_evidence(
    evidence_id: str,
    tenant_id: str,
    job_id: str,
    payload_data: Dict[str, Any],
    signing_key_pem: bytes # The private key PEM to sign with
) -> evidence_pb2.Evidence:
    """
    Creates a signed DiscoveryEvidence protobuf message.
    This is a simplified signing process for testing purposes.
    """
    evidence = evidence_pb2.Evidence()
    evidence.evidence_id = evidence_id
    evidence.tenant_id = tenant_id
    evidence.type = evidence_pb2.Evidence.DISCOVERY_EVIDENCE

    # Set collection time
    evidence.collection_time.CopyFrom(timestamp_pb2.Timestamp(seconds=int(time.time())))

    # --- Populate payload (Any) ---
    # For simplicity, we'll pack a simple dict as JSON into a bytes field.
    # A real implementation would use a specific payload protobuf message.
    import json
    payload_any = any_pb2.Any()
    payload_any.type_url = "type.xorb.test/DiscoveryResult"
    payload_any.value = json.dumps(payload_data).encode('utf-8')
    evidence.payload.CopyFrom(payload_any)

    # --- Create Chain of Custody ---
    custody_record = evidence_pb2.CustodyRecord()
    custody_record.handler = "test-discovery-service"
    custody_record.timestamp.CopyFrom(evidence.collection_time)
    custody_record.action = "collected"
    evidence.chain_of_custody.records.append(custody_record)

    # --- Sign the Evidence ---
    # Load the signing key
    signing_key = serialization.load_pem_private_key(signing_key_pem, password=None)

    # Data to sign: For simplicity, we sign the serialized payload.
    # A production system would sign a canonicalized representation of the entire evidence.
    data_to_sign = payload_any.value

    # Sign the data
    signature_bytes = signing_key.sign(data_to_sign, ec.ECDSA(hashes.SHA256()))

    # Attach the signature
    evidence.chain_of_custody.signature.algorithm = "ECDSA-SHA256"
    evidence.chain_of_custody.signature.signature = signature_bytes
    # In a real system, you'd also include the certificate or key ID

    logger.debug(f"Created and signed DiscoveryEvidence: {evidence_id}")
    return evidence

def load_evidence_from_bytes(data: bytes) -> evidence_pb2.Evidence:
    """Loads an Evidence message from bytes."""
    evidence = evidence_pb2.Evidence()
    evidence.ParseFromString(data)
    return evidence

# --- Test Key Loading (for signing in tests) ---
def load_test_signing_key() -> bytes:
    """
    Loads a pre-generated test private key for signing evidence in tests.
    This avoids generating a new key for every test run.
    In a real scenario, keys would come from a secure source like Vault.
    """
    # This is a placeholder. In practice, you would load this from a secure file
    # or generate it once and save it for tests.
    # For this example, we generate it. A better approach is to have a static test key file.
    private_key = ec.generate_private_key(ec.SECP256R1())
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    return pem

def load_test_public_key() -> bytes:
    """Loads the public key corresponding to the test signing key."""
    private_key_pem = load_test_signing_key()
    private_key = serialization.load_pem_private_key(private_key_pem, password=None)
    public_key = private_key.public_key()
    pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return pem
