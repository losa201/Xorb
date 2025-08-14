"""
XORB Platform Bus PubSub Module - Phase G2 Tenant Isolation

This module provides production-ready NATS JetStream integration with:
- Subject schema validation (v1 immutable)
- Per-tenant isolation and quotas
- Tuned consumers with flow control
- Exactly-once semantics
- Comprehensive error handling

Key Components:
- NATSClient: Main client with tenant isolation
- SubjectBuilder: Subject construction and validation
- ConsumerSettings: Tuned flow control configuration
- Domain/Event enums: Schema validation
"""

from .nats_client import (
    NATSClient,
    SubjectBuilder,
    SubjectComponents,
    ConsumerSettings,
    Domain,
    Event,
    StreamClass,
    create_nats_client
)

__all__ = [
    "NATSClient",
    "SubjectBuilder",
    "SubjectComponents",
    "ConsumerSettings",
    "Domain",
    "Event",
    "StreamClass",
    "create_nats_client"
]
