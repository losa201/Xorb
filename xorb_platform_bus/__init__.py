"""
XORB Platform Module - Core Infrastructure Components

Provides foundational platform services including:
- Bus: Two-tier messaging architecture (local + NATS)
- Subject schema validation and tenant isolation
- Performance-tuned consumers and producers
"""

from .bus import (
    NATSClient,
    SubjectBuilder,
    Domain,
    Event,
    create_nats_client
)

__all__ = [
    "NATSClient",
    "SubjectBuilder",
    "Domain",
    "Event",
    "create_nats_client"
]
