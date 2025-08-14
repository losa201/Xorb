"""
XORB Platform Bus Module - Phase G2 Two-Tier Architecture

Implements the Two-Tier Bus architecture with:
- Tier-1: Local ring buffers for same-host communication
- Tier-2: NATS JetStream for cross-node durable pub/sub

This module provides the foundation for tenant-isolated messaging
with strict subject schema validation and performance tuning.
"""

from .pubsub import (
    NATSClient,
    SubjectBuilder,
    Domain,
    Event,
    StreamClass,
    create_nats_client
)

__all__ = [
    "NATSClient",
    "SubjectBuilder",
    "Domain",
    "Event",
    "StreamClass",
    "create_nats_client"
]
