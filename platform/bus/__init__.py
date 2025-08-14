"""
XORB Platform Bus - NATS JetStream Implementation (ADR-002 Compliant)

This module provides messaging bus capabilities using NATS JetStream only.
Redis pub/sub is forbidden per ADR-002.
"""

from .pubsub.nats_client import NATSClient, StreamConfig

__all__ = ["NATSClient", "StreamConfig"]
