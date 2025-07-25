"""
Infrastructure Layer - Adapters and External Service Integrations

This layer contains:
- Repository Implementations: Concrete database repositories
- External Service Adapters: Third-party API integrations
- Message/Event Adapters: Message queue and event streaming 
- Cache Adapters: Caching implementations

Dependencies: Domain and Application layers
May NOT be imported by: Domain or Application layers
"""

from __future__ import annotations

__all__ = [
    # Repository implementations
    "PostgreSQLTargetRepository",
    "PostgreSQLAgentRepository",
    "PostgreSQLCampaignRepository", 
    "PostgreSQLFindingRepository",
    "Neo4jKnowledgeAtomRepository",
    
    # External service adapters
    "NvidiaEmbeddingService",
    "RedisCache",
    "NatsEventPublisher",
    "SlackNotificationService",
    
    # Scanner implementations
    "NucleiSecurityScanner",
    "CompositeSecurityScanner"
]

# Import repository implementations
from .repositories import (
    PostgreSQLTargetRepository,
    PostgreSQLAgentRepository,
    PostgreSQLCampaignRepository,
    PostgreSQLFindingRepository,
    Neo4jKnowledgeAtomRepository
)

# Import external service adapters
from .external import (
    NvidiaEmbeddingService,
    RedisCache,
    NatsEventPublisher,
    SlackNotificationService,
    NucleiSecurityScanner,
    CompositeSecurityScanner
)