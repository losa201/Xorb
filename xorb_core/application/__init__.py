"""
Application Layer - Use Cases and Ports

This layer contains:
- Use Cases: Application-specific business rules
- Ports: Abstract interfaces for external dependencies
- Application Services: Orchestrate domain objects and external services
- Command/Query Handlers: Handle requests from the interface layer

Dependencies: Domain layer only
May NOT depend on: Infrastructure or Interface layers
"""

from __future__ import annotations

__all__ = [
    # Ports (Abstract interfaces)
    "Repository",
    "TargetRepository",
    "AgentRepository",
    "CampaignRepository",
    "FindingRepository",
    "KnowledgeAtomRepository",
    "EmbeddingService",
    "EventPublisher",
    "CacheService",
    "NotificationService",
    "SecurityScanner",

    # Commands and Queries
    "CreateCampaignCommand",
    "StartCampaignCommand",
    "ExecuteAgentCommand",
    "TriageFindingCommand",
    "GenerateEmbeddingCommand",
    "CreateKnowledgeAtomCommand",
    "SearchSimilarFindingsQuery",
    "GetCampaignStatusQuery",
    "ListActiveCampaignsQuery",

    # Use Cases
    "CreateCampaignUseCase",
    "StartCampaignUseCase",
    "ExecuteAgentUseCase",
    "TriageFindingUseCase",
    "SearchSimilarFindingsUseCase",
    "GenerateEmbeddingUseCase",
    "CreateKnowledgeAtomUseCase",

    # Application Services
    "CampaignApplicationService",
    "FindingApplicationService",
    "KnowledgeApplicationService"
]

# Import ports
from .ports import (
    AgentRepository,
    CacheService,
    CampaignRepository,
    EmbeddingService,
    EventPublisher,
    FindingRepository,
    KnowledgeAtomRepository,
    NotificationService,
    Repository,
    SecurityScanner,
    TargetRepository,
)

# Import application services
from .services import (
    CampaignApplicationService,
    FindingApplicationService,
    KnowledgeApplicationService,
)

# Import use cases and commands/queries
from .use_cases import (
    CreateCampaignCommand,
    CreateCampaignUseCase,
    CreateKnowledgeAtomCommand,
    CreateKnowledgeAtomUseCase,
    ExecuteAgentCommand,
    ExecuteAgentUseCase,
    GenerateEmbeddingCommand,
    GenerateEmbeddingUseCase,
    GetCampaignStatusQuery,
    ListActiveCampaignsQuery,
    SearchSimilarFindingsQuery,
    SearchSimilarFindingsUseCase,
    StartCampaignCommand,
    StartCampaignUseCase,
    TriageFindingCommand,
    TriageFindingUseCase,
)
