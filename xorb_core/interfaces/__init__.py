"""
Interface Layer - REST/gRPC APIs and External Interfaces

This layer contains:
- REST API Controllers: FastAPI endpoint handlers
- gRPC Services: Protocol buffer service implementations  
- DTO Schemas: Request/response data models
- Dependency Injection: Wire up infrastructure to application layer

Dependencies: Application layer only
May NOT depend on: Domain or Infrastructure layers directly
"""

from __future__ import annotations

__all__ = [
    # REST DTOs
    "CreateCampaignRequest",
    "StartCampaignRequest",
    "CampaignResponse",
    "FindingResponse",
    "TriageFindingRequest",
    "EmbeddingRequest",
    "KnowledgeAtomResponse",

    # REST Controllers
    "CampaignController",
    "FindingController",
    "KnowledgeController",
    "HealthController",

    # gRPC Services
    "EmbeddingGrpcService",
    "CampaignGrpcService",

    # Dependency Injection
    "Dependencies",
    "get_campaign_service",
    "get_finding_service",
    "get_knowledge_service"
]

# Import REST DTOs
# Import Dependency Injection
from .dependencies import (
    Dependencies,
    get_campaign_service,
    get_finding_service,
    get_knowledge_service,
)

# Import gRPC Services
from .grpc.services import CampaignGrpcService, EmbeddingGrpcService

# Import REST Controllers
from .rest.controllers import (
    CampaignController,
    FindingController,
    HealthController,
    KnowledgeController,
)
from .rest.schemas import (
    CampaignResponse,
    CreateCampaignRequest,
    EmbeddingRequest,
    FindingResponse,
    KnowledgeAtomResponse,
    StartCampaignRequest,
    TriageFindingRequest,
)
