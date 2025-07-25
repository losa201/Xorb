"""
Domain Layer - Pure Business Logic

This layer contains:
- Entities: Objects with identity and lifecycle
- Value Objects: Immutable objects defined by their attributes
- Domain Events: Events that occur within the domain
- Domain Services: Operations that don't naturally fit within an entity

NO EXTERNAL DEPENDENCIES ALLOWED
No I/O, no frameworks, no infrastructure concerns
"""

from __future__ import annotations

__all__ = [
    # Base types
    "Entity",
    "ValueObject", 
    "DomainEvent",
    
    # IDs
    "TargetId",
    "AgentId", 
    "CampaignId",
    "FindingId",
    "AtomId",
    
    # Entities
    "Target",
    "Agent",
    "Campaign", 
    "Finding",
    "KnowledgeAtom",
    
    # Value Objects
    "Embedding",
    "TargetScope",
    "BudgetLimit",
    
    # Enums
    "Severity",
    "AgentCapability",
    "AtomType",
    "CampaignStatus",
    "FindingStatus",
    
    # Events
    "CampaignStarted",
    "CampaignCompleted",
    "FindingDiscovered", 
    "FindingTriaged",
    "AgentExecutionStarted",
    "AgentExecutionCompleted",
    "KnowledgeAtomCreated",
    "EmbeddingGenerated",
    "SimilarityThresholdExceeded",
    "BudgetThresholdExceeded",
    
    # Services
    "AgentSelectionService",
    "BudgetManagementService",
    "SimilarityService", 
    "TriageService",
    "CampaignOrchestratorService"
]

# Core domain imports
from .models import (
    Entity,
    ValueObject,
    TargetId,
    AgentId, 
    CampaignId,
    FindingId,
    AtomId,
    Target,
    Agent,
    Campaign,
    Finding,
    KnowledgeAtom,
    Embedding,
    Severity,
    AgentCapability,
    AtomType,
    CampaignStatus,
    FindingStatus,
    TargetScope,
    BudgetLimit
)

from .events import (
    DomainEvent,
    CampaignStarted,
    CampaignCompleted,
    FindingDiscovered,
    FindingTriaged,
    AgentExecutionStarted,
    AgentExecutionCompleted,
    KnowledgeAtomCreated,
    EmbeddingGenerated,
    SimilarityThresholdExceeded,
    BudgetThresholdExceeded
)

from .services import (
    AgentSelectionService,
    BudgetManagementService,
    SimilarityService,
    TriageService,
    CampaignOrchestratorService
)