"""
Domain Events - Business Events that occur within the domain

Events represent important business occurrences that other parts
of the system may need to react to.
"""

from __future__ import annotations

import uuid
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID

from .models import (
    AgentId,
    AtomId, 
    CampaignId,
    FindingId,
    Severity,
    TargetId
)

__all__ = [
    "DomainEvent",
    "CampaignStarted",
    "CampaignCompleted", 
    "FindingDiscovered",
    "FindingTriaged",
    "AgentExecutionStarted",
    "AgentExecutionCompleted",
    "KnowledgeAtomCreated",
    "EmbeddingGenerated",
    "SimilarityThresholdExceeded",
    "BudgetThresholdExceeded"
]


@dataclass(frozen=True)
class DomainEvent(ABC):
    """Base class for all domain events"""
    
    event_id: UUID = field(default_factory=uuid.uuid4)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = field(default=1)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def event_type(self) -> str:
        """Return the event type name"""
        return self.__class__.__name__


@dataclass(frozen=True)
class CampaignStarted(DomainEvent):
    """Event fired when a campaign begins execution"""
    
    campaign_id: CampaignId = None
    target_id: TargetId = None
    scheduled_agent_count: int = 0
    estimated_duration_hours: int = 0
    budget_limit_usd: float = 0.0


@dataclass(frozen=True)
class CampaignCompleted(DomainEvent):
    """Event fired when a campaign completes"""
    
    campaign_id: CampaignId = None
    target_id: TargetId = None
    findings_count: int = 0
    total_cost_usd: float = 0.0
    duration_hours: float = 0.0
    success: bool = False


@dataclass(frozen=True)
class FindingDiscovered(DomainEvent):
    """Event fired when a new security finding is discovered"""
    
    finding_id: FindingId = None
    campaign_id: CampaignId = None
    agent_id: AgentId = None
    title: str = ""
    severity: Severity = None
    target_info: str = ""


@dataclass(frozen=True)
class FindingTriaged(DomainEvent):
    """Event fired when a finding is triaged"""
    
    finding_id: FindingId = None
    campaign_id: CampaignId = None
    previous_status: str = ""
    new_status: str = ""
    confidence_score: float = 0.0
    reasoning: Optional[str] = None


@dataclass(frozen=True)
class AgentExecutionStarted(DomainEvent):
    """Event fired when an agent begins execution"""
    
    agent_id: AgentId = None
    campaign_id: CampaignId = None
    target_id: TargetId = None
    estimated_duration_minutes: int = 0
    estimated_cost_usd: float = 0.0


@dataclass(frozen=True)
class AgentExecutionCompleted(DomainEvent):
    """Event fired when an agent completes execution"""
    
    agent_id: AgentId = None
    campaign_id: CampaignId = None
    target_id: TargetId = None
    actual_duration_minutes: int = 0
    actual_cost_usd: float = 0.0
    findings_discovered: int = 0
    success: bool = False
    error_message: Optional[str] = None


@dataclass(frozen=True)
class KnowledgeAtomCreated(DomainEvent):
    """Event fired when a new knowledge atom is created"""
    
    atom_id: AtomId = None
    atom_type: str = ""
    content_length: int = 0
    confidence: float = 0.0
    source: Optional[str] = None
    tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EmbeddingGenerated(DomainEvent):
    """Event fired when a vector embedding is generated"""
    
    entity_id: UUID = None
    entity_type: str = ""
    model_name: str = ""
    vector_dimension: int = 0
    generation_time_ms: int = 0
    cached: bool = False


@dataclass(frozen=True)
class SimilarityThresholdExceeded(DomainEvent):
    """Event fired when similarity between entities exceeds threshold"""
    
    source_id: UUID = None
    target_id: UUID = None
    similarity_score: float = 0.0
    threshold: float = 0.0
    metric_type: str = ""
    

@dataclass(frozen=True)
class BudgetThresholdExceeded(DomainEvent):
    """Event fired when campaign approaches budget limits"""
    
    campaign_id: CampaignId = None
    threshold_type: str = ""
    current_value: float = 0.0
    limit_value: float = 0.0
    percentage_used: float = 0.0