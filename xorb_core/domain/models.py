"""
Core Domain Models - Pure Business Logic

Entities and Value Objects that represent the core business concepts
of the Xorb security intelligence platform.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

__all__ = [
    "Entity",
    "ValueObject", 
    "TargetId",
    "AgentId",
    "CampaignId", 
    "FindingId",
    "AtomId",
    "Target",
    "Agent",
    "Campaign",
    "Finding",
    "KnowledgeAtom",
    "Embedding",
    "Severity",
    "AgentCapability",
    "AtomType",
    "CampaignStatus",
    "FindingStatus"
]


# Base Types
class Entity(ABC):
    """Base class for domain entities with identity"""
    
    def __init__(self, id_: Any) -> None:
        self._id = id_
    
    @property
    def id(self) -> Any:
        return self._id
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return self._id == other._id
    
    def __hash__(self) -> int:
        return hash(self._id)


@dataclass(frozen=True)
class ValueObject:
    """Base class for immutable value objects"""
    pass


# Value Object IDs
@dataclass(frozen=True)
class TargetId(ValueObject):
    value: UUID
    
    @classmethod
    def generate(cls) -> TargetId:
        return cls(value=uuid.uuid4())
    
    @classmethod
    def from_string(cls, id_str: str) -> TargetId:
        return cls(value=UUID(id_str))
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class AgentId(ValueObject):
    value: UUID
    
    @classmethod
    def generate(cls) -> AgentId:
        return cls(value=uuid.uuid4())
    
    @classmethod
    def from_string(cls, id_str: str) -> AgentId:
        return cls(value=UUID(id_str))
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class CampaignId(ValueObject):
    value: UUID
    
    @classmethod
    def generate(cls) -> CampaignId:
        return cls(value=uuid.uuid4())
    
    @classmethod
    def from_string(cls, id_str: str) -> CampaignId:
        return cls(value=UUID(id_str))
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class FindingId(ValueObject):
    value: UUID
    
    @classmethod
    def generate(cls) -> FindingId:
        return cls(value=uuid.uuid4())
    
    @classmethod
    def from_string(cls, id_str: str) -> FindingId:
        return cls(value=UUID(id_str))
    
    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class AtomId(ValueObject):
    value: UUID
    
    @classmethod
    def generate(cls) -> AtomId:
        return cls(value=uuid.uuid4())
    
    @classmethod
    def from_string(cls, id_str: str) -> AtomId:
        return cls(value=UUID(id_str))
    
    def __str__(self) -> str:
        return str(self.value)


# Enums
class Severity(Enum):
    """Security finding severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    
    def __lt__(self, other: Severity) -> bool:
        order = {
            Severity.INFO: 0,
            Severity.LOW: 1,
            Severity.MEDIUM: 2,
            Severity.HIGH: 3,
            Severity.CRITICAL: 4
        }
        return order[self] < order[other]


class AgentCapability(Enum):
    """Agent capabilities for matching with targets"""
    WEB_SCANNING = "web_scanning"
    SUBDOMAIN_ENUMERATION = "subdomain_enumeration"
    PORT_SCANNING = "port_scanning"
    VULNERABILITY_SCANNING = "vulnerability_scanning"
    SOCIAL_ENGINEERING = "social_engineering"
    OSINT = "osint"
    API_TESTING = "api_testing"
    MOBILE_TESTING = "mobile_testing"
    NETWORK_ANALYSIS = "network_analysis"
    BINARY_ANALYSIS = "binary_analysis"


class AtomType(Enum):
    """Knowledge atom types"""
    VULNERABILITY = "vulnerability"
    EXPLOIT = "exploit"
    TARGET_INFO = "target_info"
    TOOL_OUTPUT = "tool_output"
    INTELLIGENCE = "intelligence"
    TECHNIQUE = "technique"
    INDICATOR = "indicator"


class CampaignStatus(Enum):
    """Campaign lifecycle status"""
    DRAFT = "draft"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FindingStatus(Enum):
    """Finding triage status"""
    NEW = "new"
    TRIAGING = "triaging"
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    DUPLICATE = "duplicate"
    RESOLVED = "resolved"


# Value Objects
@dataclass(frozen=True)
class Embedding(ValueObject):
    """Vector embedding for semantic search"""
    vector: tuple[float, ...]
    model: str
    dimension: int
    
    def __post_init__(self) -> None:
        if len(self.vector) != self.dimension:
            raise ValueError(f"Vector length {len(self.vector)} != dimension {self.dimension}")
    
    @classmethod
    def from_list(cls, vector: List[float], model: str) -> Embedding:
        return cls(
            vector=tuple(vector),
            model=model,
            dimension=len(vector)
        )
    
    def similarity_cosine(self, other: Embedding) -> float:
        """Compute cosine similarity with another embedding"""
        if self.dimension != other.dimension:
            raise ValueError("Cannot compare embeddings of different dimensions")
        
        # Simple dot product for cosine similarity (assumes normalized vectors)
        return sum(a * b for a, b in zip(self.vector, other.vector))


@dataclass(frozen=True) 
class TargetScope(ValueObject):
    """Defines what is in-scope for testing"""
    domains: Set[str]
    ip_ranges: Set[str]
    excluded_domains: Set[str] = field(default_factory=set)
    excluded_ips: Set[str] = field(default_factory=set)
    
    def is_domain_in_scope(self, domain: str) -> bool:
        """Check if a domain is within scope"""
        if domain in self.excluded_domains:
            return False
        
        # Check if domain matches any allowed domains (including subdomains)
        for allowed_domain in self.domains:
            if domain == allowed_domain or domain.endswith(f".{allowed_domain}"):
                return True
        
        return False


@dataclass(frozen=True)
class BudgetLimit(ValueObject):
    """Campaign budget constraints"""
    max_cost_usd: Decimal
    max_duration_hours: int
    max_api_calls: int
    
    def is_within_budget(
        self, 
        current_cost: Decimal, 
        duration_hours: int,
        api_calls: int
    ) -> bool:
        """Check if current usage is within budget"""
        return (
            current_cost <= self.max_cost_usd and
            duration_hours <= self.max_duration_hours and
            api_calls <= self.max_api_calls
        )


# Entities
class Target(Entity):
    """A target for security testing"""
    
    def __init__(
        self,
        id_: TargetId,
        name: str,
        scope: TargetScope,
        created_at: datetime,
        updated_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(id_)
        self.name = name
        self.scope = scope
        self.created_at = created_at
        self.updated_at = updated_at or created_at
        self.metadata = metadata or {}
    
    def update_scope(self, new_scope: TargetScope) -> None:
        """Update target scope"""
        self.scope = new_scope
        self.updated_at = datetime.now(timezone.utc)
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to target"""
        self.metadata[key] = value
        self.updated_at = datetime.now(timezone.utc)


class Agent(Entity):
    """An agent capable of performing security testing"""
    
    def __init__(
        self,
        id_: AgentId,
        name: str,
        capabilities: Set[AgentCapability],
        cost_per_execution: Decimal,
        average_duration_minutes: int,
        created_at: datetime,
        is_active: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(id_)
        self.name = name
        self.capabilities = capabilities
        self.cost_per_execution = cost_per_execution
        self.average_duration_minutes = average_duration_minutes
        self.created_at = created_at
        self.is_active = is_active
        self.metadata = metadata or {}
    
    def can_handle_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has required capability"""
        return capability in self.capabilities
    
    def deactivate(self) -> None:
        """Deactivate agent"""
        self.is_active = False
    
    def activate(self) -> None:
        """Activate agent"""
        self.is_active = True


class Campaign(Entity):
    """A campaign represents a coordinated security testing effort"""
    
    def __init__(
        self,
        id_: CampaignId,
        name: str,
        target: Target,
        budget: BudgetLimit,
        created_at: datetime,
        status: CampaignStatus = CampaignStatus.DRAFT,
        scheduled_agents: Optional[List[AgentId]] = None,
        findings: Optional[List[FindingId]] = None
    ) -> None:
        super().__init__(id_)
        self.name = name
        self.target = target
        self.budget = budget
        self.created_at = created_at
        self.status = status
        self.scheduled_agents = scheduled_agents or []
        self.findings = findings or []
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
    
    def start_campaign(self) -> None:
        """Start the campaign"""
        if self.status != CampaignStatus.QUEUED:
            raise ValueError(f"Cannot start campaign in status {self.status}")
        
        self.status = CampaignStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)
    
    def complete_campaign(self) -> None:
        """Mark campaign as completed"""
        if self.status != CampaignStatus.RUNNING:
            raise ValueError(f"Cannot complete campaign in status {self.status}")
        
        self.status = CampaignStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
    
    def add_finding(self, finding_id: FindingId) -> None:
        """Add a finding to the campaign"""
        if finding_id not in self.findings:
            self.findings.append(finding_id)
    
    def schedule_agent(self, agent_id: AgentId) -> None:
        """Schedule an agent for execution"""
        if agent_id not in self.scheduled_agents:
            self.scheduled_agents.append(agent_id)


class Finding(Entity):
    """A security finding discovered during testing"""
    
    def __init__(
        self,
        id_: FindingId,
        campaign_id: CampaignId,
        agent_id: AgentId,
        title: str,
        description: str,
        severity: Severity,
        created_at: datetime,
        status: FindingStatus = FindingStatus.NEW,
        evidence: Optional[Dict[str, Any]] = None,
        embedding: Optional[Embedding] = None
    ) -> None:
        super().__init__(id_)
        self.campaign_id = campaign_id
        self.agent_id = agent_id
        self.title = title
        self.description = description
        self.severity = severity
        self.created_at = created_at
        self.status = status
        self.evidence = evidence or {}
        self.embedding = embedding
        self.updated_at = created_at
    
    def update_status(self, new_status: FindingStatus) -> None:
        """Update finding status"""
        self.status = new_status
        self.updated_at = datetime.now(timezone.utc)
    
    def add_evidence(self, key: str, value: Any) -> None:
        """Add evidence to finding"""
        self.evidence[key] = value
        self.updated_at = datetime.now(timezone.utc)
    
    def set_embedding(self, embedding: Embedding) -> None:
        """Set vector embedding for similarity search"""
        self.embedding = embedding
        self.updated_at = datetime.now(timezone.utc)


class KnowledgeAtom(Entity):
    """A unit of knowledge in the security intelligence system"""
    
    def __init__(
        self,
        id_: AtomId, 
        content: str,
        atom_type: AtomType,
        confidence: float,
        created_at: datetime,
        embedding: Optional[Embedding] = None,
        tags: Optional[Set[str]] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(id_)
        self.content = content
        self.atom_type = atom_type
        self.confidence = self._validate_confidence(confidence)
        self.created_at = created_at
        self.embedding = embedding
        self.tags = tags or set()
        self.source = source
        self.metadata = metadata or {}
        self.updated_at = created_at
    
    @staticmethod
    def _validate_confidence(confidence: float) -> float:
        """Validate confidence score is between 0 and 1"""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")
        return confidence
    
    def update_confidence(self, new_confidence: float) -> None:
        """Update confidence score"""
        self.confidence = self._validate_confidence(new_confidence)
        self.updated_at = datetime.now(timezone.utc)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the atom"""
        self.tags.add(tag)
        self.updated_at = datetime.now(timezone.utc)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the atom"""
        self.tags.discard(tag)
        self.updated_at = datetime.now(timezone.utc)
    
    def set_embedding(self, embedding: Embedding) -> None:
        """Set vector embedding for similarity search"""
        self.embedding = embedding
        self.updated_at = datetime.now(timezone.utc)