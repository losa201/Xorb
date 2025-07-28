"""
Ports - Abstract interfaces for external dependencies

Ports define the contracts that infrastructure adapters must implement.
They represent the boundary between the application and external concerns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, TypeVar, runtime_checkable

from ..domain import (
    Agent,
    AgentId,
    AtomId,
    Campaign,
    CampaignId,
    DomainEvent,
    Embedding,
    Finding,
    FindingId,
    KnowledgeAtom,
    Target,
    TargetId,
)

__all__ = [
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
    "SecurityScanner"
]

T = TypeVar('T')

@runtime_checkable
class Repository(Protocol[T]):
    """Base repository interface"""

    @abstractmethod
    async def save(self, entity: T) -> None:
        """Save an entity"""

    @abstractmethod
    async def find_by_id(self, id_: Any) -> T | None:
        """Find entity by ID"""

    @abstractmethod
    async def delete(self, entity: T) -> None:
        """Delete an entity"""


class TargetRepository(ABC):
    """Repository for Target entities"""

    @abstractmethod
    async def save(self, target: Target) -> None:
        """Save a target"""

    @abstractmethod
    async def find_by_id(self, target_id: TargetId) -> Target | None:
        """Find target by ID"""

    @abstractmethod
    async def find_by_name(self, name: str) -> Target | None:
        """Find target by name"""

    @abstractmethod
    async def find_all(self) -> list[Target]:
        """Find all targets"""

    @abstractmethod
    async def delete(self, target: Target) -> None:
        """Delete a target"""


class AgentRepository(ABC):
    """Repository for Agent entities"""

    @abstractmethod
    async def save(self, agent: Agent) -> None:
        """Save an agent"""

    @abstractmethod
    async def find_by_id(self, agent_id: AgentId) -> Agent | None:
        """Find agent by ID"""

    @abstractmethod
    async def find_active_agents(self) -> list[Agent]:
        """Find all active agents"""

    @abstractmethod
    async def find_by_capabilities(self, capabilities: list[str]) -> list[Agent]:
        """Find agents with specific capabilities"""

    @abstractmethod
    async def delete(self, agent: Agent) -> None:
        """Delete an agent"""


class CampaignRepository(ABC):
    """Repository for Campaign entities"""

    @abstractmethod
    async def save(self, campaign: Campaign) -> None:
        """Save a campaign"""

    @abstractmethod
    async def find_by_id(self, campaign_id: CampaignId) -> Campaign | None:
        """Find campaign by ID"""

    @abstractmethod
    async def find_by_status(self, status: str) -> list[Campaign]:
        """Find campaigns by status"""

    @abstractmethod
    async def find_active_campaigns(self) -> list[Campaign]:
        """Find all running campaigns"""

    @abstractmethod
    async def delete(self, campaign: Campaign) -> None:
        """Delete a campaign"""


class FindingRepository(ABC):
    """Repository for Finding entities"""

    @abstractmethod
    async def save(self, finding: Finding) -> None:
        """Save a finding"""

    @abstractmethod
    async def find_by_id(self, finding_id: FindingId) -> Finding | None:
        """Find finding by ID"""

    @abstractmethod
    async def find_by_campaign(self, campaign_id: CampaignId) -> list[Finding]:
        """Find findings for a campaign"""

    @abstractmethod
    async def find_by_severity(self, severity: str) -> list[Finding]:
        """Find findings by severity"""

    @abstractmethod
    async def find_similar(
        self,
        embedding: Embedding,
        threshold: float = 0.8,
        limit: int = 10
    ) -> list[tuple[Finding, float]]:
        """Find similar findings using embedding similarity"""

    @abstractmethod
    async def delete(self, finding: Finding) -> None:
        """Delete a finding"""


class KnowledgeAtomRepository(ABC):
    """Repository for KnowledgeAtom entities"""

    @abstractmethod
    async def save(self, atom: KnowledgeAtom) -> None:
        """Save a knowledge atom"""

    @abstractmethod
    async def find_by_id(self, atom_id: AtomId) -> KnowledgeAtom | None:
        """Find atom by ID"""

    @abstractmethod
    async def find_by_type(self, atom_type: str) -> list[KnowledgeAtom]:
        """Find atoms by type"""

    @abstractmethod
    async def find_by_tags(self, tags: list[str]) -> list[KnowledgeAtom]:
        """Find atoms with specific tags"""

    @abstractmethod
    async def find_similar(
        self,
        embedding: Embedding,
        threshold: float = 0.7,
        limit: int = 5
    ) -> list[tuple[KnowledgeAtom, float]]:
        """Find similar knowledge atoms"""

    @abstractmethod
    async def delete(self, atom: KnowledgeAtom) -> None:
        """Delete a knowledge atom"""


class EmbeddingService(ABC):
    """Service for generating vector embeddings"""

    @abstractmethod
    async def generate_embedding(
        self,
        text: str,
        model: str = "nvidia/embed-qa-4",
        input_type: str = "query"
    ) -> Embedding:
        """Generate embedding for text"""

    @abstractmethod
    async def generate_embeddings(
        self,
        texts: list[str],
        model: str = "nvidia/embed-qa-4",
        input_type: str = "query"
    ) -> list[Embedding]:
        """Generate embeddings for multiple texts"""

    @abstractmethod
    async def compute_similarity(
        self,
        embedding1: Embedding,
        embedding2: Embedding,
        metric: str = "cosine"
    ) -> float:
        """Compute similarity between embeddings"""


class EventPublisher(ABC):
    """Service for publishing domain events"""

    @abstractmethod
    async def publish(self, event: DomainEvent) -> None:
        """Publish a domain event"""

    @abstractmethod
    async def publish_batch(self, events: list[DomainEvent]) -> None:
        """Publish multiple domain events"""


class CacheService(ABC):
    """Service for caching data"""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get value from cache"""

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None
    ) -> None:
        """Set value in cache"""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache"""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""

    @abstractmethod
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""


class NotificationService(ABC):
    """Service for sending notifications"""

    @abstractmethod
    async def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "info",
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Send an alert notification"""

    @abstractmethod
    async def send_campaign_update(
        self,
        campaign_id: CampaignId,
        status: str,
        details: str
    ) -> None:
        """Send campaign status update"""


class SecurityScanner(ABC):
    """Service for security scanning"""

    @abstractmethod
    async def scan_target(
        self,
        target: Target,
        scan_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Perform security scan on target"""

    @abstractmethod
    async def validate_scope(
        self,
        target: Target,
        url: str
    ) -> bool:
        """Validate if URL is within target scope"""

    @abstractmethod
    async def get_scan_capabilities(self) -> list[str]:
        """Get available scanning capabilities"""
