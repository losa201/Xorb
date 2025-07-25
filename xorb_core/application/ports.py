"""
Ports - Abstract interfaces for external dependencies

Ports define the contracts that infrastructure adapters must implement.
They represent the boundary between the application and external concerns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, TypeVar
from uuid import UUID

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
    TargetId
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
        pass
    
    @abstractmethod
    async def find_by_id(self, id_: Any) -> Optional[T]:
        """Find entity by ID"""
        pass
    
    @abstractmethod
    async def delete(self, entity: T) -> None:
        """Delete an entity"""
        pass


class TargetRepository(ABC):
    """Repository for Target entities"""
    
    @abstractmethod
    async def save(self, target: Target) -> None:
        """Save a target"""
        pass
    
    @abstractmethod
    async def find_by_id(self, target_id: TargetId) -> Optional[Target]:
        """Find target by ID"""
        pass
    
    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[Target]:
        """Find target by name"""
        pass
    
    @abstractmethod
    async def find_all(self) -> List[Target]:
        """Find all targets"""
        pass
    
    @abstractmethod
    async def delete(self, target: Target) -> None:
        """Delete a target"""
        pass


class AgentRepository(ABC):
    """Repository for Agent entities"""
    
    @abstractmethod
    async def save(self, agent: Agent) -> None:
        """Save an agent"""
        pass
    
    @abstractmethod
    async def find_by_id(self, agent_id: AgentId) -> Optional[Agent]:
        """Find agent by ID"""
        pass
    
    @abstractmethod
    async def find_active_agents(self) -> List[Agent]:
        """Find all active agents"""
        pass
    
    @abstractmethod
    async def find_by_capabilities(self, capabilities: List[str]) -> List[Agent]:
        """Find agents with specific capabilities"""
        pass
    
    @abstractmethod
    async def delete(self, agent: Agent) -> None:
        """Delete an agent"""
        pass


class CampaignRepository(ABC):
    """Repository for Campaign entities"""
    
    @abstractmethod
    async def save(self, campaign: Campaign) -> None:
        """Save a campaign"""
        pass
    
    @abstractmethod
    async def find_by_id(self, campaign_id: CampaignId) -> Optional[Campaign]:
        """Find campaign by ID"""
        pass
    
    @abstractmethod
    async def find_by_status(self, status: str) -> List[Campaign]:
        """Find campaigns by status"""
        pass
    
    @abstractmethod
    async def find_active_campaigns(self) -> List[Campaign]:
        """Find all running campaigns"""
        pass
    
    @abstractmethod
    async def delete(self, campaign: Campaign) -> None:
        """Delete a campaign"""
        pass


class FindingRepository(ABC):
    """Repository for Finding entities"""
    
    @abstractmethod
    async def save(self, finding: Finding) -> None:
        """Save a finding"""
        pass
    
    @abstractmethod
    async def find_by_id(self, finding_id: FindingId) -> Optional[Finding]:
        """Find finding by ID"""
        pass
    
    @abstractmethod
    async def find_by_campaign(self, campaign_id: CampaignId) -> List[Finding]:
        """Find findings for a campaign"""
        pass
    
    @abstractmethod
    async def find_by_severity(self, severity: str) -> List[Finding]:
        """Find findings by severity"""
        pass
    
    @abstractmethod
    async def find_similar(
        self, 
        embedding: Embedding, 
        threshold: float = 0.8,
        limit: int = 10
    ) -> List[tuple[Finding, float]]:
        """Find similar findings using embedding similarity"""
        pass
    
    @abstractmethod
    async def delete(self, finding: Finding) -> None:
        """Delete a finding"""
        pass


class KnowledgeAtomRepository(ABC):
    """Repository for KnowledgeAtom entities"""
    
    @abstractmethod
    async def save(self, atom: KnowledgeAtom) -> None:
        """Save a knowledge atom"""
        pass
    
    @abstractmethod
    async def find_by_id(self, atom_id: AtomId) -> Optional[KnowledgeAtom]:
        """Find atom by ID"""
        pass
    
    @abstractmethod
    async def find_by_type(self, atom_type: str) -> List[KnowledgeAtom]:
        """Find atoms by type"""
        pass
    
    @abstractmethod
    async def find_by_tags(self, tags: List[str]) -> List[KnowledgeAtom]:
        """Find atoms with specific tags"""
        pass
    
    @abstractmethod
    async def find_similar(
        self,
        embedding: Embedding,
        threshold: float = 0.7,
        limit: int = 5
    ) -> List[tuple[KnowledgeAtom, float]]:
        """Find similar knowledge atoms"""
        pass
    
    @abstractmethod
    async def delete(self, atom: KnowledgeAtom) -> None:
        """Delete a knowledge atom"""
        pass


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
        pass
    
    @abstractmethod
    async def generate_embeddings(
        self,
        texts: List[str],
        model: str = "nvidia/embed-qa-4", 
        input_type: str = "query"
    ) -> List[Embedding]:
        """Generate embeddings for multiple texts"""
        pass
    
    @abstractmethod
    async def compute_similarity(
        self,
        embedding1: Embedding,
        embedding2: Embedding,
        metric: str = "cosine"
    ) -> float:
        """Compute similarity between embeddings"""
        pass


class EventPublisher(ABC):
    """Service for publishing domain events"""
    
    @abstractmethod
    async def publish(self, event: DomainEvent) -> None:
        """Publish a domain event"""
        pass
    
    @abstractmethod
    async def publish_batch(self, events: List[DomainEvent]) -> None:
        """Publish multiple domain events"""
        pass


class CacheService(ABC):
    """Service for caching data"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[int] = None
    ) -> None:
        """Set value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        pass


class NotificationService(ABC):
    """Service for sending notifications"""
    
    @abstractmethod
    async def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "info",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send an alert notification"""
        pass
    
    @abstractmethod
    async def send_campaign_update(
        self,
        campaign_id: CampaignId,
        status: str,
        details: str
    ) -> None:
        """Send campaign status update"""
        pass


class SecurityScanner(ABC):
    """Service for security scanning"""
    
    @abstractmethod
    async def scan_target(
        self,
        target: Target,
        scan_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform security scan on target"""
        pass
    
    @abstractmethod
    async def validate_scope(
        self,
        target: Target,
        url: str
    ) -> bool:
        """Validate if URL is within target scope"""
        pass
    
    @abstractmethod
    async def get_scan_capabilities(self) -> List[str]:
        """Get available scanning capabilities"""
        pass