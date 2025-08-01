"""
XORB Core Models

Core data models and domain entities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from .enums import AgentType, CampaignStatus, ThreatSeverity


@dataclass
class Agent:
    """Core agent model."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    agent_type: AgentType = AgentType.RECONNAISSANCE
    capabilities: list[str] = field(default_factory=list)
    status: str = "idle"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Target:
    """Target model for security testing."""
    id: str = field(default_factory=lambda: str(uuid4()))
    url: str = ""
    name: str = ""
    description: str = ""
    scope: list[str] = field(default_factory=list)
    out_of_scope: list[str] = field(default_factory=list)
    authorization: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Campaign:
    """Campaign execution model."""
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    description: str = ""
    status: CampaignStatus = CampaignStatus.PENDING
    targets: list[Target] = field(default_factory=list)
    agent_requirements: list[AgentType] = field(default_factory=list)
    max_duration: int = 3600  # seconds
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    results: dict[str, Any] = field(default_factory=dict)


@dataclass
class Vulnerability:
    """Vulnerability finding model."""
    id: str = field(default_factory=lambda: str(uuid4()))
    title: str = ""
    description: str = ""
    severity: ThreatSeverity = ThreatSeverity.INFO
    cvss_score: float | None = None
    cve_id: str | None = None
    affected_asset: str = ""
    proof_of_concept: str = ""
    remediation: str = ""
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    discovered_by: str = ""  # Agent ID


@dataclass
class ThreatIntelligence:
    """Threat intelligence model."""
    id: str = field(default_factory=lambda: str(uuid4()))
    source: str = ""
    indicator_type: str = ""
    indicator_value: str = ""
    confidence: float = 0.0
    severity: ThreatSeverity = ThreatSeverity.INFO
    tags: list[str] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
