from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime

class DiscoveryTarget(BaseModel):
    """Target specification for discovery operations"""
    value: str  # Domain, IP, URL, etc.
    target_type: str  # "domain", "ip", "url", "cidr"
    scope: str = "in_scope"  # "in_scope", "out_of_scope", "unknown"
    priority: int = Field(default=5, ge=1, le=10)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Finding(BaseModel):
    """Security finding or discovery result"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    finding_type: str
    title: str
    description: str
    severity: str = "info"  # critical, high, medium, low, info
    confidence: float = Field(ge=0.0, le=1.0)
    target: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    remediation: Optional[str] = None
    references: List[str] = Field(default_factory=list)
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    agent_id: str
    campaign_id: Optional[str] = None
