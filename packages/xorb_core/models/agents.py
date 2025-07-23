
from pydantic import BaseModel, Field
from typing import Dict, Any

class DiscoveryTarget(BaseModel):
    """Target specification for discovery operations"""
    value: str  # Domain, IP, URL, etc.
    target_type: str  # "domain", "ip", "url", "cidr"
    scope: str = "in_scope"  # "in_scope", "out_of_scope", "unknown"
    priority: int = Field(default=5, ge=1, le=10)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Finding(BaseModel):
    """Security finding or discovery result"""
    title: str
    description: str
    target: str
    finding_type: str
    severity: str = "info"  # critical, high, medium, low, info
    confidence: float = Field(ge=0.0, le=1.0)
