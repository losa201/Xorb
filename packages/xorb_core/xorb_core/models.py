"""
Xorb Core Models
Core data models for XORB security intelligence platform
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class TargetType(str, Enum):
    """Types of discovery targets."""
    DOMAIN = "domain"
    IP = "ip"
    URL = "url"
    CIDR = "cidr"


class FindingSeverity(str, Enum):
    """Finding severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class DiscoveryTarget:
    """Represents a target for security discovery operations."""
    target_type: TargetType
    value: str
    scope: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Finding:
    """Represents a security finding."""
    id: str
    title: str
    description: str
    severity: FindingSeverity
    target: str
    discovery_method: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgentTask:
    """Represents a task for an agent to execute."""
    id: str
    target: DiscoveryTarget
    parameters: Dict[str, Any]
    timeout: Optional[int] = 300
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class AgentResult:
    """Represents the result of an agent execution."""
    task_id: str
    success: bool
    findings: List[Finding]
    metadata: Dict[str, Any]
    execution_time: float
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.findings is None:
            self.findings = []
        if self.metadata is None:
            self.metadata = {}


class AgentCapability(str, Enum):
    """Agent capabilities for discovery and assessment."""
    SUBDOMAIN_ENUMERATION = "subdomain_enumeration"
    PORT_SCANNING = "port_scanning"
    WEB_CRAWLING = "web_crawling"
    VULNERABILITY_SCANNING = "vulnerability_scanning"
    DNS_RESOLUTION = "dns_resolution"
    SSL_ANALYSIS = "ssl_analysis"


class AgentType(str, Enum):
    """Types of agents in the system."""
    DISCOVERY = "discovery"
    SCANNER = "scanner"
    CRAWLER = "crawler"
    ANALYZER = "analyzer"