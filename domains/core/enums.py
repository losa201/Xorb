"""
XORB Core Enumerations

Shared enums across XORB platform.
"""

from enum import Enum


class AgentType(Enum):
    """Types of XORB agents."""
    RECONNAISSANCE = "reconnaissance"
    VULNERABILITY_SCANNER = "vulnerability_scanner"
    PENETRATION_TESTING = "penetration_testing"
    THREAT_HUNTING = "threat_hunting"
    MALWARE_ANALYSIS = "malware_analysis"
    NETWORK_MONITOR = "network_monitor"
    INCIDENT_RESPONSE = "incident_response"
    BEHAVIOR_ANALYSIS = "behavior_analysis"
    ENDPOINT_PROTECTION = "endpoint_protection"
    FORENSICS_ANALYSIS = "forensics_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    SIGNATURE_DETECTION = "signature_detection"
    THREAT_INTELLIGENCE = "threat_intelligence"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"


class CampaignStatus(Enum):
    """Campaign execution statuses."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ThreatSeverity(Enum):
    """Threat severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EngineType(Enum):
    """Browser engine types for agents."""
    PLAYWRIGHT = "playwright"
    SELENIUM = "selenium"
    REQUESTS = "requests"


class EvasionTechnique(Enum):
    """Evasion techniques for stealth operations."""
    USER_AGENT_ROTATION = "user_agent_rotation"
    PROXY_ROTATION = "proxy_rotation"
    TIMING_RANDOMIZATION = "timing_randomization"
    HEADER_RANDOMIZATION = "header_randomization"
    SESSION_ROTATION = "session_rotation"
    FINGERPRINT_SPOOFING = "fingerprint_spoofing"


class KnowledgeAtomType(Enum):
    """Types of knowledge atoms in the knowledge fabric."""
    VULNERABILITY = "vulnerability"
    THREAT = "threat"
    INDICATOR = "indicator"
    TECHNIQUE = "technique"
    MITIGATION = "mitigation"
    ASSET = "asset"
    RELATIONSHIP = "relationship"
