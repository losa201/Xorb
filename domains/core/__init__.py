"""
XORB Core Domain

Central domain containing core business logic, configurations, and shared utilities.
"""

from .config import XORBConfig, config
from .enums import AgentType, CampaignStatus, ThreatSeverity
from .exceptions import ConfigurationError, ValidationError, XORBError
from .models import Agent, Campaign, Target

__all__ = [
    "config",
    "XORBConfig",
    "XORBError",
    "ConfigurationError",
    "ValidationError",
    "Agent",
    "Campaign",
    "Target",
    "AgentType",
    "CampaignStatus",
    "ThreatSeverity"
]
