"""
XORB Core Domain

Central domain containing core business logic, configurations, and shared utilities.
"""

from .config import config, XORBConfig
from .exceptions import XORBError, ConfigurationError, ValidationError
from .models import Agent, Campaign, Target
from .enums import AgentType, CampaignStatus, ThreatSeverity

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