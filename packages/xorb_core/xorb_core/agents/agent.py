from typing import Protocol, List, Dict, Any
from enum import Enum

from ..models.agents import Finding, DiscoveryTarget

class AgentCapability(str, Enum):
    DISCOVERY = "discovery"
    SCANNING = "scanning"
    EXPLOITATION = "exploitation"
    ANALYSIS = "analysis"
    REPORTING = "reporting"
    INTELLIGENCE = "intelligence"

class BaseAgent(Protocol):
    """A protocol that defines the behavior of an agent activity."""

    agent_id: str
    name: str
    capabilities: List[AgentCapability]

    async def execute_task(self, task_type: str, parameters: Dict[str, Any]) -> List[Finding]:
        """Execute a specific task and return a list of findings."""
        ...
