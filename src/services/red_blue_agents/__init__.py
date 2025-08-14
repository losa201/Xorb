"""
XORB Red/Blue Agent Framework
============================

Self-learning Red/Blue agent framework for cyber range and PTaaS services.

Features:
- Specialized agents (Recon/Exploit/Persistence/Evasion + Blue defenders) 
- Isolated sandboxes (Docker sidecar + Kata containers)
- Capability registry with environment-specific technique allow/deny
- Telemetry + learning state storage (Redis + PostgreSQL)
- Autonomous exploration and tactic learning
- Mission-based orchestration with TTL/quotas
"""

__version__ = "1.0.0"
__author__ = "XORB Security Team"

from .core.agent_scheduler import AgentScheduler
from .core.capability_registry import CapabilityRegistry
from .core.sandbox_orchestrator import SandboxOrchestrator
from .agents.red_team import ReconAgent, ExploitAgent, PersistenceAgent, EvasionAgent
from .agents.blue_team import DetectionAgent, ResponseAgent, HuntingAgent
from .telemetry.collector import TelemetryCollector
from .learning.autonomous_explorer import AutonomousExplorer

__all__ = [
    "AgentScheduler",
    "CapabilityRegistry", 
    "SandboxOrchestrator",
    "ReconAgent",
    "ExploitAgent", 
    "PersistenceAgent",
    "EvasionAgent",
    "DetectionAgent",
    "ResponseAgent",
    "HuntingAgent",
    "TelemetryCollector",
    "AutonomousExplorer"
]