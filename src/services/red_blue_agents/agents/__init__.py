"""
Specialized Red and Blue Team Agents
==================================

This module contains the implementation of specialized agents for red and blue team operations.
Each agent type focuses on specific MITRE ATT&CK techniques and defensive capabilities.
"""

from .red_team import ReconAgent, ExploitAgent, PersistenceAgent, EvasionAgent, CollectionAgent
from .blue_team import DetectionAgent, AnalysisAgent, HuntingAgent, ResponseAgent

__all__ = [
    # Red Team Agents
    "ReconAgent",
    "ExploitAgent", 
    "PersistenceAgent",
    "EvasionAgent",
    "CollectionAgent",
    
    # Blue Team Agents
    "DetectionAgent",
    "AnalysisAgent",
    "HuntingAgent", 
    "ResponseAgent"
]