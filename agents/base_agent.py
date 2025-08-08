from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):
    """Abstract base class for all agents in the simulation."""
    def __init__(self, id, position, resource_level):
        self.id = id
        self.position = position
        self.resource_level = resource_level
        self.type = "base"
        
    @abstractmethod
    def get_telemetry(self):
        """Return telemetry data for this agent."""
        pass
        
    def move(self, new_position):
        """Update agent position with resource consumption."""
        distance = np.linalg.norm(np.array(self.position) - np.array(new_position))
        self.position = new_position
        
        # Moving consumes resources
        move_cost = distance * 0.01
        self.resource_level = max(0, self.resource_level - move_cost)
        
        return True

class RedTeamAgent(Agent):
    """Abstract base class for red team agents."""
    def __init__(self, id, position, stealth_budget, resource_level, skill_level=0.5):
        super().__init__(id, position, resource_level)
        self.stealth_budget = stealth_budget
        self.skill_level = skill_level  # 0-1 scale of attacker proficiency
        self.type = "red"
        self.detected = False
        self.last_action_time = 0
        self.operation_phase = "reconnaissance"  # Initial phase
        self.knowledge_base = {}