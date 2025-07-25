
import inspect
import pkgutil
from typing import Dict, List, Optional, Type

from . import agents
from .agents.agent import BaseAgent


class AgentRegistry:
    """A singleton registry for discovering and accessing all available agents."""

    _instance = None
    _agents: Dict[str, BaseAgent] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentRegistry, cls).__new__(cls)
            cls._instance._discover_agents()
        return cls._instance

    def _discover_agents(self):
        """Dynamically import and instantiate all agent classes."""
        if self._agents:
            return

        package = agents
        for _, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            if not is_pkg:
                module = __import__(name, fromlist="dummy")
                for attribute_name in dir(module):
                    attribute = getattr(module, attribute_name)
                    if (
                        inspect.isclass(attribute)
                        and hasattr(attribute, 'name')
                        and hasattr(attribute, 'description')
                        and hasattr(attribute, 'accepted_target_types')
                        and hasattr(attribute, 'run')
                        and attribute is not BaseAgent
                        and not inspect.isabstract(attribute)
                    ):
                        # Instantiate the agent
                        agent_instance = attribute()
                        if agent_instance.name in self._agents:
                            raise ValueError(f"Duplicate agent name found: {agent_instance.name}")
                        self._agents[agent_instance.name] = agent_instance
                        print(f"Discovered and registered agent: {agent_instance.name}")

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent instance by its unique name."""
        return self._agents.get(name)

    def get_all_agents(self) -> List[BaseAgent]:
        """Return a list of all registered agent instances."""
        return list(self._agents.values())

    def get_agents_for_target_type(self, target_type: str) -> List[BaseAgent]:
        """Find all agents that can handle a specific target type."""
        return [
            agent
            for agent in self._agents.values()
            if target_type in agent.accepted_target_types
        ]


# Initialize the singleton registry instance
agent_registry = AgentRegistry()
