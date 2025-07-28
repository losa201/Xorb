"""
XORB Agent Registry

Centralized agent discovery, registration, and management system.
"""

import importlib
import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path

from domains.core import Agent, AgentType, config
from domains.core.exceptions import AgentError

logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """Agent capability definition."""
    name: str
    description: str
    required_resources: dict[str, int] = field(default_factory=dict)
    prerequisites: list[str] = field(default_factory=list)


@dataclass
class AgentDefinition:
    """Agent definition with metadata."""
    agent_class: type
    agent_type: AgentType
    capabilities: list[AgentCapability]
    resource_requirements: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


class AgentRegistry:
    """Central registry for all XORB agents."""

    def __init__(self):
        self._agents: dict[str, AgentDefinition] = {}
        self._active_agents: dict[str, Agent] = {}
        self._discovered_paths: set[Path] = set()

    async def discover_agents(self, search_paths: list[Path] = None) -> int:
        """
        Discover agents from specified paths and entry points.
        
        Returns:
            Number of agents discovered
        """
        if search_paths is None:
            search_paths = [
                config.base_path / "domains" / "agents",
                config.base_path / "services" / "*" / "agents",
                config.base_path / "ecosystem" / "agents"
            ]

        discovered_count = 0

        for search_path in search_paths:
            if search_path.exists():
                discovered_count += await self._discover_from_path(search_path)

        # Discover from entry points
        discovered_count += await self._discover_from_entry_points()

        logger.info(f"Discovered {discovered_count} agents")
        return discovered_count

    async def _discover_from_path(self, path: Path) -> int:
        """Discover agents from a filesystem path."""
        if path in self._discovered_paths:
            return 0

        self._discovered_paths.add(path)
        count = 0

        for py_file in path.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                spec = importlib.util.spec_from_file_location(
                    py_file.stem, py_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if self._is_agent_class(obj):
                        await self._register_agent_class(obj)
                        count += 1

            except Exception as e:
                logger.warning(f"Failed to load {py_file}: {e}")

        return count

    async def _discover_from_entry_points(self) -> int:
        """Discover agents from entry points."""
        try:
            import pkg_resources
            count = 0

            for entry_point in pkg_resources.iter_entry_points('xorb.agents'):
                try:
                    agent_class = entry_point.load()
                    if self._is_agent_class(agent_class):
                        await self._register_agent_class(agent_class)
                        count += 1
                except Exception as e:
                    logger.warning(f"Failed to load entry point {entry_point}: {e}")

            return count

        except ImportError:
            return 0

    def _is_agent_class(self, obj: type) -> bool:
        """Check if object is a valid agent class."""
        return (
            inspect.isclass(obj) and
            hasattr(obj, 'execute') and
            hasattr(obj, 'capabilities') and
            hasattr(obj, 'agent_type')
        )

    async def _register_agent_class(self, agent_class: type):
        """Register an agent class."""
        try:
            capabilities = []
            if hasattr(agent_class, 'capabilities'):
                for cap in agent_class.capabilities:
                    if isinstance(cap, str):
                        capabilities.append(AgentCapability(name=cap, description=""))
                    elif isinstance(cap, dict):
                        capabilities.append(AgentCapability(**cap))

            definition = AgentDefinition(
                agent_class=agent_class,
                agent_type=getattr(agent_class, 'agent_type', AgentType.RECONNAISSANCE),
                capabilities=capabilities,
                resource_requirements=getattr(agent_class, 'resource_requirements', {}),
                metadata=getattr(agent_class, 'metadata', {})
            )

            self._agents[agent_class.__name__] = definition
            logger.debug(f"Registered agent: {agent_class.__name__}")

        except Exception as e:
            raise AgentError(f"Failed to register agent {agent_class.__name__}: {e}")

    def get_agents_by_type(self, agent_type: AgentType) -> list[AgentDefinition]:
        """Get all agents of a specific type."""
        return [
            definition for definition in self._agents.values()
            if definition.agent_type == agent_type
        ]

    def get_agents_by_capability(self, capability: str) -> list[AgentDefinition]:
        """Get all agents with a specific capability."""
        result = []
        for definition in self._agents.values():
            if any(cap.name == capability for cap in definition.capabilities):
                result.append(definition)
        return result

    async def create_agent(self, agent_name: str, **kwargs) -> Agent:
        """Create an agent instance."""
        if agent_name not in self._agents:
            raise AgentError(f"Agent {agent_name} not found in registry")

        definition = self._agents[agent_name]

        try:
            agent_instance = definition.agent_class(**kwargs)
            agent = Agent(
                name=agent_name,
                agent_type=definition.agent_type,
                capabilities=[cap.name for cap in definition.capabilities],
                metadata=definition.metadata
            )

            self._active_agents[agent.id] = agent
            return agent

        except Exception as e:
            raise AgentError(f"Failed to create agent {agent_name}: {e}")

    def get_active_agents(self) -> list[Agent]:
        """Get all active agent instances."""
        return list(self._active_agents.values())

    def get_registry_stats(self) -> dict[str, int]:
        """Get registry statistics."""
        return {
            "total_registered": len(self._agents),
            "active_instances": len(self._active_agents),
            "agent_types": len(set(definition.agent_type for definition in self._agents.values()))
        }


# Global registry instance
registry = AgentRegistry()
