
from typing import Protocol, runtime_checkable

from ..models.agents import DiscoveryTarget, Finding


@runtime_checkable
class BaseAgent(Protocol):
    """
    A protocol defining the basic structure of an agent in the Xorb ecosystem.

    Agents are specialized, stateless components responsible for performing a single,
    well-defined task (e.g., subdomain enumeration, port scanning). They accept a
    target, perform their action, and return a list of findings.
    """

    @property
    def name(self) -> str:
        """A unique, machine-readable name for the agent."""
        ...

    @property
    def description(self) -> str:
        """A human-readable description of what the agent does."""
        ...

    @property
    def accepted_target_types(self) -> set[str]:
        """The set of `target_type` values this agent can process."""
        ...

    async def run(self, target: DiscoveryTarget) -> list[Finding]:
        """
        Executes the agent's primary task on a given target.

        Args:
            target: The DiscoveryTarget to be processed by the agent.

        Returns:
            A list of Findings discovered by the agent.
        """
        ...
