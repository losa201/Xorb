from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract base class for all Xorb agents."""

    @abstractmethod
    def execute(self, task: dict) -> dict:
        """Execute a task."""

    @abstractmethod
    def healthcheck(self) -> dict:
        """Perform a healthcheck."""

class BaseDataSource(ABC):
    """Abstract base class for all Xorb data sources."""

    @abstractmethod
    def fetch(self, query: dict) -> dict:
        """Fetch data from a source."""
