
import importlib
import pkgutil
from typing import List, Type
from xorb_core.agents.agent import BaseAgent

def discover_agents(package) -> List[Type[BaseAgent]]:
    """Discovers all BaseAgent implementations within a given package."""
    agents = []
    for _, name, is_pkg in pkgutil.walk_packages(package.__path__):
        if not is_pkg:
            module = importlib.import_module(f"{package.__name__}.{name}")
            for item_name in dir(module):
                item = getattr(module, item_name)
                if isinstance(item, type) and issubclass(item, BaseAgent) and item is not BaseAgent:
                    agents.append(item)
    return agents
